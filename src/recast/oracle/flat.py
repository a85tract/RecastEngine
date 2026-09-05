"""``f2py-golden-flat``: the f2py oracle for a tree whose interfaces are not flat.

Two things the plain ``f2py-golden`` cannot do for a model tree:

* **spell a derived-type interface.** For every plan the frontend stored on
  ``Facts.extra["flat_plans"]`` this oracle writes a Fortran adapter
  ``<name>_flat`` (``fortran_adapter``) that takes the object's touched
  components as flat arrays, allocates the object to its original bounds,
  calls the original, and copies the written components back. f2py is
  handed the adapter module, whose interface is the unit's spellable
  subprograms re-exported plus the adapters;
* **build against a tree of siblings.** Every module the unit ``use``s --
  the companions the frontend resolved and the stubs it did not -- is
  compiled ahead into one static library, and f2py sees only the adapter.
  Handing f2py the siblings makes it parse them, and despite ``only:`` it
  then builds a module-variable object for every sibling with public
  variables; one it cannot build is a NULL in the extension's init
  dictionary, which segfaults the interpreter on import.

Subprograms whose interface the wrapper still cannot spell (a procedure
dummy, a character argument) are left *ungated*, name and reason on the
reference's handle, for the verifier to carry into its verdict. Extra
include directories and link flags a tree needs (a netCDF the framework
stubs drag in) arrive in config -- ``include_dirs``, ``ldflags`` -- or from
a subclass that knows where to ask.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from recast.errors import OracleUnavailable, RecastError
from recast.fortran.flatten import FORTRAN_TYPES, FlatPlan, plans_from_facts, signature
from recast.fortran.tree import MODULE_DEFINITION, sources
from recast.model import Facts, OracleRef, Unit
from recast.oracle.f2py import (
    DEFAULT_FLAGS,
    F2pyGoldenOracle,
    _extra_sources,
    _regular_file,
    _resolved_root,
    _source_under_root,
)
from recast.plugins.executor import Executor, Job

__all__ = ["F2pyFlatOracle", "factory", "fortran_adapter", "stub_sources", "unspellable"]

USE = re.compile(r"^\s*use\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", re.IGNORECASE | re.MULTILINE)
INTRINSIC = frozenset({"iso_fortran_env", "iso_c_binding", "ieee_arithmetic", "netcdf"})


def _index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sources(root):
        for match in MODULE_DEFINITION.finditer(path.read_text(errors="replace")):
            index.setdefault(match.group(1).lower(), path)
    return index


def stub_sources(source: Path, root: Path, already: list[Path]) -> list[Path]:
    """Files defining the modules ``source`` uses transitively, minus
    ``already``, dependencies first."""
    index = _index(root)
    have = {p.resolve() for p in already} | {source.resolve()}
    ordered: list[Path] = []
    placed: set[str] = set()

    def place(name: str, stack: frozenset[str]) -> None:
        path = index.get(name)
        if path is None or name in placed or name in stack or name in INTRINSIC:
            return
        for dep in USE.findall(path.read_text(errors="replace")):
            place(dep.lower(), stack | {name})
        placed.add(name)
        if path.resolve() not in have:
            ordered.append(path.resolve())

    for dep in USE.findall(source.read_text(errors="replace")):
        place(dep.lower(), frozenset())
    return ordered


def unspellable(subprogram: dict[str, Any]) -> str | None:
    """Why the flat wrapper cannot take this subprogram, or ``None`` if it can.

    Naming the reason is what lets the verdict say *which* subprograms were
    left ungated and why, rather than failing the whole module on the first.
    """
    for argument in subprogram["args"]:
        if argument.get("optional"):
            continue
        dtype = str(argument["dtype"])
        if dtype == "PROCEDURE":
            return f"{argument['name']}: procedure dummy"
        if dtype == "str" and argument.get("dims"):
            return f"{argument['name']}: character array"
        if dtype not in FORTRAN_TYPES and dtype != "str":
            return f"{argument['name']}: {dtype}"
    return None


# --- the Fortran adapter -----------------------------------------------------


def _declare(argument: dict[str, Any]) -> str:
    # A character scalar (``phase = 'sun'``) selects a branch like any other
    # value; assumed length, so the caller's literal fits whatever it is.
    spelled = {"str": "character(len=*)"}.get(str(argument["dtype"])) or FORTRAN_TYPES.get(
        str(argument["dtype"])
    )
    if spelled is None:
        raise ValueError(f"{argument['name']}: {argument['dtype']} is not flat")
    intent = {"IN": "in", "OUT": "out", "INOUT": "inout", "UNKNOWN": "inout"}[argument["intent"]]
    dims = ""
    if argument.get("dims"):
        dims = "(" + ", ".join(d["ub"] or ":" for d in argument["dims"]) + ")"
    return f"    {spelled}, intent({intent}) :: {argument['name']}{dims}"


def fortran_adapter(module: str, plans: list[FlatPlan], reexport: list[str]) -> str:
    """The ``<module>_flat`` Fortran module: the adapters, and the module's
    own flat subprograms re-exported so one ``use`` line reaches both."""
    used_modules: dict[str, set[str]] = {}
    for plan in plans:
        for obj in plan.objects:
            if obj.kind == "state" and obj.module:
                used_modules.setdefault(obj.module, set()).add(obj.name)
        for state in plan.states:
            used_modules.setdefault(state.module, set()).add(state.name)
    # A type is named only where a dummy of it is declared. Module state is
    # reached by its own name (``use m, only: params_inst``): its type may be
    # private to the module, as ELM's photo_params_type is.
    types_used = {
        obj.type_name: obj.type_module
        for plan in plans
        for obj in plan.objects
        if obj.kind == "dummy"
    }
    lines = [
        "! Machine-generated by RecastEngine (recast.oracle.flat) -- DO NOT EDIT.",
        f"module {module}_flat",
        f"  use {module}, only: "
        + ", ".join(sorted({*reexport, *(p.subprogram["name"] for p in plans)})),
    ]
    for type_name, type_module in sorted(types_used.items()):
        lines.append(f"  use {type_module or type_name}, only: {type_name}")
    for mod, names in sorted(used_modules.items()):
        lines.append(f"  use {mod}, only: {', '.join(sorted(names))}")
    lines += ["  implicit none", "  public", "contains"]
    for plan in plans:
        args = plan.flat_args
        lines.append(f"  subroutine {plan.name}({', '.join(a['name'] for a in args)})")
        # Scalars first: an array's extent may be one of them.
        lines.extend(_declare(a) for a in args if not a.get("dims"))
        lines.extend(_declare(a) for a in args if a.get("dims"))
        for obj in plan.objects:
            if obj.kind == "dummy":
                lines.append(f"    type({obj.type_name}) :: {obj.name}")
        for obj in plan.objects:
            for comp in obj.components:
                target = f"{obj.name}%{comp.name}"
                if comp.bounds:
                    shape = ", ".join(f"{lb}:{ub}" for lb, ub in comp.bounds)
                    test = "associated" if comp.pointer else "allocated"
                    if obj.kind == "state":
                        lines.append(f"    if ({test}({target})) deallocate({target})")
                    lines.append(f"    allocate({target}({shape}))")
                lines.append(f"    {target} = {comp.flat}")
        for state in plan.states:
            if state.bounds:
                # An allocatable module array: allocated over the bounds its
                # own module gave it, or ``x = flat`` would rebase it to one.
                shape = ", ".join(f"{lb}:{ub}" for lb, ub in state.bounds)
                lines.append(f"    if (allocated({state.name})) deallocate({state.name})")
                lines.append(f"    allocate({state.name}({shape}))")
            lines.append(f"    {state.name} = {state.flat}")
        # Keyword actuals: an optional dummy in the middle of the list is
        # simply not passed, and Fortran resolves the rest by name.
        call_args = ", ".join(
            f"{a['name']}={a['name']}" for a in plan.subprogram["args"] if not a.get("optional")
        )
        lines.append(f"    call {plan.subprogram['name']}({call_args})")
        for obj in plan.objects:
            for comp in obj.components:
                if comp.written:
                    lines.append(f"    {comp.flat} = {obj.name}%{comp.name}")
        for state in plan.states:
            if state.written:
                lines.append(f"    {state.flat} = {state.name}")
        lines.append(f"  end subroutine {plan.name}")
        lines.append("")
    lines.append(f"end module {module}_flat")
    return "\n".join(lines) + "\n"


# --- the oracle --------------------------------------------------------------


class F2pyFlatOracle(F2pyGoldenOracle):
    """The engine's f2py oracle behind a static library and flat adapters."""

    name = "f2py-golden-flat"

    @staticmethod
    def _subprograms(facts: Facts, config: dict[str, Any]) -> list[str]:
        named = config.get("subprograms")
        if named:
            return list(named)
        return [
            s["name"]
            for s in facts.interface["subprograms"]
            if s.get("public", True) and unspellable(s) is None
        ]

    @staticmethod
    def ungated(facts: Facts, config: dict[str, Any]) -> dict[str, str]:
        named = config.get("subprograms")
        return {
            s["name"]: reason
            for s in facts.interface["subprograms"]
            if s.get("public", True)
            and (reason := unspellable(s)) is not None
            and (not named or s["name"] not in named)
        }

    # -- what a tree needs beyond the sources ---------------------------------

    def include_dirs(self, config: dict[str, Any]) -> list[str]:
        """Directories for ``-I`` beyond the sources' own; from config here,
        from wherever a subclass knows to ask."""
        return [str(d) for d in config.get("include_dirs") or []]

    def link_flags(self, config: dict[str, Any]) -> str:
        """Linker flags the extension module needs at load time."""
        return str(config.get("ldflags") or "")

    # -- the engine's root boundary -------------------------------------------

    # The engine's oracle reads the unit's source under the project root. This
    # oracle hands it the adapter it generated instead: content-hashed as
    # ``adapter_key`` the moment it was made, so the key trusts the bytes the
    # digest was taken from; written under the workspace by materialize,
    # where no project root contains it.

    def _main_source(self, facts: Facts, root: Path) -> Path:
        return _regular_file(Path(facts.provenance["source"]), label="flat adapter")

    def _main_source_digest(self, facts: Facts, root: Path) -> str:
        return str(facts.provenance["digest"])

    # -- the build plan -------------------------------------------------------

    def _plan(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> dict[str, Any]:
        """Everything the build depends on, decided without building."""
        root = _resolved_root(config.get("root", "."))
        source = _source_under_root(root, facts.provenance.get("source"), label="main source")
        # One topological order over everything the unit uses, companions
        # and stubs alike -- a companion may use a stub, so neither group can
        # be placed as a block ahead of the other. The unit's own source goes
        # in as well: f2py is handed the adapter module, which uses it.
        library = [*stub_sources(source, root, []), source]
        library += _extra_sources(config)
        plans = plans_from_facts(facts)
        module = facts.interface["module"]
        spellable = self._subprograms(facts, config)
        adapter = fortran_adapter(module, plans, spellable)
        digest = hashlib.sha256()
        for path in library:
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
        flags = config.get("fflags", DEFAULT_FLAGS)
        for include in self.include_dirs(config):
            if f"-I{include}" not in shlex.split(flags):
                flags = f"{flags} {shlex.quote(f'-I{include}')}"
        parameters = {
            **((facts.extra or {}).get("dim_parameters") or {}),
            **(config.get("wrapper_parameters") or {}),
        }
        return {
            "library": library,
            "library_key": digest.hexdigest()[:16],
            "plans": plans,
            "adapter": adapter,
            "adapter_key": hashlib.sha256(adapter.encode()).hexdigest(),
            "fflags": flags,
            "ldflags": self.link_flags(config),
            "wrapper_parameters": parameters,
        }

    def _handed(
        self,
        facts: Facts,
        config: dict[str, Any],
        plan: dict[str, Any],
        lib_dir: Path,
        adapter_path: Path,
    ) -> tuple[Facts, dict[str, Any]]:
        """What the engine's oracle is given: the adapter module as the
        source, whose interface is the unit's spellable subprograms
        (re-exported) plus one ``<name>_flat`` per adapted one; no
        companions, no extra sources; the library's ``.mod`` files on the
        include path. The engine then wraps and builds exactly as it would a
        module that was flat to begin with."""
        module = facts.interface["module"]
        chosen = set(self._subprograms(facts, config))
        flat_entries = [{**signature(p), "name": p.name, "public": True} for p in plan["plans"]]
        interface = {
            **facts.interface,
            "module": f"{module}_flat",
            "is_module": True,
            "generics": {},
            "subprograms": [
                *[s for s in facts.interface["subprograms"] if s["name"] in chosen],
                *flat_entries,
            ],
        }
        provenance = {
            **facts.provenance,
            "source": str(adapter_path),
            "digest": plan["adapter_key"],
            "companions": [],
        }
        handed_facts = dataclasses.replace(facts, interface=interface, provenance=provenance)
        handed = {
            **config,
            "extra_sources": [],
            "subprograms": [s["name"] for s in interface["subprograms"]],
            "fflags": f"{plan['fflags']} {shlex.quote(f'-I{lib_dir}')}",
            "wrapper_parameters": plan["wrapper_parameters"],
        }
        return handed_facts, handed

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        plan = self._plan(unit, facts, config)
        # Paths are keyed by content below, not by name.
        handed_facts, handed = self._handed(facts, config, plan, Path("<lib>"), Path("<adapter>"))
        base = super().key(unit, handed_facts, handed)
        # The unit's own digest as well: the library is keyed by the files'
        # content, and Facts that say the source moved have to move the key
        # even before the file on disk does.
        digest = hashlib.sha256(
            f"{base}|{facts.provenance.get('digest')}|{plan['library_key']}|"
            f"{plan['adapter_key']}|{plan['ldflags']}".encode()
        ).hexdigest()[:16]
        return f"f2py:{facts.interface['module']}:{digest}"

    def _build_library(
        self, plan: dict[str, Any], workspace: Path, executor: Executor, config: dict[str, Any]
    ) -> Path:
        """Compile the sibling and stub modules once into ``libref.a`` beside
        their ``.mod`` files, keyed by content; a second unit over the same
        siblings reuses it."""
        lib_dir = (workspace / f"reflib-{plan['library_key']}").resolve()
        if (lib_dir / "libref.a").is_file():
            return lib_dir
        lib_dir.mkdir(parents=True, exist_ok=True)
        compiler = config.get("fc", "gfortran")
        script = [
            "import subprocess, sys",
            f"fc = {compiler!r}",
            f"flags = {shlex.split(plan['fflags'])!r}",
            f"sources = {[str(p) for p in plan['library']]!r}",
            "objects = []",
            "for i, src in enumerate(sources):",
            "    obj = f'{i:03d}.o'",
            "    subprocess.run([fc, '-c', '-fPIC', *flags, '-J', '.', '-I', '.', '-o', obj, src],"
            " check=True)",
            "    objects.append(obj)",
            "subprocess.run(['ar', 'rcs', 'libref.a', *objects], check=True)",
        ]
        job = Job(
            argv=[sys.executable, "-c", "\n".join(script)],
            cwd=lib_dir,
            env={**os.environ, "FC": compiler},
            timeout_s=float(config.get("build_timeout", 600)),
            label="reference library",
        )
        try:
            result = executor.run(job)
        except RecastError:
            raise
        except Exception as error:
            # An executor that refuses is the case ``OracleUnavailable`` exists
            # for, and it has to arrive as one -- see ``F2pyGoldenOracle``.
            raise OracleUnavailable(
                f"the reference library could not be built: the executor refused: {error}"
            ) from error
        if not result.ok:
            (lib_dir / "build.log").write_text(result.stdout + "\n" + result.stderr)
            raise OracleUnavailable(
                f"the reference library did not build (exit {result.returncode}); "
                f"log at {lib_dir / 'build.log'}"
            )
        return lib_dir

    def materialize(
        self, unit: Unit, facts: Facts, workspace: Path, executor: Executor, config: dict[str, Any]
    ) -> OracleRef:
        if not self._subprograms(facts, config) and not plans_from_facts(facts):
            ungated = self.ungated(facts, config)
            raise OracleUnavailable(
                f"{unit.uid}: no subprogram the flat wrapper can spell; "
                + "; ".join(f"{n} ({why})" for n, why in sorted(ungated.items()))
            )
        plan = self._plan(unit, facts, config)
        lib_dir = self._build_library(plan, workspace, executor, config)
        adapter_dir = (workspace / f"flat-{plan['adapter_key'][:16]}").resolve()
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / f"{facts.interface['module']}_flat.f90"
        adapter_path.write_text(plan["adapter"])
        handed_facts, handed = self._handed(facts, config, plan, lib_dir, adapter_path)
        # The engine's job argv has no slot for linker flags; meson reads
        # ``LDFLAGS`` from the environment the job inherits. Scoped, restored.
        before = os.environ.get("LDFLAGS")
        os.environ["LDFLAGS"] = f"{before or ''} -L{lib_dir} -lref {plan['ldflags']}".strip()
        try:
            ref = super().materialize(unit, handed_facts, workspace, executor, handed)
        finally:
            if before is None:
                os.environ.pop("LDFLAGS", None)
            else:
                os.environ["LDFLAGS"] = before
        ref.oracle = self.name
        ref.key = self.key(unit, facts, config)
        adapted = {p.subprogram["name"] for p in plan["plans"]}
        ref.handle["ungated"] = {
            n: why for n, why in self.ungated(facts, config).items() if n not in adapted
        }
        ref.handle["flattened"] = sorted(adapted)
        ref.handle["library"] = str(lib_dir)
        return ref


def factory(**_config: Any) -> F2pyFlatOracle:
    return F2pyFlatOracle()
