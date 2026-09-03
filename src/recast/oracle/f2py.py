"""``f2py-golden``: the untouched Fortran, compiled, as the reference.

Migrated from the build half of the source pipeline's ``diff_driver.py``
and the wrapper rules of ``gen_wrapper.py``. f2py does not translate
anything: it compiles the original source with a real Fortran compiler and
generates glue so Python can call the resulting machine code directly. That
is what makes it a reference -- the thing being compared against *is* the
original program, and bit-exact agreement with it is a meaningful claim.

The wrapper layer exists because f2py's own Fortran parser is shallow. It
cannot resolve use-imported kinds (a silent ``real(4)`` truncation, learned
the hard way), stumbles on derived types, and handles optional arguments
badly. So every subprogram under test gets a flat wrapper subroutine: raw
``real(8)``/``integer`` declarations that need no kind resolution, optional
arguments dropped, dimensions spelled in terms of the other arguments so
f2py can size the outputs.

The cache key folds in everything that can move the reference: the source
digest, every extra source's digest, the compiler's identity and version,
and the flags. Two builds with the same key must behave identically, and a
compiler upgrade changes the key rather than silently invalidating every
downstream Verdict.

The compile itself goes through the executor -- on a laptop that is a
subprocess, on a batch system it is a job -- and the environment is passed
whole plus the compiler overrides, because f2py needs a real PATH to find
its toolchain.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from recast.errors import ConfigError, OracleUnavailable, RecastError
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor, Job
from recast.plugins.oracle import Oracle

__all__ = ["F2pyGoldenOracle", "factory", "wrappers_for"]

FORTRAN_TYPES = {
    "float64": "real(8)",
    "float32": "real(4)",
    "int32": "integer",
    "int64": "integer(8)",
    "bool": "logical",
    # Fixed width because f2py cannot size len=* dummies; 128 covers every
    # message and name in the corpus, and Fortran comparison semantics pad
    # the shorter operand with blanks anyway.
    "str": "character(len=128)",
}
"""Raw type spellings, no kind parameters: nothing here needs f2py's
crackfortran to resolve a use-imported kind, which it cannot."""

DEFAULT_FLAGS = "-O1 -fno-fast-math -ffp-contract=off"
"""Conservative by default. The reference must round the way the production
build rounds, and aggressive optimization is a second variable nobody asked
to test."""

_BUILD_LOG_TAIL_CHARS = 3000
"""How much of a failed build's output the error itself quotes. The tail,
because that is where crackfortran and the compiler report what they could
not parse; small enough that the error stays a message, not a log."""

_SAFE_SOURCE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}\Z")
"""A suffix that is safe to reproduce on a canonical staging filename.

f2py infers the source language and fixed/free form from the suffix, so the
staged copy must retain it.  The original basename is deliberately *not*
retained: NumPy's Meson backend joins and splits source arguments internally,
which turns whitespace (or a flag-looking basename) into additional tokens.
"""


def _resolved_root(value: str | os.PathLike[str]) -> Path:
    """Resolve and validate the project root before trusting provenance."""
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ConfigError(
            f"f2py project root {value!r} does not exist or cannot be resolved"
        ) from exc
    if not root.is_dir():
        raise ConfigError(f"f2py project root {root} is not a directory")
    return root


def _regular_file(path: Path, *, label: str) -> Path:
    """Return one canonical regular file or reject it fail-closed."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"{label} {path} does not exist or cannot be resolved") from exc
    if not resolved.is_file():
        raise ConfigError(f"{label} {resolved} is not a regular file")
    return resolved


def _source_under_root(root: Path, value: object, *, label: str) -> Path:
    """Resolve a provenance path and prove its target remains in ``root``."""
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"{label} must be a filesystem path, got {type(value).__name__}")
    resolved = _regular_file(root / Path(value), label=label)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(
            f"{label} {value!r} resolves outside the configured project root {root}"
        ) from exc
    return resolved


def _extra_sources(config: dict[str, Any]) -> list[Path]:
    """Validate explicitly configured sources (which may live outside root)."""
    values = config.get("extra_sources", []) or []
    if isinstance(values, (str, bytes, os.PathLike)):
        raise ConfigError("config['extra_sources'] must be a list of filesystem paths")
    resolved: list[Path] = []
    for index, value in enumerate(values):
        if not isinstance(value, (str, os.PathLike)):
            raise ConfigError(
                f"extra source {index} must be a filesystem path, got {type(value).__name__}"
            )
        resolved.append(_regular_file(Path(value), label=f"extra source {index}"))
    return resolved


def _staged_suffix(source: Path) -> str:
    suffix = source.suffix
    if not _SAFE_SOURCE_SUFFIX.fullmatch(suffix):
        raise ConfigError(
            f"source {source} has suffix {suffix!r}, which cannot be represented by a safe "
            "f2py staging filename"
        )
    return suffix


def _stage_build_inputs(
    stage: Path, sources: list[Path], wrapper_text: str
) -> tuple[list[str], list[str]]:
    """Copy sources and expose include directories through controlled names.

    NumPy currently performs ``' '.join(...).split()`` both while parsing
    f2py sources and while parsing include options.  Consequently *every*
    token given to it is a short relative name we generated here.  Source
    parents are reachable to the compiler only through ``includes/dNNNN``
    aliases; their original spellings never enter a flags string or Meson
    template.
    """
    source_dir = stage / "sources"
    include_dir = stage / "includes"
    backend_include_dir = stage / "f2py-build" / "includes"
    source_dir.mkdir()
    include_dir.mkdir()
    backend_include_dir.mkdir(parents=True)

    staged: list[str] = []
    for index, source in enumerate(sources):
        relative = Path("sources") / f"source_{index:04d}{_staged_suffix(source)}"
        shutil.copyfile(source, stage / relative)
        staged.append(relative.as_posix())

    wrapper = Path("sources") / "wrappers.f90"
    (stage / wrapper).write_text(wrapper_text)
    staged.append(wrapper.as_posix())

    include_args: list[str] = []
    parents = dict.fromkeys(source.parent for source in sources)
    for index, parent in enumerate(parents):
        alias = f"d{index:04d}"
        # One alias is used by crackfortran from the job cwd; the identical
        # alias below f2py's explicit build directory is used by Meson.
        (include_dir / alias).symlink_to(parent, target_is_directory=True)
        (backend_include_dir / alias).symlink_to(parent, target_is_directory=True)
        include_args.append(f"-Iincludes/{alias}")
    return staged, include_args


def _extent(dim: dict[str, Any]) -> str:
    if dim.get("ub"):
        return str(dim["ub"])
    return "*" if dim.get("assumed_size") else ":"


def _hide(
    extents: str, argument_names: list[str], parameters: dict[str, int] | None, hidden: list[str]
) -> None:
    """An extent naming neither an argument nor a local parameter is a hidden
    integer dummy the caller supplies; recorded once, in order of first use."""
    for token in re.findall(r"[A-Za-z_]\w*", extents):
        if token not in argument_names and token not in (parameters or {}):
            if token not in hidden:
                hidden.append(token)


def wrappers_for(
    record: dict[str, Any],
    subprograms: list[str],
    parameters: dict[str, int] | None = None,
    dims_override: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Flat wrapper subroutines for the named subprograms of one module.

    Returns the wrapper source text and the wrapper names, one ``w_<name>``
    per subprogram. Optional arguments are dropped -- the translation spells
    them as keyword sentinels and the differential compares the required
    surface; a specific procedure of a generic is called through the generic
    name, because the specifics are private.

    ``parameters`` are integer constants the argument dimensions name but the
    file use-imports -- a grid's ``pcols`` and ``pver``, say -- emitted as local
    PARAMETERs so f2py can fold the declared shapes (the pipeline's
    ``gen_wrapper --local-params``). A file of bare subprograms gets no
    ``use`` line at all: the callee is an external, and the borrowed module
    name would not compile.
    """
    generic_of = {
        specific: generic
        for generic, specifics in record.get("generics", {}).items()
        for specific in specifics
    }
    table = {s["name"]: s for s in record["subprograms"]}
    module = record["module"]
    is_module = record.get("is_module", True)
    # A submodule cannot be USEd; its procedures are reached through the
    # parent module whose interface declares them (#29).
    module = record.get("submodule_of") or module
    parameter_lines = [
        f"  integer, parameter :: {name} = {int(value)}"
        for name, value in (parameters or {}).items()
    ]
    pieces = ["! Machine-generated by recast (f2py-golden oracle) -- DO NOT EDIT.", ""]
    names = []
    for name in subprograms:
        sub = table[name]
        call_name = generic_of.get(name, name)
        arguments = [a for a in sub["args"] if not a.get("optional")]
        argument_names = [a["name"] for a in arguments]
        declarations = []
        hidden: list[str] = []
        for argument in arguments:
            spelled = FORTRAN_TYPES.get(argument["dtype"])
            if spelled is None:
                raise ConfigError(
                    f"{name}: argument {argument['name']!r} has dtype "
                    f"{argument['dtype']!r}, which this wrapper cannot spell; "
                    "wrap it by hand or drop the subprogram from the gate"
                )
            intent = {"IN": "in", "OUT": "out", "INOUT": "inout", "UNKNOWN": "inout"}[
                argument["intent"]
            ]
            dims = ""
            override = (dims_override or {}).get(argument["name"])
            if override and argument.get("dims"):
                # An explicit override wins: a use-imported extent is
                # invisible to f2py, and the operator names it instead (the
                # pipeline's ``gen_wrapper --dims-override``).
                dims = f"({override})"
                _hide(override, argument_names, parameters, hidden)
            elif argument.get("dims"):
                # Spelled as the source declares it: an explicit extent, ``*``
                # for an assumed-size dummy, ``:`` for an assumed-shape one.
                # f2py's interface carries either alone; what it cannot carry
                # is the mix ``(incfd, :)`` that spelling ``*`` as ``:`` made.
                dims = "(" + ", ".join(_extent(d) for d in argument["dims"]) + ")"
            declarations.append(f"  {spelled}, intent({intent}) :: {argument['name']}{dims}")
        wrapper = f"w_{name}"
        names.append(wrapper)
        use_line = [f"  use {module}, only: {call_name}"] if is_module else []
        external_line = [] if is_module else [f"  external {call_name}"]
        result_dims = sub.get("result_dims") or [] if sub["kind"] == "function" else []
        if result_dims:
            # An array-valued function result needs its extents too, spelled
            # the way a dummy's are, and f2py cannot wrap an array-valued
            # *function*: the wrapper becomes a subroutine whose ``res`` is
            # an intent(out) dummy. An extent naming neither an argument nor
            # a local parameter becomes a hidden integer dummy (#17).
            if any(d.get("ub") is None for d in result_dims):
                raise ConfigError(
                    f"{name}: array-valued result with a deferred/assumed extent "
                    "is not wrappable; wrap it by hand or drop the subprogram from the gate"
                )
            result = FORTRAN_TYPES.get(sub["result_dtype"], "real(8)")
            extents = [str(d["ub"]) for d in result_dims]
            for extent in extents:
                _hide(extent, argument_names, parameters, hidden)
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"subroutine {wrapper}({', '.join([*argument_names, *hidden, 'res'])})",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *declarations,
                f"  {result}, intent(out) :: res({', '.join(extents)})",
                *([f"  {result}, external :: {call_name}"] if not is_module else []),
                f"  res = {call_name}({', '.join(argument_names)})",
                f"end subroutine {wrapper}",
                "",
            ]
        elif sub["kind"] == "function":
            result = FORTRAN_TYPES.get(sub["result_dtype"], "real(8)")
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"function {wrapper}({', '.join([*argument_names, *hidden])}) result(res)",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *declarations,
                f"  {result} :: res",
                *([f"  {result}, external :: {call_name}"] if not is_module else []),
                f"  res = {call_name}({', '.join(argument_names)})",
                f"end function {wrapper}",
                "",
            ]
        else:
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"subroutine {wrapper}({', '.join([*argument_names, *hidden])})",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *external_line,
                *declarations,
                f"  call {call_name}({', '.join(argument_names)})",
                f"end subroutine {wrapper}",
                "",
            ]
    return "\n".join(pieces) + "\n", names


def companion_sources(facts: Facts, root: Path) -> list[Path]:
    """The sibling files this unit ``use``s, dependencies first.

    A module that takes its working precision from a kinds module one file
    over does not compile alone: gfortran wants the ``.mod``, and the file
    that would produce it was never handed to the build. The frontend already
    resolved those siblings -- ``Facts.provenance['companions']`` names them
    -- so the reference build asks the facts rather than the operator.

    ``config['extra_sources']`` stays what it always was: files from outside
    the tree, which nothing in the tree can name. Ordering is a topological
    sort over the companions' own ``use`` statements, because a Fortran
    compiler cannot read a module it has not compiled yet, and the ones that
    depend on nothing here come first.
    """
    root = _resolved_root(root)
    companions = facts.provenance.get("companions") or []
    if not isinstance(companions, list):
        raise ConfigError("Facts.provenance['companions'] must be a list")
    by_module: dict[str, tuple[dict[str, Any], Path]] = {}
    for index, companion in enumerate(companions):
        if not isinstance(companion, dict):
            raise ConfigError(f"companion {index} must be an object")
        module = str(companion.get("module", "")).lower()
        if module in by_module:
            raise ConfigError(f"duplicate companion module {module!r} in Facts provenance")
        path = _source_under_root(
            root,
            companion.get("source"),
            label=f"companion {module or index} source",
        )
        by_module[module] = (companion, path)
    ordered: list[Path] = []
    placed: set[str] = set()

    def place(name: str, stack: frozenset[str]) -> None:
        item = by_module.get(name)
        if item is None or name in placed or name in stack:
            # A cycle is not this build's to resolve -- Fortran allows mutual
            # use only through submodules, and stopping keeps the order total.
            return
        companion, path = item
        for statement in companion.get("record", {}).get("use_statements", ()):
            match = re.match(r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", statement.strip(), re.I)
            if match:
                place(match.group(1).lower(), stack | {name})
        if name in placed:
            return
        placed.add(name)
        ordered.append(path)

    for name in sorted(by_module):
        place(name, frozenset())
    return ordered


def _log_tail(output: str, limit: int = _BUILD_LOG_TAIL_CHARS) -> str:
    """The last ``limit`` characters of ``output``, cut at a line boundary and
    saying how much came before."""
    text = output.strip()
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    newline = tail.find("\n")
    if 0 <= newline < limit // 2:
        tail = tail[newline + 1 :]
    return f"… [{len(text) - len(tail)} earlier characters of the build output omitted]\n{tail}"


def _compiler_version(compiler: str) -> str:
    """The compiler's own version line. A metadata query, not a build --
    which is why it does not go through the executor: the *key* has to fold
    the version in before anyone decides whether to build at all."""
    try:
        out = subprocess.run(  # noqa: S603
            [compiler, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError as exc:
        raise ConfigError(
            f"Fortran compiler {compiler!r} is not runnable ({exc}); "
            "install gfortran or point config['fc'] at one"
        ) from exc
    if out.returncode != 0:
        raise ConfigError(f"{compiler!r} --version failed: {out.stderr.strip()[:200]}")
    return out.stdout.splitlines()[0].strip()


class F2pyGoldenOracle(Oracle):
    """Compile the untouched Fortran and hand back a callable truth module."""

    name = "f2py-golden"
    cost = "build"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        compiler = config.get("fc", "gfortran")
        digest = hashlib.sha256()
        digest.update(str(facts.provenance.get("digest")).encode())
        root = _resolved_root(config.get("root", "."))
        # Trust the bytes, not only a digest carried in mutable Facts.  This
        # also makes the key validation exercise the same root boundary as the
        # materializer before it queries a compiler or creates a workspace.
        digest.update(self._main_source_digest(facts, root).encode())
        dependencies = [*companion_sources(facts, root), *_extra_sources(config)]
        for path in sorted(dependencies, key=str):
            digest.update(str(path).encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(_compiler_version(compiler).encode())
        digest.update(config.get("fflags", DEFAULT_FLAGS).encode())
        digest.update(str(sorted((config.get("wrapper_parameters") or {}).items())).encode())
        digest.update(str(sorted((config.get("wrapper_dims") or {}).items())).encode())
        digest.update(",".join(self._subprograms(facts, config)).encode())
        return f"f2py:{facts.interface.get('module', unit.uid)}:{digest.hexdigest()[:16]}"

    def _main_source(self, facts: Facts, root: Path) -> Path:
        """The unit's own source, proven to lie under ``root``.

        An oracle that writes the source it wraps (``F2pyFlatOracle``)
        overrides this with the file it wrote.
        """
        return _source_under_root(root, facts.provenance.get("source"), label="main source")

    def _main_source_digest(self, facts: Facts, root: Path) -> str:
        """sha256 of the main source's bytes, read for the key."""
        return hashlib.sha256(self._main_source(facts, root).read_bytes()).hexdigest()

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        key = self.key(unit, facts, config)
        build = workspace / f"oracle-{key.rsplit(':', 1)[-1]}"
        build.mkdir(parents=True, exist_ok=True)

        root = _resolved_root(config.get("root", "."))
        source = self._main_source(facts, root)
        companions = companion_sources(facts, root)
        extras = _extra_sources(config)
        subprograms = self._subprograms(facts, config)
        if not subprograms:
            # Nothing callable to wrap -- a module of kind parameters, or of
            # abstract interfaces. f2py is happy to build an extension with an
            # empty ``only:`` list, and importing the result segfaults the
            # interpreter, which no ``except`` can catch and which takes the
            # whole run with it. A reference to nothing is not a reference.
            raise OracleUnavailable(
                f"{unit.uid}: no public subprogram to wrap; there is no reference to build"
            )
        wrapper_text, wrapper_names = wrappers_for(
            facts.interface,
            subprograms,
            parameters=config.get("wrapper_parameters"),
            dims_override=config.get("wrapper_dims"),
        )

        compiler = config.get("fc", "gfortran")
        module_name = f"ref_{facts.interface['module']}"
        # Companions first, then whatever the operator added, then the unit's
        # own source: gfortran compiles in argument order and a ``use`` of a
        # module later in the list is a fatal "cannot open module file".
        original_sources = [*companions, *extras, source]
        stage = Path(tempfile.mkdtemp(prefix="f2py-stage-", dir=build))
        sources, include_args = _stage_build_inputs(stage, original_sources, wrapper_text)
        # fflags remains the operator's compiler-flags string.  Source/include
        # paths never join it: NumPy splits this value internally, so appending
        # an original directory here would let whitespace and flag-looking
        # path components become compiler options.
        flags = config.get("fflags", DEFAULT_FLAGS)
        job = Job(
            argv=[
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                "--build-dir",
                "f2py-build",
                *sources,
                *include_args,
                "-m",
                module_name,
                "only:",
                *wrapper_names,
                ":",
                f"--f90flags={flags}",
                f"--f77flags={flags}",
                "--backend",
                "meson",
            ],
            cwd=stage,
            # The whole environment plus the compiler overrides: f2py needs a
            # real PATH, and the local executor passes exactly what it is given.
            # The interpreter's own bin directory rides in front so the build
            # backend (meson, ninja) installed beside numpy is found.
            env={
                **os.environ,
                "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
                "FC": compiler,
                "F90": compiler,
            },
            timeout_s=float(config.get("build_timeout", 600)),
            label=f"f2py {module_name}",
        )
        try:
            result = executor.run(job)
        except RecastError:
            raise
        except Exception as error:
            # An executor that refuses -- it cannot honestly supply what the job
            # asked for -- is the case ``OracleUnavailable`` exists for, and it
            # has to arrive as one. The runner catches ``RecastError`` and marks
            # this unit's oracle stage failed; anything else escapes it and takes
            # the whole run down, so a refusal nobody wrapped costs the other
            # units their verdicts as well as this one.
            raise OracleUnavailable(
                f"executor {getattr(executor, 'name', type(executor).__name__)!r} did not "
                f"run the f2py build for {unit.uid}: {type(error).__name__}: {error}"
            ) from error
        if not result.ok:
            log = build / "f2py.log"
            output = result.stdout + "\n" + result.stderr
            log.write_text(output)
            # The log lives in a workspace that is gone once the run returns,
            # so the error carries the end of it -- where crackfortran and the
            # compiler say what they could not parse -- not just its path.
            raise ConfigError(
                f"f2py build for {unit.uid} failed (exit {result.returncode}); log at {log}\n"
                + _log_tail(output)
            )

        sys.path.insert(0, str(stage))
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.path.remove(str(stage))
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=key,
            handle={
                "module": module,
                "wrappers": dict(zip(subprograms, wrapper_names, strict=True)),
                "build_dir": stage,
            },
            cost=self.cost,
        )

    @staticmethod
    def _subprograms(facts: Facts, config: dict[str, Any]) -> list[str]:
        named = config.get("subprograms")
        if named:
            return list(named)
        # Public only, because the wrappers `use` the module: a private
        # symbol is not importable and the build fails on the whole file.
        return [s["name"] for s in facts.interface["subprograms"] if s.get("public", True)]


def factory(**_config: Any) -> F2pyGoldenOracle:
    return F2pyGoldenOracle()
