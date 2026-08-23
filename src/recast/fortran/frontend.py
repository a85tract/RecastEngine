"""The reference Frontend: Fortran source in, Units and Facts out.

This is the module the ``recast.frontends`` entry point names, and the only one
in the package that the engine calls directly. Everything under it -- parsing,
interfaces, constants, chunking, effects -- is analysis migrated from the
CESM-language-translator pipeline; this file is the part that binds it to the
``Frontend`` contract and decides nothing else.

Two properties are worth stating because they are easy to break later.

*It imports no parser at import time.* fparser2 arrives with the
``recast-engine[fortran]`` extra. Registering the plugin must not require it,
so an installation without the extra still gets a working ``recast doctor`` and
a plugin list that mentions ``fortran``; the missing dependency surfaces on the
first ``discover`` or ``analyze`` call, named, with the install line.

*It reports, it does not repair.* A file that will not parse becomes a Unit
carrying the parse error, not a silently absent one -- a discovery pass that
quietly drops the eleven files it choked on is how a translation campaign
reports 96% coverage of a source tree it never read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from recast import WORKSPACE_DIRNAME
from recast.errors import ConfigError, RecastError
from recast.model import Facts, Unit
from recast.plugins.frontend import Frontend

UID_PREFIX = "fortran"

SUFFIXES = frozenset({".f90", ".f95", ".f03", ".f08", ".f", ".for", ".ftn"})
"""Matched case-insensitively, so ``.F90`` (the CESM convention) is included.

Fixed-form suffixes are here because fparser's reader picks free or fixed form
from the extension itself. A ``.F90`` that still needs cpp run over it is the
``preprocess`` hook's problem, not this list's.
"""

INTRINSIC_MODULES = frozenset(
    {
        "iso_fortran_env",
        "iso_c_binding",
        "ieee_arithmetic",
        "ieee_exceptions",
        "ieee_features",
        "omp_lib",
        "omp_lib_kinds",
        "openacc",
    }
)
"""Modules the standard (or a compiler) provides, so no file in a tree defines them.

A dependency resolver that did not know these would go looking for a
``iso_fortran_env`` to translate, and the pipeline this came from does exactly
that -- it is one of the defects its author catalogued.
"""

MODULE_DEFINITION = re.compile(r"^\s*module\s+(?!procedure\b)(\w+)", re.IGNORECASE | re.MULTILINE)
"""``module X``, but not ``module procedure X``."""

USE_STATEMENT = re.compile(
    r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(?P<module>\w+)"
    r"(?:\s*,\s*ONLY\s*:\s*(?P<only>.*))?$",
    re.IGNORECASE | re.DOTALL,
)
"""Both spellings, including ``USE, INTRINSIC :: iso_fortran_env``."""

SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".venv", "build", "dist", WORKSPACE_DIRNAME}
)
"""Directories that are not somebody's source.

``WORKSPACE_DIRNAME`` is in the list for a reason the others are not: it is the
engine's *own* output. An oracle build leaves generated wrappers under it, and
a discovery pass that reads them back finds units the previous run created --
so the same tree yields a different unit set before and after a run, and the
second run offers to translate the first one's scaffolding."""


class UnparsableSource(RecastError):
    """A source file the parser rejected, reported at ``analyze`` time."""


def _require_fparser() -> None:
    """Import the parser, or explain which extra is missing.

    Any other ``ImportError`` is a real bug and is re-raised untouched -- being
    helpful about the optional dependency must not swallow a typo in this
    package's own imports.
    """
    try:
        import recast.fortran._parse  # noqa: F401
    except ImportError as exc:
        if (exc.name or "").split(".")[0] != "fparser":
            raise
        raise ConfigError(
            "the fortran frontend needs fparser2, which is not installed. "
            "Install it with: pip install 'recast-engine[fortran]'"
        ) from exc


def _subprograms_of(scope: Any) -> list[tuple[str, Any]]:
    """``(lowercased name, node)`` for every subprogram in a scope, in source order."""
    from recast.fortran._parse import f03, walk

    out = []
    for sub in walk(scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        if _inside_interface(sub):
            # An interface body declares a signature and has no code; an
            # abstract interface `func` beside a real `func` would otherwise
            # be discovered twice under one uid.
            continue
        stmt = walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        name = str(stmt.children[1]).lower()
        host = _host_of(sub)
        # An internal procedure is named under its host: two hosts may each
        # contain a `func`, and one uid per subprogram is the contract.
        out.append((f"{host}/{name}" if host else name, sub))
    return out


def _host_of(node: Any) -> str | None:
    """The name of the subprogram this one is contained in, if any."""
    from recast.fortran._parse import f03, walk

    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            stmt = walk(parent, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
            return str(stmt.children[1]).lower()
        parent = getattr(parent, "parent", None)
    return None


def _inside_interface(node: Any) -> bool:
    from recast.fortran._parse import f03

    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, f03.Interface_Block):
            return True
        parent = getattr(parent, "parent", None)
    return False


def _external_calls(sub: Any, known: set[str]) -> list[str]:
    """Names this subprogram calls that are not defined alongside it.

    ``interface.extract_subprogram`` deliberately reports only resolved calls,
    because that is what a translation order needs. The call graph needs the
    other half too: an unresolved name is either a Unit in another file or a
    dependency nobody has accounted for, and both are worth seeing.
    """
    from recast.fortran._parse import f03, walk

    return sorted(
        {
            name
            for call in walk(sub, f03.Call_Stmt)
            if (name := str(call.children[0]).lower()) not in known
        }
    )


class FortranFrontend(Frontend):
    """Fortran 2008 as parsed by fparser2.

    Units come at two granularities and both are emitted: one per file for the
    module or program it defines, and one per subprogram inside it, with
    ``parent`` set. A Transform that rewrites whole modules and a Transform
    that translates one kernel then select the granularity they want instead of
    each re-deriving it.
    """

    name = "fortran"
    languages = ("fortran",)

    def __init__(
        self,
        *,
        kind_assumptions: dict[str, str] | None = None,
        extern_constants: Iterable[str] = (),
        intent_overrides: dict[str, Any] | None = None,
        externals: dict[str, dict[str, Any]] | None = None,
        stub_modules: Iterable[str] = (),
    ) -> None:
        """
        ``kind_assumptions`` maps kind parameters this tree use-imports from
        sources that are not in it (``{"r8": "float64"}``). ``extern_constants``
        names constants a sibling translation already defines, so that an
        expression over them classifies as resolvable rather than as a refusal.
        ``intent_overrides`` is ``{subprogram: {arg: "IN"|"OUT"|"INOUT"}}`` for
        dummy arguments the source declares no intent for. ``externals`` is
        ``{procedure: {"out_positions": [...]}}`` for procedures called from
        this tree whose source is not in it. ``stub_modules`` names modules a
        unit may ``use`` that are not to be resolved against the tree even if
        it contains them -- a framework whose calls a domain package answers
        with stubs rather than with a translation.

        All four are configuration rather than module state on purpose: the
        command-line ancestor of this frontend kept them in globals and in
        files it went looking for, which made the answer depend on what had run
        before and on what happened to be on disk beside the source.

        They are also all recorded in ``Facts.provenance``. Each one is a fact
        the source does not state, so Facts that carry the answer without
        carrying the assumption cannot be checked by anyone reading them later.
        """
        self.kind_assumptions = dict(kind_assumptions or {})
        self.extern_constants = frozenset(extern_constants)
        self.intent_overrides = dict(intent_overrides or {})
        self.externals = dict(externals or {})
        self.stub_modules = frozenset(m.lower() for m in stub_modules)
        self._module_indexes: dict[Path, dict[str, Path]] = {}
        self._analyzed: dict[tuple[str, str], dict[str, Any]] = {}

    # --- discovery -----------------------------------------------------------

    def discover(self, root: Path) -> Iterable[Unit]:
        _require_fparser()
        return list(self._walk(Path(root)))

    def _walk(self, root: Path) -> Iterator[Unit]:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            if SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
                continue
            yield from self._units_in(path, path.relative_to(root))

    def _units_in(self, path: Path, rel: Path) -> Iterator[Unit]:
        from recast.fortran._parse import STD, digest, f03, walk
        from recast.fortran._parse import parse as parse_file
        from recast.fortran.interface import _scope_of, node_span

        sha = digest(path)
        try:
            ast = parse_file(path)
        except Exception as exc:  # fparser raises several unrelated types
            # Named for the file, because the module name is exactly what we
            # failed to learn. Analysis of this Unit will refuse; discovery of
            # the rest of the tree carries on.
            yield Unit(
                uid=f"{UID_PREFIX}:{rel.stem.lower()}",
                kind="file",
                sources=(rel,),
                attrs={"digest": sha, "parse_error": f"{type(exc).__name__}: {exc}"},
            )
            return

        mod_name, _spec, scope = _scope_of(ast, path)
        if walk(ast, f03.Module):
            kind = "module"
        else:
            kind = "program" if walk(ast, f03.Main_Program) else "file"
        module_uid = f"{UID_PREFIX}:{mod_name}"
        yield Unit(uid=module_uid, kind=kind, sources=(rel,), attrs={"digest": sha, "std": STD})

        for sub_name, sub in _subprograms_of(scope):
            yield Unit(
                uid=f"{module_uid}/{sub_name}",
                kind="subprogram",
                sources=(rel,),
                parent=module_uid,
                attrs={"digest": sha, "line_span": list(node_span(sub))},
            )

    # --- analysis ------------------------------------------------------------

    def analyze(self, unit: Unit, root: Path) -> Facts:
        _require_fparser()
        from recast.fortran import constants as constants_mod
        from recast.fortran import interface as interface_mod
        from recast.fortran._parse import STD, digest
        from recast.fortran._parse import parse as parse_file
        from recast.fortran.effects import side_channels
        from recast.fortran.interface import _scope_of, subprogram_key
        from recast.fortran.rwset import block_rwsets, scope_for

        path = self._source_of(unit, Path(root))
        if "parse_error" in unit.attrs:
            raise UnparsableSource(
                f"{unit.uid}: {path} did not parse -- {unit.attrs['parse_error']}"
            )

        record = interface_mod.extract(
            path,
            kind_assumptions=self.kind_assumptions,
            intent_overrides=self.intent_overrides,
        )
        consts = constants_mod.extract(path, extern_names=set(self.extern_constants))

        _mod_name, _spec, scope = _scope_of(parse_file(path), path)
        nodes = dict(_subprograms_of(scope))
        defined = set(nodes)
        plain_names = {key.rsplit("/", 1)[-1] for key in defined}

        module_uid = unit.parent or unit.uid
        wanted = self._selected(unit, defined)

        subprograms = [s for s in record["subprograms"] if subprogram_key(s) in wanted]
        callgraph: dict[str, list[str]] = {}
        effects: dict[str, Any] = {}
        for sub in subprograms:
            sub_name = subprogram_key(sub)
            sub_uid = f"{module_uid}/{sub_name}"
            callgraph[sub_uid] = [f"{module_uid}/{c}" for c in sub["calls"]] + _external_calls(
                nodes[sub_name], plain_names
            )
            scope = scope_for(record, sub_name, externals=self.externals)
            effects[sub_uid] = {
                "reads": sub["module_state_read"],
                "writes": sub["module_state_written"],
                "optional_args": sub["present_calls"],
                **side_channels(nodes[sub_name]),
                # Per block, so a Verifier comparing against a translation can
                # name the piece of code that disagrees rather than the routine.
                "blocks": block_rwsets(nodes[sub_name], scope),
            }

        companions, unresolved = self._companions(record, path, Path(root))
        return Facts(
            unit=unit.uid,
            interface={**record, "subprograms": subprograms},
            constants=self._narrow_constants(consts, wanted),
            callgraph=callgraph,
            effects=effects,
            provenance={
                "frontend": self.name,
                "parser": "fparser2",
                "standard": STD,
                "source": str(unit.sources[0]) if unit.sources else str(path),
                "digest": digest(path),
                "kind_assumptions": dict(self.kind_assumptions),
                "extern_constants": sorted(self.extern_constants),
                "intent_overrides": dict(self.intent_overrides),
                "externals": dict(self.externals),
                "stub_modules": sorted(self.stub_modules),
                # What this unit ``use``s that the same tree defines. The
                # translation of a module that calls into a sibling needs the
                # sibling's declarations, and a resolver that ran on the
                # operator's config alone could only answer for a tree the
                # operator had already mapped by hand.
                "companions": companions,
                # A module the tree defines and this one uses, that could not
                # be read. Its calls will refuse; this says why.
                "companions_unresolved": unresolved,
            },
        )

    def _companions(
        self, record: dict[str, Any], path: Path, root: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Modules this one ``use``s that another file in the tree defines.

        Direct dependencies only, as the pipeline's ``auto_translate`` resolves
        them: a companion's own companions are its translation's business, not
        this one's. Three kinds of ``use`` are not companions -- an intrinsic
        module, one the operator listed as stubbed, and one nothing in the tree
        defines, which is external by definition and reaches the translation as
        a refusal or a shim instead of as a guess.

        The target's naming (what to import it as, what to call it in the
        emitted source) is deliberately not decided here. That is a property of
        the backend, and two backends over these same facts should not have to
        agree on it.
        """
        from recast.fortran import constants as constants_mod
        from recast.fortran import interface as interface_mod

        resolved_root = root.resolve()
        index = self._module_index(resolved_root)
        own = str(record.get("module", "")).lower()
        found: dict[str, dict[str, Any]] = {}
        unresolved: list[dict[str, str]] = []
        for statement in record.get("use_statements", ()):
            match = USE_STATEMENT.match(statement.strip())
            if not match:
                continue
            module = match.group("module").lower()
            if module in found or module == own:
                continue
            if module in INTRINSIC_MODULES or module in self.stub_modules:
                continue
            source = index.get(module)
            if source is None or source.resolve() == path.resolve():
                continue
            renames = {
                local.strip().lower(): remote.strip().lower()
                for local, remote in (
                    item.split("=>", 1)
                    for item in (match.group("only") or "").split(",")
                    if "=>" in item
                )
            }
            try:
                record_of = self._extracted(source, "interface", interface_mod.extract)
                constants_of = self._extracted(source, "constants", constants_mod.extract)
            except Exception as error:  # fparser raises several unrelated types
                # A sibling that does not parse is not this unit's failure. It
                # drops out of the companion set, its calls refuse the way any
                # unresolved call does, and the reason is recorded rather than
                # left to be re-derived from a stack trace.
                unresolved.append(
                    {
                        "module": module,
                        "source": str(source.relative_to(resolved_root)),
                        "reason": f"{type(error).__name__}: {error}".split("\n")[0][:200],
                    }
                )
                continue
            found[module] = {
                "module": module,
                "source": str(source.relative_to(resolved_root)),
                "record": record_of,
                "constants": constants_of,
                "renames": renames,
            }
        return list(found.values()), unresolved

    def _extracted(self, source: Path, kind: str, extract: Any) -> dict[str, Any]:
        """One extraction per (file revision, kind), however many units want it.

        A tree of forty modules is forty analyses, and without this each would
        re-extract every sibling it depends on.
        """
        from recast.fortran._parse import digest

        key = (digest(source), kind)
        cached = self._analyzed.get(key)
        if cached is None:
            if kind == "interface":
                cached = extract(source, kind_assumptions=self.kind_assumptions)
            else:
                cached = extract(source, extern_names=set(self.extern_constants))
            self._analyzed[key] = cached
        return cached

    def _module_index(self, root: Path) -> dict[str, Path]:
        """``module name -> the file that defines it``, for one tree.

        Read with a regex rather than by parsing: this runs over every file in
        the tree to answer a question about one of them, and a module statement
        is not a construct a parser is needed for. Comments and line
        continuations are removed first, which is the whole of the syntax that
        can hide one.
        """
        resolved = root.resolve()
        index = self._module_indexes.get(resolved)
        if index is not None:
            return index
        index = {}
        for candidate in sorted(resolved.rglob("*")):
            if candidate.suffix.lower() not in SUFFIXES or not candidate.is_file():
                continue
            if SKIP_DIRS & set(candidate.relative_to(resolved).parts[:-1]):
                continue
            text = candidate.read_text(errors="replace")
            text = re.sub(r"!.*", "", text)
            text = re.sub(r"&\s*\n\s*&?", " ", text)
            for name in MODULE_DEFINITION.findall(text):
                index.setdefault(name.lower(), candidate)
        self._module_indexes[resolved] = index
        return index

    def _source_of(self, unit: Unit, root: Path) -> Path:
        if not unit.sources:
            raise ConfigError(
                f"unit {unit.uid!r} carries no source file; it did not come from {self.name}"
            )
        path = root / unit.sources[0]
        if not path.is_file():
            raise ConfigError(
                f"unit {unit.uid!r} names {unit.sources[0]}, which is not a file under {root}"
            )
        return path

    @staticmethod
    def _selected(unit: Unit, defined: set[str]) -> set[str]:
        """Which subprograms this Unit's Facts should cover.

        A module-level Unit covers all of them; a subprogram Unit covers one.
        Narrowing here rather than in each analysis is what lets a Transform
        that works on one kernel be handed Facts it can trust are about that
        kernel.
        """
        if unit.kind != "subprogram":
            return defined
        name = unit.uid[len(unit.parent) + 1 :] if unit.parent else unit.uid.rsplit("/", 1)[-1]
        if name not in defined:
            raise UnparsableSource(
                f"unit {unit.uid!r} names subprogram {name!r}, which the source no longer defines"
            )
        return {name}

    @staticmethod
    def _narrow_constants(consts: dict[str, Any], wanted: set[str]) -> dict[str, Any]:
        """Drop the parts of a whole-file constants record that are about other subprograms.

        Module parameters stay whatever the Unit is: a subprogram's translated
        constants module still has to define the module-level parameters it
        reads.
        """
        hoisted = {}
        for name, entry in consts["hoisted_literals"].items():
            locations = [loc for loc in entry["locations"] if loc.split(":", 1)[0] in wanted]
            if locations:
                hoisted[name] = {**entry, "locations": locations}
        return {
            **consts,
            "local_parameters": [
                p for p in consts["local_parameters"] if p["subprogram"] in wanted
            ],
            "literal_map": {k: v for k, v in consts["literal_map"].items() if k in wanted},
            "hoisted_literals": hoisted,
        }


def factory(**config: Any) -> FortranFrontend:
    return FortranFrontend(
        kind_assumptions=config.get("kind_assumptions"),
        extern_constants=config.get("extern_constants", ()),
        intent_overrides=config.get("intent_overrides"),
        externals=config.get("externals"),
    )
