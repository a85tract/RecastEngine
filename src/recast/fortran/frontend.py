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
        stmt = walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        out.append((str(stmt.children[1]).lower(), sub))
    return out


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
    ) -> None:
        """
        ``kind_assumptions`` maps kind parameters this tree use-imports from
        sources that are not in it (``{"r8": "float64"}``). ``extern_constants``
        names constants a sibling translation already defines, so that an
        expression over them classifies as resolvable rather than as a refusal.
        ``intent_overrides`` is ``{subprogram: {arg: "IN"|"OUT"|"INOUT"}}`` for
        dummy arguments the source declares no intent for. ``externals`` is
        ``{procedure: {"out_positions": [...]}}`` for procedures called from
        this tree whose source is not in it.

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
        from recast.fortran.interface import _scope_of
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

        module_uid = unit.parent or unit.uid
        wanted = self._selected(unit, defined)

        subprograms = [s for s in record["subprograms"] if s["name"] in wanted]
        callgraph: dict[str, list[str]] = {}
        effects: dict[str, Any] = {}
        for sub in subprograms:
            sub_name = sub["name"]
            sub_uid = f"{module_uid}/{sub_name}"
            callgraph[sub_uid] = [f"{module_uid}/{c}" for c in sub["calls"]] + _external_calls(
                nodes[sub_name], defined
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
            },
        )

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
        name = unit.uid.rsplit("/", 1)[-1]
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
