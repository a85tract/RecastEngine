"""The reference Frontend: Fortran source in, Units and Facts out.

This is the module the ``recast.frontends`` entry point names, and the only one
in the package that the engine calls directly. Everything under it -- parsing,
interfaces, constants, chunking, effects -- is analysis migrated from the
source pipeline; this file is the part that binds it to the
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

from recast import OUTPUT_DIRNAME, WORKSPACE_DIRNAME
from recast.errors import ConfigError, RecastError
from recast.model import Facts, Unit
from recast.plugins.frontend import Frontend

UID_PREFIX = "fortran"

SUFFIXES = frozenset({".f90", ".f95", ".f03", ".f08", ".f", ".for", ".ftn"})
"""Matched case-insensitively, so ``.F90`` (a common convention) is included.

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

MODULE_DEFINITION = re.compile(
    r"^\s*module\s+(?!(?:procedure|function|subroutine|pure|elemental|impure"
    r"|recursive|non_recursive)\b)(\w+)",
    re.IGNORECASE | re.MULTILINE,
)
"""A module statement. ``module function f(...)`` inside an interface block
is a separate-module procedure's interface, not a module named ``function``."""

SUBMODULE_DEFINITION = re.compile(
    r"^\s*submodule\s*\(\s*(\w+)(?:\s*:\s*\w+)?\s*\)\s*(\w+)", re.IGNORECASE | re.MULTILINE
)
"""``submodule (parent[:ancestor]) name`` defines ``name`` and depends on
``parent`` (#29)."""
"""``module X``, but not ``module procedure X``."""

USE_STATEMENT = re.compile(
    r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(?P<module>\w+)"
    r"(?:\s*,\s*ONLY\s*:\s*(?P<only>.*))?$",
    re.IGNORECASE | re.DOTALL,
)
"""Both spellings, including ``USE, INTRINSIC :: iso_fortran_env``."""

ONLY_ITEM = re.compile(r"^\s*(?:[\w.]+\s*=>\s*)?(\w+)\s*$")
"""One entry of an ``ONLY`` list, reduced to the name the *used* module knows
it by -- the right-hand side of a rename. Entries that are not a plain
identifier (``operator(+)``, ``assignment(=)``) do not match, and are not
names a module can be asked whether it declares."""


def only_names(clause: str | None) -> set[str]:
    """The remote names an ``ONLY`` clause asks a module for."""
    if not clause:
        return set()
    return {
        match.group(1).lower() for item in clause.split(",") if (match := ONLY_ITEM.match(item))
    }


def declared_names(record: dict[str, Any]) -> set[str]:
    """Every name a module's own interface record declares.

    Not the same as everything the module makes visible: a module that
    ``use``s another re-exports its public entities without declaring one of
    them. That gap is the point -- it is how a re-export module is told apart
    from one that answers for itself.
    """
    names = {s["name"] for s in record.get("subprograms", ())}
    names |= set(record.get("generics", {}))
    names |= set(record.get("types", {}))
    names |= set(record.get("kind_map", {}))
    for group in ("module_parameters", "module_state"):
        names |= {entry["name"] for entry in record.get(group, ())}
    return {n.lower() for n in names}


SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "build",
        "dist",
        OUTPUT_DIRNAME,
        WORKSPACE_DIRNAME,
    }
)
"""Directories that are not somebody's source.

The last two are in the list for a reason the others are not: they are the
engine's *own* output. An oracle build leaves generated wrappers under one, and
a discovery pass that reads them back finds units the previous run created --
so the same tree yields a different unit set before and after a run, and the
second run offers to translate the first one's scaffolding. ``output/`` is
normally outside the tree, which is the real fix; it is skipped by name as well
for the case where someone points ``config["output"]`` back inside, and on the
same reading that already skips ``build`` and ``dist``. ``WORKSPACE_DIRNAME``
stays for trees carrying a run from before ``output/``."""


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


DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)", re.IGNORECASE)


def _derived_out_as_inout(records: list[dict[str, Any]]) -> list[str]:
    """Re-declare every derived-type ``intent(out)`` (or intent-less) dummy
    in these records as ``INOUT`` -- subprograms and interface bodies alike,
    because a call through an interface is spelled from the interface's
    intents and the two have to agree. Returns what was changed."""
    corrected: list[str] = []
    for record in records:
        entries = [*record.get("subprograms", ()), *record.get("interfaces", {}).values()]
        for sub in entries:
            for argument in sub.get("args", ()):
                if argument.get("intent") in ("OUT", "UNKNOWN") and DERIVED.match(
                    str(argument.get("dtype"))
                ):
                    argument["intent"] = "INOUT"
                    corrected.append(
                        f"{record.get('module', '?')}/{sub['name']}/{argument['name']}"
                    )
    return corrected


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
        exclude: Iterable[str] = (),
        buffer_out_arrays: str = "unsizable",
        constant_modules: Iterable[str] = (),
        derived_intent_out_as_inout: bool = False,
        flatten: bool | dict[str, Any] = False,
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
        self.exclude = tuple(Path(d) for d in exclude)
        self.buffer_out_arrays = buffer_out_arrays
        """``"unsizable"`` (the default) or ``"all"``: which intent(out) arrays
        are the caller's buffer -- see ``interface._mark_buffer_out_arrays``."""
        self.constant_modules = frozenset(m.lower() for m in constant_modules)
        """Modules whose parameters size things in this tree; the integer
        value of every name that spells a dummy's extent is looked up there
        and stored on ``Facts.extra["dim_parameters"]``."""
        self.derived_intent_out_as_inout = derived_intent_out_as_inout
        """Re-declare a derived-type ``intent(out)`` dummy as ``inout``.

        By the standard an ``intent(out)`` dummy is undefined on entry; a
        tree that reads such an object's components on entry works only
        because the components are pointers, whose association every
        compiler in practice leaves alone. The translation follows the
        standard and hands such a routine a fresh object, which then has no
        components; this option says the program relies on the other thing,
        and ``Facts.provenance`` records every dummy it changed."""
        self.flatten = flatten
        """Plan a flat adapter for every subroutine that takes a derived-type
        dummy (``recast.fortran.flatten``) and store the plans on
        ``Facts.extra["flat_plans"]``. ``True``, or a dict of
        ``FlatConventions`` fields the tree spells differently."""
        self._module_indexes: dict[Path, dict[str, Path]] = {}
        self._submodule_parents: dict[Path, dict[str, str]] = {}
        self._analyzed: dict[tuple[str, str, str | None], dict[str, Any]] = {}

    # --- discovery -----------------------------------------------------------

    def discover(self, root: Path) -> Iterable[Unit]:
        _require_fparser()
        return list(self._walk(Path(root)))

    def _walk(self, root: Path) -> Iterator[Unit]:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(root)
            if SKIP_DIRS & set(relative.parts[:-1]):
                continue
            if any(relative.is_relative_to(directory) for directory in self.exclude):
                continue
            yield from self._units_in(path, relative)

    def _units_in(self, path: Path, rel: Path) -> Iterator[Unit]:
        from recast.fortran._parse import STD, digest
        from recast.fortran._parse import parse as parse_file
        from recast.fortran.interface import node_span, scopes_in

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

        # Every program unit the file defines -- a file holding two modules,
        # or a module and the program that uses it, is two Units, each
        # analyzed under its own name. A submodule is a module scope.
        for mod_name, scope_kind, scope in scopes_in(ast, path):
            kind = "module" if scope_kind in ("module", "submodule") else scope_kind
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
        from recast.fortran._parse import STD, digest, f03
        from recast.fortran._parse import parse as parse_file
        from recast.fortran.effects import side_channels
        from recast.fortran.interface import _scope_of, companion_externals, subprogram_key
        from recast.fortran.rwset import block_rwsets, scope_for

        path = self._source_of(unit, Path(root))
        if "parse_error" in unit.attrs:
            raise UnparsableSource(
                f"{unit.uid}: {path} did not parse -- {unit.attrs['parse_error']}"
            )

        # Kinds the tree states before kinds the operator assumes: a sibling
        # file that defines ``wp`` is evidence, and the operator's table is
        # for what no file in the tree says. The operator still wins where
        # both speak, because an override nothing can outvote is not one.
        # Which program unit of the file this is: its uid names it. A file
        # of bare subprograms borrows its stem and has one scope, the file.
        own = (unit.parent or unit.uid).split(":", 1)[1]
        scope_name = None if unit.kind == "file" else own
        tree_kinds = self._tree_kinds(path, Path(root), own)
        record = interface_mod.extract(
            path,
            kind_assumptions={
                **{name: found["dtype"] for name, found in tree_kinds.items()},
                **self.kind_assumptions,
            },
            intent_overrides=self.intent_overrides,
            buffer_out_arrays=self.buffer_out_arrays,
            scope=scope_name,
        )
        consts = constants_mod.extract(
            path, extern_names=set(self.extern_constants), scope=scope_name
        )

        _mod_name, _spec, scope = _scope_of(parse_file(path), path, scope_name)
        nodes = dict(_subprograms_of(scope))
        if isinstance(scope, f03.Main_Program):
            nodes[own] = scope  # the program body is a subprogram of its own unit
        defined = set(nodes)
        plain_names = {key.rsplit("/", 1)[-1] for key in defined}

        module_uid = unit.parent or unit.uid
        wanted = self._selected(unit, defined)

        subprograms = [s for s in record["subprograms"] if subprogram_key(s) in wanted]

        # The read/write scope has to know the sibling modules' procedures --
        # which names are calls, and which positions they write -- or every
        # call into one scores as reads of its actuals and no write. That is
        # the fact the pipeline's ``--companions`` carried into its check, and
        # ``companion_externals`` derives it from each sibling's own record;
        # a use-rename is looked up under the local spelling the call uses.
        # The operator's table wins where both name a procedure.
        companions, unresolved = self._companions(record, path, Path(root))
        # A submodule's procedures belong to its parent's namespace -- `use
        # parent` reaches them -- so the parent's translation re-exports them
        # (#29). Which submodules, and what they define, is a fact about the
        # tree, and this is where the tree is read.
        submodules: dict[str, list[str]] = {}
        if record.get("is_module") and not record.get("submodule_of"):
            for name in self._submodules_of(str(record["module"]).lower(), Path(root)):
                source = self._module_index(Path(root).resolve()).get(name)
                record_of = self._readable(source, interface_mod.extract, name) if source else None
                if record_of is None:
                    continue
                exported = [s["name"] for s in record_of["subprograms"] if not s.get("host")]
                if exported:
                    submodules[name] = exported
        if submodules:
            record = {**record, "submodules": submodules}
        corrected = (
            _derived_out_as_inout([record, *(c["record"] for c in companions)])
            if self.derived_intent_out_as_inout
            else []
        )
        externals = dict(self.externals)
        for companion in companions:
            table = companion_externals(companion["record"])
            for local, remote in companion["renames"].items():
                if remote in table:
                    table[local] = table[remote]
            for name, entry in table.items():
                externals.setdefault(name, entry)

        callgraph: dict[str, list[str]] = {}
        effects: dict[str, Any] = {}
        for sub in subprograms:
            sub_name = subprogram_key(sub)
            sub_uid = f"{module_uid}/{sub_name}"
            callgraph[sub_uid] = [f"{module_uid}/{c}" for c in sub["calls"]] + _external_calls(
                nodes[sub_name], plain_names
            )
            scope = scope_for(
                record,
                sub_name,
                externals=externals,
                companions=tuple(c["record"] for c in companions),
            )
            effects[sub_uid] = {
                "reads": sub["module_state_read"],
                "writes": sub["module_state_written"],
                "optional_args": sub["present_calls"],
                **side_channels(nodes[sub_name]),
                # Per block, so a Verifier comparing against a translation can
                # name the piece of code that disagrees rather than the routine.
                "blocks": block_rwsets(nodes[sub_name], scope),
            }

        facts = Facts(
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
                # Kinds read out of a sibling file rather than assumed. Kept
                # apart from the assumptions above so that provenance says
                # which of the two a dtype rests on -- and names the file, so
                # a wrong precision is one lookup rather than a bisection.
                "kind_sources": tree_kinds,
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
                **({"derived_intent_out_as_inout": corrected} if corrected else {}),
            },
        )
        self._tree_facts(facts, Path(root), path)
        return facts

    def _tree_facts(self, facts: Facts, root: Path, path: Path) -> None:
        """What the tree says about this unit's interface, on ``Facts.extra``:
        the integer value of every name that sizes a dummy array, and the
        flat-adapter plans -- computed once here, read by the oracle, the
        transform and the recorder."""
        from recast.fortran.tree import integer_parameters, named_extents

        if self.constant_modules:
            names = named_extents(facts.interface.get("subprograms", []))
            values = integer_parameters(
                names, root, self.constant_modules, (path,), self.kind_assumptions
            )
            if values:
                facts.extra["dim_parameters"] = values
        if self.flatten:
            from recast.fortran.flatten import FlatConventions, plans_for

            spelled = self.flatten if isinstance(self.flatten, dict) else {}
            conventions = FlatConventions(
                kind_assumptions=dict(self.kind_assumptions),
                constant_modules=self.constant_modules,
                stub_modules=self.stub_modules,
                **spelled,
            )
            plans = plans_for(facts, root, conventions)
            if plans:
                facts.extra["flat_plans"] = [p.to_dict() for p in plans]

    def _tree_kinds(
        self, path: Path, root: Path, own: str | None = None
    ) -> dict[str, dict[str, str]]:
        """Kind parameters this file ``use``s that another file in the tree defines.

        ``integer, parameter :: wp = real64`` in one file and ``real(wp)``
        in the next is how nearly every library in the corpus spells its
        working precision, and reading only the second file leaves ``wp``
        unresolved. Unresolved is the right answer when nothing states it --
        but here something does, one file over, and the tree is already
        indexed by module name for the companion walk.

        Returns ``{kind name: {"dtype", "module", "source"}}``, keyed by the
        name the *defining* module gives it, which is what
        ``interface.kind_aliases_from_use`` matches a local rename against.
        Follows a ``use`` onward on the same rule as ``_companions``: always
        through a bare one, and through an ``only`` list whose names the used
        module does not itself declare, which is what a re-export module looks
        like. First definition wins, in module-name order, so two files that
        disagree give the same answer every run.
        """
        from recast.fortran import interface as interface_mod
        from recast.fortran._parse import f03, parse, walk

        resolved_root = root.resolve()
        index = self._module_index(resolved_root)
        found: dict[str, dict[str, str]] = {}
        ast = parse(path)
        pending = [str(u) for u in walk(ast, f03.Use_Stmt)]
        parent = interface_mod.submodule_parent(ast)
        if parent:
            # A submodule's kinds come from its parent by host association,
            # the same way ``extract`` gives it a synthetic ``USE parent``.
            pending.insert(0, f"USE {parent}")
        seen: set[str] = set()
        while pending:
            statement = pending.pop(0)
            match = USE_STATEMENT.match(statement.strip())
            if not match:
                continue
            module = match.group("module").lower()
            if module in seen or module in INTRINSIC_MODULES or module in self.stub_modules:
                continue
            seen.add(module)
            source = index.get(module)
            if source is None or module == own:
                continue
            # A sibling that does not parse costs the kinds it would have
            # supplied and nothing else; ``_companions`` records the why.
            record_of = self._readable(source, interface_mod.extract, module)
            if record_of is None:
                continue
            for name, dtype in sorted(record_of.get("kind_map", {}).items()):
                found.setdefault(
                    name,
                    {
                        "dtype": dtype,
                        "module": module,
                        "source": str(source.relative_to(resolved_root)),
                    },
                )
            if self._carries_on(match, record_of):
                pending.extend(record_of.get("use_statements", ()))
        return found

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
        # Use association is transitive through a bare ``use``: if this module
        # bare-uses B and B bare-uses C, C's public entities are visible here
        # too, and a call into one of them has to resolve. A ``use, only:``
        # does not carry anything further, so the walk stops at those.
        pending = list(record.get("use_statements", ()))
        seen_statements: set[str] = set()
        while pending:
            statement = pending.pop(0)
            if statement in seen_statements:
                continue
            seen_statements.add(statement)
            match = USE_STATEMENT.match(statement.strip())
            if not match:
                continue
            module = match.group("module").lower()
            if module in found or module == own:
                continue
            if module in INTRINSIC_MODULES or module in self.stub_modules:
                continue
            source = index.get(module)
            if source is None:
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
                record_of = self._extracted(source, "interface", interface_mod.extract, module)
                constants_of = self._extracted(source, "constants", constants_mod.extract, module)
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
            only = (
                sorted(
                    {
                        item.split("=>", 1)[0].strip().lower()
                        for item in match.group("only").split(",")
                        if item.strip()
                    }
                )
                if match.group("only") is not None
                else None
            )
            found[module] = {
                "module": module,
                "source": str(source.relative_to(resolved_root)),
                "record": record_of,
                "constants": constants_of,
                "renames": renames,
                # The names a ``use, only:`` lets in, spelled as this module
                # sees them; ``None`` for a bare use. What is not let in is
                # not visible here, and a same-named local, alias or
                # associate of this module must not be mistaken for it.
                "only": only,
            }
            if self._carries_on(match, record_of):
                pending.extend(record_of.get("use_statements", ()))
        return list(found.values()), unresolved

    @staticmethod
    def _carries_on(match: re.Match[str], record: dict[str, Any]) -> bool:
        """Whether to follow a resolved module's own ``use`` statements.

        A bare ``use`` always carries: every public entity of that module is
        visible here, including the ones it use-associated itself.

        A ``use, only:`` carries nothing further *as a rule* -- but only when
        the module can answer for the names asked of it. A module that is
        nothing but ``use basic; use strings`` re-exports; the only-list then
        names entities it does not declare, and stopping there leaves them
        unresolved, as external calls and UNKNOWN kinds, with the file that
        defines them sitting in the same tree. So the walk continues exactly
        when something asked for is not there.
        """
        if match.group("only") is None:
            return True
        return bool(only_names(match.group("only")) - declared_names(record))

    def _readable(
        self, source: Path, extract: Any, scope: str | None = None
    ) -> dict[str, Any] | None:
        """``_extracted`` for a file this unit only consults, not one it is.

        ``None`` where the extraction raised: fparser reports an unparsable
        file with any of several unrelated exception types, and a sibling's
        syntax is not a reason to fail the unit that merely ``use``s it.
        """
        try:
            return self._extracted(source, "interface", extract, scope)
        except Exception:
            return None

    def _extracted(
        self, source: Path, kind: str, extract: Any, scope: str | None = None
    ) -> dict[str, Any]:
        """One extraction per (file revision, kind, scope), however many units
        want it. ``scope`` names the program unit of the file -- the module a
        ``use`` asked for -- so a file holding two modules answers for the
        right one.

        A tree of forty modules is forty analyses, and without this each would
        re-extract every sibling it depends on.
        """
        from recast.fortran._parse import digest

        key = (digest(source), kind, scope)
        cached = self._analyzed.get(key)
        if cached is None:
            if kind == "interface":
                cached = extract(
                    source,
                    kind_assumptions=self.kind_assumptions,
                    buffer_out_arrays=self.buffer_out_arrays,
                    scope=scope,
                )
            else:
                cached = extract(source, extern_names=set(self.extern_constants), scope=scope)
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
        parents: dict[str, str] = {}
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
            for parent, name in SUBMODULE_DEFINITION.findall(text):
                index.setdefault(name.lower(), candidate)
                parents.setdefault(name.lower(), parent.lower())
        self._module_indexes[resolved] = index
        self._submodule_parents[resolved] = parents
        return index

    def _submodules_of(self, module: str, root: Path) -> list[str]:
        """The submodules of ``module`` the tree defines, in name order."""
        resolved = root.resolve()
        self._module_index(resolved)
        parents = self._submodule_parents.get(resolved, {})
        return sorted(name for name, parent in parents.items() if parent == module)

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

        ``wanted`` holds host-qualified keys; the constants record keys on the
        subprogram's own name, because two internal procedures of one name
        hoist to the same constant anyway -- the name is derived from the
        digits, not from where they were written.
        """
        wanted = {key.rsplit("/", 1)[-1] for key in wanted}
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
        exclude=config.get("exclude") or (),
        extern_constants=config.get("extern_constants", ()),
        intent_overrides=config.get("intent_overrides"),
        externals=config.get("externals"),
    )
