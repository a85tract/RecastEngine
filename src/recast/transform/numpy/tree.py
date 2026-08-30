"""``TreeTranslation``: the NumPy translation of one unit *of a tree*.

``NumpyTranslation`` translates a file. A unit of a model tree also needs:

* its **use-constants** resolved -- every name it use-imports from one of
  the tree's constants modules is read out of the tree and emitted into the
  candidate's own ``<module>_use_constants.py``, so the candidate imports
  nothing that is not in its own files and the constant is the same parsed
  expression on both sides of the differential;
* the framework's calls answered by **stub tables** rather than refused;
* a **stand-in** file for every stub module the emitted header imports
  (``recast.transform.numpy.standins``);
* its **companions bundled** -- their translations carried in this
  candidate, so a call into a sibling reaches the sibling's translation;
* a **flat adapter** beside every subprogram that takes a derived-type
  dummy (``recast.transform.numpy.flat``), from the plan the frontend
  stored on ``Facts.extra``;
* **constant overrides** from the run the reference was recorded under.

All of that is mechanical and none of it is a judgement: a name that has
no initializer is left out and the rules refuse it, with the reason on the
block. What *is* the tree's own -- which modules hold constants, which are
stubs, what the framework answers -- arrives in ``TreeConventions`` from a
domain extension; the engine has no table of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from recast.fortran.tree import USE, module_sources, named_extents, parameter_names, use_imports
from recast.plugins.transform import Transform

if TYPE_CHECKING:
    from recast.model import Candidate, Facts, Unit

__all__ = ["TreeConventions", "TreeTranslation", "factory"]


@dataclass(frozen=True)
class TreeConventions:
    """What a domain extension knows about its tree."""

    kind_assumptions: dict[str, str] = field(default_factory=dict)
    constant_modules: frozenset[str] = frozenset()
    """Modules whose public entities are initialized constants a translation
    resolves rather than ports."""
    stub_modules: frozenset[str] = frozenset()
    """Modules never resolved against the tree; the stub tables answer for
    their calls and a stand-in for their imports."""
    function_stubs: dict[str, Any] = field(default_factory=dict)
    statement_stubs: dict[str, Any] = field(default_factory=dict)
    framework: dict[str, str] = field(default_factory=dict)
    """Module -> Python text a standalone run answers its calls with."""
    profile: str = "gfortran"
    frontend: str = "fortran"
    """The registered frontend that analysed the unit, for bundling its
    companions the same way."""


class TreeTranslation(Transform):
    """Fortran to NumPy for a unit of a tree, under the tree's conventions."""

    name = "recast.translate.tree-to-numpy"
    requires = ("interface", "constants", "effects")
    deterministic = True
    notes_key = "tree"
    """Where this transform's own notes go on ``Candidate.notes``."""

    def __init__(self, conventions: TreeConventions | None = None) -> None:
        from recast.transform.numpy.translate import NumpyTranslation

        self.conventions = conventions or TreeConventions()
        self._engine = NumpyTranslation()

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return self._engine.applicable(unit, facts)

    def configured(self, facts: Facts, config: dict[str, Any]) -> dict[str, Any]:
        """The operator's config with the tree's tables and use-constants merged under it."""
        c = self.conventions
        merged = {
            **config,
            "profile": config.get("profile", c.profile),
            "function_stubs": {**c.function_stubs, **(config.get("function_stubs") or {})},
            "statement_stubs": {**c.statement_stubs, **(config.get("statement_stubs") or {})},
        }
        if config.get("use_constants") is None:
            use = self._use_constants(facts, config)
            if use is not None:
                merged["use_constants"] = use
        return merged

    def _use_constants(self, facts: Facts, config: dict[str, Any]) -> dict[str, Any] | None:
        from recast.fortran.use import UnresolvedConstant, resolve

        c = self.conventions
        modules = frozenset(
            {*c.constant_modules, *(m.lower() for m in config.get("constant_modules") or ())}
        )
        wanted = use_imports(facts.interface, modules, frozenset(c.kind_assumptions))
        if not wanted:
            return None
        root = Path(config.get("root", ".")).resolve()
        files = module_sources(root, modules)
        if not files:
            return None
        # Parameters only. A module *variable* (set at run time) is the run's
        # to say: it binds through the companion's translation and a
        # recording sets it.
        parameters = parameter_names(files, c.kind_assumptions)
        wanted = [n for n in wanted if n in parameters]
        if not wanted:
            return None
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        skipped: list[str] = []
        for name in wanted:
            try:
                records = resolve([name], files)
            except UnresolvedConstant:
                skipped.append(name)
                continue
            for entry in records:
                if entry["name"] not in seen:
                    seen.add(entry["name"])
                    resolved.append(entry)
                elif entry["requested"]:
                    for r in resolved:
                        if r["name"] == entry["name"]:
                            r["requested"] = True
        if not resolved:
            return None
        return {
            "resolved": resolved,
            "module_name": facts.interface.get("module", "unit"),
            "unresolved": skipped,
            "sources": [str(s) for s in files],
        }

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        merged = self.configured(facts, config)
        candidate = self._engine.apply(unit, facts, merged)
        candidate.transform = self.name
        self._map_use_renames(candidate, facts)
        self._add_flat_adapters(candidate, facts)
        self._bundle_companions(candidate, facts, merged)
        self._add_stand_ins(candidate, facts, merged)
        self._override_constants(candidate, merged)
        # For the differential gate: the unit's own signatures, and the
        # integer value of every name that sizes a dummy array.
        candidate.notes["signatures"] = [
            {"name": s["name"], "args": [dict(a) for a in s["args"]]}
            for s in facts.interface["subprograms"]
        ]
        dims = (facts.extra or {}).get("dim_parameters") or {}
        if dims:
            candidate.notes["dims"] = {
                k: v for k, v in dims.items() if k in named_extents(facts.interface["subprograms"])
            }
        use = merged.get("use_constants")
        if use:
            self._note(candidate)["use_constants"] = {
                "resolved": [e["name"] for e in use["resolved"]],
                "unresolved": list(use.get("unresolved", ())),
                "sources": [Path(s).name for s in use.get("sources", ())],
            }
        return candidate

    def _note(self, candidate: Candidate) -> dict[str, Any]:
        note: dict[str, Any] = candidate.notes.setdefault(self.notes_key, {})
        return note

    def _map_use_renames(self, candidate: Candidate, facts: Facts) -> None:
        """``use constants, only: pi => rpi``: the source reads ``pi`` and the
        emitted module spells the resolved constant ``RPI``. The read/write
        protocol's name table maps emitted names back to source names; add
        the rename, or the check reports ``rpi`` read where ``pi`` was."""
        protocol = candidate.notes.get("rwset")
        if not protocol:
            return
        names = protocol.setdefault("names", {})
        statements = list(facts.interface.get("use_statements") or [])
        for sub in facts.interface.get("subprograms", ()):
            statements.extend(sub.get("use_statements") or [])
        kinds = self.conventions.kind_assumptions
        for statement in statements:
            match = USE.match(statement.strip())
            if not match or not match.group("only"):
                continue
            for item in match.group("only").split(","):
                if "=>" not in item:
                    continue
                local, remote = (x.strip().lower() for x in item.split("=>", 1))
                if local != remote and remote not in kinds:
                    names.setdefault(remote.upper(), local)
                    names.setdefault(remote, local)

    def _bundle_companions(
        self, candidate: Candidate, facts: Facts, config: dict[str, Any]
    ) -> None:
        """Translate the unit's companions and carry their files in this
        candidate, so a call into a sibling reaches the sibling's translation
        rather than a stand-in that has no such function. The gate stages
        only the candidate's own files, and a companion's translation is
        verified on its own run; here it is what the unit calls into.
        Recursive over the companions' companions, once each."""
        from recast.registry import REGISTRY

        root = Path(config.get("root", ".")).resolve()
        seen: set[str] = set(config.get("_bundled") or ())
        seen.add(str(facts.interface.get("module", "")).lower())
        bundled: list[str] = []
        frontend = None
        units: dict[str, Unit] = {}
        for companion in facts.provenance.get("companions") or []:
            module = str(companion.get("module", "")).lower()
            record = companion.get("record") or {}
            if module in seen or not record.get("subprograms"):
                continue
            seen.add(module)
            if frontend is None:
                frontend = REGISTRY.get("frontend", self.conventions.frontend)()
                units = {u.uid: u for u in frontend.discover(root)}
            unit = units.get(f"fortran:{module}")
            if unit is None:
                continue
            try:
                inner = self.apply(unit, frontend.analyze(unit, root), {**config, "_bundled": seen})
            except Exception as error:  # a companion that will not translate is not our failure
                self._note(candidate).setdefault("not_bundled", {})[module] = (
                    f"{type(error).__name__}: {error}"[:200]
                )
                continue
            have = {q.name for q in candidate.files}
            fresh = {p: b for p, b in inner.files.items() if p.name not in have}
            candidate.files = {**fresh, **candidate.files}
            bundled.append(module)
            seen |= set(inner.notes.get(self.notes_key, {}).get("bundled", []))
        if bundled:
            self._note(candidate)["bundled"] = bundled

    def _add_flat_adapters(self, candidate: Candidate, facts: Facts) -> None:
        """Append a ``<name>_flat`` beside every subprogram that takes a
        derived-type dummy, from the plan the frontend stored."""
        from recast.fortran.flatten import FlatPlan
        from recast.transform.numpy.flat import python_adapter

        module = facts.interface.get("module", "")
        main = next((p for p in candidate.files if p.name == f"{module}_numpy.py"), None)
        stored = (facts.extra or {}).get("flat_plans") or []
        if main is None or not stored:
            return
        plans = [FlatPlan.from_dict(d) for d in stored]
        usable = [p for p in plans if p.usable]
        if usable:
            candidate.files[main] = candidate.files[main] + python_adapter(usable).encode()
        note = self._note(candidate)
        note["flattened"] = {
            p.subprogram["name"]: [
                f"{obj.name}%{c.name}{'*' if c.written else ''}"
                for obj in p.objects
                for c in obj.components
            ]
            for p in usable
        }
        note["not_flattened"] = {
            p.subprogram["name"]: p.unsupported for p in plans if p.unsupported
        }

    def _add_stand_ins(self, candidate: Candidate, facts: Facts, config: dict[str, Any]) -> None:
        """Write the ``<module>_numpy`` files the emitted header imports for
        stub modules, so the candidate imports with nothing but its own files
        on the path."""
        from recast.transform.numpy.standins import stand_ins

        c = self.conventions
        module = facts.interface.get("module", "")
        main = next((p for p in candidate.files if p.name == f"{module}_numpy.py"), None)
        if main is None:
            return
        files, report = stand_ins(
            candidate.files[main].decode(),
            Path(config.get("root", ".")).resolve(),
            {p.name for p in candidate.files},
            modules=c.constant_modules | c.stub_modules,
            framework=c.framework,
            kind_assumptions=c.kind_assumptions,
        )
        # Stand-ins first, the unit's own files after: the bit-exact gate takes
        # the *last* ``*_numpy.py`` it stages as the module under judgement,
        # and a stand-in is named the way the emitted import spells it.
        candidate.files = {**files, **candidate.files}
        if report:
            self._note(candidate)["stand_ins"] = report

    def _override_constants(self, candidate: Candidate, config: dict[str, Any]) -> None:
        """``config["constant_overrides"]``: a run-control variable the model
        reads from its namelist has the tree's default in every resolved
        constant here, and a recording made under the namelist is a
        recording of the other value. The override rewrites the constant's
        line in the use-constants file and every stand-in, keeps the source
        line as a comment, and is recorded on the candidate. The value is the
        operator's claim about the run, not the tree's."""
        given = config.get("constant_overrides") or {}
        overrides = {str(k).lower(): v for k, v in given.items()}
        if not overrides:
            return
        applied: dict[str, list[str]] = {}
        for path, content in list(candidate.files.items()):
            if not (path.name.endswith("_use_constants.py") or path.name.endswith("_numpy.py")):
                continue
            text = content.decode()
            changed = False
            lines = []
            for line in text.splitlines():
                match = re.match(r"^([A-Za-z_]\w*) = (.*?)(\s+#.*)?$", line)
                name = match.group(1).lower() if match else None
                if match and name in overrides and match.group(1).isupper():
                    value = overrides[name]
                    literal = f"np.float64({value!r})" if isinstance(value, float) else repr(value)
                    note = f"# constant_overrides (namelist); was: {match.group(2)}"
                    line = f"{match.group(1)} = {literal}  {note}"
                    applied.setdefault(name, []).append(path.name)
                    changed = True
                lines.append(line)
            if changed:
                candidate.files[path] = ("\n".join(lines) + "\n").encode()
        if applied:
            self._note(candidate)["constant_overrides"] = applied


def factory(**config: Any) -> TreeTranslation:
    """``translate.tree``: the conventions from config, an extension's
    factory typically passing its own tables here instead."""
    return TreeTranslation(
        TreeConventions(
            kind_assumptions=dict(config.get("kind_assumptions") or {}),
            constant_modules=frozenset(m.lower() for m in config.get("constant_modules") or ()),
            stub_modules=frozenset(m.lower() for m in config.get("stub_modules") or ()),
            function_stubs=dict(config.get("function_stubs") or {}),
            statement_stubs=dict(config.get("statement_stubs") or {}),
            framework=dict(config.get("framework") or {}),
            profile=str(config.get("profile", "gfortran")),
            frontend=str(config.get("frontend", "fortran")),
        )
    )
