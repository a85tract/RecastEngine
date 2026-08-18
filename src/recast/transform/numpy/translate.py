"""The Fortran-to-NumPy Transform: the emitter, bound to the plugin contract.

This is what the ``translate`` recipe's ``transform`` stage names, and the
end of the split that ``docs/splitting-the-translator.md`` describes: the
pipeline's ``main()`` was argument parsing, file discovery, table loading,
and rendering in one function, and everything except the plumbing has moved
into the layers below. What is left here is exactly a ``Transform.apply``:
take a Unit and its Facts, consult the operator's tables, produce a
Candidate, decide nothing about correctness.

The Candidate carries the whole product of a translation: the generated
module, its constants module, its use-constants module when the source
imports constants from modules that are not being translated -- and, in
``notes``, the block report and the name-protocol table. The report says
which blocks are mechanical and which are deferred, and the deferred list is
the agent queue: a partial Candidate with an honest list of what it could
not do is a normal result, and it is what the agentic placement of this same
slot consumes next.

Deterministic, and checkably so: the same Unit, Facts and config produce the
same ``Candidate.digest()``, which conformance holds this Transform to. The
one integrity rule enforced here is that the source on disk must still be
the source the Facts describe -- Facts carry the digest of what was
analyzed, and translating a file that changed since analysis would produce a
Candidate whose provenance quietly lies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform
from recast.transform.numpy.constants import constants_module, use_constants_module
from recast.transform.numpy.expressions import Remote
from recast.transform.numpy.modules import Modules
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.numpy.vocabulary import pysafe
from recast.transform.profiles import DEFAULT, PROFILES

__all__ = ["NumpyTranslation", "companion_tables", "factory"]


def companion_tables(
    companions: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Remote], dict[str, str], tuple[str, ...]]:
    """The four views of the companion configuration.

    Each entry describes one already-translated sibling module: its interface
    ``record``, the ``alias`` the generated code calls it by, the ``module_py``
    file that alias imports, and any use-``renames``. One entry feeds four
    consumers -- semantics wants the records, the emitters want the remotes
    and the globals, the module renderer wants the import lines -- and
    deriving all four here keeps them from drifting apart.
    """
    records: list[dict[str, Any]] = []
    remotes: dict[str, Remote] = {}
    globals_: dict[str, str] = {}
    imports: list[str] = []
    for companion in companions:
        record = companion["record"]
        records.append(record)
        alias = companion["alias"]
        imports.append(f"import {companion['module_py']} as {alias}")
        subprograms = {s["name"]: s for s in record["subprograms"]}
        renames = {k.lower(): v.lower() for k, v in (companion.get("renames") or {}).items()}
        for local, remote in renames.items():
            if remote in subprograms:
                remotes[local] = Remote(alias, remote)
        for subprogram in record["subprograms"]:
            remotes.setdefault(subprogram["name"], Remote(alias, subprogram["name"]))
        for parameter in record["module_parameters"]:
            globals_.setdefault(parameter["name"], f"{alias}.{pysafe(parameter['name'].upper())}")
        for state in record["module_state"]:
            globals_.setdefault(state["name"], f"{alias}.{pysafe(state['name'])}")
    return tuple(records), remotes, globals_, tuple(imports)


class NumpyTranslation(Transform):
    """Rule-driven Fortran to NumPy, refusing what it cannot prove out."""

    name = "recast.translate.fortran-to-numpy"
    requires = ("interface", "constants")
    deterministic = True

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return (
            unit.kind in ("module", "program")
            and "parse_error" not in unit.attrs
            and bool(facts.interface.get("subprograms"))
        )

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        source = self._verified_source(unit, facts, config)
        records, remotes, companion_globals, companion_imports = companion_tables(
            config.get("companions", [])
        )
        use = config.get("use_constants")
        # The map the generated module resolves use-imported names through is
        # derived from what resolve() was actually asked for -- the pipeline
        # wrote the same map to use_params.json.
        use_parameters = (
            {e["name"]: e["name"].upper() for e in use["resolved"] if e["requested"]} if use else {}
        )

        assembler = Subprograms(
            record=facts.interface,
            constants=facts.constants,
            profile=PROFILES[config.get("profile", DEFAULT)],
            companions=records,
            use_parameters=use_parameters,
            companion_globals=companion_globals,
            externals=facts.provenance.get("externals", {}),
            remotes=remotes,
            function_stubs=config.get("function_stubs", {}),
            statement_stubs=config.get("statement_stubs", {}),
            patches=config.get("patches", {}),
        )
        module = facts.interface["module"]
        stem = config.get("constants_stem", f"{module}_constants")
        use_stem = config.get("use_constants_stem", f"{module}_use_constants")
        renderer = Modules(
            subprograms=assembler,
            constants_stem=stem,
            use_constants_stem=use_stem,
            externals_module=config.get("externals_module"),
            companion_imports=companion_imports,
        )

        text, report = renderer.render(source)
        files: dict[Path, bytes] = {
            Path(f"{module}_numpy.py"): text.encode(),
            Path(f"{stem}.py"): constants_module(
                facts.constants,
                extern=tuple((e["stem"], e["count"]) for e in config.get("extern_constants", [])),
            ).encode(),
        }
        if use:
            files[Path(f"{use_stem}.py")] = use_constants_module(
                use["resolved"], use["module_name"]
            ).encode()

        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files=files,
            deferred=[
                f"{entry['subprogram']}/{entry['block']}: {entry['reason']}"
                for entry in report
                if entry["status"] == "agent_queue"
            ],
            notes={
                "blocks": report,
                "profile": assembler.profile.name,
                "source_digest": facts.provenance.get("digest"),
                "companions": [c["alias"] for c in config.get("companions", [])],
                # Emitted name -> source name, per subprogram: the record the
                # read/write cross-check needs to undo the constant renames.
                # Producing it is part of this Transform's obligation, not an
                # internal detail (see ``names.as_protocol_table``).
                "renames": {
                    record["name"]: assembler.floors(record["name"]).names.as_protocol_table()
                    for record in facts.interface["subprograms"]
                },
            },
        )

    @staticmethod
    def _verified_source(unit: Unit, facts: Facts, config: dict[str, Any]) -> Path:
        from recast.fortran._parse import digest

        root = Path(config.get("root", "."))
        named = facts.provenance.get("source") or (unit.sources[0] if unit.sources else None)
        if named is None:
            raise ConfigError(f"unit {unit.uid!r} carries no source path to translate")
        source = root / named
        if not source.is_file():
            raise ConfigError(f"unit {unit.uid!r} names {source}, which is not a file")
        analyzed = facts.provenance.get("digest")
        if analyzed and digest(source) != analyzed:
            raise ConfigError(
                f"{source} changed since analysis: its Facts describe digest "
                f"{analyzed[:12]}..., so this translation would carry provenance "
                "that quietly lies. Re-run analysis."
            )
        return source


def factory(**config: Any) -> NumpyTranslation:
    """The entry-point hook. Configuration arrives per-``apply``, not here:
    the tables are per-module, and a Transform instance is per-run."""
    del config
    return NumpyTranslation()
