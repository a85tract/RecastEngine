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

Like the frontend, this module imports none of its heavy dependencies at
import time. fparser2 and NumPy arrive with the ``fortran`` and ``translate``
extras; registering the plugin must not require them, so a bare install
still gets a working ``recast doctor``, and the missing extras surface on
the first ``apply``, named, with the install line.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from recast.errors import ConfigError
from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform

if TYPE_CHECKING:
    from recast.transform.numpy.expressions import Remote

__all__ = ["NumpyTranslation", "companion_tables", "factory"]

BACKEND_PACKAGES = ("fparser", "numpy", "mpmath")
"""What the emitter stack needs installed before it can be imported."""


def _require_backend() -> None:
    """Import the emitter stack's dependencies, or name the missing extras.

    Any other ``ImportError`` is a real bug and is re-raised untouched --
    being helpful about the optional dependencies must not swallow a typo in
    this package's own imports.
    """
    try:
        import recast.transform.numpy.modules  # noqa: F401
    except ImportError as exc:
        if (exc.name or "").split(".")[0] not in BACKEND_PACKAGES:
            raise
        raise ConfigError(
            "the numpy translation backend needs fparser2 and numpy, which are "
            "not installed. Install them with: pip install 'recast-engine[fortran,translate]'"
        ) from exc


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
    from recast.transform.numpy.constants import defined_module_parameters
    from recast.transform.numpy.expressions import Remote
    from recast.transform.numpy.vocabulary import pysafe

    records: list[dict[str, Any]] = []
    remotes: dict[str, Remote] = {}
    globals_: dict[str, str] = {}
    imports: list[str] = []
    aliases = _aliases(companions)
    for companion in companions:
        record = companion["record"]
        records.append(record)
        alias = companion.get("alias") or aliases[id(companion)]
        module_py = companion.get("module_py") or f"{_module_of(companion)}_numpy"
        imports.append(f"import {module_py} as {alias}")
        subprograms = {s["name"]: s for s in record["subprograms"]}
        renames = {k.lower(): v.lower() for k, v in (companion.get("renames") or {}).items()}
        for local, remote in renames.items():
            if remote in subprograms:
                remotes[local] = Remote(alias, remote)
        for subprogram in record["subprograms"]:
            remotes.setdefault(subprogram["name"], Remote(alias, subprogram["name"]))
        # A parameter is spelled as the companion's constants file spells it:
        # upper-case when that file defines it, lower-case when the file has
        # only a SKIPPED line for it. Without the companion's constants record
        # nothing is known to be defined, and every name is lower-case -- which
        # is what the pipeline does with no constants.py beside the interface.
        defined = (
            defined_module_parameters(companion["constants"])
            if companion.get("constants")
            else set()
        )
        for parameter in record["module_parameters"]:
            name = parameter["name"]
            attr = pysafe(name.upper()) if name.upper() in defined else pysafe(name)
            globals_.setdefault(name, f"{alias}.{attr}")
        for state in record["module_state"]:
            globals_.setdefault(state["name"], f"{alias}.{pysafe(state['name'])}")
        reverse = {v: k for k, v in renames.items()}
        for parameter in record["module_parameters"]:
            local = reverse.get(parameter["name"])
            if local:
                # A use-rename shadows a same-named global another companion
                # registered first: assigned, never setdefault-guarded.
                name = parameter["name"]
                attr = pysafe(name.upper()) if name.upper() in defined else pysafe(name)
                globals_[local] = f"{alias}.{attr}"
        for state in record["module_state"]:
            local = reverse.get(state["name"])
            if local:
                globals_[local] = f"{alias}.{pysafe(state['name'])}"
    return tuple(records), remotes, globals_, tuple(imports)


def _aliases(companions: list[dict[str, Any]]) -> dict[int, str]:
    """What the emitted module imports each companion as.

    The pipeline's rule -- ``_<module>``, shortened to ``_<first three>`` past
    ten characters -- because a translation has to import its
    siblings under the names the pipeline's output uses or the two files
    disagree on every reference. The shortening collides
    (``micro_mg_utils`` and ``micro_mg2_0`` are both ``_mic``), which the
    pipeline does not notice because its companion lists are written by hand;
    a resolver that derives them has to, so a collision lengthens the prefix
    until it stops.
    """
    taken: dict[str, str] = {}
    chosen: dict[int, str] = {}
    for companion in companions:
        if companion.get("alias"):
            taken.setdefault(companion["alias"], "")
            continue
        module = _module_of(companion)
        length = 3 if len(module) > 10 else len(module)
        while length < len(module) and taken.get(f"_{module[:length]}", module) != module:
            length += 1
        alias = f"_{module[:length]}"
        taken.setdefault(alias, module)
        chosen[id(companion)] = alias
    return chosen


def _module_of(companion: dict[str, Any]) -> str:
    """The Fortran module a companion entry is about."""
    return str(companion.get("module") or companion["record"].get("module", "")).lower()


def _scaffolding_names() -> set[str]:
    """Every emitted name that is machinery rather than data.

    The runtime's definitions are read out of the runtime module itself
    rather than kept as a list, so a shim added there is scaffolding here
    without anyone remembering to say so twice.
    """
    import ast

    from recast.transform.numpy import runtime
    from recast.transform.numpy.vocabulary import (
        ELEMENTAL_ARRAY,
        ELEMENTAL_SCALAR,
        REDUCTIONS,
    )

    names = {"np", "math", "os", "_ext", "_RUNTIME", "_SIGNATURES", "range", "SystemExit"}
    # The emitter's own temporaries. ``_out`` holds a multi-output call's
    # tuple for one statement and is immediately unpacked -- the dataflow is
    # to the names it is unpacked *into*, and counting it made every such
    # block read and write a variable the Fortran has no counterpart for.
    # ``_g``/``_be``/``_lc``/``_le`` are the ``except ... as`` bindings of the
    # goto, block-exit and named-loop catchers; the exception classes are
    # already scaffolding because they are defined in the runtime, but the
    # names they are caught under are not defined anywhere a reader could
    # find them.
    names |= {"_out", "_g", "_be", "_lc", "_le"}
    # The backend spells some intrinsics as bare Python builtins -- abs, int,
    # max, len. They are its vocabulary, so it declares them; a verifier that
    # skipped builtin-looking names on its own would drop real dataflow on a
    # Fortran variable that happens to be called `sum`.
    for table in (ELEMENTAL_SCALAR, ELEMENTAL_ARRAY, REDUCTIONS):
        names |= {spelling for spelling in table.values() if "." not in spelling}
    for node in ast.parse(runtime.emit()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


class NumpyTranslation(Transform):
    """Rule-driven Fortran to NumPy, refusing what it cannot prove out."""

    name = "recast.translate.fortran-to-numpy"
    requires = ("interface", "constants", "effects")
    deterministic = True

    def __init__(self, *, deterministic: bool | None = None) -> None:
        """``deterministic=False`` is how a caller takes responsibility for a
        behaviour hook.

        The flag is read at plan time, off the plugin the recipe names, to
        decide whether the run needs a hard gate. A transform that claimed
        determinism and then consulted a model would slip past that rule, so
        this one refuses a ``deferred_handler`` unless it was constructed
        having said otherwise -- and the thing that constructs it that way is
        a Transform somebody wrote, registered, and declared
        ``deterministic = False`` on. See ``recast.transform.numpy.agentic``.
        """
        if deterministic is not None:
            self.deterministic = deterministic

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        """A module or program this frontend could parse.

        Deliberately not "and it has procedures". A module of nothing but
        kind parameters is what half the libraries in the corpus put their
        working precision in, and skipping it emitted no file -- so every
        sibling that ``use``s it failed to import, which took whole cases
        down. The pipeline has no such condition: pointed at such a file it
        writes the module out, empty of subprograms and importable, and its
        parameters ride in the constants module beside it.
        """
        return (
            unit.kind in ("module", "program")
            and "parse_error" not in unit.attrs
            and bool(facts.interface)
        )

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        _require_backend()
        from recast.fortran.interface import subprogram_key
        from recast.transform.numpy.constants import constants_module, use_constants_module
        from recast.transform.numpy.modules import Modules
        from recast.transform.numpy.subprograms import Subprograms
        from recast.transform.profiles import DEFAULT, PROFILES

        source = self._verified_source(unit, facts, config)
        # The operator's list wins where there is one; otherwise the frontend
        # resolved the unit's own ``use`` statements against the tree. A tree
        # nobody has mapped by hand is the common case, and its
        # translation refusing every sibling call was the corpus's largest
        # single finding.
        declared = config.get("companions")
        if declared is None:
            declared = facts.provenance.get("companions", [])
        records, remotes, companion_globals, companion_imports = companion_tables(declared)
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
            intrinsics=config.get("intrinsic_overrides", {}),
            runtime_imports=tuple(config.get("runtime_imports", ())),
            call_transforms=config.get("call_transforms", {}),
            function_transforms=config.get("function_transforms", {}),
            handle_producers=frozenset(config.get("handle_producers", ())),
            type_bound_procedures=frozenset(config.get("type_bound_procedures", ())),
            patches=config.get("patches", {}),
            poison_undefined=bool(config.get("poison_undefined", False)),
            poison_integers=bool(config.get("poison_integers", False)),
            deferred_handler=self._handler(config),
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
            keep_unbound_stub_imports=bool(config.get("keep_unbound_stub_imports", False)),
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

        # Emitted name -> source name, per subprogram: the record the
        # read/write cross-check needs to undo the constant renames.
        # Producing it is part of this Transform's obligation, not an
        # internal detail (see ``names.as_protocol_table``).
        renames = {
            subprogram_key(record): assembler.floors(
                subprogram_key(record)
            ).names.as_protocol_table()
            for record in facts.interface["subprograms"]
        }
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
                # Stable, source-free coverage protocol for an outer
                # acceptance controller.  ``blocks`` remains the diagnostic
                # ledger; callers should not have to reverse-engineer that
                # transform-specific shape merely to prove which requested
                # subprograms this Candidate attempted.
                "coverage": {
                    "subprograms": sorted(
                        subprogram_key(record) for record in facts.interface["subprograms"]
                    )
                },
                "profile": assembler.profile.name,
                "source_digest": facts.provenance.get("digest"),
                "companions": [c["alias"] for c in config.get("companions", [])],
                "renames": renames,
                "rwset": self._rwset_protocol(
                    unit, facts, report, renames, f"{module}_numpy.py", assembler
                ),
            },
        )

    def _rwset_protocol(
        self,
        unit: Unit,
        facts: Facts,
        report: list[dict[str, Any]],
        renames: dict[str, dict[str, str]],
        emitted_file: str,
        assembler: Any,
    ) -> dict[str, Any]:
        """What ``static.rwset`` needs to cross-check this candidate.

        Blocks pair the source's read/write sets (from ``Facts.effects``,
        which the frontend computed per block) with the emitted line spans
        this Transform produced for them. Deferred blocks are left out: a
        ``raise NotImplementedError`` is not a translation, and the gate's
        job is to judge translations.
        """
        from recast.transform.numpy.vocabulary import RESERVED, pysafe

        blocks = []
        for entry in report:
            if entry["status"] == "agent_queue":
                continue
            effects = facts.effects.get(f"{unit.uid}/{entry.get('key', entry['subprogram'])}", {})
            sets = next((b for b in effects.get("blocks", []) if b["id"] == entry["block"]), None)
            if sets is None:
                raise ConfigError(
                    f"Facts.effects carries no read/write sets for "
                    f"{entry['subprogram']}/{entry['block']}; the frontend and this "
                    "transform chunked the source differently, which the gate "
                    "cannot paper over"
                )
            blocks.append(
                {
                    "subprogram": entry["subprogram"],
                    "block": entry["block"],
                    "reads": sets["reads"],
                    "writes": sets["writes"],
                    "lines": entry["py_lines"],
                }
            )
        names: dict[str, str] = {}
        for table in renames.values():
            names.update(table)
        # ``use, intrinsic :: iso_fortran_env, only: stdout => output_unit``
        # reads ``stdout`` on the source side and ``_iso_fortran_env.
        # output_unit`` on this one. The alias rule below undoes exactly that
        # shape for a companion; an intrinsic namespace is the same shape and
        # is invisible here otherwise, because no import line announces it.
        for local, spelling in assembler.use_bindings.items():
            alias, _, attribute = spelling.partition(".")
            if alias in assembler.intrinsic_aliases and attribute:
                names.setdefault(attribute, local)
        return {
            "file": emitted_file,
            "blocks": blocks,
            "names": names,
            "procedures": sorted(
                {pysafe(record["name"]) for record in facts.interface["subprograms"]}
                # The siblings' procedures too: `_wv.wv_sat_svp_water(t)` is a
                # call, and without these the alias rule would read it as data.
                | {remote.name for remote in assembler.remotes.values()}
                # ... and the stubbed modules' (the frontend's list): a call to
                # a stand-in function is a call, not a read of its name.
                | {pysafe(name) for name in facts.interface.get("stub_procedures") or ()}
            ),
            "aliases": sorted(
                {remote.alias for remote in assembler.remotes.values()}
                | set(assembler.intrinsic_aliases)
                # A stub module's alias too: ``_pftconmod.pftcon`` is a read
                # of ``pftcon``, and without the alias the check read ``_pftconmod``.
                | {line.rsplit(" as ", 1)[-1] for line in assembler.stub_imports}
                # ...and a companion reached only for its globals, whose alias
                # the emitter binds through ``companion_globals``.
                | {spelling.split(".")[0] for spelling in assembler.companion_globals.values()}
            ),
            "reserved": sorted(RESERVED),
            "scaffolding": sorted(
                _scaffolding_names()
                | {f"_make_{t}" for t in assembler.record.get("types", {})}
                | {
                    f"_make_{t}"
                    for companion in assembler.companions
                    for t in companion.get("types", {})
                }
            ),
        }

    def _handler(self, config: dict[str, Any]) -> Any:
        """The behaviour hook, if this instance is allowed to have one."""
        handler = config.get("deferred_handler")
        if handler is None:
            return None
        if self.deterministic:
            raise ConfigError(
                f"{self.name!r} was given a 'deferred_handler' while declaring "
                "deterministic = True. A handler runs during the translation and its "
                "answers vary, so the transform the recipe names has to be one that "
                "says so -- construct this with NumpyTranslation(deterministic=False) "
                "from a Transform of your own that declares it, or precompute the "
                "sites into config['patches'] instead."
            )
        if not callable(handler):
            raise ConfigError(
                f"'deferred_handler' is {type(handler).__name__}, not callable. A "
                "behaviour hook cannot come from a JSON config; it is supplied by a "
                "Transform in Python."
            )
        return handler

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
