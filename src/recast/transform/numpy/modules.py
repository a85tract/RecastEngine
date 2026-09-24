"""Whole module files, assembled around the subprograms.

The top of the emitter: everything in a generated file that is not inside a
``def``. Type factories, because a Fortran derived type is storage with a
layout and the translation needs something that constructs one. Module state,
because a Fortran module carries SAVE variables that outlive any call, and
each one needs the initialization its declaration promised -- or an honest
``None`` when initialization is the init routine's job. The signature table,
because the differential harness on the other side generates driver data from
it. And the runtime, pasted in whole so the file stands alone: it is the
product, imported by comparison harnesses with no reason to have the engine
installed.

Two deliberate divergences from the pipeline this reproduces, both above the
body and neither observable by a gate:

* The header -- docstring, imports, the runtime text itself -- is the
  engine's own. The pipeline kept its runtime inside a string constant; this
  repository keeps it as real, typed, tested code (``runtime.emit``), and the
  emitted text follows that code, not the string.
* The pipeline strips ``_fstr_eq`` from the runtime when no statement used
  it. The runtime here ships whole: an unused definition changes no number,
  and a runtime whose contents depend on emission bookkeeping is harder to
  reason about than one that is always the same text.

Below the first factory, the output is byte-for-byte the pipeline's, and
``tools/emit_diff.py`` holds it there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from recast import references
from recast.errors import ConfigError
from recast.fortran._parse import f03, parse, walk
from recast.fortran.expr import int_div
from recast.fortran.interface import emit_name, subprogram_key
from recast.transform.numpy import runtime
from recast.transform.numpy.constants import constant_expression
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.numpy.vocabulary import pysafe

__all__ = ["Modules"]

STATE_DTYPES = {
    "float64": "np.float64",
    "int32": "np.int32",
    "bool": "np.bool_",
    "str": "object",
}
"""Module-array state dtypes. Narrower than the allocate map on purpose:
this is what the pipeline recognized here, and a float32 module array in the
corpus would have silently become float64 -- so it must keep doing so until
a gate says otherwise."""

INTEGER_TEXT = re.compile(r"-?\s*\d+")
INTEGER_KIND_TEXT = re.compile(r"(-?\s*\d+)_\w+")
"""An integer literal with a kind suffix, ``2_i4``. Apart from ``INTEGER_TEXT``
because the spelling is the digits without it, and ahead of ``REAL_TEXT``,
which matches it too -- a mantissa with no point -- and made it a float."""
REAL_TEXT = re.compile(r"-?\s*(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_\w+)?", re.I)
CHARACTER_TEXT = re.compile(r"'[^']*'|\"[^\"]*\"")

SAVE_ARRAY_REFUSAL = "save-init array module state translated as scalar"


def _character_literal(fortran: str) -> str:
    """A Fortran character constant as a Python string literal.

    This was ``text.replace('"', "'")`` -- the Fortran spelling emitted
    verbatim with its quotes swapped -- which is wrong twice. A
    double-quoted constant with an apostrophe in it became a Python syntax
    error; and one with ``'; import os; ...`` in it became a statement
    sequence in a module the verifier imports. The source under translation
    is the input this engine takes from other people, so the second one is
    an injection, found by the security review on 2026-08-21.

    ``repr`` of the *value* is what ``expressions.py`` already does for the
    same constant in an expression, and it cannot be escaped. The value is
    the text between the quotes with Fortran's doubled-quote escape undone.
    """
    quote = fortran[0]
    return repr(fortran[1:-1].replace(quote * 2, quote))


ARRAY_TEXT = re.compile(r"\(/.*?/\)", re.S)
DIVISION_TEXT = re.compile(r"-?\s*(?:\d+\.?\d*|\.\d+)(?:_\w+)?\s*/\s*(?:\d+\.?\d*|\.\d+)(?:_\w+)?")
MARKER = re.compile(r"^    # (B\d{3}) <- ")
DEFINITION = re.compile(r"^def (\w+)\(")
STATE_REFUSAL = re.compile(r"^(\w+) = None  # AGENT_QUEUE: ")
"""The one line a refused module-state binding emits, at module scope. Its
report entry (block ``S001``) is located by this line rather than by a block
marker, because module state has no function to carry a marker inside."""
DERIVED_TYPE = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")


@dataclass
class Modules:
    """Render one translated module file.

    Wraps a ``Subprograms`` (which carries all the per-module context) with
    the file-level decisions: what to import, what to name the constants
    module, where the externals shim lives.
    """

    subprograms: Subprograms

    constants_stem: str = "constants"
    """Module name of the generated constants. Each target needs a UNIQUE
    stem when several translated modules coexist in one process."""

    use_constants_stem: str = "use_constants"
    externals_module: str | None = None
    """Where the audited externals shims live; default ``<module>_externals``."""

    keep_unbound_stub_imports: bool = False
    """Keep ``import <mod>_numpy as _<mod>`` for a USE'd module nothing in the
    body binds to. Off, such an import names a module that is not part of the
    run -- a kinds-only USE is the common case -- and only raises
    ``ModuleNotFoundError`` (#18). On, for a harness that provides a runtime
    stub per USE'd module and wants every one imported, as the pipeline's
    CESM project does."""

    companion_imports: tuple[str, ...] = ()
    """``import micro_mg_utils_numpy as _mgu`` lines, one per companion.

    Supplied alongside ``remotes`` rather than derived from it: the remotes
    table knows aliases, but which *file* an alias binds to is a deployment
    decision the operator's companion config owns.
    """

    # -- the whole file -------------------------------------------------------

    def render(self, source: Path) -> tuple[str, list[dict[str, Any]]]:
        """The complete generated file, and the block report with its
        ``py_lines`` re-based to final-file line numbers."""
        nodes = self._subprogram_nodes(source)
        body, report = self.body(nodes)
        text = self.header(body) + "\n".join(body) + self._submodule_exports()
        self._rebase(text, report)
        return text, report

    def _submodule_exports(self) -> str:
        """A lazy re-export of every procedure whose body lives in one of this
        module's submodules (#29): ``use parent`` reaches them in Fortran, so
        ``import parent_numpy`` has to here. PEP 562 ``__getattr__``, so it is
        correct whichever of parent and submodule is imported first. The text
        is the pipeline's, appended after the body so no block line moves."""
        submodules = self.subprograms.record.get("submodules") or {}
        if not submodules:
            return ""
        lines = [
            "",
            "",
            "# -- submodule re-exports (#29) --",
            "_SUBMODULE_EXPORTS = {}",
            "",
            "",
            "def __getattr__(name):",
            "    mod = _SUBMODULE_EXPORTS.get(name)",
            "    if mod is None:",
            "        raise AttributeError(name)",
            "    import importlib",
            "    return getattr(importlib.import_module(mod), name)",
        ]
        for submodule, names in submodules.items():
            lines.extend(f"_SUBMODULE_EXPORTS[{n!r}] = {submodule + '_numpy'!r}" for n in names)
        return "\n".join(lines) + "\n"

    def _stub_imports(self, body: list[str] | None) -> list[str]:
        """The auto-stub imports the file needs: every one when told to keep
        them, otherwise only those whose alias the body binds to."""
        return self._bound_imports(self.subprograms.stub_imports, body)

    def _companion_imports(self, body: list[str] | None) -> list[str]:
        """The companions' imports the file needs.

        The same rule as the auto-stubs, and for the same reason (#18): a
        ``use`` that brought nothing but a kind parameter binds no alias, and
        the file that imports its sibling's translation anyway raises
        ``ModuleNotFoundError`` before running a line -- whether that sibling
        rides along in the candidate or not. Naming only what it calls is what
        lets the candidate be self-contained without carrying the tree.
        """
        return self._bound_imports(self.companion_imports, body)

    def _bound_imports(self, imports: Iterable[str], body: list[str] | None) -> list[str]:
        """Every one when told to keep them, otherwise only those whose alias
        the body binds to."""
        lines = list(imports)
        if self.keep_unbound_stub_imports or body is None:
            return lines
        text = "\n".join(body)
        return [line for line in lines if f"{line.rsplit(' as ', 1)[1]}." in text]

    def header(self, body: list[str] | None = None) -> str:
        record = self.subprograms.record
        init = record["subprograms"][0]["name"] if record["subprograms"] else "<none>"
        # The file's name, never the path it happened to be found at. An
        # absolute path in the emitted text makes the artifact -- and so
        # ``Candidate.digest()`` -- differ between two machines translating the
        # same source, which breaks exactly the reproducibility a
        # ``deterministic`` Transform promises and conformance checks.
        source_name = PurePosixPath(str(record["source_file"])).name
        pieces = [
            f'"""Machine-translated from {source_name} by recast.\n\n'
            f"NumPy/scalar direct translation. Module state mirrors the Fortran\n"
            f"module exactly; call {init} before use.\n"
            f'DO NOT hand-edit mechanical blocks -- fix the engine instead.\n"""',
            "",
        ]
        pieces.extend(runtime.REQUIRED_IMPORTS)
        # Header lines the domain package's emitted code needs: the module
        # its intrinsic spellings live in, the shims its call transforms
        # emit calls to. The engine does not know what they are.
        pieces.extend(self.subprograms.runtime_imports)
        pieces.append("")
        pieces.append(f"from {self.constants_stem} import *  # noqa: F401,F403")
        if self.subprograms.use_parameters:
            pieces.append(f"from {self.use_constants_stem} import *  # noqa: F401,F403")
        if self.subprograms.externals:
            shims = self.externals_module or (record["module"] + "_externals")
            pieces.append(f"import {shims} as _ext")
        extra = sorted(
            set(self._companion_imports(body))
            | set(self._stub_imports(body))
            | {
                imported
                for patch in self.subprograms.patches.values()
                for imported in patch.get("imports", [])
            }
        )
        pieces.extend(extra)
        pieces.append("")
        # A runtime side-effect channel for agent patches (abort flags etc.).
        pieces.append("_RUNTIME = {'abort_msg': None}")
        pieces.append("")
        pieces.append(f"_SIGNATURES = {self._signatures()!r}")
        pieces.append("")
        pieces.append(runtime.emit())
        pieces.append("")
        return "\n".join(pieces)

    def body(self, nodes: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        """Factories, module state, and every subprogram, in the pipeline's
        order -- the part of the file the differential compares byte-for-byte."""
        lines: list[str] = []
        report: list[dict[str, Any]] = []
        semantics_types = self._all_types()
        for type_name, components in semantics_types.items():
            lines.extend(self._factory(type_name, components))
        for state in self.subprograms.record["module_state"]:
            lines.extend(self._state(state, report))
        lines.append("")
        # Ahead of the subprograms, the way the source declares a procedure
        # ahead of the code that calls it -- and so the last subprogram's span
        # still ends at the end of the file (``_rebase``).
        lines.extend(references.python_for(self._reference_externals()))
        for record in self.subprograms.record["subprograms"]:
            node = nodes.get(subprogram_key(record))
            if node is None:
                # The interface record and the parse pass disagree about what
                # exists. Dropping the subprogram here shipped a file whose
                # coverage note claimed it was attempted; a broken invariant
                # is a crash, not a gap.
                raise RuntimeError(
                    f"subprogram {subprogram_key(record)!r} has an interface "
                    "record but no parse node; refusing to emit a module "
                    "with a silent hole"
                )
            rendered, entries = self.subprograms.render(node, subprogram_key(record))
            lines.extend(rendered)
            report.extend(entries)
        return lines, report

    # -- procedures declared here and defined nowhere --------------------------

    def _reference_externals(self) -> list[str]:
        """Names this module declares, nothing defines, and recast can supply.

        ``use lapack, only: dgesv, dgbsv`` names a module whose whole content
        is interface blocks; the bodies are in a compiled library.  Those
        declarations bind like module procedures (``semantics.for_subprogram``
        merges them in), so a caller is translated to ``_lapack.dgbsv(...)`` --
        and this file, which *is* the ``lapack`` translation, had nothing of
        that name in it.  ``recast.references`` holds an implementation for a
        few of them, and the reference build compiles the Fortran twin of the
        same one, so the call means the same thing on both sides.

        Three kinds of name are left alone, each because something else
        already defines it: one this module or a companion has a body for; one
        a submodule of this module defines, which ``_submodule_exports``
        re-exports (#29); and one the operator gave an audited shim in the
        externals module.  Everything else recast has no implementation for
        stays as it was -- declared here, defined nowhere, and disclaiming its
        callers in the oracle.
        """
        record = self.subprograms.record
        defined = {s["name"] for s in record["subprograms"]}
        for names in (record.get("submodules") or {}).values():
            defined |= set(names)
        declared_elsewhere: set[str] = set()
        for companion in self.subprograms.companions:
            defined |= {s["name"] for s in companion.get("subprograms") or ()}
            declared_elsewhere |= {
                declared["name"]
                for declared in (companion.get("interfaces") or {}).values()
                if declared.get("kind") in ("subroutine", "function")
            }
        declared_here = {
            declared["name"]
            for declared in (record.get("interfaces") or {}).values()
            if declared.get("kind") in ("subroutine", "function")
        }
        # A bare ``call dgbsv(...)`` -- declared by nothing, an implicit
        # external -- is bound against ``references.interface`` and spelled
        # as a call into this module (``Statements._call``), so the
        # definition lands here too. One a companion declares is spelled
        # through that companion's alias and is that module's to define.
        called = {
            str(name).lower()
            for subprogram in record["subprograms"]
            for name in subprogram.get("external_calls") or ()
        }
        return references.supported(
            name
            for name in declared_here | (called - declared_elsewhere)
            if name not in defined and name not in self.subprograms.externals
        )

    # -- derived-type factories -----------------------------------------------

    def _all_types(self) -> dict[str, dict[str, Any]]:
        """Companion types first, the module's own updating over them --
        the order the pipeline built its table in, kept because it decides
        the order the factories appear in the file."""
        merged: dict[str, dict[str, Any]] = {}
        for companion in self.subprograms.companions:
            merged.update(companion.get("types", {}))
        merged.update(self.subprograms.record.get("types", {}))
        return merged

    def _factory(self, type_name: str, components: dict[str, Any]) -> list[str]:
        lines = [
            f"def _make_{type_name}():",
            f'    """factory for type({type_name}) (components per Derived_Type_Def)."""',
            "    o = _new_derived()",
        ]
        for name, component in components.items():
            safe = pysafe(name)
            dims = component.get("dims")
            shape = self.subprograms.component_shape(component) if dims else None
            # A CHARACTER component starts at its length in blanks, or at a
            # literal initializer fitted to it, as a module variable does.
            character = (
                self._character_value(component, str(component.get("init") or "").strip())
                if component["dtype"] == "str" and (shape is not None or not dims)
                else None
            )
            if character is not None and shape is not None:
                lines.append(f"    o.{safe} = np.full(({shape},), {character}, dtype=object)")
            elif character is not None:
                lines.append(f"    o.{safe} = {character}")
            elif component.get("init"):
                lines.append(f"    o.{safe} = {self._component_default(component, shape)}")
            elif shape is not None:
                lines.append(f"    o.{safe} = np.zeros(({shape},))")
            elif dims:
                lines.append(f"    o.{safe} = None")
            elif component["dtype"] in ("float64", "float32"):
                lines.append(f"    o.{safe} = 0.0")
            elif component["dtype"] in ("int32", "int64"):
                lines.append(f"    o.{safe} = 0")
            elif component["dtype"] == "bool":
                lines.append(f"    o.{safe} = False")
            else:
                lines.append(f"    o.{safe} = None")
        lines.append("    return o")
        lines.append("")
        return lines

    def _component_default(self, component: dict[str, Any], shape: str | None) -> str:
        """A component's default initialization, as the factory's value.

        Every object of the type starts with it (F2018 7.5.4.6), and the
        factory started every component at a zero of its type whatever the
        type said -- ``real(r8) :: tol = 1.0d-6`` came out ``o.tol = 0.0``
        (FNP-D0030). The initializer is rendered by the forms a module
        variable's is, since it is the same kind of constant text: a scalar
        through ``_state_value``, an array's broadcast through
        ``_broadcast_fill``. A bare integer given to a REAL component is
        that real, as Fortran's conversion makes it. What neither renders
        -- an expression, an array whose shape is not static -- is ``None``
        with the text beside it rather than the zero that was not the
        source's: a ``None`` read before it is written raises, where a
        zero is a wrong number nothing reports.
        """
        initializer = str(component["init"]).strip()
        dtype = str(component.get("dtype") or "")
        parameters = {p["name"] for p in self.subprograms.record["module_parameters"]}
        unrendered = f"None  # default initialization not rendered: {initializer!r}"
        if component.get("dims"):
            if shape is None or initializer.lower() == "null()":
                return "None" if initializer.lower() == "null()" else unrendered
            fill = self._broadcast_fill(initializer.lower(), parameters)
            if fill is not None:
                return f"np.full(({shape},), {fill}, dtype={STATE_DTYPES.get(dtype, 'np.float64')})"
            if dtype in ("int32", "int64") and ARRAY_TEXT.fullmatch(initializer):
                # An integer constructor stays integer: ``_state_value``'s
                # constructor rule reads ``1`` as a real before an integer.
                items = [i.strip() for i in initializer[2:-2].split(",")]
                if all(INTEGER_TEXT.fullmatch(i) for i in items):
                    values = ", ".join(i.replace(" ", "") for i in items)
                    return f"np.array([{values}], dtype=np.{dtype})"
            value = self._state_value({"init_expr": initializer, "dtype": dtype}, parameters)
            return unrendered if value.startswith("None") else value
        whole = INTEGER_KIND_TEXT.fullmatch(initializer)
        if dtype.startswith("float") and (whole or INTEGER_TEXT.fullmatch(initializer)):
            digits = (whole.group(1) if whole else initializer).replace(" ", "")
            return (
                f"np.float64(np.float32({digits}))"
                if dtype == "float32"
                else f"np.float64({digits})"
            )
        value = self._state_value({"init_expr": initializer, "dtype": dtype}, parameters)
        if value.startswith("None  # pointer"):
            return "None"
        if value.startswith("None"):
            # An expression over the module's parameters -- ``n + 1_wi`` --
            # spelled as the constants file spells a parameter's, and
            # converted to the component's type: ``integer :: kk = grav * 2``
            # is 19, where the expression's own value was 19.62.
            spelled = constant_expression(initializer, self.subprograms.constants, dtype)
            return unrendered if spelled is None else spelled
        # A lone real given to an INTEGER component is converted the same
        # way, and a double parameter given to a single one is rounded.
        lowered = initializer.lower()
        declared = {
            p["name"]: str(p.get("dtype") or "")
            for p in self.subprograms.constants.get("module_parameters") or []
        }
        literal = not whole and not INTEGER_TEXT.fullmatch(initializer)
        real = (literal and REAL_TEXT.fullmatch(initializer) is not None) or declared.get(
            lowered, ""
        ).startswith("float")
        if dtype.startswith("int") and real:
            return f"int({value})"
        if dtype == "float32" and declared.get(lowered) == "float64":
            return f"np.float64(np.float32({value}))"
        return value

    # -- module state ---------------------------------------------------------

    def _state(
        self, state: dict[str, Any], report: list[dict[str, Any]] | None = None
    ) -> list[str]:
        # Policy: module state the renderer cannot honestly initialize keeps
        # its None binding (the module must import for anything else to be
        # checked) but is RECORDED as deferred work -- a comment alone let
        # `allocated(x)` silently read "never allocated" with no entry
        # anywhere saying the translation is incomplete.
        def _refuse(reason: str) -> list[str]:
            if report is not None:
                report.append(
                    {
                        "subprogram": str(state["name"]),
                        "key": f"module-state:{state['name']}",
                        "block": "S001",
                        "src_span": [0, 0],
                        "status": "agent_queue",
                        "reason": reason,
                        "py_lines": [0, 0],
                    }
                )
            return [f"{pysafe(state['name'])} = None  # AGENT_QUEUE: {reason}"]

        parameters = {p["name"] for p in self.subprograms.record["module_parameters"]}
        # A private scalar one public argument-less setter fixes to a
        # constant (the record's ``constant_state``) starts at that constant:
        # it is what the run's init leaves there, and a ``None`` for the
        # declaration's missing initializer would be read by the first
        # ``select case`` on it.
        constant = (self.subprograms.record.get("constant_state") or {}).get(state["name"])
        if constant and not state.get("dims"):
            value = self._state_value({**state, "init_expr": constant["value"]}, parameters)
            if not value.startswith("None  # TODO"):
                return [
                    f"{pysafe(state['name'])} = {value}  # module state ({state['dtype']}), "
                    f"what {constant['setter']} sets it to and nothing else writes"
                ]
        initializer = str(state.get("init_expr") or "").strip()
        lowered = initializer.lower()
        # An array's branch, whether or not it carries an initializer. Only
        # this was reached before, when it did not -- so a saved array with a
        # scalar init was emitted as that scalar: ``dim_theta = 0.0`` for a
        # PDF_N_THETA-long buffer, and ``lq = False`` for an array of
        # logicals. An array-constructor init and ``null()`` still take the
        # scalar path below, which is where they belong.
        if state.get("dims") and not lowered.startswith("(/") and lowered != "null()":
            if all(d["ub"] is not None for d in state["dims"]):
                extents = []
                renderable = True
                for dim in state["dims"]:
                    text = dim["ub"]
                    if re.fullmatch(r"\d+", text):
                        extents.append(text)
                    elif text.lower() in parameters:
                        extents.append(text.upper())
                    else:
                        renderable = False
                if renderable:
                    dtype = STATE_DTYPES.get(state["dtype"], "np.float64")
                    shape = ", ".join(extents)
                    fill = self._broadcast_fill(lowered, parameters)
                    if fill is not None:
                        # Fortran broadcasts a scalar save-init across the
                        # whole array; every element starts at it.
                        return [
                            f"{state['name']} = np.full(({shape},), {fill}, "
                            f"dtype={dtype})  # module array state (save-init)"
                        ]
                    if not initializer:
                        # No init: a zero buffer, filled by the module's init
                        # routine (Fortran SAVE semantics).
                        if dtype == "object":
                            # A character array: np.zeros of dtype object is
                            # an array of the integer 0, and the first thing
                            # done to it is a string comparison. Its elements
                            # are blanks of its length where that is known.
                            blank = self._character_value(state) or "''"
                            return [
                                f"{state['name']} = np.full(({shape},), {blank}, dtype=object)"
                                "  # module array state (str)"
                            ]
                        return [
                            f"{state['name']} = np.zeros(({shape},), "
                            f"dtype={dtype})  # module array state"
                        ]
                    return _refuse(f"{SAVE_ARRAY_REFUSAL} (init {initializer!r})")
            if initializer:
                # A bound no module-scope name resolves, and an initializer to
                # broadcast across it: the shape is not knowable here, and
                # guessing one would be a silently wrong buffer.
                return _refuse(f"{SAVE_ARRAY_REFUSAL} (dims not static)")
            return [
                f"{pysafe(state['name'])} = None  # allocatable/assumed module array, set by init"
            ]
        if state["dtype"] == "str" and not state.get("dims"):
            blanks = self._character_value(state, initializer)
            if blanks is not None:
                note = "Fortran save-init" if initializer else "set by init"
                return [f"{pysafe(state['name'])} = {blanks}  # module state (str), {note}"]
        if state["init_expr"]:
            value = self._state_value(state, parameters)
            if value.startswith("None  # TODO"):
                return _refuse(f"module-state initializer not renderable: {state['init_expr']!r}")
            return [
                f"{pysafe(state['name'])} = {value}  # module state "
                f"({state['dtype']}), Fortran save-init"
            ]
        derived = DERIVED_TYPE.match(str(state.get("dtype", "")))
        if derived:
            name = derived.group(1).lower()
            factory = f"_make_{name}()" if name in self._all_types() else "_new_derived()"
            return [
                f"{pysafe(state['name'])} = {factory}  # module state "
                f"({state['dtype']}), set by init"
            ]
        return [f"{pysafe(state['name'])} = None  # module state ({state['dtype']}), set by init"]

    def _character_value(self, entity: dict[str, Any], initializer: str = "") -> str | None:
        """What a CHARACTER module variable or component starts as: its
        declared length in blanks, or a character-literal ``initializer``
        fitted to it; ``None`` for a length not known at module scope.

        A ``character(len=6) :: ms`` holds six characters before any store
        as after one, and ``ms = 'ab'`` leaves ``'ab    '``. The module
        bound it to ``None`` and started a component at ``None`` too, and a
        literal initializer kept its own length (FNP-D0021). A length is
        known here when it is digits or one of the module's parameters,
        which the module spells in capitals.
        """
        value = ""
        if CHARACTER_TEXT.fullmatch(initializer):
            value = initializer[1:-1].replace(initializer[0] * 2, initializer[0])
        elif initializer:
            return None
        length = str(entity.get("char_len") or "").strip()
        parameters = {p["name"] for p in self.subprograms.record["module_parameters"]}
        if length.isdigit():
            return repr(value.ljust(int(length))[: int(length)])
        if length.lower() in parameters:
            return f"({value!r}).ljust({length.upper()})[:{length.upper()}]"
        return None

    def _broadcast_fill(self, expression: str, parameters: set[str]) -> str | None:
        """A scalar initializer simple enough to broadcast, or ``None``.

        Deliberately fewer forms than ``_state_value``: what goes into every
        element of a saved array has to be a value this stage is certain of,
        and anything else is a site for a human rather than a guess.
        """
        if not expression:
            return None
        if expression in (".true.", ".false."):
            return "True" if expression == ".true." else "False"
        if expression in parameters:
            return expression.upper()
        if expression in self.subprograms.companion_globals:
            return self.subprograms.companion_globals[expression]
        if INTEGER_TEXT.fullmatch(expression):
            return expression.replace(" ", "")
        if REAL_TEXT.fullmatch(expression):
            return f"np.float64('{expression.replace(' ', '').split('_')[0].replace('d', 'e')}')"
        return None

    def _state_value(self, state: dict[str, Any], parameters: set[str]) -> str:
        """A saved variable's compile-time initializer, rendered from text.

        A long chain of recognized forms with an honest ``None  # TODO`` at
        the end -- an initializer this cannot render is a site for a human,
        not a guess.
        """
        expression: str = str(state["init_expr"]).strip().lower()
        if expression in parameters:
            return expression.upper()
        if INTEGER_TEXT.fullmatch(expression):
            return expression.replace(" ", "")
        whole = INTEGER_KIND_TEXT.fullmatch(expression)
        if whole:
            # ``integer(i4) :: j = 2_i4`` is the integer 2. ``REAL_TEXT`` read
            # it as a real, ``np.float64('2')``, and ``x(j)`` then raised --
            # a float is no subscript.
            return whole.group(1).replace(" ", "")
        if expression in (".true.", ".false."):
            return "True" if expression == ".true." else "False"
        if REAL_TEXT.fullmatch(expression):
            base = expression.replace(" ", "").split("_")[0].replace("d", "e")
            if state.get("dtype") == "float32":
                single = self._single(expression)
                return (
                    single or f"None  # TODO: real(4) init of unknown kind {state['init_expr']!r}"
                )
            return f"np.float64('{base}')"
        if CHARACTER_TEXT.fullmatch(str(state["init_expr"]).strip()):
            return _character_literal(str(state["init_expr"]).strip())
        if expression == "null()":
            return "None  # pointer, null-init"
        logical = re.fullmatch(r"\.(true|false)\.(?:_\w+)?", expression)
        if logical:
            # A LOGICAL literal of a named kind, ``.false._wi``: the value is
            # the same whatever the kind, and the plain spelling above only
            # knew the default one.
            return "True" if logical.group(1) == "true" else "False"
        if re.fullmatch(r"huge\(1\)", expression):
            return "np.int32(2147483647)  # HUGE(default int)"
        if re.fullmatch(r"huge\(1\.0?_?\w*\)", expression):
            return "np.finfo(np.float64).max  # HUGE(real(r8))"
        if re.fullmatch(r"-\s*huge\(1\.0?_?\w*\)", expression):
            return "-np.finfo(np.float64).max  # -HUGE(real(r8))"
        if re.fullmatch(r"-\s*huge\(1\)", expression):
            return "np.int32(-2147483647)  # -HUGE(int)"
        if re.fullmatch(r"epsilon\(\w+\)", expression):
            return "np.finfo(np.float64).eps  # EPSILON"
        if ARRAY_TEXT.fullmatch(expression):
            items = [item.strip() for item in expression[2:-2].split(",")]
            if all(re.fullmatch(r"'[^']*'", item) for item in items):
                return "np.array([" + ", ".join(items) + "])  # char array init"
            if all(REAL_TEXT.fullmatch(item) for item in items):
                values = ", ".join(
                    "np.float64('{}')".format(item.replace(" ", "").split("_")[0].replace("d", "e"))
                    for item in items
                )
                return f"np.array([{values}])"
            if all(re.fullmatch(r"\.\s*(true|false)\s*\.", item, re.I) for item in items):
                values = ", ".join("True" if "true" in item.lower() else "False" for item in items)
                return f"np.array([{values}])"
            if all(re.fullmatch(r"\d+", item.strip()) for item in items):
                return f"np.array([{', '.join(item.strip() for item in items)}], dtype=np.int32)"
            return f"None  # TODO: array init {state['init_expr']!r}"
        if DIVISION_TEXT.fullmatch(expression):
            return self._quotient_value(expression, str(state.get("dtype") or ""))
        return f"None  # TODO: init {state['init_expr']!r}"

    def _real_kind(self, literal: str) -> str | None:
        """The dtype of a real literal: a ``d`` exponent is a double, no
        suffix the default REAL, a single, and a suffix what the module's
        kind map says it is -- ``None`` for one it does not name."""
        mantissa, _, suffix = literal.replace(" ", "").lower().partition("_")
        if not suffix:
            return "float64" if "d" in mantissa else "float32"
        kinds = {"4": "float32", "8": "float64", **(self.subprograms.record.get("kind_map") or {})}
        found = kinds.get(suffix)
        return found if found in ("float32", "float64") else None

    def _single(self, literal: str) -> str | None:
        """A real literal as a REAL(4) variable holds it, or ``None``.

        The value is the single nearest the literal, widened: ``real :: s =
        0.1`` holds 0.10000000149011612, where the ``np.float64('0.1')`` this
        was holds 0.1. A double literal is rounded to double first, as the
        compiler converts it; the default kind's is read as a single, as
        the constants file's ``_real`` reads it.
        """
        base = literal.replace(" ", "").lower().split("_")[0].replace("d", "e")
        kind = self._real_kind(literal)
        if kind == "float32":
            return f"np.float64(np.float32('{base}'))"
        if kind == "float64":
            return f"np.float64(np.float32(np.float64('{base}')))"
        return None

    def _quotient_value(self, expression: str, dtype: str) -> str:
        """``a / b`` over two literals, as the variable it initializes holds it.

        Fortran divides first, by the operands' types, and converts the
        quotient to the variable's type after: ``real(r8) :: third = 1/3``
        is an integer division, 0, and then 0.0 -- this folded every such
        quotient as a real one, 0.333... (FNP-D0029). Two integer literals
        truncate toward zero (``expr.int_div``, exact); a quotient with a
        real operand is the real one it was; an INTEGER variable takes the
        quotient truncated, as the assignment converts it. The sign is
        read off the numerator with its spaces gone -- fparser writes
        ``-7/2`` as ``- 7 / 2``, which ``float`` refused, and the whole unit
        with it.

        A REAL(4) variable holds the quotient rounded to single: ``real ::
        ms = 1.0/3.0`` is 0.3333333432674408, and was the double third. Its
        real quotient is spelled for NumPy to divide at the operands' kind --
        two singles in single, a double operand in double, an integer at the
        other side's -- and then rounded; an operand of a kind the module
        does not name is not guessed at.
        """
        sides = [side.replace(" ", "") for side in expression.split("/")]
        numerator, denominator = (side.split("_")[0].replace("d", "e") for side in sides)
        integers = [bool(INTEGER_TEXT.fullmatch(side)) for side in (numerator, denominator)]
        if all(integers):
            value: float = int_div(int(numerator), int(denominator))
        else:
            value = float(numerator) / float(denominator)
        if dtype.startswith("int"):
            return str(int(value))  # the conversion truncates, as ``int`` does
        if dtype == "float32":
            if all(integers):
                return f"np.float64(np.float32({int(value)}))"
            kinds = {self._real_kind(s) for s, i in zip(sides, integers, strict=True) if not i}
            if None in kinds:
                return f"None  # TODO: real(4) init of unknown kind {expression!r}"
            ctor = "np.float64" if "float64" in kinds else "np.float32"
            quotient = f"{ctor}('{numerator}') / {ctor}('{denominator}')"
            return f"np.float64(np.float32({quotient}))"
        return f"np.float64({float(value)!r})"

    # -- the signature table --------------------------------------------------

    def _signatures(self) -> dict[str, dict[str, Any]]:
        """Full type signatures per subprogram, embedded for the comparison
        harness on the other side to generate driver data from."""
        table = {}
        for subprogram in self.subprograms.record["subprograms"]:
            arguments = []
            for argument in subprogram["args"]:
                entry: dict[str, Any] = {
                    "name": argument["name"],
                    "dtype": argument["dtype"],
                    "intent": argument["intent"],
                    "optional": argument.get("optional", False),
                }
                if argument.get("domain"):
                    # The values the source lets this integer take (a
                    # ``select case`` whose default stops): the harness
                    # draws within them unless the operator's ranges say.
                    entry["domain"] = list(argument["domain"])
                if argument.get("dims"):
                    entry["dims"] = [
                        {"lb": d.get("lb", "1"), "ub": d.get("ub")} for d in argument["dims"]
                    ]
                if argument.get("path"):
                    # A character dummy the body opens as a file, and what its
                    # OPEN asks of it. The harness that supplies arguments has
                    # to know a scratch path from a message.
                    entry["path"] = argument["path"]
                if argument.get("buffer") and self.subprograms.buffer_out_arrays:
                    # The caller's storage: a harness has to pass one in.
                    entry["buffer"] = True
                if argument.get("procedure"):
                    # A procedure argument's "type" is what calling it means,
                    # so the harness on the other side gets the interface
                    # itself: it has to build something callable, and the
                    # emitted body calls it by the same in/out split.
                    entry["interface"] = self._interface(argument.get("interface"))
                arguments.append(entry)
            table[emit_name(subprogram)] = {
                "kind": subprogram["kind"],
                "public": bool(subprogram.get("public", True)),
                "args": arguments,
                "result": subprogram.get("result"),
                "result_dtype": subprogram.get("result_dtype"),
                # What the body's own entry checks say its dummies' shapes
                # must be. An assumed-shape dummy declares neither extent, so
                # for a harness that has to supply one this is the only
                # statement of it there is.
                **(
                    {"shape_guards": subprogram["shape_guards"]}
                    if subprogram.get("shape_guards")
                    else {}
                ),
                # ... and what they say about their values. An integer
                # dummy the body will only take two values of is a mode
                # selector nothing else declares as one.
                **(
                    {"value_guards": subprogram["value_guards"]}
                    if subprogram.get("value_guards")
                    else {}
                ),
            }
        return table

    def _interface(self, name: Any) -> dict[str, Any] | None:
        """One abstract interface, in the same shape a signature entry has.

        ``None`` when the declaration named no interface -- ``procedure()``
        says a name is callable and nothing about the call -- because a
        harness cannot build a callable it has no argument list for, and an
        empty one would be a guess.
        """
        record = (self.subprograms.record.get("interfaces") or {}).get(str(name))
        if record is None:
            return None
        arguments = []
        for argument in record["args"]:
            entry: dict[str, Any] = {
                "name": argument["name"],
                "dtype": argument["dtype"],
                "intent": argument["intent"],
                "optional": argument.get("optional", False),
            }
            if argument.get("dims"):
                entry["dims"] = [
                    {"lb": d.get("lb", "1"), "ub": d.get("ub")} for d in argument["dims"]
                ]
            arguments.append(entry)
        return {
            "kind": record["kind"],
            "args": arguments,
            "result": record.get("result"),
            "result_dtype": record.get("result_dtype"),
        }

    # -- plumbing -------------------------------------------------------------

    @staticmethod
    def _subprogram_nodes(source: Path) -> dict[str, Any]:
        tree = parse(source)
        found = walk(tree, f03.Module)
        scope = found[0] if found else tree
        from recast.fortran.frontend import _subprograms_of

        return dict(_subprograms_of(scope))

    def _rebase(self, text: str, report: list[dict[str, Any]]) -> None:
        """Rewrite every entry's ``py_lines`` as final-file line numbers.

        Scanned back out of the finished text by its block markers rather
        than accumulated during emission, so the numbers cannot drift from
        the file they describe.
        """
        lines = text.splitlines()
        starts: dict[str, int] = {}
        markers: list[tuple[str, str, int]] = []
        refused_state: dict[str, int] = {}
        current = None
        for number, line in enumerate(lines, 1):
            defined = DEFINITION.match(line)
            if defined:
                current = defined.group(1)
                starts[current] = number
                continue
            marked = MARKER.match(line)
            if marked and current:
                markers.append((current, marked.group(1), number))
                continue
            refused = STATE_REFUSAL.match(line)
            if refused:
                # Anchored at column 0, so it cannot match inside a function.
                refused_state[refused.group(1)] = number
        ordered = [
            pysafe(emit_name(s))
            for s in self.subprograms.record["subprograms"]
            if pysafe(emit_name(s)) in starts
        ]
        ends = {}
        for at, name in enumerate(ordered):
            low = starts[name]
            high = starts[ordered[at + 1]] - 1 if at + 1 < len(ordered) else len(lines)
            while high > low and not lines[high - 1].strip():
                high -= 1
            ends[name] = high  # the function's trailing `return` line
        spans = {}
        for at, (name, block, number) in enumerate(markers):
            following = (
                markers[at + 1][2] - 1
                if at + 1 < len(markers) and markers[at + 1][0] == name
                else ends[name] - 1
            )
            spans[(name, block)] = [number, following]
        for entry in report:
            key = (pysafe(entry["subprogram"]), entry["block"])
            if key in spans:
                entry["py_lines"] = spans[key]
                continue
            if str(entry.get("key", "")).startswith("module-state:"):
                # A refused module-state binding is one line at module scope,
                # found by its AGENT_QUEUE comment; a report entry with no
                # such line is the report and the text disagreeing.
                binding = refused_state.get(key[0])
                if binding is None:
                    raise ConfigError(
                        f"module state {entry['subprogram']!r} is recorded as deferred but "
                        "the emitted file carries no AGENT_QUEUE binding for it"
                    )
                entry["py_lines"] = [binding, binding]
                continue
            # A DATA block carries no marker -- the pipeline emits none and
            # the emitted text is compared to it byte for byte -- so its
            # lines are shifted by where its subprogram landed instead.
            offset = starts.get(key[0])
            if offset is None:
                raise ConfigError(
                    f"block {entry['block']} of {entry['subprogram']!r} has no place in the "
                    "emitted file; the report and the text disagree"
                )
            low, high = entry["py_lines"]
            entry["py_lines"] = [offset + low, offset + high - 1]
