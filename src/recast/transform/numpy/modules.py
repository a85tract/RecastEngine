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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from recast.errors import ConfigError
from recast.fortran._parse import f03, parse, walk
from recast.fortran.interface import emit_name, subprogram_key
from recast.transform.numpy import runtime
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.numpy.vocabulary import pysafe

__all__ = ["Modules"]

STATE_DTYPES = {"float64": "np.float64", "int32": "np.int32", "bool": "np.bool_"}
"""Module-array state dtypes. Narrower than the allocate map on purpose:
this is what the pipeline recognized here, and a float32 module array in the
corpus would have silently become float64 -- so it must keep doing so until
a gate says otherwise."""

INTEGER_TEXT = re.compile(r"-?\s*\d+")
REAL_TEXT = re.compile(r"-?\s*(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_\w+)?", re.I)
CHARACTER_TEXT = re.compile(r"'[^']*'|\"[^\"]*\"")


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
        text = self.header() + "\n".join(body)
        self._rebase(text, report)
        return text, report

    def header(self) -> str:
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
        if self.subprograms.intrinsics.get("import"):
            pieces.append(str(self.subprograms.intrinsics["import"]))
        pieces.append("")
        pieces.append(f"from {self.constants_stem} import *  # noqa: F401,F403")
        if self.subprograms.use_parameters:
            pieces.append(f"from {self.use_constants_stem} import *  # noqa: F401,F403")
        if self.subprograms.externals:
            shims = self.externals_module or (record["module"] + "_externals")
            pieces.append(f"import {shims} as _ext")
        extra = sorted(
            set(self.companion_imports)
            | set(self.subprograms.stub_imports)
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
            lines.extend(self._state(state))
        lines.append("")
        for record in self.subprograms.record["subprograms"]:
            node = nodes.get(subprogram_key(record))
            if node is None:
                continue
            rendered, entries = self.subprograms.render(node, subprogram_key(record))
            lines.extend(rendered)
            report.extend(entries)
        return lines, report

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

    def _component_shape(self, component: dict[str, Any]) -> str | None:
        """A derived-type component's allocation shape, or None.

        Module scope, so no subprogram-local name applies: an extent is a
        digit or a constant reachable here -- a module parameter, a
        use-imported one, a companion's global. Anything else leaves the
        component None in the factory rather than a guessed size.
        """
        extents = []
        for dim in component.get("dims") or []:
            if dim.get("lb") not in (None, "1") or not dim.get("ub"):
                return None
            text = dim["ub"].strip().lower()
            if re.fullmatch(r"\d+", text):
                extents.append(text)
                continue
            if not re.fullmatch(r"[a-z_]\w*", text):
                return None
            for table in (
                {
                    p["name"]: p["name"].upper()
                    for p in self.subprograms.record["module_parameters"]
                },
                self.subprograms.use_parameters,
                self.subprograms.companion_globals,
            ):
                if text in table:
                    extents.append(table[text])
                    break
            else:
                return None
        return ", ".join(extents) or None

    def _factory(self, type_name: str, components: dict[str, Any]) -> list[str]:
        lines = [
            f"def _make_{type_name}():",
            f'    """factory for type({type_name}) (components per Derived_Type_Def)."""',
            "    o = _new_derived()",
        ]
        for name, component in components.items():
            safe = pysafe(name)
            dims = component.get("dims")
            shape = self._component_shape(component) if dims else None
            if shape is not None:
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

    # -- module state ---------------------------------------------------------

    def _state(self, state: dict[str, Any]) -> list[str]:
        parameters = {p["name"] for p in self.subprograms.record["module_parameters"]}
        if state.get("dims") and not state["init_expr"]:
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
                    # Module-level array state: a zero buffer, filled by the
                    # module's init routine (Fortran SAVE semantics).
                    dtype = STATE_DTYPES.get(state["dtype"], "np.float64")
                    shape = ", ".join(extents)
                    if dtype == "object":
                        # A character array: np.zeros of dtype object is an
                        # array of the integer 0, and the first thing done to
                        # it is a string comparison.
                        return [
                            f"{state['name']} = np.full(({shape},), '', dtype=object)"
                            "  # module array state"
                        ]
                    return [
                        f"{state['name']} = np.zeros(({shape},), "
                        f"dtype={dtype})  # module array state"
                    ]
            return [
                f"{pysafe(state['name'])} = None  # allocatable/assumed module array, set by init"
            ]
        if state["init_expr"]:
            value = self._state_value(state, parameters)
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
        if expression in (".true.", ".false."):
            return "True" if expression == ".true." else "False"
        if REAL_TEXT.fullmatch(expression):
            base = expression.replace(" ", "").split("_")[0].replace("d", "e")
            return f"np.float64('{base}')"
        if CHARACTER_TEXT.fullmatch(str(state["init_expr"]).strip()):
            return _character_literal(str(state["init_expr"]).strip())
        if expression == "null()":
            return "None  # pointer, null-init"
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
            # An arithmetic constant expression: 2.0_r8 / 7.0_r8.
            numerator, denominator = expression.split("/")
            top = numerator.strip().split("_")[0].replace("d", "e")
            bottom = denominator.strip().split("_")[0].replace("d", "e")
            return f"np.float64({float(top) / float(bottom)!r})"
        return f"None  # TODO: init {state['init_expr']!r}"

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
                if argument.get("dims"):
                    entry["dims"] = [
                        {"lb": d.get("lb", "1"), "ub": d.get("ub")} for d in argument["dims"]
                    ]
                arguments.append(entry)
            table[emit_name(subprogram)] = {
                "kind": subprogram["kind"],
                "args": arguments,
                "result": subprogram.get("result"),
                "result_dtype": subprogram.get("result_dtype"),
            }
        return table

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
