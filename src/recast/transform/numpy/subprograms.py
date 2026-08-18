"""Whole subprograms, assembled from the floors below.

The statement layer renders one statement; this one renders the function
around them, and everything it adds is about Fortran's model of a subprogram
rather than about any statement in it:

* The signature reorders arguments -- Fortran passes everything by
  reference, so an ``intent(out)`` argument is not a parameter at all but a
  slot in the return tuple, and an optional one becomes a ``want_<name>``
  keyword carrying ``present()``.
* The prologue exists because Fortran leaves things undefined that Python
  cannot: locals hold stack garbage until assigned, an ``intent(out)`` array
  arrives as raw storage, a function result may never be set on some path.
  Each gets a determinizing initialization, recorded as a UB-only deviation.
* The body is emitted block by block, each with a marker naming the source
  lines it came from, because everything downstream -- the notary, the
  coverage gate, the agent queue -- keys on those block ids. A block with an
  operator-supplied patch takes the patch; a block the statement layer
  refuses becomes a ``raise NotImplementedError`` carrying the reason, which
  is the deferred site the agent layer consumes.
* A ``goto`` jumping forward across top-level blocks becomes the same
  labelled-exception region the statement layer builds inside a loop, spelled
  here because only this layer sees the whole block list.

The report this returns alongside the lines is not a log; it is the record
of which blocks are mechanical and which are deferred, and it travels with
the Candidate so a verifier can gate on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03, walk
from recast.fortran.chunk import chunk_subprogram
from recast.fortran.semantics import Semantics, for_subprogram
from recast.transform.numpy.expressions import Expressions, Remote
from recast.transform.numpy.names import for_subprogram as names_for
from recast.transform.numpy.statements import ALLOCATED_DTYPES, REFUSED, Statements
from recast.transform.numpy.vocabulary import pysafe
from recast.transform.profiles import Profile
from recast.transform.rules import NoRule

__all__ = ["Subprograms"]

KIND_SUFFIX = re.compile(r"(?<![A-Za-z_])(\d\.?\d*(?:[eE][+-]?\d+)?)\s*_\w+", re.I)
D_EXPONENT = re.compile(r"(\d\.?\d*)D([+-]?\d+)", re.I)
ARRAY_CONSTRUCTOR = re.compile(r"\(/\s*(.*?)\s*/\)", re.S)
DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")
INTEGER_TEXT = re.compile(r"-?\s*\d+")
REAL_TEXT = re.compile(r"-?\s*(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_\w+)?", re.I)


def strip_kind(expression: str) -> str:
    """``'2._r8 * PI'`` -> ``'2. * PI'``, ``'1.0D-3'`` -> ``'1.0e-3'``."""
    return KIND_SUFFIX.sub(r"\1", D_EXPONENT.sub(r"\1e\2", expression))


@dataclass
class Subprograms:
    """Assemble subprograms for one module.

    Holds the per-module context -- the frontend's two records, the
    operator's tables, the compiler profile -- and builds the per-subprogram
    stack (semantics, names, expressions, statements) fresh for each
    ``render``, because everything in that stack is scoped to one subprogram.
    """

    record: dict[str, Any]
    """The ``interface.extract`` record for the module being translated."""

    constants: dict[str, Any]
    """The ``constants.extract`` record: hoisted literals and their names."""

    profile: Profile

    companions: tuple[dict[str, Any], ...] = ()
    use_parameters: dict[str, str] = field(default_factory=dict)
    companion_globals: dict[str, str] = field(default_factory=dict)
    externals: dict[str, dict[str, Any]] = field(default_factory=dict)
    remotes: dict[str, Remote] = field(default_factory=dict)
    function_stubs: dict[str, str] = field(default_factory=dict)
    statement_stubs: dict[str, str] = field(default_factory=dict)

    patches: dict[str, dict[str, Any]] = field(default_factory=dict)
    """``"subprogram/block"`` -> an operator-audited replacement for a block
    the mechanical rules refuse. Applied verbatim, and recorded as such."""

    # -- the per-subprogram stack ---------------------------------------------

    def floors(self, name: str) -> Statements:
        """Semantics, names, expressions and statements for one subprogram."""
        semantics = for_subprogram(self.record, name, companions=self.companions)
        names = names_for(
            semantics,
            self.constants,
            use_parameters=self.use_parameters,
            companion_globals=self.companion_globals,
        )
        expressions = Expressions(
            semantics,
            names,
            self.profile,
            externals=self.externals,
            remotes=self.remotes,
            stubs=dict(self.function_stubs),
            elemental=_is_elemental(semantics.subprogram),
        )
        return Statements(
            semantics,
            names,
            expressions,
            externals=self.externals,
            stubs=dict(self.statement_stubs),
        )

    # -- assembly -------------------------------------------------------------

    def render(self, node: Any, name: str) -> tuple[list[str], list[dict[str, Any]]]:
        """One whole subprogram: its emitted lines and its block report."""
        statements = self.floors(name)
        statements.scan(node)
        semantics = statements.semantics
        subprogram = semantics.subprogram

        lines = [self.signature(subprogram)]
        span = subprogram["line_span"]
        lines.append(f'    """L{span[0]}-L{span[1]} {subprogram["kind"]} (machine-translated)."""')

        if subprogram["module_state_written"]:
            shadowed = {a["name"] for a in subprogram["args"]} | {
                local["name"] for local in subprogram.get("locals") or []
            }
            written = sorted(
                pysafe(n) for n in subprogram["module_state_written"] if n not in shadowed
            )
            if written:
                lines.append("    global " + ", ".join(written))

        lines.extend(self._result_initializer(subprogram, semantics))

        prologue = self._prologue(subprogram, semantics, statements)
        if prologue:
            lines.append(
                "    # UB-guard + automatic-array allocation "
                "(Fortran locals undefined until assignment)"
            )
            lines.extend(prologue)

        report: list[dict[str, Any]] = []
        blocks = list(chunk_subprogram(node))

        # A subprogram-level forward goto-region: `goto L` in top-level
        # blocks i..j-1 jumping to a top-level `L continue` block j wraps
        # those blocks in try/except. Detection mirrors the do-body rule,
        # and only the first such region is taken.
        region_open: dict[int, str] = {}
        region_close: dict[int, str] = {}
        for at, (_, statement, _) in enumerate(blocks):
            if isinstance(statement, f03.Continue_Stmt):
                label = statements.label(statement)
                if not label or label in statements.consumed_labels:
                    continue
                jumps = [
                    earlier
                    for earlier in range(at)
                    if any(
                        str(goto.children[0]) == label
                        for goto in walk(blocks[earlier][1], f03.Goto_Stmt)
                    )
                ]
                if jumps and not region_open:
                    region_open[jumps[0]] = label
                    region_close[at] = label

        in_region = None
        for at, (block, statement, span) in enumerate(blocks):
            if at in region_open:
                label = region_open[at]
                lines.append(f"    try:  # forward-goto region (label {label})")
                statements.active_labels.append(("region", label))
                in_region = label
            if at in region_close:
                statements.active_labels.pop()
                lines.append("    except _FGoto as _g:")
                lines.append(f"        if _g.args[0] != '{region_close[at]}':")
                lines.append("            raise")
                lines.append(f"        pass  # {region_close[at]} continue (region exit)")
                in_region = None
                # The label block itself is consumed by the wrapper, but
                # still emits its marker (re-scans of the output key on it).
                lines.append(f"    # {block} <- L{span[0]}-L{span[1]}")
                lines.append("    pass  # label block consumed by region")
                report.append(
                    {
                        "subprogram": subprogram["name"],
                        "block": block,
                        "src_span": list(span),
                        "status": "mechanical",
                        "py_lines": [len(lines) - 1, len(lines)],
                    }
                )
                continue
            entry: dict[str, Any] = {
                "subprogram": subprogram["name"],
                "block": block,
                "src_span": list(span),
            }
            start = len(lines)
            patch = self.patches.get(f"{subprogram['name']}/{block}")
            if patch is not None:
                lines.append(
                    f"    # {block} <- L{span[0]}-L{span[1]} AGENT-PATCHED ({patch['reason']})"
                )
                lines.extend("    " + patched for patched in patch["python"])
                entry["status"] = "agent_patched"
                entry["reason"] = patch["reason"]
                entry["py_lines"] = [start + 1, len(lines)]
                report.append(entry)
                continue
            try:
                body = statements.render(statement, 2 if in_region else 1)
                lines.append(f"    # {block} <- L{span[0]}-L{span[1]}")
                lines.extend(body)
                entry["status"] = "mechanical"
            except REFUSED as refusal:
                pad = "    " * (2 if in_region else 1)
                lines.append(f"{pad}# {block} <- L{span[0]}-L{span[1]} AGENT_QUEUE: {refusal}")
                lines.append(f"{pad}raise NotImplementedError({str(refusal)!r})  # {block}")
                entry["status"] = "agent_queue"
                entry["reason"] = str(refusal)
            entry["py_lines"] = [start + 1, len(lines)]
            report.append(entry)

        tail = statements.returned_value()
        last_code = next(
            (line for line in reversed(lines) if line.strip() and not line.strip().startswith("#")),
            "",
        )
        if not last_code.strip().startswith("return"):
            lines.append(f"    return {tail}" if tail else "    return")
        lines.append("")
        return lines, report

    def signature(self, subprogram: dict[str, Any]) -> str:
        """The ``def`` line: in-arguments positionally, optionals as
        keywords, optional outputs as their ``want_<name>`` sentinels."""
        positional: list[str] = []
        keyword: list[str] = []
        for argument in subprogram["args"]:
            if Statements.is_optional_output(argument):
                keyword.append(f"want_{argument['name']}=False")
            elif argument["intent"] in ("IN", "INOUT", "UNKNOWN"):
                (keyword if argument["optional"] else positional).append(
                    pysafe(argument["name"]) + ("=None" if argument["optional"] else "")
                )
        return f"def {pysafe(subprogram['name'])}({', '.join(positional + keyword)}):"

    # -- the determinizing prologue -------------------------------------------

    def _result_initializer(self, subprogram: dict[str, Any], semantics: Semantics) -> list[str]:
        """A function result is undefined until assigned; a path that never
        assigns it -- a SELECT CASE with no matching branch -- is UB in
        Fortran. Pre-initializing is a documented UB-only deviation."""
        if subprogram["kind"] != "function":
            return []
        lines = []
        if subprogram.get("result_dims"):
            shape = ", ".join(d.get("ub", "1") for d in subprogram["result_dims"])
            lines.append(f"    {subprogram['result']} = np.zeros(({shape},), dtype=np.float64)")
        elif subprogram["result_dtype"] in ("float64", "float32"):
            lines.append(f"    {subprogram['result']} = 0.0")
        if subprogram["result_dtype"] in ("int32", "int64"):
            lines.append(f"    {subprogram['result']} = 0")
        if subprogram["result_dtype"] == "bool":
            lines.append(f"    {subprogram['result']} = False")
        if subprogram["result_dtype"] == "str":
            lines.append(f"    {subprogram['result']} = ''")
        derived = DERIVED.match(str(subprogram["result_dtype"]))
        if derived is not None:
            type_name = derived.group(1).lower()
            constructor = (
                f"_make_{type_name}()" if type_name in semantics.types else "_new_derived()"
            )
            lines.append(f"    {subprogram['result']} = {constructor}")
        return lines

    def _prologue(
        self, subprogram: dict[str, Any], semantics: Semantics, statements: Statements
    ) -> list[str]:
        lines: list[str] = []
        # intent(out)-only arguments are NOT parameters (return convention):
        # the function owns their buffers. Arrays get np.empty -- contents
        # undefined, exactly like Fortran -- and scalars the UB-guard zero.
        for argument in subprogram["args"]:
            if argument["intent"] != "OUT":
                continue
            lines.extend(self._out_argument(argument, subprogram, statements))
        # Local PARAMETERs, as local assignments (matches Fortran scope).
        for parameter in subprogram["local_parameters"]:
            initializer = parameter.get("init_expr", "0")
            if initializer is None:
                continue
            lines.append(
                f"    {pysafe(parameter['name'])} = {self._parameter_value(initializer.strip())}"
            )
        parameter_names = {p["name"] for p in subprogram["local_parameters"]}
        for local in subprogram["locals"]:
            if local["name"] in parameter_names:
                continue
            lines.extend(self._local(local, semantics, statements))
        return lines

    def _out_argument(
        self, argument: dict[str, Any], subprogram: dict[str, Any], statements: Statements
    ) -> list[str]:
        name = pysafe(argument["name"])
        if argument.get("optional"):
            if argument.get("dims"):
                value = "None"
            elif argument["dtype"] in ("float64", "float32"):
                value = "0.0"
            elif argument["dtype"] in ("int32", "int64"):
                value = "0"
            else:
                value = "None"
            return [f"    {name} = {value}  # optional OUT: may not be assigned"]
        dims = argument.get("dims")
        if dims and any(d["ub"] is None for d in dims):
            # An assumed-shape OUT: Fortran takes the extent from the actual;
            # the convention here allocates inside, borrowing the shape of a
            # same-rank assumed-shape IN argument.
            donor = next(
                (
                    other["name"]
                    for other in subprogram["args"]
                    if other["intent"] in ("IN", "INOUT")
                    and len(other.get("dims") or []) == len(dims)
                    and any(d["ub"] is None for d in other["dims"])
                ),
                None,
            )
            if donor is None:
                # An allocatable OUT with no donor: None, so that translated
                # allocated() -> `is not None` checks keep working.
                return [f"    {name} = None  # out-arg: assumed shape, no donor"]
            dtype = ALLOCATED_DTYPES.get(argument["dtype"], "np.float64")
            return [f"    {name} = np.empty(np.shape({pysafe(donor)}), dtype={dtype})"]
        if dims and all(d["ub"] is not None for d in dims):
            try:
                shape = ", ".join(self._extent(d, statements) for d in dims)
            except REFUSED as refusal:
                return [f"    # out-arg {argument['name']}: allocation skipped ({refusal})"]
            dtype = ALLOCATED_DTYPES.get(argument["dtype"], "np.float64")
            return [f"    {name} = np.empty(({shape},), dtype={dtype})"]
        if not dims and argument["dtype"] in ("float64", "float32"):
            return [f"    {name} = 0.0"]
        if not dims and argument["dtype"] in ("int32", "int64"):
            return [f"    {name} = 0"]
        if not dims and argument["dtype"] == "bool":
            return [f"    {name} = False"]
        return []

    def _local(
        self, local: dict[str, Any], semantics: Semantics, statements: Statements
    ) -> list[str]:
        name = pysafe(local["name"])
        dims = local.get("dims")
        if dims:
            if any(d["ub"] is None for d in dims):
                # Allocatable: None until its Allocate_Stmt (allocated() rule).
                return [f"    {name} = None"]
            try:
                shape = ", ".join(self._extent(d, statements) for d in dims)
            except REFUSED as refusal:
                return [
                    f"    # {local['name']}: array prologue skipped"
                    f" ({refusal}) — first use will AgentQueue"
                ]
            dtype = ALLOCATED_DTYPES.get(local["dtype"], "np.float64")
            return [f"    {name} = np.empty(({shape},), dtype={dtype})"]
        if local.get("array_spec"):
            return []
        derived = DERIVED.match(str(local["dtype"]))
        if derived is not None:
            type_name = derived.group(1).lower()
            constructor = (
                f"_make_{type_name}()" if type_name in semantics.types else "_new_derived()"
            )
            return [f"    {name} = {constructor}"]
        if local["dtype"] in ("float64", "float32"):
            return [f"    {name} = 0.0"]
        if local["dtype"] in ("int32", "int64"):
            return [f"    {name} = 0"]
        if local["dtype"] == "bool":
            return [f"    {name} = False"]
        if local["dtype"] == "str":
            return [f"    {name} = ''"]
        return []

    @staticmethod
    def _parameter_value(text: str) -> str:
        """A local parameter's initializer, rendered from its source text.

        Text-level, not node-level: declarations were extracted as text, and
        the pipeline this reproduces rendered them the same way.
        """
        if text.lower() in (".true.", ".false."):
            return "True" if "true" in text.lower() else "False"
        if INTEGER_TEXT.fullmatch(text):
            return str(int(text.replace(" ", "")))
        if REAL_TEXT.fullmatch(text):
            return "np.float64('" + text.replace(" ", "").split("_")[0].replace("d", "e") + "')"
        constructed = ARRAY_CONSTRUCTOR.search(text)
        if constructed:
            items = [strip_kind(item.strip()) for item in constructed.group(1).split(",")]
            if all(re.fullmatch(r"'[^']*'", item) for item in items):
                return f"np.array([{', '.join(items)}])"
            if all(re.fullmatch(r"-?\d+", item.strip()) for item in items):
                return f"np.array([{', '.join(items)}], dtype=np.int32)"
            return f"np.array([{', '.join(items)}])"
        return strip_kind(text).upper()

    # -- declared extents -----------------------------------------------------

    @staticmethod
    def _extent(dim: dict[str, Any], statements: Statements) -> str:
        """The length of a declared dimension: ``ub - lb + 1``."""
        if dim["ub"] is None:
            raise NoRule("extent of assumed/deferred dim")
        upper = statements.bound(dim["ub"])
        lower = dim.get("lb")
        if lower in (None, "1", ":"):
            return upper
        return f"({upper}) - ({statements.bound(lower)}) + 1"


def _is_elemental(subprogram: dict[str, Any]) -> bool:
    return any("ELEMENTAL" in str(p).upper() for p in (subprogram.get("prefixes") or []))
