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
from recast.fortran.interface import emit_name, node_span, subprogram_key
from recast.fortran.semantics import Semantics, for_subprogram
from recast.transform.numpy.agentic import DeferredHandler, DeferredSite
from recast.transform.numpy.expressions import CONFLICTING_BOUNDS, Expressions, Remote
from recast.transform.numpy.names import bind_use_statements
from recast.transform.numpy.names import for_subprogram as names_for
from recast.transform.numpy.statements import (
    ALLOCATED_DTYPES,
    REFUSED,
    Statements,
    derived_array,
    undefined_array,
)
from recast.transform.numpy.vocabulary import (
    ELEMENTAL_ARRAY,
    ELEMENTAL_SCALAR,
    REDUCTIONS,
    pysafe,
)
from recast.transform.profiles import Profile
from recast.transform.rules import NoRule

__all__ = ["Subprograms"]

KIND_SUFFIX = re.compile(r"(?<![A-Za-z_])(\d\.?\d*(?:[eE][+-]?\d+)?)\s*_\w+", re.I)
D_EXPONENT = re.compile(r"(\d\.?\d*)D([+-]?\d+)", re.I)
ARRAY_CONSTRUCTOR = re.compile(r"\(/\s*(.*?)\s*/\)", re.S)
DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")
INTEGER_TEXT = re.compile(r"-?\s*\d+")
BOZ_TEXT = re.compile(r"[zboZBO]'([0-9a-fA-F]+)'")
IDENTIFIER = re.compile(r"[a-zA-Z_]\w*")
REAL_TEXT = re.compile(r"-?\s*(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_\w+)?", re.I)


UPPERCASED_CALL = re.compile(r"\b[A-Z_][A-Z0-9_]*\s*\(")
"""A name the token pass spelled as a module constant, followed by ``(``.

A constant is never called and an array constant is not indexed with
parentheses in Python, so this is the token pass having uppercased an
intrinsic (``SIN(0.5)``) or an array reference: syntactically Python, and a
NameError or TypeError the first time the function runs."""


def _token_pass_guessed(text: str, spelled: str) -> bool:
    """Whether the token pass produced Python that cannot mean the Fortran.

    ``_is_expression`` only asks whether Python can *parse* it. Three shapes
    parse and are still wrong: a call or array reference of an uppercased
    name; a ``//`` concatenation, which Python reads as floor division; and an
    array constructor that was only part of the text (``reshape((/.../),
    ...)``), where the search silently dropped everything around it.
    """
    if UPPERCASED_CALL.search(spelled):
        return True
    if "//" in text:
        return True
    constructed = ARRAY_CONSTRUCTOR.search(text)
    return constructed is not None and constructed.span() != (0, len(text))


def _is_expression(text: str) -> str | bool:
    """Whether ``text`` is something Python can evaluate.

    The check the token pass never had. Cheap, and it is the only thing
    standing between an initializer nobody could translate and a module that
    does not import.
    """
    try:
        compile(text, "<initializer>", "eval")
    except SyntaxError:
        return False
    return True


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

    The two floors below are named here rather than constructed inline so a
    second backend can replace them. A backend that re-emits the same Fortran
    with different spellings -- Numba's, whose kernels take the module state
    they read as explicit parameters -- differs from this one in perhaps a
    dozen rules spread across the expression and statement layers, and in
    nothing about the assembly around them. Subclassing the floors and
    naming the subclasses here is how it says that, instead of copying the
    assembly to change twelve lines inside it.
    """

    expressions_class = Expressions
    statements_class = Statements

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
    intrinsics: dict[str, Any] = field(default_factory=dict)
    runtime_imports: tuple[str, ...] = ()
    """Import lines the emitted module needs at its top, from the package
    that supplies whatever its transforms and spellings call."""

    call_transforms: dict[str, Any] = field(default_factory=dict)
    function_transforms: dict[str, Any] = field(default_factory=dict)
    handle_producers: frozenset[str] = frozenset()

    buffer_out_arrays: bool = True
    """Apply the caller-buffer convention to intent(out) arrays (#36).

    An ``intent(out)`` array the callee cannot size is the caller's storage,
    so it stays a parameter and is returned like an INOUT. Off, a rank-3
    assumed-shape OUT with no donor is bound to ``None`` and the first write
    into it raises -- ``ndrop.dropmixnuc``'s ``factnum`` is that case.

    Upstream defaults this on and turns it off only for its own CESM project,
    where the validated slots and their post-patches were built on the older
    convention. That carve-out is about a patch baseline this repository does
    not carry, so it belongs to whoever has one -- set the field, as upstream
    sets it from its CLI -- rather than to the default here.
    """
    type_bound_procedures: frozenset[str] = frozenset()

    poison_undefined: bool = False
    """Fill an undefined float local with NaN instead of leaving it undefined.

    Off by default, and the default is not the safe-looking one. A Fortran
    local is undefined until assigned; this backend gives every one of them an
    initializer, which makes a read-before-write *reproducible* -- and a
    reproducible wrong number is the kind that survives a run, a re-run and a
    reviewer. Unpoisoned, that initializer is a zero fill, which is
    reproducible but says nothing -- see ``undefined_array``.

    Turned on, every float array Fortran would have left undefined is
    NaN-filled instead of merely allocated, so a read of an unwritten cell
    propagates to the outputs the gate compares and ``differential.bitexact``
    counts it as ``nan_mismatch``.

    Integer arrays are a separate switch, ``poison_integers``, because the
    detector is separate -- see there.

    Scalars are untouched. A scalar local gets the UB-guard zero either way,
    which is the same reproducible-not-visible problem one level down, and no
    smaller: it is simply not what the tool this came from covers, and
    covering it is a separate question rather than a free extension of this
    flag.

    A poisoned candidate is not the one being shipped -- which is why this is
    a separate run rather than a default, and why its digest differs.
    """

    poison_integers: bool = False
    """The integer arm of ``poison_undefined``, and off even when that is on.

    An undefined integer read cannot announce itself the way a float one can:
    there is no integer NaN, so the fill is ``INT32_MIN + 1`` -- an impossible
    index -- and reading an unwritten cell either crashes on the subscript or
    shifts the poisoned run's outputs. Nothing propagates to a NaN scan; the
    detector is an A/B diff against the unpoisoned run.

    Two switches rather than one because of that, which is how the tool this
    came from has it: the float arm is answered by the gate already running,
    the integer arm needs a second run and a comparison, and turning both on
    together would report the first while quietly requiring the second.

    Has no effect unless ``poison_undefined`` is on.
    """

    patches: dict[str, dict[str, Any]] = field(default_factory=dict)
    """``"subprogram/block"`` -> an operator-audited replacement for a block
    the mechanical rules refuse. Applied verbatim, and recorded as such."""

    use_bindings: dict[str, str] = field(init=False, default_factory=dict)
    intrinsic_aliases: frozenset[str] = field(init=False, default=frozenset())
    """Aliases bound to the runtime's intrinsic-module namespaces.

    No header line imports them -- they are already in the inlined runtime --
    so nothing downstream can infer them from the emitted imports the way it
    can for a companion. Carried here for the read/write protocol."""

    stub_imports: tuple[str, ...] = field(init=False, default=())
    """Derived in ``__post_init__`` from the module's USE statements; see
    ``names.bind_use_statements``. ``stub_imports`` are the header lines for
    USE'd modules that are not companions."""

    deferred_handler: DeferredHandler | None = None
    """Consulted at the moment a block is refused, with the refusal in hand.

    The other half of ``patches``, and not a duplicate of it: a patch is known
    before rendering starts and may add imports, a handler runs during
    rendering and cannot. Supplying one makes the run non-deterministic; see
    ``recast.transform.numpy.agentic`` for who is allowed to.
    """

    def _fill(
        self,
        subprogram: str,
        block: str,
        statement: Any,
        span: Any,
        refusal: Exception,
        statements: Statements,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Ask the handler for this block. Returns ``(filled, why_not)``.

        A handler that raises, or answers with something that is not a list of
        source lines, leaves the site deferred and says what went wrong on the
        block. Refusing is already a normal answer here, so a broken handler
        degrades to the answer the site would have had anyway -- rather than
        ending the run over one block, or swallowing the failure so that a
        transform silently stops filling anything.
        """
        if self.deferred_handler is None:
            return None, None
        site = DeferredSite(
            subprogram=subprogram,
            block=block,
            fortran=str(statement),
            src_span=(int(span[0]), int(span[1])),
            reason=str(refusal),
            names=statements.names.as_protocol_table(),
        )
        try:
            filled = self.deferred_handler(site)
        except Exception as error:  # a plugin's handler is not the run's life
            return None, f"handler raised {type(error).__name__}: {error}"
        if filled is None:
            return None, None
        body = filled.get("python")
        if not isinstance(body, list) or not all(isinstance(line, str) for line in body):
            return None, "handler returned no 'python' list of source lines"
        return dict(filled), None

    def __post_init__(self) -> None:
        aliases = {remote.alias for remote in self.remotes.values()}
        aliases |= {spelling.split(".")[0] for spelling in self.companion_globals.values()}
        self.use_bindings, stubs, intrinsic = bind_use_statements(
            self.record, aliases, set(self.remotes), self.companion_globals
        )
        self.intrinsic_aliases = frozenset(intrinsic)
        self.stub_imports = tuple(f"import {mod}_numpy as {alias}" for mod, alias in stubs.items())

    # -- the per-subprogram stack ---------------------------------------------

    def floors(self, name: str) -> Statements:
        """Semantics, names, expressions and statements for one subprogram."""
        semantics = for_subprogram(self.record, name, companions=self.companions)
        names = names_for(
            semantics,
            self.constants,
            use_parameters=self.use_parameters,
            companion_globals=self.companion_globals,
            use_bindings=self.use_bindings,
        )
        shadowed = {a["name"] for a in semantics.subprogram["args"]}
        shadowed |= {loc["name"] for loc in semantics.subprogram.get("locals") or ()}
        # A companion's allocatable keeps the lower bound its own ALLOCATE
        # gave it when this module subscripts it (``vcmax_np1(itype)`` over
        # pftvarcon's ``allocate (vcmax_np1 (0:mxpft))``): the reader's
        # declaration is the companion's ``(:)``, which says nothing, and
        # the blanket one-based shift lands a slot off. Companions that
        # disagree on a name leave it conflicting; the module's own record
        # wins over any of them. A USE rename is not followed.
        allocated: dict[str, Any] = {}
        for companion in self.companions:
            for name, bounds in (companion.get("module_allocate_bounds") or {}).items():
                if name in allocated and allocated[name] != bounds:
                    allocated[name] = CONFLICTING_BOUNDS
                else:
                    allocated[name] = bounds
        allocated.update(self.record.get("module_allocate_bounds", {}))
        allocated = {name: bounds for name, bounds in allocated.items() if name not in shadowed}
        expressions = self.expressions_class(
            semantics,
            names,
            self.profile,
            externals=self.externals,
            remotes=self.remotes,
            stubs=dict(self.function_stubs),
            function_transforms=dict(self.function_transforms),
            handle_producers=self.handle_producers,
            type_bound=self.type_bound_procedures,
            intrinsics={k: v for k, v in self.intrinsics.items() if isinstance(v, dict)},
            elemental=_is_elemental(semantics.subprogram),
            allocated_bounds=allocated,
        )
        return self.statements_class(
            semantics,
            names,
            expressions,
            externals=self.externals,
            stubs=dict(self.statement_stubs),
            call_transforms=dict(self.call_transforms),
            poison_undefined=self.poison_undefined,
            buffer_out_arrays=self.buffer_out_arrays,
            poison_integers=self.poison_integers,
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

        lines.extend(self._result_initializer(subprogram, semantics, statements))

        prologue_refusals: list[tuple[str, int, int]] = []
        prologue = self._prologue(subprogram, semantics, statements, prologue_refusals)
        # Where the prologue's first line lands: after the header comment.
        prologue_at = len(lines) + 1
        if prologue:
            lines.append(
                "    # UB-guard + automatic-array allocation "
                "(Fortran locals undefined until assignment)"
            )
            lines.extend(prologue)

        report: list[dict[str, Any]] = []
        # Prologue refusals are deferred work like any block: without these
        # entries a skipped allocation or parameter was a comment the bundle
        # never carried, and acceptance had nothing to refuse. Each entry
        # points at the lines the refusal emitted, so the rebase can place
        # it in the finished file like a DATA block.
        for at, (reason, low, high) in enumerate(prologue_refusals, start=1):
            report.append(
                {
                    "subprogram": emit_name(subprogram),
                    "key": subprogram_key(subprogram),
                    "block": f"P{at:03d}",
                    "src_span": [0, 0],
                    "status": "agent_queue",
                    "reason": reason,
                    "py_lines": [prologue_at + low, prologue_at + high],
                }
            )
        # DATA sits in the specification part, and is a static
        # initialisation: its assignments go after the prologue, before any
        # statement can read the names. Its own block ids, because the
        # execution-part chunking never saw it.
        for at, statement in enumerate(walk(_specification(node), f03.Data_Stmt), start=1):
            block = f"D{at:03d}"
            span = node_span(statement)
            data_entry: dict[str, Any] = {
                "subprogram": emit_name(subprogram),
                "key": subprogram_key(subprogram),
                "block": block,
                "src_span": [int(span[0] or 0), int(span[1] or 0)],
            }
            before = len(lines)
            try:
                lines.extend(statements.data_statement(statement, 1))
                data_entry["status"] = "mechanical"
            except REFUSED as refusal:
                reason = f"DATA deferred: {refusal}"
                lines.append(f"    # AGENT_QUEUE: {reason}")
                lines.append(f"    raise NotImplementedError({reason!r})")
                data_entry["status"] = "agent_queue"
                data_entry["reason"] = str(refusal)
            data_entry["py_lines"] = [before, len(lines)]
            report.append(data_entry)

        blocks = list(chunk_subprogram(node))

        # A subprogram-level forward goto-region: `goto L` in top-level
        # blocks i..j-1 jumping to a top-level `L continue` block j wraps
        # those blocks in try/except. Detection mirrors the do-body rule,
        # and only the first such region is taken.
        region_open: dict[int, str] = {}
        region_close: dict[int, str] = {}
        # A subprogram-level *backward* goto-region: `L continue` at block i
        # with `goto L` in a later block j is a loop -- everything from the
        # label to the last such goto runs again. Taken before the forward
        # case, as the statement-level rule orders it, and only one region
        # per subprogram either way.
        back_open: dict[int, str] = {}
        back_close: dict[int, str] = {}
        for at, (_, statement, _) in enumerate(blocks):
            if isinstance(statement, f03.Continue_Stmt):
                label = statements.label(statement)
                if not label or label in statements.consumed_labels:
                    continue
                jumps = [
                    later
                    for later in range(at + 1, len(blocks))
                    if any(
                        str(goto.children[0]) == label
                        for goto in walk(blocks[later][1], f03.Goto_Stmt)
                    )
                ]
                if jumps and not back_open:
                    back_open[at] = label
                    back_close[jumps[-1]] = label
        for at, (_, statement, _) in enumerate(blocks):
            if back_open:
                break
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
        in_loop_region = None
        for at, (block, statement, span) in enumerate(blocks):
            if at in back_open:
                label = back_open[at]
                lines.append(f"    while True:  # backward-goto region (label {label})")
                lines.append("        try:")
                statements.active_labels.append(("region", label))
                in_loop_region = label
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
                lines.append(f"        pass  # {region_close[at]} (region exit)")
                in_region = None
                # The label block itself is consumed by the wrapper, but
                # still emits its marker (re-scans of the output key on it).
                lines.append(f"    # {block} <- L{span[0]}-L{span[1]}")
                lines.append("    pass  # label block consumed by region")
                report.append(
                    {
                        "subprogram": emit_name(subprogram),
                        "key": subprogram_key(subprogram),
                        "block": block,
                        "src_span": list(span),
                        "status": "mechanical",
                        "py_lines": [len(lines) - 1, len(lines)],
                    }
                )
                continue
            entry: dict[str, Any] = {
                "subprogram": emit_name(subprogram),
                "key": subprogram_key(subprogram),
                "block": block,
                "src_span": list(span),
            }
            start = len(lines)
            # Inside a goto region every line of the block is one level
            # deeper, its marker comment included. The refusing path below
            # worked this out and the accepting one did not, so a region's
            # blocks carried their markers at the outer indent.
            depth = 3 if in_loop_region else (2 if in_region else 1)
            pad = "    " * depth
            patch = self.patches.get(f"{subprogram['name']}/{block}")
            if patch is not None:
                lines.append(
                    f"{pad}# {block} <- L{span[0]}-L{span[1]} AGENT-PATCHED ({patch['reason']})"
                )
                lines.extend(pad + patched for patched in patch["python"])
                entry["status"] = "agent_patched"
                entry["reason"] = patch["reason"]
                entry["py_lines"] = [start + 1, len(lines)]
                report.append(entry)
                continue
            try:
                body = statements.render(statement, depth)
                lines.append(f"{pad}# {block} <- L{span[0]}-L{span[1]}")
                lines.extend(body)
                entry["status"] = "mechanical"
            except REFUSED as refusal:
                filled, why_not = self._fill(
                    emit_name(subprogram), block, statement, span, refusal, statements
                )
                if filled is not None:
                    lines.append(
                        f"{pad}# {block} <- L{span[0]}-L{span[1]} "
                        f"AGENT-FILLED ({filled.get('reason', 'no reason given')})"
                    )
                    lines.extend(pad + line for line in filled["python"])
                    # Everything the handler said rides into the report, which
                    # is where a non-deterministic transform's provenance has
                    # to end up -- the model and the prompt, not just the fact
                    # that something answered.
                    entry.update({k: v for k, v in filled.items() if k != "python"})
                    entry["status"] = "agent_filled"
                else:
                    lines.append(f"{pad}# {block} <- L{span[0]}-L{span[1]} AGENT_QUEUE: {refusal}")
                    lines.append(f"{pad}raise NotImplementedError({str(refusal)!r})  # {block}")
                    entry["status"] = "agent_queue"
                    entry["reason"] = str(refusal)
                    if why_not:
                        entry["handler_error"] = why_not
            entry["py_lines"] = [start + 1, len(lines)]
            report.append(entry)
            if at in back_close:
                label = back_close[at]
                statements.active_labels.pop()
                lines.append("            break  # natural exit")
                lines.append("        except _FGoto as _g:")
                lines.append(f"            if _g.args[0] != '{label}':")
                lines.append("                raise")
                lines.append(f"            pass  # {label} (loop restart)")
                in_loop_region = None

        tail = statements.returned_value()
        last_code = next(
            (line for line in reversed(lines) if line.strip() and not line.strip().startswith("#")),
            "",
        )
        # Only a function-level return (indent exactly 4) may stand in for the
        # final one. A return nested in a branch used to match via .strip(),
        # and the path that skipped the branch fell off the end returning
        # None (the translator's T45 rule, hetfrz_classnuc_calc).
        if not last_code.startswith("    return"):
            lines.append(f"    return {tail}" if tail else "    return")
        lines.append("")
        return lines, report

    def _is_caller_buffer(self, argument: dict[str, Any]) -> bool:
        """Whether this formal is an OUT array the caller supplies (#36)."""
        return bool(argument.get("buffer") and self.buffer_out_arrays)

    def _passes(self, argument: dict[str, Any]) -> bool:
        """Whether this formal is a Python parameter of the emitted def."""
        return argument["intent"] in ("IN", "INOUT", "UNKNOWN") or self._is_caller_buffer(argument)

    def signature(self, subprogram: dict[str, Any]) -> str:
        """The ``def`` line: in-arguments positionally, optionals as
        keywords, optional outputs as their ``want_<name>`` sentinels.

        An intent(out) array marked ``buffer`` is a parameter too, when the
        caller-buffer convention is on: the caller owns its storage, and this
        subprogram cannot size it.
        """
        positional: list[str] = []
        keyword: list[str] = []
        for argument in subprogram["args"]:
            if Statements.is_optional_output(argument):
                keyword.append(f"want_{argument['name']}=False")
            elif self._passes(argument):
                (keyword if argument["optional"] else positional).append(
                    pysafe(argument["name"]) + ("=None" if argument["optional"] else "")
                )
        # Host association: an internal procedure receives the host variables
        # it touches as trailing parameters.
        positional.extend(pysafe(hv) for hv in subprogram.get("host_vars") or ())
        return f"def {pysafe(emit_name(subprogram))}({', '.join(positional + keyword)}):"

    # -- the determinizing prologue -------------------------------------------

    def _result_initializer(
        self, subprogram: dict[str, Any], semantics: Semantics, statements: Statements
    ) -> list[str]:
        """A function result is undefined until assigned; a path that never
        assigns it -- a SELECT CASE with no matching branch -- is UB in
        Fortran. Pre-initializing is a documented UB-only deviation."""
        if subprogram["kind"] != "function":
            return []
        lines = []
        # The emitted spelling, not the source one: ``result(lambda)`` is
        # legal Fortran, and every other mention of the result is renamed.
        result = pysafe(subprogram["result"])
        if subprogram.get("result_dims"):
            shape = ", ".join(
                statements.bound(d["ub"]) if d.get("ub") else "1" for d in subprogram["result_dims"]
            )
            lines.append(f"    {result} = np.zeros(({shape},), dtype=np.float64)")
        elif subprogram["result_dtype"] in ("float64", "float32"):
            lines.append(f"    {result} = 0.0")
        if subprogram["result_dtype"] in ("int32", "int64"):
            lines.append(f"    {result} = 0")
        if subprogram["result_dtype"] == "bool":
            lines.append(f"    {result} = False")
        if subprogram["result_dtype"] == "str":
            lines.append(f"    {result} = ''")
        derived = DERIVED.match(str(subprogram["result_dtype"]))
        if derived is not None:
            type_name = derived.group(1).lower()
            constructor = (
                f"_make_{type_name}()" if type_name in semantics.types else "_new_derived()"
            )
            lines.append(f"    {result} = {constructor}")
        return lines

    def _prologue(
        self,
        subprogram: dict[str, Any],
        semantics: Semantics,
        statements: Statements,
        refusals: list[tuple[str, int, int]] | None = None,
    ) -> list[str]:
        # Policy: a refusal is never a comment. Every site below records its
        # reason (the caller turns them into agent_queue report entries) and
        # emits a raise, so partial translation can neither pass a gate nor
        # compute quietly wrong numbers at runtime. Each recorded refusal is
        # ``(reason, low, high)``: the reason and the half-open span, in the
        # returned lines, that it emitted.
        if refusals is None:
            refusals = []
        lines: list[str] = []

        def _emit(emitted: list[str], pending: list[str]) -> None:
            low, high = len(lines), len(lines) + len(emitted)
            lines.extend(emitted)
            for reason in pending:
                refusals.append((reason, low, high))

        # intent(out)-only arguments are NOT parameters (return convention):
        # the function owns their buffers. Arrays go through
        # ``undefined_array`` and scalars get the UB-guard zero.
        for argument in subprogram["args"]:
            if argument["intent"] != "OUT" or self._is_caller_buffer(argument):
                # A buffer argument arrives already allocated; allocating a
                # fresh one here would write the caller's result into an array
                # the caller never sees.
                continue
            pending: list[str] = []
            _emit(
                self._out_argument(argument, subprogram, semantics, statements, pending),
                pending,
            )
        # Local PARAMETERs, as local assignments (matches Fortran scope).
        own_parameters = frozenset(p["name"].lower() for p in subprogram["local_parameters"])
        # #47: a character parameter is the value the frontend folded and
        # fitted to its declared length, referenced from the constants module
        # rather than re-rendered here -- the token pass below would
        # upper-case 'maxi' and let a Fortran // through as Python's.
        folded = {
            (str(rec["subprogram"]).lower(), str(rec["name"]).lower()): rec
            for rec in self.constants.get("local_parameters", ())
        }
        this = str(subprogram.get("name", "")).lower()
        for parameter in subprogram["local_parameters"]:
            initializer = parameter.get("init_expr", "0")
            if initializer is None:
                continue
            name = pysafe(parameter["name"])
            rec = folded.get((this, parameter["name"].lower()))
            if rec is not None and rec["kind"] == "str":
                lines.append(f"    {name} = {rec['const']}")
                continue
            if rec is not None and str(rec["payload"]).startswith("character expression"):
                lines.append(
                    f"    {name} = None  # AGENT_QUEUE: local parameter "
                    f"{parameter['name']} ({rec['payload']})"
                )
                continue
            try:
                value = self._parameter_value(initializer.strip(), own_parameters, statements)
            except REFUSED as refusal:
                reason = f"local parameter {parameter['name']} ({refusal}): {initializer.strip()}"
                _emit(
                    [
                        f"    # AGENT_QUEUE: {reason}",
                        f"    raise NotImplementedError({reason!r})",
                    ],
                    [reason],
                )
                continue
            lines.append(f"    {name} = {value}")
        parameter_names = {p["name"] for p in subprogram["local_parameters"]}
        for local in subprogram["locals"]:
            if local["name"] in parameter_names:
                continue
            if self._types_an_intrinsic(local, statements):
                continue
            pending = []
            _emit(self._local(local, semantics, statements, pending), pending)
        return lines

    @staticmethod
    def _types_an_intrinsic(local: dict[str, Any], statements: Statements) -> bool:
        """Whether ``real(8) :: abs`` types an INTRINSIC rather than declaring
        a local.

        F77 let a program give an intrinsic a type without shadowing it, and
        the declaration reads identically to a scalar local. The body settles
        it: a name it calls and never assigns is the intrinsic, and giving it
        a prologue ``abs = 0.0`` shadows the builtin, so every later ``abs(x)``
        raises "float object is not callable" rather than anything this file
        would notice.
        """
        name = local["name"]
        return (
            not local.get("dims")
            and name not in statements.assigned_names
            and name in statements.called_names
            and (name in ELEMENTAL_SCALAR or name in ELEMENTAL_ARRAY or name in REDUCTIONS)
        )

    def component_shape(self, component: dict[str, Any]) -> str | None:
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
                {p["name"]: p["name"].upper() for p in self.record["module_parameters"]},
                self.use_parameters,
                self.companion_globals,
            ):
                if text in table:
                    extents.append(table[text])
                    break
            else:
                return None
        return ", ".join(extents) or None

    def _out_argument(
        self,
        argument: dict[str, Any],
        subprogram: dict[str, Any],
        semantics: Semantics,
        statements: Statements,
        refusals: list[str] | None = None,
    ) -> list[str]:
        if refusals is None:
            refusals = []
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
            shape = f"np.shape({pysafe(donor)})"
            return [f"    {name} = {undefined_array(self, shape, dtype)}"]
        if dims and all(d["ub"] is not None for d in dims):
            try:
                shape = ", ".join(self._extent(d, statements) for d in dims)
            except REFUSED as refusal:
                reason = f"out-arg {argument['name']}: allocation refused ({refusal})"
                refusals.append(reason)
                return [
                    f"    # AGENT_QUEUE: {reason}",
                    f"    raise NotImplementedError({reason!r})",
                ]
            dtype = ALLOCATED_DTYPES.get(argument["dtype"], "np.float64")
            return [f"    {name} = {undefined_array(self, f'({shape},)', dtype)}"]
        if not dims and argument["dtype"] in ("float64", "float32"):
            return [f"    {name} = 0.0"]
        if not dims and argument["dtype"] in ("int32", "int64"):
            return [f"    {name} = 0"]
        if not dims and argument["dtype"] == "bool":
            return [f"    {name} = False"]
        derived = DERIVED.match(str(argument["dtype"]))
        if not dims and derived is not None:
            # An INTENT(OUT) derived-type dummy is a fresh object at entry:
            # Fortran says the callee sees it undefined, and the return
            # convention makes the callee its owner. It was falling through
            # to nothing, so the body's first ``cart2%x = ...`` ran against
            # whatever the caller passed -- or against a name never bound.
            type_name = derived.group(1).lower()
            components = semantics.types.get(type_name) or {}
            unresolved = next(
                (
                    component_name
                    for component_name, component in components.items()
                    if component.get("dims")
                    and not component.get("allocatable")
                    and not component.get("pointer")
                    and self.component_shape(component) is None
                ),
                None,
            )
            if unresolved is not None:
                # A fixed-shape component whose extent no module-scope name
                # resolves would materialize as None and read as absent.
                # Queue it; never guess a size.
                reason = (
                    f"out-arg {argument['name']}: INTENT(OUT) derived-type dummy not "
                    f"materialized at function entry (type({type_name})%{unresolved} "
                    "dims not statically resolvable)"
                )
                refusals.append(reason)
                return [
                    f"    # AGENT_QUEUE: {reason}",
                    f"    raise NotImplementedError({reason!r})",
                ]
            constructor = (
                f"_make_{type_name}()" if type_name in semantics.types else "_new_derived()"
            )
            return [f"    {name} = {constructor}"]
        return []

    def _local(
        self,
        local: dict[str, Any],
        semantics: Semantics,
        statements: Statements,
        refusals: list[str] | None = None,
    ) -> list[str]:
        if refusals is None:
            refusals = []
        name = pysafe(local["name"])
        dims = local.get("dims")
        if dims:
            if any(d["ub"] is None for d in dims):
                # Allocatable: None until its Allocate_Stmt (allocated() rule).
                return [f"    {name} = None"]
            try:
                shape = ", ".join(self._extent(d, statements) for d in dims)
            except REFUSED as refusal:
                reason = f"local array {local['name']}: extent not resolvable ({refusal})"
                refusals.append(reason)
                return [
                    f"    # AGENT_QUEUE: {reason}",
                    f"    raise NotImplementedError({reason!r})",
                ]
            derived = DERIVED.match(str(local["dtype"]))
            if derived is not None:
                # An array of a derived type: the elements exist the moment
                # the array does, so they are constructed here rather than
                # left as the ``None``s an object ``np.empty`` would hold.
                filled = derived_array(
                    derived.group(1).lower(),
                    [self._extent(d, statements) for d in dims],
                    semantics.types,
                )
                if filled is not None:
                    return [f"    {name} = {filled}"]
            dtype = ALLOCATED_DTYPES.get(local["dtype"], "np.float64")
            return [f"    {name} = {undefined_array(self, f'({shape},)', dtype)}"]
        if local.get("array_spec"):
            return []
        derived = DERIVED.match(str(local["dtype"]))
        if derived is not None:
            type_name = derived.group(1).lower()
            constructor = (
                f"_make_{type_name}()" if type_name in semantics.types else "_new_derived()"
            )
            return [f"    {name} = {constructor}"]
        initializer = local.get("init_expr")
        if initializer and local["dtype"] in ("float64", "float32", "int32", "int64", "bool"):
            # ``real(r8) :: minlwp = -2._r8``: the declaration's value, not
            # the UB-guard zero. (Fortran also SAVEs such a local; a value
            # the body never changes is the same on every call, and one it
            # does change is a deviation this note is the record of.)
            own = frozenset(p["name"].lower() for p in semantics.subprogram["local_parameters"])
            try:
                value = self._parameter_value(str(initializer).strip(), own, statements)
            except REFUSED as refusal:
                value = None
                note = f"  # initializer not translated ({refusal})"
            if value is not None:
                return [f"    {name} = {value}  # declared initializer"]
        else:
            note = ""
        if local["dtype"] in ("float64", "float32"):
            return [f"    {name} = 0.0{note}"]
        if local["dtype"] in ("int32", "int64"):
            return [f"    {name} = 0{note}"]
        if local["dtype"] == "bool":
            return [f"    {name} = False{note}"]
        if local["dtype"] == "str":
            return [f"    {name} = ''"]
        return []

    def _parameter_value(
        self,
        text: str,
        local_parameters: frozenset[str] = frozenset(),
        statements: Statements | None = None,
    ) -> str:
        """A local parameter's initializer, as something Python can evaluate.

        The token pass below classifies; it does not *parse*, and until this
        wrapper existed it had no way to say so. Every branch of it returns a
        string whether or not that string is Python -- an implied-do array
        constructor came out as upper-cased Fortran inside the emitted
        module, a ``SyntaxError`` that takes the whole file down at import
        rather than the one initializer nobody could read. The check belongs
        here, at the pass's boundary, rather than on any one branch: the
        branch that gets it wrong is by definition the one that did not know
        it was wrong.
        """
        spelled = self._token_parameter_value(text, local_parameters)
        if _is_expression(spelled) and not _token_pass_guessed(text, spelled):
            return spelled
        return self._reparsed_parameter_value(text, spelled, statements)

    @staticmethod
    def _token_parameter_value(text: str, local_parameters: frozenset[str]) -> str:
        """A local parameter's initializer, rendered from its source text.

        Text-level, not node-level: declarations were extracted as text, and
        the pipeline this reproduces rendered them the same way.

        ``local_parameters`` are the names of this subprogram's own
        parameters, which decides the casing of an expression that names one
        -- see the token pass at the end.
        """
        if text.lower() in (".true.", ".false."):
            return "True" if "true" in text.lower() else "False"
        boz = BOZ_TEXT.fullmatch(text)
        if boz:
            return str(int(boz.group(1), {"Z": 16, "B": 2, "O": 8}[text[0].upper()]))
        if INTEGER_TEXT.fullmatch(text):
            return str(int(text.replace(" ", "")))
        if REAL_TEXT.fullmatch(text):
            # Both exponent letters: fparser hands this text back with the D
            # upper-cased, and np.float64('0.5D0') is a ValueError at run time
            # rather than anything this file would notice.
            exponent = text.replace(" ", "").split("_")[0]
            return "np.float64('" + exponent.replace("d", "e").replace("D", "e") + "')"
        constructed = ARRAY_CONSTRUCTOR.search(text)
        if constructed:
            items = [strip_kind(item.strip()) for item in constructed.group(1).split(",")]
            if all(re.fullmatch(r"'[^']*'", item) for item in items):
                return f"np.array([{', '.join(items)}])"
            if all(re.fullmatch(r"-?\d+", item.strip()) for item in items):
                return f"np.array([{', '.join(items)}], dtype=np.int32)"
            return f"np.array([{', '.join(items)}])"

        # A module constant is spelled upper case in the emitted source, and a
        # reference to one of this subprogram's own parameters is not -- that
        # one was emitted as a local assignment just above, under its own
        # name. Uppercasing the whole text bound it to a module constant that
        # does not exist: ``1._r8/hplanck`` came out ``1./HPLANCK`` beside the
        # ``hplanck`` it meant.
        def case_of(match: re.Match[str]) -> str:
            token = match.group()
            return pysafe(token.lower()) if token.lower() in local_parameters else token.upper()

        return IDENTIFIER.sub(case_of, strip_kind(text))

    @staticmethod
    def _reparsed_parameter_value(text: str, spelled: str, statements: Statements | None) -> str:
        """What the token pass could not classify, read as an expression.

        Reached only once the token pass has produced something that is not
        valid Python, so there is nothing to lose by parsing: the answer is
        either better than the text or it is a refusal, and the text was
        already unusable. The renderer spells an array constructor as
        ``np.array(...)`` itself, so only a bare comprehension -- an
        implied-do standing alone -- needs wrapping; a Fortran array is not a
        Python list.
        """
        if statements is None:
            raise NoRule(f"initializer needs a parse and none was available: {spelled}")
        from recast.fortran._parse import f03

        try:
            node = f03.Expr(text)
        except Exception as exc:  # fparser raises its own hierarchy
            raise NoRule(f"initializer does not parse: {exc}") from exc
        rendered = statements.expressions.render(node)
        if rendered.startswith("[") and rendered.endswith("]"):
            rendered = f"np.array({rendered})"
        if not _is_expression(rendered):
            raise NoRule(f"initializer did not render as an expression: {rendered}")
        return rendered

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


def _specification(node: Any) -> Any:
    """A subprogram's specification part, or an empty list if it has none."""
    return next((c for c in node.children if isinstance(c, f03.Specification_Part)), [])


def _is_elemental(subprogram: dict[str, Any]) -> bool:
    return any("ELEMENTAL" in str(p).upper() for p in (subprogram.get("prefixes") or []))
