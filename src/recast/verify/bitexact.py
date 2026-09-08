"""``differential.bitexact``: the translation against the compiled truth.

Migrated from the comparison half of the source pipeline
``diff_driver.py`` and the tolerance ladder of its ``tests/test_diff.py``.
The candidate's emitted module and the oracle's compiled Fortran are called
side by side on the same generated inputs, and every output is compared bit
for bit. The ladder has two rungs and both are spelled in the Verdict:

* ``BIT_EXACT`` -- every compared value identical to the last bit. The
  strongest empirical claim there is, and the default acceptance bar.
* ``TOLERANCED`` -- everything agreed within an operator-stated ``rtol``.
  Only awarded when the operator *asked* for a tolerance; loosening the bar
  is a decision someone must make, never a default.

Anything else is ``FAILED``, including every way the comparison could not
run: the candidate does not import, the oracle handle is not a compiled
module, a subprogram raises. Fail closed -- a gate that cannot run did not
pass.

Inputs are generated deterministically per (subprogram, trial): shapes come
from the interface's dimensions resolved against the operator's ``dims``
table, values from per-name ``ranges``. The physical ranges that make a model
kernels behave -- temperatures in kelvin, pressures in pascals -- are domain
knowledge and arrive in config; the engine's defaults are only wide, not
wise. An extent nobody pinned is the harness's own to choose, so a shape the
body will not take -- a packed workspace whose ``lr`` must be ``n(n+1)/2``
for the order it goes with -- is grown until the subscripts fit rather than
left to the operator. Structure in the *values* that no per-name range can
express -- a mode the source stops on, a column that must be monotone --
comes from the project itself: a ``recast_inputs.py`` at the root, whose
``prepare(unit, subprogram, inputs, rng)`` shapes each generated draw before
both sides receive it. Subprograms with deferred blocks are skipped and said
so: their translation raises ``NotImplementedError`` by construction, and the
gate's job is to judge translations, not queues.

Not every output is an argument. A subprogram whose only product is the file
it writes -- ``saveppm(filename, img)``, which declares two inputs and
nothing else -- has nothing for the comparison to pair, and comparing it on
its arguments would be comparing what the caller already knew. So a character
dummy the source hands to an OPEN that *creates* a file is drawn as a scratch
path, one per side, and the bytes each side left there are compared like any
other declared-integer output (``drawable_path``, ``_file_outputs``).
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import keyword
import operator
import re
import shutil
import signal
import sys
import tempfile
import threading
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from recast.errors import InputProfileError
from recast.model import Candidate, Confidence, OracleRef, Unit, Verdict
from recast.oracle.isolated import ReferenceAborted
from recast.plugins.executor import Executor
from recast.plugins.verifier import Verifier
from recast.verify.ulp import ulp_audit

__all__ = ["BitexactVerifier", "factory", "flatten_derived"]

DEFAULT_RANGE = (-1000.0, 1000.0)
DEFAULT_INTEGER_RANGE = (1, 8)


def _f2py_name(name: str) -> str:
    """How f2py spells a dummy on the Python side: lower-cased, and a C
    keyword or a name f2py reserves (``switch``, ``int``, ``len``) with the
    ``_bn`` it appends -- PCHIP's ``dpchic(ic, vc, switch, ...)`` is
    ``w_dpchic(ic, vc, switch_bn, ...)``. f2py's own table where it is
    installed; the keywords alone where it is not."""
    lowered = name.lower()
    try:
        from numpy.f2py import crackfortran
    except Exception:  # numpy without f2py, or no numpy: the reference is not f2py's
        return lowered
    return str(dict(crackfortran.badnames).get(lowered, lowered))


def _bounds_violation(runtime_error: str | None) -> bool:
    """libgfortran's ``-fcheck=bounds`` diagnostics all name the bound: a
    subscript ``below lower bound`` or ``above upper bound``, a substring
    ``out of bounds``, an ``Array bound mismatch``."""
    return "bound" in (runtime_error or "").lower()


def _complex_parts(np: Any, value: Any) -> Any:
    """A complex value as float64 parts: shape ``(..., 2)``, real then imaginary."""
    widened = np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
    return widened.view(np.float64).reshape(*widened.shape, 2)


def _declined_summary(declined_by: dict[str, int]) -> str:
    """``"3 error stop, 1 NaN on both sides"``: the declined draws by kind."""
    return (
        ", ".join(f"{count} {why}" for why, count in sorted(declined_by.items()))
        or "no reason recorded"
    )


def _redrawn_note(totals: dict[str, Any]) -> str:
    """What a passing verdict says about the draws it did not compare on."""
    redrawn = int(totals.get("redrawn") or 0)
    if not redrawn:
        return ""
    reasons = _declined_summary(totals.get("declined") or {})
    return f"; {redrawn} draw(s) declined and drawn again ({reasons})"


DEFAULT_DIMENSION = 8
GROWTH_FACTORS = (2, 4, 8, 16, 32, 64)
"""What an unpinned extent is multiplied by while a shape is being fitted.

Multiples of the default rather than a walk upward: the extent a packed
workspace wants grows with the square of the order it goes with, so a search
that adds one at a time never arrives. Sixty-four times the default of eight
covers an order-eight triangle (36) with room over.
"""
MAX_FITTED_EXTENT = 1024
"""Ceiling on a grown extent, so a subprogram no shape fits costs a bounded
amount of memory rather than the machine's."""
SUPPORTED_DTYPES = frozenset(
    {"float32", "float64", "int32", "int64", "bool", "complex64", "complex128"}
)
COMPLEX_DTYPES = frozenset({"complex64", "complex128"})
"""A complex is compared as its two parts, each a point: bit-exact means
both parts are, and an ULP distance is a part's. Draws give both parts the
argument's range."""
PROCEDURE_DTYPE = "PROCEDURE"


class _NegativeSubscript(IndexError):
    """A translated subscript computed below a dummy's declared lower bound.

    A positive overrun (``IndexError`` from a plain ndarray) is a shape the
    harness's own default got wrong, and growing an unpinned extent can fix
    it (see ``_fit_extents``). A negative one is not a shape at all -- no
    extent, grown or not, changes whether an index is negative -- it is a
    value outside the domain the source itself takes (PCHIP's ``dpchkt``
    forms ``x(n-1)`` and is only ever called with N>=2), so it is drawn
    again exactly like an ``ERROR STOP`` or a NaN-inducing value, and left
    out of the ``reshaped`` accounting a shape refusal earns.
    """


def _reject_negative_subscript(key: Any) -> None:
    """Refuse a negative integer subscript; leave slices and arrays alone.

    A translated subscript is always ``expr - lb``: a body that reads a
    dummy below its declared lower bound -- PCHIP's ``dpchkt`` forms
    ``x(n-1)`` and is only ever called with N>=2, so ``x(0)`` is a draw
    outside the source's own domain, not a shape this harness chose --
    computes a negative Python index. Plain ndarray wraps that to the
    *other* end of the array instead of refusing it the way a positive
    overrun already does (an ``IndexError`` the redraw loop below already
    knows how to answer without ever calling the reference on it), so the
    candidate would silently read the wrong element instead of raising.
    """
    indices = key if isinstance(key, tuple) else (key,)
    for index in indices:
        try:
            value = operator.index(index)
        except TypeError:
            continue  # a slice, a mask, a fancy index -- not a bare subscript
        if value < 0:
            raise _NegativeSubscript(
                f"index {value} is out of bounds for a Fortran dummy "
                "(subscript below its declared lower bound)"
            )


_NO_WRAP_ARRAY_TYPES: dict[int, type] = {}
"""One ``_NoWrapArray`` class per ``np`` module handed in, built lazily.

``numpy`` is imported lazily throughout this file, so nothing here can
subclass ``np.ndarray`` at module scope; a subclass is built once per
``np`` (keyed by ``id()``, since a project's own ``np`` and any test
double share nothing else stable) and reused after that.
"""


def _no_wrap_array_type(np: Any) -> type:
    """The ``_NoWrapArray`` class for this ``np`` module, built on first use.

    Only single-index (or all-integer tuple) access is guarded: a slice,
    a boolean mask, or a fancy index is the harness's or the translation's
    own choice of view, not a subscript the source computed, and is left
    to ndarray's ordinary rules.
    """
    cached = _NO_WRAP_ARRAY_TYPES.get(id(np))
    if cached is not None:
        return cached

    class _NoWrapArray(np.ndarray):  # type: ignore[misc]  # ``np`` is a parameter, not the typed module
        def __getitem__(self, key: Any) -> Any:
            _reject_negative_subscript(key)
            return super().__getitem__(key)

        def __setitem__(self, key: Any, value: Any) -> None:
            _reject_negative_subscript(key)
            super().__setitem__(key, value)

    _NO_WRAP_ARRAY_TYPES[id(np)] = _NoWrapArray
    return _NoWrapArray


INPUT_PROFILE = "recast_inputs.py"
"""The project's input profile, looked for at the root the run was given.

Its ``prepare(unit, subprogram, inputs, rng)`` receives the inputs this
harness drew for one trial, by argument name, and returns them shaped into
the source's domain -- or ``None`` to leave that subprogram's draw as it is.
A shaped draw is an assertion that the reference takes it, and is judged as
one: it is never redrawn, a candidate that refuses it has failed, and a
reference that refuses it means the profile is wrong.
"""
"""What a dummy *procedure* argument is declared as.

Not a value the harness can sample: it is something to call, and what both
sides need is the same callable. See :func:`callback_for`.
"""


def drawable_path(argument: dict[str, Any]) -> bool:
    """Whether this dummy is a scratch path the harness may draw.

    A character dummy has no sampling story in general -- an init routine's
    ``errstring`` is a message, and a default that drew one would fail the
    whole gate on it. One shape does: a dummy the source hands to an OPEN
    that *creates* the file (``path: "created"``, from the frontend's
    ``opened_files``). Any name works there, because the subprogram makes the
    file rather than finding it, so the harness can give each side a scratch
    path of its own and compare the two files afterwards. A path the source
    opens ``STATUS='OLD'`` is the opposite case: the draw would have to be a
    file that already holds something, which is not a value anything here can
    produce, and the oracle leaves that subprogram ungated.
    """
    return argument.get("dtype") == "str" and argument.get("path") == "created"


def _file_bytes(path: Any) -> bytes | None:
    """What a side left at a path argument, or ``None`` if it left nothing."""
    try:
        return Path(str(path)).read_bytes()
    except OSError:
        return None


class _CallTimedOut(Exception):
    """The candidate did not return from a draw within its bound.

    Not an error in the translation: a draw can put a subprogram in a loop
    the source itself never leaves -- ``bisect``'s ``do while (b - a > tol)``
    with a negative tolerance halves the interval to zero and keeps going --
    and the reference, being the same algorithm, would not leave it either.
    So it is one more way a draw is refused, beside the ERROR STOP and the
    out-of-bounds subscript below it, and it is answered the same way: draw
    again.
    """


@contextlib.contextmanager
def _bounded(seconds: float) -> Iterator[None]:
    """Run the block under a wall-clock bound, or unbounded where none can be.

    An interval timer, because the thing to bound is a call inside this
    process and the point is to *get back*: a subprocess would need the
    candidate module and the same call-back object, and a thread cannot be
    stopped. The bound therefore lands where Python next checks for signals,
    which is between bytecodes -- a translated loop, which is what runs long
    here. It is not a way to interrupt a long call inside a C extension, and
    it does not claim to be one.

    Unavailable off the main thread and on platforms without an interval
    timer; there the block runs as it always did rather than not at all.
    """
    if (
        seconds <= 0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def expire(_signum: int, _frame: Any) -> None:
        raise _CallTimedOut(f"did not return within {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _callback_split(interface: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A call-back interface's arguments, split the way a call to it is made.

    The convention is f2py's, and the emitted translation's: the arguments the
    Fortran side supplies are passed in, in declaration order, and the ones it
    reads back come out of the return, in declaration order. An ``intent(inout)``
    argument is in both.

    A function call-back answers through its result, which both sides read off
    the return the same way, so the result stands in the outputs as the one
    thing the call produces.
    """
    inputs = [a for a in interface["args"] if a["intent"] in ("IN", "INOUT")]
    outputs = [a for a in interface["args"] if a["intent"] in ("OUT", "INOUT")]
    if interface["kind"] == "function":
        outputs = [
            {
                "name": interface.get("result") or "result",
                "dtype": interface.get("result_dtype"),
                "intent": "OUT",
            }
        ]
    return inputs, outputs


def callback_for(np: Any, name: str, interface: dict[str, Any]) -> Any:
    """One deterministic stand-in for a procedure argument.

    A subprogram that takes a procedure -- a residual, a Jacobian, a
    user-supplied right-hand side -- cannot be compared without one, and there
    is no such thing as a *sampled* function. So the harness supplies a fixed
    one, and hands the *same Python object* to both sides: the reference calls
    it back through f2py, the candidate calls it directly, and any difference
    between them is therefore a difference in the code under test rather than
    in what was called.

    What it computes is deliberately dull -- a bounded, smooth function of the
    values it was given, with a per-position offset so the outputs are not all
    equal. It is not a physics problem and does not claim to be one; the claim
    a differential makes is that both sides did the same arithmetic, and any
    total function of the inputs supports it.
    """
    inputs, outputs = _callback_split(interface)
    unsupported = [
        f"{a['name']!r}={a['dtype']!r}"
        for a in interface["args"]
        if a["dtype"] not in SUPPORTED_DTYPES
    ]
    if unsupported:
        raise ValueError(
            f"call-back {name!r} takes argument(s) {', '.join(unsupported)}, which this "
            "harness cannot supply"
        )
    if interface["kind"] == "function":
        # The reference reads the result off the return and so does the
        # translation, which is the whole convention -- but only when the
        # result is the one thing the call produces. A function that also
        # writes an argument hands two things back in an order this harness
        # would be inventing, and the reference wrapper refuses it too.
        if interface.get("result_dtype") not in SUPPORTED_DTYPES:
            raise ValueError(
                f"call-back {name!r} returns {interface.get('result_dtype')!r}, which this "
                "harness cannot supply"
            )
        written = [a["name"] for a in interface["args"] if a["intent"] != "IN"]
        if written:
            raise ValueError(
                f"call-back {name!r} is a function that writes argument(s) "
                f"{', '.join(written)}; this harness supplies functions that only read theirs"
            )

    def shape_of(argument: dict[str, Any], bound: dict[str, Any]) -> tuple[int, ...]:
        axes = []
        for dim in argument.get("dims") or []:
            axis = str(dim.get("ub") or "").strip().lower()
            if axis not in bound:
                raise ValueError(
                    f"call-back {name!r} output {argument['name']!r} has extent {axis!r}, "
                    "which the call does not supply"
                )
            axes.append(int(np.asarray(bound[axis]).item()))
        return tuple(axes)

    def call(*values: Any) -> Any:
        if len(values) != len(inputs):
            raise ValueError(
                f"call-back {name!r} takes {len(inputs)} argument(s), called with {len(values)}"
            )
        bound = {a["name"].lower(): v for a, v in zip(inputs, values, strict=True)}
        # Only the real ``intent(in)`` arguments feed the value. An integer
        # flag is a mode rather than data; and an ``intent(inout)`` argument
        # is where an answer goes, so on the first call it holds whatever the
        # caller's uninitialized buffer held -- reading it would make the
        # call-back's answer depend on memory neither side defines, which is
        # a difference between the two sides that is not a difference in the
        # code under test.
        supplied = [
            np.ravel(np.asarray(bound[a["name"].lower()], dtype=np.float64))
            for a in inputs
            if a["dtype"] in ("float32", "float64") and a["intent"] == "IN"
        ]
        pool = np.concatenate(supplied) if supplied else np.zeros(1)
        if pool.size == 0:
            pool = np.zeros(1)
        produced = []
        for argument in outputs:
            if argument["dtype"] in ("int32", "int64", "bool"):
                # An integer or logical the caller reads back is a control
                # flag -- ``iflag`` says "keep going" -- and inventing a value
                # for it would steer the algorithm rather than answer it.
                produced.append(bound.get(argument["name"].lower(), 0))
                continue
            shape = shape_of(argument, bound)
            count = int(np.prod(shape)) if shape else 1
            picks = pool[np.arange(count) % pool.size]
            offsets = (np.arange(count) % 5) * 0.25
            values_out = picks + 0.5 * picks / (1.0 + picks * picks) - offsets
            produced.append(np.float64(values_out[0]) if not shape else values_out.reshape(shape))
        return produced[0] if len(produced) == 1 else tuple(produced)

    # f2py reads the *arity* of the Python object it was handed -- how many
    # of the call-back's arguments to pass is ``__code__.co_argcount`` -- and
    # a ``*values`` function reports none, so the reference called it with
    # nothing. The arity is part of the calling convention, so it is spelled.
    parameters = ", ".join(f"_cb{index}" for index in range(len(inputs)))
    namespace: dict[str, Any] = {"_call": call}
    exec(  # noqa: S102 - the text is this function's own, over generated names
        f"def _callback({parameters}):\n    return _call({parameters})\n", namespace
    )
    callback = namespace["_callback"]
    callback.__name__ = f"callback_{name}"
    return callback


def _returned(translated_out: Any) -> list[Any]:
    """The values a candidate call handed back, as a list: a tuple's items,
    one bare value, or none at all -- a subroutine with no OUT argument
    returns ``None`` (CLUBB's finalize_tau_sponge_damp_api deallocates and
    returns), and that is zero values, not one."""
    if translated_out is None:
        return []
    return list(translated_out) if isinstance(translated_out, tuple) else [translated_out]


def _extent(dim: dict[str, Any], dims: dict[str, int]) -> int:
    """An axis's extent: ``ub - lb + 1`` when a lower bound is declared
    (CLUBB's ``lhs(-2:2, ...)`` has five rows, not two), ``ub`` otherwise."""
    upper = _resolve_extent(dim.get("ub"), dims)
    lower = str(dim.get("lb") or "1").strip()
    if lower == "1" or dim.get("ub") is None:
        return upper
    return upper - _resolve_extent(lower, dims) + 1


def _resolve_extent(text: str | None, dims: dict[str, int]) -> int:
    """A declared dimension's extent under the operator's table.

    A name the table does not pin is the default dimension -- which is also
    what the harness draws the scalar of that name as (``_generated_inputs``)
    -- so ``g(n + 1)`` over an unpinned ``n`` is nine cells beside an ``n`` of
    eight, not eight beside eight. Left as a name, the arithmetic failed and
    the whole bound fell back to the default, and the reference refused the
    array ("0-th dimension must be fixed to 9 but got 8").
    """
    default = int(dims.get("default_dim", DEFAULT_DIMENSION))
    if text is None:
        return default
    spelled = str(text).strip().lower()
    if spelled.isdigit():
        return int(spelled)
    resolved = spelled
    for name, value in dims.items():
        resolved = re.sub(rf"\b{re.escape(name.lower())}\b", str(value), resolved)
    resolved = re.sub(r"\b[a-z_]\w*\b", str(default), resolved)
    try:
        return int(_arithmetic(resolved))
    except Exception:
        return default


_SIZE_TERM = re.compile(r"size\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.I)
"""A ``size(name, axis)`` term of a shape guard's extent (axis from zero)."""


def _guarded_shapes(
    required: list[dict[str, Any]],
    guards: Sequence[dict[str, Any]],
    dims: dict[str, int],
) -> dict[str, list[int]]:
    """The shape to draw each array argument at, under the body's own checks.

    Every extent starts where it always did -- the declared bound under the
    operator's table, ``default_dim`` for an assumed-shape one -- and a guard
    then says what one of them has to be. ``size(c,1) /= 5`` makes ``c``'s
    first extent five; ``size(c,2) /= size(xi)-1`` makes its second one less
    than the length of ``xi``, which is a *relation*, so the guards are
    applied until they stop changing anything rather than in one pass.

    A guard whose extent does not resolve, or resolves to nothing an array
    can have, is left alone: the draw it would make is worse than the default
    it replaces, and the subprogram refusing it says so where a shape nobody
    can name would not.
    """
    shapes = {
        str(argument["name"]).lower(): [
            _resolve_extent(dim.get("ub"), dims) for dim in argument["dims"]
        ]
        for argument in required
        if argument.get("dims")
    }
    for _ in range(len(guards) + 1):
        settled = True
        for guard in guards:
            extent = shapes.get(str(guard.get("arg", "")).lower())
            axis = int(guard.get("axis", 0))
            if extent is None or not 0 <= axis < len(extent):
                continue
            spelled = _SIZE_TERM.sub(
                lambda m: str(
                    (shapes.get(m.group(1).lower()) or [0])[int(m.group(2))]
                    if int(m.group(2)) < len(shapes.get(m.group(1).lower()) or [])
                    else 0
                ),
                str(guard.get("extent", "")),
            )
            try:
                wanted = int(_arithmetic(spelled))
            except Exception:  # noqa: S112 - an extent this cannot resolve keeps its default
                continue
            if wanted < 1 or wanted > MAX_FITTED_EXTENT or wanted == extent[axis]:
                continue
            extent[axis] = wanted
            settled = False
        if settled:
            break
    return shapes


# Split by arity rather than kept in one table. A single dict of both is a
# dict whose value type is the join of a two-argument and a one-argument
# callable, which is a type nothing can call -- the checker says so, and the
# result was an ``Any`` leaking out of a function declared to return ``float``.
_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _arithmetic(text: str) -> float:
    """Integer arithmetic over the operators a Fortran extent can use.

    This was ``eval`` with empty builtins, on text that came from a declared
    dimension in the source under verification -- and the source under
    verification is the input this engine exists to take from other people.
    An empty ``__builtins__`` does not make ``eval`` safe, and a dimension
    expression has no business reaching anything but arithmetic. Anything
    that is not a number or one of the operators above is a ``ValueError``,
    which the caller turns into the default extent exactly as before.
    """

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError(f"not arithmetic: {ast.dump(node)}")

    return walk(ast.parse(text, mode="eval"))


def _delegation_chain(name: str, delegated: dict[str, str]) -> str:
    """The backend's reasons, followed to the root: a wrapper delegated because
    it ``calls non-emitted subprogram f`` says nothing until f's own reason
    is beside it."""
    import re

    parts: list[str] = []
    seen: set[str] = set()
    while name in delegated and name not in seen:
        seen.add(name)
        why = str(delegated[name])
        parts.append(f"{name}: {why}" if parts else why)
        match = re.search(r"calls non-emitted subprogram (\w+)", why)
        if not match:
            break
        name = match.group(1)
    return f" ({' <- '.join(parts)})" if parts else ""


def flatten_derived(
    sub: dict[str, Any],
    translated_fn: Any,
    plan: dict[str, dict[str, Any]],
    translated: Any = None,
) -> tuple[dict[str, Any], Any]:
    """The candidate's signature and function with derived-type dummies split
    into the flat scalars the reference wrapper takes.

    f2py cannot marshal a derived type, so the oracle spells a dummy of a
    type made of scalar components as one dummy per component
    (``recast.oracle.f2py.derived_components``) and puts the plan on its
    handle: ``{argument: {"type": name, "components": [{"name", "component",
    "dtype"}, ...]}}``. This is the same split on the candidate's side. The
    returned signature carries the components in the argument's place, with
    its intent, so the harness draws, passes and pairs them like any other
    scalar; the returned function assembles the object from those draws --
    through the emitted ``_make_<type>()`` factory where the module has one
    -- calls the translation, and hands back the object's components where
    the translation handed back the object. Nothing about the comparison
    changes: every component is a point, an integer one compared exactly.
    """
    from types import SimpleNamespace

    from recast.transform.numpy.vocabulary import pysafe

    flat_args: list[dict[str, Any]] = []
    split: list[tuple[str, str, list[dict[str, Any]]]] = []
    for argument in sub["args"]:
        entry = plan.get(argument["name"])
        if entry is None or argument.get("optional"):
            flat_args.append(argument)
            continue
        components = list(entry.get("components") or [])
        for component in components:
            flat_args.append(
                {
                    "name": component["name"],
                    "dtype": component["dtype"],
                    "intent": argument["intent"],
                    "optional": False,
                }
            )
        split.append((argument["name"], str(entry.get("type", "")), components))
    if not split:
        return sub, translated_fn
    flat_sub = {**sub, "args": flat_args}
    outs = [a for a in sub["args"] if a["intent"] in ("OUT", "INOUT")]
    by_name = {name: (type_name, components) for name, type_name, components in split}

    def assemble(type_name: str) -> Any:
        factory = getattr(translated, f"_make_{type_name}", None) if translated else None
        return factory() if callable(factory) else SimpleNamespace()

    def flat_fn(**kwargs: Any) -> Any:
        objects: dict[str, Any] = {}
        for name, type_name, components in split:
            obj = assemble(type_name)
            for component in components:
                key = pysafe(component["name"])
                if key in kwargs:
                    setattr(obj, pysafe(component["component"]), kwargs.pop(key))
            objects[name] = obj
            kwargs[pysafe(name)] = obj
        result = translated_fn(**kwargs)
        if sub["kind"] == "function":
            return result
        values = (
            list(result) if isinstance(result, tuple) else ([result] if result is not None else [])
        )
        if len(values) != len(outs):
            return result  # the harness reports the count mismatch itself
        expanded: list[Any] = []
        for argument, value in zip(outs, values, strict=True):
            if argument["name"] in by_name:
                _type_name, components = by_name[argument["name"]]
                expanded.extend(
                    getattr(value, pysafe(component["component"])) for component in components
                )
            else:
                expanded.append(value)
        return tuple(expanded)

    return flat_sub, flat_fn


class BitexactVerifier(Verifier):
    """Call both sides on the same inputs; count the bits that disagree."""

    name = "differential.bitexact"
    provides = Confidence.BIT_EXACT

    draws_per_trial: int = 24
    """How many draws one trial may take before the harness gives up on it.

    A generated draw is not always one the subprogram accepts: an argument
    outside the domain the source itself declares (``error stop 'invalid
    mode'``), an extent too small for the subscripts the body forms, a value
    that drives the arithmetic into NaN, a value the source's own loop never
    leaves (``call_seconds``). None of those is a difference between the two
    sides -- the reference cannot even be *called* on the first two without
    ending the process or reading memory it does not own, and would not come
    back from the last one either -- so the trial is drawn again, with a
    fresh seed and fresh unpinned extents, rather than reported as a
    comparison that failed.

    Bounded, and the bound is the point: a subprogram whose every draw is
    refused is reported as one that could not be compared, which is what
    ``uncovered`` and the coverage gate above are for. And a subprogram whose
    declined draws outnumber the trials it was compared on fails by name:
    the trials that survived are a minority of what the configured draw
    produces, and a translation that stops or overruns on inputs the source
    accepts would pass on the handful it did not, one survivor at a time.
    The number of redraws, and the reason for each, is recorded per
    subprogram and repeated on the verdict, so the count is never silent.

    Two things a redraw is not for. A NaN on one side only is a mismatch
    between the sides, not a domain draw, and is counted as one; only a trial
    where both sides produce NaN at the same points is drawn again. And a
    trial compared only after a shape refusal moved the free extents is
    evidence at the extents it landed on, not the configured ones: a
    subprogram that passes mostly that way fails by name, with the number of
    such trials recorded as ``reshaped``.

    A shape refusal is answered before that, and not by a redraw: the extents
    nobody pinned are this harness's own choice, so the first one refused is
    *grown* until the body's subscripts fit, and the subprogram is compared
    again from its first trial at that one shape, which the outcome records
    as ``extents`` (:meth:`_fit_extents`). The redraw and the ``reshaped``
    floor are what remains for a body no growth fits.

    None of this applies to a draw the project's ``recast_inputs.py`` shaped.
    That draw is the project saying the source takes it, so there is nothing
    to draw again: the reference is called first, and if it refuses, the
    profile is wrong and the verification stops there
    (``InputProfileError``); if it answers and the candidate refuses, the
    candidate has failed on inputs the source takes.
    """

    call_seconds: float = 5.0
    """How long one generated draw may keep the candidate before it is refused.

    Generous by three orders of magnitude: the gate's draws are small -- a
    default extent of eight -- and a subprogram that has not answered one in
    five seconds is not answering it. What the bound buys is that a draw the
    source does not terminate on is a redraw rather than a run that never
    ends, and the reference is never called on it. ``0`` turns it off.
    """

    dominant_at: float | None = None
    """Fraction of a row's maximum above which an element is *dominant*.

    ``None`` here: bit-exactness has no use for the distinction, since every
    element has to agree. A gate that tolerates ULP drift does have one --
    see ``recast.verify.tolerance`` -- and sets it, which is what turns the
    per-element mask on in the comparison below.
    """

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        # Where a subprogram that writes a file writes it. Outside the run's
        # workspace on purpose: the reference takes a path through a
        # ``character(len=128)`` wrapper dummy, and a workspace path is
        # already most of that budget before a file name is added to it.
        scratch = Path(tempfile.mkdtemp(prefix="recast-gate-io-"))
        try:
            verdict = self._compare_all(
                unit, candidate, oracle, workspace, executor, config, scratch
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        # An oracle that could not spell every subprogram lists the rest on
        # its handle; a module that passes with three of its eleven
        # subprograms compared must say so where the evidence is read. This
        # neither weakens the verdict for what was compared nor strengthens
        # it for what was not.
        handle = oracle.handle if isinstance(oracle.handle, dict) else {}
        # ... and so must an operator who declared one ungated in config; the
        # comparison put those it found in the table on the metrics.
        ungated = {
            **dict(handle.get("ungated") or {}),
            **dict(verdict.metrics.get("ungated") or {}),
        }
        # A procedure the build could not link and recast defined instead, on
        # both sides (``recast.references``). Every point a subprogram
        # reaching one was compared at was computed with that stand-in, not
        # with the library the source names, and a reader of this verdict has
        # no other way to know it.
        substituted = dict(handle.get("substituted") or {})
        if not ungated and not substituted:
            return verdict
        detail = verdict.detail
        if substituted:
            detail += (
                f"; {len(substituted)} external(s) stood in for by recast's own "
                "reference implementation, on both sides: " + ", ".join(sorted(substituted))
            )
        if ungated:
            detail += f"; {len(ungated)} subprogram(s) ungated, no reference: " + ", ".join(
                f"{name} ({why})" for name, why in sorted(ungated.items())
            )
        return Verdict(
            unit=verdict.unit,
            candidate=verdict.candidate,
            verifier=verdict.verifier,
            confidence=verdict.confidence,
            metrics={
                **verdict.metrics,
                **({"ungated": ungated} if ungated else {}),
                **({"substituted": substituted} if substituted else {}),
            },
            detail=detail,
        )

    def _compare_all(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
        scratch: Path | None = None,
    ) -> Verdict:
        try:
            import numpy as np
        except ImportError:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                "numpy is not installed; install recast-engine[translate]",
            )

        handle = oracle.handle if isinstance(oracle.handle, dict) else {}
        truth = handle.get("module")
        wrappers = handle.get("wrappers", {})
        # Which direction this comparison runs. Every reference that *computes*
        # answers inputs the harness chose, so the harness generates them. A
        # reference that only *replays* cannot be asked anything it was not
        # already asked -- the inputs are whatever the recorded run used -- so
        # it supplies them, and says so here rather than being detected.
        recorded = handle.get("input_source") == "recorded"
        samples = list(handle.get("samples") or []) if recorded else []
        if recorded and not samples:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                f"oracle {oracle.oracle!r} supplies the inputs and handed over no samples",
            )
        if truth is None and not recorded:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                f"oracle {oracle.oracle!r} handed no compiled module to compare against",
            )

        try:
            translated = self._load_candidate(
                candidate,
                workspace,
                config.get("module_suffix", "_numpy.py"),
                companions=config.get("companion_paths") or (),
            )
        except (Exception, SystemExit) as error:  # fail closed, whatever broke
            return self._verdict(
                candidate, Confidence.FAILED, {}, f"candidate does not import: {error}"
            )

        profile = None if recorded else self._load_input_profile(config)

        table = getattr(translated, "_SIGNATURES", None)
        if not isinstance(table, dict) or not table:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                "the emitted module carries no _SIGNATURES table to generate "
                "inputs from; the transform embeds one for exactly this harness",
            )

        deferred_subprograms = {entry.split("/", 1)[0] for entry in candidate.deferred}
        if recorded:
            # A recording is text and parses as float64 throughout; the
            # signature says which of its inputs are integers.
            self._type_recorded(np, samples, table)

        def judged(name: str) -> bool:
            """Not deferred -- and not a flat adapter around a subprogram
            that is, since the adapter would call into a NotImplementedError
            and the skip has to map the adapter's name back to its own."""
            if name in deferred_subprograms:
                return False
            return not (name.endswith("_flat") and name[: -len("_flat")] in deferred_subprograms)

        def not_generable(name: str) -> str | None:
            """Why this harness cannot produce every required input, or None.

            Character arguments have no sampling story yet, beyond the one
            shape ``drawable_path`` names; a default that tried would fail the
            whole gate on an init routine's errstring. Explicit config still
            wins -- and then fails loudly. The reason goes on the verdict by
            name (numfor's ``print_msg``, a message to stderr): not compared,
            and not silent about it.
            """
            for a in table[name]["args"]:
                if a["intent"] == "OUT" or a.get("optional"):
                    continue
                if a["dtype"] == "str" and not drawable_path(a):
                    return f"character argument {a['name']}: no generated draw for one"
                if a["dtype"] == PROCEDURE_DTYPE and not isinstance(a.get("interface"), dict):
                    # A procedure argument the frontend could not resolve an
                    # interface for: nothing here can say what calling it
                    # means, so nothing here can supply one.
                    return f"procedure argument {a['name']} of unresolved interface"
            return None

        def generable(name: str) -> bool:
            return not_generable(name) is None

        # One the operator declared ungated is not compared: the declaration
        # says the reference cannot be held -- on generated inputs (CLUBB's
        # rcm_sat_adj iterates and error-stops on them) or on a recording
        # (its sponge initializer leaves the levels below the layer
        # undefined on both sides). The reason is reported beside the verdict.
        declared_ungated = set(config.get("ungated") or {})
        harness_ungated: dict[str, str] = {}
        if recorded:
            # A recording names what it is a recording of, so the set to
            # compare is the set that was captured -- not every subprogram the
            # module exports. ``generable`` does not apply: nothing is
            # generated, and a character argument that was recorded can be
            # replayed.
            by_subprogram: dict[str, list[dict[str, Any]]] = {}
            for sample in samples:
                by_subprogram.setdefault(str(sample.get("subprogram", "")), []).append(sample)
            offered = sorted(by_subprogram)
            wanted = config.get("subprograms") or [
                name
                for name in offered
                if name in table and judged(name) and name not in declared_ungated
            ]
            skipped = sorted(set(offered) - set(wanted))
        else:
            by_subprogram = {}
            # A subprogram the oracle listed as ungated has no reference to
            # compare against -- it says so, and says why, and the reason
            # lands on the verdict below. Comparing one anyway compares the
            # candidate against a wrapper the oracle has already disclaimed.
            disclaimed = set(handle.get("ungated") or {}) | set(config.get("ungated") or {})
            wanted = config.get("subprograms") or [
                name
                for name in wrappers
                if name in table and judged(name) and generable(name) and name not in disclaimed
            ]
            skipped = sorted(set(wrappers) - set(wanted))
            # A translated subprogram the harness has no draw for is named
            # with its reason, beside the ones the oracle and the operator
            # declared: what the verdict does not cover, said aloud.
            for name in table:
                why = not_generable(name)
                if why is not None and name not in declared_ungated and judged(name):
                    harness_ungated[name] = why

        trials = int(config.get("trials", 10))
        # The transform may have read the tree for the value of every name
        # that sizes a dummy array (``Candidate.notes["dims"]``); the
        # operator's table wins where both speak.
        dims = {**(candidate.notes.get("dims") or {}), **dict(config.get("dims", {}))}
        ranges = {str(k).lower(): tuple(v) for k, v in (config.get("ranges") or {}).items()}

        # Module state first: the emitted header says "call <init> before
        # use", and the Fortran side's SAVE variables need the same call with
        # the same constants, or the two sides are computing under different
        # physics and every mismatch is noise.
        try:
            self._run_setup(
                config.get("setup") or [],
                translated,
                truth,
                wrappers,
                str(handle.get("arg_naming", "lower")),
            )
        except (Exception, SystemExit) as error:  # fail closed
            return self._verdict(candidate, Confidence.FAILED, {}, f"setup call failed: {error}")

        per_subprogram: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        worst_rel = 0.0
        totals: dict[str, Any] = {
            "points": 0,
            "bit_exact": 0,
            "max_ulp": 0,
            "nan_mismatch": 0,
            "integer_points": 0,
            "integer_mismatch": 0,
            "redrawn": 0,
            "declined": {},
        }
        # A backend that says which names it lowered (the JAX module's
        # ``_JAX_KERNELS``) is judged on those alone. A name it forwarded to
        # its host module (``f = _host.f``) is the anchor's code, already
        # judged at the anchor's tier; comparing it here awarded the JAX
        # port a bit-exact verdict on a kernel it never emitted (ELM's
        # hydraulic-stress routine). It fails by name, with the backend's
        # reason, unless declared ungated like any other silence.
        lowered = getattr(translated, "_JAX_KERNELS", None)
        delegated = (candidate.notes.get("jax") or {}).get("delegated") or {}
        declared_flat = handle.get("flattened")
        flattened: dict[str, Any] = declared_flat if isinstance(declared_flat, dict) else {}
        for name in wanted:
            sub = table[name]
            if isinstance(lowered, (list, tuple, set)) and name not in lowered:
                failures.append(
                    f"{name}: not lowered by this backend, forwarded to its host module"
                    + _delegation_chain(name, delegated)
                )
                continue
            translated_fn = self._candidate_function(translated, name)
            truth_fn = None if recorded else getattr(truth, wrappers.get(name, f"w_{name}"), None)
            if translated_fn is None or (truth_fn is None and not recorded):
                side = "candidate" if translated_fn is None else "oracle"
                failures.append(f"{name}: missing on the {side} side")
                continue
            if isinstance(flattened.get(name), dict):
                # The reference takes this subprogram's derived-type dummies
                # component by component; so, for this comparison, does the
                # candidate.
                sub, translated_fn = flatten_derived(
                    sub, translated_fn, flattened[name], translated
                )
            outcome = self._compare_subprogram(
                np,
                name,
                sub,
                translated_fn,
                truth_fn,
                trials,
                dims,
                ranges,
                profile=profile,
                unit_uid=unit.uid,
                dominant_at=config.get("dominant_at", self.dominant_at),
                dominant_axis=config.get("dominant_axis", -1),
                rel_scale=str(config.get("rel_scale", "element")),
                draws=int(config.get("draws", self.draws_per_trial)),
                call_seconds=float(config.get("call_seconds", self.call_seconds)),
                arg_naming=str(handle.get("arg_naming", "lower")),
                convention=str(handle.get("return_convention", "f2py")),
                samples=by_subprogram.get(name) if recorded else None,
                scratch=None if scratch is None else scratch / name,
                reference_isolated=handle.get("isolation") == "process",
            )
            per_subprogram[name] = outcome
            if "error" in outcome:
                failures.append(f"{name}: {outcome['error']}")
                continue
            totals["points"] += outcome["points"]
            totals["bit_exact"] += outcome["bit_exact"]
            totals["max_ulp"] = max(totals["max_ulp"], outcome["max_ulp"])
            totals["nan_mismatch"] += outcome["nan_mismatch"]
            totals["integer_points"] += outcome["integer_points"]
            totals["integer_mismatch"] += outcome["integer_mismatch"]
            totals["redrawn"] += outcome.get("redrawn", 0)
            for why, count in (outcome.get("declined") or {}).items():
                totals["declined"][why] = totals["declined"].get(why, 0) + count
            worst_rel = max(worst_rel, outcome["max_rel"])
            if "max_ulp_dominant" in outcome:
                totals["max_ulp_dominant"] = max(
                    totals.get("max_ulp_dominant", 0), outcome["max_ulp_dominant"]
                )
                totals["dominant_points"] = (
                    totals.get("dominant_points", 0) + outcome["dominant_points"]
                )

        # Policy gate: a public subprogram the module declares (its
        # _SIGNATURES), that is not deferred, and that no comparison attempt
        # covered, is a translation claim with no evidence. Every silent-
        # narrowing filter -- oracle-side wrapper drops, generability skips,
        # config subsets -- lands here by construction, because coverage is
        # judged against what was TRANSLATED, not against whatever survived
        # the filters. Three things are not silence: a private subprogram,
        # which no wrapper can reach and every public caller exercises; a
        # subprogram compared through its ``<name>_flat`` adapter, which calls
        # it on both sides; and one the oracle listed as ungated, whose
        # reason ``verify`` carries onto the verdict; and one the operator
        # declared ungated in config, ``{name: why}``, for a reference this
        # oracle cannot hold -- a routine a recording never reached because it
        # is compared inside another unit's replay, say. The reason goes on
        # the verdict with the oracle's, and a name not in this unit's table
        # is not this unit's to report.
        declared = {
            **harness_ungated,
            **{
                name: str(why)
                for name, why in (config.get("ungated") or {}).items()
                if name in table
            },
        }
        compared = set(wanted)
        ungated = set(handle.get("ungated") or {}) | set(declared)
        uncovered = sorted(
            name
            for name, sub in table.items()
            if sub.get("public", True)
            and judged(name)
            and name not in compared
            and f"{name}_flat" not in compared
            and name not in ungated
        )
        metrics = {
            "subprograms": per_subprogram,
            "trials": trials,
            "skipped": skipped,
            "uncovered": uncovered,
            "input_profile": INPUT_PROFILE if profile is not None else None,
            # Where the reference ran: its own process (an ``error stop``
            # there declines a draw) or this one (a call-back argument keeps
            # it here, and an ``error stop`` would end the run).
            "reference_isolation": handle.get("isolation"),
            "shaped": sorted(name for name, out in per_subprogram.items() if out.get("shaped")),
            "max_rel": worst_rel,
            **({"ungated": declared} if declared else {}),
            **totals,
            **self._devices(translated, handle),
        }
        if uncovered:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(uncovered)} translated subprogram(s) were never compared: "
                + ", ".join(uncovered[:5])
                + " -- defer them or drop them from the unit; silence is not a pass",
            )
        if failures:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(failures)} subprogram(s) could not be compared: " + "; ".join(failures[:3]),
            )
        if not per_subprogram:
            return self._verdict(
                candidate, Confidence.FAILED, metrics, "nothing was compared; that is not a pass"
            )
        if totals["points"] == 0:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                "zero numerical points were compared; that is not a pass",
            )
        if totals["integer_mismatch"]:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{totals['integer_mismatch']}/{totals['integer_points']} integer point(s) "
                "differ exactly; integer mismatches cannot be tolerance-excused",
            )
        if totals["nan_mismatch"]:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{totals['nan_mismatch']} point(s) where one side produced NaN "
                "and the other a number",
            )
        # What a gate needs to come back to the comparison: the loaded
        # candidate, the table, the recorded samples, the staged files.
        context = {
            "np": np,
            "translated": translated,
            "table": table,
            "recorded": recorded,
            "by_subprogram": by_subprogram,
            "workspace": workspace,
            "handle": handle,
            "trials": trials,
            "dims": dims,
            "ranges": ranges,
        }
        return self._award(candidate, totals, per_subprogram, metrics, config, context=context)

    def _award(
        self,
        candidate: Candidate,
        totals: dict[str, Any],
        per_subprogram: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Verdict:
        """Which confidence the numbers earn.

        Everything above this point -- generating inputs, calling both sides,
        counting ULP -- is the same comparison whatever the gate. What differs
        between gates is only the policy, so that is the part a subclass
        overrides. Splitting it here is what lets a second differential gate
        exist without a second harness to keep in step with this one.
        """
        rtol = config.get("rtol")
        worst_rel = metrics["max_rel"]
        if totals["bit_exact"] == totals["points"]:
            return self._verdict(
                candidate,
                Confidence.BIT_EXACT,
                metrics,
                f"{totals['points']} points across {len(per_subprogram)} "
                f"subprogram(s), all bit-exact" + _redrawn_note(totals),
            )
        if rtol is not None and worst_rel <= float(rtol):
            return self._verdict(
                candidate,
                Confidence.TOLERANCED,
                metrics,
                f"{totals['bit_exact']}/{totals['points']} bit-exact, "
                f"max_rel={worst_rel:.3e} within rtol={rtol}",
            )
        return self._verdict(
            candidate,
            Confidence.FAILED,
            metrics,
            f"{totals['points'] - totals['bit_exact']}/{totals['points']} points differ "
            f"(max {totals['max_ulp']} ULP, max_rel={worst_rel:.3e}) and no rtol excuses them",
        )

    @staticmethod
    def _run_setup(
        setup: list[dict[str, Any]],
        translated: Any,
        truth: Any,
        wrappers: dict[str, str],
        arg_naming: str = "lower",
    ) -> None:
        from recast.transform.numpy.vocabulary import pysafe

        spell = pysafe if arg_naming == "pysafe" else _f2py_name
        for call in setup:
            name = call["subprogram"]
            inputs = call.get("inputs", {})
            getattr(translated, pysafe(name))(**{pysafe(k): v for k, v in inputs.items()})
            if truth is None:
                # A replayed reference has no state to set: whatever the
                # production run's module state was is already folded into the
                # numbers it recorded. The candidate still needs the call, and
                # an operator whose ``setup`` does not match the run's own
                # initialization gets a difference rather than a silent pass --
                # which is the correct outcome and worth naming, because it is
                # the one thing about a replay that cannot be checked from
                # here.
                continue
            getattr(truth, wrappers.get(name, f"w_{name}"))(
                **{spell(k): v for k, v in inputs.items()}
            )

    # -- one subprogram -------------------------------------------------------

    def _compare_subprogram(
        self,
        np: Any,
        name: str,
        sub: dict[str, Any],
        translated_fn: Any,
        truth_fn: Any,
        trials: int,
        dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        profile: Any = None,
        unit_uid: str = "",
        dominant_at: float | None = None,
        dominant_axis: Any = -1,
        rel_scale: str = "element",
        draws: int = 1,
        call_seconds: float = 0.0,
        arg_naming: str = "lower",
        convention: str = "f2py",
        samples: list[dict[str, Any]] | None = None,
        fitted: dict[str, int] | None = None,
        scratch: Path | None = None,
        reference_isolated: bool = False,
    ) -> dict[str, Any]:
        """Compare one subprogram over ``trials`` draws.

        ``fitted`` is what a first pass grew an unpinned extent to
        (:meth:`_fit_extents`): those extents arrive pinned in ``dims``, and
        the comparison starts again from the first trial so that every trial
        is compared at one shape. It is recorded on the outcome, because the
        shape the points were bit-exact at is part of what they say.
        """
        from recast.transform.numpy.vocabulary import pysafe

        if convention not in {"f2py", "emitted", "recorded"}:
            return {"error": f"oracle declares unsupported return convention {convention!r}"}

        declared_dtypes = [
            (f"argument {a.get('name', '<unnamed>')!r}", a.get("dtype"))
            for a in sub["args"]
            if a.get("dtype") != PROCEDURE_DTYPE and not drawable_path(a)
        ]
        if sub["kind"] == "function":
            declared_dtypes.append(("function result", sub.get("result_dtype")))
        unsupported = [
            f"{place}={dtype!r}"
            for place, dtype in declared_dtypes
            if not isinstance(dtype, str)
            or (dtype not in SUPPORTED_DTYPES and not (dtype == "str" and convention == "recorded"))
        ]
        # A character scalar (``phase = 'sun'``) is not sampled, but a
        # recording carries the value the run passed, and it is replayed
        # as it was written; no cast, no comparison, an input only.
        if unsupported:
            return {
                "error": "unsupported declared dtype(s) "
                f"{', '.join(unsupported)}; supported dtypes are "
                f"{', '.join(sorted(SUPPORTED_DTYPES))}"
            }

        required = [a for a in sub["args"] if not a.get("optional")]
        # Character dummies the source opens as files it creates. Not values
        # to compare -- both sides get a scratch path of their own, and what
        # is compared is the file each one left there. None of that applies to
        # a replay: its inputs are the recorded run's, including the path it
        # actually wrote to, and there is no second side to write a file.
        path_arguments = [] if samples is not None else [a for a in required if drawable_path(a)]
        if path_arguments and scratch is None:
            return {
                "error": "argument(s) "
                + ", ".join(a["name"] for a in path_arguments)
                + " name files the subprogram writes, and this comparison has "
                "nowhere to let the two sides write them"
            }
        outs_all = [a for a in sub["args"] if a["intent"] in ("OUT", "INOUT")]
        outs_required = [a for a in outs_all if not a.get("optional")]
        # Over the arguments the comparison passes, not every argument the
        # subprogram declares. An optional one is dropped from both calls --
        # the wrapper does not take it and the translation spells it as a
        # keyword sentinel -- so neither side reads or writes it and its
        # intent decides nothing here. ``integer, optional :: maxiter``, which
        # is how the corpus's ``secant`` declares its iteration cap, states no
        # intent and cost that subprogram its comparison over an argument no
        # call made.
        unknown_intents = [a["name"] for a in required if a["intent"] == "UNKNOWN"]
        if unknown_intents:
            return {
                "error": "argument(s) "
                f"{', '.join(unknown_intents)} have UNKNOWN intent; this verifier cannot "
                "know whether their post-call values are outputs"
            }
        if sub["kind"] == "function" and outs_required:
            # Required ones only, for the reason the comment above gives: an
            # optional dummy is dropped from both calls, so a function with
            # an optional intent(out) argument -- ``newunit(unit)``, whose
            # argument exists for callers that want the number twice -- has
            # no side effect to pair with its result on the call being made.
            names = ", ".join(a["name"] for a in outs_required)
            return {
                "error": f"function {name!r} declares OUT/INOUT dummy argument(s) "
                f"{names}; this verifier cannot pair both its result and side effects"
            }
        # A scalar LOGICAL INOUT goes through the wrapper as an integer, 0
        # or 1, converted on both sides of the call; an array of them has no
        # such path and no portable buffer ABI, and is refused as before.
        logical_inouts = [
            a["name"]
            for a in outs_all
            if a["intent"] == "INOUT" and a.get("dtype") == "bool" and a.get("dims")
        ]
        if convention == "f2py" and logical_inouts:
            return {
                "error": "f2py LOGICAL INOUT array dummy argument(s) "
                f"{', '.join(logical_inouts)} have no portable Python buffer ABI; "
                "refusing to guess the compiler's raw true representation"
            }

        # A scalar that names another argument's extent is not free data: it
        # must equal the extent the arrays are generated with, or every call
        # is a shape error rather than a comparison.
        dimension_names = {
            token
            for argument in sub["args"]
            for dim in argument.get("dims") or []
            for token in re.findall(
                r"[a-z_]\w*", f"{dim.get('lb') or ''} {dim.get('ub') or ''}".lower()
            )
        }
        # What the body's own entry checks say its dummies' shapes must be.
        # An assumed-shape dummy declares neither extent, and a subprogram
        # that stops unless ``size(c,1)`` is five has said the one thing this
        # harness could otherwise only guess -- and guess wrongly on every
        # draw it makes.
        guards = list(sub.get("shape_guards") or [])
        # ... and what they say about their values. ``bctype`` is a plain
        # integer dummy that the body stops on unless it is 1 or 2, so a draw
        # from this harness's default integer range is refused fifteen times
        # in sixteen and the subprogram runs out of attempts having compared
        # nothing. The operator's own range still wins: a project that has
        # said what it wants drawn has said it about this argument too.
        ranges = {
            **{
                str(guard["arg"]).lower(): (float(guard["low"]), float(guard["high"]))
                for guard in sub.get("value_guards") or []
            },
            **ranges,
        }

        points = bit_exact = nan_mismatch = 0
        integer_points = integer_mismatch = 0
        max_ulp = 0
        max_ulp_dominant = 0
        dominant_points = 0
        max_rel = 0.0
        redrawn = 0
        # Why each declined draw was declined, by kind, so the verdict can say.
        declined_by: dict[str, int] = {}
        # Trials that were compared only after a shape refusal moved the free
        # extents off the configured ones.
        reshaped = 0
        # Trials the project's input profile shaped. Those are never redrawn.
        shaped_trials = 0
        # Extents no operator pinned. A redraw varies these along with the
        # values, because some of what a subprogram will not accept is a
        # *shape*: a packed triangular workspace wants an extent that is a
        # function of the order it goes with, and one drawn independently of
        # that order is a subscript past the end rather than a comparison.
        # An extent this harness already grew is pinned, and is in ``dims``
        # rather than here.
        free_extents = sorted(dimension_names - {str(k).lower() for k in dims})
        # Whether a shape refusal is still answered by growing the extents.
        # Off once a growth has been fitted, once a search has found none --
        # a second search over the same shapes would find the same nothing --
        # and for a replay or a profiled draw, whose extents are not this
        # harness's to choose.
        fitting = samples is None and profile is None and not fitted
        attempts = 1 if samples is not None else max(1, int(draws))
        # Replayed samples are the trials, and there are as many as were
        # recorded. ``trials`` is a sampling parameter and does not apply: a
        # recording cannot be asked for more points than it holds, and
        # truncating it to a count chosen here would silently narrow the
        # evidence.
        rounds: list[Any] = list(samples) if samples is not None else list(range(trials))
        per_sample: list[dict[str, Any]] = []
        for round_index, round_item in enumerate(rounds):
            declined = ""
            reshape = False
            for attempt in range(attempts):
                reason = ""
                # hash() is salted per process; a seed must not be.
                salt = f"{name}:{round_index}"
                if attempt:
                    salt = f"{salt}:{attempt}"
                rng = np.random.default_rng(int.from_bytes(salt.encode(), "big") % 2**32)
                trial_dims = dict(dims)
                if reshape:
                    # Only once a draw has been refused for its *shape*. A
                    # value the subprogram will not take -- a mode it stops
                    # on, an argument that drives the arithmetic to NaN -- is
                    # answered by drawing values again, and moving the extents
                    # as well would break the relations the default choice
                    # keeps: every unpinned extent is the same number, so a
                    # leading dimension is never smaller than the order it
                    # carries.
                    ceiling = int(dims.get("default_dim", DEFAULT_DIMENSION))
                    for extent in free_extents:
                        trial_dims[extent] = int(rng.integers(1, ceiling + 1))
                staged: list[dict[str, Any]] = []
                # A path argument is drawn as a scratch name, one per side:
                # both sides create the file the source's OPEN creates, and
                # writing to one path would have the second call overwrite
                # what the comparison is about to read.
                drawn_paths: dict[str, str] = {}
                truth_paths: dict[str, str] = {}
                if path_arguments:
                    trial_root = Path(str(scratch)) / f"{round_index}.{attempt}"
                    for side in ("c", "r"):
                        (trial_root / side).mkdir(parents=True, exist_ok=True)
                    drawn_paths = {
                        a["name"]: str(trial_root / "c" / a["name"]) for a in path_arguments
                    }
                    truth_paths = {
                        a["name"]: str(trial_root / "r" / a["name"]) for a in path_arguments
                    }
                if samples is not None:
                    bound = self._recorded_inputs(np, required, round_item)
                    if isinstance(bound, str):
                        return {"error": bound}
                    inputs = bound
                else:
                    drawn = self._generated_inputs(
                        np,
                        required,
                        dimension_names,
                        dims,
                        trial_dims,
                        ranges,
                        rng,
                        drawn_paths,
                        guards,
                    )
                    if isinstance(drawn, str):
                        return {"error": drawn}
                    inputs = drawn

                recorded_outputs = None
                if samples is not None:
                    recorded_outputs = round_item.get("outputs")
                    if not isinstance(recorded_outputs, dict):
                        source = round_item.get("source", "sample")
                        return {"error": f"{source} has no OUTPUT mapping"}
                    required_output_names = (
                        [sub.get("result") or "result"]
                        if sub["kind"] == "function"
                        else [a["name"] for a in outs_required]
                    )
                    missing_outputs = [
                        output
                        for output in required_output_names
                        if output.lower() not in recorded_outputs
                    ]
                    if missing_outputs:
                        return {
                            "error": "the recorded sample carries no value for required output(s) "
                            f"{', '.join(missing_outputs)}; partial output evidence is not a pass"
                        }

                shaped = False
                if profile is not None and samples is None:
                    # The project's profile shapes *generated* inputs only. A
                    # recording's inputs are already in the domain by
                    # construction, and letting anything edit the production
                    # run's own numbers before the artifact is judged on them
                    # would let the exam be rewritten.
                    inputs, shaped = self._shape_inputs(
                        np, profile, unit_uid, name, round_index, inputs, rng
                    )
                    if shaped:
                        shaped_trials += 1
                site = _profile_site(unit_uid, name, round_index)

                # Keyword calls on both sides: f2py reorders inferred-dimension
                # scalars into trailing keywords, so positional order is not a
                # shared vocabulary -- names are.
                translated_kwargs = {
                    pysafe(a["name"]): inputs[a["name"]]
                    for a in required
                    if a["intent"] != "OUT" or (a.get("buffer") and a["name"] in inputs)
                }
                # How the reference spells an argument is the reference's business,
                # and it declares which on its handle. f2py lowercases every dummy
                # name, because Fortran is case-insensitive and the source's
                # spelling is not a fact about the interface -- a candidate that
                # reports `sl_prePBL` still reaches the same oracle argument. An
                # anchor emitted by this engine's own backend spells names the
                # emitted way instead, because both sides of that comparison came
                # out of the same emitter.
                #
                # A caller-buffer OUT array is handed to the reference as well:
                # it is the caller's storage on both sides, and the reference
                # cannot allocate what its wrapper never sized. The copy
                # ``_truth_input`` makes keeps the two sides independent.
                spell = pysafe if arg_naming == "pysafe" else _f2py_name
                handed = [
                    a
                    for a in required
                    if a["intent"] != "OUT" or (a.get("buffer") and a["name"] in inputs)
                ]
                try:
                    truth_kwargs = {
                        spell(a["name"]): self._truth_input(np, a, inputs[a["name"]], convention)
                        for a in handed
                    }
                except Exception as error:
                    if shaped:
                        raise InputProfileError(
                            f"{site} returned inputs the reference does not take: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    return {
                        "error": f"oracle input preparation failed: {type(error).__name__}: {error}"
                    }
                for argument_name, reference_path in truth_paths.items():
                    truth_kwargs[spell(argument_name)] = reference_path
                truth_args = [truth_kwargs[spell(a["name"])] for a in handed]
                if shaped:
                    # The profile asserts the reference takes this draw, so the
                    # reference goes first and decides. Refused there, the
                    # assertion is what is wrong and nothing about the
                    # candidate is being judged; taken there and refused by the
                    # candidate, the candidate has failed on inputs the source
                    # accepts -- not a draw to make again.
                    try:
                        truth_out = truth_fn(**truth_kwargs)
                    except Exception as error:
                        raise InputProfileError(
                            f"{site} returned inputs the reference does not take: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    try:
                        translated_out = translated_fn(**translated_kwargs)
                    except (Exception, SystemExit) as error:
                        return {
                            "error": "candidate raised on shaped inputs the reference took: "
                            f"{type(error).__name__}: {error}"
                        }
                else:
                    try:
                        with _bounded(call_seconds):
                            translated_out = translated_fn(**translated_kwargs)
                    except _CallTimedOut as error:
                        # The draw, not the translation: the reference runs
                        # the same loop and would not come back from it
                        # either, so it is not called on this one.
                        declined = f"candidate {error}"
                        redrawn += 1
                        continue
                    except (SystemExit, IndexError) as error:
                        if (
                            isinstance(error, _NegativeSubscript)
                            and reference_isolated
                            and truth_fn is not None
                            and samples is None
                        ):
                            # The candidate refused a subscript below the
                            # dummy's declared lower bound rather than wrapping
                            # it to the other end (see ``_NoWrapArray``). A
                            # bounds-checked reference in its own process is the
                            # authority on such a draw: it names the array and
                            # the bound and ends cleanly, so the draw is
                            # declined under the reference's own reason (#42).
                            # An in-process reference cannot be trusted to
                            # survive the read, so it is left uncalled and the
                            # candidate's refusal stands.
                            try:
                                truth_fn(**truth_kwargs)
                            except ReferenceAborted as ref_error:
                                declined = f"reference aborted: {ref_error}"
                                why = (
                                    "reference subscript out of bounds"
                                    if _bounds_violation(ref_error.runtime_error)
                                    else "reference error stop"
                                )
                                declined_by[why] = declined_by.get(why, 0) + 1
                                redrawn += 1
                                continue
                            except Exception:  # noqa: S110 - did not abort: the candidate's refusal stands
                                pass
                        # Not a comparison that failed -- a draw the subprogram
                        # does not take. ``SystemExit`` is a translated ERROR
                        # STOP: the source itself saying these arguments are not
                        # its own, and the reference would say the same by
                        # ending the process, taking every other unit's verdict
                        # with it. ``IndexError`` is a subscript past a dummy
                        # array's declared extent, where the reference, compiled
                        # without bounds checking, reads memory the call does
                        # not own. Either way the reference must not be called
                        # on this draw; draw again.
                        declined = f"candidate raised: {type(error).__name__}: {error}"
                        overrun = isinstance(error, IndexError)
                        why = "subscript past extent" if overrun else "error stop"
                        declined_by[why] = declined_by.get(why, 0) + 1
                        # A ``_NegativeSubscript`` is a value outside the
                        # source's own domain, not a shape this harness's
                        # default got wrong (see the class) -- no extent
                        # grows its way out of a negative index, so it is
                        # not a candidate for ``_fit_extents`` and does not
                        # earn the subprogram a ``reshaped`` count below.
                        growable = isinstance(error, IndexError) and not isinstance(
                            error, _NegativeSubscript
                        )
                        if growable and fitting and free_extents:
                            # A subscript past the end at extents nobody
                            # pinned is this harness's own default being
                            # wrong about the shape, not the draw being
                            # wrong about the values. Grow the default until
                            # the body's subscripts fit and compare the
                            # subprogram again from its first trial, so that
                            # every trial is compared at one shape. Only
                            # once: what the growth finds is pinned, and what
                            # it does not find is what the redraw below is
                            # for. A project that shapes its own inputs has
                            # said what its subprograms take, and a replay's
                            # extents are the recording's.
                            table = self._fit_extents(
                                np,
                                name,
                                required,
                                dimension_names,
                                translated_fn,
                                dims,
                                ranges,
                                free_extents,
                                trials,
                                call_seconds,
                                path_arguments,
                                scratch,
                                guards,
                            )
                            grown = {e: int(table[e]) for e in free_extents if e in table}
                            if grown:
                                return self._compare_subprogram(
                                    np,
                                    name,
                                    sub,
                                    translated_fn,
                                    truth_fn,
                                    trials,
                                    table,
                                    ranges,
                                    profile=profile,
                                    unit_uid=unit_uid,
                                    dominant_at=dominant_at,
                                    dominant_axis=dominant_axis,
                                    rel_scale=rel_scale,
                                    draws=draws,
                                    call_seconds=call_seconds,
                                    arg_naming=arg_naming,
                                    convention=convention,
                                    samples=samples,
                                    fitted=grown,
                                    scratch=scratch,
                                    reference_isolated=reference_isolated,
                                )
                            fitting = False
                        # Only where there is an extent left to move: with
                        # every extent pinned or fitted there is nothing to
                        # reshape, and counting the trial as reshaped would
                        # name extents that did not move.
                        reshape = reshape or (bool(free_extents) and growable)
                        redrawn += 1
                        continue
                    except Exception as error:
                        return {"error": f"candidate raised: {type(error).__name__}: {error}"}
                    if samples is not None:
                        # Nothing to call: the reference already ran, in
                        # production, and what it produced is the recording.
                        truth_out = recorded_outputs
                    else:
                        try:
                            truth_out = truth_fn(**truth_kwargs)
                        except ReferenceAborted as error:
                            # The reference ended its process on this draw:
                            # an ERROR STOP the source takes on inputs that
                            # are not its own, the way the candidate's
                            # SystemExit says the same. Not a comparison;
                            # draw again, and say so (#21). A bounds check
                            # tripping is the same answer with a different
                            # name: the draw put a subscript outside an
                            # array, and what the reference would have read
                            # there is this process's memory, not the
                            # source's arithmetic (#42) -- the kind says
                            # which, so a profile can be written for it.
                            declined = f"reference aborted: {error}"
                            why = (
                                "reference subscript out of bounds"
                                if _bounds_violation(error.runtime_error)
                                else "reference error stop"
                            )
                            declined_by[why] = declined_by.get(why, 0) + 1
                            redrawn += 1
                            continue
                        except Exception as error:
                            return {"error": f"oracle raised: {type(error).__name__}: {error}"}

                pairs = self._paired_outputs(
                    sub,
                    outs_all,
                    outs_required,
                    required,
                    translated_out,
                    truth_out,
                    truth_args,
                    convention,
                )
                if isinstance(pairs, str):
                    return {"error": pairs}
                output_dtypes = (
                    {sub.get("result") or "result": sub.get("result_dtype")}
                    if sub["kind"] == "function"
                    else {a["name"]: a.get("dtype") for a in outs_all}
                )
                if path_arguments:
                    produced = self._file_outputs(np, path_arguments, drawn_paths, truth_paths)
                    if isinstance(produced, str):
                        return {"error": produced}
                    pairs = [*pairs, *produced]
                    output_dtypes = {
                        **output_dtypes,
                        **{label: "int32" for label, _ours, _theirs in produced},
                    }
                for label, ours, theirs in pairs:
                    declared_dtype = output_dtypes.get(label)
                    if declared_dtype in {"int32", "int64"}:
                        shaped_ours = self._integer_output(
                            np, ours, declared_dtype, label=label, side="candidate"
                        )
                        if isinstance(shaped_ours, str):
                            return {"error": shaped_ours}
                        shaped_theirs = self._integer_output(
                            np, theirs, declared_dtype, label=label, side="oracle"
                        )
                        if isinstance(shaped_theirs, str):
                            return {"error": shaped_theirs}
                        if shaped_ours.shape != shaped_theirs.shape:
                            return {
                                "error": f"{label}: shape {shaped_ours.shape} "
                                f"vs {shaped_theirs.shape}"
                            }
                        exact = shaped_ours == shaped_theirs
                        compared = int(exact.size)
                        agreed = int(np.count_nonzero(exact))
                        staged.append(
                            {
                                "points": compared,
                                "bit_exact": agreed,
                                "integer_points": compared,
                                "integer_mismatch": compared - agreed,
                            }
                        )
                        continue
                    if declared_dtype == "bool":
                        # f2py exposes Fortran LOGICAL as a C int.  A true value
                        # need only be nonzero: gfortran commonly emits 1/-1/-2,
                        # and another compiler may choose a different bit pattern.
                        # Compare the declared logical meaning, not that private
                        # representation.
                        shaped_ours = np.asarray(np.asarray(ours) != 0, dtype=np.float64)
                        shaped_theirs = np.asarray(np.asarray(theirs) != 0, dtype=np.float64)
                    elif declared_dtype in COMPLEX_DTYPES:
                        # Both parts, interleaved along a last axis of 2: a
                        # complex64 is widened first, which loses nothing.
                        shaped_ours = _complex_parts(np, ours)
                        shaped_theirs = _complex_parts(np, theirs)
                    else:
                        shaped_ours = np.asarray(ours, dtype=np.float64)
                        shaped_theirs = np.asarray(theirs, dtype=np.float64)
                    if shaped_ours.shape != shaped_theirs.shape:
                        return {
                            "error": f"{label}: shape {shaped_ours.shape} vs {shaped_theirs.shape}"
                        }
                    a = shaped_ours.ravel()
                    b = shaped_theirs.ravel()
                    audit = ulp_audit(
                        a.tolist(),
                        b.tolist(),
                        dominant=self._dominance(np, shaped_theirs, dominant_at, dominant_axis),
                    )
                    if samples is None:
                        nan_ours = np.isnan(a)
                        nan_theirs = np.isnan(b)
                        if shaped and nan_theirs.any():
                            raise InputProfileError(
                                f"{site} returned inputs on which the reference produced "
                                f"NaN in {label}; a NaN-tainted trial compares the compiler's "
                                "scheduling rather than the translation, and a shaped draw "
                                "is not drawn again"
                            )
                        if nan_ours.any() and bool((nan_ours == nan_theirs).all()):
                            # A draw that put the subprogram outside its numeric
                            # domain on *both* sides: a square root of a negative,
                            # a division that overflowed. Fortran does not say
                            # what MIN and MAX return for a NaN operand, and
                            # gfortran's answer is whichever operand its register
                            # allocator made the second one, so what the two
                            # sides do with the NaN afterwards is the compiler's
                            # scheduling rather than the translation; the trial is
                            # drawn again. A NaN on one side where the other has
                            # a number is not that: it is a ``nan_mismatch``, and
                            # it fails the unit.
                            reason = f"{label}: a NaN in the compared values on both sides"
                            continue
                    measured: dict[str, Any] = {
                        "points": audit["total_points"],
                        "bit_exact": audit["bit_exact"],
                        "nan_mismatch": audit["nan_mismatch"],
                        "max_ulp": audit["max_ulp"],
                    }
                    if "max_ulp_dominant" in audit:
                        measured["max_ulp_dominant"] = audit["max_ulp_dominant"]
                        measured["dominant_points"] = audit["dominant_points"]
                    if audit["bit_exact"] != audit["total_points"]:
                        # ``rel_scale``: each element against itself (the default),
                        # or ``"array"`` -- against the array's largest magnitude,
                        # for a layout where a cancellation residual of 1e-17 sits
                        # beside values of order one and its own relative error
                        # says nothing about the translation.
                        if rel_scale == "array":
                            scale = np.maximum(float(np.abs(b).max()) if b.size else 0.0, 1e-300)
                        else:
                            scale = np.maximum(np.abs(b), 1e-300)
                        with np.errstate(invalid="ignore"):
                            rel = np.abs(a - b) / scale
                        # A one-sided NaN is counted in ``nan_mismatch``; it has
                        # no relative error to report.
                        rel = rel[~np.isnan(rel)]
                        measured["max_rel"] = float(rel.max()) if rel.size else 0.0
                    staged.append(measured)
                if reason:
                    declined = reason
                    declined_by["NaN on both sides"] = declined_by.get("NaN on both sides", 0) + 1
                    redrawn += 1
                    continue
                if reshape:
                    reshaped += 1
                if samples is not None:
                    # What each recorded sample measured, for a gate that
                    # comes back to the worst ones (the conditioning check).
                    per_sample.append(
                        {
                            "sample": round_index,
                            "max_ulp": max((m.get("max_ulp", 0) for m in staged), default=0),
                            "max_ulp_dominant": max(
                                (m.get("max_ulp_dominant", 0) for m in staged), default=0
                            ),
                        }
                    )
                for measured in staged:
                    points += measured["points"]
                    bit_exact += measured["bit_exact"]
                    nan_mismatch += measured.get("nan_mismatch", 0)
                    integer_points += measured.get("integer_points", 0)
                    integer_mismatch += measured.get("integer_mismatch", 0)
                    max_ulp = max(max_ulp, measured.get("max_ulp", 0))
                    if "max_ulp_dominant" in measured:
                        max_ulp_dominant = max(max_ulp_dominant, measured["max_ulp_dominant"])
                        dominant_points += measured["dominant_points"]
                    max_rel = max(max_rel, measured.get("max_rel", 0.0))
                break
            else:
                # Every attempt declined: say what kinds, then the last
                # reason in full -- the kinds are what a profile answers.
                kinds = _declined_summary(declined_by)
                return {
                    "error": f"no draw this harness could compare in {attempts} attempt(s)"
                    + (f" ({kinds})" if kinds else "")
                    + ": "
                    + (declined or "reason not recorded")
                }
        if samples is None and reshaped * 2 > len(rounds):
            # A trial compared only after its extents were moved is evidence
            # about the extents the redraw happened to land on, not the ones
            # asked for. A subprogram that passes mostly that way -- a packed
            # workspace whose ``lr`` must be ``n(n+1)/2`` at every order, so the
            # default extents never work and the redraws that do are n = 1 --
            # is unverified at the configured extents, and says so by name
            # rather than passing on the handful of points that did fit.
            return {
                "error": f"{reshaped} of {len(rounds)} trial(s) were compared only after the "
                f"free extent(s) {', '.join(free_extents)} were moved off the configured "
                f"values; the {points} point(s) that fit are not evidence at those extents. "
                "Pin `dims` to extents the subprogram takes"
            }
        if samples is None and redrawn > len(rounds):
            # More draws were declined than trials compared: the survivors
            # are a minority of what the configured draw produces, and a
            # candidate that stops or overruns where the source does not
            # would pass on exactly that minority. Unverified at the
            # configured draw, and says so by name rather than passing on it.
            return {
                "error": f"{redrawn} draw(s) were declined ({_declined_summary(declined_by)}) "
                f"to compare {len(rounds)} trial(s); the trials compared are a minority of "
                "what the configured draw produces and are not evidence about the rest. "
                "Narrow the draw with `ranges`, or pin `dims`, to values the subprogram takes"
            }
        outcome: dict[str, Any] = {
            "points": points,
            "bit_exact": bit_exact,
            "max_ulp": max_ulp,
            "max_rel": max_rel,
            "nan_mismatch": nan_mismatch,
            "integer_points": integer_points,
            "integer_mismatch": integer_mismatch,
            "redrawn": redrawn,
            "declined": declined_by,
            "reshaped": reshaped,
            "shaped": shaped_trials,
        }
        if fitted:
            # What was compared, at extents this harness chose: a reader who
            # is told the points were bit-exact is owed the shape they were
            # bit-exact at.
            outcome["extents"] = fitted
        if dominant_at is not None:
            outcome["max_ulp_dominant"] = max_ulp_dominant
            outcome["dominant_points"] = dominant_points
        if samples is not None:
            outcome["per_sample"] = per_sample
        return outcome

    def _fit_extents(
        self,
        np: Any,
        name: str,
        required: list[dict[str, Any]],
        dimension_names: set[str],
        translated_fn: Any,
        dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        free_extents: list[str],
        trials: int,
        call_seconds: float,
        path_arguments: list[dict[str, Any]] | None = None,
        scratch: Path | None = None,
        guards: Sequence[dict[str, Any]] = (),
    ) -> dict[str, int]:
        """Grow the extents nobody pinned until the body's subscripts fit.

        An extent no operator pinned is this harness's own choice rather than
        a value the run asked for: every one of them is ``default_dim``. For a
        packed workspace that choice is never right and cannot be -- MINPACK's
        ``dogleg`` reads the upper triangle of an order-``n`` matrix out of
        ``r(lr)``, so ``lr`` has to be ``n(n+1)/2`` and is never ``n`` -- and
        at it the subprogram is not comparable at all: the first subscript the
        body forms is already past the end.

        So the default is *grown*, and only ever grown. A longer workspace at
        the same order is the problem the operator configured, one size larger;
        the extents a shape redraw moves to are a *smaller* problem, and a
        different one every trial, which is why a subprogram compared that way
        fails by name (see the ``reshaped`` floor). The growth is decided once,
        at the first shape a trial refused, and the subprogram is compared
        again from its first trial at it -- so every trial holds one shape,
        and the metrics say which.

        The candidate's own refusal is what a shape is judged by: an
        ``IndexError`` is a subscript past a dummy's declared extent and there
        is nothing else here that knows what the body needs. The reference is
        not called -- on these draws it would read memory the call does not
        own -- and a draw refused for its *values* (an ``ERROR STOP``, a loop
        it does not come back from) says nothing about the shape, so it is
        neither a fit nor a reason to grow.
        """
        from recast.transform.numpy.vocabulary import pysafe

        def fits(table: dict[str, int]) -> bool:
            for index in range(trials):
                # The trials' own draws, at the shape under test: a shape
                # fitted against draws of this pass's own would be a shape
                # nothing that gets compared was ever made at.
                rng = np.random.default_rng(
                    int.from_bytes(f"{name}:{index}".encode(), "big") % 2**32
                )
                paths = {}
                if path_arguments and scratch is not None:
                    root = Path(str(scratch)) / f"fit.{index}"
                    root.mkdir(parents=True, exist_ok=True)
                    paths = {a["name"]: str(root / a["name"]) for a in path_arguments}
                inputs = self._generated_inputs(
                    np, required, dimension_names, dims, table, ranges, rng, paths, guards
                )
                if isinstance(inputs, str):
                    return True  # no draw to make: not a shape this can fit
                kwargs = {
                    pysafe(a["name"]): inputs[a["name"]]
                    for a in required
                    if a["intent"] != "OUT" or (a.get("buffer") and a["name"] in inputs)
                }
                try:
                    with _bounded(call_seconds):
                        translated_fn(**kwargs)
                except _NegativeSubscript:
                    continue  # a value refusal, not a shape one -- see the class
                except IndexError:
                    return False
                except (Exception, SystemExit):  # noqa: S112 - a value refusal, not a shape
                    continue  # this draw has nothing to say about the shape
            return True

        if fits(dims):
            return dims
        base = {extent: _resolve_extent(extent, dims) for extent in free_extents}
        # Each extent alone, and the one sizing fewest of the subprogram's
        # arrays first: a packed workspace is the extent of one array, where
        # an order is what every other array is cut to. Growing the order
        # raises the requirement along with the supply and arrives nowhere,
        # and it is the order the operator's own default is a statement
        # about. Every extent together is tried last, for a body whose
        # workspaces are more than one.
        sized = {
            extent: sum(
                1
                for argument in required
                for dim in argument.get("dims") or []
                if extent in re.findall(r"[a-z_]\w*", str(dim.get("ub") or "").lower())
            )
            for extent in free_extents
        }
        groups = [[extent] for extent in sorted(free_extents, key=lambda e: (sized[e], e))]
        if len(free_extents) > 1:
            groups.append(sorted(free_extents))
        for group in groups:
            for factor in GROWTH_FACTORS:
                grown = {e: min(base[e] * factor, MAX_FITTED_EXTENT) for e in group}
                if all(grown[e] == base[e] for e in group):
                    continue
                table = {**dims, **grown}
                if fits(table):
                    return table
        return dims

    @staticmethod
    def _devices(translated: Any, handle: dict[str, Any]) -> dict[str, str]:
        """Which device each side ran on, when either side says.

        Every rung of the ladder is a claim about an environment rather than
        about the code -- the same candidate and the same oracle can agree to
        the bit on one machine and differ on the next -- and for an accelerator
        backend the device is the half of that environment most likely to move.
        A verdict that does not record it cannot be re-argued later.

        Asked for rather than detected, and by the same convention as
        ``_SIGNATURES``: the emitted module declares
        ``_DEVICE`` if it knows, and an Oracle puts one on its handle. Reaching
        for ``jax.devices()`` here instead would put an accelerator import in
        the core, which is the one thing the core does not do.
        """
        found = {
            "candidate_device": getattr(translated, "_DEVICE", None),
            "reference_device": handle.get("device"),
        }
        return {name: str(value) for name, value in found.items() if value}

    @staticmethod
    def _dominance(
        np: Any, reference: Any, dominant_at: float | None, axis: Any = -1
    ) -> list[bool] | None:
        """Which elements a ULP bound is allowed to be held to.

        ``|v| >= fraction * the maximum along the last axis``, so an element is
        judged against its own row: a column of small values is not excused by
        a large value somewhere else in the array. The *reference* side decides,
        because whether an element matters is a fact about what it should have
        been, not about the candidate being judged.

        ``axis`` is the operator's (``dominant_axis``): the last axis by
        default, or ``"all"`` for the whole array -- for a layout whose last
        axis is not a row of comparable values (a two-element sun/shade pair,
        say), where a cancellation residual of 1e-17 would otherwise be the
        maximum of its own row and judged at the ULP tier.
        """
        if dominant_at is None:
            return None
        magnitude = np.abs(reference)
        if magnitude.size == 0:
            # A zero-extent output (CLUBB's scalar tracers under
            # sclr_dim = 0): nothing to weigh, and no maximum to take.
            return []
        if axis in ("all", None) or magnitude.ndim <= 1:
            scale = magnitude.max()
        else:
            scale = magnitude.max(axis=int(axis), keepdims=True)
        mask: list[bool] = (magnitude >= dominant_at * scale).ravel().tolist()
        return mask

    @staticmethod
    def _type_recorded(np: Any, samples: list[dict[str, Any]], table: dict[str, Any]) -> int:
        """Cast each recorded sample's inputs to the dtypes its signature declares.

        The cast changes no value -- an integer written as ``3`` is 3 -- and
        a scalar recorded as a one-element array is shaped back to a scalar,
        on the output side as well, so it compares against a scalar."""
        kinds = {
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
            "float32": np.float32,
            "float64": np.float64,
            "complex64": np.complex64,
            "complex128": np.complex128,
        }
        cast = 0
        for sample in samples:
            sig = table.get(str(sample.get("subprogram", "")))
            if not sig:
                continue
            for argument in sig["args"]:
                key = argument["name"].lower()
                value = sample.get("outputs", {}).get(key)
                if isinstance(value, np.ndarray) and not argument.get("dims") and value.size == 1:
                    sample["outputs"][key] = value.reshape(-1)[0]
            for argument in sig["args"]:
                key = argument["name"].lower()
                dtype = kinds.get(str(argument["dtype"]))
                if key not in sample.get("inputs", {}) or dtype is None:
                    continue
                value = sample["inputs"][key]
                if isinstance(value, np.ndarray) and not argument.get("dims") and value.size == 1:
                    sample["inputs"][key] = dtype(value.reshape(-1)[0])
                    cast += 1
                elif isinstance(value, np.ndarray):
                    if value.dtype != dtype:
                        sample["inputs"][key] = np.asfortranarray(value.astype(dtype))
                        cast += 1
                elif not isinstance(value, dtype):
                    sample["inputs"][key] = dtype(value)
                    cast += 1
        return cast

    @staticmethod
    def _recorded_inputs(
        np: Any, required: list[dict[str, Any]], sample: dict[str, Any]
    ) -> dict[str, Any] | str:
        """Bind one recorded sample's INPUT sections to the declared arguments.

        By exact name, lowercased, and nothing else. The script this oracle
        came from matched fuzzily -- exact, then with ``in``/``out`` stripped,
        then any substring either way -- and filled whatever was left with
        zeros. That is defensible in a one-shot investigation and not in a
        gate: a substring match binds ``t`` to ``theta``, a zero fill invents
        an input the run never had, and either one produces numbers that can
        be compared and mean nothing. So a required argument the recording does
        not name is a refusal, which is what a verifier that fails closed owes
        its reader.
        """
        recorded = sample.get("inputs", {})
        inputs: dict[str, Any] = {}
        missing = []
        for argument in required:
            if argument["intent"] == "OUT":
                continue
            key = argument["name"].lower()
            if key not in recorded:
                missing.append(argument["name"])
                continue
            value = recorded[key]
            inputs[argument["name"]] = np.copy(value) if isinstance(value, np.ndarray) else value
        if missing:
            return (
                f"{sample.get('source', 'sample')} records no value for "
                f"{', '.join(missing)}; a replay does not invent one"
            )
        return inputs

    @staticmethod
    def _truth_input(
        np: Any,
        argument: dict[str, Any],
        value: Any,
        convention: str,
    ) -> Any:
        """Give the reference an independent input with its required ABI shape.

        f2py represents a scalar ``intent(inout)`` dummy as an in/output
        rank-0 array.  It accepts a NumPy scalar too, but that object is
        immutable: the wrapper updates a temporary and Python observes the
        original value.  A writable zero-dimensional ndarray is therefore
        part of the f2py calling convention, not a change to the sampled
        value.  Array INOUTs already arrive as independent writable copies;
        emitted and recorded references retain their own conventions.
        """
        if argument.get("dtype") == PROCEDURE_DTYPE:
            # The same object on both sides. Copying it would be meaningless
            # and independence is not the point here: what makes the
            # comparison a comparison is that both sides call the same thing.
            return value
        if convention == "f2py" and argument["intent"] == "INOUT" and not argument.get("dims"):
            if argument.get("dtype") == "bool":
                # The wrapper carries a scalar LOGICAL INOUT as an integer,
                # 0 or 1 (no portable buffer ABI for the logical itself).
                return np.asarray(np.int32(1 if value else 0)).copy()
            buffered = np.asarray(value).copy()
            if buffered.ndim != 0:
                raise ValueError(f"scalar INOUT {argument['name']!r} became rank {buffered.ndim}")
            return buffered
        return np.copy(value) if isinstance(value, np.ndarray) else value

    @staticmethod
    def _integer_output(
        np: Any,
        value: Any,
        declared_dtype: str,
        *,
        label: str,
        side: str,
    ) -> Any | str:
        """Validate and preserve one declared integer output exactly.

        Casting through float64 aliases adjacent int64 values above 2**53.
        Casting a float *to* an integer is no safer: it lets a candidate that
        violated its declared interface masquerade as one that did not.  Only
        actual integer values in the declared signed range enter the exact
        comparison.
        """
        try:
            raw = np.asarray(value)
        except Exception as error:
            return (
                f"{label}: {side} {declared_dtype} output cannot be represented as an "
                f"array: {type(error).__name__}: {error}"
            )

        if raw.dtype.kind not in {"i", "u", "O"}:
            return (
                f"{label}: {side} declared {declared_dtype} but produced non-integer "
                f"dtype {raw.dtype}"
            )
        if raw.dtype.kind == "O":
            for item in raw.flat:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                    return (
                        f"{label}: {side} declared {declared_dtype} but produced "
                        f"non-integer value of type {type(item).__name__}"
                    )

        target = np.dtype(np.int32 if declared_dtype == "int32" else np.int64)
        limits = np.iinfo(target)
        if raw.size:
            if raw.dtype.kind == "O":
                smallest = min(int(item) for item in raw.flat)
                largest = max(int(item) for item in raw.flat)
            else:
                smallest = int(raw.min())
                largest = int(raw.max())
            if smallest < int(limits.min) or largest > int(limits.max):
                offending = smallest if smallest < int(limits.min) else largest
                return (
                    f"{label}: {side} {declared_dtype} output value {offending} is outside "
                    f"[{int(limits.min)}, {int(limits.max)}]"
                )
        return raw.astype(target, copy=False)

    @staticmethod
    def _file_outputs(
        np: Any,
        path_arguments: list[dict[str, Any]],
        drawn_paths: dict[str, str],
        truth_paths: dict[str, str],
    ) -> list[tuple[str, Any, Any]] | str:
        """The file each side left at a path argument, as an output to compare.

        A subprogram whose only product is a file has no output argument for
        ``_paired_outputs`` to pair -- ``saveppm(filename, img)`` declares two
        inputs and nothing else -- and comparing its arguments compares what
        the caller already knew. What it produced is the bytes it wrote, and
        those are compared as declared integers: the bit-exact bar for a file
        is that it holds the same bytes, in the same order, and one that is a
        byte longer is a different file rather than a near-miss.

        A side that wrote nothing is not an empty file to compare against an
        empty file: the source's OPEN creates one, so its absence is the call
        having done nothing, and it is named rather than passed over.
        """
        produced: list[tuple[str, Any, Any]] = []
        for argument in path_arguments:
            label = f"{argument['name']} (file)"
            ours = _file_bytes(drawn_paths[argument["name"]])
            theirs = _file_bytes(truth_paths[argument["name"]])
            if ours is None or theirs is None:
                side = "candidate" if ours is None else "oracle"
                return f"{label}: the {side} left no file at the path it was given"
            produced.append(
                (
                    label,
                    np.frombuffer(ours, dtype=np.uint8).astype(np.int32),
                    np.frombuffer(theirs, dtype=np.uint8).astype(np.int32),
                )
            )
        return produced

    @staticmethod
    def _paired_outputs(
        sub: dict[str, Any],
        outs_all: list[dict[str, Any]],
        outs_required: list[dict[str, Any]],
        required: list[dict[str, Any]],
        translated_out: Any,
        truth_out: Any,
        truth_args: list[Any],
        convention: str = "f2py",
    ) -> list[tuple[str, Any, Any]] | str:
        """Match the two sides' outputs by argument.

        The translation returns every OUT/INOUT argument, optional ones
        included, in declaration order (a function returns its result). What
        the *reference* returns depends on what kind of reference it is, and it
        says which on its handle rather than being guessed at here.

        ``f2py`` returns the wrapper's ``intent(out)`` arguments and mutates
        the ``inout`` ones in place, so INOUT values are read back from the
        independent arrays that were passed (including rank-0 buffers for
        scalar INOUTs). ``emitted`` is a reference this engine's own backend
        produced -- a NumPy anchor for a port -- and returns exactly what the
        candidate does, because the same emitter wrote both.
        """
        if convention == "recorded":
            # ``truth_out`` is not a return value here -- it is the recorded
            # OUTPUT section, keyed by the name the probe wrote. The match is
            # therefore by exact name on both sides. Every required output was
            # preflighted before the candidate call; keep the same check here
            # as a fail-closed local invariant for direct callers.
            mine = _returned(translated_out)
            names = (
                [sub.get("result") or "result"]
                if sub["kind"] == "function"
                else [a["name"] for a in outs_all]
            )
            if len(mine) != len(names):
                return (
                    f"candidate returned {len(mine)} value(s) for {len(names)} "
                    "out-intent argument(s)"
                )
            ours_by_name = dict(zip(names, mine, strict=True))
            wanted = names if sub["kind"] == "function" else [a["name"] for a in outs_required]
            missing = [name for name in wanted if name.lower() not in truth_out]
            if missing:
                return (
                    "the recorded sample carries no value for required output(s) "
                    f"{', '.join(missing)}; partial output evidence is not a pass"
                )
            pairs: list[tuple[str, Any, Any]] = []
            for name in wanted:
                ours = ours_by_name[name]
                theirs = truth_out[name.lower()]
                # The probe format has no rank-0 section: a scalar result is
                # written as a section holding exactly one value and parses as
                # shape (1,). When the candidate returned a scalar, read the
                # recording back as the scalar it is. The one value is still
                # compared bit for bit, so the reshape hides nothing; without
                # it every recorded scalar function was "shape () vs (1,)".
                import numpy as np

                if getattr(theirs, "shape", None) == (1,) and np.ndim(ours) == 0:
                    theirs = theirs.reshape(())
                pairs.append((name, ours, theirs))
            return pairs

        if sub["kind"] == "function":
            # Both sides return the result, whatever kind of reference this is.
            return [(sub.get("result") or "result", translated_out, truth_out)]

        if convention == "emitted":
            mine = list(translated_out) if isinstance(translated_out, tuple) else [translated_out]
            yours = list(truth_out) if isinstance(truth_out, tuple) else [truth_out]
            names = [a["name"] for a in outs_all]
            if len(mine) != len(names) or len(yours) != len(names):
                return (
                    f"candidate returned {len(mine)} and reference {len(yours)} value(s) "
                    f"for {len(names)} out-intent argument(s)"
                )
            ours_by_name = dict(zip(names, mine, strict=True))
            theirs_by_name = dict(zip(names, yours, strict=True))
            return [
                (a["name"], ours_by_name[a["name"]], theirs_by_name[a["name"]])
                for a in outs_required
            ]

        ours = _returned(translated_out)
        if len(ours) != len(outs_all):
            return (
                f"candidate returned {len(ours)} value(s) for "
                f"{len(outs_all)} out-intent argument(s)"
            )
        by_name = dict(zip([a["name"] for a in outs_all], ours, strict=True))

        theirs_out = (
            list(truth_out)
            if isinstance(truth_out, tuple)
            else ([truth_out] if truth_out is not None else [])
        )
        # A caller-buffer OUT array is not among them: the wrapper spells it
        # ``intent(in out)``, because the caller owns the storage on both
        # sides, so it is read back from the array that was passed exactly as
        # an INOUT is.
        pure_out = [a for a in outs_required if a["intent"] == "OUT" and not a.get("buffer")]
        if len(theirs_out) != len(pure_out):
            return (
                f"oracle returned {len(theirs_out)} value(s) for "
                f"{len(pure_out)} intent(out) argument(s)"
            )
        theirs = dict(zip([a["name"] for a in pure_out], theirs_out, strict=True))
        passed_in = [a["name"] for a in required if a["intent"] != "OUT" or a.get("buffer")]
        read_back = [a for a in outs_required if a["intent"] == "INOUT" or a.get("buffer")]
        if read_back and len(truth_args) != len(passed_in):
            return (
                f"the reference was handed {len(truth_args)} argument(s) for "
                f"{len(passed_in)} the gate has to read an updated value back from"
            )
        for argument in read_back:
            theirs[argument["name"]] = truth_args[passed_in.index(argument["name"])]
            if argument["intent"] == "INOUT" and argument.get("dtype") == "bool":
                if not argument.get("dims") and convention == "f2py":
                    # Back from the wrapper's integer to the logical.
                    theirs[argument["name"]] = bool(int(theirs[argument["name"]]) != 0)

        return [(a["name"], by_name[a["name"]], theirs[a["name"]]) for a in outs_required]

    def _generated_inputs(
        self,
        np: Any,
        required: list[dict[str, Any]],
        dimension_names: set[str],
        dims: dict[str, int],
        trial_dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        rng: Any,
        paths: dict[str, str] | None = None,
        guards: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any] | str:
        """One draw's inputs by argument name, or why there is no draw.

        ``dims`` is what the operator pinned and ``trial_dims`` what this draw
        is being made at; they differ where an extent nobody pinned has been
        moved or grown for this trial. ``paths`` is the scratch name each
        path argument is drawn as, chosen by the caller because the two sides
        need different ones.
        """
        inputs: dict[str, Any] = {}
        shapes = _guarded_shapes(required, guards, trial_dims)
        for argument in required:
            if argument["intent"] == "OUT" and not argument.get("buffer"):
                continue
            # An intent(out) buffer is the caller's storage: generated
            # like an input, handed to the candidate, and compared
            # after the call the way any output is.
            lowered = argument["name"].lower()
            if argument["name"] in (paths or {}):
                inputs[argument["name"]] = (paths or {})[argument["name"]]
            elif argument.get("dtype") == PROCEDURE_DTYPE:
                interface = argument.get("interface")
                if not isinstance(interface, dict):
                    return (
                        f"procedure argument {argument['name']!r} carries no "
                        "interface; there is nothing to build a call-back from"
                    )
                try:
                    inputs[argument["name"]] = callback_for(np, argument["name"], interface)
                except ValueError as error:
                    return str(error)
            elif not argument.get("dims") and (lowered in dimension_names or lowered in dims):
                inputs[argument["name"]] = np.int32(_resolve_extent(lowered, trial_dims))
            else:
                inputs[argument["name"]] = self._value(
                    np, argument, trial_dims, ranges, rng, shapes.get(lowered)
                )
        return inputs

    def _value(
        self,
        np: Any,
        argument: dict[str, Any],
        dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        rng: Any,
        extents: list[int] | None = None,
    ) -> Any:
        name = argument["name"].lower()
        kinds = {
            "float64": np.float64,
            "float32": np.float32,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
            "complex128": np.complex128,
            "complex64": np.complex64,
        }
        if argument["dtype"] not in kinds:
            # Unsupported dtypes are refused before any draw; this is the
            # invariant, not a default.
            raise ValueError(f"{name}: no draw for declared dtype {argument['dtype']!r}")
        dtype = kinds[argument["dtype"]]
        shape = None
        if argument.get("dims"):
            shape = (
                tuple(extents)
                if extents is not None
                else tuple(_extent(d, dims) for d in argument["dims"])
            )
        if dtype in (np.complex128, np.complex64):
            low, high = ranges.get(name, DEFAULT_RANGE)
            part = np.float32 if dtype is np.complex64 else np.float64
            if shape is None:
                return dtype(complex(part(rng.uniform(low, high)), part(rng.uniform(low, high))))
            re_part = rng.uniform(low, high, size=shape).astype(part)
            im_part = rng.uniform(low, high, size=shape).astype(part)
            return np.asfortranarray((re_part + 1j * im_part).astype(dtype))
        if dtype in (np.float64, np.float32):
            low, high = ranges.get(name, DEFAULT_RANGE)
            if shape is None:
                return dtype(rng.uniform(low, high))
            drawn = np.asfortranarray(rng.uniform(low, high, size=shape).astype(dtype))
            return drawn.view(_no_wrap_array_type(np))
        if dtype in (np.int32, np.int64):
            # The source's own domain for the dummy (``select case (mode)``
            # with a stopping default) bounds the draw; the operator's
            # ``ranges`` still win where they speak.
            fallback = tuple(argument.get("domain") or DEFAULT_INTEGER_RANGE)
            low, high = ranges.get(name, fallback)
            if shape is None:
                return dtype(rng.integers(int(low), int(high) + 1))
            drawn = np.asfortranarray(
                rng.integers(int(low), int(high) + 1, size=shape).astype(dtype)
            )
            return drawn.view(_no_wrap_array_type(np))
        # A logical takes a range like anything else: ``pivot`` decides
        # whether ``qrfac`` writes ``ipvt`` at all, and an operator with no
        # way to pin it is comparing an array one side never defined.
        low, high = ranges.get(name, (0, 1))
        low, high = min(int(low), int(high)), max(int(low), int(high))
        if shape is None:
            return np.bool_(rng.integers(low, high + 1))
        drawn = np.asfortranarray(rng.integers(low, high + 1, size=shape).astype(np.bool_))
        return drawn.view(_no_wrap_array_type(np))

    # -- loading --------------------------------------------------------------

    @staticmethod
    def _load_input_profile(config: dict[str, Any]) -> Callable[..., Any] | None:
        """The project's ``recast_inputs.py``, when the tree carries one.

        Looked for at the root the run was given -- the same root the
        sources were read from, so the profile is part of the source
        artifact and travels, and is digested, with it. A tree without one
        is the generated path unchanged. A tree with one that does not
        import, or defines no ``prepare``, is a project whose assertion
        about its own inputs cannot be read, and that stops the gate rather
        than being taken as "no profile".
        """
        root = config.get("root")
        if root is None:
            return None
        path = Path(root) / INPUT_PROFILE
        if not path.is_file():
            return None
        # Compiled and run by hand rather than through the import machinery:
        # a SourceFileLoader would drop ``__pycache__/`` into the project
        # root, and the root is a checkout whose cleanliness is checked.
        module = types.ModuleType("recast_inputs")
        module.__file__ = str(path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise InputProfileError(
                f"{INPUT_PROFILE} could not be read: {type(error).__name__}: {error}"
            ) from error
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
        except (Exception, SystemExit) as error:
            raise InputProfileError(
                f"{INPUT_PROFILE} does not import: {type(error).__name__}: {error}"
            ) from error
        prepare: Callable[..., Any] | None = getattr(module, "prepare", None)
        if not callable(prepare):
            raise InputProfileError(
                f"{INPUT_PROFILE} defines no callable prepare(unit, subprogram, inputs, rng)"
            )
        return prepare

    @staticmethod
    def _shape_inputs(
        np: Any,
        prepare: Callable[..., Any],
        unit_uid: str,
        name: str,
        trial: int,
        drawn: dict[str, Any],
        rng: Any,
    ) -> tuple[dict[str, Any], bool]:
        """Hand one trial's draw to the profile; say whether it shaped it.

        The profile sees a copy, so the only way its work reaches the
        comparison is by returning it: a hook that edits the draw in place
        and returns ``None`` would otherwise be a shaped trial judged under
        the unshaped rules, silently. Returned inputs must name exactly the
        arguments drawn -- the profile shapes values, it does not rewrite
        the interface.
        """
        site = _profile_site(unit_uid, name, trial)
        offered = {key: _copy_input(value) for key, value in drawn.items()}
        try:
            shaped = prepare(unit_uid, name, offered, rng)
        except Exception as error:
            raise InputProfileError(f"{site} raised {type(error).__name__}: {error}") from error
        if shaped is None:
            edited = sorted(key for key in drawn if not _same_input(np, offered[key], drawn[key]))
            if edited:
                raise InputProfileError(
                    f"{site} edited {', '.join(edited)} in place and returned None; "
                    "return the shaped inputs, or None to leave the draw as it is"
                )
            return drawn, False
        if not isinstance(shaped, dict):
            raise InputProfileError(
                f"{site} returned {type(shaped).__name__}; return the inputs by argument "
                "name, or None"
            )
        if set(shaped) != set(drawn):
            missing = sorted(set(drawn) - set(shaped))
            extra = sorted(set(shaped) - set(drawn))
            raise InputProfileError(
                f"{site} returned inputs that do not name the arguments drawn"
                + (f"; missing {', '.join(missing)}" if missing else "")
                + (f"; unknown {', '.join(extra)}" if extra else "")
            )
        return shaped, True

    @staticmethod
    def _candidate_function(translated: Any, name: str) -> Any:
        """The emitted translation of the subprogram ``_SIGNATURES`` names.

        The table is in the source's vocabulary, and a Fortran name that is a
        Python keyword cannot be a Python definition of the same spelling:
        ``subroutine assert`` is emitted ``def assert_``. The trailing
        underscore is PEP 8's convention and a fact about Python rather than
        about any one backend -- ``static.rwset`` strips it back by the same
        rule -- so it is read back here, and a subprogram whose name needed
        it stops being invisible to this gate.
        """
        found = getattr(translated, name, None)
        if found is None and keyword.iskeyword(name):
            found = getattr(translated, f"{name}_", None)
        return found

    @staticmethod
    def _load_candidate(
        candidate: Candidate,
        workspace: Path,
        suffix: str = "_numpy.py",
        companions: Sequence[str | Path] = (),
    ) -> Any:
        """Write the candidate's files and import its generated module.

        The candidate is self-contained by design -- module, constants,
        use-constants -- so importing it needs nothing but its own files on
        the path. Companion modules, when a scheme has them, are the run's
        business: it names the directories of the candidates it emitted
        before this one (``config["companion_paths"]``), and they go on the
        path behind the candidate's own for the import.

        ``suffix`` picks which of those files is the one under judgement. A
        port carries more than one: the JAX module, and the NumPy module it
        host-delegates to and imports. Both are staged, only one is imported
        as the candidate, and the recipe says which by setting
        ``config["module_suffix"]``. Defaulting to the NumPy module keeps
        every existing config meaning what it did.

        Among the files that carry the suffix, the one under judgement is the
        unit's own -- ``<unit>_numpy.py`` for ``fortran:<unit>``. It used to
        be whichever came last in ``Candidate.files``, which was this unit's
        for as long as a candidate held exactly one such file. It no longer
        does: a candidate carries the translations of the siblings it
        imports, and a bundle written and read back is ordered by path, so
        ``sorting`` was judged against ``utils_numpy.py`` -- its signature
        table, its coverage, none of them the unit's. Which module a gate
        judges is not the file order's to decide.
        """
        staged = workspace / "candidate"
        staged.mkdir(parents=True, exist_ok=True)
        staged_stems = set()
        offered: list[Path] = []
        for path, content in candidate.files.items():
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            staged_stems.add(target.stem)
            if str(path).endswith(suffix):
                offered.append(target)
        module_path = _module_under_judgement(candidate.unit, offered, suffix)

        entries = [str(staged), *(str(p) for p in companions if str(p) != str(staged))]
        for entry in reversed(entries):
            sys.path.insert(0, entry)
        try:
            # Every staged name, not just the module under judgement: a
            # sibling left in ``sys.modules`` by an earlier unit would be
            # imported instead of the copy this candidate carries.
            for name in list(sys.modules):
                if name in staged_stems or name.endswith("_constants"):
                    del sys.modules[name]
            spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path.stem] = module
            spec.loader.exec_module(module)
        finally:
            for entry in entries:
                sys.path.remove(entry)
        return module

    def _verdict(
        self, candidate: Candidate, confidence: Confidence, metrics: dict[str, Any], detail: str
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics=metrics,
            detail=detail,
        )


def _module_under_judgement(unit: str, offered: list[Path], suffix: str) -> Path:
    """Which of a candidate's suffix-carrying files is the unit's own.

    One file is the unit's translation; the rest are the siblings it imports.
    Picking by name rather than by position keeps the answer the same whether
    the candidate came straight from the transform or through a bundle, which
    orders files by path. Two files that both claim the name, or none that
    does, is a candidate this gate cannot judge -- and a verifier says so
    rather than guessing.
    """
    if not offered:
        raise FileNotFoundError(f"candidate carries no *{suffix} module")
    if len(offered) == 1:
        return offered[0]
    own = f"{unit.rpartition(':')[2].rpartition('/')[2].lower()}{suffix}"
    named = [path for path in offered if path.name.lower() == own]
    if len(named) == 1:
        return named[0]
    carried = ", ".join(sorted(path.name for path in offered))
    raise FileNotFoundError(
        f"candidate for {unit} carries {len(offered)} *{suffix} modules ({carried}); "
        + ("none of them is" if not named else f"{len(named)} of them are")
        + f" the unit's own {own}, and which one is under judgement cannot be "
        "guessed from the file order"
    )


def _profile_site(unit_uid: str, name: str, trial: int) -> str:
    return f"{INPUT_PROFILE}: prepare({unit_uid!r}, {name!r}) at trial {trial}"


def _copy_input(value: Any) -> Any:
    """A copy of one drawn input for the profile to shape, in the layout the
    draw has: ``ndarray.copy()`` alone is C order, and a two-dimensional
    Fortran-ordered INOUT the profile returned untouched reached f2py as a
    copy that was "not fortran contiguous" (CLUBB's advance_helper_module,
    whose profile shapes the grid and leaves the rest)."""
    copy = getattr(value, "copy", None)
    if not callable(copy):
        return value
    try:
        return copy(order="K")
    except TypeError:
        return copy()


def _same_input(np: Any, offered: Any, drawn: Any) -> bool:
    if offered is drawn:
        return True
    try:
        return bool(np.array_equal(np.asarray(offered), np.asarray(drawn), equal_nan=True))
    except (TypeError, ValueError):
        return False


def factory(**_config: Any) -> BitexactVerifier:
    return BitexactVerifier()
