"""The runtime a translated module needs to mean what the Fortran meant.

Pasted verbatim into every generated file, which is why it lives here as real
code rather than as the string constant it used to be. A string is not linted,
not type-checked, and not testable, and this is the last place in the pipeline
that should be none of those things: these are the definitions of ``sign``,
``mod``, ``nint`` and ``transfer``, and each one exists precisely because
Python's answer differs from Fortran's.

The differences are not exotic. ``MOD`` truncates toward zero and ``%`` floors,
so they disagree on every negative operand. ``NINT`` rounds half away from zero
and ``round`` rounds to even, so they disagree on every exact ``.5``. Integer
division truncates in Fortran and floors in Python. A translation that uses the
Python spelling produces numbers that are close, plausible, and wrong in a way
no structural check notices -- only a bit-exact comparison against the original
catches it, and only on inputs that happen to reach the case.

``emit`` renders this module's source for inlining. The generated file has to
stand alone -- it is the product, and it is imported by a comparison harness
with no reason to have the engine installed -- so the text is copied in rather
than imported.
"""

from __future__ import annotations

import inspect
import math
import os
import re as _re
from typing import Any

import numpy as np

__all__ = ["REQUIRED_IMPORTS", "emit"]

REQUIRED_IMPORTS = (
    "import math",
    "import os",
    "import re as _re",
    "from typing import Any",
    "",
    "import numpy as np",
)
"""What a file containing ``emit()``'s text has to import for it to run.

Declared here rather than known by a header template in the Transform: this
module decides what it needs, and a template that has to be edited in step with
it would be a second copy of the same fact.
"""

_LIBM_STRICT = os.environ.get("PY_LIBM_STRICT", "1") == "1"
"""Strict libm, on by default.

``np.exp``/``log``/``power`` (npy_math, SIMD) differ from glibc libm by one
ULP. This backend serves the bit-exact gates, so agreement wins; setting
``PY_LIBM_STRICT=0`` buys throughput, and the njit and CUDA backends are the
ones to reach for when throughput is the point.
"""


def _f_vexp(x: Any) -> Any:
    if _LIBM_STRICT:
        return np.array([math.exp(v) for v in np.ravel(x)]).reshape(np.shape(x))
    return np.exp(x)


def _f_vlog(x: Any) -> Any:
    if _LIBM_STRICT:
        return np.array([_f_log(v) for v in np.ravel(x)]).reshape(np.shape(x))
    return np.log(x)


def _f_vlog10(x: Any) -> Any:
    if _LIBM_STRICT:
        return np.array([_f_log10(v) for v in np.ravel(x)]).reshape(np.shape(x))
    return np.log10(x)


def _f_vpow(a: Any, b: Any) -> Any:
    """array ** : CPython scalar pow == gfortran (libm pow / squaring for
    int exponents); np.power is 1 ULP off either."""
    if _LIBM_STRICT:
        bc = np.broadcast(a, b)
        return np.array([x**y for x, y in bc]).reshape(bc.shape)
    return a**b


def _f_powi(x: Any, n: Any) -> Any:
    """``x ** n`` for an integer exponent the compiler could not read off.

    ``(xe(i)-x0)**(j-1)`` inside a loop is one: there is no literal to
    expand, so what the emitter had left was Python's ``**``, which is a
    libm ``pow`` call. gfortran does not call ``pow`` for an integer
    exponent at any point -- libgcc's ``__powidf2`` squares and multiplies,
    LSB first, exactly as ``expand_power`` spells out when the exponent *is*
    a literal -- and the two are one to two ULP apart, which is enough to
    fail a bit-exact gate and not enough to notice by looking.
    """
    count = int(n)
    remaining = -count if count < 0 else count
    result: Any = None
    square = x
    while remaining:
        if remaining & 1:
            result = square if result is None else result * square
        remaining >>= 1
        if remaining:
            square = square * square
    if result is None:
        return x**0
    return 1.0 / result if count < 0 else result


def _f_cfold(fn: Any, *args: Any) -> Any:
    """gfortran evaluates constant-argument intrinsics at COMPILE time
    with MPFR (correctly rounded) — that value matches no runtime libm
    (proven: gamma(1.8) differs from BOTH libgfortran and glibc)."""
    import mpmath as mp

    with mp.workprec(200):
        return float(getattr(mp, fn)(*[mp.mpf(float(a)) for a in args]))


def _f_vachar(x: Any) -> Any:
    """Fortran ACHAR over an array: one character per element.

    ``chr`` elementwise rather than a NumPy ``str_`` view, because the
    result is an item list a WRITE hands to the formatter one value at a
    time, and a fixed-width NumPy string array would pad every one of them.
    """
    return np.array([chr(int(v)) for v in np.ravel(x)], dtype=object).reshape(np.shape(x))


def _f_vceil(x: Any) -> Any:
    """Fortran CEILING returns default INTEGER."""
    return np.ceil(x).astype(np.int32)


def _f_vfloor(x: Any) -> Any:
    """Fortran FLOOR returns default INTEGER; np.floor returns float."""
    return np.floor(x).astype(np.int32)


class _FLoopExit(Exception):
    """``EXIT <name>`` naming a DO that is not the innermost one.

    Python's ``break`` leaves one loop, and Fortran's named EXIT leaves the
    one it names. Emitting ``break`` for both is not a shape difference: it
    leaves the *inner* loop and then runs whatever follows it inside the
    outer one, so the program keeps going down a path the Fortran had
    abandoned. The named loop catches this and checks the name, so an EXIT
    crossing two loops passes through the first.
    """


class _FLoopCycle(Exception):
    """``CYCLE <name>`` naming a DO that is not the innermost one.

    The counterpart of ``_FLoopExit``, and the more dangerous of the two: a
    wrong ``continue`` re-runs an inner loop that was supposed to be
    finished, which usually still terminates and still produces numbers.
    """


class _FBlockExit(Exception):
    """``EXIT <name>`` naming an enclosing BLOCK construct.

    A BLOCK is inlined rather than emitted as a scope of its own, so there
    is no Python construct for a ``break`` to leave -- it would bind to
    whatever loop happens to be outside.
    """


class _FGoto(Exception):
    """forward-goto region jump (structured replacement for `goto L`)."""


def _f_ecall(fn: Any, *args: Any, **kw: Any) -> Any:
    """ELEMENTAL procedure broadcast over array actuals: run the scalar
    translation per element (keeps the strict-libm scalar paths and the
    scalar control flow intact). Keywords (optional/want_ sentinels)
    broadcast alongside."""
    return np.vectorize(fn)(*args, **kw)


def _f_copy_out(dst: Any, src: Any) -> None:
    """Copy a callee's returned OUT array into the caller's buffer.

    ``dst[...] = src`` when the shapes agree; when they do not -- a
    ``pcols``-wide buffer receiving an ``ncol``-wide result, or a rank-1
    buffer receiving a section -- the overlap is copied and the rest left as
    it was, which is what Fortran's by-reference OUT did. ``None`` is an
    unsupplied optional, and nothing is written."""
    if dst is None:
        return
    if not isinstance(src, np.ndarray):
        dst[...] = src
        return
    if src.shape == dst.shape:
        dst[...] = src
        return
    if src.ndim == dst.ndim and src.ndim > 1:
        slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape, strict=True))
        dst[slices] = src[slices]
    else:
        n = min(src.size, dst.size)
        dst.ravel()[:n] = src.ravel()[:n]


def _f_seq_tail(arr: Any, start: Any, *leading: Any) -> Any:
    """Sequence association of an array element with an assumed-size dummy.

    The caller's storage from 0-based column-major position ``start`` to the
    end of ``arr``, which is what ``x(*)`` spans when ``a(i, j)`` is passed
    for it. A view when ``arr`` is Fortran-contiguous -- every array the gate
    draws and every reshaped window is -- so the callee's in-place writes
    land in the caller's array; otherwise a copy, which ``_f_seq_tail_out``
    writes back.

    ``leading`` are the extents of a rank-2 dummy's leading axes, ``u(iue,
    *)``. The BLAS idiom ``h12(..., a(i, 1), mda, ...)`` -- a row of the
    matrix walked with the matrix's own leading extent -- is the matrix from
    that row and column on, ``a[i-1:, :]``: a view whatever the memory order,
    and its last column is the partial one Fortran's storage has. Any other
    leading extent folds the tail in column-major order onto as many whole
    columns as the storage holds."""
    at = int(start)
    extents = [int(e) for e in leading]
    if len(extents) == 1 and np.ndim(arr) == 2 and extents[0] == np.shape(arr)[0] > 0:
        return arr[at % extents[0] :, at // extents[0] :]
    flat = np.ravel(arr, order="F")[at:]
    if not extents:
        return flat
    block = 1
    for extent in extents:
        block *= extent
    whole = (flat.size // block) * block if block else 0
    return np.reshape(flat[:whole], (*extents, -1), order="F")


def _f_seq_tail_out(dst: Any, start: Any, src: Any) -> None:
    """Copy a callee's returned assumed-size array back onto the storage
    ``_f_seq_tail(dst, start, ...)`` handed it.

    Where the tail was a view the callee wrote the caller's array in place
    and there is nothing to do; where ``dst`` is not Fortran-contiguous the
    tail was a copy, and the callee's writes reach ``dst`` only through
    here, at the same column-major positions."""
    if dst is None or not isinstance(src, np.ndarray):
        return
    if np.may_share_memory(src, dst):
        return
    values = np.ravel(src, order="F")
    flat = np.ravel(dst, order="F")
    at = int(start)
    n = min(values.size, max(flat.size - at, 0))
    if n <= 0:
        return
    flat[at : at + n] = values[:n]
    if not np.may_share_memory(flat, dst):
        dst[...] = np.reshape(flat, np.shape(dst), order="F")


def _f_rstep(lo: Any, hi: Any, st: Any) -> Any:
    """Fortran lo:hi:st (st<0, inclusive, 1-based) -> python slice; the
    exclusive stop edge underflows at hi==1, which needs None."""
    return slice(lo - 1, hi - 2 if hi >= 2 else None, st)


def _f_rstep_lb(lo: Any, hi: Any, st: Any, lb: Any) -> Any:
    """Fortran lo:hi:st (st<0, inclusive) with declared lower bound lb.

    Either edge may be None: Fortran lets a section leave one implied."""
    start = None if lo is None else lo - lb
    stop = None
    if hi is not None:
        index = hi - lb
        stop = index - 1 if index >= 1 else None
    return slice(start, stop, st)


def _f_vdot(a: Any, b: Any) -> Any:
    """Fortran DOT_PRODUCT accumulates in order; np.dot (BLAS/pairwise)
    rounds differently."""
    if _LIBM_STRICT:
        s = 0.0
        # Unchecked on purpose: this is the emitted runtime, and it has
        # been through bit-exact gates in this form. A length mismatch is
        # invalid Fortran that never reaches here.
        for x, y in zip(np.ravel(a), np.ravel(b)):  # noqa: B905
            s += x * y
        return s
    return np.dot(a, b)


def _f_vsum(a: Any, axis: Any = None) -> Any:
    """Fortran SUM accumulates in element order; np.sum pairs terms and
    rounds differently (CLUBB's vertical_integral: 12 ULP)."""
    if _LIBM_STRICT:
        arr = np.asarray(a)
        if axis is None:
            s = arr.dtype.type(0) if arr.dtype.kind in "fc" else 0
            for x in np.ravel(arr, order="F"):
                s = s + x
            return s
        out = np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=arr.dtype)
        for i in range(arr.shape[axis]):
            out = out + np.take(arr, i, axis=axis)
        return out
    return np.sum(a, axis=axis)


def _fstr_eq(a: str, b: str) -> bool:
    """Fortran character equality: pad shorter operand with blanks."""
    return a.rstrip(" ") == b.rstrip(" ")


class _new_derived:  # noqa: N801  (the emitted name; not a class the engine exposes)
    """Fortran derived-type local: attribute container (components are
    attached by translated allocate statements)."""

    pass


def _copy_derived(obj: Any) -> Any:
    """Fortran derived-type assignment is a DEEP copy (incl. array
    components); python name binding is not."""
    import copy

    return copy.deepcopy(obj)


def _f_trim(s: str) -> str:
    """Fortran TRIM: strip trailing blanks only."""
    return s.rstrip(" ")


def _f_len_trim(s: str) -> int:
    return len(s.rstrip(" "))


def _f_adjustl(s: str) -> str:
    """Fortran ADJUSTL keeps length (pads right)."""
    return s.lstrip(" ").ljust(len(s))


def _f_sqrt(x: Any) -> Any:
    """Fortran SQRT of a negative real is a NaN, not an exception.

    ``math.sqrt`` raises ValueError there, which turns a number the Fortran
    would have carried on computing with into a crash -- and a crash the
    differential gate reports as "the candidate raised" rather than as the
    NaN both sides produce. Non-negative arguments go through ``math.sqrt``
    unchanged: the correctly-rounded hardware square root, bit for bit what
    the compiled reference does.
    """
    return math.sqrt(x) if x >= 0.0 else float("nan")


def _f_log(x: Any) -> Any:
    """Fortran LOG outside its domain is an IEEE value, not an exception.

    ``math.log`` raises ValueError at zero and below, where the compiled
    reference carries on with ``-Infinity`` and ``NaN`` -- and where the body
    that reached it usually clamps the result a line later, as
    ``iixexp`` does with an index off an exponential mesh. Raising there
    turns a number both sides agree on into "the candidate raised", which
    the differential gate reports as no comparison at all. Positive
    arguments go through ``math.log`` unchanged, which is the compiled
    reference's own libm call.
    """
    if x > 0.0:
        return math.log(x)
    return float("-inf") if x == 0.0 else float("nan")


def _f_log10(x: Any) -> Any:
    """Fortran LOG10 outside its domain, for the reason ``_f_log`` gives."""
    if x > 0.0:
        return math.log10(x)
    return float("-inf") if x == 0.0 else float("nan")


_INT32_LIMIT = 2147483648.0
"""One past the largest default INTEGER, as a float: the conversion's edge."""


def _f_int(x: Any, kind: Any = None) -> Any:
    """Fortran INT: truncate toward zero, into an integer of the given kind.

    Python's ``int`` raises on a NaN and grows without bound on a value no
    INTEGER can hold; the compiled reference does neither. gfortran emits the
    hardware conversion, which answers every value it cannot represent --
    NaN and overflow alike -- with the most negative integer of the kind, and
    a body that has just taken ``LOG`` of something non-positive is exactly
    where that happens. ``iixexp`` then clamps the index to ``1``, on both
    sides, which is the number the comparison is about.
    """
    dtype = np.int64 if int(kind or 4) == 8 else np.int32
    limit = _INT32_LIMIT * (2**32 if dtype is np.int64 else 1)
    if isinstance(x, (int, np.integer)):
        return dtype(x)
    value = float(x)
    if not -limit <= value < limit:  # NaN compares false and lands here too
        return dtype(-limit)
    return dtype(int(value))


def _f_min(*xs: Any) -> Any:
    """gfortran MIN, as the f2py reference actually computes it (measured
    against the built module, not a standalone toy): a NaN operand is
    absorbed and the other returned wherever it falls -- ``min(NaN, x)`` and
    ``min(x, NaN)`` are both ``x`` -- and the result is NaN only when every
    operand is. This is IEEE ``fmin`` order. gfortran at the golden ``-O1
    -fno-fast-math`` flags emits the comparison with the *computed* operand in
    the position that makes the constant win, so a body reaching ``min(1.0_wp,
    x)`` with ``x`` gone NaN keeps the 1.0 -- a comparison model
    (``(a<b)?a:b``) would let that NaN through and mismatch the reference,
    which is what the earlier "propagate on the right" rule did. Python's
    builtin ``min`` returns its first argument on a NaN, a different trap
    again."""
    r = xs[0]
    for b in xs[1:]:
        if r != r:
            r = b
        elif b == b and b < r:
            r = b
    return r


def _f_max(*xs: Any) -> Any:
    r = xs[0]
    for b in xs[1:]:
        if r != r:
            r = b
        elif b == b and b > r:
            r = b
    return r


def _f_vmin(a: Any, b: Any) -> Any:
    """elementwise gfortran MIN semantics (see _f_min): a NaN in either
    operand is absorbed, so ``fmin`` rather than a comparison that would let
    an operand's NaN through."""
    return np.fmin(a, b)


def _f_vmax(a: Any, b: Any) -> Any:
    return np.fmax(a, b)


def _f_nint(x: Any) -> Any:
    """Fortran NINT: round half away from zero (not banker's rounding).
    Python round() uses banker's rounding: round(0.5)=0, round(2.5)=2.
    Fortran NINT: NINT(0.5)=1, NINT(2.5)=3."""
    if isinstance(x, (float, np.floating)):
        return np.int32(math.floor(x + 0.5)) if x >= 0 else np.int32(math.ceil(x - 0.5))
    return np.int32(x)


def _f_sign(a: Any, b: Any) -> Any:
    """Fortran SIGN(a,b). Real b: IEEE copysign (gfortran distinguishes
    -0.0 -> -|a|). Integer b: value compare, b == 0 -> +|a| (differs from
    copysign there — the classic port trap)."""
    if isinstance(b, (float, np.floating)):
        return math.copysign(abs(a), b)
    return abs(a) if b >= 0 else -abs(a)


def _f_mod(a: Any, p: Any) -> Any:
    """Fortran MOD(a,p) = a - int(a/p)*p (truncated, sign follows a).
    Python % floors (sign follows p) — not equivalent for negatives."""
    return a - int(a / p) * p


def _f_int_div(a: Any, b: Any) -> Any:
    """Fortran integer division truncates toward zero; Python // floors."""
    return int(a / b)


def _f_io_values(items: Any) -> list[Any]:
    """An output item list, one value per element.

    A whole array is as many values as it has elements -- Fortran writes them
    all, and repeats the format over them -- so an item that is an array is
    flattened here rather than handed to a formatter that would ask an array
    for its single float.
    """
    values: list[Any] = []
    for item in items:
        if isinstance(item, np.ndarray) and item.ndim:
            values.extend(np.ravel(item).tolist())
        else:
            values.append(item)
    return values


def _f_list_write(*items: Any) -> Any:
    """gfortran list-directed WRITE: the record it produces, item by item.

    Measured against gfortran, because the point of this function is to
    reproduce another language's output byte for byte. An ``integer`` is a
    12-column right-justified field; a ``real(8)`` is 26 columns (a G25.17E3
    field and the blank that separates it from the next); a ``character`` is a
    blank and then its own characters; a ``logical`` is a blank and ``T`` or
    ``F``. Nothing is prepended to the record: the leading blank a
    list-directed record famously starts with is the first field's own
    padding, and adding one as well put every record a column out.

    Percent formatting throughout, deliberately: ``%`` is the spelling whose
    width, precision and sign rules match the Fortran edit descriptors this
    is emulating, and restating them in ``format`` would be a re-derivation
    of something already validated against real output.
    """
    out = ""
    for it in _f_io_values(items):
        if isinstance(it, str):
            out += " " + it
        elif isinstance(it, (bool, np.bool_)):
            out += " T" if it else " F"
        elif isinstance(it, (int, np.integer)):
            out += "%12d" % int(it)  # noqa: UP031
        else:
            out += _f_list_real(float(it))
    return out


def _f_list_real(v: float) -> str:
    """One ``real(8)`` of a list-directed record.

    The F form is right-justified in 21 columns with five blanks after it;
    the E form is right-justified in 26. How many decimals the F form carries
    is decided by the digits before the point -- seventeen significant
    figures in all -- and zero has one such digit, which is why ``0.0`` comes
    out with sixteen decimals where ``0.5`` has seventeen.
    """
    av = abs(v)
    if v == 0.0 or (0.1 <= av < 1e17):
        int_digits = 1 if v == 0.0 else (0 if av < 1.0 else len(str(int(av))))
        return ("%.*f" % (17 - int_digits, v)).rjust(21) + " " * 5  # noqa: UP031
    mant, ex = ("%.16E" % v).split("E")  # noqa: UP031
    return ("%sE%+04d" % (mant, int(ex))).rjust(26)  # noqa: UP031


_FMT_TOKEN = _re.compile(
    r"\s*(?:(?P<rep>\d+)?\s*(?P<ed>I\d+(?:\.\d+)?|F\d+\.\d+"
    r"|E[SN]?\d+\.\d+(?:E\d+)?|G\d+\.\d+|A(?:\d+)?|L\d+|\d*X|/"
    r"|'[^']*'|\"[^\"]*\"))\s*,?",
    _re.I,
)


def _f_fmt_records(fmt: str, values: list[Any]) -> list[str]:
    """The records one formatted transfer writes, for the edit descriptors
    the corpus uses: ``Iw[.m]``, ``Fw.d``, ``Ew.d`` / ``ESw.d``, ``Gw.d``,
    ``A[w]``, ``Lw``, ``nX``, ``/``, literals, repeat counts. Fortran field
    semantics: right-justified, asterisks on overflow, ``Iw.m`` zero-filled
    to ``m`` digits, ``E`` as ``0.dddE+ee``.

    A *list* because a format shorter than its item list does not truncate:
    Fortran reverts to the start of the format and, in doing so, ends the
    record and begins another. ``write(u, '(3a1)') achar(pixel)`` on a
    four-component pixel writes two records, and a translation that wrote one
    would be a file the source never produced. ``/`` ends a record the same
    way.
    """
    body = fmt.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    records: list[str] = []
    out: list[str] = []
    pos = 0
    while True:
        if pos >= len(body):
            if not values:
                break
            if not body:
                raise ValueError(f"_f_fmt_write: format {fmt!r} has no edit descriptor")
            records.append("".join(out))  # format reversion ends the record
            out, pos = [], 0
            continue
        m = _FMT_TOKEN.match(body, pos)
        if not m or m.end() == pos:
            raise ValueError(f"_f_fmt_write: cannot parse {fmt!r}")
        pos = m.end()
        rep = int(m.group("rep")) if m.group("rep") else 1
        ed = m.group("ed")
        for _ in range(rep):
            u = ed.upper()
            if u.startswith(("'", '"')):
                out.append(ed[1:-1])
            elif u.endswith("X"):
                out.append(" " * (int(u[:-1]) if u[:-1] else 1))
            elif u == "/":
                records.append("".join(out))
                out = []
            else:
                if not values:
                    # A data edit descriptor with no value left: the transfer
                    # ends here, and whatever the format has after it is not
                    # written.
                    records.append("".join(out))
                    return records
                out.append(_fmt_one(u, values.pop(0)))
    records.append("".join(out))
    return records


def _f_fmt_write(fmt: str, *vals: Any) -> str:
    """Formatted internal WRITE (#16): every record the format produces, the
    record terminators between them spelled ``\n`` the way an internal write
    to a character array would hold them."""
    return "\n".join(_f_fmt_records(fmt, _f_io_values(vals)))


def _fmt_one(u: str, v: Any) -> str:
    def fit(s: str, w: int) -> str:
        # ``w = 0`` asks for the shortest field the value fits in -- what
        # ``(i0)`` and ``(f0.6)`` are written for; the value never overflows
        # a width it chooses itself.
        if w == 0:
            return s
        return s.rjust(w) if len(s) <= w else "*" * w

    if u[0] == "I":
        width, _, minimum = u[1:].partition(".")
        iv = int(v)
        digits = str(abs(iv))
        if minimum:
            digits = digits.rjust(int(minimum), "0")
        return fit(("-" if iv < 0 else "") + digits, int(width))
    if u[0] == "F":
        width, decimals = u[1:].split(".")
        s = f"{float(v):.{int(decimals)}f}"
        if s.startswith("0.") and len(s) > int(width):
            s = s[1:]
        elif s.startswith("-0.") and len(s) > int(width):
            s = "-" + s[2:]
        return fit(s, int(width))
    if u[0] == "E":
        sci = u.startswith("ES")
        spec = u[2:] if u.startswith(("ES", "EN")) else u[1:]
        wd, _, ee = spec.partition("E")
        w, d = (int(x) for x in wd.split("."))
        ew = int(ee) if ee else 2
        x = float(v)
        if x == 0.0:
            mant, exp = 0.0, 0
        else:
            exp = int(np.floor(np.log10(abs(x))))
            if sci:
                mant = round(x / 10.0**exp, d)
                if abs(mant) >= 10.0:
                    mant /= 10.0
                    exp += 1
            else:
                exp += 1
                mant = round(x / 10.0**exp, d)
                if abs(mant) >= 1.0:
                    mant /= 10.0
                    exp += 1
        s = f"{mant:.{d}f}E{'+' if exp >= 0 else '-'}{abs(exp):0{ew}d}"
        return fit(s, w)
    if u[0] == "G":
        width, decimals = u[1:].split(".")
        return fit(f"{float(v):.{int(decimals)}g}", int(width))
    if u[0] == "A":
        s = str(v)
        if len(u) > 1:
            w = int(u[1:])
            return s[:w] if len(s) >= w else s.rjust(w)
        return s
    if u[0] == "L":
        return fit("T" if bool(v) else "F", int(u[1:]))
    raise ValueError(f"_f_fmt_write: descriptor {u}")


def _f_unpack(vector: Any, mask: Any, field: Any) -> Any:
    """Fortran UNPACK: scatter vector elements into field where mask is True."""
    result = field.copy()
    result[mask] = vector[: np.count_nonzero(mask)]
    return result


def _f_eoshift(array: Any, shift: Any, axis: Any = 0) -> Any:
    """Fortran EOSHIFT: shift and fill with zeros (no wrap-around)."""
    result = np.zeros_like(array)
    n = array.shape[axis]
    s = int(shift)
    if s > 0:
        slc_src = [slice(None)] * array.ndim
        slc_dst = [slice(None)] * array.ndim
        slc_src[axis] = slice(s, None)
        slc_dst[axis] = slice(None, n - s)
        result[tuple(slc_dst)] = array[tuple(slc_src)]
    elif s < 0:
        slc_src = [slice(None)] * array.ndim
        slc_dst = [slice(None)] * array.ndim
        slc_src[axis] = slice(None, s)
        slc_dst[axis] = slice(-s, None)
        result[tuple(slc_dst)] = array[tuple(slc_src)]
    else:
        result[...] = array
    return result


def _f_index(string: str, substring: str) -> int:
    """Fortran INDEX: 1-based position, 0 if not found."""
    p = string.find(substring)
    return p + 1 if p >= 0 else 0


def _f_huge(x: Any) -> Any:
    """Fortran HUGE: largest representable value of same type."""
    if isinstance(x, (float, np.floating)):
        return np.finfo(np.float64).max
    return np.iinfo(np.int32).max


def _f_tiny(x: Any) -> Any:
    """Fortran TINY: smallest positive normalized value."""
    return np.finfo(np.float64).tiny


def _f_epsilon(x: Any) -> Any:
    """Fortran EPSILON: smallest difference from 1.0 of same type."""
    return np.finfo(np.float64).eps


def _f_modulo(a: Any, p: Any) -> Any:
    """Fortran MODULO(a,p): result has sign of p (floored). Python % semantics."""
    return a % p


def _int_bits(*vals: Any) -> int | None:
    """The widest integer KIND among the operands, in bits, or ``None`` when
    none carries a dtype -- a bare literal -- in which case the operation
    stays unbounded, as it always was.

    Fortran's bit intrinsics work on the operand's KIND, 32- or 64-bit two's
    complement; a Python int is unbounded and keeps bits Fortran drops, which
    put ``mt19937_64`` wrong past its first tempering step (#15)."""
    width = None
    for v in vals:
        dt = getattr(v, "dtype", None)
        if dt is not None and np.issubdtype(dt, np.integer):
            b = dt.itemsize * 8
            width = b if width is None else max(width, b)
    return width


def _wrap_signed(v: Any, bits: int | None) -> Any:
    if bits is None:
        return int(v)
    v = int(v) & ((1 << bits) - 1)
    if v >= (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _f_iand(a: Any, b: Any) -> Any:
    bits = _int_bits(a, b)
    m = (1 << bits) - 1 if bits is not None else -1
    return _wrap_signed((int(a) & m) & (int(b) & m), bits)


def _f_ior(a: Any, b: Any) -> Any:
    bits = _int_bits(a, b)
    m = (1 << bits) - 1 if bits is not None else -1
    return _wrap_signed((int(a) & m) | (int(b) & m), bits)


def _f_ieor(a: Any, b: Any) -> Any:
    bits = _int_bits(a, b)
    m = (1 << bits) - 1 if bits is not None else -1
    return _wrap_signed((int(a) & m) ^ (int(b) & m), bits)


def _f_ishft(i: Any, shift: Any) -> Any:
    """Fortran ISHFT: a LOGICAL shift (zero-fill), positive = left, negative
    = right, within the operand's bit width (#15)."""
    s = int(shift)
    bits = _int_bits(i)
    v = int(i)
    if bits is not None:
        v &= (1 << bits) - 1  # the unsigned view, for a logical shift
    r = (v << s) if s >= 0 else (v >> (-s))
    return _wrap_signed(r, bits)


def _f_scan(string: str, set_chars: str) -> int:
    """Fortran SCAN: 1-based index of first char in set, 0 if none."""
    for i, c in enumerate(string):
        if c in set_chars:
            return i + 1
    return 0


def _f_kind(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return 8
    if isinstance(x, (int, np.integer)):
        return 4
    return 1


def _f_precision(x: Any) -> Any:
    return 15


def _f_radix(x: Any) -> Any:
    """Fortran RADIX: the base of the model number system, 2 for every
    integer and IEEE real kind numpy has. A Python int, as RADIX is a
    default integer: ``base**l`` then stays exact where an int32 would
    wrap, and it widens to float64 the moment it meets one."""
    return 2


def _f_transfer(source: Any, mold: Any) -> Any:
    """Fortran TRANSFER: reinterpret bit pattern."""
    src = np.array(source)
    viewed = src.view(np.array(mold).dtype)
    if hasattr(mold, "__len__"):
        return viewed.reshape(np.shape(mold))
    return viewed.flat[0]


def _f_is_iostat_end(stat: int) -> bool:
    return stat < 0


def _f_lbound(arr: Any, dim: Any = None) -> Any:
    """Fortran LBOUND is always 1 for standard arrays."""
    if dim is not None:
        return 1
    return np.ones(arr.ndim, dtype=np.int32)


def _f_c_loc(x: Any) -> Any:
    return id(x)


def _f_dim(x: Any, y: Any) -> Any:
    """Fortran DIM(x,y) = max(x-y, 0)."""
    return max(x - y, 0)


def _f_mvbits(from_val: Any, frompos: Any, length: Any, to_val: Any, topos: Any) -> Any:
    """Fortran MVBITS: copy bits. Returns modified to_val."""
    mask = (1 << length) - 1
    bits = (int(from_val) >> int(frompos)) & mask
    to_int = int(to_val)
    to_int &= ~(mask << int(topos))
    to_int |= bits << int(topos)
    return type(to_val)(to_int)


def _f_verf(x: Any) -> Any:
    from scipy.special import erf as _sp_erf

    return _sp_erf(x)


def _f_verfc(x: Any) -> Any:
    from scipy.special import erfc as _sp_erfc

    return _sp_erfc(x)


class _FIntrinsicModule:
    """The public names of one Fortran intrinsic module, as a namespace.

    ``USE ISO_FORTRAN_ENV`` names a module the standard (or the compiler)
    provides rather than one sitting in the tree, so there is no companion to
    translate and nothing to import. Binding such a USE the way an ordinary
    one is bound emits ``import iso_fortran_env_numpy``, a module that can
    never exist, and the translated file fails at import before any number is
    wrong. The emitter binds it to one of the objects below instead, and
    because they are part of this runtime they are already in the generated
    file -- no import line is emitted for them at all.
    """

    def __init__(self, **names: Any) -> None:
        self.__dict__.update(names)


def _f_ieee_value(x: Any, cls: Any) -> Any:
    """``IEEE_VALUE(X, CLASS)``: the class constants are spelled as themselves."""
    return {
        "ieee_positive_inf": np.inf,
        "ieee_negative_inf": -np.inf,
        "ieee_quiet_nan": np.nan,
        "ieee_signaling_nan": np.nan,
        "ieee_positive_zero": 0.0,
        "ieee_negative_zero": -0.0,
    }[cls]


# The named constants of these modules are kind *numbers* -- ``real64`` is 8,
# not a dtype -- because that is what the source reads when it compares one
# (``if (kind(x) /= real64)``). The frontend has a table of its own mapping the
# same names to dtypes for a declaration; the two are different questions.
_iso_fortran_env = _FIntrinsicModule(
    int8=np.int32(1),
    int16=np.int32(2),
    int32=np.int32(4),
    int64=np.int32(8),
    real32=np.int32(4),
    real64=np.int32(8),
    real128=np.int32(16),
    input_unit=np.int32(5),
    output_unit=np.int32(6),
    error_unit=np.int32(0),
    iostat_end=np.int32(-1),
    iostat_eor=np.int32(-2),
    numeric_storage_size=np.int32(32),
    character_storage_size=np.int32(8),
    file_storage_size=np.int32(8),
)

# ``c_null_char`` and ``c_new_line`` are the characters, not the two-character
# escapes that spell them in source: ``C_NULL_CHAR`` is ``ACHAR(0)``, and a
# string terminated with a literal backslash-zero is not terminated at all.
_iso_c_binding = _FIntrinsicModule(
    c_int=np.int32(4),
    c_short=np.int32(2),
    c_long=np.int32(8),
    c_long_long=np.int32(8),
    c_size_t=np.int32(8),
    c_int8_t=np.int32(1),
    c_int16_t=np.int32(2),
    c_int32_t=np.int32(4),
    c_int64_t=np.int32(8),
    c_float=np.int32(4),
    c_double=np.int32(8),
    c_long_double=np.int32(16),
    c_float_complex=np.int32(4),
    c_double_complex=np.int32(8),
    c_bool=np.int32(1),
    c_char=np.int32(1),
    c_null_char=chr(0),
    c_new_line=chr(10),
    c_carriage_return=chr(13),
    c_horizontal_tab=chr(9),
    c_null_ptr=None,
    c_loc=_f_c_loc,
)

_ieee_arithmetic = _FIntrinsicModule(
    ieee_is_nan=np.isnan,
    ieee_is_finite=np.isfinite,
    ieee_is_negative=np.signbit,
    ieee_is_normal=lambda x: np.isfinite(x) & (x != 0.0),
    ieee_value=_f_ieee_value,
    ieee_support_datatype=lambda *_a: True,
    ieee_positive_inf="ieee_positive_inf",
    ieee_negative_inf="ieee_negative_inf",
    ieee_quiet_nan="ieee_quiet_nan",
    ieee_signaling_nan="ieee_signaling_nan",
    ieee_positive_zero="ieee_positive_zero",
    ieee_negative_zero="ieee_negative_zero",
)

_ieee_exceptions = _FIntrinsicModule()
_ieee_features = _FIntrinsicModule()

# The translated module is serial, so the OpenMP enquiries answer as the
# runtime library does outside a parallel region. A translation that reported
# more than one thread would be describing a program that is not running.
_omp_lib = _FIntrinsicModule(
    omp_get_num_threads=lambda: np.int32(1),
    omp_get_max_threads=lambda: np.int32(1),
    omp_get_thread_num=lambda: np.int32(0),
    omp_get_num_procs=lambda: np.int32(1),
    omp_in_parallel=lambda: False,
    omp_get_wtime=lambda: 0.0,
)
_omp_lib_kinds = _FIntrinsicModule()
_openacc = _FIntrinsicModule(acc_get_num_devices=lambda *_a: np.int32(0))


# -- external files ----------------------------------------------------------
#
# The I/O statements that write a variable -- READ, INQUIRE, OPEN's NEWUNIT=
# and IOSTAT= -- are translated rather than stubbed, because a stub drops
# those writes silently and leaves the variable at whatever it held. So is a
# WRITE to a unit the translation itself connected to a file, because for a
# subprogram whose only product is that file the stub leaves nothing at all.
# That needs a unit table, and the table needs a file position Fortran would
# recognise: between records after every advancing statement, inside one
# after a non-advancing transfer.
#
# Formatted sequential and stream access. Direct access, unformatted
# sequential records and namelists are refused by the emitter rather than
# approximated here.

_F_UNITS: dict[int, Any] = {}
"""Connected unit number -> its connection. Module state, as Fortran's is."""

_F_PRECONNECTED = (0, 5, 6)
"""stderr, stdin, stdout: connected before the program starts, so INQUIRE
reports them OPENED without anything having opened them."""


class _FConnection:
    """One connected external file, positioned the way Fortran positions one.

    ``record`` is the record the file is positioned *inside* -- what a
    non-advancing READ leaves behind. ``None`` means positioned between
    records, which is where every advancing statement leaves it. ``starts``
    is where each record read so far began, which is what BACKSPACE needs.
    """

    def __init__(self, unit: int, path: Any, handle: Any, form: str, access: str) -> None:
        self.unit = int(unit)
        self.path = path
        self.handle = handle
        self.form = form
        self.access = access
        self.record: Any = None
        self.column = 0
        self.starts: list[int] = []
        self.partial = False
        """Whether a non-advancing WRITE left the file inside a record.

        Fortran has no incomplete record: CLOSE (and the end of the program)
        terminates the one a ``advance='no'`` transfer left open, so a file
        whose last write was non-advancing still ends with a record
        terminator. ``saveppm``'s last pixel is written that way, and a
        translation that dropped the terminator would be one byte short of
        the file the source writes.
        """


def _f_default_form(access: str) -> str:
    """The FORM a connection has when OPEN did not say: what the standard
    says, which is unformatted for direct and stream access and formatted
    otherwise.

    Stream is the one worth spelling out. ``open(newunit=u, file=f,
    access='stream')`` with no FORM= is how a Fortran program reads a file
    byte by byte -- gfortran reports UNFORMATTED for it -- and reading those
    bytes as text records would take a PPM's pixels for a record.
    """
    return "unformatted" if access in ("direct", "stream") else "formatted"


def _f_newunit() -> int:
    """A unit number NEWUNIT= can hand out: negative, the way gfortran's is,
    so a routine scanning 10..999 with INQUIRE never collides with one."""
    n = -10
    while n in _F_UNITS:
        n -= 1
    return n


def _f_open(
    unit: Any = None,
    file: Any = None,
    status: str = "unknown",
    access: str = "sequential",
    form: Any = None,
    position: str = "asis",
    action: str = "readwrite",
    recl: Any = None,
    strict: bool = True,
) -> tuple[Any, Any]:
    """Fortran OPEN. ``unit=None`` is NEWUNIT=. Returns ``(iostat, unit)``.

    ``strict`` is False only where the source wrote IOSTAT=: a statement
    without it aborts the program in Fortran, so this raises there.
    """
    number = _f_newunit() if unit is None else int(unit)
    st, act, acc = str(status).lower(), str(action).lower(), str(access).lower()
    shape = str(form).lower() if form is not None else _f_default_form(acc)
    path = None if file is None else str(file)
    if path is None:
        return _f_io_error(5000, "OPEN without FILE= (STATUS='SCRATCH' is refused)", strict, number)
    exists = os.path.exists(path)
    if st == "old" and not exists:
        missing = f"OPEN(STATUS='OLD') on {path!r}, which does not exist"
        return _f_io_error(2, missing, strict, number)
    if st == "new" and exists:
        return _f_io_error(17, f"OPEN(STATUS='NEW') on {path!r}, which exists", strict, number)
    if act == "read":
        mode = "r"
    elif st in ("new", "replace") or not exists:
        mode = "w+"
    else:
        mode = "r+"
    try:
        # A formatted connection is text, but Fortran's characters are bytes:
        # a program that reads a PPM header with FORM='FORMATTED' and its
        # pixels through a second connection would hit a decode error on the
        # binary tail of the very first buffered read. latin-1 is the codec
        # that maps every byte to one character, and ``newline=""`` leaves the
        # record terminators as they lie, so POS= counts what the file holds.
        handle = (
            open(path, mode + "b")
            if shape == "unformatted"
            else open(path, mode, encoding="latin-1", newline="")
        )
    except OSError as error:
        return _f_io_error(error.errno or 5000, str(error), strict, number)
    if str(position).lower() == "append":
        handle.seek(0, 2)
    _f_close(number, strict=False)
    _F_UNITS[number] = _FConnection(number, path, handle, shape, acc)
    return np.int32(0), np.int32(number)


def _f_io_error(code: int, message: str, strict: bool, unit: Any = -1) -> tuple[Any, Any]:
    if strict:
        raise OSError(f"Fortran I/O error {code}: {message}")
    return np.int32(code), np.int32(unit)


def _f_close(unit: Any, status: Any = None, strict: bool = True) -> Any:
    """Fortran CLOSE. Closing a unit nothing connected is not an error."""
    conn = _F_UNITS.pop(int(unit), None)
    if conn is None:
        return np.int32(0)
    if conn.partial:
        conn.handle.write("\n")
    conn.handle.close()
    if str(status).lower() == "delete" and conn.path is not None:
        try:
            os.remove(conn.path)
        except OSError as error:
            return _f_io_error(error.errno or 5000, str(error), strict)[0]
    return np.int32(0)


def _f_connection(unit: Any, strict: bool) -> Any:
    conn = _F_UNITS.get(int(unit))
    if conn is None and strict:
        raise OSError(f"Fortran I/O error: unit {int(unit)} is not connected")
    return conn


def _f_rewind(unit: Any, strict: bool = True) -> Any:
    conn = _f_connection(unit, strict)
    if conn is None:
        return np.int32(5001)
    conn.handle.seek(0)
    conn.record, conn.column, conn.starts = None, 0, []
    return np.int32(0)


def _f_backspace(unit: Any, strict: bool = True) -> Any:
    """Fortran BACKSPACE: position before the record just read."""
    conn = _f_connection(unit, strict)
    if conn is None:
        return np.int32(5001)
    if conn.starts:
        conn.handle.seek(conn.starts.pop())
    conn.record, conn.column = None, 0
    return np.int32(0)


def _f_endfile(unit: Any, strict: bool = True) -> Any:
    conn = _f_connection(unit, strict)
    if conn is None:
        return np.int32(5001)
    conn.handle.truncate()
    conn.record, conn.column = None, 0
    return np.int32(0)


def _f_flush(unit: Any, strict: bool = True) -> Any:
    conn = _f_connection(unit, strict)
    if conn is None:
        return np.int32(5001)
    conn.handle.flush()
    return np.int32(0)


def _f_inquire(unit: Any, file: Any, what: str) -> Any:
    """One INQUIRE output specifier's value.

    One call per specifier, because every specifier is a write and the
    emitter renders each as its own assignment.
    """
    key = str(what).lower()
    conn = None
    path = None if file is None else str(file)
    if path is not None:
        conn = next((c for c in _F_UNITS.values() if c.path == path), None)
    elif unit is not None:
        conn = _F_UNITS.get(int(unit))
    connected = conn is not None or (
        path is None and unit is not None and int(unit) in _F_PRECONNECTED
    )
    if key == "opened":
        return bool(connected)
    if key == "exist":
        return bool(os.path.exists(path)) if path is not None else bool(connected)
    if key == "named":
        return bool(conn is not None and conn.path is not None)
    if key == "name":
        return conn.path if conn is not None and conn.path is not None else ""
    if key == "number":
        return np.int32(conn.unit if conn is not None else -1)
    if key == "size":
        target = path if path is not None else (conn.path if conn is not None else None)
        return np.int32(os.path.getsize(target) if target and os.path.exists(target) else -1)
    if key == "pos":
        return np.int32(conn.handle.tell() + 1 if conn is not None else -1)
    if key == "iostat":
        return np.int32(0)
    if key == "iomsg":
        return ""
    if key in (
        "form",
        "access",
        "action",
        "position",
        "sequential",
        "direct",
        "formatted",
        "unformatted",
        "recl",
        "nextrec",
    ):
        return _f_inquire_connection(conn, key)
    raise ValueError(f"_f_inquire: unsupported specifier {what!r}")


def _f_inquire_connection(conn: Any, key: str) -> Any:
    """The specifiers that describe *how* a unit is connected."""
    if key == "recl":
        return np.int32(-1)
    if key == "nextrec":
        return np.int32(0)
    if conn is None:
        return np.int32(-1) if key in ("recl", "nextrec") else "UNDEFINED"
    if key == "form":
        return conn.form.upper()
    if key == "access":
        return conn.access.upper()
    if key == "action":
        return "READWRITE"
    if key == "position":
        return "ASIS"
    if key == "sequential":
        return "YES" if conn.access == "sequential" else "NO"
    if key == "direct":
        return "YES" if conn.access == "direct" else "NO"
    if key == "formatted":
        return "YES" if conn.form == "formatted" else "NO"
    return "YES" if conn.form == "unformatted" else "NO"


def _f_print(fmt: Any, *items: Any) -> None:
    """PRINT: the record it writes to standard output, formatted the way
    gfortran formats one. Kept rather than stubbed because the item list is
    a read, and a stub told the read/write gate nothing was read."""
    print(_f_list_write(*items) if fmt is None else _f_fmt_write(str(fmt), *items))


def _f_write(unit: Any, fmt: Any = None, items: Any = (), advance: Any = "yes") -> Any:
    """Fortran WRITE to an external unit: the records it puts in the file.

    Translated rather than stubbed for a unit the program itself connected,
    because for a routine whose whole purpose is the file it produces --
    ``saveppm`` writes a PPM and returns nothing else -- a stub is not a
    lossy translation but an empty one, and there is nothing left for a
    differential to compare.

    A unit no OPEN in this translation connected is the log destination it
    always was: the preconnected standard output, or a diagnostic the source
    sends to a unit the caller connected. Those records are written where
    ``PRINT``'s go, so the item list is still read, and no file is invented
    for a connection this translation does not hold.

    ``advance='no'`` leaves the file inside the record, which is what the
    next WRITE then continues; every other transfer ends it.
    """
    values = _f_io_values(items)
    conn = _F_UNITS.get(int(unit))
    if conn is None:
        print(_f_list_write(*values) if fmt is None else _f_fmt_write(str(fmt), *values))
        return np.int32(0)
    if conn.form == "unformatted":
        conn.handle.write(_f_unformatted_bytes(values))
        return np.int32(0)
    records = [_f_list_write(*values)] if fmt is None else _f_fmt_records(str(fmt), values)
    text = "\n".join(records)
    nonadvancing = str(advance).strip().lower() == "no"
    if not nonadvancing:
        text += "\n"
    conn.handle.write(text)
    conn.partial = nonadvancing
    return np.int32(0)


def _f_unformatted_bytes(values: list[Any]) -> bytes:
    """One unformatted transfer's item list, as the bytes gfortran writes for
    a stream connection: a character is its own byte, a number its raw
    little-endian image. A LOGICAL is refused for the reason
    ``_f_read_stream`` refuses to read one -- four bytes to gfortran, one to
    NumPy."""
    out = bytearray()
    for value in values:
        if isinstance(value, str):
            out += value.encode("latin-1")
        elif isinstance(value, (bool, np.bool_)):
            raise OSError("Fortran I/O error: unformatted stream WRITE of bool")
        else:
            out += np.asarray(value).tobytes()
    return bytes(out)


_F_READ_DTYPES = {
    "float64": np.float64,
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}
"""Item dtype -> what a parsed field becomes. ``str`` is the field itself."""

_F_REPEAT = _re.compile(r"(\d+)\*(.*)")
"""``3*1.0`` in list-directed input: three values, not one."""


def _f_seek(conn: Any, pos: Any) -> None:
    """POS=: put the connection at the ``pos``th byte of the file, counting
    from one, the way Fortran counts a stream position. Whatever record the
    connection was positioned inside is left behind with it."""
    conn.handle.seek(max(int(pos) - 1, 0))
    conn.record, conn.column = None, 0


def _f_read(
    unit: Any,
    fmt: Any = None,
    items: Any = (),
    advance: str = "yes",
    pos: Any = None,
    strict: bool = True,
) -> tuple[Any, ...]:
    """Fortran READ from a connected unit.

    ``items`` is one ``(dtype, count, width)`` per input item: ``count`` is
    None for a scalar and the element count for an array item, ``width`` the
    declared character length. Returns ``(iostat, value, ...)`` in item
    order, so the emitter unpacks the statement's item list straight out of
    it -- which is the point of translating READ at all.
    """
    blanks = tuple(_f_read_blank(spec) for spec in items)
    conn = _F_UNITS.get(int(unit))
    if conn is None:
        if strict:
            raise OSError(f"Fortran I/O error: READ on unit {int(unit)}, which is not connected")
        return (np.int32(5002), *blanks)
    if pos is not None:
        _f_seek(conn, pos)
    if conn.form != "formatted":
        if conn.access != "stream":
            # Refused rather than approximated: an unformatted *record's*
            # layout -- the length markers around it -- is the compiler's, and
            # guessing at it would put wrong numbers in the right variables.
            # An unformatted stream has no records and nothing to guess: the
            # file is the values, laid end to end, which is why it is read
            # below rather than refused with them.
            raise OSError(f"Fortran I/O error: READ from unformatted unit {int(unit)}")
        values, ios = _f_read_stream(conn, items)
        if ios == 0:
            return (np.int32(0), *values)
        if strict:
            raise EOFError(f"Fortran I/O error {ios} reading unit {int(unit)}")
        return (np.int32(ios), *blanks)
    spelled = "*" if fmt is None else str(fmt).strip()
    if spelled == "*":
        values, ios = _f_read_list(conn, items)
    else:
        values, ios = _f_read_formatted(conn, spelled, items, str(advance).lower() == "no")
    if ios != 0:
        if strict:
            raise EOFError(f"Fortran I/O error {ios} reading unit {int(unit)}")
        return (np.int32(ios), *blanks)
    return (np.int32(0), *values)


def _f_read_blank(spec: Any) -> Any:
    """What an item keeps when the READ that would have written it failed."""
    dtype, count, _width = spec
    if count is not None:
        if dtype == "str":
            return np.array([""] * int(count), dtype=object)
        return np.zeros(int(count), dtype=_F_READ_DTYPES.get(dtype, np.float64))
    if dtype == "str":
        return ""
    return _F_READ_DTYPES.get(dtype, np.float64)(0)


def _f_read_stream(conn: Any, items: Any) -> tuple[Any, int]:
    """An unformatted stream READ: the items, taken from the file as bytes.

    No records and no length markers -- ``access='stream'`` with no FORM= is
    the connection a program reads a PPM's pixels through, one ``character``
    per byte -- so each item takes exactly the storage its type occupies.
    A short read is end of file, and leaves the items alone.
    """
    values: list[Any] = []
    for dtype, count, width in items:
        size = int(count) if count is not None else 1
        if dtype == "str":
            each = int(width) if width else 1
            raw = conn.handle.read(size * each)
            if len(raw) < size * each:
                return (), -1  # IOSTAT_END
            text = raw.decode("latin-1")
            taken: Any = [text[at * each : (at + 1) * each] for at in range(size)]
            if count is None:
                values.append(taken[0])
                continue
            values.append(np.array(taken, dtype=object))
            continue
        if dtype not in _F_READ_DTYPES or dtype == "bool":
            # A LOGICAL is four bytes to gfortran and one to NumPy; reading it
            # here would put the wrong bytes in the right variable.
            raise OSError(f"Fortran I/O error: unformatted stream READ of {dtype}")
        element = np.dtype(_F_READ_DTYPES[dtype])
        raw = conn.handle.read(size * element.itemsize)
        if len(raw) < size * element.itemsize:
            return (), -1
        parsed = np.frombuffer(raw, dtype=element)
        values.append(element.type(parsed[0]) if count is None else parsed.copy())
    return tuple(values), 0


def _f_next_record(conn: Any) -> Any:
    """The next record, or None at end of file."""
    start = conn.handle.tell()
    line = conn.handle.readline()
    if line == "":
        return None
    conn.starts.append(start)
    return line.rstrip("\n").rstrip("\r")


def _f_list_tokens(text: str) -> list[str]:
    """One record's list-directed values. Blanks and commas separate them,
    quotes group them, ``r*v`` repeats one."""
    tokens: list[str] = []
    current, quote = "", ""
    for ch in text:
        if quote:
            if ch == quote:
                tokens.append(current)
                current, quote = "", ""
            else:
                current += ch
        elif ch in "'\"":
            quote = ch
        elif ch in " \t,":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    expanded: list[str] = []
    for token in tokens:
        repeat = _F_REPEAT.fullmatch(token)
        if repeat and repeat.group(2):
            expanded.extend([repeat.group(2)] * int(repeat.group(1)))
        else:
            expanded.append(token)
    return expanded


def _f_read_list(conn: Any, items: Any) -> tuple[Any, int]:
    """A list-directed READ: values, across as many records as it takes."""
    needed = sum(1 if count is None else int(count) for _d, count, _w in items)
    tokens: list[str] = []
    while len(tokens) < needed:
        if conn.record is None:
            record = _f_next_record(conn)
            if record is None:
                return (), -1  # IOSTAT_END
            conn.record, conn.column = record, 0
        tokens.extend(_f_list_tokens(conn.record[conn.column :]))
        conn.column = len(conn.record)
        if len(tokens) < needed:
            conn.record = None  # the item list is not satisfied: read on
    conn.record, conn.column = None, 0  # a list-directed READ advances
    return _f_group(items, tokens[:needed]), 0


def _f_read_formatted(conn: Any, fmt: str, items: Any, nonadvancing: bool) -> tuple[Any, int]:
    """A formatted READ, over the edit descriptors ``_f_fmt_write`` writes."""
    body = fmt.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    slots = [
        (dtype, width)
        for dtype, count, width in items
        for _ in range(1 if count is None else int(count))
    ]
    if conn.record is None:
        record = _f_next_record(conn)
        if record is None:
            return (), -1
        conn.record, conn.column = record, 0
    values: list[Any] = []
    pos = 0
    while len(values) < len(slots):
        if pos >= len(body):
            # The format is exhausted before the item list: Fortran starts it
            # again on the next record.
            record = _f_next_record(conn)
            if record is None:
                return (), -1
            conn.record, conn.column, pos = record, 0, 0
            if not body:
                raise ValueError(f"_f_read: format {fmt!r} has no edit descriptor")
        match = _FMT_TOKEN.match(body, pos)
        if not match or match.end() == pos:
            raise ValueError(f"_f_read: cannot parse {fmt!r}")
        pos = match.end()
        repeat = int(match.group("rep")) if match.group("rep") else 1
        edit = match.group("ed")
        upper = edit.upper()
        for _ in range(repeat):
            if upper.startswith(("'", '"')):
                conn.column += len(edit) - 2
            elif upper.endswith("X"):
                conn.column += int(upper[:-1]) if upper[:-1] else 1
            elif upper == "/":
                record = _f_next_record(conn)
                if record is None:
                    return (), -1
                conn.record, conn.column = record, 0
            elif len(values) < len(slots):
                if nonadvancing and conn.column >= len(conn.record):
                    # End of record: the file is positioned after it, and the
                    # loop that reads a record character by character stops
                    # here rather than running into the next one.
                    conn.record, conn.column = None, 0
                    return (), -2  # IOSTAT_EOR
                values.append(_f_take_field(conn, upper, slots[len(values)][1]))
    if not nonadvancing:
        conn.record, conn.column = None, 0
    return _f_group(items, values), 0


def _f_take_field(conn: Any, upper: str, width: Any) -> str:
    """The characters one data edit descriptor consumes, blank-padded (the
    default PAD='YES') when the record ends inside the field."""
    if upper[0] == "A":
        rest = len(conn.record) - conn.column
        size = int(upper[1:]) if len(upper) > 1 else (int(width) if width else rest)
    else:
        spec = upper[2:] if upper.startswith(("ES", "EN")) else upper[1:]
        size = int(spec.split(".")[0].split("E")[0])
    field = str(conn.record[conn.column : conn.column + size])
    conn.column += size
    return field.ljust(size)


def _f_read_value(field: str, dtype: Any) -> Any:
    """One field, as the value its item's declared type gives it."""
    text = field.strip()
    if dtype == "str":
        return field
    if dtype == "bool":
        return np.bool_(text[:1].upper() == "T" or text[:2].upper() == ".T")
    if not text:
        return _F_READ_DTYPES.get(dtype, np.float64)(0)
    if dtype in ("int32", "int64"):
        return _F_READ_DTYPES[dtype](int(float(text.replace("d", "e").replace("D", "e"))))
    return _F_READ_DTYPES.get(dtype, np.float64)(float(text.replace("d", "e").replace("D", "e")))


def _f_group(items: Any, fields: Any) -> tuple[Any, ...]:
    """The fields read, grouped back onto the items that asked for them."""
    grouped: list[Any] = []
    at = 0
    for dtype, count, _width in items:
        if count is None:
            grouped.append(_f_read_value(fields[at], dtype))
            at += 1
        else:
            size = int(count)
            taken = [_f_read_value(f, dtype) for f in fields[at : at + size]]
            at += size
            grouped.append(
                np.array(taken, dtype=object if dtype == "str" else _F_READ_DTYPES[dtype])
            )
    return tuple(grouped)


def emit() -> str:
    """This module's runtime definitions, as source text for a generated file.

    Everything from ``_LIBM_STRICT`` onwards, minus ``emit`` itself. Read out
    of the live module rather than kept as a second copy, so the code that is
    tested and the code that ships cannot drift apart.
    """
    module = inspect.getmodule(emit)
    assert module is not None
    source = inspect.getsource(module)
    return source[source.index("_LIBM_STRICT =") : source.index("def emit()")].rstrip() + "\n"
