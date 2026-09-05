"""The ``_f_*`` runtime a jaxized module needs, in JAX rather than NumPy.

Migrated from ``13_jax_backend/jax_shim.py``, behaviour intact and layout
reformatted to this repository's style, so a diff against the origin is not
what holds it: that is done by the anchor check in
``tests/test_jax_backend.py`` and, for the numbers, by the ULP gate every
ported kernel has to pass. Counterpart of the
shim library ``recast.transform.numpy.runtime`` inlines into every translated
module, and deliberately not identical to it:

  - **no strict libm.** ``jnp.exp``/``log``/``power`` lower to XLA's own
    implementations, which differ from glibc by ULPs. That is why this backend
    gates at the ULP tier and never at bit-exactness.
  - ``_f_min``/``_f_max`` reproduce the gfortran SSE ``minsd``/``maxsd`` NaN
    order exactly -- the left operand's NaN absorbed, the right's propagated --
    which matches the NumPy shim bit for bit on every non-transcendental path.

Its twenty anchors are a strict subset of the NumPy runtime's forty-four, and
that is the property worth keeping: two backends held to one set of anchors
rather than drifting into separate notions of correct. The twenty-four it does
not implement are string, bit and pointer intrinsics a numeric kernel does not
reach.

Importing this enables float64, which JAX does not do by default and which has
to happen before any array is created. That is also why nothing in the engine
imports this module: ``backend.emit_runtime`` reads its text off disk instead,
so emitting JAX code never requires JAX to be installed.
"""

import re as _re
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

# A star-import from the generated module must see the underscore names.
__all__ = [
    "_f_adjustl",
    "_f_cfold",
    "_f_concrete",
    "_f_dim",
    "_f_ecall",
    "_f_epsilon",
    "_f_fmt_write",
    "_f_fori",
    "_f_huge",
    "_f_int_div",
    "_f_len_trim",
    "_f_list_write",
    "_f_max",
    "_f_min",
    "_f_mod",
    "_f_modulo",
    "_f_nint",
    "_f_pyfloat",
    "_f_pyint",
    "_f_pymax",
    "_f_pymin",
    "_f_rstep",
    "_f_rstep_lb",
    "_f_sign",
    "_f_sqrt",
    "_f_tiny",
    "_f_trim",
    "_f_trips",
    "_f_vceil",
    "_f_vdot",
    "_f_verf",
    "_f_verfc",
    "_f_vexp",
    "_f_vfloor",
    "_f_vlog",
    "_f_vlog10",
    "_f_vmax",
    "_f_vmin",
    "_f_vpow",
    "_f_vsum",
    "_fstr_eq",
    "jax",
    "jnp",
    "lax",
]


def _f_min(*xs):
    """gfortran MIN, SSE minsd fold order: per step the FIRST operand's
    NaN is absorbed, the second's propagates (x != x is the NaN test —
    valid for int operands too, unlike jnp.isnan)."""
    r = xs[0]
    for b in xs[1:]:
        r = jnp.where(r != r, b, jnp.where(r < b, r, b))
    return r


def _f_max(*xs):
    r = xs[0]
    for b in xs[1:]:
        r = jnp.where(r != r, b, jnp.where(r > b, r, b))
    return r


def _f_vmin(a, b):
    return jnp.where(jnp.isnan(a), b, jnp.where(a < b, a, b))


def _f_vmax(a, b):
    return jnp.where(jnp.isnan(a), b, jnp.where(a > b, a, b))


def _f_sign(a, b):
    """Fortran SIGN(a,b): real b -> copysign (-0.0 aware); integer b ->
    value compare with b == 0 giving +|a|. dtype dispatch is static at
    trace time."""
    b_ = jnp.asarray(b)
    if jnp.issubdtype(b_.dtype, jnp.floating):
        return jnp.copysign(jnp.abs(a), b_)
    return jnp.where(b_ >= 0, jnp.abs(a), -jnp.abs(a))


def _f_dim(x, y):
    """Fortran DIM(x,y) = max(x-y, 0)."""
    return jnp.maximum(jnp.asarray(x) - y, 0)


def _f_mod(a, p):
    """Fortran MOD: truncated, sign follows a."""
    a_, p_ = jnp.asarray(a), jnp.asarray(p)
    q = jnp.trunc(a_ / p_).astype(jnp.result_type(a_, p_))
    return a_ - q * p_


def _f_modulo(a, p):
    """Fortran MODULO: floored, sign follows p."""
    return jnp.mod(jnp.asarray(a), p)


def _f_int_div(a, b):
    """Fortran integer division truncates toward zero."""
    a_, b_ = jnp.asarray(a), jnp.asarray(b)
    return jnp.trunc(a_ / b_).astype(jnp.result_type(a_, b_))


def _f_nint(x):
    """Fortran NINT: round half away from zero."""
    x_ = jnp.asarray(x)
    r = jnp.where(x_ >= 0, jnp.floor(x_ + 0.5), jnp.ceil(x_ - 0.5))
    return r.astype(jnp.int32)


def _f_vexp(x):
    return jnp.exp(x)


def _f_vlog(x):
    return jnp.log(x)


def _f_vlog10(x):
    return jnp.log10(x)


def _f_vpow(a, b):
    return jnp.asarray(a) ** b


def _f_concrete(*values):
    """Whether every value is known at trace time -- a Python or NumPy
    scalar, or a concrete array -- rather than a tracer. A branch over a
    kernel's static scalar argument is a Python ``if`` when the kernel runs
    through its jit wrapper and a ``lax.cond`` when another kernel's traced
    body calls its implementation; this decides which, over the leaves of
    the test (its comparisons and names), since the test itself is spelled
    with Python logic the Python ``if`` evaluates."""
    return not any(isinstance(x, jax.core.Tracer) for x in values)


def _f_fori(lo, hi, body, init):
    """``lax.fori_loop`` unless the trip count is static and empty.

    A Fortran DO over an array's zero-extent axis (CLUBB's scalar tracers
    under ``sclr_dim = 0``) runs no iteration; ``fori_loop`` would still
    trace the body once, and JAX refuses any index into a size-0 axis at
    trace time. A dynamic bound is left to ``fori_loop``.
    """
    static = (int, np.integer)
    if isinstance(lo, static) and isinstance(hi, static):
        if int(hi) <= int(lo):
            return init
        # A static trip count: a scan over the indices (reverse-
        # differentiable, as fori_loop's own scan form is), spelled int32
        # -- Fortran's default integer, the dtype every integer local and
        # dummy carries, so a store of the index into one keeps its dtype
        # across a lax.cond. fori_loop's scan form would count in the
        # default int, int64 under x64.
        indices = jnp.arange(int(lo), int(hi), dtype=jnp.int32)

        def step(carry, i):
            return body(i, carry), None

        carry, _ = lax.scan(step, init, indices)
        return carry
    return lax.fori_loop(lo, hi, body, init)


def _f_pyint(x):
    """A trace-time integer as a Python int (a NumPy int32 from the gate,
    a Python int from the jit wrapper: one kind, whatever the spelling);
    a tracer, when a traced body reached the kernel, stays one."""
    if isinstance(x, jax.core.Tracer):
        return x
    return int(x)


def _f_pyfloat(x):
    """The same for a trace-time real."""
    if isinstance(x, jax.core.Tracer):
        return x
    return float(x)


def _f_pymax(*values):
    """Python's ``max`` where every value is a Python or NumPy scalar (a
    static bound stays a Python int), ``jnp.maximum`` folded otherwise."""
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return max(values)
    out = values[0]
    for v in values[1:]:
        out = jnp.maximum(out, v)
    return out


def _f_pymin(*values):
    """Python's ``min`` the same way, ``jnp.minimum`` folded."""
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return min(values)
    out = values[0]
    for v in values[1:]:
        out = jnp.minimum(out, v)
    return out


def _f_trips(lo, hi, step):
    """How many times ``range(lo, hi, step)`` runs -- Fortran's DO trip
    count, ``max(0, (hi - lo + step - sign(step)) // step)`` -- as a
    Python int when every bound is, else traced."""
    static = (int, np.integer)
    if isinstance(lo, static) and isinstance(hi, static) and isinstance(step, static):
        return len(range(int(lo), int(hi), int(step)))
    return jnp.maximum(0, (hi - lo + step - jnp.sign(step)) // step)


def _f_ecall(fn, *args, **kw):
    """ELEMENTAL procedure broadcast over array actuals: the scalar kernel
    per element, in sequence, as the NumPy runtime's np.vectorize runs it.

    Not jnp.vectorize: under jit that is a vmap, which turns the kernel's
    branches into selects and its loops into batched ones, and CLUBB's
    hybrid PDF closure came out 1e7 ULP from the anchor that way (2 ULP
    this way). lax.map keeps each element's own control flow."""
    arrays = [jnp.asarray(a) for a in args]
    shape = jnp.broadcast_shapes(*[a.shape for a in arrays])
    flat = tuple(jnp.broadcast_to(a, shape).reshape(-1) for a in arrays)
    outs = lax.map(lambda xs: fn(*xs, **kw), flat)
    return jax.tree_util.tree_map(lambda o: o.reshape(shape), outs)


def _f_sqrt(x):
    """Fortran SQRT: a NaN for a negative real, not an exception, and the
    correctly rounded root otherwise -- what the NumPy shim does with
    math.sqrt, and what jnp.sqrt does by itself."""
    return jnp.sqrt(x)


def _f_vsum(a, axis=None):
    """Fortran SUM accumulates in element order; a sequential fori_loop
    keeps the fold order the NumPy anchor uses (its ``_f_vsum``), so the
    two sides differ by XLA's rounding alone and not by association."""
    arr = jnp.asarray(a)
    if axis is None:
        flat = jnp.ravel(arr, order="F")

        def body(i, s):
            return s + flat[i]

        return lax.fori_loop(0, flat.shape[0], body, jnp.zeros((), dtype=arr.dtype))
    moved = jnp.moveaxis(arr, axis, 0)

    def body_axis(i, s):
        return s + moved[i]

    return lax.fori_loop(0, moved.shape[0], body_axis, jnp.zeros(moved.shape[1:], dtype=arr.dtype))


def _f_verf(x):
    from jax.scipy.special import erf as _erf

    return _erf(x)


def _f_verfc(x):
    from jax.scipy.special import erfc as _erfc

    return _erfc(x)


def _f_vdot(a, b):
    """Fortran DOT_PRODUCT accumulates in order; sequential fori_loop
    keeps the fold order (XLA may still contract the FMA)."""
    af, bf = jnp.ravel(a), jnp.ravel(b)

    def body(i, s):
        return s + af[i] * bf[i]

    return lax.fori_loop(0, af.shape[0], body, jnp.float64(0.0))


def _f_vceil(x):
    return jnp.ceil(x).astype(jnp.int32)


def _f_vfloor(x):
    return jnp.floor(x).astype(jnp.int32)


def _f_huge(x):
    d = jnp.asarray(x).dtype
    if jnp.issubdtype(d, jnp.floating):
        return jnp.finfo(d).max
    return jnp.iinfo(d).max


def _f_tiny(x):
    return jnp.finfo(jnp.asarray(x).dtype).tiny


def _f_epsilon(x):
    return jnp.finfo(jnp.asarray(x).dtype).eps


def _fstr_eq(a: str, b: str) -> bool:
    """Fortran character equality: pad the shorter operand with blanks.
    Characters are static under tracing -- plain Python strings."""
    return a.rstrip(" ") == b.rstrip(" ")


def _f_trim(s: str) -> str:
    """Fortran TRIM: strip trailing blanks only."""
    return s.rstrip(" ")


def _f_len_trim(s: str) -> int:
    return len(s.rstrip(" "))


def _f_adjustl(s: str) -> str:
    return s.lstrip(" ").ljust(len(s))


# -- the NumPy runtime's shims a kernel body still reaches -------------------
# Character and constant-folding shims are pure Python at trace time; the
# section shims build Python slices from static bounds. A tracer handed to a
# writer (a statistics name written from a traced loop index) prints as a
# placeholder: with statistics off nothing reads the name.


def _traced_placeholder(items):
    return tuple("<traced>" if isinstance(it, jax.core.Tracer) else it for it in items)


def _f_cfold(fn: Any, *args: Any) -> Any:
    """gfortran evaluates constant-argument intrinsics at COMPILE time
    with MPFR (correctly rounded) — that value matches no runtime libm
    (proven: gamma(1.8) differs from BOTH libgfortran and glibc)."""
    import mpmath as mp

    with mp.workprec(200):
        return float(getattr(mp, fn)(*[mp.mpf(float(a)) for a in args]))


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


def _f_list_write(*items: Any) -> Any:
    """gfortran list-directed internal WRITE shim.

    Byte-exact against reference probes: a record starts with one blank,
    strings print verbatim, ``int32`` becomes I12 plus a blank separator,
    ``real(8)`` becomes G25.17E3 plus a blank.

    Percent formatting throughout, deliberately. The point of this function
    is to reproduce another language's output byte for byte, and ``%`` is
    the spelling whose width, precision and sign rules match the Fortran
    edit descriptors it is emulating. Restating them in ``format`` would be
    a re-derivation of something already validated against real output.
    """
    out = " "
    for it in _traced_placeholder(items):
        if isinstance(it, str):
            out += it
        elif isinstance(it, (int, np.integer)):
            out += "%12d " % int(it)  # noqa: UP031
        else:
            v = float(it)
            av = abs(v)
            if v == 0.0 or (0.1 <= av < 1e17):
                int_digits = 0 if av < 1.0 else len(str(int(av)))
                out += "%21.*f" % (17 - int_digits, v) + " " * 6  # noqa: UP031
            else:
                mant, ex = ("%.16E" % v).split("E")  # noqa: UP031
                out += ("%sE%+04d" % (mant, int(ex))).rjust(26) + " "  # noqa: UP031
    return out


_FMT_TOKEN = _re.compile(
    r"\s*(?:(?P<rep>\d+)?\s*(?P<ed>I\d+(?:\.\d+)?|F\d+\.\d+"
    r"|E[SN]?\d+\.\d+(?:E\d+)?|G\d+\.\d+|A(?:\d+)?|L\d+|\d*X|/"
    r"|'[^']*'|\"[^\"]*\"))\s*,?",
    _re.I,
)


def _f_fmt_write(fmt: str, *vals: Any) -> str:
    """Formatted internal WRITE (#16), for the edit descriptors the corpus
    uses: ``Iw[.m]``, ``Fw.d``, ``Ew.d`` / ``ESw.d``, ``Gw.d``, ``A[w]``,
    ``Lw``, ``nX``, ``/``, literals, repeat counts. Fortran field semantics:
    right-justified, asterisks on overflow, ``Iw.m`` zero-filled to ``m``
    digits, ``E`` as ``0.dddE+ee``."""
    body = fmt.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    out: list[str] = []
    values = list(vals)
    pos = 0
    while pos < len(body):
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
                out.append("\n")
            else:
                if not values:
                    return "".join(out)
                out.append(_fmt_one(u, values.pop(0)))
    return "".join(out)


def _fmt_one(u: str, v: Any) -> str:
    def fit(s: str, w: int) -> str:
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
