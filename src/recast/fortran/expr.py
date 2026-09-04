"""A language-neutral constant expression, and one fold to render it.

Migrated from the ``ExprPrinter`` in the source pipeline
``pipeline/resolve_use.py``, which printed the same parsed expression as both
Fortran and Python so that a stand-in module and a translated constants file
would agree bit-for-bit by construction rather than by review.

Splitting that into a tree plus a fold keeps the guarantee and moves the two
target languages out of the frontend. The tree records grouping and operators;
``render`` decides nothing except how to join them. Two renderers that disagree
about a value are then a bug in one callback, not a divergence in two
independently written printers -- which is the failure the original was written
to prevent.

Deliberately small. These are physical-constant initializers -- sums, products
and powers over literals and earlier constants, and a short list of intrinsic
calls both languages fold to the same bits (``INTRINSICS``). Anything richer
raises ``UnsupportedExpression`` rather than being approximated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03

BINARY_OPS = ("+", "-", "*", "/", "**")
UNARY_OPS = ("+", "-")

INTRINSICS = frozenset(
    {"max", "min", "abs", "sqrt", "epsilon", "huge", "tiny", "real", "dble", "int"}
)
"""The intrinsic calls an initializer may make.

Each is one the compiler folds to the same value NumPy computes: ``max`` /
``min`` / ``abs`` select or negate, ``sqrt`` is correctly rounded on both
sides (IEEE 754 requires it), the three kind inquiries are the kind's own
constants, and ``real`` / ``dble`` / ``int`` are conversions. ``exp``,
``log`` and the trigonometric functions are not here: gfortran folds them
with MPFR and libm need not agree in the last bit, so a constant made of one
is declined rather than approximated.
"""

KIND_INQUIRIES = frozenset({"epsilon", "huge", "tiny"})
"""Calls whose argument contributes its kind, not its value."""

KINDS_64 = frozenset({"8", "dp", "r8", "core_rknd", "shr_kind_r8", "real64"})
"""Spellings of a 64-bit real kind a ``kind=`` argument may name. Any other
spelling refuses: the fold renders every real as 64-bit and must not claim a
conversion it cannot honour."""


class UnsupportedExpression(RecastError):
    """An initializer this frontend will not claim to understand.

    Raised rather than returned because the caller's only correct response is
    to leave the constant unresolved and say so. A silently approximated
    physical constant is the kind of defect a bit-exact gate cannot attribute.
    """


@dataclass(frozen=True)
class Expr:
    """One node of a constant initializer.

    ``kind`` is ``real``, ``int``, ``name``, ``paren``, ``unary``, ``binary``
    or ``call``. ``text`` carries the literal text, the identifier, the
    operator, or the intrinsic's lower-case name.
    """

    kind: str
    text: str = ""
    args: tuple[Expr, ...] = field(default_factory=tuple)


def _normalize_real(text: str) -> str:
    """``7.90298_r8`` -> ``7.90298``; ``1.d0`` -> ``1.e0``.

    Strips the kind suffix and folds Fortran's ``d`` exponent to ``e``. The
    digits themselves are never touched, so no rounding happens here.
    """
    return text.split("_")[0].lower().replace("d", "e")


def build(node: Any) -> Expr:
    """Turn an fparser2 initializer node into an ``Expr``."""
    if isinstance(node, f03.Real_Literal_Constant):
        return Expr("real", _normalize_real(str(node)))
    if isinstance(node, f03.Int_Literal_Constant):
        return Expr("int", str(node).split("_")[0])
    if isinstance(node, f03.Name):
        return Expr("name", str(node).lower())
    if isinstance(node, f03.Parenthesis):
        return Expr("paren", "", (build(node.children[1]),))
    children = getattr(node, "children", None)
    if children and len(children) == 2 and isinstance(children[0], str):
        if children[0] in UNARY_OPS:
            return Expr("unary", children[0], (build(children[1]),))
    if children and len(children) == 3 and isinstance(children[1], str):
        if children[1] in BINARY_OPS:
            return Expr("binary", children[1], (build(children[0]), build(children[2])))
    if isinstance(node, f03.Intrinsic_Function_Reference):
        return _call(node)
    raise UnsupportedExpression(f"unsupported initializer node {type(node).__name__}: {node}")


def _call(node: Any) -> Expr:
    """An intrinsic call from ``INTRINSICS``, its arguments built in order.

    A ``kind=`` argument on a conversion is checked and dropped: the fold
    renders every real as 64-bit, so a 64-bit kind is the identity and any
    other kind is refused. Every other keyword argument refuses -- the
    intrinsics listed take positional arguments in every initializer seen.
    """
    fname = str(node.children[0]).lower()
    if fname not in INTRINSICS:
        raise UnsupportedExpression(f"unsupported intrinsic in initializer: {node}")
    spec = node.children[1]
    args: list[Expr] = []
    for item in spec.items if spec is not None else ():
        if isinstance(item, f03.Actual_Arg_Spec):
            keyword, value = (str(c).lower() for c in item.children)
            if keyword == "kind" and fname in {"real", "dble", "int"} and value in KINDS_64:
                continue
            raise UnsupportedExpression(f"unsupported keyword argument in initializer: {node}")
        args.append(build(item))
    if fname in {"real", "dble"} and len(args) == 2:
        # ``real(x, r8)``: the positional form of the same kind argument.
        if args[1].kind == "name" and args[1].text in KINDS_64:
            args = args[:1]
        elif args[1].kind == "int" and args[1].text in KINDS_64:
            args = args[:1]
        else:
            raise UnsupportedExpression(f"unsupported kind in initializer: {node}")
    arity = {
        "abs": 1,
        "sqrt": 1,
        "epsilon": 1,
        "huge": 1,
        "tiny": 1,
        "real": 1,
        "dble": 1,
        "int": 1,
    }
    if len(args) != arity.get(fname, len(args)) or (fname in {"max", "min"} and len(args) < 2):
        raise UnsupportedExpression(f"unsupported argument count in initializer: {node}")
    return Expr("call", fname, tuple(args))


def substitute(expr: Expr, name: str, replacement: Expr) -> Expr:
    """``expr`` with every reference to ``name`` replaced.

    For the one legal self-reference in an initializer, a kind inquiry on the
    constant being declared (``tol = max( 1.e-10_r8, epsilon(tol) )``): the
    reference carries the constant's kind and nothing else, and the fold
    renders reals as 64-bit, so a 64-bit literal stands in for it.
    """
    if expr.kind == "name" and expr.text == name:
        return replacement
    if not expr.args:
        return expr
    return Expr(expr.kind, expr.text, tuple(substitute(a, name, replacement) for a in expr.args))


def render(
    expr: Expr,
    *,
    real: Callable[[str], str],
    integer: Callable[[str], str],
    name: Callable[[str], str],
    call: Callable[[str, list[str]], str] | None = None,
) -> str:
    """Fold an ``Expr`` to text, given how to spell its three kinds of atom
    and, optionally, an intrinsic call over already-rendered arguments.

    Grouping and spacing are fixed here so that every target language brackets
    the arithmetic identically. That is the whole point: two renderings of one
    tree can differ in how a literal is spelled and not in what is multiplied
    by what. A renderer given no ``call`` refuses a tree with one in it
    rather than guessing a spelling.
    """
    if expr.kind == "real":
        return real(expr.text)
    if expr.kind == "int":
        return integer(expr.text)
    if expr.kind == "name":
        return name(expr.text)
    sub = [render(a, real=real, integer=integer, name=name, call=call) for a in expr.args]
    if expr.kind == "call":
        if call is None:
            raise UnsupportedExpression(f"no rendering for intrinsic {expr.text!r} in this target")
        return call(expr.text, sub)
    if expr.kind == "paren":
        return f"({sub[0]})"
    if expr.kind == "unary":
        return f"({expr.text}{sub[0]})"
    if expr.kind == "binary":
        return f"({sub[0]} {expr.text} {sub[1]})"
    raise UnsupportedExpression(f"unknown Expr kind {expr.kind!r}")


REAL_CALLS = frozenset({"real", "dble", "sqrt"} | KIND_INQUIRIES)


def typed(expr: Expr, env: dict[str, str | None] | None = None) -> str | None:
    """``"real"``, ``"int"``, or ``None`` when a bare name leaves it open.

    Type inference the fold needs for exactly one decision: whether a ``/``
    is Fortran's integer division. A real literal or a real-valued call
    anywhere in an operand makes the quotient real; ``int(...)`` and integer
    literals make it integer; a name is what ``env`` says its declaration
    was -- CLUBB's ``ep = Rd / Rv`` over two real parameters is a real
    quotient, and a fold that guessed integer made it zero.
    """
    if expr.kind == "real":
        return "real"
    if expr.kind == "int":
        return "int"
    if expr.kind == "name":
        return (env or {}).get(expr.text)
    if expr.kind == "call":
        if expr.text == "int":
            return "int"
        if expr.text in REAL_CALLS:
            return "real"
    kinds = {typed(a, env) for a in expr.args}
    if "real" in kinds:
        return "real"
    if kinds == {"int"}:
        return "int"
    return None


def with_integer_division(
    expr: Expr, *, default_integer: bool | None = None, env: dict[str, str | None] | None = None
) -> Expr:
    """The tree with every integer ``/`` spelled ``//``.

    Fortran divides two integers to an integer: ``nrk = runge_kutta_type / 10``
    is 4, not 4.1. A quotient whose operands are both known integers is
    marked; one with a name in it is typed by ``env`` (the declared types
    of the constants resolved so far) and otherwise falls back to
    ``default_integer``, which the caller sets from the whole initializer.
    """
    if default_integer is None:
        default_integer = typed(expr, env) != "real"
    if not expr.args:
        return expr
    args = tuple(
        with_integer_division(a, default_integer=default_integer, env=env) for a in expr.args
    )
    text = expr.text
    if expr.kind == "binary" and expr.text == "/":
        kinds = {typed(a, env) for a in args}
        if kinds == {"int"} or ("real" not in kinds and default_integer):
            text = "//"
    return Expr(expr.kind, text, args)


def names_used(expr: Expr) -> list[str]:
    """Every identifier the expression depends on, in traversal order."""
    if expr.kind == "name":
        return [expr.text]
    out: list[str] = []
    for a in expr.args:
        out.extend(names_used(a))
    return out


def python_call(fname: str, args: list[str], *, real64: str = "np.float64") -> str:
    """Spell one whitelisted intrinsic in Python over rendered arguments.

    ``real64`` names the 64-bit real constructor the caller renders literals
    with (``np.float64`` in an emitted module, ``float`` in an evaluator that
    imports nothing). A kind inquiry is spelled over the argument's own value
    where NumPy is available -- ``np.finfo(PI).eps`` is the epsilon of PI's
    kind -- and over the 64-bit kind otherwise.
    """
    numpy = real64 == "np.float64"
    if fname in {"max", "min", "abs", "int"}:
        return f"{fname}({', '.join(args)})"
    if fname == "sqrt":
        return f"np.sqrt({args[0]})" if numpy else f"math.sqrt({args[0]})"
    if fname in {"real", "dble"}:
        return f"{real64}({args[0]})"
    if fname in KIND_INQUIRIES:
        attr = {"epsilon": "eps", "huge": "max", "tiny": "tiny"}[fname]
        if numpy:
            return f"np.finfo({args[0]}).{attr}"
        field = {"epsilon": "epsilon", "huge": "max", "tiny": "min"}[fname]
        return f"sys.float_info.{field}"
    raise UnsupportedExpression(f"no Python spelling for intrinsic {fname!r}")
