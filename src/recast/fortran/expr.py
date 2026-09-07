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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03

BINARY_OPS = ("+", "-", "*", "/", "**")
UNARY_OPS = ("+", "-")

INTRINSICS = frozenset(
    {"max", "min", "abs", "sqrt", "epsilon", "huge", "tiny", "real", "dble", "float", "int"}
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

STANDARD_REAL_KINDS = {"4": "float32", "8": "float64", "real32": "float32", "real64": "float64"}
"""The kind spellings the language itself fixes. Every other name -- ``r8``,
``dp``, ``core_rknd`` -- is a parameter of some module, and what it means is
read from that module's initializer or supplied by the extension's
``kind_assumptions``; nothing here assumes a domain's conventions."""

CONVERSIONS = frozenset({"real", "dble", "float"})
"""Conversions whose result kind is the point: ``real(x)`` is the *default*
real -- single precision -- unless a kind says otherwise, ``dble`` is double."""


class UnsupportedExpression(RecastError):
    """An initializer this frontend will not claim to understand.

    Raised rather than returned because the caller's only correct response is
    to leave the constant unresolved and say so. A silently approximated
    physical constant is the kind of defect a bit-exact gate cannot attribute.
    """


@dataclass(frozen=True)
class Expr:
    """One node of a constant initializer.

    ``kind`` is ``real``, ``int``, ``str``, ``name``, ``paren``, ``unary``,
    ``binary`` or ``call``. ``text`` carries the literal text, the identifier,
    the operator, or the intrinsic's lower-case name. A ``str`` node's text is
    the character constant's *value*, the Fortran quoting undone (``'it''s'``
    -> ``it's``), so a renderer quotes it for its own language and never
    re-parses Fortran's.
    """

    kind: str
    text: str = ""
    args: tuple[Expr, ...] = field(default_factory=tuple)
    dtype: str | None = None
    """``float32``, ``float64``, ``int``, ``str`` or ``None`` when the kind
    could not be settled -- a name whose declaration the resolver did not
    see. The kind is what decides whether a fold is exact: a compiler folds
    ``0.1 * x`` in the wider of the two kinds, and an initializer folded at
    another width is a different number."""


def _normalize_real(text: str) -> str:
    """``7.90298_r8`` -> ``7.90298``; ``1.d0`` -> ``1.e0``.

    Strips the kind suffix and folds Fortran's ``d`` exponent to ``e``. The
    digits themselves are never touched, so no rounding happens here.
    """
    return text.split("_")[0].lower().replace("d", "e")


def _character_value(text: str) -> str:
    """The value of a character constant: the quotes off, a doubled quote
    inside them folded to one. A kind prefix (``k_'text'``) is dropped."""
    text = text.strip()
    if "_" in text and text[0] not in ("'", '"'):
        text = text.split("_", 1)[1]
    quote = text[0]
    return text[1:-1].replace(quote * 2, quote)


def real_kind_of(spelling: str, kinds: Mapping[str, str] | None) -> str:
    """The dtype a kind spelling names, or a refusal.

    ``8`` and ``real64`` are the language's; ``r8`` is whatever the tree or
    the extension said it is. A spelling neither knows is not guessed: the
    fold would otherwise claim a precision it cannot honour.
    """
    key = spelling.strip().lower()
    if key in STANDARD_REAL_KINDS:
        return STANDARD_REAL_KINDS[key]
    dtype = (kinds or {}).get(key)
    if dtype in ("float32", "float64"):
        return dtype
    raise UnsupportedExpression(f"real kind {spelling!r} is not one this fold knows the width of")


def literal_kind(text: str, kinds: Mapping[str, str] | None) -> str:
    """The kind of a real literal: its suffix, else ``d`` for double, else
    the default real -- single precision in every build here."""
    if "_" in text:
        return real_kind_of(text.split("_", 1)[1], kinds)
    return "float64" if "d" in text.lower() else "float32"


def promoted(*dtypes: str | None) -> str | None:
    """Fortran's promotion over an operator's operands: the wider real wins,
    an integer defers to any real, and an operand of unknown kind leaves the
    result unknown -- ``None`` is a fact the fold reports, not a default."""
    if any(d is None for d in dtypes):
        return None
    if "float64" in dtypes:
        return "float64"
    if "float32" in dtypes:
        return "float32"
    if all(d == "int" for d in dtypes):
        return "int"
    return None


def build(
    node: Any,
    kinds: Mapping[str, str] | None = None,
    declared: Mapping[str, str | None] | None = None,
) -> Expr:
    """Turn an fparser2 initializer node into an ``Expr``.

    ``kinds`` maps kind-parameter names to dtypes (``{"r8": "float64"}``) and
    ``declared`` maps constant names to the dtype they were declared with;
    both feed the ``dtype`` every node carries. Given neither, literals are
    still typed by their own spelling and names are left unknown.
    """
    if isinstance(node, f03.Real_Literal_Constant):
        text = str(node)
        return Expr("real", _normalize_real(text), dtype=literal_kind(text, kinds))
    if isinstance(node, f03.Int_Literal_Constant):
        return Expr("int", str(node).split("_")[0], dtype="int")
    if isinstance(node, f03.Char_Literal_Constant):
        # A character parameter is a value too: ``namep = 'pft'`` names a
        # level to an abort message, and a tree that use-imports it is
        # resolvable, not refused.
        return Expr("str", _character_value(str(node)))
    if isinstance(node, f03.Name):
        name = str(node).lower()
        return Expr("name", name, dtype=(declared or {}).get(name))
    if isinstance(node, f03.Parenthesis):
        inner = build(node.children[1], kinds, declared)
        return Expr("paren", "", (inner,), dtype=inner.dtype)
    children = getattr(node, "children", None)
    if children and len(children) == 2 and isinstance(children[0], str):
        if children[0] in UNARY_OPS:
            inner = build(children[1], kinds, declared)
            return Expr("unary", children[0], (inner,), dtype=inner.dtype)
    if children and len(children) == 3 and isinstance(children[1], str):
        if children[1] in BINARY_OPS:
            left = build(children[0], kinds, declared)
            right = build(children[2], kinds, declared)
            return Expr(
                "binary", children[1], (left, right), dtype=promoted(left.dtype, right.dtype)
            )
    if isinstance(node, f03.Intrinsic_Function_Reference):
        return _call(node, kinds, declared)
    raise UnsupportedExpression(f"unsupported initializer node {type(node).__name__}: {node}")


def _call(
    node: Any,
    kinds: Mapping[str, str] | None = None,
    declared: Mapping[str, str | None] | None = None,
) -> Expr:
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
    kind_arg: str | None = None
    for item in spec.items if spec is not None else ():
        if isinstance(item, f03.Actual_Arg_Spec):
            keyword, value = (str(c).lower() for c in item.children)
            if keyword == "kind" and fname in CONVERSIONS | {"int"}:
                kind_arg = value
                continue
            raise UnsupportedExpression(f"unsupported keyword argument in initializer: {node}")
        args.append(build(item, kinds, declared))
    if fname in CONVERSIONS | {"int"} and len(args) == 2:
        # ``real(x, r8)``: the positional form of the same kind argument.
        if args[1].kind in ("name", "int"):
            kind_arg = args[1].text
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
        "float": 1,
        "int": 1,
    }
    if len(args) != arity.get(fname, len(args)) or (fname in {"max", "min"} and len(args) < 2):
        raise UnsupportedExpression(f"unsupported argument count in initializer: {node}")
    if fname == "int":
        return Expr("call", fname, tuple(args), dtype="int")
    if fname in CONVERSIONS:
        # The result kind is the whole point of a conversion, and it must be
        # known: ``real(x)`` is single precision, ``real(x, r8)`` whatever
        # ``r8`` is, ``dble(x)`` double. A kind the fold cannot place refuses.
        if kind_arg is not None:
            dtype: str | None = real_kind_of(kind_arg, kinds)
        else:
            dtype = "float64" if fname == "dble" else "float32"
        return Expr("call", fname, tuple(args), dtype=dtype)
    if fname in KIND_INQUIRIES:
        # The argument contributes its kind and nothing else. Unknown, the
        # inquiry has no answer -- ``epsilon`` of a single is 2**-23, of a
        # double 2**-52, and a fold that picked one would be silently off
        # by sixteen orders of magnitude on the other.
        asked = args[0].dtype
        if asked not in ("float32", "float64"):
            raise UnsupportedExpression(
                f"kind of the argument of {fname}({args[0].text or args[0].kind}) is not known"
            )
        return Expr("call", fname, (Expr("dtype", asked, dtype=asked),), dtype=asked)
    return Expr("call", fname, tuple(args), dtype=promoted(*(a.dtype for a in args)))


def substitute(expr: Expr, name: str, replacement: Expr) -> Expr:
    """``expr`` with every reference to ``name`` replaced.

    For the one legal self-reference in an initializer, a kind inquiry on the
    constant being declared (``tol = max( 1.e-10_r8, epsilon(tol) )``): the
    reference carries the constant's kind and nothing else, and the fold
    renders reals as 64-bit, so a 64-bit literal stands in for it. Every
    occurrence is replaced: a parameter cannot name itself anywhere else in
    its own initializer, so there is no other occurrence to preserve.
    """
    if expr.kind == "name" and expr.text == name:
        return replacement
    if not expr.args:
        return expr
    return Expr(
        expr.kind, expr.text, tuple(substitute(a, name, replacement) for a in expr.args), expr.dtype
    )


def render(
    expr: Expr,
    *,
    real: Callable[[str], str],
    integer: Callable[[str], str],
    name: Callable[[str], str],
    string: Callable[[str], str] = repr,
    call: Callable[[str, list[str], str | None], str] | None = None,
    real32: Callable[[str], str] | None = None,
    dtype: Callable[[str], str] | None = None,
) -> str:
    """Fold an ``Expr`` to text, given how to spell its four kinds of atom
    and, optionally, an intrinsic call over already-rendered arguments.

    ``string`` spells a character value; Python's ``repr`` is the default,
    a quoting that never re-parses the Fortran one. ``call`` receives the
    intrinsic's name, its rendered arguments and the call's own kind
    (``float32`` for ``real(x)``, the argument's for ``epsilon(x)``);
    ``real32`` spells a single-precision literal where the target can, and
    ``dtype`` spells the kind an inquiry asks about.

    Grouping and spacing are fixed here so that every target language brackets
    the arithmetic identically. That is the whole point: two renderings of one
    tree can differ in how a literal is spelled and not in what is multiplied
    by what. A renderer given no ``call`` refuses a tree with one in it
    rather than guessing a spelling.
    """
    if expr.kind == "real":
        # A single-precision literal is spelled as one when the renderer
        # can (``real32``); a renderer without that spelling gets the digits
        # and its caller has already ruled on whether that is exact.
        if expr.dtype == "float32" and real32 is not None:
            return real32(expr.text)
        return real(expr.text)
    if expr.kind == "int":
        return integer(expr.text)
    if expr.kind == "name":
        return name(expr.text)
    if expr.kind == "str":
        return string(expr.text)
    if expr.kind == "dtype":
        # The argument of a kind inquiry: the kind itself, no value.
        if dtype is None:
            raise UnsupportedExpression("no rendering for a kind in this target")
        return dtype(expr.text)
    sub = [
        render(
            a,
            real=real,
            integer=integer,
            name=name,
            string=string,
            call=call,
            real32=real32,
            dtype=dtype,
        )
        for a in expr.args
    ]
    if expr.kind == "call":
        if call is None:
            raise UnsupportedExpression(f"no rendering for intrinsic {expr.text!r} in this target")
        return call(expr.text, sub, expr.dtype)
    if expr.kind == "paren":
        return f"({sub[0]})"
    if expr.kind == "unary":
        return f"({expr.text}{sub[0]})"
    if expr.kind == "binary":
        return f"({sub[0]} {expr.text} {sub[1]})"
    raise UnsupportedExpression(f"unknown Expr kind {expr.kind!r}")


REAL_CALLS = frozenset({"real", "dble", "float", "sqrt"} | KIND_INQUIRIES)

REAL_LEAVES = frozenset({"real", "name", "call"})
"""Node kinds that carry a value of their own kind into an expression."""


def _leaves(expr: Expr) -> list[Expr]:
    if expr.kind in ("real", "int", "name", "str", "dtype"):
        return [expr]
    if expr.kind == "call" and (expr.text in CONVERSIONS or expr.text in KIND_INQUIRIES):
        return [expr]  # the conversion is the leaf: its kind is the point
    out: list[Expr] = []
    for a in expr.args:
        out.extend(_leaves(a))
    return out


def fold_check(expr: Expr, declared: str | None) -> str:
    """Whether rendering ``expr`` with 64-bit arithmetic and 32-bit atoms is
    exactly what a compiler stores into a constant declared ``declared``.

    Returns how to store it -- ``"as-is"``, ``"single"`` (a lone
    single-precision value, to be rounded to single before it is widened),
    ``"int"`` -- or raises ``UnsupportedExpression`` naming why the fold
    would be a different number:

    * a leaf whose kind is unknown (a name the resolver did not see declared,
      a kind spelling nothing defines);
    * two single-precision operands, or a single beside an integer, in a
      double constant: the compiler combines them in single first, and this
      renderer has no single-precision arithmetic;
    * arithmetic in a single-precision constant, for the same reason;
    * a real constant whose own kind is unknown.

    A lone default-real literal in a double constant is exact -- the single
    value widened, which is what the rule for unsuffixed literals has spelled
    all along -- and so is a lone single value stored into a single.
    """
    leaves = _leaves(expr)
    for leaf in leaves:
        if leaf.kind in ("str", "dtype"):
            continue
        if leaf.dtype is None:
            what = leaf.text if leaf.kind != "call" else f"{leaf.text}(...)"
            raise UnsupportedExpression(
                f"kind of {what!r} is not known; the fold would guess a width"
            )
    reals32 = sum(1 for leaf in leaves if leaf.dtype == "float32")
    ints = sum(1 for leaf in leaves if leaf.dtype == "int")
    lone = len(leaves) == 1 and expr.kind in REAL_LEAVES
    if declared in ("int", "complex"):
        return "int"  # no real storage rounding to apply; spelled as before
    if declared == "float64":
        if expr.dtype == "int":
            return "int"
        if expr.dtype == "float32":
            if lone:
                return "single"
            raise UnsupportedExpression(
                "single-precision arithmetic in a double constant: the compiler folds it "
                "in single, this fold has no single-precision arithmetic"
            )
        if reals32 > 1 or (reals32 == 1 and ints):
            raise UnsupportedExpression(
                "a single-precision operand meets another single or an integer before a "
                "double does: the compiler folds that step in single"
            )
        return "as-is"
    if declared == "float32":
        if lone:
            return "single"
        raise UnsupportedExpression(
            "arithmetic in a single-precision constant: the compiler folds it in single, "
            "this fold has no single-precision arithmetic"
        )
    raise UnsupportedExpression(
        f"declared kind {declared!r} is not one this fold knows the width of"
    )


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
    return Expr(expr.kind, text, args, expr.dtype)


def names_used(expr: Expr) -> list[str]:
    """Every identifier the expression depends on, in traversal order."""
    if expr.kind == "name":
        return [expr.text]
    out: list[str] = []
    for a in expr.args:
        out.extend(names_used(a))
    return out


def python_call(
    fname: str, args: list[str], *, real64: str = "np.float64", result_kind: str | None = None
) -> str:
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
    if fname in CONVERSIONS:
        # ``result_kind`` is the conversion's own kind, threaded by the
        # caller; without it the 64-bit spelling stands, as before.
        if result_kind == "float32":
            return f"np.float32({args[0]})" if numpy else f"_f32({args[0]})"
        return f"{real64}({args[0]})"
    if fname in KIND_INQUIRIES:
        attr = {"epsilon": "eps", "huge": "max", "tiny": "tiny"}[fname]
        if numpy:
            return f"np.finfo({args[0]}).{attr}"
        # ``args[0]`` is the kind's name here (see ``render``'s ``dtype``).
        if args[0] == "float32":
            return {
                "epsilon": "2.0 ** -23",
                "huge": "3.4028234663852886e+38",
                "tiny": "2.0 ** -126",
            }[fname]
        field = {"epsilon": "epsilon", "huge": "max", "tiny": "min"}[fname]
        return f"sys.float_info.{field}"
    raise UnsupportedExpression(f"no Python spelling for intrinsic {fname!r}")
