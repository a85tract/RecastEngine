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
and powers over literals and earlier constants. Anything richer raises
``UnsupportedExpression`` rather than being approximated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03

BINARY_OPS = ("+", "-", "*", "/", "**")
UNARY_OPS = ("+", "-")


class UnsupportedExpression(RecastError):
    """An initializer this frontend will not claim to understand.

    Raised rather than returned because the caller's only correct response is
    to leave the constant unresolved and say so. A silently approximated
    physical constant is the kind of defect a bit-exact gate cannot attribute.
    """


@dataclass(frozen=True)
class Expr:
    """One node of a constant initializer.

    ``kind`` is ``real``, ``int``, ``str``, ``name``, ``paren``, ``unary`` or
    ``binary``. ``text`` carries the literal text, the identifier, or the
    operator. A ``str`` node's text is the character constant's *value*,
    the Fortran quoting undone (``'it''s'`` -> ``it's``), so a renderer
    quotes it for its own language and never re-parses Fortran's.
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


def _character_value(text: str) -> str:
    """The value of a character constant: the quotes off, a doubled quote
    inside them folded to one. A kind prefix (``k_'text'``) is dropped."""
    text = text.strip()
    if "_" in text and text[0] not in ("'", '"'):
        text = text.split("_", 1)[1]
    quote = text[0]
    return text[1:-1].replace(quote * 2, quote)


def build(node: Any) -> Expr:
    """Turn an fparser2 initializer node into an ``Expr``."""
    if isinstance(node, f03.Real_Literal_Constant):
        return Expr("real", _normalize_real(str(node)))
    if isinstance(node, f03.Int_Literal_Constant):
        return Expr("int", str(node).split("_")[0])
    if isinstance(node, f03.Char_Literal_Constant):
        # A character parameter is a value too: ``namep = 'pft'`` names a
        # level to an abort message, and a tree that use-imports it is
        # resolvable, not refused.
        return Expr("str", _character_value(str(node)))
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
    raise UnsupportedExpression(f"unsupported initializer node {type(node).__name__}: {node}")


def render(
    expr: Expr,
    *,
    real: Callable[[str], str],
    integer: Callable[[str], str],
    name: Callable[[str], str],
    string: Callable[[str], str] = repr,
) -> str:
    """Fold an ``Expr`` to text, given how to spell its four kinds of atom.

    ``string`` spells a character value; Python's ``repr`` is the default,
    a quoting that never re-parses the Fortran one.

    Grouping and spacing are fixed here so that every target language brackets
    the arithmetic identically. That is the whole point: two renderings of one
    tree can differ in how a literal is spelled and not in what is multiplied
    by what.
    """
    if expr.kind == "real":
        return real(expr.text)
    if expr.kind == "int":
        return integer(expr.text)
    if expr.kind == "name":
        return name(expr.text)
    if expr.kind == "str":
        return string(expr.text)
    sub = [render(a, real=real, integer=integer, name=name, string=string) for a in expr.args]
    if expr.kind == "paren":
        return f"({sub[0]})"
    if expr.kind == "unary":
        return f"({expr.text}{sub[0]})"
    if expr.kind == "binary":
        return f"({sub[0]} {expr.text} {sub[1]})"
    raise UnsupportedExpression(f"unknown Expr kind {expr.kind!r}")


def names_used(expr: Expr) -> list[str]:
    """Every identifier the expression depends on, in traversal order."""
    if expr.kind == "name":
        return [expr.text]
    out: list[str] = []
    for a in expr.args:
        out.extend(names_used(a))
    return out
