"""Parameters and inline literals, classified but not yet spelled.

Migrated from the source pipeline's ``pipeline/extract_constants.py``, and
split where the ``Frontend`` contract says it has to be. The original did two
jobs: it worked out what every parameter and every inline numeric literal was,
and it wrote a NumPy constants module. Only the first job is analysis.

So ``classify_init`` returns a decision -- integer, real, logical, reference,
array, an expression over earlier constants, or an honest refusal -- and never
a line of target-language source. The rendering it used to do belongs to the
Transform that has a target language, and lives with it.

The zero-literal rule survives the split intact: every non-structural numeric
literal in an execution part is hoisted to a deterministic name, so that a
translated routine contains no bare magic numbers and every constant has one
definition that both sides of a differential check can be pointed at.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recast.fortran._parse import f03, parse, walk
from recast.fortran.expr import _normalize_real
from recast.fortran.interface import _scope_of

WHITELIST_INT = frozenset({"0", "1", "2"})
WHITELIST_REAL = frozenset({0.0, 1.0, 0.5})
"""Literals that stay inline.

These are structural -- loop bounds, indices, halves -- not physical. Hoisting
them would bury the arithmetic in names without making anything checkable.
"""

_ARRAY_ELEMENTS = r"(?:[0-9eE.,+\-\s]|'[^']*')+"
_ARRAY_MODULE_RE = re.compile(rf"[0-9eE.+\-*/\s]*\[{_ARRAY_ELEMENTS}\][0-9eE.+\-*/\s]*")
_ARRAY_LOCAL_RE = re.compile(rf"\[{_ARRAY_ELEMENTS}\]")
_ARRAY_OLD_FORM = re.compile(r"\(/(.*)/\)", re.S)
_KIND_SUFFIX_RE = re.compile(r"_\w+")
_ARRAY_CTOR_OLD = re.compile(r"\(/(.+)/\)", re.S)
_ARRAY_CTOR_NEW = re.compile(r"\[(.+)\]", re.S)
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_LEADING_DOT = re.compile(r"^(-?)\.")
_NUMERIC = re.compile(r"-?\d+\.?\d*(?:e[+-]?\d+)?")
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*(?:[edED][+-]?\d+)?(?:_\w+)?|\*\*|[()+\-*/,]")


def strip_kind(text: str) -> str:
    """``7.90298_r8`` -> ``7.90298``; ``10_i8`` -> ``10``."""
    return text.split("_")[0]


def is_default_real(text: str) -> bool:
    """Is this real literal written in Fortran's *default* real kind?

    An unsuffixed literal with no ``d`` exponent is default REAL, which is
    single precision in every build here -- and a compiler evaluates it at
    that precision before promoting. ``real(real64) :: x`` then ``x = 0.1``
    stores 0.10000000149011612, not 0.1, and a translation that writes
    ``0.1`` has changed the number. Assigning the same literal ``0.1_r8``
    stores 0.1. The two are different constants and get different names.
    """
    return "_" not in text and "d" not in text.lower()


def canon_name(text: str, is_real: bool) -> str:
    """Deterministic constant name for a literal.

    Derived from the literal's own digits, so the same number hoisted from two
    different routines lands on one name and one definition. ``F_`` for reals,
    ``F32_`` for reals written in the default kind (a different value; see
    ``is_default_real``), ``I_`` for integers, ``P`` for the point, ``M`` for
    a minus.
    """
    t = strip_kind(text).lower().replace("d", "e")
    t = t.replace(".", "P").replace("+", "").replace("-", "M").replace("e", "E")
    if not is_real:
        return f"I_{t}"
    return f"{'F32' if is_default_real(text) else 'F'}_{t}"


def is_whitelisted(text: str, is_real: bool) -> bool:
    t = strip_kind(text)
    if is_real:
        try:
            return float(t.lower().replace("d", "e")) in WHITELIST_REAL
        except ValueError:
            return False
    return t in WHITELIST_INT


def literals_with_lines(
    node: Any, line: int | None = None, out: list[tuple[str, bool, int | None]] | None = None
) -> list[tuple[str, bool, int | None]]:
    """Depth-first ``(literal_text, is_real, nearest_line)`` under a node.

    Descends into plain tuples and lists as well as node children: some
    fparser2 nodes -- ``Loop_Control`` among them -- hold their children in bare
    Python containers, and a traversal that only follows ``.children`` silently
    misses every do-bound literal in the file.
    """
    if out is None:
        out = []
    if isinstance(node, list | tuple):
        for ch in node:
            if ch is not None and not isinstance(ch, str):
                literals_with_lines(ch, line, out)
        return out
    item = getattr(node, "item", None)
    if item is not None and getattr(item, "span", None):
        line = item.span[0]
    if isinstance(node, (f03.Real_Literal_Constant, f03.Signed_Real_Literal_Constant)):
        out.append((str(node), True, line))
        return out
    if isinstance(node, (f03.Int_Literal_Constant, f03.Signed_Int_Literal_Constant)):
        out.append((str(node), False, line))
        return out
    children = getattr(node, "children", None)
    if children is None or not isinstance(children, list | tuple):
        return out
    for ch in children:
        if ch is not None and not isinstance(ch, str):
            literals_with_lines(ch, line, out)
    return out


INTRINSICS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "cos",
        "dble",
        "epsilon",
        "exp",
        "float",
        "huge",
        "int",
        "log",
        "log10",
        "max",
        "min",
        "mod",
        "modulo",
        "nint",
        "real",
        "sign",
        "sin",
        "sqrt",
        "tan",
        "tiny",
    }
)
"""Intrinsics a constant expression may call, and this stage can carry across.

The target evaluates the call; this stage only says which name it is. Folding
here would fold at a different precision than the compiler did.
"""

_KIND_ARGUMENT = re.compile(
    r"^(r4|r8|r16|i4|i8|dp|sp|wp|kind|real32|real64|int32|int64"
    r"|selected_real_kind|selected_int_kind)$",
    re.I,
)


def _split_arguments(tokens: list[str]) -> list[list[str]]:
    """Argument token lists, split on the commas at depth zero."""
    arguments: list[list[str]] = []
    current: list[str] = []
    depth = 0
    for piece in tokens:
        if piece in ("(", "(/", "["):
            depth += 1
        elif piece in (")", "/)", "]"):
            depth -= 1
        if piece == "," and depth == 0:
            arguments.append(current)
            current = []
        else:
            current.append(piece)
    if current:
        arguments.append(current)
    return arguments


def _argument_tokens(tokens: list[str], known_names: set[str]) -> list[dict[str, str]] | None:
    """One argument as tokens of the same vocabulary; ``None`` if it names
    something no earlier constant defines."""
    spelled: list[dict[str, str]] = []
    for piece in tokens:
        if re.match(r"[A-Za-z_]", piece):
            if piece.lower() in known_names:
                spelled.append({"t": "ref", "v": piece.lower()})
            elif _KIND_ARGUMENT.match(piece):
                continue
            else:
                return None
        elif re.match(r"\d", piece):
            base = _normalize_real(piece)
            if "." in base or "e" in base:
                spelled.append({"t": "real32" if is_default_real(piece) else "real", "v": base})
            else:
                spelled.append({"t": "int", "v": base})
        else:
            spelled.append({"t": "op", "v": piece})
    return spelled


def _intrinsic_call(
    tokens: list[str], at: int, known_names: set[str]
) -> tuple[dict[str, Any] | None, int]:
    """The call starting at ``tokens[at]``, and the index just past it.

    A trailing kind argument -- ``real(x, r8)`` -- is dropped: it says what
    precision the compiler evaluated in, which the target's own float64 is,
    and it is not a value to pass on.
    """
    name = tokens[at].lower()
    depth = 0
    end = at + 1
    while end < len(tokens):
        if tokens[end] == "(":
            depth += 1
        elif tokens[end] == ")":
            depth -= 1
            if depth == 0:
                break
        end += 1
    arguments = _split_arguments(tokens[at + 2 : end])
    kept = []
    for index, argument in enumerate(arguments):
        stripped = [token for token in argument if token.strip()]
        last = index == len(arguments) - 1
        if last and len(stripped) == 1 and _KIND_ARGUMENT.match(stripped[0]):
            continue
        kept.append(argument)
    spelled = [_argument_tokens(argument, known_names) for argument in kept]
    if any(text is None for text in spelled):
        return None, end + 1
    return {"t": "call", "v": name, "args": spelled}, end + 1


_CHAR_INTRINSICS = ("achar", "char", "new_line", "repeat", "trim", "adjustl", "adjustr")
_CHAR_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")|(?P<int>\d+)"
    r"|(?P<name>[A-Za-z_]\w*)|(?P<op>//|[(),]))"
)
_CHAR_LITERAL = re.compile(r"'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\"")


def _unquote(token: str) -> str:
    quote = token[0]
    return token[1:-1].replace(quote * 2, quote)


def fold_char_expr(expr: str, char_values: Mapping[str, str]) -> str | None:
    """A character parameter initializer folded to its value at extraction time.

    Literals, earlier character parameters of the scope (``char_values``),
    ``//``, and the intrinsics with a fixed meaning -- ``achar``/``char`` of
    an integer literal, ``new_line``, ``repeat`` by an integer literal,
    ``trim``, ``adjustl``, ``adjustr``. Anything else is None: a skip, never
    a rendered expression, because the token route has no rule for ``//``
    and would emit ``A / / B``. CESM-language-translator PR #48's rule.
    """
    tokens: list[re.Match[str]] = []
    pos = 0
    while pos < len(expr):
        m = _CHAR_TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip():
                return None
            break
        tokens.append(m)
        pos = m.end()
    at = 0

    def peek(kind: str, value: str | None = None) -> bool:
        if at >= len(tokens) or tokens[at].lastgroup != kind:
            return False
        return value is None or tokens[at].group(kind) == value

    def primary() -> str | int:
        nonlocal at
        if at >= len(tokens):
            raise ValueError("expression ends early")
        tok = tokens[at]
        at += 1
        group = tok.lastgroup
        if group == "str":
            return _unquote(tok.group("str"))
        if group == "op" and tok.group("op") == "(":
            inner = concat()
            if not peek("op", ")"):
                raise ValueError("unbalanced")
            at += 1
            return inner
        if group == "name":
            name = tok.group("name").lower()
            if peek("op", "("):
                at += 1
                args: list[str | int] = []
                if not peek("op", ")"):
                    args.append(argument())
                    while peek("op", ","):
                        at += 1
                        args.append(argument())
                if not peek("op", ")"):
                    raise ValueError("unbalanced call")
                at += 1
                return call(name, args)
            if name in char_values:
                return char_values[name]
        raise ValueError(f"not foldable: {tok.group(0)!r}")

    def argument() -> str | int:
        nonlocal at
        if peek("int"):
            value = int(tokens[at].group("int"))
            at += 1
            return value
        return concat()

    def call(name: str, args: list[str | int]) -> str:
        if name in ("achar", "char") and len(args) == 1 and isinstance(args[0], int):
            return chr(args[0])
        if name == "new_line" and len(args) == 1:
            return "\n"
        if name == "repeat" and len(args) == 2:
            text, count = args
            if isinstance(text, str) and isinstance(count, int):
                return text * count
        if len(args) == 1 and isinstance(args[0], str):
            text = args[0]
            if name == "trim":
                return text.rstrip(" ")
            if name == "adjustl":
                return text.lstrip(" ") + " " * (len(text) - len(text.lstrip(" ")))
            if name == "adjustr":
                return " " * (len(text) - len(text.rstrip(" "))) + text.rstrip(" ")
        raise ValueError(f"no rule for {name}")

    def concat() -> str:
        nonlocal at
        value = primary()
        if not isinstance(value, str):
            raise ValueError("integer where character expected")
        while peek("op", "//"):
            at += 1
            right = primary()
            if not isinstance(right, str):
                raise ValueError("integer where character expected")
            value = value + right
        return value

    try:
        folded = concat()
    except (ValueError, IndexError):
        return None
    return folded if at == len(tokens) else None


def char_length(type_spec_text: str, entity_text: str = "") -> int | str | None:
    """The declared length of a CHARACTER parameter: an int, ``"*"`` for
    ``len=*`` (the initializer's own length), None when it is a name this
    pass cannot evaluate. A bare ``character`` is length 1."""
    m = re.search(r"\*\s*(\d+|\*)\s*$", entity_text)  # ``c*4`` on the entity
    if m:
        return "*" if m.group(1) == "*" else int(m.group(1))
    text = type_spec_text.strip()
    if not re.match(r"character", text, re.I):
        return None
    m = re.search(r"\(\s*(?:len\s*=\s*)?(\*|\d+|[A-Za-z_]\w*)", text, re.I)
    if not m:
        return 1
    value = m.group(1)
    if value == "*":
        return "*"
    return int(value) if value.isdigit() else None


def fit_char(value: str, length: int | str | None) -> str:
    """Fortran assignment to a fixed-length CHARACTER: blank-padded or
    truncated to the declared length; ``*``/unknown leaves it as written."""
    if isinstance(length, int):
        return value[:length] if len(value) >= length else value + " " * (length - len(value))
    return value


def classify_init(
    init_expr: str,
    known_names: set[str],
    char_values: Mapping[str, str] | None = None,
    char_len: int | str | None = None,
) -> tuple[str, Any]:
    """Classify a parameter initializer. Returns ``(kind, payload)``.

    ``int`` / ``int64`` -> a Python ``int``.
    ``real``            -> normalized decimal text, digits untouched.
    ``logical``         -> a Python ``bool``.
    ``ref``             -> the lower-cased name of another constant.
    ``expr``            -> tokens over earlier constants, in source order.
    ``skip``            -> a reason, in words.

    ``skip`` is a result, not an error. A kind parameter has no runtime value to
    carry, and an initializer this frontend cannot evaluate must be reported as
    unresolved rather than approximated.
    """
    e = init_expr.strip()

    m = re.search(r"selected_real_kind\s*\(\s*(\d+)", e, re.I)
    if m:
        # The kind value itself is referenceable at runtime (`if (kind /= r8)`).
        # gfortran: selected_real_kind(p >= 10) -> 8, otherwise 4.
        return "int", 8 if int(m.group(1)) >= 10 else 4
    m = re.search(r"selected_int_kind\s*\(\s*(\d+)", e, re.I)
    if m:
        # The smallest kind holding 10**N. gfortran: N <= 4 -> 2, N <= 9 -> 4,
        # otherwise 8; a value, because the source can compare against it.
        n = int(m.group(1))
        return "int", 2 if n <= 4 else (4 if n <= 9 else 8)
    if re.search(r"kind\s*\(", e, re.I):
        return "skip", "kind parameter (compile-time only)"

    m = re.fullmatch(r"int\s*\(\s*z'([0-9a-f]+)'\s*,\s*\w+\s*\)", e, re.I)
    if m:  # BOZ literal: int(Z'...', i8)
        return "int64", int(m.group(1), 16)

    m = re.fullmatch(r"z'([0-9a-f]+)'", e, re.I)
    if m:  # a bare BOZ literal, without the int() wrapper
        return "int", int(m.group(1), 16)

    # A character parameter is the value the source compares at runtime
    # (``calkindflag = 'GREGORIAN'``), fitted to its declared length as a
    # Fortran assignment would. A bare literal is its text with the doubled
    # quote undone; an expression is folded here or skipped -- never handed to
    # the token route, which has no rule for ``//`` and would render
    # ``A / / B``, a SyntaxError in a constants module every unit of the tree
    # imports (numfor's ``strings.f90:11`` took 20 units down before the
    # guard). CESM-language-translator PR #48's rule.
    m = _CHAR_LITERAL.fullmatch(e)
    if m:
        return "str", fit_char(_unquote(m.group(0)), char_len)
    if "//" in e or re.match(rf"({'|'.join(_CHAR_INTRINSICS)})\s*\(", e, re.I):
        folded = fold_char_expr(e, char_values or {})
        if folded is None:
            return "skip", f"character expression: {e}"
        return "str", fit_char(folded, char_len)

    compact = e.replace(" ", "")
    if compact.lower() in (".true.", ".false."):
        return "logical", compact.lower() == ".true."
    if re.fullmatch(r"-?\d+", compact):
        return "int", int(compact)
    m = re.fullmatch(r"(-?)(\d+\.?\d*(?:[edED][+-]?\d+)?)(_\w+)?", compact)
    if m and ("." in compact or "e" in compact.lower() or "d" in compact.lower()):
        kind = "real32" if is_default_real(compact) else "real"
        return kind, m.group(1) + _normalize_real(m.group(2))
    if compact.lower() in known_names:
        return "ref", compact.lower()

    # An array constructor. ``_array_literal`` takes the ones whose *shape*
    # is on the entity; this takes the rest, which is how a parameter written
    # ``integer, dimension(4), parameter :: side = (/1,0,1,1/)`` gets a value
    # at all -- it was being skipped as "more than literals" when every
    # element is one.
    spelled = _array_elements(e)
    if spelled is not None:
        return "expr", [{"t": "spelled", "v": spelled}]

    # One that holds more than literals -- a name, an implied do. The token
    # walk below would spell its delimiters as arithmetic, so it stops here.
    # Naming the first unresolved name says which one, the way every other
    # unevaluable expression here does; "more than literals" said only that
    # something was wrong.
    if "(/" in compact or "[" in compact:
        for name in _TOKEN_RE.findall(e):
            if name[0].isalpha() or name[0] == "_":
                if name.lower() not in known_names:
                    return "skip", f"unknown name {name!r} in expression: {e}"
        return "skip", f"array constructor over more than literals: {e}"

    # A constant expression over earlier parameters. Re-emitted token-wise so
    # the target language evaluates the same arithmetic the compiler folded,
    # rather than this stage folding it at a different precision.
    toks = _TOKEN_RE.findall(e)
    if toks and "".join(toks).replace(" ", "") == compact:
        out: list[dict[str, str]] = []
        at = 0
        while at < len(toks):
            t = toks[at]
            if re.match(r"[A-Za-z_]", t) and t.lower() in INTRINSICS:
                if at + 1 < len(toks) and toks[at + 1] == "(":
                    call, at = _intrinsic_call(toks, at, known_names)
                    if call is None:
                        return "skip", f"unresolved argument in expression: {e}"
                    out.append(call)
                    continue
                if t.lower() not in known_names:
                    return "skip", f"unknown name {t!r} in expression: {e}"
                out.append({"t": "ref", "v": t.lower()})
            elif re.match(r"[A-Za-z_]", t):
                if t.lower() not in known_names:
                    return "skip", f"unknown name {t!r} in expression: {e}"
                out.append({"t": "ref", "v": t.lower()})
            elif re.match(r"\d", t):
                base = _normalize_real(t)
                if "." in base or "e" in base:
                    out.append({"t": "real32" if is_default_real(t) else "real", "v": base})
                else:
                    out.append({"t": "int", "v": base})
            else:
                out.append({"t": "op", "v": t})
            at += 1
        return "expr", out

    return "skip", f"unevaluated expression: {e}"


def _array_elements(init: str) -> str | None:
    """A constructor of literal elements, as ``np.array([...])``, or ``None``.

    Both spellings, split at top-level commas so a nested call does not break
    the scan. Character elements are allowed. An element that is not a literal
    makes the whole thing not-an-array, which is the honest answer: this stage
    cannot evaluate a name.
    """
    compact = init.strip()
    inner = None
    for pattern in (_ARRAY_CTOR_OLD, _ARRAY_CTOR_NEW):
        match = pattern.fullmatch(compact)
        if match:
            inner = match.group(1)
            break
    if inner is None:
        return None
    elements, depth, current = [], 0, ""
    for character in inner:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            elements.append(current.strip())
            current = ""
            continue
        current += character
    if current.strip():
        elements.append(current.strip())

    spelled = []
    for element in elements:
        quoted = _QUOTED.fullmatch(element)
        if quoted:
            spelled.append(
                f"'{quoted.group(1) if quoted.group(1) is not None else quoted.group(2)}'"
            )
            continue
        # fparser writes a negative literal as ``- 1``; the spacing is its
        # own, not the source's, and an element is a single literal or it is
        # not one at all.
        cleaned = _LEADING_DOT.sub(
            r"\g<1>0.",
            _KIND_SUFFIX_RE.sub("", element).lower().replace("d", "e").replace(" ", ""),
        )
        if not _NUMERIC.fullmatch(cleaned):
            return None
        if "." in cleaned or "e" in cleaned:
            spelled.append(
                f"np.float64(np.float32('{cleaned}'))"
                if is_default_real(element)
                else f"np.float64('{cleaned}')"
            )
        else:
            spelled.append(cleaned)
    if not spelled:
        return None
    dtype = ", dtype=object" if any(s.startswith("'") for s in spelled) else ""
    return f"np.array([{', '.join(spelled)}]{dtype})"


def _array_literal(init: str, module_level: bool) -> str | None:
    """Kind-stripped text of a pure-literal array constructor, or ``None``.

    Lookup tables are declared as ``dnu(16) = [ ... ]`` or, in the older
    spelling, ``(/ ... /)``. Stripping the kind suffix textually is exact when
    the constructor holds nothing but literals, which is what the pattern
    check enforces before the strip is trusted. Character elements are
    allowed; a name or an implied do is not, and falls through to be skipped
    rather than approximated.
    """
    # ``(/ ... /)`` is the same constructor in the older spelling; which one a
    # source uses is a house style, not a difference in meaning.
    init = _ARRAY_OLD_FORM.sub(r"[\1]", init)
    txt = _KIND_SUFFIX_RE.sub("", init).replace("D", "E").replace("d", "e")
    pattern = _ARRAY_LOCAL_RE if module_level is False else _ARRAY_MODULE_RE
    return txt if pattern.fullmatch(txt) else None


def _decl_line(decl: Any) -> int | None:
    for n in walk(decl):
        item = getattr(n, "item", None)
        if item is not None and getattr(item, "span", None):
            return int(item.span[0])
    return None


def extract(
    path: Path,
    *,
    extern_names: set[str] | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """Every parameter and hoisted literal in one Fortran source file.

    ``extern_names`` are constants this module use-imports and that a sibling
    translation already defines; they count as known when classifying an
    expression. The command-line ancestor learned them by ``exec``-ing a
    generated Python file, which made the frontend's answer depend on a
    previously generated artifact being present and importable.
    """
    ast = parse(path)
    mod_name, mod_spec, sub_scope = _scope_of(ast, path, scope)

    known: set[str] = set(extern_names or ())
    module_parameters: list[dict[str, Any]] = []
    char_values: dict[str, str] = {}  # every character parameter's value so far

    for decl in walk(mod_spec, f03.Type_Declaration_Stmt) if mod_spec is not None else []:
        type_spec, attr_list, _ = decl.children
        attrs = [str(a).upper() for a in (attr_list.children if attr_list else [])]
        if "PARAMETER" not in attrs:
            continue
        base = (
            str(type_spec.children[0]).upper()
            if isinstance(type_spec, f03.Intrinsic_Type_Spec)
            else "?"
        )
        line = _decl_line(decl)
        for ent in walk(decl, f03.Entity_Decl):
            name = str(ent.children[0]).lower()
            init = str(ent.children[3].children[1]) if ent.children[3] is not None else None
            rec: dict[str, Any] = {
                "name": name,
                "base_type": base,
                "init_expr": init,
                "line": line,
            }
            # ``[...]`` only, as the pipeline has it. A ``(/.../)`` parameter
            # goes to the classifier instead, which spells its elements
            # ``np.float64('...')`` rather than as kind-stripped source text
            # -- two routes with two spellings, and this is the one that
            # decides which a parameter takes.
            array_text = (
                _array_literal(init, module_level=True)
                if init and "[" in init and ent.children[1] is not None
                else None
            )
            if array_text is not None:
                rec["kind"], rec["payload"] = "array", array_text
            else:
                rec["kind"], rec["payload"] = classify_init(
                    init or "",
                    known,
                    char_values,
                    char_length(str(type_spec), str(ent)) if base == "CHARACTER" else None,
                )
            if rec["kind"] == "str":
                char_values[name] = rec["payload"]
            module_parameters.append(rec)
            if rec["kind"] != "skip":
                # Only a parameter that got a value is a name later ones may
                # be written in terms of. Adding every declared name meant an
                # expression over a *skipped* parameter was emitted anyway,
                # referring to something the constants module never defines --
                # a NameError at import, which is how twenty units of the
                # corpus failed to load. The pipeline adds the name inside the
                # branches that emit, and so does this now.
                known.add(name)

    local_parameters: list[dict[str, Any]] = []
    literal_map: dict[str, dict[str, str]] = {}
    hoisted: dict[str, dict[str, Any]] = {}

    def hoist(text: str, is_real: bool, location: str, subprogram: str) -> None:
        cname = canon_name(text, is_real)
        entry = hoisted.setdefault(
            cname,
            {
                "value": _normalize_real(text) if is_real else strip_kind(text),
                "is_real": is_real,
                "locations": [],
            },
        )
        entry["locations"].append(location)
        literal_map[subprogram][text] = cname

    for sub in walk(sub_scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        st = walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        sname = str(st.children[1]).lower()
        literal_map[sname] = {}

        spec = next((c for c in sub.children if isinstance(c, f03.Specification_Part)), None)
        local_chars = dict(char_values)  # the module's values, then this subprogram's
        if spec is not None:
            # F77 separate form: PARAMETER (NAME = expr, ...)
            for pstmt in walk(spec, f03.Parameter_Stmt):
                for pdef in walk(pstmt, f03.Named_Constant_Def):
                    pname = str(pdef.children[0]).lower()
                    init = str(pdef.children[1])
                    kind, payload = classify_init(init, known, local_chars)
                    if kind == "str":
                        local_chars[pname] = payload
                    local_parameters.append(
                        {
                            "subprogram": sname,
                            "name": pname,
                            "const": f"{sname.upper()}__{pname.upper()}",
                            "form": "parameter_stmt",
                            "init_expr": init,
                            "kind": kind,
                            "payload": payload,
                        }
                    )
            for decl in walk(spec, f03.Type_Declaration_Stmt):
                local_type_spec, attr_list, _ = decl.children
                attrs = [str(a).upper() for a in (attr_list.children if attr_list else [])]
                if "PARAMETER" not in attrs:
                    continue
                for ent in walk(decl, f03.Entity_Decl):
                    pname = str(ent.children[0]).lower()
                    init = str(ent.children[3].children[1]) if ent.children[3] is not None else None
                    const = f"{sname.upper()}__{pname.upper()}"
                    array_text = (
                        _array_literal(init, module_level=False)
                        if init and init.strip().startswith("[") and ent.children[1] is not None
                        else None
                    )
                    if array_text is not None:
                        kind, payload = "array", array_text
                    else:
                        kind, payload = classify_init(
                            init or "",
                            known,
                            local_chars,
                            char_length(str(local_type_spec), str(ent)),
                        )
                    if kind == "str":
                        local_chars[pname] = payload
                    local_parameters.append(
                        {
                            "subprogram": sname,
                            "name": pname,
                            "const": const,
                            "form": "declaration",
                            "init_expr": init,
                            "kind": kind,
                            "payload": payload,
                        }
                    )
                    literal_map[sname][f"@param:{pname}"] = const

            # Declaration bounds take part in the zero-literal rule too: the
            # prologue that allocates capeten(pcols, 5) must name the 5.
            for shp in walk(spec, f03.Explicit_Shape_Spec):
                for text, is_real, _line in literals_with_lines(shp):
                    if is_real or is_whitelisted(text, is_real):
                        continue
                    hoist(text, False, f"{sname}:decl", sname)

            # A DATA statement's values are literals like any other, and they
            # live in the specification part, so the sweep over the execution
            # part below never sees them. Without a name here the translation
            # of the DATA statement has nothing to emit and refuses -- which
            # is what it was doing, on statements the pipeline translates.
            for data in walk(spec, f03.Data_Stmt):
                for group in data.children or ():
                    children = getattr(group, "children", None)
                    if not children or len(children) < 2:
                        continue
                    for text, is_real, _line in literals_with_lines(children[1]):
                        if is_whitelisted(text, is_real):
                            continue
                        hoist(text, is_real, f"{sname}:data", sname)

        exec_part = next((c for c in sub.children if isinstance(c, f03.Execution_Part)), None)
        if exec_part is None:
            continue
        for text, is_real, line in literals_with_lines(exec_part):
            if is_whitelisted(text, is_real):
                continue
            hoist(text, is_real, f"{sname}:{line}", sname)

    return {
        "source_file": str(path),
        "module": mod_name,
        "module_parameters": module_parameters,
        "local_parameters": local_parameters,
        "hoisted_literals": hoisted,
        "literal_map": literal_map,
    }
