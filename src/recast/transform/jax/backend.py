"""The JAX backend: ``<module>_numpy.py`` -> ``<module>_jax.py``.

Migrated from ``13_jax_backend/jaxize.py`` in the agent-produced script
collection, faithfully: ``tools/jax_diff.py`` holds every emitted byte to that
script across 109 modules, so the rules below are its rules and not a
reinterpretation of them. What was dropped is the command-line front end,
because the caller is now a ``Transform``; what was changed is the emitted
header, which is the one thing that harness deliberately does not compare.

**It carries no type annotations, on purpose.** What holds this module is the
byte-for-byte diff rather than a type checker, and that diff reaches 96% of it;
the exemption and the condition that would reverse it are in ``pyproject.toml``
beside the mypy override. The short version: annotate this before widening the
emitter's subset, not before anything else, because widening the subset is what
retires the diff.

It transforms the *Python* the NumPy backend emitted. Fortran is never
re-parsed -- the anchor is the validated ``<module>_numpy.py``, and its
emission grammar is stable enough to pattern-match. That is why the port
recipe's Transform runs the NumPy translation first and this second, inside
one ``apply``: they are two halves of one pass, not two stages.

**This backend cannot be bit-exact and that is not a defect.** XLA lowers
transcendentals to its own implementations rather than to libm and fuses
multiply-add, so the honest ceiling is a ULP bound; see
``recast.verify.tolerance`` for the gate that awards one. Anything outside the
subset raises ``JaxQueue`` and the subprogram is host-delegated to the NumPy
module rather than guessed at.

Emission rules, unchanged from the source:
  - kernel eligibility mirrors the Numba backend's -- numeric arguments only,
    no derived types, no module-state writes; a subprogram whose NumPy body
    still holds an ``AGENT_QUEUE`` placeholder is delegated, because the
    anchor itself is incomplete;
  - ``for v in range(a, b)`` becomes ``lax.fori_loop`` with the body's
    assigned-name set threaded as an explicit carry tuple; step -1 loops are
    index-remapped over a hoisted trip count;
  - ``if``/``elif``/``else`` becomes ``lax.cond`` carrying the union of both
    branches' assigned names; an absent ``else`` is the identity, which is
    Fortran's one-armed IF;
  - ``and``/``or``/``not`` become ``jnp.logical_*``, because Fortran's
    ``.AND.``/``.OR.`` do not short-circuit;
  - subscript stores become functional ``x = x.at[idx].set(v)``;
  - intra-module calls are rewritten to the callee's kernel with its state
    closure appended, iterating to a fixpoint so a caller never references a
    kernel that failed to emit;
  - ``math.*``/``np.*`` become ``jnp.*``; the ``_f_*`` shims come from the
    runtime this package emits beside the module.

Known limits, also unchanged: ``while``, ``AugAssign``, derived types,
optional ``want_`` parameters and ``endrun`` raises inside kernels are all
queued, the DO-variable's end value is not reproduced, and B-block provenance
comments do not survive ``ast.unparse`` -- which is why ``static.rwset``
cannot gate a jaxized module and the port recipe does not ask it to.
"""

import ast
import copy
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

NP_RENAME = {"empty": "zeros"}
# scalar/dtype constructors stay in numpy land: they run at trace time on
# literals (jnp.float64('0.01') rejects strings) and trace as constants
NP_KEEP = {"float64", "float32", "int64", "int32", "int8", "bool_"}


class JaxQueue(Exception):
    """Subprogram outside the supported subset -> host-delegate."""


# ---------------------------------------------------------------- eligibility


def eligible(sub):
    if any(a["dtype"] == "str" and (a.get("dims") or a["intent"] != "IN") for a in sub["args"]):
        return False  # a scalar IN str is a static argument; the rest are not flat
    if any("UNKNOWN(TYPE" in str(a["dtype"]) for a in sub["args"]):
        return False
    rd = str(sub.get("result_dtype") or "")
    if rd == "str" or "UNKNOWN(TYPE" in rd:
        return False
    return True


def inelig_reason(sub):
    if any(a["dtype"] == "str" and (a.get("dims") or a["intent"] != "IN") for a in sub["args"]):
        return "[elig] str arg"
    if any("UNKNOWN(TYPE" in str(a["dtype"]) for a in sub["args"]) or "UNKNOWN(TYPE" in str(
        sub.get("result_dtype") or ""
    ):
        return "[elig] derived type"
    return "[elig] calls outside kernel set"


def anchor_incomplete(fn):
    """True if the numpy body still contains AGENT_QUEUE placeholders —
    the anchor itself cannot run this subprogram, nothing to gate on."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "NotImplementedError"
        ):
            return True
    return False


def state_closure(
    subs: Mapping[str, Any], kernels: Collection[str], name: str, memo: dict[str, set[str]]
) -> set[str]:
    """Transitive module-state reads through intra-module kernel calls."""
    if name in memo:
        return memo[name]
    memo[name] = set()  # cycle guard
    st = set(subs[name]["module_state_read"])
    for c in subs[name]["calls"]:
        if c in kernels and c in subs:
            st |= state_closure(subs, kernels, c, memo)
    memo[name] = st
    return st


def write_closure(
    subs: Mapping[str, Any], kernels: Collection[str], name: str, memo: dict[str, set[str]]
) -> set[str]:
    """Transitive module-state writes through intra-module kernel calls."""
    if name in memo:
        return memo[name]
    memo[name] = set()  # cycle guard
    st = set(subs[name].get("module_state_written") or ())
    for c in subs[name]["calls"]:
        if c in kernels and c in subs:
            st |= write_closure(subs, kernels, c, memo)
    memo[name] = st
    return st


# ------------------------------------------------------------ expr rewriting


class ExprMap(ast.NodeTransformer):
    """math.X / np.X -> jnp.X (np.empty -> jnp.zeros, dtype ctors kept);
    and/or/not -> jnp.logical_* (traced bools reject Python short-circuit;
    matches Fortran .AND./.OR. non-short-circuit semantics)."""

    def visit_BinOp(self, node):
        # ``x ** 0.67`` -- a fractional constant exponent has an infinite
        # derivative at x == 0 (fwet from a dry canopy), and a linearized
        # graph turns inf * 0 into NaN that poisons every tangent downstream.
        # For x > 0 the value is untouched to the bit; at x == 0 the primal
        # is the same 0 and the subgradient taken is 0; a negative base was
        # NaN in Fortran already.
        self.generic_visit(node)
        fractional = (
            isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, float)
            and not float(node.right.value).is_integer()
        ) or isinstance(node.right, ast.Name)  # a run-set exponent (fwet_exponent)
        if isinstance(node.op, ast.Pow) and fractional:
            base = node.left
            positive = ast.Compare(
                left=copy.deepcopy(base), ops=[ast.Gt()], comparators=[ast.Constant(0.0)]
            )
            safe = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()), attr="where", ctx=ast.Load()
                ),
                args=[positive, base, ast.Constant(1.0)],
                keywords=[],
            )
            return ast.copy_location(
                ast.BinOp(
                    left=ast.BinOp(left=safe, op=ast.Pow(), right=node.right),
                    op=ast.Mult(),
                    right=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="where",
                            ctx=ast.Load(),
                        ),
                        args=[copy.deepcopy(positive), ast.Constant(1.0), ast.Constant(0.0)],
                        keywords=[],
                    ),
                ),
                node,
            )
        return node

    def visit_Subscript(self, node):
        # ``MDAYLEAP[mcmnth - 1]``: a module constant is a numpy array, and
        # numpy indexing with a traced index calls __array__ on the tracer.
        # Read the table through jnp instead; a static index is unharmed.
        self.generic_visit(node)
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id.isupper()
            and len(node.value.id) > 1
        ):
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()), attr="asarray", ctx=ast.Load()
                ),
                args=[node.value],
                keywords=[],
            )
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            if node.value.id == "math":
                return ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()), attr=node.attr, ctx=node.ctx
                    ),
                    node,
                )
            if node.value.id == "np" and node.attr not in NP_KEEP:
                return ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr=NP_RENAME.get(node.attr, node.attr),
                        ctx=node.ctx,
                    ),
                    node,
                )
        return node

    @staticmethod
    def _jnp_call(fn, args):
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=fn, ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        expr = node.values[0]
        for v in node.values[1:]:
            expr = self._jnp_call(fn, [expr, v])
        return ast.copy_location(expr, node)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(self._jnp_call("logical_not", [node.operand]), node)
        return node


class CallRewrite(ast.NodeTransformer):
    """Intra-module calls to emitted kernels -> _<name>_k_impl(...) with
    the callee's state closure appended. The callee signature is
    [required..., closure..., optional-with-defaults...] (numbaize
    convention), so positional actuals beyond the required count are
    normalized to keywords before the closure is appended positionally.
    A call to a module subprogram that is NOT being emitted must queue
    the caller: leaving the bare name would resolve to the host-delegated
    numpy function at trace time and run numpy/math code on tracers."""

    def __init__(self, call_map, known_subs):
        self.map = call_map
        self.subs = known_subs

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Name) and f.id == "_f_ecall" and node.args:
            # ``_f_ecall(split, x[i - 1, :])``: the elemental broadcast of an
            # emitted kernel is the runtime's jnp.vectorize over the kernel's
            # implementation, its state closure appended; of a subprogram
            # not being emitted, a host function no tracer can run.
            callee = node.args[0]
            if isinstance(callee, ast.Name) and callee.id in self.map:
                info = self.map[callee.id]
                node.args = [
                    ast.Name(id=f"_{callee.id}_k_impl", ctx=ast.Load()),
                    *node.args[1:],
                    *[ast.Name(id=c, ctx=ast.Load()) for c in info["closure"]],
                ]
            elif isinstance(callee, ast.Name) and callee.id in self.subs:
                raise JaxQueue(f"elemental call of non-emitted subprogram {callee.id}")
            return node
        if isinstance(f, ast.Name):
            if f.id in self.map:
                info = self.map[f.id]
                pos = list(node.args)
                if len(pos) > info["nreq"]:
                    for i, extra in enumerate(pos[info["nreq"] :]):
                        node.keywords.append(
                            ast.keyword(arg=info["params"][info["nreq"] + i], value=extra)
                        )
                    pos = pos[: info["nreq"]]
                node.func = ast.Name(id=f"_{f.id}_k_impl", ctx=ast.Load())
                node.args = pos + [ast.Name(id=c, ctx=ast.Load()) for c in info["closure"]]
            elif f.id in self.subs:
                raise JaxQueue(f"calls non-emitted subprogram {f.id}")
        return node


# ------------------------------------------------------- statement lowering


def _has_cycle(stmts) -> bool:
    """Whether a ``continue`` of *this* loop sits in ``stmts`` -- at the top
    level or under an ``if``; one inside a nested ``for`` is that loop's."""
    for s in stmts:
        if isinstance(s, ast.Continue):
            return True
        if isinstance(s, ast.If) and (_has_cycle(s.body) or _has_cycle(s.orelse)):
            return True
    return False


def _cycle_to_else(stmts, rest):
    """``stmts`` followed by ``rest``, with every ``continue`` folded away.

    Fortran's ``if ( ... ) then ... cycle end if`` followed by the rest of
    the loop body (CLUBB's interpolators) is ``if c: A else: R`` -- the
    continuation of the loop body moves into the branches that do not
    cycle. Done on the Python AST before the fori_loop lowering, which has
    no place for a ``continue``: a ``lax.cond`` branch is a function.
    """
    out: list[ast.stmt] = []
    for at, s in enumerate(stmts):
        if isinstance(s, ast.Continue):
            return out  # what follows is never reached on this path
        if isinstance(s, ast.If) and (_has_cycle(s.body) or _has_cycle(s.orelse)):
            tail = [*stmts[at + 1 :], *rest]
            folded = ast.If(
                test=s.test,
                body=_cycle_to_else(s.body, copy.deepcopy(tail)) or [ast.Pass()],
                orelse=_cycle_to_else(s.orelse, copy.deepcopy(tail)),
            )
            return [*out, ast.copy_location(folded, s)]
        out.append(s)
    return [*out, *rest]


def _assigned_names(stmts):
    """Names stored by Assign statements, first-assignment order
    (nested fori_loop/cond results arrive as Tuple targets; static
    Python ifs survive lowering, so recurse into them)."""
    out = []

    def add(n):
        if n not in out:
            out.append(n)

    for s in stmts:
        if isinstance(s, ast.Assign):
            for t in s.targets:
                # The hoisted bounds of a step -1 loop (``_hi_n``, ``_cnt_n``)
                # and a tuple-target temporary (``_tn``) are assigned in the
                # body before their use and never read after it; carried, they
                # would have to be initialized at the enclosing level, where
                # nothing assigns them.
                # ``_out`` is the anchor's call-result tuple, assigned and
                # unpacked on consecutive lines of one block: the block's own,
                # never a carry (its shape changes from call to call).
                if isinstance(t, ast.Name) and not re.fullmatch(r"_(?:hi_|cnt_|t)\d+|_out", t.id):
                    add(t.id)
                elif isinstance(t, ast.Tuple):
                    for e in t.elts:
                        if isinstance(e, ast.Name) and not re.fullmatch(r"_t\d+|_out", e.id):
                            add(e.id)
        elif isinstance(s, ast.If):
            for n in _assigned_names(s.body) + _assigned_names(s.orelse):
                add(n)
    return out


def _static_test(test):
    """True for branch conditions decidable at trace time per the
    translate.py grammar: `x is [not] None` (Fortran PRESENT) and bare
    `want_*` sentinels (optional-output flags, static under jit)."""
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Is, ast.IsNot))
    ):
        return True
    if isinstance(test, ast.Name) and test.id.startswith("want_"):
        return True

    def constant(node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            return node.id.isupper() and len(node.id) > 1
        if isinstance(node, ast.BinOp):
            return constant(node.left) and constant(node.right)
        if isinstance(node, ast.UnaryOp):
            return constant(node.operand)
        return False

    # ``RUNGE_KUTTA_TYPE == I_10``: a comparison over module constants
    # decides at trace time; lowering it to lax.cond would put a Python-int
    # assignment in one arm of a traced carry.
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        return constant(test.left) and all(constant(c) for c in test.comparators)
    if isinstance(test, ast.BoolOp):
        return all(_static_test(v) for v in test.values)
    return False


def _names(ids, ctx):
    return [ast.Name(id=i, ctx=ctx()) for i in ids]


def _trace_constant_stores(stmts):
    """Names whose every store is a trace-time constant (a literal, an
    upper-case constant expression, or ``int()`` of one): re-established
    identically on every pass, so carrying them through a loop or cond
    would only force a Python value into a traced carry slot --
    ``nrk_steps = int(RUNGE_KUTTA_TYPE / I_10)`` inside the substep loop."""

    def const(node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            return node.id.isupper() and len(node.id) > 1
        if isinstance(node, ast.BinOp):
            return const(node.left) and const(node.right)
        if isinstance(node, ast.UnaryOp):
            return const(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("int", "float")
            and len(node.args) == 1
        ):
            return const(node.args[0])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "jnp")
            and node.func.attr in ("int32", "int64", "float32", "float64")
            and len(node.args) == 1
        ):
            # A dtype-ctor around a constant (the guard-init spelling of a
            # zero) is still a trace-time constant store.
            return const(node.args[0])
        return False

    verdicts: dict[str, bool] = {}

    def note(name: str, good: bool) -> None:
        verdicts[name] = verdicts.get(name, True) and good

    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                targets = (
                    node.targets[0].elts
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    else node.targets
                )
                single = len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                for t in targets:
                    if isinstance(t, ast.Name):
                        note(t.id, single and const(node.value))
            elif isinstance(node, (ast.AugAssign, ast.For)):
                target = getattr(node, "target", None)
                if isinstance(target, ast.Name):
                    note(target.id, False)
    return {n for n, good in verdicts.items() if good}


def _const_int(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


class KernelLowerer:
    BANNED = (
        ast.While,
        ast.AugAssign,
        ast.With,
        ast.Try,
        ast.Raise,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
    )

    def __init__(self, bound=()):
        self.n = 0
        # The names bound so far, in statement order: a loop carries only a
        # name that exists before it. One first assigned inside the body
        # (the anchor's ``_out = callee(...)`` result tuple, unpacked on the
        # next lines) is the body's own -- carried, its initial value would
        # be read before any assignment, and its shape may change between
        # two calls in the same body.
        self.bound = set(bound)

    def _bind(self, stmts):
        for name in _assigned_names(stmts):
            self.bound.add(name)

    def lower_block(self, stmts, depth):
        out: list[ast.stmt] = []
        for s in stmts:
            start = len(out)
            if isinstance(s, self.BANNED):
                raise JaxQueue(f"unsupported stmt {type(s).__name__}")
            if isinstance(s, ast.Continue | ast.Break):
                # A CYCLE the loop pass could not fold into a branch, or an
                # EXIT: neither has a place in a fori_loop body. Delegated,
                # not emitted -- a ``continue`` inside a ``lax.cond`` branch
                # is a SyntaxError that takes the whole module down.
                raise JaxQueue(f"{type(s).__name__.lower()} inside a lowered loop")
            if isinstance(s, ast.Return) and depth > 0:
                raise JaxQueue("return inside loop/branch body")
            if isinstance(s, ast.For):
                out.extend(self.lower_for(s, depth))
            elif isinstance(s, ast.If):
                out.extend(self.lower_if(s, depth))
            elif isinstance(s, ast.Assign):
                lowered = self.lower_assign(s)
                out.extend(lowered if isinstance(lowered, list) else [lowered])
            else:
                out.append(s)  # Expr (docstring), Return at top level, Pass
            # Bound by what was *emitted*: a lowered loop or branch binds its
            # carried names and nothing else -- a body-local of one loop is
            # not bound for the next.
            self._bind(out[start:])
        return out

    def lower_assign(self, s):
        if len(s.targets) != 1:
            raise JaxQueue("multi-target assign")
        t = s.targets[0]
        if isinstance(t, ast.Tuple):
            # multi-output intra-module call: a, b = _callee_k_impl(...)
            if all(isinstance(e, ast.Name) for e in t.elts):
                return s
            # ``lo[i - 1, :], hi[i - 1, :] = split(...)`` (CLUBB's column
            # loops): the tuple through a temporary, each element a store
            # of its own -- a subscript store is lowered on its own line.
            self.n += 1
            tmp = f"_t{self.n}"
            stores = [ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=s.value)]
            for index, element in enumerate(t.elts):
                piece = ast.Subscript(
                    value=ast.Name(id=tmp, ctx=ast.Load()),
                    slice=ast.Constant(value=index),
                    ctx=ast.Load(),
                )
                stores.append(self.lower_assign(ast.Assign(targets=[element], value=piece)))
            return stores
        if isinstance(t, ast.Name):
            # strengthen scalar literal inits so fori_loop carries keep a
            # stable strong dtype (0.0 -> jnp.float64(0.0))
            if isinstance(s.value, ast.Constant) and not isinstance(s.value.value, (bool, str)):
                ctor = (
                    "float64"
                    if isinstance(s.value.value, float)
                    else "int32"
                    if isinstance(s.value.value, int)
                    else None
                )
                if ctor:
                    s = ast.Assign(
                        targets=s.targets,
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()), attr=ctor, ctx=ast.Load()
                            ),
                            args=[s.value],
                            keywords=[],
                        ),
                    )
            return s
        if isinstance(t, ast.Subscript):
            if not isinstance(t.value, ast.Name):
                raise JaxQueue("subscript store base is not a plain name")
            base = t.value.id
            # x[...] = c  ->  x = jnp.full_like(x, c)
            if (
                isinstance(t.slice, ast.Constant)
                and t.slice.value is Ellipsis
                and isinstance(s.value, ast.Constant)
            ):
                return ast.Assign(
                    targets=[ast.Name(id=base, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="full_like",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=base, ctx=ast.Load()), s.value],
                        keywords=[],
                    ),
                )
            at = ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id=base, ctx=ast.Load()), attr="at", ctx=ast.Load()
                ),
                slice=t.slice,
                ctx=ast.Load(),
            )
            return ast.Assign(
                targets=[ast.Name(id=base, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=at, attr="set", ctx=ast.Load()),
                    args=[s.value],
                    keywords=[],
                ),
            )
        raise JaxQueue(f"unsupported assign target {type(t).__name__}")

    def lower_for(self, s, depth):
        if s.orelse:
            raise JaxQueue("for-else")
        if not isinstance(s.target, ast.Name):
            raise JaxQueue("tuple loop target")
        it = s.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
        ):
            raise JaxQueue("non-range for")
        step = 1
        # Annotated because the first branch would otherwise pin these to
        # Constant and the others assign general expressions to them.
        lo: ast.expr
        hi: ast.expr
        if len(it.args) == 1:
            lo, hi = ast.Constant(value=0), it.args[0]
        elif len(it.args) == 2:
            lo, hi = it.args
        elif len(it.args) == 3:
            step = _const_int(it.args[2])
            lo, hi = it.args[0], it.args[1]
            if step not in (1, -1):
                raise JaxQueue("range step not +-1")
        else:
            raise JaxQueue("malformed range")

        bound_before = set(self.bound)
        self.bound.add(s.target.id)
        body = self.lower_block(_cycle_to_else(s.body, []), depth + 1)
        settled = _trace_constant_stores(body)
        carried = [
            n
            for n in _assigned_names(body)
            if n != s.target.id and n not in settled and n in bound_before
        ]
        if not carried:
            raise JaxQueue("loop with no carried effects")
        self.n += 1
        fname = f"_body_{self.n}"
        # Through the runtime's ``_f_fori``: a trip count that is static
        # and empty (a loop over an array's zero-extent axis -- CLUBB's
        # scalar tracers under sclr_dim = 0) is skipped rather than traced,
        # because JAX refuses any index into a size-0 axis at trace time.
        carry_call = ast.Call(func=ast.Name(id="_f_fori", ctx=ast.Load()), args=[], keywords=[])
        result = ast.Assign(
            targets=[ast.Tuple(elts=_names(carried, ast.Store), ctx=ast.Store())], value=carry_call
        )
        init = ast.Tuple(elts=_names(carried, ast.Load), ctx=ast.Load())

        if step == 1:
            fn = self._carry_fn(fname, [s.target.id, "_c"], carried, body)
            carry_call.args = [lo, hi, ast.Name(id=fname, ctx=ast.Load()), init]
            return [fn, result]

        # step -1: Fortran DO k=hi,lo,-1 arrived as range(hi, stop, -1)
        # iterating hi..stop+1. Remap: t in [0, hi-stop), k = hi - t.
        # Bounds are hoisted (evaluated once, Fortran DO semantics).
        hi_name, cnt_name = f"_hi_{self.n}", f"_cnt_{self.n}"
        pre = [
            ast.Assign(targets=[ast.Name(id=hi_name, ctx=ast.Store())], value=lo),
            ast.Assign(
                targets=[ast.Name(id=cnt_name, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=hi_name, ctx=ast.Load()), op=ast.Sub(), right=hi),
            ),
        ]
        remap = ast.Assign(
            targets=[ast.Name(id=s.target.id, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=hi_name, ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Name(id="_r", ctx=ast.Load()),
            ),
        )
        fn = self._carry_fn(fname, ["_r", "_c"], carried, [remap, *body])
        carry_call.args = [
            ast.Constant(value=0),
            ast.Name(id=cnt_name, ctx=ast.Load()),
            ast.Name(id=fname, ctx=ast.Load()),
            init,
        ]
        return [*pre, fn, result]

    def lower_if(self, s, depth):
        """if/elif/else -> lax.cond with the union of both branches'
        assigned names as carry. An absent else branch becomes the
        identity (carry passes through unchanged) — Fortran one-armed IF
        semantics. NOTE lax.cond under vmap lowers to select (both sides
        evaluated); guarded-domain expressions inside branches will need
        the double-where pattern at Level-2 vectorization, not here.

        PRESENT/want_ tests are trace-time static: they stay as Python
        ifs (executed during tracing; the dead branch — which may touch
        a None arg — is never traced)."""
        body = self.lower_block(s.body, depth + 1)
        orelse = self.lower_block(s.orelse, depth + 1)
        if _static_test(s.test):
            return [ast.If(test=s.test, body=body, orelse=orelse or [])]
        carried = _assigned_names(body)
        for n in _assigned_names(orelse):
            if n not in carried:
                carried.append(n)
        if not carried:
            trivial = all(
                isinstance(st, ast.Pass)
                or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))
                for st in [*body, *orelse]
            )
            if trivial:
                return []  # a guard around dropped logs/aborts: nothing to carry
            raise JaxQueue("IF with no carried effects")
        self.n += 1
        t_name, f_name = f"_true_{self.n}", f"_false_{self.n}"
        t_fn = self._carry_fn(t_name, ["_c"], carried, body)
        f_fn = self._carry_fn(f_name, ["_c"], carried, orelse)
        cond = ast.Assign(
            targets=[ast.Tuple(elts=_names(carried, ast.Store), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="lax", ctx=ast.Load()), attr="cond", ctx=ast.Load()
                ),
                args=[
                    s.test,
                    ast.Name(id=t_name, ctx=ast.Load()),
                    ast.Name(id=f_name, ctx=ast.Load()),
                    ast.Tuple(elts=_names(carried, ast.Load), ctx=ast.Load()),
                ],
                keywords=[],
            ),
        )
        return [t_fn, f_fn, cond]

    @staticmethod
    def _carry_fn(name, params, carried, stmts):
        """def name(*params): (carried) = _c; <stmts>; return (carried)"""
        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=p) for p in params],
                vararg=None,
                kwarg=None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Tuple(elts=_names(carried, ast.Store), ctx=ast.Store())],
                    value=ast.Name(id="_c", ctx=ast.Load()),
                ),
                *stmts,
                ast.Return(value=ast.Tuple(elts=_names(carried, ast.Load), ctx=ast.Load())),
            ],
            decorator_list=[],
            returns=None,
        )


# ----------------------------------------------------------------- emission


def split_params(fn_src):
    """(required names, optional names, default exprs) of the original fn."""
    params = [a.arg for a in fn_src.args.args]
    n_def = len(fn_src.args.defaults)
    n_req = len(params) - n_def
    return params[:n_req], params[n_req:], fn_src.args.defaults


def emit_kernel(fn_src, sub, closure, call_map=None, known_subs=None, writes=()):
    """Original numpy FunctionDef -> unparsed _<name>_k_impl source.

    Kernel signature: [required..., closure..., optional-with-defaults].
    Optional params keep their None/False defaults; their PRESENT/want_
    branches stay Python ifs and resolve at trace time. ``writes`` are the
    module-state names the body (or a callee) stores: the closure passes
    their current values in, the ``global`` statement is dropped (the name
    is a local from here), every return carries them out, and a call to a
    writing kernel binds them back -- state threads, nothing is written to
    a module under tracing."""
    fn = copy.deepcopy(fn_src)
    fn.name = f"_{sub['name']}_k_impl"
    fn.decorator_list = []
    if fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise JaxQueue("kwonly/vararg params")
    if any(not isinstance(d, ast.Constant) for d in fn.args.defaults):
        raise JaxQueue("non-literal default")
    n_def = len(fn.args.defaults)
    head = fn.args.args[: len(fn.args.args) - n_def]
    tail = fn.args.args[len(fn.args.args) - n_def :]
    fn.args.args = head + [ast.arg(arg=st) for st in sorted(closure)] + tail
    write_list = sorted(writes)
    if write_list:
        kept = []
        for stmt in fn.body:
            if isinstance(stmt, ast.Global):
                if not set(stmt.names) <= set(write_list):
                    raise JaxQueue(f"global beyond the written closure: {stmt.names}")
                continue
            kept.append(stmt)
        fn.body = kept

        class Extend(ast.NodeTransformer):
            def visit_Return(self, node):
                elems = (
                    list(node.value.elts)
                    if isinstance(node.value, ast.Tuple)
                    else ([node.value] if node.value is not None else [])
                )
                elems += [ast.Name(id=w, ctx=ast.Load()) for w in write_list]
                value = elems[0] if len(elems) == 1 else ast.Tuple(elts=elems, ctx=ast.Load())
                return ast.Return(value=value)

            def visit_FunctionDef(self, node):
                return node  # a nested function's returns are its own

        for at, stmt in enumerate(fn.body):
            fn.body[at] = Extend().visit(stmt)
    _bind_writer_calls(fn, call_map or {})
    ExprMap().visit(fn)
    CallRewrite(call_map or {}, known_subs or set()).visit(fn)
    fn.body = KernelLowerer(bound={a.arg for a in fn.args.args}).lower_block(fn.body, 0)
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


def _bind_writer_calls(fn, call_map):
    """A statement calling a writing kernel binds the written state:
    ``a, b = f(...)`` -> ``a, b, X = f(...)`` and a bare ``f(...)`` becomes
    the assignment. A writer nested inside an expression has nowhere to put
    the state and queues the caller."""

    def writer(call):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in call_map
        ):
            return call_map[call.func.id].get("writes") or []
        return []

    class Bind(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node

        def _stmt(self, node):
            call = node.value
            writes = writer(call)
            for inner in ast.walk(node):
                if inner is not call and writer(inner):
                    raise JaxQueue(
                        f"a state-writing kernel inside an expression: {ast.unparse(inner)}"
                    )
            if not writes:
                return node
            targets = (
                list(node.targets[0].elts)
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                else ([node.targets[0]] if isinstance(node, ast.Assign) else [])
            )
            targets += [ast.Name(id=w, ctx=ast.Store()) for w in writes]
            target = targets[0] if len(targets) == 1 else ast.Tuple(elts=targets, ctx=ast.Store())
            return ast.copy_location(ast.Assign(targets=[target], value=call), node)

        def visit_Assign(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call):
                return self._stmt(node)
            if any(writer(inner) for inner in ast.walk(node)):
                raise JaxQueue("a state-writing kernel inside an expression")
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call):
                return self._stmt(node)
            return node

    for at, stmt in enumerate(fn.body):
        fn.body[at] = Bind().visit(stmt)


def static_spec(fn_src, sub, traced_scalars=frozenset()):
    """(static_argnums, static_argnames) for the jitted kernel.

    argnums: scalar int32 IN args among the REQUIRED params — their
    positions are unchanged by the closure insertion (closure goes after
    the required block). Optional scalars stay traced. argnames: want_*
    sentinels (must be concrete Python bools; their branches are static)."""
    recs = {a["name"]: a for a in sub["args"]}
    req, opt, _ = split_params(fn_src)
    nums = []
    for pos, p in enumerate(req):
        r = recs.get(p)
        if r and not r.get("dims") and r["intent"] == "IN" and p not in traced_scalars:
            # int32 scalars and character dummies are run constants (filter
            # counts, calendar kinds). A float/int64 scalar is static ONLY
            # when it is module state (the ``__`` spelling: dtime_ml, dtstep
            # -- namelist configuration): an ordinary real dummy (xa, xb,
            # tol) must stay traced, or jvp/grad against it breaks.
            if r["dtype"] in ("int32", "str") or (
                r["dtype"] in ("int64", "float64") and "__" in r["name"]
            ):
                nums.append(pos)
    names = tuple(p for p in opt if p.startswith("want_"))
    return tuple(nums), names


def build_module(
    interface: dict[str, Any], tree: ast.Module, traced_scalars: frozenset[str] = frozenset()
) -> tuple[list[str], list[str], dict[str, str]]:
    """Emit all kernels of one module to a fixpoint.

    Returns (pieces, jitted, delegated) where pieces are source chunks
    (kernels + jit lines + wrappers + delegations + _JAX_KERNELS) and
    delegated maps name -> reason."""
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    subs = {s["name"]: s for s in interface["subprograms"]}

    delegated = {}
    kernels = set()
    for name, rec in subs.items():
        if name not in fns:
            delegated[name] = "[elig] no python fn"
        elif not eligible(rec):
            delegated[name] = inelig_reason(rec)
        elif anchor_incomplete(fns[name]):
            delegated[name] = "[anchor] AGENT_QUEUE placeholder in numpy body"
        else:
            kernels.add(name)

    memo: dict[str, set[str]] = {}
    wmemo: dict[str, set[str]] = {}
    wclosures = {n: sorted(write_closure(subs, kernels, n, wmemo)) for n in kernels}
    closures = {
        n: sorted(state_closure(subs, kernels, n, memo) | set(wclosures[n])) for n in kernels
    }

    # fixpoint: drop kernels that fail, re-emit (callers of dropped
    # kernels then fail their CallRewrite and drop too)
    emit_set = set(kernels)
    srcs: dict[str, str] = {}
    while True:
        srcs, failed = {}, {}
        call_map_all = {}
        for n in emit_set:
            req, _, _ = split_params(fns[n])
            call_map_all[n] = {
                "closure": closures[n],
                "writes": wclosures[n],
                "params": [a.arg for a in fns[n].args.args],
                "nreq": len(req),
            }
        for name in sorted(emit_set):
            call_map = {k: v for k, v in call_map_all.items() if k != name}
            try:
                srcs[name] = emit_kernel(
                    fns[name], subs[name], closures[name], call_map, set(subs), wclosures[name]
                )
            except JaxQueue as e:
                failed[name] = f"[emit] {e}"
        if not failed:
            break
        for n, r in failed.items():
            delegated[n] = r
            emit_set.discard(n)

    pieces, jitted = [], []
    for rec in interface["subprograms"]:
        name = rec["name"]
        if name not in srcs:
            continue
        sa, sn = static_spec(fns[name], rec, traced_scalars)
        req, opt, defaults = split_params(fns[name])
        pieces.append(srcs[name])
        jit_kw = f"static_argnums={sa!r}"
        if sn:
            jit_kw += f", static_argnames={sn!r}"
        pieces.append(f"_{name}_k = jax.jit(_{name}_k_impl, {jit_kw})")
        host = [f"_host.{s}" for s in closures[name]]
        # ``strict=False`` preserves the source's truncation. Tightening it
        # here would be a behaviour change the emitted bytes cannot reveal
        # unless a corpus module happens to have mismatched lengths.
        sig = req + [f"{p}={ast.unparse(d)}" for p, d in zip(opt, defaults, strict=False)]
        call = req + host + [f"{p}={p}" for p in opt]
        writes = wclosures[name]
        if not writes:
            pieces.append(
                "\n".join(
                    [
                        f"def {name}({', '.join(sig)}):",
                        '    """Host wrapper: state read from the validated numpy module."""',
                        f"    return _{name}_k({', '.join(call)})",
                    ]
                )
            )
        else:
            # A writing kernel returns [original outs..., writes...]; the
            # wrapper stores the state back on the host module and hands the
            # caller what the numpy signature promised.
            final = next(
                (
                    st
                    for st in reversed(fns[name].body)
                    if isinstance(st, ast.Return) and st.value is not None
                ),
                None,
            )
            n_orig = (
                len(final.value.elts)
                if final is not None and isinstance(final.value, ast.Tuple)
                else (1 if final is not None else 0)
            )
            lines = [
                f"def {name}({', '.join(sig)}):",
                '    """Host wrapper: state threads through; writes land on the host."""',
                f"    _res = _{name}_k({', '.join(call)})",
            ]
            if n_orig + len(writes) == 1:
                lines.append("    _res = (_res,)")
            for at, w in enumerate(writes):
                lines.append(f"    _host.{w} = _res[{n_orig + at}]")
            if n_orig == 0:
                lines.append("    return None")
            elif n_orig == 1:
                lines.append("    return _res[0]")
            else:
                lines.append(f"    return _res[:{n_orig}]")
            pieces.append("\n".join(lines))
        jitted.append(name)
    for rec in interface["subprograms"]:
        if rec["name"] in delegated and rec["name"] in fns:
            pieces.append(f"{rec['name']} = _host.{rec['name']}")
    pieces.append(f"_JAX_KERNELS = {sorted(jitted)!r}")
    return pieces, jitted, delegated


HEADER = '''"""Machine-generated by recast.transform.jax -- JAX backend (EXPERIMENTAL).

Loop-faithful lax emission from {module}_numpy.py. Cannot pass a bit-exact
gate: XLA lowers transcendentals to its own implementations and fuses
multiply-add. The gate is the ULP tier against the validated numpy module --
see recast.verify.tolerance. Do not hand-edit; fix the backend instead.
B-block provenance lives in the numpy module.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from {runtime} import *  # noqa: F401,F403  (enables x64, _f_* jnp shims)
from {constants} import *  # noqa: F401,F403
import {module}_numpy as _host

# Which device this module will run on. A ULP bound between a GPU and a CPU is
# a different claim from one between two CPUs, and the gate records what the
# module says rather than importing an accelerator to find out.
_DEVICE = str(jax.devices()[0])

'''


def emit_runtime() -> str:
    """The shim library's source, read from disk rather than imported.

    ``recast.transform.numpy.runtime`` renders itself through
    ``inspect.getsource``, which works because importing it costs nothing. This
    one cannot: importing the JAX shim imports JAX and enables x64, and
    emitting JAX code must not require an accelerator to be installed -- the
    emitter is pure AST work, and the script this came from needed nothing
    either. So the text is read beside this file.
    """
    return Path(__file__).with_name("runtime.py").read_text()
