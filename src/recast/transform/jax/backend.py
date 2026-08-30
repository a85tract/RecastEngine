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
    if sub["module_state_written"]:
        return False
    if any(a["dtype"] == "str" for a in sub["args"]):
        return False
    if any("UNKNOWN(TYPE" in str(a["dtype"]) for a in sub["args"]):
        return False
    rd = str(sub.get("result_dtype") or "")
    if rd == "str" or "UNKNOWN(TYPE" in rd:
        return False
    return True


def inelig_reason(sub):
    if sub["module_state_written"]:
        return "[elig] module-state write"
    if any(a["dtype"] == "str" for a in sub["args"]):
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


def state_closure(subs, kernels, name, memo):
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


# ------------------------------------------------------------ expr rewriting


class ExprMap(ast.NodeTransformer):
    """math.X / np.X -> jnp.X (np.empty -> jnp.zeros, dtype ctors kept);
    and/or/not -> jnp.logical_* (traced bools reject Python short-circuit;
    matches Fortran .AND./.OR. non-short-circuit semantics)."""

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
                if isinstance(t, ast.Name) and not re.fullmatch(r"_(?:hi_|cnt_|t)\d+", t.id):
                    add(t.id)
                elif isinstance(t, ast.Tuple):
                    for e in t.elts:
                        if isinstance(e, ast.Name) and not re.fullmatch(r"_t\d+", e.id):
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
    return False


def _names(ids, ctx):
    return [ast.Name(id=i, ctx=ctx()) for i in ids]


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

    def __init__(self):
        self.n = 0

    def lower_block(self, stmts, depth):
        out = []
        for s in stmts:
            if isinstance(s, self.BANNED):
                raise JaxQueue(f"unsupported stmt {type(s).__name__}")
            if isinstance(s, ast.Return) and depth > 0:
                raise JaxQueue("return inside loop/branch body")
            if isinstance(s, ast.For):
                out.extend(self.lower_for(s, depth))
            elif isinstance(s, ast.If):
                out.extend(self.lower_if(s, depth))
            elif isinstance(s, ast.Assign):
                out.append(self.lower_assign(s))
            else:
                out.append(s)  # Expr (docstring), Return at top level, Pass
        return out

    def lower_assign(self, s):
        if len(s.targets) != 1:
            raise JaxQueue("multi-target assign")
        t = s.targets[0]
        if isinstance(t, ast.Tuple):
            # multi-output intra-module call: a, b = _callee_k_impl(...)
            if all(isinstance(e, ast.Name) for e in t.elts):
                return s
            raise JaxQueue("tuple target with non-name elements")
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

        body = self.lower_block(s.body, depth + 1)
        carried = [n for n in _assigned_names(body) if n != s.target.id]
        if not carried:
            raise JaxQueue("loop with no carried effects")
        self.n += 1
        fname = f"_body_{self.n}"
        carry_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="lax", ctx=ast.Load()), attr="fori_loop", ctx=ast.Load()
            ),
            args=[],
            keywords=[],
        )
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


def emit_kernel(fn_src, sub, closure, call_map=None, known_subs=None):
    """Original numpy FunctionDef -> unparsed _<name>_k_impl source.

    Kernel signature: [required..., closure..., optional-with-defaults].
    Optional params keep their None/False defaults; their PRESENT/want_
    branches stay Python ifs and resolve at trace time."""
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
    ExprMap().visit(fn)
    CallRewrite(call_map or {}, known_subs or set()).visit(fn)
    fn.body = KernelLowerer().lower_block(fn.body, 0)
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


def static_spec(fn_src, sub):
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
        if r and r["dtype"] == "int32" and not r.get("dims") and r["intent"] == "IN":
            nums.append(pos)
    names = tuple(p for p in opt if p.startswith("want_"))
    return tuple(nums), names


def build_module(
    interface: dict[str, Any], tree: ast.Module
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
    closures = {n: sorted(state_closure(subs, kernels, n, memo)) for n in kernels}

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
                "params": [a.arg for a in fns[n].args.args],
                "nreq": len(req),
            }
        for name in sorted(emit_set):
            call_map = {k: v for k, v in call_map_all.items() if k != name}
            try:
                srcs[name] = emit_kernel(fns[name], subs[name], closures[name], call_map, set(subs))
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
        sa, sn = static_spec(fns[name], rec)
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
        pieces.append(
            "\n".join(
                [
                    f"def {name}({', '.join(sig)}):",
                    '    """Host wrapper: module state read from the validated numpy module."""',
                    f"    return _{name}_k({', '.join(call)})",
                ]
            )
        )
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
