"""``port.tree-jax``: a unit of a model tree retargeted to JAX.

The JAX backend's eligibility rule excludes a subprogram that takes a
derived-type dummy, and in a model tree that is every physics routine: the
object with a few hundred pointer components is the interface. The
backend is not widened for it -- ``tools/jax_diff.py`` holds its emitted
bytes to the script it came from -- and it does not need to be. The same
``FlatPlan`` that gives the oracle and the NumPy gate a flat adapter gives
this transform a *flat function*: the anchor's Python body with every
``object.component`` spelled as the flat argument the plan names, every
module-state read the plan carries spelled the same way, and the written
components returned. That function has numeric arguments only, no module
state and no object, so the backend lowers it as it lowers any kernel --
and ``<name>_flat`` in the ported module is a kernel, not a wrapper around
a host-delegated original.

What the rewrite does to the anchor's grammar:

* ``alias = obj.comp`` (the emitted form of an ``associate``) is dropped
  and every use of ``alias`` becomes the flat name -- in loads, in stores,
  as the base of a subscript store -- so there is one binding per
  component and a functional ``.at[].set`` on it is the component written;
* ``obj.comp``, ``_mod.var`` and ``_mod.obj.comp`` become their flat
  names; the plan already lists every one the body (and its callees)
  reaches;
* a call to another subprogram of the module that takes the object is
  rewritten to *its* flat function: the object's actual is replaced by the
  callee's components spelled on the caller's names, and the callee's
  written components are assigned back;
* a call into a bundled companion (``_mlw.latvap(...)``) is rewritten to
  the companion's own ported kernel, its module-state closure spelled as
  the flat state arguments the plan carries -- the companions are ported
  alongside, so a kernel never calls NumPy code on a traced value;
* the anchor's ``return`` becomes the flat signature's outputs, in the
  order the ``_SIGNATURES`` entry declares them.

And to every function of the module, flat or not, three things a kernel
cannot carry, each named on the candidate: an abort check (``if bad:
raise``, the anchor's ``endrun``) is dropped -- the recorded run it is
gated on never took it; ``int(x)`` / ``np.float64(x)`` on a traced value
become ``jnp`` casts; ``_f_copy_out(dst, src)`` becomes ``dst = src`` and a
tuple target with a subscript in it goes through temporaries.

Anything the rewrite cannot express leaves the subprogram host-delegated
with the reason on the candidate, the way the backend leaves its own
refusals. The gate is ``dump-replay`` plus ``differential.tolerance`` on
the flat signature, the same recording the NumPy translation was held
bit-exact against.
"""

from __future__ import annotations

import ast
import copy
import keyword
import re
from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.fortran.flatten import FlatPlan, plans_from_facts, signature
from recast.model import Candidate, Facts, Unit
from recast.transform.jax.translate import KernelToJax, _signatures_of
from recast.transform.numpy.tree import TreeConventions, TreeTranslation

__all__ = ["TreeToJax", "factory", "flattened_module"]

CASTS = ("float64", "float32", "int32", "int64")
WIDENED = ast.Name(id="_widened_", ctx=ast.Load())
"""A placeholder upper bound marking a slice this pass widened."""


TRACED_SCALARS: frozenset[str] = frozenset()
"""Scalar dummies the operator declares per-call-varying (config
``traced_scalars``): kept traced so the kernel does not recompile per
call -- the step counter ``itim`` above all."""


class NotFlat(Exception):
    """This subprogram's body cannot be spelled on the flat signature."""


def _py(name: str) -> str:
    return name + "_" if keyword.iskeyword(name) else name


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """``_alias -> module`` from the anchor's ``import X_numpy as _alias`` lines."""
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.asname and name.name.endswith("_numpy"):
                    aliases[name.asname] = name.name[: -len("_numpy")].lower()
    return aliases


class _Spelling:
    """What the plan says every object, component and state is called."""

    def __init__(self, plan: FlatPlan | None, aliases: dict[str, str]) -> None:
        self.dummies: dict[str, dict[str, str]] = {}
        self.state_objects: dict[tuple[str, str], dict[str, str]] = {}
        self.state_vars: dict[tuple[str, str], str] = {}
        self.states: set[str] = set()
        self.used: set[str] = set()  # the flat state names a body actually spelled
        if plan is not None:
            for obj in plan.objects:
                table = {c.name: c.flat for c in obj.components}
                if obj.kind == "dummy":
                    self.dummies[obj.name] = table
                else:
                    self.state_objects[(obj.module or "", obj.name)] = table
            for state in plan.states:
                self.state_vars[(state.module, state.name)] = state.flat
                self.states.add(state.flat)
        self.modules = aliases

    def of(self, node: ast.expr) -> str | None:
        """The flat name an attribute chain spells, or None."""
        found = self._of(node)
        if found is not None:
            self.used.add(found)
        return found

    def _of(self, node: ast.expr) -> str | None:
        if not isinstance(node, ast.Attribute):
            return None
        base = node.value
        if isinstance(base, ast.Name):
            if base.id in self.dummies:
                return self.dummies[base.id].get(node.attr)
            module = self.modules.get(base.id)
            if module is not None:
                return self.state_vars.get((module, node.attr))
            return None
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            module = self.modules.get(base.value.id)
            if module is not None:
                return self.state_objects.get((module, base.attr), {}).get(node.attr)
        return None

    def object_of(self, node: ast.expr) -> str | None:
        """The plan object a bare actual names: a dummy, or ``_mod.obj``."""
        if isinstance(node, ast.Name) and node.id in self.dummies:
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = self.modules.get(node.value.id)
            if module is not None and (module, node.attr) in self.state_objects:
                return node.attr
        return None


def _outputs(plan: FlatPlan) -> list[str]:
    """What the flat function returns, in ``_SIGNATURES`` order."""
    names = [_py(a["name"]) for a in plan.flat_args if a["intent"] in ("OUT", "INOUT")]
    if plan.subprogram["kind"] == "function":
        names = ["_result", *names]
    return names


class _Rewrite(ast.NodeTransformer):
    """The anchor's body on the flat signature -- or, with no plan, just
    scrubbed of what a kernel cannot carry."""

    def __init__(
        self,
        plan: FlatPlan | None,
        spelling: _Spelling,
        plans: dict[str, FlatPlan],
        ports: dict[str, dict[str, Any]],
        bundled: frozenset[str] = frozenset(),
        statics: frozenset[str] = frozenset(),
        dim_sources: dict[str, tuple[str, int]] | None = None,
    ) -> None:
        self.statics = statics  # scalar integer arguments the kernel takes static
        self.dim_sources = dim_sources or {}  # dummy extent name -> (array dummy, axis)
        self.own: dict[str, Any] = {}
        """This module's ``fns`` and ``records``."""
        self.specialized: dict[str, tuple[ast.FunctionDef, FlatPlan, list[str]]] = {}
        """Root finders specialized per callback, emitted beside the flat functions."""
        self.plan = plan
        self.spelling = spelling
        self.plans = plans
        self.ports = ports  # companion module -> {"kernels", "closures", "plans"}
        self.bundled = bundled  # companions the anchor carries a translation of
        self.aliases: dict[str, str] = {}  # alias -> flat name, from dropped assignments
        self.calls: list[str] = []
        self.aborts: list[str] = []
        self.companions: set[str] = set()  # ``_alias`` names whose ported module is called
        self.temps = 0
        self.associates: frozenset[str] = frozenset()  # names an alias drop may claim
        self.inits: dict[str, str] = {}  # local -> "int32" | "float64", from its guard init
        self.loop_vars: set[str] = set()  # loop counters: int64 under x64, cast when stored
        self.state_params: list[str] = []  # a helper's module state, taken as parameters
        self.buffer_outs: dict[str, list[str]] = {}  # anchor _out buffers, elided
        self.masked: list[str] = []  # statements whose dynamic slices became masks
        self.static_loops: list[str] = []  # loops whose trip count became static

    # -- what is static under jit ---------------------------------------------
        self.seeded: list[str] = []

    def _is_dynamic(self, expr: ast.expr) -> bool:
        """A bound that is not a literal, a constant, a static argument or an
        array's static shape."""

        def visit(node: ast.AST) -> bool:
            if isinstance(node, ast.Attribute) and node.attr == "shape":
                return False  # ``x.shape[k]`` is static under jit
            if isinstance(node, ast.Subscript) and not (
                isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
            ):
                return True
            if (
                isinstance(node, ast.Name)
                and not node.id.isupper()
                and node.id not in self.statics
                and node.id not in ("min", "max", "len")  # static over static shapes
            ):
                return True
            return any(visit(child) for child in ast.iter_child_nodes(node))

        return visit(expr)

    def _mask_slices(self, expr: ast.expr) -> tuple[ast.expr, ast.expr | None]:
        """Every ``x[i, lo:hi]`` with a traced ``hi`` becomes ``x[i, lo:]`` --
        the axis's whole static extent -- and the mask ``arange(lo, extent) <
        hi`` says which of it the statement meant. One bound per statement,
        or the rewrite refuses."""
        found: list[tuple[ast.expr, int, ast.expr | None, ast.expr, int]] = []
        rewrite = self

        class Widen(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                self.generic_visit(node)
                elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
                widened: list[ast.expr] = []
                for axis, element in enumerate(elts):
                    if (
                        isinstance(element, ast.Slice)
                        and element.upper is not None
                        and rewrite._is_dynamic(element.upper)
                    ):
                        trailing = sum(isinstance(e, ast.Slice) for e in elts[axis + 1 :])
                        found.append((node.value, axis, element.lower, element.upper, trailing))
                        widened.append(ast.Slice(lower=element.lower, upper=WIDENED, step=None))
                    else:
                        widened.append(element)
                if isinstance(node.slice, ast.Tuple):
                    node.slice = ast.Tuple(elts=widened, ctx=ast.Load())
                else:
                    node.slice = widened[0]
                return node

        widened_expr: Any = Widen().visit(copy.deepcopy(expr))
        if not found:
            return expr, None
        keys = {
            (ast.unparse(_fold(lo)) if lo else "0", ast.unparse(_fold(hi)))
            for _, _, lo, hi, _ in found
        }
        if len(keys) != 1:
            raise NotFlat("dynamic slices with different bounds in one statement")
        _base, _axis, lo, hi, trailing = found[0]
        # One static length for every widened slice -- the SMALLEST of the
        # sliced arrays' axes -- so ``x[i, 0:n] + y[i, 0:n]`` stays
        # conformable when ``x`` and ``y`` are not allocated alike (the
        # accumulator's interface axis is nlev+1 beside a nlev profile).
        # Shapes are Python ints under jit, so ``min`` folds at trace time.
        seen: dict[str, ast.expr] = {}
        for b, ax, _, _, _ in found:
            shape_ref: ast.expr = ast.Subscript(
                value=ast.Attribute(value=copy.deepcopy(b), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(ax),
                ctx=ast.Load(),
            )
            seen.setdefault(ast.unparse(shape_ref), shape_ref)
        refs = list(seen.values())
        extent = (
            refs[0]
            if len(refs) == 1
            else ast.Call(func=ast.Name(id="min", ctx=ast.Load()), args=refs, keywords=[])
        )
        low: ast.expr = copy.deepcopy(lo) if lo else ast.Constant(0)
        length = ast.BinOp(left=extent, op=ast.Sub(), right=copy.deepcopy(low))

        class Bound(ast.NodeTransformer):
            def visit_Slice(self, node: ast.Slice) -> ast.AST:
                if node.upper is WIDENED:
                    start: ast.expr = copy.deepcopy(node.lower) if node.lower else ast.Constant(0)
                    return ast.Slice(
                        lower=node.lower,
                        upper=ast.BinOp(left=start, op=ast.Add(), right=copy.deepcopy(length)),
                        step=None,
                    )
                return node

        mask: ast.expr = ast.Compare(
            left=_jnp(
                "arange",
                [
                    low,
                    ast.BinOp(left=copy.deepcopy(low), op=ast.Add(), right=copy.deepcopy(length)),
                ],
            ),
            ops=[ast.Lt()],
            comparators=[copy.deepcopy(hi)],
        )
        if trailing:
            # ``x[i, 0:n, 0:m]``: the mask is along one axis and the slice
            # has more; ``mask[:, None]`` broadcasts it over the rest.
            mask = ast.Subscript(
                value=mask,
                slice=ast.Tuple(
                    elts=[ast.Slice(), *[ast.Constant(None) for _ in range(trailing)]],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
        return Bound().visit(widened_expr), mask

    # -- names and attributes -------------------------------------------------

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.buffer_outs:
            raise NotFlat(
                f"the buffer {node.id} was elided at its call; this later use has no value"
            )
        flat = self.aliases.get(node.id)
        if flat is not None:
            return ast.copy_location(ast.Name(id=flat, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        flat = self.spelling.of(node)
        if flat is not None:
            return ast.copy_location(ast.Name(id=flat, ctx=node.ctx), node)
        self.generic_visit(node)
        return node

    # -- statements -----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> Any:
        # ``alias = obj.comp`` -> dropped; the alias is the component from
        # here. Only an *associate*-style alias: assigned once, at the top of
        # the body. ``kn = _mod.kn_val`` in one branch of an ``if`` is a
        # value, and dropping it would make every branch read that one.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.associates
            and self.spelling.of(node.value) is not None
        ):
            self.aliases[node.targets[0].id] = self.spelling.of(node.value) or ""
            return None
        if isinstance(node.value, ast.Call) and self._flat_callee(node.value) is not None:
            return self._rewrite_call(node.targets[0], node.value)
        pending = self._companion_writes(node.value) if isinstance(node.value, ast.Call) else []
        self.generic_visit(node)
        if pending:
            # A companion kernel that writes module state returns it after
            # its own outputs; the caller binds it onto the flat names.
            targets = (
                list(node.targets[0].elts)
                if isinstance(node.targets[0], ast.Tuple)
                else list(node.targets)
            )
            targets += [ast.Name(id=flat, ctx=ast.Store()) for flat in pending]
            node.targets = [ast.Tuple(elts=targets, ctx=ast.Store())]
            return node
        target = node.targets[0]
        if (
            len(node.targets) == 1
            and isinstance(target, ast.Name)
            and target.id in self.inits
            and not isinstance(node.value, ast.Constant)
            and (_constant_expression(node.value, self.loop_vars) or _table_read(node.value))
        ):
            # ``l1 = NL - 1``: a Python int under x64 is an int64, and a
            # ``lax.cond`` arm that assigns it beside one that keeps an
            # int32 has a different output type. The variable's guard init
            # (``l1 = 0``, ``obu0 = 0.0``) says which strong type it is.
            node.value = _jnp(self.inits[target.id], [node.value])

        if len(node.targets) == 1 and isinstance(target, ast.Subscript):
            # ``x[i, lo:hi] = v`` with a traced ``hi``: the whole axis, masked.
            pair: Any
            pair, mask = self._mask_slices(ast.Tuple(elts=[target, node.value], ctx=ast.Load()))
            if mask is not None:
                full_target, full_value = pair.elts
                keep = copy.deepcopy(full_target)
                keep.ctx = ast.Load()
                full_target.ctx = ast.Store()
                self.masked.append(ast.unparse(node))
                return ast.copy_location(
                    ast.Assign(
                        targets=[full_target],
                        value=_jnp("where", [mask, full_value, keep]),
                    ),
                    node,
                )
        # ``x[p - 1, :], y = f(...)``: the backend lowers a subscript store on
        # its own statement, not inside a tuple target.
        if (
            len(node.targets) == 1
            and isinstance(target, ast.Tuple)
            and not all(isinstance(e, ast.Name) for e in target.elts)
        ):
            names: list[ast.expr] = []
            follow: list[ast.stmt] = []
            for element in target.elts:
                if isinstance(element, ast.Name):
                    names.append(element)
                else:
                    self.temps += 1
                    temp = f"_t{self.temps}"
                    names.append(ast.Name(id=temp, ctx=ast.Store()))
                    follow.append(
                        ast.Assign(targets=[element], value=ast.Name(id=temp, ctx=ast.Load()))
                    )
            first = ast.Assign(targets=[ast.Tuple(elts=names, ctx=ast.Store())], value=node.value)
            return [first, *follow]
        return node

    def visit_Expr(self, node: ast.Expr) -> Any:
        call = node.value
        if isinstance(call, ast.Call) and self._flat_callee(call) is not None:
            return self._rewrite_call(None, call)
        # ``_f_copy_out(dst, src)``: an in-place copy is an assignment here.
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_f_copy_out"
            and len(call.args) == 2
        ):
            dst, src = call.args
            if (
                isinstance(src, ast.Subscript)
                and isinstance(src.value, ast.Name)
                and src.value.id in self.buffer_outs
            ):
                # The buffer was elided at the call: this copy already
                # happened when the flat outputs bound to the actuals.
                return None
            return self.visit(ast.copy_location(ast.Assign(targets=[dst], value=src), node))
        self.generic_visit(node)
        return node

    def visit_For(self, node: ast.For) -> Any:
        if isinstance(node.target, ast.Name):
            self.loop_vars.add(node.target.id)  # before the body: its stores cast
        self.generic_visit(node)
        it = node.iter
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and isinstance(node.target, ast.Name)
            and len(it.args) == 3
        ):
            return self._static_descending(node)
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and isinstance(node.target, ast.Name)
            and 1 <= len(it.args) <= 2
        ):
            return node
        stop = it.args[-1]
        start: ast.expr = it.args[0] if len(it.args) == 2 else ast.Constant(0)
        self.loop_vars.add(node.target.id)
        # A bound over a dummy extent (``do i = 2, n`` beside ``a(n)``) is
        # static in the kernel's own trace and traced when the kernel is
        # inlined into another; the array's static shape serves both.
        # ... and so is any scalar integer dummy (``maxit``, ``n``): static
        # in the kernel's own trace, a tracer when the kernel is inlined.
        sized = any(
            isinstance(n, ast.Name) and (n.id.lower() in self.dim_sources or n.id in self.statics)
            for bound in (start, stop)
            for n in ast.walk(bound)
        )
        dynamic_start = self._is_dynamic(start) or any(
            isinstance(n, ast.Name) and n.id in self.statics for n in ast.walk(start)
        )
        # Bounds stay as the anchor spells them: ``lax.fori_loop`` lowers a
        # loop whose bounds are Python ints to a ``scan``, which reverse
        # mode can transpose, and a ``jnp.int32`` bound -- even a concrete
        # one -- makes it a ``while_loop``, which it cannot. A weakly typed
        # counter is fine: what the arms of a ``lax.cond`` must agree on is
        # the dtype, and the guard-typed casts keep the other arm an int32.
        if not (self._is_dynamic(stop) or dynamic_start or sized):
            return node
        # ``do ic = 1, ncan(p)``: the trip count is the run's, and a loop
        # whose count is traced has no reverse-mode rule. The body indexes
        # some array by ``ic - 1``; that axis's static extent is the count
        # the loop runs to, and ``if ic < stop`` keeps the iterations the
        # source meant -- a ``lax.cond`` with the identity as its other arm.
        var = node.target.id
        self.loop_vars.add(var)
        extent = _indexed_extent(node.body, var)
        if extent is None and any(isinstance(n, ast.Subscript) for n in ast.walk(stop)):
            # Only a subscripted per-patch count (``ncan[p - 1] + 1``) may
            # take its extent from the callee's indexing: a scalar bound
            # (``nrk_steps + 1 + 1``) can legitimately run PAST the axis the
            # callee indexes (the RK combine pass), and a too-short range is
            # a wrong answer no guard can widen.
            extent = self._callee_extent(node.body, var)
        if extent is None:
            return node
        test: ast.expr = ast.Compare(
            left=ast.Name(id=var, ctx=ast.Load()), ops=[ast.Lt()], comparators=[stop]
        )
        if dynamic_start:
            # ``do ic = nbot(p), ntop(p)``: from the axis's first index, the
            # start a guard as well.
            test = ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Name(id=var, ctx=ast.Load()), ops=[ast.GtE()], comparators=[start]
                    ),
                    test,
                ],
            )
            it.args[0] = ast.Constant(1)
        guard = ast.If(test=test, body=node.body, orelse=[])
        self.static_loops.append(f"{var}: {ast.unparse(start)}..{ast.unparse(stop)}")
        it.args[-1] = extent
        node.body = [guard]
        return node

    def _callee_extent(self, body: list[ast.stmt], var: str) -> ast.expr | None:
        """The loop counter is not an index here -- it rides into a companion
        kernel (``LeafFluxes(p, ic, il, inst)``) and the indexing lives in
        the callee. The flat spelling is the same on both sides, so the
        extent the callee's body reads is valid in the caller."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("_")
                and node.func.attr.endswith("_k_impl")
                and isinstance(node.func.value, ast.Name)
            ):
                continue
            at = next(
                (i for i, a in enumerate(node.args) if isinstance(a, ast.Name) and a.id == var),
                None,
            )
            if at is None:
                continue
            kernel = node.func.attr[1 : -len("_k_impl")]
            module_alias = node.func.value.id[: -len("_jax")]
            module = self.spelling.modules.get(module_alias)
            port = self.ports.get(module) if module is not None else None
            fn = (port or {}).get("fns", {}).get(kernel)
            record = (port or {}).get("records", {}).get(kernel)
            if (fn is None or record is None) and kernel.endswith("_flat"):
                # The port's records and anchor functions are keyed by the
                # ORIGINAL subprogram; the flat call's leading scalars keep
                # the original order, and the original body carries the
                # subscripts the flat rewrite respelled.
                base = kernel[: -len("_flat")]
                fn = (port or {}).get("fns", {}).get(base)
                record = (port or {}).get("records", {}).get(base)
            if fn is None or record is None or at >= len(record["args"]):
                continue
            dummy = _py(record["args"][at]["name"])
            found = _callee_component_extent(fn.body, dummy, node.args)
            if found is not None:
                return found
        return None

    def _static_descending(self, node: ast.For) -> Any:
        """``do ic = ntop(p), nbot(p), -1``: the anchor's ``range(a, b - 1,
        -1)`` with a traced ``a`` or ``b``. The loop runs the indexed axis's
        whole extent downward and ``if b - 1 < ic <= a`` keeps the iterations
        the source meant -- static, so reverse mode has a rule."""
        it = node.iter
        assert isinstance(it, ast.Call) and isinstance(node.target, ast.Name)
        start, stop, step = it.args
        if not (
            isinstance(step, ast.UnaryOp)
            and isinstance(step.op, ast.USub)
            and isinstance(step.operand, ast.Constant)
            and step.operand.value == 1
        ):
            return node
        traced = any(
            isinstance(n, ast.Name) and n.id in self.statics
            for bound in (start, stop)
            for n in ast.walk(bound)
        )
        if not (self._is_dynamic(start) or self._is_dynamic(stop) or traced):
            return node
        var = node.target.id
        extent = _indexed_extent(node.body, var)
        if extent is None:
            return node
        # ``extent`` is the exclusive bound of an ascending loop; the highest
        # value the variable takes is one less.
        top: ast.expr = ast.BinOp(left=extent, op=ast.Sub(), right=ast.Constant(1))
        guard = ast.If(
            test=ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Name(id=var, ctx=ast.Load()), ops=[ast.LtE()], comparators=[start]
                    ),
                    ast.Compare(
                        left=ast.Name(id=var, ctx=ast.Load()), ops=[ast.Gt()], comparators=[stop]
                    ),
                ],
            ),
            body=node.body,
            orelse=[],
        )
        self.static_loops.append(f"{var}: {ast.unparse(start)} down to {ast.unparse(stop)}")
        it.args = [top, ast.Constant(0), step]
        node.body = [guard]
        return node

    def visit_If(self, node: ast.If) -> Any:
        # ``if bad: raise`` -- the anchor's ``endrun``. A kernel cannot raise
        # under tracing, and the recorded run it is gated on never did; the
        # check is dropped and named on the candidate, so the evidence says
        # the kernel no longer aborts where the model would.
        if node.body and all(isinstance(s, (ast.Raise, ast.Expr, ast.Pass)) for s in node.body):
            if any(isinstance(s, ast.Raise) for s in node.body):
                self.aborts.append(ast.unparse(node.test))
                if not node.orelse:
                    return None
                replaced: list[ast.stmt] = []
                for stmt in node.orelse:
                    seen = self.visit(stmt)
                    if seen is None:
                        continue
                    replaced.extend(seen if isinstance(seen, list) else [seen])
                return replaced or None
        self.generic_visit(node)
        if all(isinstance(s, (ast.Pass, ast.Expr)) for s in node.body):
            # ``if cond: write(iulog, ...)`` -- a log line the anchor already
            # left as ``pass``; nothing to carry.
            return [*node.orelse] if node.orelse else None
        return node

    def visit_Raise(self, node: ast.Raise) -> Any:
        if (
            isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "_FGoto"
        ):
            # A backward-goto restart, not an abort: control flow the while
            # lowering turns into a flag. Dropping it would silently unroll
            # the loop to one pass.
            return node
        self.aborts.append(ast.unparse(node))
        return None

    def visit_Return(self, node: ast.Return) -> Any:
        if self.plan is None:
            self.generic_visit(node)
            return node
        outs = _outputs(self.plan)
        if self.plan.subprogram["kind"] == "function":
            value = self.visit(node.value) if node.value is not None else ast.Constant(None)
            result = ast.Assign(targets=[ast.Name(id="_result", ctx=ast.Store())], value=value)
            return [result, ast.Return(value=_tuple(outs))]
        return ast.Return(value=_tuple(outs))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self._flat_callee(node) is not None:
            raise NotFlat(
                f"{ast.unparse(node.func)} takes the object and is called inside an expression"
            )
        self.generic_visit(node)
        # ``int(x)`` and ``np.float64(x)`` on a traced value: the cast the
        # anchor spells with a Python or NumPy constructor is ``jnp``'s here.
        # ``jnp.int32`` truncates toward zero the way ``int`` does.
        if isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
            if _static_expression(node.args[0], self.statics):
                # ``int(dtime_clm / dtime_ml)`` over static scalars: a Python
                # int at trace time, so the loop it bounds lowers to a scan
                # reverse mode can transpose. A jnp cast -- even concrete --
                # would make the loop a while_loop.
                return node
            return ast.copy_location(_jnp("int32", node.args), node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "jnp")
            and node.func.attr in ("empty", "zeros", "ones")
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            # A local sized by a dummy extent (``gam(n)`` beside ``a(n)``):
            # the extent is the array dummy's static shape, which stays
            # static when the kernel is inlined into another with a traced
            # ``n``.
            shape = node.args[0]
            for at, element in enumerate(shape.elts):
                if isinstance(element, ast.Name) and element.id.lower() in self.dim_sources:
                    array, axis = self.dim_sources[element.id.lower()]
                    shape.elts[at] = ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=array, ctx=ast.Load()), attr="shape", ctx=ast.Load()
                        ),
                        slice=ast.Constant(axis),
                        ctx=ast.Load(),
                    )
            return node
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "jnp")
            and node.func.attr == "sum"
            and len(node.args) == 1
        ):
            # ``sum(x[i, lo:hi] * y[i, lo:hi])`` over a traced ``hi``: the
            # whole axis, the rest zeroed.
            full, mask = self._mask_slices(node.args[0])
            if mask is not None:
                self.masked.append(ast.unparse(node))
                node.args = [_jnp("where", [mask, full, ast.Constant(0.0)])]
                return node
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in CASTS
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Constant)
            and not _static_expression(node.args[0], frozenset(self.statics))
        ):
            return ast.copy_location(_jnp(node.func.attr, node.args), node)
        return self._companion_call(node)

    def _companion_writes(self, node: ast.Call) -> list[str]:
        """The flat names a companion kernel call writes, for its caller's
        statement to bind; refused if the plan does not carry one."""
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            return []
        module = self.spelling.modules.get(func.value.id)
        port = self.ports.get(module) if module is not None else None
        if port is None:
            return []
        flats = []
        for state in (port.get("write_closures") or {}).get(func.attr, []):
            flat = f"{module}__{state}"
            if flat not in self.spelling.states:
                raise NotFlat(f"{module}.{func.attr} writes {state}, which the plan does not carry")
            flats.append(flat)
        return flats

    # -- calls into companions ------------------------------------------------

    def _companion_call(self, node: ast.Call) -> ast.AST:
        """``_mlw.latvap(x)`` -> ``_mlw_jax._latvap_k_impl(x, <closure>)``."""
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            return node
        module = self.spelling.modules.get(func.value.id)
        if module is None:
            return node
        port = self.ports.get(module)
        if port is None:
            if module in self.bundled:
                raise NotFlat(f"calls {module}.{func.attr}, and {module} was not ported")
            # Not a companion: a stand-in. Its functions are the framework's
            # answers, not physics, and are left as they are.
            return node
        if func.attr not in port["kernels"]:
            raise NotFlat(f"calls {module}.{func.attr}, which its port did not lower")
        closure: list[ast.expr] = []
        for state in port["closures"].get(func.attr, []):
            flat = f"{module}__{state}"
            if flat not in self.spelling.states:
                raise NotFlat(f"{module}.{func.attr} reads {state}, which the plan does not carry")
            closure.append(ast.Name(id=flat, ctx=ast.Load()))
        self.companions.add(func.value.id)
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=f"{func.value.id}_jax", ctx=ast.Load()),
                    attr=f"_{func.attr}_k_impl",
                    ctx=ast.Load(),
                ),
                args=[*node.args, *closure],
                keywords=node.keywords,
            ),
            node,
        )

    # -- calls into this module's flat functions ------------------------------

    def _flat_callee(self, call: ast.Call) -> FlatPlan | None:
        callee: FlatPlan | None = None
        source: dict[str, Any] = self.own
        if isinstance(call.func, ast.Name):
            callee = self.plans.get(call.func.id.lower())
            name = call.func.id.lower()
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            module = self.spelling.modules.get(call.func.value.id)
            port = self.ports.get(module or "")
            if port is None:
                return None
            callee = port["plans"].get(call.func.attr.lower())
            source = port
            name = call.func.attr.lower()
        else:
            return None
        if callee is None:
            callee = self._specialize(name, call, source)
        if callee is None:
            return None
        if any(self.spelling.object_of(a) is not None for a in call.args) or any(
            self.spelling.object_of(k.value) is not None for k in call.keywords
        ):
            return callee
        return None

    def _specialize(self, name: str, call: ast.Call, source: dict[str, Any]) -> FlatPlan | None:
        """``hybrid(msg, p, ic, il, inst, cifunc, xa, xb, tol)``: a subprogram
        taking the object *and a procedure* has no plan of its own -- what
        it touches is the callback's business. Specialized per callback,
        it is the callback's plan on the root finder's body: the procedure
        dummy is the callback's flat function, the object is the
        callback's components, and the specialization is emitted beside the
        flat functions as ``<callee>__<callback>_flat``."""
        record = (source.get("records") or {}).get(name)
        fn = (source.get("fns") or {}).get(name)
        if record is None or fn is None:
            return None
        procedure = [a["name"].lower() for a in record["args"] if str(a["dtype"]) == "PROCEDURE"]
        if len(procedure) != 1:
            return None
        dummies = [a["name"].lower() for a in record["args"]]
        actual_by_dummy: dict[str, ast.expr] = {}
        for at, given in enumerate(call.args):
            if at < len(dummies):
                actual_by_dummy[dummies[at]] = given
        for keyword_ in call.keywords:
            if keyword_.arg:
                actual_by_dummy[keyword_.arg.lower()] = keyword_.value
        passed = actual_by_dummy.get(procedure[0])
        if not isinstance(passed, ast.Name):
            return None
        callback = self.plans.get(passed.id.lower())
        if callback is None:
            return None
        spec_name = f"{name}__{callback.subprogram['name']}"
        if spec_name in self.specialized:
            return self.specialized[spec_name][1]
        if spec_name.lower() in self.plans:
            return self.plans[spec_name.lower()]
        objects = copy.deepcopy(callback.objects)
        derived = [a["name"].lower() for a in record["args"] if "UNKNOWN(TYPE" in str(a["dtype"])]
        dummy_objects = [o for o in objects if o.kind == "dummy"]
        # The object is called by the root finder's own dummy name. One
        # derived dummy pairs with the callback's one object; more would
        # pair by position, which is a guess.
        if len(derived) != 1 or len(dummy_objects) != 1:
            raise NotFlat(
                f"{name}: {len(derived)} derived dummies against {len(dummy_objects)} "
                f"objects of {callback.subprogram['name']}; the pairing is not by position"
            )
        for obj, own_name in zip(dummy_objects, derived, strict=True):
            for component in obj.components:
                component.owner = own_name
            obj.name = own_name
        plan = FlatPlan(
            subprogram={
                **record,
                "name": spec_name,
                "public": True,
                "args": list(record["args"]),
            },
            objects=objects,
            states=copy.deepcopy(callback.states),
            patch_count=callback.patch_count,
            counter_prefix=callback.counter_prefix,
        )
        plans = {**self.plans, procedure[0]: callback, spec_name.lower(): plan}
        inner = _Rewrite(
            plan,
            _Spelling(plan, self.spelling.modules),
            plans,
            self.ports,
            self.bundled,
            _static_names(plan.flat_args),
            _dim_sources(plan.flat_args),
        )
        inner.own = source
        inner.specialized = self.specialized
        body = _rewritten_body(fn, inner)
        if not body or not isinstance(body[-1], ast.Return):
            body.append(ast.Return(value=_tuple(_outputs(plan))))
        taken = [_py(a["name"]) for a in plan.flat_args if a["intent"] != "OUT"]
        flat = ast.FunctionDef(
            name=plan.name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=n) for n in taken],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
        )
        ast.fix_missing_locations(flat)
        # The iteration is one function; the specialization the caller sees
        # is the implicit-function wrapper around it.
        iterate = copy.deepcopy(plan)
        iterate.subprogram = {**plan.subprogram, "name": f"{spec_name}_iterate"}
        flat.name = iterate.name
        self.specialized[iterate.subprogram["name"]] = (flat, iterate, inner.calls)
        wrapper = self._implicit_wrapper(plan, iterate, fn, procedure[0], callback, inner)
        self.specialized[spec_name] = (wrapper, plan, [iterate.name, callback.name])
        if source.get("module"):
            self.companions.add(f"module:{source['module']}")
        self.aborts.extend(f"{spec_name}: {a}" for a in inner.aborts)
        self.companions |= inner.companions
        self.plans[spec_name.lower()] = plan
        return plan

    def _implicit_wrapper(
        self,
        plan: FlatPlan,
        iterate: FlatPlan,
        fn: ast.FunctionDef,
        procedure: str,
        callback: FlatPlan,
        inner: _Rewrite,
    ) -> ast.FunctionDef:
        """The implicit-function adjoint around the iteration (Blondel et al.
        2022, as the paper applies it): the converged root is detached, one
        Newton step on the callback's residual ``x - F(x)/sg(dF/dx)`` leaves
        the value where it was and makes the derivative the implicit one,
        ``dF/dtheta / dF/dx`` -- spelled ``F - sg(F)`` so the value is exactly
        the iteration's, since the Fortran stopped on a tolerance and a
        Newton step would move it; the components come from one last call of the
        callback at that root, so they are differentiable through it too.
        Reverse mode then never has to go through the loop."""
        taken = [_py(a["name"]) for a in plan.flat_args if a["intent"] != "OUT"]
        outs = _outputs(plan)
        # Each component's value is the iteration's own (the Fortran's last
        # callback evaluation is not necessarily at the root), its tangent
        # the one at the root.
        outs_at_root = [
            f"lax.stop_gradient(_it[{i + 1}]) + _at_root[{i + 1}]"
            f" - lax.stop_gradient(_at_root[{i + 1}])"
            for i in range(len(outs) - 1)
        ]
        # The callback call, as the root finder spells it: the first call of
        # the procedure dummy in its body, with the unknown's actual replaced.
        template = next(
            (
                node
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.lower() == procedure
            ),
            None,
        )
        if template is None:
            raise NotFlat(f"{plan.subprogram['name']}: no call of {procedure} to invert")
        cb_args = callback.subprogram["args"]
        unknown = next(
            (
                at
                for at, a in enumerate(cb_args)
                if a["intent"] == "IN" and str(a["dtype"]).startswith("float") and not a.get("dims")
            ),
            None,
        )
        if unknown is None or unknown >= len(template.args):
            raise NotFlat(f"{plan.subprogram['name']}: the callback's unknown is not a scalar real")

        def callback_call(x: str) -> ast.expr:
            call = copy.deepcopy(template)
            call.args[unknown] = ast.Name(id=x, ctx=ast.Load())
            probe = _Rewrite(
                plan, _Spelling(plan, self.spelling.modules), inner.plans, self.ports, self.bundled
            )
            probe.own = inner.own
            probe.specialized = self.specialized
            # A target per output the callback declares, so the rewrite has
            # somewhere to put them; only the call expression is kept.
            slots = ast.Tuple(
                elts=[
                    ast.Name(id=f"_o{i}", ctx=ast.Store())
                    for i, a in enumerate(cb_args)
                    if a["intent"] in ("OUT", "INOUT") and not a.get("optional")
                ],
                ctx=ast.Store(),
            )
            rewritten = probe._rewrite_call(slots, call)
            if isinstance(rewritten, list):
                rewritten = rewritten[0]
            value = rewritten if isinstance(rewritten, ast.Expr) else rewritten.value
            return value.value if isinstance(value, ast.Expr) else value

        src = "\n".join(
            [
                f"def {plan.name}({', '.join(taken)}):",
                # No tangent enters the iteration: its derivative is the
                # residual step's, and a loop nothing differentiates through
                # is one reverse mode never has to transpose.
                f"    _it = {iterate.name}({', '.join(f'lax.stop_gradient({n})' for n in taken)})",
                "    _root = lax.stop_gradient(_it[0])",
                "",
                "    def _residual(_x):",
                "        return CALLBACK_X[0]",
                "",
                "    _f, _dfdx = jax.jvp(_residual, (_root,), (jnp.ones_like(_root),))",
                # The value stays the iteration's own (F - sg(F) is zero), the
                # tangent becomes dF/dtheta / dF/dx; a flat residual (no light
                # on the leaf, say) has no root to speak of and no derivative
                # through it.
                "    _slope = lax.stop_gradient(jnp.where(jnp.abs(_dfdx) > 0.0, _dfdx, 1.0))",
                "    _root = _root - jnp.where("
                "jnp.abs(_dfdx) > 0.0, (_f - lax.stop_gradient(_f)) / _slope, 0.0)",
                "    _at_root = CALLBACK_ROOT",
                "    return (" + ", ".join(["_root", *outs_at_root]) + ",)",
            ]
        )
        module = ast.parse(src)
        wrapper = module.body[0]
        assert isinstance(wrapper, ast.FunctionDef)

        class Fill(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == "CALLBACK_X":
                    return callback_call("_x")
                if node.id == "CALLBACK_ROOT":
                    return callback_call("_root")
                return node

        filled: Any = Fill().visit(wrapper)
        assert isinstance(filled, ast.FunctionDef)
        ast.fix_missing_locations(filled)
        return filled

    def _rewrite_call(self, target: ast.expr | None, call: ast.Call) -> Any:
        callee = self._flat_callee(call)
        assert callee is not None and self.plan is not None
        # The anchor calls its translation by the emitted convention: the
        # non-OUT, non-optional dummies in declaration order, then the OUT
        # arrays it hands in as buffers. An OUT scalar before an object
        # (``hybrid(..., gs_mol, iter, atm2lnd_vars, photosyns_vars)``) is
        # not in the actual list, and counting it shifted the objects.
        arguments = callee.subprogram["args"]
        dummies = [
            a["name"] for a in arguments if a["intent"] != "OUT" and not a.get("optional")
        ] + [a["name"] for a in arguments if a["intent"] == "OUT" and a.get("dims")]
        actual_by_dummy: dict[str, ast.expr] = {}
        for at, given in enumerate(call.args):
            if at < len(dummies):
                actual_by_dummy[dummies[at].lower()] = given
        for keyword_ in call.keywords:
            if keyword_.arg:
                actual_by_dummy[keyword_.arg.lower().rstrip("_")] = keyword_.value
        # Which callee object is which caller object.
        objects: dict[str, str] = {}
        for obj in callee.objects:
            if obj.kind == "dummy":
                actual = actual_by_dummy.get(obj.name)
                passed: str | None = self.spelling.object_of(actual) if actual is not None else None
                if passed is None:
                    raise NotFlat(f"{callee.subprogram['name']}: object {obj.name} not passed")
                objects[obj.name] = passed
            else:
                objects[obj.name] = obj.name  # module state: the same name everywhere
        caller_components = {
            obj.name: {c.name: c.flat for c in obj.components} for obj in self.plan.objects
        }
        args: list[ast.expr] = []
        for entry in callee.flat_args:
            name = entry["name"]
            if entry["intent"] == "OUT":
                continue  # returned, not taken -- the emitted convention
            if name == callee.patch_count:
                args.append(ast.Name(id=self.plan.patch_count, ctx=ast.Load()))
            elif "__" in name and (owner := name.split("__", 1)[0]) in objects:
                comp = name.split("__", 1)[1]
                flat_name: str | None = caller_components.get(objects[owner], {}).get(comp)
                if flat_name is None:
                    raise NotFlat(
                        f"{callee.subprogram['name']}: {owner}%{comp} not in the caller's plan"
                    )
                args.append(ast.Name(id=flat_name, ctx=ast.Load()))
            elif "__" in name:
                if name not in self.spelling.states:
                    raise NotFlat(
                        f"{callee.subprogram['name']}: state {name} not in the caller's plan"
                    )
                args.append(ast.Name(id=name, ctx=ast.Load()))
            else:
                actual = actual_by_dummy.get(name.lower())
                if actual is None:
                    raise NotFlat(f"{callee.subprogram['name']}: no actual for {name}")
                rewritten: Any = self.visit(copy.deepcopy(actual))
                if str(entry["dtype"]).startswith("float") and _integer_constant_expression(
                    rewritten
                ):
                    # ``hybrid(..., 0, 1, tol)``: an integer literal for a
                    # real dummy is an int32 the callee's arithmetic would
                    # carry into a ``lax.cond`` arm as the wrong dtype.
                    rewritten = _jnp(str(entry["dtype"]), [rewritten])
                args.append(rewritten)
        func: ast.expr = ast.Name(id=callee.name, ctx=ast.Load())
        if callee.name[: -len("_flat")] in self.specialized:
            self.calls.append(callee.name)
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            # A companion's flat function: its port's kernel, or nothing.
            module = self.spelling.modules[call.func.value.id]
            if callee.name not in self.ports[module]["kernels"]:
                raise NotFlat(f"calls {module}.{callee.name}, which its port did not lower")
            self.companions.add(call.func.value.id)
            func = ast.Attribute(
                value=ast.Name(id=f"{call.func.value.id}_jax", ctx=ast.Load()),
                attr=f"_{callee.name}_k_impl",
                ctx=ast.Load(),
            )
        else:
            self.calls.append(callee.name)
        new_call = ast.Call(func=func, args=args, keywords=[])
        # Outputs back onto the caller's names. The anchor's target tuple
        # lists the callee's OUT/INOUT dummies in declaration order, objects
        # included (the emitted convention); an original output lands on the
        # element at its position, a component on the caller's spelling. A
        # target that is not a plain name (``x[p - 1]``) goes through a
        # temporary, because the backend lowers a subscript store on its own
        # statement and not inside a tuple target.
        anchor_targets: list[ast.expr] = []
        if isinstance(target, ast.Tuple):
            anchor_targets = list(target.elts)
        elif target is not None:
            anchor_targets = [target]
        original_outs = [
            _py(a["name"])
            for a in callee.subprogram["args"]
            if a["intent"] in ("OUT", "INOUT") and not a.get("optional")
        ]
        if callee.subprogram["kind"] == "function":
            original_outs = ["_result", *original_outs]
        slot = dict(zip(original_outs, anchor_targets, strict=False))
        # ``_out = callee(...); _f_copy_out(a, _out[0]); ...`` -- the anchor's
        # buffer convention for several array outputs: one name receives the
        # tuple and the copies unpack it. The flat outputs bind to the actuals
        # directly, and the buffer name is rebuilt as their tuple so the
        # anchor's unpacking assignments stay true (each becomes ``a = a``).
        buffered: ast.expr | None = None
        if (
            callee.subprogram["kind"] != "function"
            and len(original_outs) > 1
            and len(anchor_targets) == 1
            and isinstance(anchor_targets[0], ast.Name)
        ):
            # ``_out = callee(...); _f_copy_out(a, _out[0]); ...`` -- the
            # anchor's buffer for several array outputs. The flat outputs
            # bind to the actuals directly and the buffer never exists: a
            # name rebound with a different tuple type at another call site
            # would poison a lax carry, and the copies are self-assignments.
            buffered = anchor_targets[0]
            slot = {
                name: actual_by_dummy[name.lower()]
                for name in original_outs
                if name.lower() in actual_by_dummy
            }
            self.buffer_outs[buffered.id] = [
                ast.unparse(slot[name]) for name in original_outs if name in slot
            ]
        targets: list[ast.expr] = []
        follow: list[ast.stmt] = []
        for name in _outputs(callee):
            if "__" in name and (owner := name.split("__", 1)[0]) in objects:
                comp = name.split("__", 1)[1]
                targets.append(
                    ast.Name(id=caller_components[objects[owner]][comp], ctx=ast.Store())
                )
            elif "__" in name:
                targets.append(ast.Name(id=name, ctx=ast.Store()))
            else:
                where = slot.get(name)
                if where is None:
                    raise NotFlat(f"{callee.subprogram['name']}: output {name} has no target")
                bound: Any = self.visit(copy.deepcopy(where))
                if isinstance(bound, ast.Name):
                    targets.append(ast.Name(id=bound.id, ctx=ast.Store()))
                else:
                    self.temps += 1
                    temp = f"_t{self.temps}"
                    targets.append(ast.Name(id=temp, ctx=ast.Store()))
                    follow.append(
                        ast.Assign(targets=[bound], value=ast.Name(id=temp, ctx=ast.Load()))
                    )
        if not targets:
            return ast.Expr(value=new_call)
        # One output comes back bare (the flat function returns a name, not
        # a 1-tuple); a 1-tuple target would unpack a (1,) array into its
        # element.
        target_node: ast.expr = (
            targets[0] if len(targets) == 1 else ast.Tuple(elts=targets, ctx=ast.Store())
        )
        assign = ast.Assign(targets=[target_node], value=new_call)
        return [assign, *follow] if follow else assign


def _callee_component_extent(
    body: list[ast.stmt], var: str, call_args: list[ast.expr]
) -> ast.expr | None:
    """The callee's body subscripts ``inst.comp[..., ic - 1, ...]`` (the
    anchor's record-attribute spelling); the caller passed the same
    component flat as ``<obj>__<comp>``, whose static axis is the extent.
    Also takes plain flat-Name subscripts, which spell the same on both
    sides."""

    def caller_array(comp: str) -> ast.expr | None:
        for given in call_args:
            if isinstance(given, ast.Name) and given.id.endswith(f"__{comp}"):
                return ast.Name(id=given.id, ctx=ast.Load())
        return None

    # The anchor keeps the associate bindings (``tleaf =
    # mlcanopy_inst.tleaf_leaf``) and subscripts the bare alias after; the
    # alias table routes those to the component the caller passed flat.
    aliases: dict[str, str] = {}
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Attribute)
            and isinstance(stmt.value.value, ast.Name)
        ):
            aliases[stmt.targets[0].id] = stmt.value.attr

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Subscript):
            continue
        base: ast.expr | None = node.value
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and not base.value.id.startswith("_")
        ):
            base = caller_array(base.attr)
        elif isinstance(base, ast.Name) and base.id in aliases:
            base = caller_array(aliases[base.id])
        elif not (isinstance(base, ast.Name) and "__" in base.id):
            base = None
        if base is None:
            continue
        elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        for axis, element in enumerate(elts):
            shape = ast.Subscript(
                value=ast.Attribute(value=copy.deepcopy(base), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(axis),
                ctx=ast.Load(),
            )
            if (
                isinstance(element, ast.BinOp)
                and isinstance(element.op, ast.Sub)
                and isinstance(element.left, ast.Name)
                and element.left.id == var
                and isinstance(element.right, ast.Constant)
                and element.right.value == 1
            ):
                return ast.BinOp(left=shape, op=ast.Add(), right=ast.Constant(1))
            if isinstance(element, ast.Name) and element.id == var:
                return shape
    return None


def _indexed_extent(body: list[ast.stmt], var: str) -> ast.expr | None:
    """``base.shape[k] + 1`` for the first ``base[..., var - 1, ...]`` in the
    body (the loop is 1-based and the index 0-based), or ``base.shape[k]``
    for a bare ``var``; None when nothing is indexed by the variable."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Subscript):
            continue
        elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        for axis, element in enumerate(elts):
            shape = ast.Subscript(
                value=ast.Attribute(value=copy.deepcopy(node.value), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(axis),
                ctx=ast.Load(),
            )
            if (
                isinstance(element, ast.BinOp)
                and isinstance(element.op, ast.Sub)
                and isinstance(element.left, ast.Name)
                and element.left.id == var
                and isinstance(element.right, ast.Constant)
                and isinstance(element.right.value, int)
            ):
                # ``x[var - c]``: the variable runs to ``shape + c - 1``, so
                # the exclusive bound is ``shape + c`` (``(ic) - (0)`` is a
                # zero-based array indexed by a one-based loop).
                offset = element.right.value
                if offset == 0:
                    return shape
                return ast.BinOp(left=shape, op=ast.Add(), right=ast.Constant(offset))
            if isinstance(element, ast.Name) and element.id == var:
                return shape
    return None


def _jnp(attr: str, args: list[ast.expr]) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=attr, ctx=ast.Load()),
        args=args,
        keywords=[],
    )


def _tuple(names: list[str]) -> ast.expr:
    if len(names) == 1:
        return ast.Name(id=names[0], ctx=ast.Load())
    return ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in names], ctx=ast.Load())


def _associates(fn: ast.FunctionDef) -> frozenset[str]:
    """Names assigned exactly once in the body, by a top-level statement."""
    counts: dict[str, int] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    counts[target.id] = counts.get(target.id, 0) + 1
                elif isinstance(target, ast.Tuple):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            counts[element.id] = counts.get(element.id, 0) + 1
        elif isinstance(node, (ast.For, ast.AugAssign)) and isinstance(node.target, ast.Name):
            counts[node.target.id] = counts.get(node.target.id, 0) + 1
    top = {
        s.targets[0].id
        for s in fn.body
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
    }
    return frozenset(name for name in top if counts.get(name) == 1)


def _concrete_scalars(fn: ast.FunctionDef, rewrite: _Rewrite) -> frozenset[str]:
    """Locals that hold a trace-time value: assigned only at the body's top
    level, from arithmetic over statics/constants/other concrete locals,
    casts of those, or a companion scalar kernel called on them
    (``dtime_clm = get_step_size()`` reading the static ``dtstep``). A name
    also stored under a branch or loop is a carried value, not a constant."""
    nested: set[str] = set()
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            continue
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                nested.add(inner.id)
    concrete: set[str] = set()

    def scalar_port_call(call: ast.Call) -> bool:
        func = call.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            return False
        module = rewrite.spelling.modules.get(func.value.id)
        port = rewrite.ports.get(module) if module is not None else None
        if port is None:
            return False
        record = port.get("records", {}).get(func.attr)
        if record is None or func.attr not in port.get("kernels", ()):
            return False
        if any(a.get("dims") for a in record["args"]) or record.get("result_dtype") is None:
            return False
        if (port.get("write_closures") or {}).get(func.attr):
            return False
        closure_static = all(
            f"{module}__{state}" in rewrite.statics
            for state in (port.get("closures") or {}).get(func.attr, [])
        )
        return closure_static and all(ok(a) for a in call.args)

    def ok(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            return node.id.isupper() or node.id in rewrite.statics or node.id in concrete
        if isinstance(node, ast.Attribute):
            flat = rewrite.spelling.of(node)
            return flat is not None and flat in rewrite.statics
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
        ):
            return ok(node.left) and ok(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return ok(node.operand)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
                return ok(node.args[0])
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr in ("float64", "float32", "int32", "int64")
                and len(node.args) == 1
            ):
                return ok(node.args[0])
            return scalar_port_call(node)
        return False

    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign):
            continue
        target = stmt.targets[0]
        names = [
            t.id
            for t in (target.elts if isinstance(target, ast.Tuple) else stmt.targets)
            if isinstance(t, ast.Name)
        ]
        if (
            len(stmt.targets) == 1
            and isinstance(target, ast.Name)
            and target.id not in nested
            and ok(stmt.value)
        ):
            concrete.add(target.id)
        else:
            # A later store that is not a concrete scalar (a tuple from a
            # kernel call, a traced expression) un-makes the name: the
            # UB-guard ``day = 0`` alone does not a constant make.
            for name in names:
                concrete.discard(name)
    return frozenset(concrete)


def _pure_reference(node: ast.expr) -> bool:
    """A name or an attribute chain: an alias of something that exists, not
    a computation."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _seed_pointer_locals(body: list[ast.stmt]) -> list[str]:
    """A pointer local the anchor starts as ``None`` and points in a branch
    (``par_z => solarabs_vars%parsun_z_patch`` under ``phase == 'sun'``,
    the shade array under ``'sha'``) starts the kernel as its first target
    instead: a ``lax.cond`` carries it, and a carry has to have the same
    shape on both arms, which ``None`` is not. The value is the branch's
    own -- the run takes one arm, and the seed is overwritten there; an
    arm the run never takes leaves a Fortran pointer undefined, and the
    kernel a view it never reads. Returns the names seeded."""
    seeded: list[str] = []
    for at, statement in enumerate(body):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is None
        ):
            continue
        name = statement.targets[0].id
        first = next(
            (
                n.value
                for later in body[at + 1 :]
                for n in ast.walk(later)
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name
            ),
            None,
        )
        if first is not None and _pure_reference(first):
            statement.value = copy.deepcopy(first)
            seeded.append(name)
    return seeded


def _rewritten_body(fn: ast.FunctionDef, rewrite: _Rewrite) -> list[ast.stmt]:
    rewrite.seeded = _seed_pointer_locals(fn.body)
    rewrite.associates = _associates(fn)
    rewrite.inits = _guard_inits(fn)
    rewrite.statics = frozenset(rewrite.statics) | _concrete_scalars(fn, rewrite)
    lowered: list[ast.stmt] = []
    # Single exit first, on the anchor's own returns: the flat return that
    # replaces them is one statement at the end.
    for statement in _single_exit([copy.deepcopy(s) for s in fn.body]):
        result = rewrite.visit(statement)
        if result is None:
            continue
        lowered.extend(result if isinstance(result, list) else [result])
    int_locals = frozenset(_integer_inits(fn)) | frozenset(
        n for n, kind in (rewrite.inits or {}).items() if kind == "int32"
    )
    lowered = _WhileLoops(_integer_inits(fn), int_locals).visit_block(lowered)
    # The goto-region flags are synthetic locals with no UB-guard init; an
    # enclosing loop carries them, and the initial carry tuple needs a value
    # before the loop. Every flag starts False at the top of the function.
    flags = sorted(
        {
            n.id
            for stmt in lowered
            for n in ast.walk(stmt)
            if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and re.fullmatch(r"_(?:skip|restart)_\d+", n.id)
        }
    )
    lowered = [
        *(
            ast.fix_missing_locations(
                ast.Assign(targets=[ast.Name(id=f, ctx=ast.Store())], value=ast.Constant(False))
            )
            for f in flags
        ),
        *lowered,
    ]
    for statement in lowered:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Slice)
                and node.upper is not None
                and rewrite._is_dynamic(node.upper)
            ):
                raise NotFlat(f"a dynamic slice outside a store or a sum: {ast.unparse(node)}")
    return lowered


def _dim_sources(args: list[dict[str, Any]]) -> dict[str, tuple[str, int]]:
    """``{extent name: (array dummy, axis)}`` for every dummy array whose
    declared extent is a bare name -- ``a(n)`` says ``n == a.shape[0]``."""
    sources: dict[str, tuple[str, int]] = {}
    for a in args:
        for axis, dim in enumerate(a.get("dims") or []):
            ub = str(dim.get("ub") or "").strip().lower()
            if ub.isidentifier() and str(dim.get("lb") or "1").strip() == "1":
                sources.setdefault(ub, (_py(a["name"]), axis))
    return sources


def _has_return(stmts: list[ast.stmt]) -> bool:
    return any(isinstance(n, ast.Return) for s in stmts for n in ast.walk(s))


def _single_exit(body: list[ast.stmt]) -> list[ast.stmt]:
    """Early returns -- ``if f0 == 0: root = x0; return root`` -- become a
    flag and a value, every later statement runs under ``if not _ret``, and
    the one return at the end merges: the backend refuses a return inside a
    branch, and a kernel has one exit."""
    early = [s for s in body[:-1] if _has_return([s])]
    if not early or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return body
    final = body[-1]
    # A subroutine's translation returns the same tuple of its output
    # names at every return (the emitted convention): nothing to select,
    # the values are the variables' at the moment of the return, which the
    # guarded statements after it leave alone. Only the flag is needed.
    # A function whose returns are expressions merges one value through
    # ``jnp.where``; several distinct values would need one ``where`` each
    # and are refused.
    same_names = isinstance(final.value, ast.Tuple) and all(
        isinstance(n.value, ast.Tuple) and ast.unparse(n.value) == ast.unparse(final.value)
        for statement in body
        for n in ast.walk(statement)
        if isinstance(n, ast.Return)
    )
    if not same_names:
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                    raise NotFlat(
                        f"an early return in a function with several outputs: {ast.unparse(node)}"
                    )

    class Returns(ast.NodeTransformer):
        def visit_Return(self, node: ast.Return) -> Any:
            flag = ast.Assign(
                targets=[ast.Name(id="_ret", ctx=ast.Store())], value=ast.Constant(True)
            )
            if same_names:
                return [flag]
            value = node.value if node.value is not None else ast.Constant(None)
            return [ast.Assign(targets=[ast.Name(id="_r", ctx=ast.Store())], value=value), flag]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node  # a nested function's returns are its own

    out: list[ast.stmt] = [
        ast.Assign(targets=[ast.Name(id="_ret", ctx=ast.Store())], value=ast.Constant(False)),
    ]
    if not same_names:
        out.append(
            ast.Assign(targets=[ast.Name(id="_r", ctx=ast.Store())], value=ast.Constant(0.0))
        )
    exited = False
    for statement in body[:-1]:
        had_return = _has_return([statement])  # before the visit rewrites it away
        rewritten: Any = Returns().visit(statement)
        rewritten = rewritten if isinstance(rewritten, list) else [rewritten]
        not_returned = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_ret", ctx=ast.Load()))
        if exited and len(rewritten) == 1 and isinstance(rewritten[0], ast.While):
            # A loop is not wrapped -- its done flag is the loop's own carry
            # and must not become an enclosing branch's -- but its condition
            # takes the flag: a loop after a return does not run.
            loop = rewritten[0]
            forever = isinstance(loop.test, ast.Constant) and loop.test.value is True
            loop.test = (
                not_returned
                if forever
                else ast.BoolOp(op=ast.And(), values=[loop.test, copy.deepcopy(not_returned)])
            )
            out.append(loop)
        elif exited:
            out.append(ast.If(test=not_returned, body=rewritten, orelse=[]))
        else:
            out.extend(rewritten)
        if had_return:
            exited = True
    assert isinstance(final, ast.Return) and final.value is not None
    if same_names:
        out.append(final)
        return out
    merged = _jnp(
        "where",
        [ast.Name(id="_ret", ctx=ast.Load()), ast.Name(id="_r", ctx=ast.Load()), final.value],
    )
    out.append(ast.Return(value=merged))
    return out


def _require_trailing_raises(body: list[ast.stmt], label: Any) -> None:
    """The flag rewrite guards statements AFTER the statement containing a
    raise, not statements after the raise inside its own block: a matching
    ``raise _FGoto`` must be the last statement of whatever block holds it,
    or the rewrite would run its trailing siblings."""

    def matching(node: ast.AST) -> bool:
        return bool(
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "_FGoto"
            and (
                label is None
                or (
                    node.exc.args
                    and isinstance(node.exc.args[0], ast.Constant)
                    and node.exc.args[0].value == label
                )
            )
        )

    for stmt in body:
        for node in ast.walk(stmt):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    for at, inner in enumerate(block):
                        if matching(inner) and at != len(block) - 1:
                            raise NotFlat("a goto that is not the last statement of its branch")


def _fold(expr: ast.expr) -> ast.expr:
    """Trivial constant folding, so two spellings of one bound compare equal:
    ``(0) - (0)`` is ``0`` and ``n - 0 + 1`` is ``n + 1`` -- the translation
    spells a Fortran bound differently by context, not by value."""

    class Fold(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
            self.generic_visit(node)
            left, right = node.left, node.right
            if (
                isinstance(left, ast.Constant)
                and isinstance(right, ast.Constant)
                and isinstance(left.value, int | float)
                and isinstance(right.value, int | float)
            ):
                if isinstance(node.op, ast.Add):
                    return ast.Constant(left.value + right.value)
                if isinstance(node.op, ast.Sub):
                    return ast.Constant(left.value - right.value)
            if (
                isinstance(right, ast.Constant)
                and right.value == 0
                and isinstance(node.op, (ast.Add, ast.Sub))
            ):
                return left
            if isinstance(left, ast.Constant) and left.value == 0 and isinstance(node.op, ast.Add):
                return right
            return node

    folded: ast.expr = Fold().visit(copy.deepcopy(expr))
    return ast.fix_missing_locations(folded)


def _advances(body: list[ast.stmt], name: str) -> bool:
    """Whether ``name`` is stepped by a positive integer constant at the
    body's top level -- ``name = name + c`` or ``name += c``."""

    def positive(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) and isinstance(value.value, int) and value.value > 0

    for statement in body:
        if (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
            and isinstance(statement.op, ast.Add)
            and positive(statement.value)
        ):
            return True
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and isinstance(statement.value, ast.BinOp)
            and isinstance(statement.value.op, ast.Add)
        ):
            left, right = statement.value.left, statement.value.right
            if (isinstance(left, ast.Name) and left.id == name and positive(right)) or (
                isinstance(right, ast.Name) and right.id == name and positive(left)
            ):
                return True
    return False


class _WhileLoops(ast.NodeTransformer):
    """``while True: ... break`` as ``lax.while_loop``: every ``break``
    sets a flag, every later statement of the body runs under ``if not
    flag``, the body is lowered the way the backend lowers a kernel, and
    the loop carries what the body assigns.

    A ``while`` whose condition bounds a counter by a constant (``n <=
    nmax`` with ``nmax = 50`` above) becomes instead a ``for`` over that
    constant with the condition as a guard -- a fixed trip count, which
    ``lax.fori_loop`` lowers to a scan reverse mode can transpose; the
    fixed-point iterations of the paper's Phase 5 are run this way."""

    def __init__(
        self,
        constants: dict[str, int] | None = None,
        int_locals: frozenset[str] | None = None,
    ) -> None:
        self.n = 0
        self.constants = constants or {}
        self.int_locals = int_locals or frozenset()

    def visit_block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for statement in stmts:
            result: Any = self.visit(statement)
            if result is None:
                continue
            out.extend(result if isinstance(result, list) else [result])
        return out

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node  # nested (already lowered) bodies stay as they are

    def _int_typed(self, name: str) -> bool:
        return name in self.int_locals

    def _counted_bound(self, test: ast.expr, body: list[ast.stmt]) -> int | None:
        """The constant a counter is compared against in ``test`` --
        ``n <= nmax`` or ``n < nmax`` -- when ``nmax`` is a known integer
        and the body advances ``n`` by a positive constant on every pass
        (``n = n + 1`` or ``n += 1`` at the body's top level, outside any
        branch): only then is ``nmax + 1`` the trip count."""
        for node in ast.walk(test):
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.Lt, ast.LtE))
                and isinstance(node.left, ast.Name)
            ):
                bound = node.comparators[0]
                if not _advances(body, node.left.id):
                    continue
                if isinstance(bound, ast.Constant) and isinstance(bound.value, int):
                    return int(bound.value)
                if isinstance(bound, ast.Name) and bound.id in self.constants:
                    return self.constants[bound.id]
        return None

    def _unwrap_goto_region(self, node: ast.While) -> str | None:
        """``while True: try: ... break; except _FGoto: pass`` -- the numpy
        side's backward-goto region. The raise restarts the loop; spelled as
        a flag (raise -> ``_restart = True``, later statements and the final
        break guarded on it), the body is a plain ``while True: ... break``
        the lowering below already takes."""
        if not (
            isinstance(node.test, ast.Constant)
            and node.test.value is True
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Try)
        ):
            return None
        wrapper = node.body[0]
        if len(wrapper.handlers) != 1 or wrapper.orelse or wrapper.finalbody:
            return None
        handler = wrapper.handlers[0]
        if not (isinstance(handler.type, ast.Name) and handler.type.id == "_FGoto"):
            return None
        # A label is whatever constant ``_FGoto`` was raised with -- the
        # translation spells them as it read them, so no one type is assumed.
        labels: set[Any] = {
            n.exc.args[0].value
            for stmt in wrapper.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
            and n.exc.func.id == "_FGoto"
            and n.exc.args
            and isinstance(n.exc.args[0], ast.Constant)
        }
        if not labels:
            # No restart survived (each raise sat on a dropped abort path, or
            # the region never loops): the wrapper is the body.
            node.body = [s for s in wrapper.body if not isinstance(s, ast.Break)] + [ast.Break()]
            return None
        if len(labels) > 1:
            raise NotFlat(f"one goto region, several labels: {sorted(labels, key=str)}")
        label = labels.pop()
        self.n += 1
        restart = f"_restart_{self.n}"

        _require_trailing_raises(wrapper.body, label)

        class Restarts(ast.NodeTransformer):
            def visit_Raise(self, raised: ast.Raise) -> Any:
                exc = raised.exc
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "_FGoto"
                    and exc.args
                    and isinstance(exc.args[0], ast.Constant)
                    and exc.args[0].value == label
                ):
                    return ast.Assign(
                        targets=[ast.Name(id=restart, ctx=ast.Store())], value=ast.Constant(True)
                    )
                return raised

            def visit_While(self, inner: ast.While) -> ast.AST:
                return inner  # a raise crossing a nested loop is not a flag

            def visit_For(self, inner: ast.For) -> ast.AST:
                return inner

            def visit_FunctionDef(self, inner: ast.FunctionDef) -> ast.AST:
                return inner

        guarded: list[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=restart, ctx=ast.Store())], value=ast.Constant(False))
        ]
        restarted = False
        for statement in wrapper.body:
            if isinstance(statement, ast.Break):
                continue  # the natural exit; re-emitted guarded below
            had = any(
                isinstance(n, ast.Raise)
                and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name)
                and n.exc.func.id == "_FGoto"
                for n in ast.walk(statement)
            )
            rewritten = Restarts().visit(statement)
            if any(
                isinstance(n, ast.Raise)
                and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name)
                and n.exc.func.id == "_FGoto"
                for n in ast.walk(rewritten)
            ):
                raise NotFlat(f"a goto crossing a nested loop (label {label})")
            if restarted:
                guarded.append(
                    ast.If(
                        test=ast.UnaryOp(
                            op=ast.Not(), operand=ast.Name(id=restart, ctx=ast.Load())
                        ),
                        body=[rewritten],
                        orelse=[],
                    )
                )
            else:
                guarded.append(rewritten)
            if had:
                restarted = True
        guarded.append(
            ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=restart, ctx=ast.Load())),
                body=[ast.Break()],
                orelse=[],
            )
        )
        node.body = guarded
        return restart

    def visit_Try(self, node: ast.Try) -> Any:
        """A forward-goto region -- ``try: ...; raise _FGoto('100') ...
        except _FGoto: pass`` outside any loop: the raise skips to the end
        of the region. As a flag: the raise sets it, every later statement
        runs under its negation, the handler disappears."""
        self.generic_visit(node)
        if len(node.handlers) != 1 or node.orelse or node.finalbody:
            return node
        handler = node.handlers[0]
        if not (isinstance(handler.type, ast.Name) and handler.type.id == "_FGoto"):
            return node

        def restarts(stmt: ast.stmt) -> bool:
            return any(
                isinstance(n, ast.Raise)
                and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name)
                and n.exc.func.id == "_FGoto"
                for n in ast.walk(stmt)
            )

        label = next(
            (
                n.value
                for stmt in handler.body
                for n in ast.walk(stmt)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            ),
            None,
        )
        self.n += 1
        skip = f"_skip_{self.n}"
        _require_trailing_raises(node.body, label)

        class Skips(ast.NodeTransformer):
            def visit_Raise(self, raised: ast.Raise) -> Any:
                if (
                    isinstance(raised.exc, ast.Call)
                    and isinstance(raised.exc.func, ast.Name)
                    and raised.exc.func.id == "_FGoto"
                    and (
                        label is None
                        or (
                            raised.exc.args
                            and isinstance(raised.exc.args[0], ast.Constant)
                            and raised.exc.args[0].value == label
                        )
                    )
                ):
                    return ast.Assign(
                        targets=[ast.Name(id=skip, ctx=ast.Store())], value=ast.Constant(True)
                    )
                return raised

            def visit_While(self, inner: ast.While) -> ast.AST:
                return inner  # a goto crossing a loop is not a flag

            def visit_For(self, inner: ast.For) -> ast.AST:
                return inner

            def visit_FunctionDef(self, inner: ast.FunctionDef) -> ast.AST:
                return inner

        out: list[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=skip, ctx=ast.Store())], value=ast.Constant(False))
        ]
        skipped = False
        for statement in node.body:
            had = restarts(statement)
            rewritten = Skips().visit(statement)
            if restarts(rewritten):
                raise NotFlat("a forward goto crossing a nested loop")
            if skipped:
                out.append(
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=skip, ctx=ast.Load())),
                        body=[rewritten],
                        orelse=[],
                    )
                )
            else:
                out.append(rewritten)
            if had:
                skipped = True
        for statement in out:
            ast.fix_missing_locations(statement)
        return out

    def visit_While(self, node: ast.While) -> Any:
        from recast.transform.jax.backend import JaxQueue, KernelLowerer, _assigned_names

        restart = self._unwrap_goto_region(node)
        self.generic_visit(node)
        if node.orelse:
            return node
        self.n += 1
        done = f"_done_{self.n}"
        forever = isinstance(node.test, ast.Constant) and node.test.value is True
        counted = self._counted_bound(node.test, node.body)
        if counted is not None and not any(isinstance(n, ast.Break) for n in ast.walk(node)):
            self.n -= 1
            return ast.For(
                target=ast.Name(id=f"_w{self.n + 1}", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(0), ast.Constant(counted + 1)],
                    keywords=[],
                ),
                body=[ast.If(test=node.test, body=node.body, orelse=[])],
                orelse=[],
            )

        class Breaks(ast.NodeTransformer):
            def visit_Break(self, node: ast.Break) -> ast.AST:
                return ast.Assign(
                    targets=[ast.Name(id=done, ctx=ast.Store())], value=ast.Constant(True)
                )

            def visit_For(self, node: ast.For) -> ast.AST:
                return node  # a break inside an inner loop is that loop's

            def visit_While(self, node: ast.While) -> ast.AST:
                return node

        guarded: list[ast.stmt] = []
        exited = False
        for statement in node.body:
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(statement))
            rewritten: Any = Breaks().visit(statement)
            if exited:
                guarded.append(
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=done, ctx=ast.Load())),
                        body=[rewritten],
                        orelse=[],
                    )
                )
            else:
                guarded.append(rewritten)
            if has_break:
                exited = True
        for statement in guarded:
            ast.fix_missing_locations(statement)
        carried = [n for n in _assigned_names(guarded) if n != done]  # type: ignore[no-untyped-call]
        state = [*carried, done]
        try:
            lowered = KernelLowerer().lower_block(guarded, 1)  # type: ignore[no-untyped-call]
        except JaxQueue as why:
            raise NotFlat(f"while loop body: {why}") from why
        unpack = ast.Assign(
            targets=[
                ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Store()) for n in state], ctx=ast.Store())
            ],
            value=ast.Name(id="_c", ctx=ast.Load()),
        )
        pack = ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in state], ctx=ast.Load())
        arguments = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="_c")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        going: ast.expr = _jnp("logical_not", [ast.Name(id=done, ctx=ast.Load())])
        if not forever:
            # ``while cond:`` -- the condition over the carried state, and
            # not after a break either.
            going = _jnp("logical_and", [going, copy.deepcopy(node.test)])
        cond_fn = ast.FunctionDef(
            name=f"_wcond_{self.n}",
            args=arguments,
            body=[unpack, ast.Return(value=going)],
            decorator_list=[],
            returns=None,
        )
        body_fn = ast.FunctionDef(
            name=f"_wbody_{self.n}",
            args=copy.deepcopy(arguments),
            body=[copy.deepcopy(unpack), *lowered, ast.Return(value=pack)],
            decorator_list=[],
            returns=None,
        )
        loop = ast.Assign(
            targets=[copy.deepcopy(unpack.targets[0])],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="lax", ctx=ast.Load()), attr="while_loop", ctx=ast.Load()
                ),
                args=[
                    ast.Name(id=cond_fn.name, ctx=ast.Load()),
                    ast.Name(id=body_fn.name, ctx=ast.Load()),
                    copy.deepcopy(pack),
                ],
                keywords=[],
            ),
        )
        if all(
            n == done or n.startswith(("_restart_", "_skip_")) or self._int_typed(n) for n in state
        ):
            # A loop over integers alone (the calendar normalization): its
            # outputs are piecewise-constant in every real input, the
            # cotangent is zero, and reverse mode need not transpose it.
            loop.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="lax", ctx=ast.Load()),
                    attr="stop_gradient",
                    ctx=ast.Load(),
                ),
                args=[loop.value],
                keywords=[],
            )
        init = ast.Assign(targets=[ast.Name(id=done, ctx=ast.Store())], value=ast.Constant(False))
        pre: list[ast.stmt] = [init]
        if restart is not None:
            # The restart flag is loop-carried; without a value before the
            # loop the initial carry tuple cannot be built.
            pre.insert(
                0,
                ast.Assign(
                    targets=[ast.Name(id=restart, ctx=ast.Store())], value=ast.Constant(False)
                ),
            )
        return [*pre, cond_fn, body_fn, loop]


def _bind_state(piece: str, plans: list[FlatPlan]) -> str:
    """The host wrapper of a flat kernel sets the module state its plan
    carries before the call, the way the NumPy adapter does: a helper the
    kernel calls (a lookup over a grid the run filled) reads that state
    through its module at trace time, and the value has to be there. Read
    at trace time, it is a constant of the compiled kernel -- right for a
    table the run set once, and the reason such state is not differentiated.
    """
    for plan in plans:
        head = f"def {plan.name}("
        if not piece.startswith(head):
            continue
        lines = piece.split("\n")
        binds: list[str] = []
        for obj in plan.objects:
            if obj.kind == "state" and obj.module:
                binds.append(f"    import {obj.module}_numpy as _{obj.module}")
                binds.append(
                    f"    if not hasattr(getattr(_{obj.module}, {obj.name!r}, None), '__dict__'):"
                )
                binds.append(f"        _{obj.module}.{obj.name} = _host._Record()")
                for comp in obj.components:
                    binds.append(
                        f"    _{obj.module}.{obj.name}.{comp.name} = jnp.asarray({comp.flat})"
                    )
        for state in plan.states:
            binds.append(f"    import {state.module}_numpy as _{state.module}")
            binds.append(f"    _{state.module}.{state.name} = jnp.asarray({state.flat})")
        if not binds:
            return piece
        # def line, docstring line, then the binds, then the return.
        return "\n".join([*lines[:2], *binds, *lines[2:]])
    return piece


def _table_read(node: ast.expr) -> bool:
    """``MDAYLEAP[mcmnth - 1]``: a module constant table read. Its dtype is
    the table's (int64 for a spelled integer array), not the guard-typed
    local's -- the same cast the constant-expression case needs."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id.isupper()
        and len(node.value.id) > 1
    )


def _static_expression(node: ast.expr, statics: frozenset[str]) -> bool:
    """Arithmetic over numeric literals, upper-case constants, static scalar
    dummies and concrete locals only: a Python value at trace time."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id.isupper() or node.id in statics
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
    ):
        return _static_expression(node.left, statics) and _static_expression(node.right, statics)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _static_expression(node.operand, statics)
    return False


def _constant_expression(node: ast.expr, loop_vars: set[str] | None = None) -> bool:
    """Numeric literals, upper-case constants and (given) loop counters under
    arithmetic only -- what a Python int or an int64 counter would type
    differently from the local's own int32."""
    names = loop_vars or set()
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id.isupper() or node.id in names
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)
    ):
        return _constant_expression(node.left, names) and _constant_expression(node.right, names)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _constant_expression(node.operand, names)
    return False


def _integer_constant_expression(node: ast.expr) -> bool:
    """Integer literals under arithmetic only -- no names, whose type is
    not known here."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
    ):
        return _integer_constant_expression(node.left) and _integer_constant_expression(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _integer_constant_expression(node.operand)
    return False


def _integer_inits(fn: ast.FunctionDef) -> dict[str, int]:
    """Locals assigned one integer literal at the top of the body
    (``itmax = 40``), by value."""
    values: dict[str, int] = {}
    for statement in fn.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            continue
        name = statement.targets[0].id
        value = statement.value
        # The last assignment wins: the guard init (``nmax = 0``) comes
        # first, the value the source gives it after -- a literal, or the
        # emitter's ``I_<n>`` name for one; anything else is not a constant.
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, int)
            and not isinstance(value.value, bool)
        ):
            values[name] = int(value.value)
        elif isinstance(value, ast.Name) and re.fullmatch(r"I_\d+", value.id):
            values[name] = int(value.id[2:])
        else:
            values.pop(name, None)
    return values


def _guard_inits(fn: ast.FunctionDef) -> dict[str, str]:
    """The strong type of each local from its guard init at the top of the
    body: ``x = 0`` is an int32, ``x = 0.0`` a float64."""
    inits: dict[str, str] = {}
    for statement in fn.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, (int, float))
            and not isinstance(statement.value.value, bool)
        ):
            inits.setdefault(
                statement.targets[0].id,
                "float64" if isinstance(statement.value.value, float) else "int32",
            )
    return inits


def _union_states(plans: list[FlatPlan]) -> FlatPlan | None:
    """One plan carrying every module-state variable and object any plan
    in the module carries, for the helpers to spell theirs by."""
    if not plans:
        return None
    seen_vars: dict[str, Any] = {}
    seen_objs: dict[str, Any] = {}
    for plan in plans:
        for state in plan.states:
            seen_vars.setdefault(state.flat, state)
        for obj in plan.objects:
            if obj.kind == "state":
                held = seen_objs.setdefault(obj.name, copy.deepcopy(obj))
                have = {c.name for c in held.components}
                held.components.extend(c for c in obj.components if c.name not in have)
    union = copy.deepcopy(plans[0])
    union.objects = list(seen_objs.values())
    union.states = list(seen_vars.values())
    return union


def _pass_helper_state(
    rewritten: dict[str, ast.FunctionDef],
    specialized: dict[str, tuple[ast.FunctionDef, FlatPlan, list[str]]],
    helper_state: dict[str, list[str]],
    refused: dict[str, str],
) -> None:
    """Every call of a helper that took module state as parameters passes
    it, and a helper that calls such a helper takes those parameters too --
    to a fixpoint. A flat function has them as its own arguments; one that
    does not (its plan never reached that state) is refused."""
    if not helper_state:
        return
    functions = {**rewritten, **{n: f for n, (f, _, _) in specialized.items()}}
    changed = True
    while changed:
        changed = False
        for name, fn in functions.items():
            own_params = {a.arg for a in fn.args.args}
            needed: list[str] = []
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in helper_state
                ):
                    for state in helper_state[node.func.id]:
                        if state not in needed:
                            needed.append(state)
            if not needed:
                continue
            missing = [n for n in needed if n not in own_params]
            if (missing and name in helper_state) or (missing and name not in specialized):
                if name in helper_state:
                    helper_state[name] = [*helper_state[name], *missing]
                    fn.args.args.extend(ast.arg(arg=n) for n in missing)
                    changed = True
                elif name.endswith("_flat"):
                    refused[name] = f"a helper needs {missing[0]}, which this plan does not carry"
                    continue
                else:
                    helper_state[name] = missing
                    fn.args.args.extend(ast.arg(arg=n) for n in missing)
                    changed = True

    class Pass(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in helper_state:
                have = {ast.unparse(a) for a in node.args}
                for state in helper_state[node.func.id]:
                    if state not in have:
                        node.args.append(ast.Name(id=state, ctx=ast.Load()))
            return node

    for fn in functions.values():
        Pass().visit(fn)
        ast.fix_missing_locations(fn)


def _static_names(args: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        _py(a["name"])
        for a in args
        if not a.get("dims")
        and a["intent"] == "IN"
        and _py(a["name"]) not in TRACED_SCALARS
        and (a["dtype"] == "int32" or (a["dtype"] in ("int64", "float64") and "__" in a["name"]))
    )


def flat_function(
    fn: ast.FunctionDef,
    plan: FlatPlan,
    plans: dict[str, FlatPlan],
    aliases: dict[str, str],
    ports: dict[str, dict[str, Any]],
    bundled: frozenset[str] = frozenset(),
    own: dict[str, Any] | None = None,
    specialized: dict[str, tuple[ast.FunctionDef, FlatPlan, list[str]]] | None = None,
) -> tuple[ast.FunctionDef, _Rewrite]:
    """The anchor's function on the flat signature, and the rewrite that
    made it (its callees, dropped aborts, companions used)."""
    rewrite = _Rewrite(
        plan,
        _Spelling(plan, aliases),
        plans,
        ports,
        bundled,
        _static_names(plan.flat_args),
        _dim_sources(plan.flat_args),
    )
    rewrite.own = own or {}
    if specialized is not None:
        rewrite.specialized = specialized
    body = _rewritten_body(fn, rewrite)
    if not body or not isinstance(body[-1], ast.Return):
        body.append(ast.Return(value=_tuple(_outputs(plan))))
    taken = [_py(a["name"]) for a in plan.flat_args if a["intent"] != "OUT"]
    flat = ast.FunctionDef(
        name=plan.name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in taken],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
    )
    ast.fix_missing_locations(flat)
    return flat, rewrite


def scrubbed_function(
    fn: ast.FunctionDef,
    aliases: dict[str, str],
    ports: dict[str, dict[str, Any]],
    bundled: frozenset[str] = frozenset(),
    statics: frozenset[str] = frozenset(),
    dim_sources: dict[str, tuple[str, int]] | None = None,
    own: dict[str, Any] | None = None,
    specialized: dict[str, tuple[ast.FunctionDef, FlatPlan, list[str]]] | None = None,
    states: FlatPlan | None = None,
) -> tuple[ast.FunctionDef, _Rewrite]:
    """A function with no plan, scrubbed the same way. ``states`` is a plan
    of every module-state variable and object the module's plans carry: a
    helper that reads one through its module (``_mlcl.zdtgridm``, a grid
    the run filled) gets it as a parameter instead, so nothing is read from
    a module at trace time -- the callers pass their own flat argument."""
    spelling = _Spelling(states, aliases)
    spelling.dummies = {}  # only module state: a helper has no object dummy
    rewrite = _Rewrite(None, spelling, {}, ports, bundled, statics, dim_sources)
    rewrite.own = own or {}
    if specialized is not None:
        rewrite.specialized = specialized
    scrubbed = copy.deepcopy(fn)
    scrubbed.body = _rewritten_body(fn, rewrite)
    for name in sorted(spelling.used):
        scrubbed.args.args.append(ast.arg(arg=name))
    rewrite.state_params = sorted(spelling.used)
    ast.fix_missing_locations(scrubbed)
    return scrubbed, rewrite


def _entry(plan: FlatPlan, calls: list[str]) -> dict[str, Any]:
    """The interface record of the flat function, for the backend."""
    sub = plan.subprogram
    return {
        **signature(plan),
        "name": plan.name,
        "public": True,
        "kind": sub["kind"],
        "result": "_result" if sub["kind"] == "function" else None,
        "result_dtype": sub.get("result_dtype") if sub["kind"] == "function" else None,
        "module_state_read": [],
        "module_state_written": [],
        "calls": calls,
        "present_calls": [],
    }


def flattened_module(
    tree: ast.Module,
    interface: dict[str, Any],
    plans: list[FlatPlan],
    ports: dict[str, dict[str, Any]] | None = None,
    bundled: frozenset[str] = frozenset(),
) -> tuple[ast.Module, dict[str, Any], dict[str, Any]]:
    """The anchor with each planned subprogram's flat function in place of
    its ``_flat`` wrapper, every other function scrubbed, and the interface
    with the flat records. Returns ``(module, interface, notes)``:
    ``notes["refused"]`` names what the rewrite could not spell and why,
    ``notes["aborts_dropped"]`` the abort checks each function lost,
    ``notes["companions"]`` the ``_alias`` names whose ported kernels are
    called."""
    ports = ports or {}
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    by_name = {p.subprogram["name"].lower(): p for p in plans}
    aliases = _module_aliases(tree)
    rewritten: dict[str, ast.FunctionDef] = {}
    entries: list[dict[str, Any]] = []
    refused: dict[str, str] = {}
    helper_state: dict[str, list[str]] = {}  # helper -> the state parameters it took
    own = {"fns": fns, "records": {s["name"]: s for s in interface["subprograms"]}}
    specialized: dict[str, tuple[ast.FunctionDef, FlatPlan, list[str]]] = {}
    aborts: dict[str, list[str]] = {}
    masked: dict[str, list[str]] = {}
    static_loops: dict[str, list[str]] = {}
    companions: set[str] = set()
    planned = {p.subprogram["name"] for p in plans}
    states = _union_states(plans)
    for plan in plans:
        fn = fns.get(plan.subprogram["name"])
        if fn is None:
            refused[plan.name] = "no python function in the anchor"
            continue
        try:
            flat, rewrite = flat_function(
                fn, plan, by_name, aliases, ports, bundled, own, specialized
            )
        except NotFlat as why:
            refused[plan.name] = str(why)
            entries.append(_entry(plan, []))  # known, unlowered: callers delegate by name
            continue
        if rewrite.aborts:
            aborts[plan.name] = rewrite.aborts
        if rewrite.masked:
            masked[plan.name] = rewrite.masked
        if rewrite.static_loops:
            static_loops[plan.name] = rewrite.static_loops
        companions |= rewrite.companions
        rewritten[plan.name] = flat
        entries.append(_entry(plan, rewrite.calls))
    for name, fn in fns.items():
        if name in planned or name.endswith("_flat") or name.startswith("_"):
            continue
        try:
            record = next((s for s in interface["subprograms"] if s["name"] == name), None)
            statics = _static_names(record["args"]) if record else frozenset()
            sources = _dim_sources(record["args"]) if record else {}
            scrubbed, rewrite = scrubbed_function(
                fn, aliases, ports, bundled, statics, sources, own, specialized, states
            )
        except NotFlat as why:
            refused[name] = str(why)
            continue
        if rewrite.aborts:
            aborts[name] = rewrite.aborts
        if rewrite.masked:
            masked[name] = rewrite.masked
        if rewrite.static_loops:
            static_loops[name] = rewrite.static_loops
        if rewrite.state_params:
            helper_state[name] = list(rewrite.state_params)
        companions |= rewrite.companions
        rewritten[name] = scrubbed
    _pass_helper_state(rewritten, specialized, helper_state, refused)
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in rewritten:
            body.append(rewritten.pop(node.name))
        else:
            body.append(node)
    body.extend(rewritten.values())  # plans without a NumPy wrapper (private, functions)
    for spec_fn, spec_plan, spec_calls in specialized.values():
        body.append(spec_fn)
        entries.append(_entry(spec_plan, spec_calls))
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    notes = {
        "specialized_plans": [spec_plan for _, spec_plan, _ in specialized.values()],
        "refused": refused,
        "aborts_dropped": aborts,
        "masked": masked,
        "static_loops": static_loops,
        "companions": sorted(companions),
    }
    return module, {**interface, "subprograms": [*interface["subprograms"], *entries]}, notes


class TreeToJax(KernelToJax):
    """The tree translation first, the flat functions second, JAX third."""

    name = "recast.port.tree-to-jax"

    def __init__(self, conventions: TreeConventions | None = None) -> None:
        self.conventions = conventions or TreeConventions()
        self._tree = TreeTranslation(self.conventions)
        self._anchor = self._tree  # type: ignore[assignment]

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        from recast.transform.jax.backend import HEADER, build_module, emit_runtime

        anchor = self._anchor.apply(unit, facts, config)
        module = facts.interface["module"]
        constants_stem = config.get("constants_stem", f"{module}_constants")
        runtime_stem = config.get("jax_runtime_stem", f"{module}_jax_runtime")

        global TRACED_SCALARS
        TRACED_SCALARS = frozenset(config.get("traced_scalars") or ())
        ported, files = self._port_companions(anchor, facts, config)
        tree = ast.parse(anchor.files[Path(f"{module}_numpy.py")].decode())
        plans = plans_from_facts(facts, gated=False)
        bundled = frozenset(anchor.notes.get(self._tree.notes_key, {}).get("bundled") or [])
        flat_tree, interface, flat_notes = flattened_module(
            tree, facts.interface, plans, ported, bundled
        )
        pieces, jitted, delegated = build_module(interface, flat_tree, TRACED_SCALARS)
        # A flat function the backend delegated has a host to fall back on
        # only if the NumPy module carries its wrapper -- the gated ones. A
        # private subprogram's or a function's flat form has none, and a line
        # binding it to a host attribute that does not exist would break the
        # import; its callers were delegated with it, so nothing needs it.
        pieces = [
            _bind_state(piece, [*plans, *flat_notes["specialized_plans"]]) for piece in pieces
        ]
        hosted = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)}
        pieces = [
            piece
            for piece in pieces
            if not (
                "\n" not in piece
                and " = _host." in piece
                and piece.split(" = _host.")[0] not in hosted
            )
        ]
        # The anchor's own imports: its use-constants, the stand-ins a kernel
        # may still read a resolved constant through, and the ported
        # companions its kernels call into.
        extra_imports = ""
        if Path(f"{module}_use_constants.py") in anchor.files:
            extra_imports = f"from {module}_use_constants import *  # noqa: F401,F403\n"
        aliases = _module_aliases(tree)
        for node in tree.body:
            if isinstance(node, ast.Import) and any(
                n.asname and n.name.endswith("_numpy") for n in node.names
            ):
                extra_imports += ast.unparse(node) + "\n"
        for alias in flat_notes["companions"]:
            if alias.startswith("module:"):
                # A specialized root finder's body is a companion's: it
                # reads the companion's constants.
                other = alias.split(":", 1)[1]
                extra_imports += f"from {other}_constants import *  # noqa: F401,F403\n"
                if Path(f"{other}_use_constants.py") in {**anchor.files, **files}:
                    extra_imports += f"from {other}_use_constants import *  # noqa: F401,F403\n"
                continue
            extra_imports += f"import {aliases[alias]}_jax as {alias}_jax\n"
        emitted = (
            HEADER.format(module=module, constants=constants_stem, runtime=runtime_stem)
            + extra_imports
            + "\n"
            + _signatures_of(tree)
            + "\n\n\n".join(pieces)
        )
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={
                **anchor.files,
                **files,
                Path(f"{runtime_stem}.py"): emit_runtime().encode(),
                Path(f"{module}_jax.py"): (emitted + "\n").encode(),
            },
            deferred=list(anchor.deferred),
            notes={
                **anchor.notes,
                "jax": {
                    "anchor": f"{module}_numpy.py",
                    "kernels": sorted(jitted),
                    "delegated": dict(sorted(delegated.items())),
                    "flat_refused": dict(sorted(flat_notes["refused"].items())),
                    "aborts_dropped": dict(sorted(flat_notes["aborts_dropped"].items())),
                    "masked": dict(sorted(flat_notes["masked"].items())),
                    "static_loops": dict(sorted(flat_notes["static_loops"].items())),
                    "companions": sorted(ported),
                    "runtime": f"{runtime_stem}.py",
                    "_ports": ported,
                },
            },
        )

    def _port_companions(
        self, anchor: Candidate, facts: Facts, config: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], dict[Path, bytes]]:
        """Port every companion bundled into the anchor, so a kernel that
        calls into one calls its kernel. Returns what each port lowered and
        the closure of each kernel (for the caller to spell), and the files."""
        from recast.registry import REGISTRY
        from recast.transform.jax.backend import state_closure, write_closure

        bundled = list(anchor.notes.get(self._tree.notes_key, {}).get("bundled") or [])
        done: set[str] = set(config.get("_ported") or ())
        done.add(str(facts.interface.get("module", "")).lower())
        # Ports the caller already made are reused: a companion of a
        # companion is the caller's companion too, and its kernels are what
        # the inner rewrite has to spell.
        ports: dict[str, dict[str, Any]] = dict(config.get("_ports") or {})
        files: dict[Path, bytes] = {}
        if not bundled:
            return ports, files
        root = Path(config.get("root", ".")).resolve()
        frontend = REGISTRY.get("frontend", self.conventions.frontend)()
        units = {u.uid: u for u in frontend.discover(root)}
        for module in bundled:
            if module in done:
                continue
            done.add(module)
            unit = units.get(f"fortran:{module}")
            if unit is None:
                raise ConfigError(
                    f"companion {module!r} of {facts.unit!r} was bundled into the anchor but "
                    f"has no unit under {root} to port"
                )
            companion_facts = frontend.analyze(unit, root)
            inner = self.apply(unit, companion_facts, {**config, "_ported": done, "_ports": ports})
            subs = {s["name"]: s for s in companion_facts.interface["subprograms"]}
            kernels = set(inner.notes["jax"]["kernels"])
            memo: dict[str, set[str]] = {}
            wmemo: dict[str, set[str]] = {}
            anchor_tree = ast.parse(inner.files[Path(f"{module}_numpy.py")].decode())
            ports[module] = {
                "module": module,
                "kernels": sorted(kernels),
                "fns": {n.name: n for n in anchor_tree.body if isinstance(n, ast.FunctionDef)},
                "records": dict(subs),
                "plans": {
                    p.subprogram["name"].lower(): p
                    for p in plans_from_facts(companion_facts, gated=False)
                },
                "closures": {
                    k: sorted(
                        state_closure(subs, kernels, k, memo)
                        | write_closure(subs, kernels, k, wmemo)
                    )
                    for k in kernels
                    if k in subs
                },
                "write_closures": {
                    k: sorted(write_closure(subs, kernels, k, wmemo)) for k in kernels if k in subs
                },
            }
            for other, port in (inner.notes["jax"].get("_ports") or {}).items():
                ports.setdefault(other, port)
            for path, content in inner.files.items():
                if path.name.endswith("_jax.py") or path.name.endswith("_jax_runtime.py"):
                    files.setdefault(path, content)
        return ports, files


def factory(**config: Any) -> TreeToJax:
    from recast.transform.numpy.tree import factory as tree_factory

    return TreeToJax(tree_factory(**config).conventions)
