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
from pathlib import Path
from typing import Any

from recast.fortran.flatten import FlatPlan, plans_from_facts, signature
from recast.model import Candidate, Facts, Unit
from recast.transform.jax.translate import KernelToJax, _signatures_of
from recast.transform.numpy.tree import TreeConventions, TreeTranslation

__all__ = ["TreeToJax", "factory", "flattened_module"]

CASTS = ("float64", "float32", "int32", "int64")


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
    ) -> None:
        self.plan = plan
        self.spelling = spelling
        self.plans = plans
        self.ports = ports  # companion module -> {"kernels", "closures"}
        self.aliases: dict[str, str] = {}  # alias -> flat name, from dropped assignments
        self.calls: list[str] = []
        self.aborts: list[str] = []
        self.companions: set[str] = set()  # ``_alias`` names whose ported module is called
        self.temps = 0

    # -- names and attributes -------------------------------------------------

    def visit_Name(self, node: ast.Name) -> ast.AST:
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
        # ``alias = obj.comp`` -> dropped; the alias is the component from here.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self.spelling.of(node.value) is not None
        ):
            self.aliases[node.targets[0].id] = self.spelling.of(node.value) or ""
            return None
        if isinstance(node.value, ast.Call) and self._flat_callee(node.value) is not None:
            return self._rewrite_call(node.targets[0], node.value)
        self.generic_visit(node)
        # ``x[p - 1, :], y = f(...)``: the backend lowers a subscript store on
        # its own statement, not inside a tuple target.
        target = node.targets[0]
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
            return self.visit(ast.copy_location(ast.Assign(targets=[dst], value=src), node))
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> Any:
        # ``if bad: raise`` -- the anchor's ``endrun``. A kernel cannot raise
        # under tracing, and the recorded run it is gated on never did; the
        # check is dropped and named on the candidate, so the evidence says
        # the kernel no longer aborts where the model would.
        if node.body and all(isinstance(s, (ast.Raise, ast.Expr, ast.Pass)) for s in node.body):
            if any(isinstance(s, ast.Raise) for s in node.body):
                self.aborts.append(ast.unparse(node.test))
                return [self.visit(s) for s in node.orelse] if node.orelse else None
        self.generic_visit(node)
        if not node.body:
            return [*node.orelse] if node.orelse else None
        return node

    def visit_Raise(self, node: ast.Raise) -> Any:
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
            return ast.copy_location(_jnp("int32", node.args), node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in CASTS
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Constant)
        ):
            return ast.copy_location(_jnp(node.func.attr, node.args), node)
        return self._companion_call(node)

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
            # Not a ported companion: a stand-in. Its functions are the
            # framework's answers, not physics, and are left as they are.
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
        if isinstance(call.func, ast.Name):
            callee = self.plans.get(call.func.id.lower())
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            module = self.spelling.modules.get(call.func.value.id)
            port = self.ports.get(module or "")
            if port is not None:
                callee = port["plans"].get(call.func.attr.lower())
        if callee is None:
            return None
        if any(self.spelling.object_of(a) is not None for a in call.args) or any(
            self.spelling.object_of(k.value) is not None for k in call.keywords
        ):
            return callee
        return None

    def _rewrite_call(self, target: ast.expr | None, call: ast.Call) -> Any:
        callee = self._flat_callee(call)
        assert callee is not None and self.plan is not None
        dummies = [a["name"] for a in callee.subprogram["args"]]
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
                args.append(rewritten)
        func: ast.expr = ast.Name(id=callee.name, ctx=ast.Load())
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
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
        assign = ast.Assign(targets=[ast.Tuple(elts=targets, ctx=ast.Store())], value=new_call)
        return [assign, *follow] if follow else assign


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


def _rewritten_body(fn: ast.FunctionDef, rewrite: _Rewrite) -> list[ast.stmt]:
    lowered: list[ast.stmt] = []
    for statement in [copy.deepcopy(s) for s in fn.body]:
        result = rewrite.visit(statement)
        if result is None:
            continue
        lowered.extend(result if isinstance(result, list) else [result])
    return lowered


def flat_function(
    fn: ast.FunctionDef,
    plan: FlatPlan,
    plans: dict[str, FlatPlan],
    aliases: dict[str, str],
    ports: dict[str, dict[str, Any]],
) -> tuple[ast.FunctionDef, _Rewrite]:
    """The anchor's function on the flat signature, and the rewrite that
    made it (its callees, dropped aborts, companions used)."""
    rewrite = _Rewrite(plan, _Spelling(plan, aliases), plans, ports)
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
    fn: ast.FunctionDef, aliases: dict[str, str], ports: dict[str, dict[str, Any]]
) -> tuple[ast.FunctionDef, _Rewrite]:
    """A function with no plan, scrubbed the same way."""
    rewrite = _Rewrite(None, _Spelling(None, aliases), {}, ports)
    scrubbed = copy.deepcopy(fn)
    scrubbed.body = _rewritten_body(fn, rewrite)
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
    aborts: dict[str, list[str]] = {}
    companions: set[str] = set()
    planned = {p.subprogram["name"] for p in plans}
    for plan in plans:
        fn = fns.get(plan.subprogram["name"])
        if fn is None:
            refused[plan.name] = "no python function in the anchor"
            continue
        try:
            flat, rewrite = flat_function(fn, plan, by_name, aliases, ports)
        except NotFlat as why:
            refused[plan.name] = str(why)
            entries.append(_entry(plan, []))  # known, unlowered: callers delegate by name
            continue
        if rewrite.aborts:
            aborts[plan.name] = rewrite.aborts
        companions |= rewrite.companions
        rewritten[plan.name] = flat
        entries.append(_entry(plan, rewrite.calls))
    for name, fn in fns.items():
        if name in planned or name.endswith("_flat") or name.startswith("_"):
            continue
        try:
            scrubbed, rewrite = scrubbed_function(fn, aliases, ports)
        except NotFlat as why:
            refused[name] = str(why)
            continue
        if rewrite.aborts:
            aborts[name] = rewrite.aborts
        companions |= rewrite.companions
        rewritten[name] = scrubbed
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in rewritten:
            body.append(rewritten.pop(node.name))
        else:
            body.append(node)
    body.extend(rewritten.values())  # plans without a NumPy wrapper (private, functions)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    notes = {"refused": refused, "aborts_dropped": aborts, "companions": sorted(companions)}
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

        ported, files = self._port_companions(anchor, facts, config)
        tree = ast.parse(anchor.files[Path(f"{module}_numpy.py")].decode())
        plans = plans_from_facts(facts, gated=False)
        flat_tree, interface, flat_notes = flattened_module(tree, facts.interface, plans, ported)
        pieces, jitted, delegated = build_module(interface, flat_tree)
        # A flat function the backend delegated has a host to fall back on
        # only if the NumPy module carries its wrapper -- the gated ones. A
        # private subprogram's or a function's flat form has none, and a line
        # binding it to a host attribute that does not exist would break the
        # import; its callers were delegated with it, so nothing needs it.
        hosted = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)}
        pieces = [
            piece
            for piece in pieces
            if not (" = _host." in piece and piece.split(" = _host.")[0] not in hosted)
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
        from recast.transform.jax.backend import state_closure

        bundled = list(anchor.notes.get(self._tree.notes_key, {}).get("bundled") or [])
        done: set[str] = set(config.get("_ported") or ())
        done.add(str(facts.interface.get("module", "")).lower())
        ports: dict[str, dict[str, Any]] = {}
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
                continue
            companion_facts = frontend.analyze(unit, root)
            inner = self.apply(unit, companion_facts, {**config, "_ported": done})
            subs = {s["name"]: s for s in companion_facts.interface["subprograms"]}
            kernels = set(inner.notes["jax"]["kernels"])
            memo: dict[str, set[str]] = {}
            ports[module] = {
                "kernels": sorted(kernels),
                "plans": {
                    p.subprogram["name"].lower(): p
                    for p in plans_from_facts(companion_facts, gated=False)
                },
                "closures": {
                    k: sorted(state_closure(subs, kernels, k, memo))  # type: ignore[no-untyped-call]
                    for k in kernels
                    if k in subs
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
