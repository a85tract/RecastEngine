"""Pure-AST Python/NumPy to Numba and JAX transforms."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform

__all__ = ["PythonNumpyToJax", "PythonNumpyToNumba", "jax_factory", "numba_factory"]

_COMMON_REFUSALS = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Import,
    ast.ImportFrom,
)
_JAX_REFUSALS = (
    *_COMMON_REFUSALS,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Delete,
)
_JAX_NUMPY_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "arange",
        "array",
        "asarray",
        "clip",
        "concatenate",
        "cos",
        "dot",
        "einsum",
        "empty_like",
        "exp",
        "expm1",
        "log",
        "log1p",
        "matmul",
        "maximum",
        "mean",
        "minimum",
        "ones",
        "ones_like",
        "power",
        "reshape",
        "sin",
        "sqrt",
        "stack",
        "sum",
        "tanh",
        "transpose",
        "where",
        "zeros",
        "zeros_like",
    }
)
_JAX_ARRAY_METHODS = frozenset(
    {"astype", "clip", "max", "mean", "min", "reshape", "sum", "transpose"}
)
_DYNAMIC_CALLS = frozenset({"eval", "exec", "globals", "locals", "open", "compile", "input"})
_RESERVED_NAMES = frozenset(
    {
        "__recast_backend__",
        "__recast_compiled_functions__",
        "__recast_jax",
        "__recast_jnp",
        "__recast_numba",
    }
)


class _PythonAccelerator(Transform):
    requires = ("interface", "effects")
    deterministic = True
    backend: str
    suffix: str

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return (
            unit.kind == "python-module"
            and len(unit.sources) == 1
            and facts.provenance.get("frontend") == "python-numpy"
            and bool(facts.interface.get("exports"))
        )

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        source = self._source(unit, config)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(unit.sources[0]))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ConfigError(f"cannot transform {unit.sources[0]}: {error}") from error
        _reject_reserved_names(tree)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        exports = set(facts.interface.get("exports", ()))
        selected = _reachable(exports, functions)
        reasons = self._refusals(selected, functions, facts)
        translated = sorted(selected - reasons.keys())
        self._rewrite_module(tree, translated)
        ast.fix_missing_locations(tree)

        relative = unit.sources[0]
        target = relative.with_name(f"{relative.stem}_{self.suffix}.py")
        deferred = [f"{name}: {reasons[name]}" for name in sorted(reasons)]
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={target: (ast.unparse(tree).rstrip() + "\n").encode("utf-8")},
            deferred=deferred,
            notes={
                "backend": self.backend,
                "source": relative.as_posix(),
                "output": target.as_posix(),
                "exports": sorted(exports),
                "translated_functions": translated,
                "deferred_functions": dict(sorted(reasons.items())),
                # Language-neutral, source-free acceptance ledger consumed by
                # phased verification. A deferred entry is still attempted
                # coverage; static.complete/no-deferred decide readiness.
                "coverage": {"subprograms": sorted(selected)},
            },
        )

    @staticmethod
    def _source(unit: Unit, config: dict[str, Any]) -> Path:
        root = Path(config.get("root", ".")).resolve()
        source = (root / unit.sources[0]).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ConfigError(
                f"python source escapes the project root: {unit.sources[0]}"
            ) from error
        return source

    def _refusals(
        self,
        selected: set[str],
        functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        facts: Facts,
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for name in sorted(selected):
            reason = self._function_refusal(functions[name], facts)
            if reason:
                reasons[name] = reason
        # A translated caller must never silently call a local function that
        # the backend refused.  Propagate the refusal through the call graph.
        changed = True
        while changed:
            changed = False
            for name in sorted(selected - reasons.keys()):
                calls = _local_calls(functions[name], functions)
                refused = sorted(calls.intersection(reasons))
                if refused:
                    reasons[name] = f"calls deferred local function {refused[0]!r}"
                    changed = True
        return reasons

    def _function_refusal(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, facts: Facts
    ) -> str | None:
        raise NotImplementedError

    def _rewrite_module(self, tree: ast.Module, translated: list[str]) -> None:
        raise NotImplementedError


class PythonNumpyToNumba(_PythonAccelerator):
    """Add reviewed ``njit`` decorators to the reachable numerical closure."""

    name = "recast.translate.python-numpy-to-numba"
    backend = "numba"
    suffix = "numba"

    def _function_refusal(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, facts: Facts
    ) -> str | None:
        if node.decorator_list:
            return "existing decorators require manual ordering review"
        if node.args.vararg is not None or node.args.kwarg is not None:
            return "variadic arguments do not have a stable compilation signature"
        for child in ast.walk(node):
            if child is not node and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                return f"nested {type(child).__name__} has no reviewed Numba lowering"
            if isinstance(child, _COMMON_REFUSALS):
                return f"unsupported {type(child).__name__} in nopython mode"
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in _DYNAMIC_CALLS
            ):
                return f"dynamic call {child.func.id!r} is not compilable"
            if isinstance(child, ast.Call):
                imported = _imported_call(child, facts)
                if imported and not imported.startswith(("numpy", "math.")):
                    return f"imported call {imported!r} has no reviewed Numba lowering"
        return None

    def _rewrite_module(self, tree: ast.Module, translated: list[str]) -> None:
        _insert_after_future(
            tree,
            ast.Import(names=[ast.alias(name="numba", asname="__recast_numba")]),
        )
        selected = set(translated)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in selected:
                node.decorator_list.insert(
                    0,
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="__recast_numba", ctx=ast.Load()),
                            attr="njit",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[
                            ast.keyword(arg="cache", value=ast.Constant(False)),
                            ast.keyword(arg="fastmath", value=ast.Constant(False)),
                        ],
                    ),
                )
        _append_metadata(tree, "numba", translated)


class PythonNumpyToJax(_PythonAccelerator):
    """Lower a conservative, pure NumPy AST subset to JAX and ``jit`` it."""

    name = "recast.translate.python-numpy-to-jax"
    backend = "jax"
    suffix = "jax"

    def _function_refusal(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, facts: Facts
    ) -> str | None:
        if node.decorator_list:
            return "existing decorators require manual ordering review"
        if node.args.vararg is not None or node.args.kwarg is not None:
            return "variadic arguments do not have a stable JIT signature"
        for child in ast.walk(node):
            if child is not node and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                return f"nested {type(child).__name__} has no reviewed JAX lowering"
            if isinstance(child, _JAX_REFUSALS):
                return f"unsupported {type(child).__name__} in the pure JAX subset"
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets: list[ast.expr]
                if isinstance(child, ast.Assign):
                    targets = child.targets
                else:
                    targets = [child.target]
                if any(isinstance(target, (ast.Subscript, ast.Attribute)) for target in targets):
                    return "in-place array or attribute mutation is not JAX-functional"
            if isinstance(child, ast.Call):
                reason = _jax_call_refusal(
                    child, set(str(alias) for alias in facts.effects.get("numpy_aliases", ()))
                )
                if reason:
                    return reason
        return None

    def _rewrite_module(self, tree: ast.Module, translated: list[str]) -> None:
        aliases: set[str] = set()
        rewritten_body: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.name == "numpy":
                        alias = item.asname or "numpy"
                        aliases.add(alias)
                # Keep NumPy itself for evaluated annotations and module-level
                # constants. Only translated function bodies are lowered.
                rewritten_body.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
                rewritten_body.append(ast.ImportFrom(module="jax.numpy", names=node.names, level=0))
            else:
                rewritten_body.append(node)
        tree.body = rewritten_body
        _insert_after_future(
            tree,
            ast.Import(names=[ast.alias(name="jax.numpy", asname="__recast_jnp")]),
        )
        _insert_after_future(
            tree,
            ast.Import(names=[ast.alias(name="jax", asname="__recast_jax")]),
        )
        import_end = 0
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                import_end = index + 1
                continue
            break
        tree.body.insert(
            import_end,
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="__recast_jax", ctx=ast.Load()),
                            attr="config",
                            ctx=ast.Load(),
                        ),
                        attr="update",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant("jax_enable_x64"), ast.Constant(True)],
                    keywords=[],
                )
            ),
        )
        selected = set(translated)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in selected:
                _lower_numpy_names(node, aliases)
                node.decorator_list.insert(
                    0,
                    ast.Attribute(
                        value=ast.Name(id="__recast_jax", ctx=ast.Load()),
                        attr="jit",
                        ctx=ast.Load(),
                    ),
                )
        _append_metadata(tree, "jax", translated)


def _jax_call_refusal(call: ast.Call, numpy_aliases: set[str]) -> str | None:
    if isinstance(call.func, ast.Name):
        if call.func.id in _DYNAMIC_CALLS:
            return f"dynamic call {call.func.id!r} is not JIT-safe"
        return None
    if not isinstance(call.func, ast.Attribute):
        return "dynamic callable expressions are not in the reviewed JAX subset"
    path = _attribute_path(call.func)
    if len(path) == 2 and path[0] in numpy_aliases | {"jnp"}:
        if path[1] not in _JAX_NUMPY_CALLS:
            return f"NumPy call {'.'.join(path)!r} has no reviewed JAX lowering"
    elif len(path) >= 2 and path[0] in numpy_aliases | {"jnp"}:
        return f"nested NumPy API {'.'.join(path)!r} has no reviewed JAX lowering"
    elif path[-1] not in _JAX_ARRAY_METHODS:
        return f"method call {path[-1]!r} has no reviewed JAX lowering"
    return None


def _imported_call(call: ast.Call, facts: Facts) -> str | None:
    aliases = {
        str(alias): str(module)
        for alias, module in dict(facts.effects.get("import_aliases", {})).items()
    }
    if isinstance(call.func, ast.Name):
        return aliases.get(call.func.id)
    if not isinstance(call.func, ast.Attribute):
        return None
    path = _attribute_path(call.func)
    if not path or path[0] not in aliases:
        return None
    suffix = ".".join(path[1:])
    return f"{aliases[path[0]]}.{suffix}" if suffix else aliases[path[0]]


def _attribute_path(node: ast.Attribute) -> tuple[str, ...]:
    parts = [node.attr]
    value: ast.expr = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return tuple(reversed(parts))


def _reject_reserved_names(tree: ast.Module) -> None:
    """Prevent source bindings from capturing engine-owned generated names."""

    for node in ast.walk(tree):
        value: str | None = None
        if isinstance(node, ast.Name):
            value = node.id
        elif isinstance(node, ast.arg):
            value = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            value = node.name
        elif isinstance(node, ast.alias):
            value = node.asname or node.name.split(".", 1)[0]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        if value in _RESERVED_NAMES:
            raise ConfigError(f"source uses reserved accelerator binding {value!r}")


def _reachable(
    roots: set[str], functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
) -> set[str]:
    reached: set[str] = set()
    pending = sorted(roots)
    while pending:
        name = pending.pop()
        if name in reached or name not in functions:
            continue
        reached.add(name)
        pending.extend(sorted(_local_calls(functions[name], functions) - reached))
    return reached


def _local_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in functions
    }


def _insert_after_future(tree: ast.Module, statement: ast.stmt) -> None:
    index = 0
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            index = 1
    while index < len(tree.body):
        node = tree.body[index]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        index += 1
    tree.body.insert(index, statement)


def _append_metadata(tree: ast.Module, backend: str, translated: list[str]) -> None:
    tree.body.extend(
        [
            ast.Assign(
                targets=[ast.Name(id="__recast_backend__", ctx=ast.Store())],
                value=ast.Constant(backend),
            ),
            ast.Assign(
                targets=[ast.Name(id="__recast_compiled_functions__", ctx=ast.Store())],
                value=ast.List(elts=[ast.Constant(name) for name in translated], ctx=ast.Load()),
            ),
        ]
    )


class _NumpyNameLowerer(ast.NodeTransformer):
    def __init__(self, aliases: set[str]) -> None:
        self.aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            return ast.copy_location(ast.Name(id="__recast_jnp", ctx=node.ctx), node)
        return node


def _lower_numpy_names(node: ast.FunctionDef, aliases: set[str]) -> None:
    lowerer = _NumpyNameLowerer(aliases)
    node.body = [lowerer.visit(statement) for statement in node.body]
    node.args.defaults = [lowerer.visit(value) for value in node.args.defaults]
    node.args.kw_defaults = [
        lowerer.visit(value) if value is not None else None for value in node.args.kw_defaults
    ]


def numba_factory(**_config: Any) -> PythonNumpyToNumba:
    return PythonNumpyToNumba()


def jax_factory(**_config: Any) -> PythonNumpyToJax:
    return PythonNumpyToJax()
