"""Isolated execution worker for Python accelerator differential gates.

This module is launched through an :class:`~recast.plugins.executor.Executor`.
It deliberately has a small, canonical-JSON interface: project modules and
backend objects never cross back into the verifier process.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from recast.verify._python_accelerator_protocol import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    ProtocolError,
    canonical_bytes,
    decode_document,
    decode_value,
    encode_value,
)

_TRIALS = 5


def _require_keys(document: dict[str, Any], expected: set[str], label: str) -> None:
    if set(document) != expected:
        raise ProtocolError(f"{label} fields are invalid")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ProtocolError(f"{label} is invalid")
    return value


def _require_functions(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 10_000
        or any(
            not isinstance(name, str) or not name or len(name) > 512 or not name.isidentifier()
            for name in value
        )
        or len(set(value)) != len(value)
    ):
        raise ProtocolError("worker function list is invalid")
    return tuple(value)


def _request_paths(request: dict[str, Any]) -> tuple[Path, Path, str]:
    root = Path(_require_string(request.get("root"), "project root")).resolve()
    module_path = Path(_require_string(request.get("module_path"), "module path")).resolve()
    module_name = _require_string(request.get("module_name"), "module name")
    if not root.is_dir() or not module_path.is_file():
        raise ProtocolError("worker import paths are unavailable")
    parts = module_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ProtocolError("worker module name is invalid")
    return root, module_path, module_name


def _import_file(source: Path, module_name: str, root: Path, *, label: str) -> Any:
    """Load a file under a unique sibling name so package imports still work."""

    package, _, stem = module_name.rpartition(".")
    loaded_name = f"{package + '.' if package else ''}__recast_{label}_{stem}"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(loaded_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create an import spec for {source.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[loaded_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))


def _sample_arguments(
    np: Any, function: Any, name: str, trial: int
) -> tuple[list[Any], dict[str, Any]]:
    signature = inspect.signature(function)
    seed = int.from_bytes(f"{name}:{trial}".encode(), "little") % (2**32)
    rng = np.random.default_rng(seed)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError("variadic exports do not have a deterministic sample contract")
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = _sample_value(np, rng, parameter.name, parameter.annotation)
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
        else:
            args.append(value)
    return args, kwargs


def _sample_value(np: Any, rng: Any, name: str, annotation: Any) -> Any:
    spelled = "" if annotation is inspect.Parameter.empty else str(annotation).lower()
    if annotation is int or spelled in {"int", "<class 'int'>"}:
        return 4
    if annotation is bool or spelled in {"bool", "<class 'bool'>"}:
        return True
    if annotation is float or spelled in {"float", "<class 'float'>"}:
        return float(rng.uniform(0.5, 2.0))
    if name.lower() in {"n", "m", "k", "axis", "size", "count", "steps"}:
        return 4
    return rng.uniform(0.5, 2.0, size=(8,)).astype(np.float64)


def _oracle_response(request: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        request,
        {"schema", "mode", "root", "module_path", "module_name", "functions", "trials"},
        "oracle request",
    )
    if request["trials"] != _TRIALS:
        raise ProtocolError("oracle trial count is invalid")
    root, module_path, module_name = _request_paths(request)
    functions = _require_functions(request["functions"])
    np = importlib.import_module("numpy")
    original = _import_file(module_path, module_name, root, label="oracle")

    rows: list[dict[str, Any]] = []
    for name in functions:
        function = getattr(original, name, None)
        if not callable(function):
            raise RuntimeError(f"export {name!r} is not callable on the oracle side")
        trial_rows: list[dict[str, Any]] = []
        for trial in range(_TRIALS):
            args, kwargs = _sample_arguments(np, function, name, trial)
            input_args = encode_value(np, args)
            input_kwargs = encode_value(np, kwargs)
            reference_args = copy.deepcopy(args)
            reference_kwargs = copy.deepcopy(kwargs)
            result = function(*reference_args, **reference_kwargs)
            trial_rows.append(
                {
                    "input_args": input_args,
                    "input_kwargs": input_kwargs,
                    "return": encode_value(np, result),
                    "post_args": encode_value(np, reference_args),
                    "post_kwargs": encode_value(np, reference_kwargs),
                }
            )
        rows.append({"name": name, "trials": trial_rows})
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": True,
        "mode": "oracle",
        "functions": rows,
    }


def _candidate_calls(value: object, functions: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(functions):
        raise ProtocolError("candidate call set is invalid")
    calls: list[dict[str, Any]] = []
    for expected_name, call in zip(functions, value, strict=True):
        if not isinstance(call, dict):
            raise ProtocolError("candidate call row is invalid")
        _require_keys(call, {"name", "trials"}, "candidate call row")
        if call["name"] != expected_name:
            raise ProtocolError("candidate call ordering is invalid")
        trials = call["trials"]
        if not isinstance(trials, list) or len(trials) != _TRIALS:
            raise ProtocolError("candidate trial set is invalid")
        for trial in trials:
            if not isinstance(trial, dict):
                raise ProtocolError("candidate trial row is invalid")
            _require_keys(trial, {"input_args", "input_kwargs"}, "candidate trial row")
        calls.append(call)
    return calls


def _capture_backend(backend: str) -> tuple[object, object | None]:
    if backend == "numba":
        registry = importlib.import_module("numba.core.registry")
        dispatcher = getattr(registry, "CPUDispatcher", None)
        if not isinstance(dispatcher, type):
            raise RuntimeError("Numba backend identity is unavailable")
        return dispatcher, None
    if backend == "jax":
        jax = importlib.import_module("jax")
        config = getattr(jax, "config", None)
        if config is None or bool(getattr(config, "jax_disable_jit", False)):
            raise RuntimeError("JAX JIT execution is disabled")
        jitted = jax.jit(lambda value: value)
        return type(jitted), config
    raise ProtocolError("candidate backend is invalid")


def _attest_and_compile(
    backend: str,
    function: Any,
    backend_type: object,
    backend_config: object | None,
    args: list[Any],
    kwargs: dict[str, Any],
) -> Any | None:
    if not isinstance(backend_type, type) or not isinstance(function, backend_type):
        if backend == "numba":
            raise RuntimeError("candidate is not a genuine Numba CPU dispatcher")
        raise RuntimeError("candidate is not a genuine JAX jitted function")
    if backend == "numba":
        if not getattr(function, "signatures", ()):
            raise RuntimeError("Numba dispatcher has no compiled signature")
        return None
    if backend_config is None or bool(getattr(backend_config, "jax_disable_jit", False)):
        raise RuntimeError("JAX JIT execution is disabled")
    lower = getattr(function, "lower", None)
    if not callable(lower):
        raise RuntimeError("JAX callable has no lowering interface")
    compiled = lower(*copy.deepcopy(args), **copy.deepcopy(kwargs)).compile()
    if not callable(compiled):
        raise RuntimeError("JAX lowering did not produce a compiled executable")
    return compiled(*copy.deepcopy(args), **copy.deepcopy(kwargs))


def _candidate_response(request: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        request,
        {
            "schema",
            "mode",
            "backend",
            "root",
            "module_path",
            "module_name",
            "functions",
            "trials",
            "calls",
        },
        "candidate request",
    )
    if request["trials"] != _TRIALS:
        raise ProtocolError("candidate trial count is invalid")
    backend = _require_string(request["backend"], "candidate backend")
    root, module_path, module_name = _request_paths(request)
    functions = _require_functions(request["functions"])
    calls = _candidate_calls(request["calls"], functions)
    np = importlib.import_module("numpy")

    # Capture the installed backend's real runtime type before project or
    # candidate code can import a shared helper and mutate backend modules.
    backend_type, backend_config = _capture_backend(backend)
    translated = _import_file(module_path, module_name, root, label="candidate")
    if getattr(translated, "__recast_backend__", None) != backend:
        raise RuntimeError(f"candidate is not marked as backend {backend!r}")

    rows: list[dict[str, Any]] = []
    for name, call in zip(functions, calls, strict=True):
        function = getattr(translated, name, None)
        if not callable(function):
            raise RuntimeError(f"export {name!r} is not callable on the candidate side")
        trial_rows: list[dict[str, Any]] = []
        last_args: list[Any] = []
        last_kwargs: dict[str, Any] = {}
        for trial in call["trials"]:
            args = decode_value(np, trial["input_args"])
            kwargs = decode_value(np, trial["input_kwargs"])
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ProtocolError("candidate inputs have invalid container types")
            pristine_args = copy.deepcopy(args)
            pristine_kwargs = copy.deepcopy(kwargs)
            result = function(*args, **kwargs)
            trial_rows.append(
                {
                    "return": encode_value(np, result),
                    "post_args": encode_value(np, args),
                    "post_kwargs": encode_value(np, kwargs),
                }
            )
            last_args = pristine_args
            last_kwargs = pristine_kwargs
        compiled_result = _attest_and_compile(
            backend,
            function,
            backend_type,
            backend_config,
            last_args,
            last_kwargs,
        )
        rows.append(
            {
                "name": name,
                "trials": trial_rows,
                "compiled_return": (
                    None if backend == "numba" else encode_value(np, compiled_result)
                ),
            }
        )
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": True,
        "mode": "candidate",
        "backend": backend,
        "functions": rows,
    }


def _failure(mode: str, error: BaseException) -> dict[str, Any]:
    reason = str(error).replace("\x00", "?")[:4000]
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": False,
        "mode": mode if mode in {"oracle", "candidate"} else "invalid",
        "reason": reason or type(error).__name__,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        return 2
    request_path = Path(arguments[0])
    response_path = Path(arguments[1])
    mode = "invalid"
    try:
        request = decode_document(request_path.read_bytes())
        mode_value = request.get("mode")
        mode = mode_value if isinstance(mode_value, str) else "invalid"
        if request.get("schema") != REQUEST_SCHEMA:
            raise ProtocolError("worker request schema is invalid")
        if mode == "oracle":
            response = _oracle_response(request)
        elif mode == "candidate":
            response = _candidate_response(request)
        else:
            raise ProtocolError("worker mode is invalid")
    except BaseException as error:
        response = _failure(mode, error)
    try:
        response_path.write_bytes(canonical_bytes(response))
    except BaseException:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
