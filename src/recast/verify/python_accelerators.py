"""Independent numerical gates for Python/NumPy accelerator candidates."""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from recast.model import Candidate, Confidence, OracleRef, Unit, Verdict
from recast.plugins.executor import Executor, Job
from recast.plugins.verifier import Verifier
from recast.verify._python_accelerator_protocol import (
    MAX_DOCUMENT_BYTES,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    ProtocolError,
    canonical_bytes,
    decode_document,
    decode_value,
)

__all__ = [
    "PythonJaxDifferentialVerifier",
    "PythonNumbaDifferentialVerifier",
    "jax_factory",
    "numba_factory",
]

_TRIALS = 5
_RTOL = 1e-12
_ATOL = 1e-12
_WORKER_ENV_NAMES = frozenset(
    {
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
    }
)
_WORKER_ENV_PREFIXES = (
    "BLAS_",
    "CUDA_",
    "JAX_",
    "LAPACK_",
    "MKL_",
    "NUMBA_",
    "NVIDIA_",
    "OMP_",
    "OPENBLAS_",
    "ROCR_",
    "XLA_",
)


class _WorkerFailure(RuntimeError):
    """An isolated worker returned a controlled, fail-closed result."""


class _PythonDifferentialVerifier(Verifier):
    """Compare two separately isolated executions on deterministic samples."""

    dependency: str
    suffix: str
    backend: str
    provides = Confidence.TOLERANCED

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        del unit
        if importlib.util.find_spec("numpy") is None:
            return self._failed(candidate, "numpy is not installed; install recast-engine[verify]")
        if importlib.util.find_spec(self.dependency) is None:
            return self._failed(
                candidate,
                f"{self.dependency} is not installed; this backend cannot be verified",
            )
        if candidate.deferred:
            return self._failed(
                candidate,
                f"candidate has {len(candidate.deferred)} deferred function(s); "
                "a partial accelerator port is not a numerical pass",
            )

        try:
            root, source, original_module, functions = _oracle_inputs(oracle, config)
            workspace.mkdir(parents=True, exist_ok=True)
            job_root = Path(
                tempfile.mkdtemp(
                    prefix=f"python-{self.backend}-{candidate.digest()[:12]}-",
                    dir=workspace,
                )
            )
            candidate_path, candidate_module = _stage_candidate(
                candidate,
                job_root / "candidate",
                self.suffix,
            )
        except Exception as error:
            return self._failed(candidate, f"accelerator verifier input is invalid: {error}")

        oracle_request = {
            "schema": REQUEST_SCHEMA,
            "mode": "oracle",
            "root": str(root),
            "module_path": str(source),
            "module_name": original_module,
            "functions": list(functions),
            "trials": _TRIALS,
        }
        try:
            oracle_response = _run_worker(
                executor,
                oracle_request,
                job_root / "oracle",
                label=f"python-{self.backend}-oracle",
            )
            oracle_rows = _oracle_rows(oracle_response, functions)
        except Exception as error:
            return self._failed(candidate, f"isolated Python oracle could not run: {error}")

        calls = [
            {
                "name": row["name"],
                "trials": [
                    {
                        "input_args": trial["input_args"],
                        "input_kwargs": trial["input_kwargs"],
                    }
                    for trial in row["trials"]
                ],
            }
            for row in oracle_rows
        ]
        candidate_request = {
            "schema": REQUEST_SCHEMA,
            "mode": "candidate",
            "backend": self.backend,
            "root": str(root),
            "module_path": str(candidate_path),
            "module_name": candidate_module,
            "functions": list(functions),
            "trials": _TRIALS,
            "calls": calls,
        }
        try:
            candidate_response = _run_worker(
                executor,
                candidate_request,
                job_root / "candidate-run",
                label=f"python-{self.backend}-candidate",
            )
            candidate_rows = _candidate_rows(candidate_response, functions, self.backend)
        except Exception as error:
            return self._failed(
                candidate,
                f"candidate could not prove an isolated real {self.backend} execution: {error}",
            )

        try:
            np = importlib.import_module("numpy")
            return self._compare_rows(candidate, np, oracle_rows, candidate_rows)
        except Exception as error:
            return self._failed(
                candidate, f"isolated accelerator observations are invalid: {error}"
            )

    def _compare_rows(
        self,
        candidate: Candidate,
        np: Any,
        oracle_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
    ) -> Verdict:
        comparisons = 0
        checked_points = 0
        exact = 0
        max_abs = 0.0
        max_rel = 0.0
        compiled: list[str] = []

        for oracle_row, candidate_row in zip(oracle_rows, candidate_rows, strict=True):
            name = str(oracle_row["name"])
            last_expected: Any = None
            for trial_number, (reference, translated) in enumerate(
                zip(oracle_row["trials"], candidate_row["trials"], strict=True)
            ):
                baseline_args = decode_value(np, reference["input_args"])
                baseline_kwargs = decode_value(np, reference["input_kwargs"])
                reference_args = decode_value(np, reference["post_args"])
                reference_kwargs = decode_value(np, reference["post_kwargs"])
                candidate_args = decode_value(np, translated["post_args"])
                candidate_kwargs = decode_value(np, translated["post_kwargs"])
                expected = decode_value(np, reference["return"])
                actual = decode_value(np, translated["return"])
                if not all(
                    (
                        isinstance(baseline_args, list),
                        isinstance(baseline_kwargs, dict),
                        isinstance(reference_args, list),
                        isinstance(reference_kwargs, dict),
                        isinstance(candidate_args, list),
                        isinstance(candidate_kwargs, dict),
                    )
                ):
                    raise ProtocolError("worker argument observations have invalid containers")
                if set(reference_kwargs) != set(candidate_kwargs):
                    raise ProtocolError("post-call keyword argument sets differ")

                leaves = [
                    ("return", expected, actual, None),
                    *(
                        (f"arg[{index}]", left, right, baseline_args[index])
                        for index, (left, right) in enumerate(
                            zip(reference_args, candidate_args, strict=True)
                        )
                    ),
                    *(
                        (
                            f"kwarg[{key}]",
                            reference_kwargs[key],
                            candidate_kwargs[key],
                            baseline_kwargs[key],
                        )
                        for key in sorted(reference_kwargs)
                    ),
                ]
                for label, expected_value, actual_value, baseline_value in leaves:
                    result = _compare(
                        np,
                        expected_value,
                        actual_value,
                        rtol=_RTOL,
                        atol=_ATOL,
                    )
                    if result is None:
                        continue
                    points, same, absolute, relative, discrete_mismatch = result
                    checked_points += points
                    meaningful = (
                        label == "return"
                        or _observable_changed(np, baseline_value, expected_value)
                        or _observable_changed(np, baseline_value, actual_value)
                    )
                    if meaningful:
                        comparisons += points
                        exact += same
                    max_abs = max(max_abs, absolute)
                    max_rel = max(max_rel, relative)
                    if discrete_mismatch or (absolute > _ATOL and relative > _RTOL):
                        comparison = "differs exactly" if discrete_mismatch else "differs"
                        return self._failed(
                            candidate,
                            f"{name} trial {trial_number} {label} {comparison}: "
                            f"max_abs={absolute:.3e}, max_rel={relative:.3e}",
                            metrics={
                                "points": comparisons,
                                "checked_points": checked_points,
                                "max_abs": max_abs,
                                "max_rel": max_rel,
                            },
                        )
                last_expected = expected

            compiled_document = candidate_row["compiled_return"]
            if compiled_document is not None:
                compiled_actual = decode_value(np, compiled_document)
                compiled_result = _compare(
                    np,
                    last_expected,
                    compiled_actual,
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                if compiled_result is not None:
                    _points, _same, absolute, relative, discrete_mismatch = compiled_result
                    if discrete_mismatch or (absolute > _ATOL and relative > _RTOL):
                        comparison = "differs exactly" if discrete_mismatch else "differs"
                        return self._failed(
                            candidate,
                            f"{name!r} explicitly compiled output {comparison}: "
                            f"max_abs={absolute:.3e}, max_rel={relative:.3e}",
                        )
            compiled.append(name)

        if comparisons == 0:
            return self._failed(candidate, "nothing numerical was compared; that is not a pass")
        confidence = Confidence.BIT_EXACT if exact == comparisons else Confidence.TOLERANCED
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics={
                "functions": compiled,
                "trials": _TRIALS,
                "points": comparisons,
                "checked_points": checked_points,
                "bit_exact": exact,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "rtol": _RTOL,
                "atol": _ATOL,
                "backend": self.backend,
            },
            detail=(
                f"{len(compiled)} function(s), {_TRIALS} trial(s) each, "
                f"{comparisons} numerical point(s) compared on an isolated real "
                f"{self.backend} backend"
            ),
        )

    def _failed(
        self, candidate: Candidate, detail: str, *, metrics: dict[str, Any] | None = None
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=Confidence.FAILED,
            metrics=metrics or {},
            detail=detail,
        )


class PythonNumbaDifferentialVerifier(_PythonDifferentialVerifier):
    name = "differential.python-numba"
    dependency = "numba"
    suffix = "_numba.py"
    backend = "numba"


class PythonJaxDifferentialVerifier(_PythonDifferentialVerifier):
    name = "differential.python-jax"
    dependency = "jax"
    suffix = "_jax.py"
    backend = "jax"


def _oracle_inputs(
    oracle: OracleRef, config: dict[str, Any]
) -> tuple[Path, Path, str, tuple[str, ...]]:
    handle = oracle.handle
    if not isinstance(handle, dict) or set(handle) != {
        "root",
        "source",
        "module_name",
        "functions",
    }:
        raise ValueError("python-source oracle did not provide an inert source handle")
    root_name = handle["root"]
    if not isinstance(root_name, str) or not root_name:
        raise ValueError("python-source oracle root is invalid")
    root = Path(root_name)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("python-source oracle root is unavailable or non-canonical")
    configured_root = config.get("root")
    if configured_root is not None and Path(configured_root).resolve() != root:
        raise ValueError("python-source oracle root differs from verifier configuration")
    source_name = handle["source"]
    module_name = handle["module_name"]
    raw_functions = handle["functions"]
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("python-source oracle source is invalid")
    relative = Path(source_name)
    if relative.is_absolute():
        raise ValueError("python-source oracle source must be relative")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("python-source oracle source escapes the project root") from error
    if not source.is_file():
        raise ValueError("python-source oracle source is unavailable")
    if (
        not isinstance(module_name, str)
        or not module_name
        or any(not part.isidentifier() for part in module_name.split("."))
    ):
        raise ValueError("python-source oracle module name is invalid")
    if (
        not isinstance(raw_functions, (tuple, list))
        or not raw_functions
        or any(not isinstance(name, str) or not name.isidentifier() for name in raw_functions)
        or len(set(raw_functions)) != len(raw_functions)
    ):
        raise ValueError("python-source oracle exports are invalid")
    return root, source, module_name, tuple(raw_functions)


def _stage_candidate(candidate: Candidate, staged: Path, suffix: str) -> tuple[Path, str]:
    matches = [path for path in candidate.files if str(path).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"candidate must contain exactly one {suffix} module")
    staged.mkdir(parents=True, exist_ok=False)
    for raw_path, content in candidate.files.items():
        if not isinstance(raw_path, Path) or raw_path.is_absolute() or ".." in raw_path.parts:
            raise ValueError("candidate contains an unsafe output path")
        if not isinstance(content, bytes):
            raise ValueError("candidate file content is not bytes")
        target = (staged / raw_path).resolve()
        try:
            target.relative_to(staged.resolve())
        except ValueError as error:
            raise ValueError("candidate output escapes its staging directory") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    relative = matches[0]
    module_parts = relative.with_suffix("").parts
    if not module_parts or any(not part.isidentifier() for part in module_parts):
        raise ValueError("candidate module path is not importable")
    module_path = (staged / relative).resolve()
    if not module_path.is_file():
        raise ValueError("candidate module was not staged")
    return module_path, ".".join(module_parts)


def _run_worker(
    executor: Executor,
    request: dict[str, Any],
    directory: Path,
    *,
    label: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    request_path = directory / "request.json"
    response_path = directory / "response.json"
    request_path.write_bytes(canonical_bytes(request))
    job = Job(
        argv=(
            sys.executable,
            "-I",
            "-m",
            "recast.verify.python_accelerators_worker",
            str(request_path),
            str(response_path),
        ),
        cwd=directory,
        env=_worker_environment(),
        timeout_s=120.0,
        label=label,
    )
    try:
        result = executor.run(job)
    except Exception as error:
        raise _WorkerFailure(f"executor refused worker job: {error}") from error
    if not result.ok:
        raise _WorkerFailure(f"worker job exited with status {result.returncode}")
    if response_path.is_symlink() or not response_path.is_file():
        raise ProtocolError("worker produced no regular response")
    size = response_path.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise ProtocolError("worker response exceeds the protocol bound")
    response = decode_document(response_path.read_bytes())
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ProtocolError("worker response schema is invalid")
    ok = response.get("ok")
    if type(ok) is not bool:
        raise ProtocolError("worker response status is invalid")
    if ok is False:
        if set(response) != {"schema", "ok", "mode", "reason"}:
            raise ProtocolError("worker failure response fields are invalid")
        reason = response["reason"]
        if not isinstance(reason, str) or not reason or len(reason) > 4000:
            raise ProtocolError("worker failure reason is invalid")
        raise _WorkerFailure(reason)
    return response


def _worker_environment() -> dict[str, str]:
    """Preserve backend/runtime controls without exposing arbitrary secrets.

    ``-I`` already ignores Python path/startup variables.  The explicit
    allowlist additionally keeps project code from receiving unrelated tokens
    while retaining accelerator, numerical-library and dynamic-loader controls
    that define the environment being attested.
    """

    return {
        name: value
        for name, value in os.environ.items()
        if name in _WORKER_ENV_NAMES or name.startswith(_WORKER_ENV_PREFIXES)
    }


def _oracle_rows(response: dict[str, Any], functions: tuple[str, ...]) -> list[dict[str, Any]]:
    if set(response) != {"schema", "ok", "mode", "functions"}:
        raise ProtocolError("oracle response fields are invalid")
    if response["mode"] != "oracle":
        raise ProtocolError("oracle response mode is invalid")
    rows = response["functions"]
    if not isinstance(rows, list) or len(rows) != len(functions):
        raise ProtocolError("oracle response function set is invalid")
    for expected_name, row in zip(functions, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"name", "trials"}:
            raise ProtocolError("oracle response function row is invalid")
        if row["name"] != expected_name:
            raise ProtocolError("oracle response function ordering is invalid")
        trials = row["trials"]
        if not isinstance(trials, list) or len(trials) != _TRIALS:
            raise ProtocolError("oracle response trial set is invalid")
        for trial in trials:
            if not isinstance(trial, dict) or set(trial) != {
                "input_args",
                "input_kwargs",
                "return",
                "post_args",
                "post_kwargs",
            }:
                raise ProtocolError("oracle response trial row is invalid")
    return rows


def _candidate_rows(
    response: dict[str, Any], functions: tuple[str, ...], backend: str
) -> list[dict[str, Any]]:
    if set(response) != {"schema", "ok", "mode", "backend", "functions"}:
        raise ProtocolError("candidate response fields are invalid")
    if response["mode"] != "candidate" or response["backend"] != backend:
        raise ProtocolError("candidate response identity is invalid")
    rows = response["functions"]
    if not isinstance(rows, list) or len(rows) != len(functions):
        raise ProtocolError("candidate response function set is invalid")
    for expected_name, row in zip(functions, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "name",
            "trials",
            "compiled_return",
        }:
            raise ProtocolError("candidate response function row is invalid")
        if row["name"] != expected_name:
            raise ProtocolError("candidate response function ordering is invalid")
        if (backend == "numba") != (row["compiled_return"] is None):
            raise ProtocolError("candidate compiled response is invalid")
        trials = row["trials"]
        if not isinstance(trials, list) or len(trials) != _TRIALS:
            raise ProtocolError("candidate response trial set is invalid")
        for trial in trials:
            if not isinstance(trial, dict) or set(trial) != {
                "return",
                "post_args",
                "post_kwargs",
            }:
                raise ProtocolError("candidate response trial row is invalid")
    return rows


def _compare(
    np: Any, expected: Any, actual: Any, *, rtol: float, atol: float
) -> tuple[int, int, float, float, bool] | None:
    if expected is None and actual is None:
        return None
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
            raise ValueError("result container shape differs")
        combined = [
            _compare(np, left, right, rtol=rtol, atol=atol)
            for left, right in zip(expected, actual, strict=True)
        ]
        rows = [row for row in combined if row is not None]
        if not rows:
            return None
        return (
            sum(row[0] for row in rows),
            sum(row[1] for row in rows),
            max(row[2] for row in rows),
            max(row[3] for row in rows),
            any(row[4] for row in rows),
        )
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape:
        raise ValueError(f"result shape differs: {left.shape} != {right.shape}")
    if left.dtype.kind not in "buifc" or right.dtype.kind not in "buifc":
        if np.array_equal(left, right):
            return None
        raise ValueError("non-numerical result differs")
    points = int(left.size)
    if points == 0:
        # An empty observation proves no numerical behavior.  Returning zero
        # lets the caller's non-vacuity check reject a run made only of empty
        # arrays instead of manufacturing one point with ``size or 1``.
        return 0, 0, 0.0, 0.0, False

    left_discrete = left.dtype.kind in "bui"
    right_discrete = right.dtype.kind in "bui"
    if left_discrete or right_discrete:
        # Integer and logical disagreements are semantic disagreements, not
        # floating-point error.  In particular, converting int64 values above
        # 2**53 to complex128 aliases adjacent integers and can make a real
        # mismatch look like zero error.  Compare discrete values before any
        # floating conversion and carry an explicit flag past the tolerance
        # decision.  Bool and integer are distinct value categories; signed
        # and unsigned integers share exact integer-value comparison.
        left_category = "bool" if left.dtype.kind == "b" else "integer"
        right_category = "bool" if right.dtype.kind == "b" else "integer"
        same_category = left_discrete and right_discrete and left_category == right_category
        equal = np.equal(left, right) if same_category else np.zeros(left.shape, dtype=bool)
        same = int(np.count_nonzero(equal))
        mismatch = same != points
        if mismatch and same_category:
            differences = [
                abs(int(left_value) - int(right_value))
                for left_value, right_value in zip(
                    left.reshape(-1).tolist(), right.reshape(-1).tolist(), strict=True
                )
            ]
            max_difference = max(differences)
            max_abs = float(max_difference)
            max_rel = max(
                difference / max(abs(int(left_value)), 1)
                for difference, left_value in zip(
                    differences, left.reshape(-1).tolist(), strict=True
                )
            )
        elif mismatch:
            # A bool/integer or discrete/floating category change is itself an
            # exact mismatch even when the displayed scalar values coincide.
            max_abs = 1.0
            max_rel = 1.0
        else:
            max_abs = 0.0
            max_rel = 0.0
        return points, same, max_abs, float(max_rel), mismatch

    both_nan = np.isnan(left) & np.isnan(right)
    equal = np.equal(left, right) | both_nan
    raw_difference = np.abs(left.astype(np.complex128) - right.astype(np.complex128))
    difference = np.where(both_nan, 0.0, raw_difference)
    raw_magnitude = np.maximum(np.abs(left.astype(np.complex128)), np.finfo(np.float64).tiny)
    magnitude = np.where(both_nan, 1.0, raw_magnitude)
    max_abs = float(np.max(difference, initial=0.0))
    max_rel = float(np.max(difference / magnitude, initial=0.0))
    if not math.isfinite(max_abs) or not math.isfinite(max_rel):
        raise ValueError("comparison produced a non-finite error")
    return points, int(np.count_nonzero(equal)), max_abs, max_rel, False


def _observable_changed(np: Any, before: Any, after: Any) -> bool:
    """Whether an input argument became a meaningful mutated observation."""

    if before is None and after is None:
        return False
    if isinstance(before, (tuple, list)):
        if not isinstance(after, (tuple, list)) or len(before) != len(after):
            return True
        return any(
            _observable_changed(np, left, right) for left, right in zip(before, after, strict=True)
        )
    try:
        left = np.asarray(before)
        right = np.asarray(after)
        if left.shape != right.shape:
            return True
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            return not bool(np.array_equal(left, right, equal_nan=True))
        return not bool(np.array_equal(left, right))
    except Exception:
        return True


def numba_factory(**_config: Any) -> PythonNumbaDifferentialVerifier:
    return PythonNumbaDifferentialVerifier()


def jax_factory(**_config: Any) -> PythonJaxDifferentialVerifier:
    return PythonJaxDifferentialVerifier()
