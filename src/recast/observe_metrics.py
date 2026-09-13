"""Bounded, source-free numerical observations for the read-only event stream.

Verdict metrics are an extension-owned mapping and can contain source, paths,
or arbitrarily large objects. Only this explicit numerical vocabulary crosses
the observer boundary; it is never an acceptance policy or a raw passthrough.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

_SCOPE = "sampled_outputs_and_mutations_including_explicit_compilation"
_DTYPE = re.compile(
    r"(?:bool|u?int(?:8|16|32|64)|float(?:16|32|64|96|128)|complex(?:64|128|192|256)"
    r"|[<>|=]?[biufc](?:1|2|4|8|16|32))\Z"
)
_REASONS = {
    None,
    "not_a_supported_float_pair",
    "exact_discrete_comparison",
    "dtype_mismatch",
    "no_finite_values",
}
_COUNTS = (
    "points",
    "finite_components",
    "matched_nan_components",
    "matched_infinity_components",
    "nonfinite_mismatches",
    "signed_zero_differences",
)


def _count(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError("numerical event count is invalid")
    return value


def _ulp(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"0|[1-9][0-9]{0,19}", value)
        or int(value) >= 2**64
    ):
        raise ValueError("numerical event ULP must be a bounded decimal string or null")
    return value


def _number(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("numerical event error must be finite and nonnegative or null")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("numerical event error exceeds finite range") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("numerical event error must be finite and nonnegative or null")
    return result


def _dtype(value: object) -> str:
    if not isinstance(value, str) or len(value) > 16 or not _DTYPE.fullmatch(value):
        raise ValueError("numerical event dtype is unsupported")
    return value


def _group(raw: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("numerical event dtype group must be a mapping")
    measurement = raw.get("measurement_dtype")
    reason = raw.get("ulp_not_measured_reason")
    if (
        measurement not in (None, "float32", "float64")
        or (reason is not None and not isinstance(reason, str))
        or reason not in _REASONS
    ):
        raise ValueError("numerical event ULP scope is invalid")
    result = {
        "reference_dtype": _dtype(raw.get("reference_dtype")),
        "candidate_dtype": _dtype(raw.get("candidate_dtype")),
        "measurement_dtype": measurement,
        "max_observed_ulp": _ulp(raw.get("max_observed_ulp")),
        "ulp_not_measured_reason": reason,
        **{key: _count(raw.get(key)) for key in _COUNTS},
    }
    return MappingProxyType(result)


def numerical_event_metrics(raw: object) -> Mapping[str, Any] | None:
    """Copy/freeze recognized accuracy observations; never retain plugin data.

    Unrecognized metric families are omitted. A recognized but malformed
    measurement is refused rather than inventing a value or exposing text.
    No NumPy/JAX import is needed by the core or by an observer.
    """

    if not isinstance(raw, Mapping) or raw.get("accuracy") is None:
        return None
    accuracy = raw["accuracy"]
    if (
        not isinstance(accuracy, Mapping)
        or accuracy.get("reference") != "numpy"
        or accuracy.get("scope") != _SCOPE
        or accuracy.get("ulp_bound") is not None
    ):
        raise ValueError("numerical event accuracy vocabulary is unsupported")
    groups = accuracy.get("dtypes")
    if not isinstance(groups, (list, tuple)) or len(groups) > 64:
        raise ValueError("numerical event dtype groups exceed their bound")
    result: dict[str, Any] = {
        "accuracy": MappingProxyType(
            {
                "reference": "numpy",
                "scope": _SCOPE,
                "ulp_bound": None,
                "max_observed_ulp": _ulp(accuracy.get("max_observed_ulp")),
                "dtypes": tuple(_group(group) for group in groups),
            }
        ),
        **{key: _number(raw.get(key)) for key in ("max_abs", "max_rel", "atol", "rtol")},
    }
    execution = raw.get("execution")
    if execution is not None:
        if not isinstance(execution, Mapping):
            raise ValueError("numerical event execution must be a mapping")
        devices = execution.get("devices")
        if (
            execution.get("backend") not in ("jax", "numba")
            or not isinstance(devices, (list, tuple))
            or len(devices) > 8
            or any(device not in ("cpu", "gpu", "tpu") for device in devices)
            or type(execution.get("explicit_compilation")) is not bool
            or (
                execution.get("x64_enabled") is not None
                and type(execution.get("x64_enabled")) is not bool
            )
        ):
            raise ValueError("numerical event execution vocabulary is unsupported")
        result["execution"] = MappingProxyType(
            {
                "backend": execution["backend"],
                "devices": tuple(devices),
                "x64_enabled": execution.get("x64_enabled"),
                "explicit_compilation": execution["explicit_compilation"],
            }
        )
    return MappingProxyType(result)


def numerical_metrics_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return fresh JSON containers from already validated immutable metrics."""

    def copy(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: copy(value) for key, value in item.items()}
        if isinstance(item, tuple):
            return [copy(value) for value in item]
        return item

    return {key: copy(item) for key, item in value.items()}
