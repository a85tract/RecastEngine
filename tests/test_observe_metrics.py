"""Numerical events are bounded observations, never raw verifier metadata."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from recast.observe import RunEvent, RunEventAction, RunEventEntity


def measurement() -> dict[str, Any]:
    return {
        "accuracy": {
            "reference": "numpy",
            "scope": "sampled_outputs_and_mutations_including_explicit_compilation",
            "ulp_bound": None,
            "max_observed_ulp": "9007199254740993",
            "dtypes": [
                {
                    "reference_dtype": "float64",
                    "candidate_dtype": "float64",
                    "measurement_dtype": "float64",
                    "points": 6,
                    "finite_components": 6,
                    "max_observed_ulp": "9007199254740993",
                    "matched_nan_components": 0,
                    "matched_infinity_components": 0,
                    "nonfinite_mismatches": 0,
                    "signed_zero_differences": 0,
                    "ulp_not_measured_reason": None,
                }
            ],
        },
        "max_abs": 1e-14,
        "max_rel": 1e-15,
        "atol": 1e-12,
        "rtol": 1e-12,
        "execution": {
            "backend": "jax",
            "devices": ["cpu"],
            "x64_enabled": True,
            "explicit_compilation": True,
        },
    }


def event(metrics: Any = None, **changes: Any) -> RunEvent:
    fields = {
        "run_id": "run",
        "sequence": 1,
        "emitted_at": "2026-01-01T00:00:00Z",
        "entity": RunEventEntity.VERDICT,
        "action": RunEventAction.FINISHED,
        "recipe": "python-jax",
        "status": "ok",
        "reason_code": "verdict_passed",
        "metrics": metrics,
        **changes,
    }
    return RunEvent(**fields)


def test_unmeasured_events_preserve_exact_legacy_record() -> None:
    legacy = event().to_record()
    assert "metrics" not in legacy
    assert event({}).to_record() == legacy
    assert event({"arbitrary_plugin": {"source": "private"}}).to_record() == legacy


def test_observation_is_deeply_copied_and_immutable() -> None:
    source = measurement()
    expected = copy.deepcopy(source)
    observed = event(source)
    assert observed.to_record()["metrics"] == expected
    source["accuracy"]["dtypes"][0]["max_observed_ulp"] = "0"
    source["execution"]["devices"].append("gpu")
    assert observed.to_record()["metrics"] == expected
    assert observed.metrics is not None
    with pytest.raises(TypeError):
        observed.metrics["accuracy"]["max_observed_ulp"] = "0"
    record = observed.to_record()
    record["metrics"]["accuracy"]["dtypes"][0]["max_observed_ulp"] = "0"
    assert observed.to_record()["metrics"] == expected
    assert '"9007199254740993"' in json.dumps(observed.to_record(), allow_nan=False)


def test_unknown_source_bearing_data_never_crosses_the_numerical_boundary() -> None:
    source = measurement()
    source["source"] = "PRIVATE"
    source["accuracy"]["source"] = "PRIVATE"
    source["accuracy"]["dtypes"][0]["path"] = "PRIVATE"
    source["execution"]["device_description"] = "PRIVATE"
    source["gradient"] = {"function": "PRIVATE", "detail": "PRIVATE"}
    assert event(source).to_record()["metrics"] == measurement()
    assert "PRIVATE" not in json.dumps(event(source).to_record())


@pytest.mark.parametrize(
    "damage",
    [
        "scope",
        "dtype",
        "device",
        "count",
        "ulp_number",
        "ulp_range",
        "nonfinite",
        "groups",
        "flag",
        "reason",
    ],
)
def test_unreviewed_or_unbounded_numerical_values_are_refused(damage: str) -> None:
    source = measurement()
    accuracy = source["accuracy"]
    group = accuracy["dtypes"][0]
    if damage == "scope":
        accuracy["scope"] = "PRIVATE"
    elif damage == "dtype":
        group["reference_dtype"] = "PRIVATE"
    elif damage == "device":
        source["execution"]["devices"] = ["PRIVATE"]
    elif damage == "count":
        group["points"] = 2**53
    elif damage == "ulp_number":
        accuracy["max_observed_ulp"] = 1
    elif damage == "ulp_range":
        accuracy["max_observed_ulp"] = str(2**64)
    elif damage == "nonfinite":
        source["max_abs"] = float("inf")
    elif damage == "groups":
        accuracy["dtypes"] *= 65
    elif damage == "flag":
        source["execution"]["explicit_compilation"] = 1
    else:
        group["ulp_not_measured_reason"] = "PRIVATE"
    with pytest.raises(ValueError):
        event(source)


def test_missing_measurement_is_not_zero_and_native_groups_stay_separate() -> None:
    source = measurement()
    source["accuracy"]["max_observed_ulp"] = None
    source["accuracy"]["dtypes"].append(
        {
            **source["accuracy"]["dtypes"][0],
            "reference_dtype": "float32",
            "candidate_dtype": "float32",
            "measurement_dtype": "float32",
            "max_observed_ulp": "2",
        }
    )
    source["max_abs"] = None
    source["max_rel"] = None
    assert event(source).to_record()["metrics"] == source


def test_only_a_finished_verdict_can_carry_numerical_observations() -> None:
    with pytest.raises(ValueError):
        event(measurement(), entity=RunEventEntity.CANDIDATE)
    with pytest.raises(ValueError):
        event(measurement(), action=RunEventAction.STARTED)
