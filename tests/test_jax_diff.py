"""Tests for the JAX migration harness itself.

An instrument that cannot be seen to fail is not evidence, and this one will
be the only thing standing between a 600-line AST migration and a silent
behaviour change. So each check below plants exactly one difference and
requires it to be reported -- and one plants a difference that must *not* be,
because a harness that flags a reworded diagnostic will be switched off by the
third person who hits it.

No collection and no corpus needed: what is under test is the comparison, and
the emitters are two dictionaries.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_spec = importlib.util.spec_from_file_location(
    "jax_diff", Path(__file__).resolve().parent.parent / "tools" / "jax_diff.py"
)
assert _spec is not None and _spec.loader is not None
jax_diff = importlib.util.module_from_spec(_spec)
sys.modules["jax_diff"] = jax_diff
_spec.loader.exec_module(jax_diff)


def emission(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "pieces": ["def k(x):\n    return x + 1.0\n"],
        "jitted": ["k"],
        "delegated": {"host": "[emit] unsupported stmt Raise: in k"},
    }
    return jax_diff.Emission(**{**base, **kw})


def test_identical_emitters_disagree_about_nothing() -> None:
    assert jax_diff.differences("m", emission(), emission()) == []


def test_a_changed_byte_is_reported_with_the_line() -> None:
    """The failure this exists for: a reflowed parenthesis is indistinguishable
    from a wrong number until run time."""
    other = emission(pieces=["def k(x):\n    return x + 1.00\n"])
    problems = jax_diff.differences("m", emission(), other)
    assert len(problems) == 1
    assert "piece 0 differs" in problems[0]
    assert "line 2" in problems[0]
    assert "1.00" in problems[0]


def test_a_kernel_that_stopped_being_emitted_is_reported() -> None:
    problems = jax_diff.differences("m", emission(), emission(jitted=[]))
    assert any("kernels differ" in p and "'k'" in p for p in problems)


def test_a_subprogram_that_changed_side_is_reported() -> None:
    """Host-delegation is a decision, and a migration that moves one has
    changed what the backend does even if every emitted byte still matches."""
    problems = jax_diff.differences("m", emission(), emission(delegated={}))
    assert any("host-delegation differs" in p for p in problems)


def test_a_different_reason_category_is_reported() -> None:
    other = emission(delegated={"host": "[elig] derived type"})
    problems = jax_diff.differences("m", emission(), other)
    assert any("delegated for" in p for p in problems)


def test_a_reworded_diagnostic_is_not() -> None:
    """Same decision, same category, different tail. The tail is a diagnostic
    and the two sides are entitled to word one differently."""
    other = emission(delegated={"host": "[emit] unsupported stmt Raise: inside kernel k"})
    assert jax_diff.differences("m", emission(), other) == []


def test_refusing_the_same_way_is_agreement_and_refusing_differently_is_not() -> None:
    """Which modules an emitter refuses is part of what is being compared."""
    boom = jax_diff.Emission(error="ValueError: no")
    assert jax_diff.differences("m", boom, jax_diff.Emission(error="ValueError: no")) == []
    problems = jax_diff.differences("m", boom, jax_diff.Emission(error="KeyError: other"))
    assert any("refused differently" in p for p in problems)


def test_a_missing_piece_stops_at_the_count() -> None:
    """Reporting a hundred line differences after a count mismatch buries the
    one fact that explains them."""
    problems = jax_diff.differences("m", emission(), emission(pieces=[]))
    assert problems == ["m: 1 emitted piece(s) vs 0"]


def test_the_category_is_the_part_before_the_colon() -> None:
    assert (
        jax_diff.category("[emit] unsupported stmt Raise: in k") == "[emit] unsupported stmt Raise"
    )
    assert jax_diff.category("[elig] derived type") == "[elig] derived type"


def test_the_migrated_backend_is_named_before_it_exists() -> None:
    """The harness expects ``recast.transform.jax.backend.build_module``. That
    expectation is the migration's specification, so it fails loudly and says
    what is missing rather than skipping."""
    with pytest.raises(SystemExit, match="not written yet"):
        jax_diff.load_migrated()
