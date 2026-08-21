"""Cross-cutting rules about how a recipe is put together.

None of these run anything. They are properties of the declaration, which is
the point: a recipe that cannot produce trustworthy evidence should be caught
when it is written, not three hours into a run that was never going to mean
anything.
"""

from __future__ import annotations

from typing import Any

import pytest

from recast.errors import PluginNotFound
from recast.model import Confidence
from recast.registry import KINDS, REGISTRY

# What counts as a hard gate: a verdict that can only be reached by executing
# the candidate against the reference. ``SAMPLED`` is not on this list, and
# that exclusion is the whole rule -- agreement on inputs somebody generated
# is exactly what a plausible wrong answer produces.
_HARD = frozenset(
    {Confidence.TOLERANCED, Confidence.ULP_BOUNDED, Confidence.BIT_EXACT, Confidence.SYMBOLIC}
)

_PROBE_EXECUTOR = "conformance-probe-executor"


def test_the_config_is_one_the_recipe_accepts(recipe_case: Any, build_recipe: Any) -> None:
    """Everything below reads a plan; a plan from a refused config is not one."""
    problems = build_recipe(recipe_case).validate(dict(recipe_case.config))
    assert not problems, (
        f"the declared config for {recipe_case.name!r} does not validate: {problems}"
    )


def test_the_recipe_has_a_gate(recipe_case: Any, build_recipe: Any) -> None:
    stages = build_recipe(recipe_case).stages(dict(recipe_case.config))
    assert any(stage.gate for stage in stages), (
        f"recipe {recipe_case.name!r} gates on nothing, so it can only produce "
        "candidates, never evidence"
    )


def test_no_stage_is_both_a_gate_and_optional(recipe_case: Any, build_recipe: Any) -> None:
    """An optional gate is a suggestion. It downgrades to nothing when absent."""
    for stage in build_recipe(recipe_case).stages(dict(recipe_case.config)):
        assert not (stage.gate and stage.optional), (
            f"{recipe_case.name}: stage {stage.plugin!r} is declared as both"
        )


def test_stage_kinds_are_ones_the_engine_walks(recipe_case: Any, build_recipe: Any) -> None:
    for stage in build_recipe(recipe_case).stages(dict(recipe_case.config)):
        assert stage.kind in KINDS, f"{recipe_case.name}: unknown stage kind {stage.kind!r}"


def test_the_plan_is_reproducible(recipe_case: Any, build_recipe: Any) -> None:
    """Same config, same stages -- or the Evidence names a run nobody can repeat."""
    config = dict(recipe_case.config)
    first = [
        (s.kind, s.plugin, s.gate, s.optional) for s in build_recipe(recipe_case).stages(config)
    ]
    second = [
        (s.kind, s.plugin, s.gate, s.optional) for s in build_recipe(recipe_case).stages(config)
    ]
    assert first == second


def test_it_declares_at_most_one_transform(recipe_case: Any, build_recipe: Any) -> None:
    """A Unit has one Candidate, so a second transform stage does not compose --
    it replaces, and what it replaces includes the ``deferred`` list the second
    stage would have been there to consume. Composition happens inside a
    Transform: rules first, an ``AgentProvider`` for what they refused, one
    Candidate out."""
    transforms = [
        stage.plugin
        for stage in build_recipe(recipe_case).stages(dict(recipe_case.config))
        if stage.kind == "transform"
    ]
    assert len(transforms) <= 1, (
        f"{recipe_case.name!r} declares {len(transforms)} transform stages "
        f"({transforms}); all but the last would be discarded"
    )


def test_it_declares_the_executor_it_needs(recipe_case: Any, build_recipe: Any) -> None:
    """An Oracle or a Verifier is handed an executor, so the recipe must name one.

    First in the list, because it is not a step: it has to be resolved before
    anything it is handed to runs.
    """
    stages = build_recipe(recipe_case).stages(dict(recipe_case.config))
    if not {stage.kind for stage in stages} & {"oracle", "verifier"}:
        pytest.skip(f"{recipe_case.name!r} has neither an oracle nor a verifier stage")

    executors = [stage for stage in stages if stage.kind == "executor"]
    assert len(executors) == 1, f"{recipe_case.name!r} declares {len(executors)} executor stages"
    assert stages[0].kind == "executor", (
        f"{recipe_case.name!r} declares its executor at position "
        f"{stages.index(executors[0])}, after work that is handed one"
    )


def test_the_executor_comes_from_config(recipe_case: Any, build_recipe: Any) -> None:
    """A real executor's name (``pbs-<site>``) is site knowledge and a leak."""
    stages = build_recipe(recipe_case).stages(dict(recipe_case.config))
    if not any(stage.kind == "executor" for stage in stages):
        pytest.skip(f"{recipe_case.name!r} declares no executor stage")

    asked = {**dict(recipe_case.config), "executor": _PROBE_EXECUTOR}
    replanned = build_recipe(recipe_case).stages(asked)
    named = [stage.plugin for stage in replanned if stage.kind == "executor"]
    assert named == [_PROBE_EXECUTOR], (
        f"{recipe_case.name!r} planned executor {named} for a config asking for "
        f"{_PROBE_EXECUTOR!r}; the name is hardcoded"
    )


def test_an_agentic_transform_is_under_a_hard_gate(recipe_case: Any, build_recipe: Any) -> None:
    """A model emits plausible output for exactly the cases the rules refused.

    Only running it against the oracle separates that from a correct one, so a
    ``deterministic = False`` Transform needs a gate that awards ``BIT_EXACT``
    or an explicit tolerance. A transform this process cannot resolve is not
    judged -- an uninstalled plugin's determinism is not knowable from here,
    and the suite says which ones it skipped rather than assuming.
    """
    stages = build_recipe(recipe_case).stages(dict(recipe_case.config))
    agentic = [
        stage.plugin
        for stage in stages
        if stage.kind == "transform" and _deterministic(stage.plugin) is False
    ]
    if not agentic:
        return

    strengths = [
        _provides(stage.plugin) for stage in stages if stage.kind == "verifier" and stage.gate
    ]
    assert any(strength in _HARD for strength in strengths), (
        f"{recipe_case.name!r} runs agentic transform(s) {agentic} but its gates award "
        f"{[s.value if s else 'unresolvable' for s in strengths]}; none of those catch a "
        "plausible wrong answer"
    )


def _deterministic(plugin: str) -> bool | None:
    """``None`` when the transform is not installed here, so nothing is claimed."""
    try:
        transform = REGISTRY.get("transform", plugin)()
    except PluginNotFound:
        return None
    determinism: bool = transform.deterministic
    return determinism


def _provides(plugin: str) -> Confidence | None:
    try:
        verifier = REGISTRY.get("verifier", plugin)()
    except PluginNotFound:
        return None
    strength: Confidence = verifier.provides
    return strength
