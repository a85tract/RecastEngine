"""Contract checks for declarative TranslationEngine manifests."""

from __future__ import annotations

from typing import Any

from recast.engines import TranslationEngine
from recast.registry import REGISTRY


def _default_recipe(engine_case: Any, engine: TranslationEngine) -> Any:
    if engine_case.build_recipe is not None:
        return engine_case.build_recipe()
    return REGISTRY.get("recipe", engine.default_recipe)()


def test_the_factory_returns_the_named_engine(engine_case: Any, build_engine: Any) -> None:
    engine = build_engine(engine_case)
    assert isinstance(engine, TranslationEngine)
    assert engine.id == engine_case.name
    assert engine.name == engine.id


def test_the_manifest_digest_is_reproducible(engine_case: Any, build_engine: Any) -> None:
    first = build_engine(engine_case)
    second = build_engine(engine_case)
    assert first.to_dict() == second.to_dict()
    assert first.digest() == second.digest()
    assert first.to_dict()["digest"] == first.digest()


def test_the_default_recipe_names_this_engine(engine_case: Any, build_engine: Any) -> None:
    engine = build_engine(engine_case)
    recipe = _default_recipe(engine_case, engine)
    config = dict(engine.default_config)
    assert recipe.validate(config) == []
    assert recipe.resolved_engine_id(config) == engine.id


def test_required_gates_are_in_the_default_plan(engine_case: Any, build_engine: Any) -> None:
    engine = build_engine(engine_case)
    recipe = _default_recipe(engine_case, engine)
    stages = recipe.stages(dict(engine.default_config))
    gates = {stage.plugin for stage in stages if stage.gate}
    assert set(engine.required_gates) <= gates


def test_translation_changes_the_artifact_contract(engine_case: Any, build_engine: Any) -> None:
    engine = build_engine(engine_case)
    assert engine.input_artifact_contract != engine.output_artifact_contract
