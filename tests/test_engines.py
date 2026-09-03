"""Translation engine manifests are immutable, canonical catalog metadata."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from recast.cli import main
from recast.engines import (
    ArtifactContract,
    TranslationEngine,
    canonical_digest,
    catalog_document,
    engines,
    fortran_numpy_engine,
    python_jax_engine,
    python_numba_engine,
)
from recast.errors import PluginError
from recast.recipes import TranslateRecipe
from recast.registry import Registry

_DIGEST = "sha256:" + "0" * 64


def _engine(name: str = "example.a-b") -> TranslationEngine:
    return TranslationEngine(
        id=name,
        version="1",
        implementation_digest=_DIGEST,
        default_recipe="translate",
        input_artifact_contract=ArtifactContract(
            id="example.source.a",
            version="1",
            media_type="application/vnd.example.source",
            language="a",
        ),
        output_artifact_contract=ArtifactContract(
            id="example.source.b",
            version="1",
            media_type="application/vnd.example.source",
            language="b",
        ),
        config_schema={"type": "object", "properties": {"mode": {"type": "string"}}},
        default_config={"mode": "safe"},
        required_gates=("example.gate",),
        capabilities=("translation", "deterministic"),
        owning_repository="https://example.invalid/engine",
    )


def _registry(*manifests: TranslationEngine) -> Registry:
    registry = Registry()
    registry._loaded.add("engine")
    for manifest in manifests:
        registry.register("engine", manifest.id, lambda manifest=manifest: manifest)
    return registry


def test_manifest_digest_is_canonical_and_covers_semantics() -> None:
    first = _engine()
    reordered = replace(
        first,
        config_schema={"properties": {"mode": {"type": "string"}}, "type": "object"},
    )
    changed = replace(first, required_gates=("example.stronger-gate",))

    assert first.digest() == reordered.digest()
    assert first.digest() != changed.digest()
    assert hash(first) == hash(reordered)
    assert first.digest().startswith("sha256:")
    assert first.config_schema_digest == reordered.config_schema_digest
    assert first.input_artifact_contract.to_dict()["digest"].startswith("sha256:")


def test_nested_config_documents_are_copied_and_frozen() -> None:
    schema = {"type": "object", "properties": {"mode": {"type": "string"}}}
    manifest = replace(_engine(), config_schema=schema)
    before = manifest.digest()
    schema["properties"] = {}

    assert manifest.digest() == before
    with pytest.raises(TypeError):
        manifest.config_schema["type"] = "array"  # type: ignore[index]
    properties = manifest.config_schema["properties"]
    assert isinstance(properties, dict) is False
    with pytest.raises(TypeError):
        properties["new"] = {}  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "Not Namespaced", "lowercase dotted identifier"),
        ("implementation_digest", "latest", "sha256"),
        ("owning_repository", "", "owning_repository"),
        ("required_gates", (), "required_gates"),
    ],
)
def test_invalid_manifests_fail_at_registration_boundary(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(PluginError, match=message):
        replace(_engine(), **{field: value})


def test_catalog_is_ordered_and_has_an_envelope_digest() -> None:
    second = replace(_engine("example.z-a"), version="2")
    registry = _registry(second, _engine())
    document = catalog_document(registry=registry)

    listed = engines(registry=registry)
    assert [manifest.id for manifest in listed] == ["example.a-b", "example.z-a"]
    assert [item["id"] for item in document["engines"]] == [  # type: ignore[index]
        "example.a-b",
        "example.z-a",
    ]
    payload = {"schema": document["schema"], "engines": document["engines"]}
    assert document["digest"] == canonical_digest(payload)


def test_entry_point_address_must_equal_manifest_identity() -> None:
    registry = Registry()
    registry._loaded.add("engine")
    registry.register("engine", "example.wrong", lambda: _engine())
    with pytest.raises(PluginError, match="must match"):
        engines(registry=registry)


def test_in_process_registration_accepts_an_immutable_manifest() -> None:
    manifest = _engine()
    registry = Registry()
    registry._loaded.add("engine")
    registry.register("engine", manifest.id, manifest)
    assert engines(registry=registry) == (manifest,)


def test_builtin_describes_only_the_numpy_variant() -> None:
    manifest = fortran_numpy_engine()
    recipe = TranslateRecipe()

    assert manifest.id == "recast.fortran-python.numpy"
    assert manifest.input_artifact_contract.language == "fortran"
    assert manifest.output_artifact_contract.language == "python"
    assert manifest.output_artifact_contract.profile == "numpy"
    assert manifest.config_schema["additionalProperties"] is False
    assert manifest.config_schema["properties"]["executor"]["const"] == "local"  # type: ignore[index]
    assert recipe.resolved_engine_id({}) == manifest.id
    assert recipe.resolved_engine_id({"target": "numba"}) is None
    assert recipe.resolved_engine_id({"target": "cuda"}) is None
    assert recipe.resolved_engine_id({"target": "numpy", "frontend": "extension"}) is None


def test_python_accelerator_manifests_share_only_the_numpy_input_contract() -> None:
    numba = python_numba_engine()
    jax = python_jax_engine()

    assert numba.input_artifact_contract == jax.input_artifact_contract
    assert numba.input_artifact_contract.id == "recast.source-tree.python.numpy"
    assert numba.output_artifact_contract != jax.output_artifact_contract
    assert numba.default_recipe == "python-to-numba"
    assert jax.default_recipe == "python-to-jax"
    assert numba.required_gates[-1] != jax.required_gates[-1]


def test_cli_exports_catalog_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["engines", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "recast.translation-engine-catalog.v1"
    assert document["digest"].startswith("sha256:")
    assert [engine["id"] for engine in document["engines"]] == [
        "recast.fortran-python.numpy",
        "recast.python-numpy.jax",
        "recast.python-numpy.numba",
    ]
