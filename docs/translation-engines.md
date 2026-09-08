# Translation engine catalog

A `Recipe` says how one run executes. A `TranslationEngine` says what that run
accepts and produces, so an outer pipeline builder can select it and connect
artifacts without hardcoding source/target languages.

The declaration is deliberately not executable. Transforms, frontends and
verifiers remain plugins named by its `default_recipe`; the manifest records:

- stable engine id and version;
- digest of the installed implementation;
- exact input and output `ArtifactContract`;
- default recipe, default config and JSON Schema for configuration;
- canonical config-schema and input/output contract digests;
- gates that must appear in the default plan;
- machine-readable capabilities used for UI filtering; and
- repository that owns changes to the engine.

Every manifest and the ordered catalog have canonical `sha256:` digests. Nested
configuration documents are copied into immutable mappings when the manifest
is constructed, so its digest cannot change because a caller later mutates the
dictionary it passed in.

## Inspecting the catalog

```console
$ recast engines
recast.fortran-python.numpy  v1  fortran -> python/numpy  recipe=translate  sha256:...
```

```console
$ recast engines --json
{
  "schema": "recast.translation-engine-catalog.v1",
  "digest": "sha256:...",
  "engines": [ ... ]
}
```

The JSON form is the control-plane/UI boundary. A campaign should lock the
manifest digest, implementation digest, recipe and effective config before it
runs; discovering a newer entry point must not silently change an existing
campaign.

## Registering another engine

```python
from recast.engines import ArtifactContract, TranslationEngine


def engine():
    return TranslationEngine(
        id="example.python-numba",
        version="1",
        implementation_digest="sha256:" + "...64 lowercase hex...",
        default_recipe="example-python-numba",
        input_artifact_contract=ArtifactContract(
            id="example.source-tree.python.numpy",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="python",
            profile="numpy",
        ),
        output_artifact_contract=ArtifactContract(
            id="example.source-tree.python.numba",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="python",
            profile="numba",
        ),
        config_schema={"type": "object"},
        default_config={},
        required_gates=("example.correctness",),
        capabilities=("translation", "deterministic"),
        owning_repository="https://example.invalid/translator",
    )
```

```toml
[project.entry-points."recast.engines"]
"example.python-numba" = "example.engine:engine"
```

The entry-point address must equal the manifest `id`. Its default recipe should
set `Recipe.engine_id`, or override `resolved_engine_id(config)` when only some
configurations represent that engine. The conformance suite checks this link
and checks that every `required_gates` entry is a real gate in the default plan.

