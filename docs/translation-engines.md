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
recast.python-numpy.jax      v1  python/numpy -> python/jax  recipe=python-to-jax  sha256:...
recast.python-numpy.numba    v1  python/numpy -> python/numba  recipe=python-to-numba  sha256:...

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

## Shipped Python accelerator engines

`recast.python-numpy.numba` and `recast.python-numpy.jax` accept the same
`recast.source-tree.python.numpy` contract and produce different, non-interchangeable
contracts. They are peers of the Fortran→NumPy engine, not target flags on it.

The `python-numpy` frontend discovers `.py` modules that import NumPy, skips the
engine's `.recast` workspace, and uses `__all__` or `_RECAST_EXPORTS` when one is
present; otherwise public module functions are exports. It never imports source
during discovery. Each recipe then runs:

```text
python-numpy frontend -> backend transform -> static.complete
                      -> untouched python-source oracle
                      -> backend differential verifier -> evidence
```

The Numba transform decorates the reachable export closure with
`@njit(cache=False, fastmath=False)`. The JAX transform lowers a conservative,
functional NumPy AST subset to `jax.numpy`, enables 64-bit values, and applies
`@jit`. A construct outside either reviewed subset is recorded in
`Candidate.deferred`; `static.complete` rejects it before numerical verification.
That is deliberate input to an outer repair/agent queue, not a silent fallback.
Generated decorators use engine-reserved module aliases, and source which binds
one of those reserved names is rejected.

The differential verifier calls the original source and candidate on identical,
deterministic numerical samples. The two sides run as separate `Executor` jobs
in isolated `python -I` processes, so ordinary project imports and backend
monkeypatches cannot leak through the verifier's `sys.modules` or between
project roots. The parent exchanges only bounded canonical JSON with those
workers; it never deserializes project objects or pickle data. A timeout,
non-canonical response, missing response, or malformed observation fails closed.

Equal strings/objects and unchanged input arguments cannot make the numerical
gate non-vacuous. The candidate worker captures the installed backend identity
before importing project code. It checks that Numba produced a genuine
`CPUDispatcher` with a compiled signature; for JAX it first requires a genuine
JAX-jitted callable and JIT to be enabled, then explicitly lowers and compiles
the sampled signature, executes that compiled artifact, and compares its result
again. Missing NumPy/backend dependencies, an unavailable oracle, a
non-importable candidate, zero numerical observations, deferred work, and
runtime errors all produce `FAILED`.

```console
recast run python-to-numba ./project --config numba.json
recast run python-to-jax ./project --config jax.json
```

The locked configs are respectively
`{"target":"numba","frontend":"python-numpy","executor":"local"}` and
`{"target":"jax","frontend":"python-numpy","executor":"local"}`. Existing
direct Fortran→Numba/CUDA recipe variants retain their legacy recipe-level
behavior and do not borrow either Python engine identity.
