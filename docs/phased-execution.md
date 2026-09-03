# Durable transform and verify phases

`run_recipe(...)` remains the local, backwards-compatible API for walking a
whole recipe. Distributed runners use a stricter boundary:

```python
from recast.phases import transform_recipe, verify_recipe_candidates

bundle = transform_recipe(
    recipe,
    source_root,
    semantic_config,
    source_artifact_digest=source_tree_digest,
)

report = verify_recipe_candidates(
    recipe,  # compatibility identity only; verification never calls it
    source_root,
    bundle,
    semantic_config,
    expected_source_artifact_digest=locked_source_tree_digest,
    expected_engine=locked_engine_binding,
    project_required_gates=("project.correctness",),
    required_units=("language:module",),
    required_subprograms=("routine",),
    require_no_deferred=True,
)
```

The first call resolves only frontend and transform plugins and freezes the
resulting verification-stage projection in the bundle. The second strictly
decodes that inert plan and walks only executor, oracle, verifier and
evidence-store stages. It never calls `Recipe.validate`, `Recipe.stages` or
`Recipe.resolved_engine_id`; the positional recipe argument remains only for
source compatibility. Its registry facade rejects every other plugin kind. In
particular, it cannot resolve or instantiate a Transform; a failed Verdict
cannot become transform feedback inside this API.

## `recast.candidate-bundle.v2`

A `CandidateBundle` binds the following immutable subjects:

- recipe and canonical semantic-config digest;
- caller-supplied source artifact digest;
- engine manifest, implementation and input/output contract digests;
- exact frontend and transform declarations;
- a recursively frozen `recast.verification-plan.v1` containing only safe
  executor/oracle/verifier/store declarations and their canonical configs;
- the complete frontend discovery set, including parent/child units;
- selected units and the exact `Facts` used by their Transform; and
- candidates, deferred ledgers, notes, patches and generated files.

Generated-file descriptors contain a normalized POSIX relative path, a
`sha256:` blob digest, byte size and the exact
`application/octet-stream` transport type. The v2 codec is self-contained and
inlines canonical base64, with hard limits of 64 MiB per blob, 256 MiB of
unique blobs and 16 MiB per patch. A larger-artifact transport should keep the
same descriptors and externalize their bytes; it must not silently raise these
limits in an untrusted control-plane message.

`CandidateBundle.to_json()` and `decode_candidate_bundle(...)` are inverse only
for the single canonical JSON representation. The decoder rejects duplicate or
unknown keys, non-finite values, non-canonical base64, mismatched size/digest,
unreferenced blobs, absolute/traversing paths and non-canonical JSON text.
Bundle identity contains no source root, workspace, output path or timestamp.
Those paths are separate API keyword arguments; putting `workspace`, `output`
or `store_root` in semantic config is refused.

The source artifact digest is supplied by the surrounding content store. The
engine records and checks that identity rather than inventing another tree
hashing convention. A runner still has to mount that exact artifact at
`source_root`; this API proves the binding presented to it, not the integrity
of an untrusted mount.

## `recast.verification-report.v1`

The report is an acceptance proof, not a diagnostic log. It contains:

- boolean checks for recipe/config/source/composition/installed-engine locks;
- per-unit transform state, candidate digest, deferred count, verification
  status and evidence-record count;
- exact selected/transformed/gated coverage for every project-required unit;
- each engine/project-required gate, its origin, and passed/failed/missing/
  unrecorded unit sets;
- every discovered subprogram, its owning selected unit and gate coverage;
- explicit coverage of project-required subprogram UIDs; and
- the aggregate no-deferred decision.

Acceptance fails closed unless there is at least one selected unit and one
required gate, every selected unit produced a candidate, all declared
verification stages passed, every required gate passed for every unit and was
recorded as Evidence, all required units are selected and gated, all required
subprogram selectors occur in a gated Candidate's coverage ledger, and the
deferred policy passes. Engine-required gates come from the
currently installed manifest and project-required gates are an independent
input; neither can replace the other.

Subprogram selection is not inferred from a language-specific UID convention.
A Transform publishes the source-free protocol
`Candidate.notes["coverage"] = {"subprograms": ["stable-selector", ...]}`.
Required selectors must occur in a well-formed ledger, and the Candidate which
owns each claim must pass every required gate with recorded Evidence. The
bundle's complete discovery set remains in the report as independent trace
data; discovery by itself is not proof that a Transform covered a routine.

The canonical report deliberately omits verifier `detail` and `metrics`, source
bytes, evidence-store URIs, machine paths and timestamps. Operator diagnostics
belong in the run event stream and durable store. This keeps the report safe to
place in a campaign trace and makes its digest stable across workspaces.

Any verifier exception, identity mismatch, wrong candidate digest, missing
required plugin, failed evidence write, or engine-binding drift raises or
produces `accepted=false`. None is weakened to a pass.

Verification first canonical-round-trips the bundle into an isolated copy and
captures both its bundle digest and every transform-produced candidate digest.
After every executor/oracle/verifier/store boundary it checks both the caller's
bundle and the isolated copy byte-for-byte. A plugin which mutates any reachable
`Unit`, `Facts`, `Candidate`, patch, note, deferred item or file is rejected;
the report and every Verdict remain bound to the pre-verification transform
artifact rather than a digest recomputed after plugin code returns.
