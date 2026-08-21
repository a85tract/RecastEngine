# Repository Guidelines

## Scope

RecastEngine is the Core Layer of SciRecast: framework and basic functionality,
no domain knowledge. If a change would make this repository know something
specific about CESM, about Fortran dialects beyond the reference frontend, or
about NCAR's machines, it belongs in a plugin repository instead.

## Layout

- `src/recast/plugins/` — the extension contract. Ten ABCs. Changing a signature
  here is a breaking change for every plugin, in-tree and out.
- `src/recast/model.py` — the shared vocabulary. Keep it small; a field only one
  workload uses belongs in that plugin's `attrs`/`extra`.
- `src/recast/recipes/` — workload declarations. Stage lists, no logic.
- `src/recast/executors/`, `src/recast/store/` — the shipped implementations.
- `conformance/` — what a plugin must pass, and the suite that checks it.
- `tools/check_hygiene.py` — the site-leakage gate. Runs in CI on every push.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . && uv run mypy
uv run recast plan translate --config '{"target":"numba"}'
```

Four-space indentation, type hints, `snake_case` functions, `PascalCase`
classes. `from __future__ import annotations` in every module.

## Rules that are load-bearing

**The core imports no domain package.** Not numpy, sympy, numba, jax, anthropic,
or netCDF4 — anywhere under `recast.*`. `tests/test_contract.py` checks this by
walking the package. If you need one, you are writing a plugin.

**A transform never judges its own output.** `Transform` → `Candidate`;
`Verifier` → `Verdict`; gated `Verdict` → `Evidence`. Do not add a shortcut.

**Verifiers fail closed.** Could not build, scheduler refused, oracle missing →
`Confidence.FAILED`. Never a weaker pass.

**Findings default to embargoed.** Never lower `Access` or `Disclosure` in code.
See `SECURITY.md`.

**No site-specific values in source.** No `/glade` paths, allocation accounts,
usernames, or scheduler hostnames. They go in config or a case repository. This
is enforced, not requested — the migration from two NCAR-bound repositories is
exactly when it would otherwise slip in.

## Evidence format

CC-Test owns `evidence-manifest.v1.json`. `Evidence.to_manifest()` is the only
place the two vocabularies meet. If CC-Test revises the schema, change the
mapping there — do not reshape the engine's model to match.

## Commits

Imperative subject: `Add oracle cache key`, `Split translate rules from emitter`.
Include the tests. Do not commit workspace output, evidence, or findings.
