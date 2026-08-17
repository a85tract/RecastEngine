# Contributing

## Where to contribute

RecastEngine is the Core Layer. Contributions here are new formal methods, new
verification strategies, new plugin kinds, and improvements to the framework.

Things that belong elsewhere:

| You want to | Go to |
|---|---|
| Fix a translated CESM kernel | the Product Layer repository that owns it |
| Add CESM-specific rules or catalogs | `recast-cesm`, in CESM-modernization-overview |
| Add a benchmark or validation case | [CC-Test](https://github.com/a85tract/CESM-CC-Test) |
| Report a vulnerability | [`SECURITY.md`](SECURITY.md) — never a public issue |

Human developers do not directly modify the Product Layer. When end users open
issues there, the engine generates, tests, and merges the fix.

## Before you open a PR

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy
python tools/check_hygiene.py .
```

All five run in CI. The last one is the one people forget: no `/glade` paths,
allocation accounts, usernames, or scheduler hostnames anywhere in the tree.

## What a good PR looks like

- **New verifier?** State `provides` honestly and show the numbers it produces,
  not just that it passes. `{"max_ulp": 0, "bit_exact": 512}` is reviewable;
  `{"ok": true}` is not.
- **New transform?** Show what lands in `deferred` as well as what translates.
  A transform with an empty deferred list on a hard input is usually hiding
  something rather than handling it.
- **Changing a `plugins/` signature?** That breaks every plugin, including
  ones you cannot see. Say so in the PR, and expect a major version bump.
- **Adding a dependency to the core?** Almost certainly no — make it an extra.
  The core installs with zero dependencies and CI asserts it stays importable
  that way.

## Contributor agreement

Undecided (DCO vs CLA) — see [`docs/roadmap.md`](docs/roadmap.md) P0. Until it is
settled, contributions are accepted under Apache-2.0 as stated in `LICENSE`.

## Contact

**Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu>
