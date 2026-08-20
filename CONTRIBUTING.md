# Contributing

## Where to contribute

RecastEngine is the Core Layer. Contributions here are new formal methods, new
verification strategies, new plugin kinds, and improvements to the framework.

Things that belong elsewhere:

| You want to | Go to |
|---|---|
| Fix a translated CESM kernel | the Product Layer repository that owns it |
| Add a new extension | its own repository — [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md) |
| Add a benchmark or validation case | [CC-Test](https://github.com/a85tract/CESM-CC-Test) |
| Report a vulnerability | [`SECURITY.md`](SECURITY.md) — never a public issue |

## Before you open a PR

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy
python tools/check_hygiene.py .
python tools/check_signoff.py origin/main..HEAD
```

All six run in CI. The last two are the ones people forget. `check_hygiene`:
no `/glade` paths, allocation accounts, usernames, or scheduler hostnames
anywhere in the tree. `check_signoff`: every commit carries its DCO trailer,
which is cheap to add as you go and a rebase to add afterwards.

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

## Contributor agreement — DCO

This project uses the [Developer Certificate of Origin](DCO), not a CLA. You
keep the copyright in what you write; you certify that you had the right to
send it. Sign each commit:

```bash
git commit -s
```

which appends `Signed-off-by: Your Name <your@email>` using your `git config`
identity. The e-mail has to be the one you author with — a sign-off in someone
else's name certifies nothing. Forgot it? `git rebase --signoff origin/main`
and force-push.

**Why DCO and not a CLA.** A CLA buys the right to relicense later, and its
price is that every contributor signs a legal document before their first patch
— which is the wrong toll to charge the graduate students and domain scientists
this engine is built for. Apache-2.0 already grants what the project needs to
ship, including the patent grant, so a CLA would be collecting a right we have
no plan to use. If relicensing ever becomes necessary it will be by asking
contributors, which is the honest way to ask.

Contributions are accepted under Apache-2.0, as stated in `LICENSE`.

Sign-off is required from its adoption forward. The commits before it are not
rewritten: back-dating a certification nobody was asked for would be a worse
record than none, and `tools/check_signoff.py` therefore checks the range a
pull request adds rather than the whole history.

## Contact

**Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu>
