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

### When the work arrives through someone else

Most of what lands here starts in a private repository someone else owns — the
translator and the agent-produced collection are both a student's — and reaches
this repository through a maintainer. That is deliberate: a single hand between
private material and a public repository is what the hygiene gate and the
disclosure ledger assume. It does not let that hand sign for the author.

**Migrate with history wherever the material allows it.** A `git filter-repo`
path rewrite keeps each commit's author, so the person who wrote the code is
still recorded as having written it, with the maintainer as committer. That is
the honest record, and it is the first thing to try.

**It did not turn out to be available here, and the qualifier is doing real
work.** P2 planned to move the pipeline that way and did not: the source
repository is a single commit — one import of a finished tool, not a history —
and the material was decomposed into the plugin contract as it landed, so no
commit could have carried a module across intact. Both of this repository's
relays therefore fall under the rule below, and no commit here carries an
outside author. That is a fact about what was available, and it is recorded
rather than smoothed over, because "migrated with history" was written into
this file and the roadmap before anyone checked whether the history existed.

**When history cannot come along, say so in the file and in `NOTICE`.** P3's
promotions were rewritten as they moved — de-site-ified, split, renamed — so no
commit could carry them intact, and P2's pipeline arrived the same way. Each
file names the source file in its header, and the receiving repository's
`NOTICE` names whose work it was. A provenance line is not attribution; both
are needed, and the header alone was what P2 shipped until this was noticed.

**Sign-off follows authorship, not delivery.** `Signed-off-by` certifies the
right to submit, and only the author can certify that. A maintainer relaying
someone's work adds their own sign-off as the submitter, and does not
manufacture one in the author's name; if the author's certification is needed
and absent, ask for it rather than supply it. Where a commit is rewritten
heavily enough that the maintainer is genuinely a co-author, `Co-authored-by`
records that and both sign.

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
