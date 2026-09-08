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
tools/ci_local.sh
```

That runs every job in `.github/workflows/`, here, reading the commands out of
the workflow files rather than keeping its own copy of them — so it cannot
drift into checking something else. Three exits: 0 is "CI would pass", 1 is
"it would not", and 2 is that a job could not run here at all, which is not a
pass with a caveat. `tools/ci_local.sh --list` shows the jobs; naming one runs
only it.

Two of them need a tool this repository does not install for you, and report
`NOT RUN` rather than passing when it is missing:

| Job | Needs | Why it is worth having |
|---|---|---|
| `spine`, `conformance` | `gfortran` | the only jobs that compile Fortran, and the only ones that run a verification chain to a verdict |
| `secrets` | `gitleaks` | scans full history; the gate P6 depends on |
| `sbom` | `syft` | proves the SBOM scanner runs |

`port-spine` needs no compiler, which is a property of its oracle rather than a
convenience — see the shipped-trees section of `corpus/README.md`.

The short version, when you only want the fast ones:

```bash
uv run --isolated --extra dev --extra fortran --extra translate pytest -q
uv run --isolated --extra dev ruff check . && uv run --isolated --extra dev ruff format --check .
uv run --isolated --extra dev mypy
python3 tools/check_hygiene.py .
python3 tools/check_signoff.py origin/main..HEAD
```

`--isolated` is not decoration. Without it `uv run` uses this checkout's
`.venv`, which on a working machine usually has a domain extension installed in
it — and the engine passing *with* an extension present is a different and
weaker claim than the one CI makes, since the runner never has one. For the
same reason, do not open with `uv sync`: it makes the environment match the
lockfile exactly, which removes anything you installed alongside it.

The last two are the ones people forget. `check_hygiene`: no `/glade` paths,
allocation accounts, usernames, or scheduler hostnames anywhere in the tree.
`check_signoff`: you are in `CLA-SIGNATORIES.md` and every commit carries its
sign-off trailer, which is cheap to add as you go and a rebase to add afterwards.

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

### Names and paths already taken

The registry refuses two plugins under one name, and a module path can hold
one module. Both are shared with plugins that install alongside this tree
rather than inside it, so the names and paths below are taken even where
this tree has nothing at them. A contribution that adds a plugin of the same
kind — another JAX target, another tolerance gate, another flat oracle —
picks a new name and a new path, and the two can then be installed together
and chosen between in a recipe.

Plugin names: `c-kernel`; `translate.tree`, `port.jax`, `port.tree-jax`,
`translate.numba`, `translate.cuda`, `translate.python-numba`,
`translate.python-jax`; `differential.tolerance`, `differential.probes`,
`differential.python-numba`, `differential.python-jax`,
`performance.benchmark`; `f2py-golden-flat`, `dump-replay`,
`executable-golden`; the recipes `port`, `python-to-numba`, `python-to-jax`;
the engines `recast.python-numpy.numba`, `recast.python-numpy.jax`.

Module paths under `src/recast/`: `fortran/flatten.py`,
`transform/numpy/tree.py`, `transform/numpy/flat.py`, `transform/jax/`,
`transform/numba/`, `transform/cuda/`, `transform/python_accelerators.py`,
`c/`, `verify/tolerance.py`, `verify/conditioning.py`, `verify/probes.py`,
`verify/probe_inject.py`, `verify/gpu_time.py`,
`verify/python_accelerators.py`, `verify/python_accelerators_worker.py`,
`oracle/flat.py`, `oracle/record.py`, `oracle/dump_replay.py`,
`oracle/executable.py`, `recipes/optional.py`,
`conformance/builtin_optional.py`.

The `flatten` option of the Fortran frontend is such a plugin's: without it
the frontend translates flat interfaces, and asking for `flatten` says what
is not installed.

## Contributor agreement — CLA

This project takes contributions under a [Contributor License Agreement](CLA.md).
You keep the copyright in what you write; you license it to the project's
maintainer under terms that let the project be relicensed, and you certify
that you had the right to send it (the Developer Certificate of Origin is
section 4 of the agreement). Two steps, the first once:

1. **Sign, once.** Read `CLA.md`, then add a row for yourself to
   [`CLA-SIGNATORIES.md`](CLA-SIGNATORIES.md) in your first pull request,
   listing every e-mail you author commits with.
2. **Sign off every commit.**

   ```bash
   git commit -s
   ```

   appends `Signed-off-by: Your Name <your@email>` using your `git config`
   identity. That trailer is your statement that the commit is submitted under
   the agreement. The e-mail has to be the one you author with — a sign-off in
   someone else's name certifies nothing. Forgot it? `git rebase --signoff
   origin/main` and force-push.

`tools/check_signoff.py` enforces both: the trailer against the author, the
author against the signatories file as it stands at the head of your branch.
If your university or employer has rights in what you write, CLA section 5
applies; ask before your first patch rather than after.

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

**Why a CLA, and why it replaced the DCO.** The project began under the DCO,
on the argument that a CLA collects a right — relicensing — the project had no
plan to use, and that signing a legal document is the wrong toll for the
graduate students and domain scientists the engine is built for. The first
half of that argument stopped holding on 2026-09-07: the maintainer may need
to offer the engine under other terms — to an institution whose policy
requires them, or in a form that funds the project's upkeep — and a project
that may one day do so has to hold that right from every contributor, or it
cannot. Asking afterwards is not a plan when the answer can be no. The toll is
kept as low as a CLA can be: one row in a file, once, and the same
`git commit -s` the DCO already asked for. The switch was made while the
contributors were the people in one research group, which is the cheapest
moment it will ever have.

Contributions are published under Apache-2.0, as stated in `LICENSE`; what
the agreement adds is that they may also be offered under other terms.

The agreement is required from its adoption forward. The commits before it are
not rewritten: back-dating a certification nobody was asked for would be a
worse record than none, and `tools/check_signoff.py` therefore checks the range
a pull request adds rather than the whole history. A signatory can extend the
agreement to their earlier commits with the "covers from" column.

## Contact

**Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu>
