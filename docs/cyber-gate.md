# The Cyber gate under the engine: a handoff

2026-09-25. For Chien-Wei Huang, who wrote `hpc-devsecops` in CC-Test and is
picking up the engine side. What of that gate exists here now, where, what
was kept, what was changed on purpose, what is not done, and how to run it.
Everything below is on unmerged branches unless it says `main`; the PR each
piece sits on is in the last table. The LLM audit and the sanitizer plane
live in the private domain extension, which this document does not name by
repository (the hygiene check, `tools/check_hygiene.py`, forbids it here).

## What runs, and how

| `hpc-devsecops` did | the engine does |
|---|---|
| `tools/devsecops-local.sh <repo>` | `recast run audit <repo>` |
| `tools/devsecops-local.sh --range A..B --block` | `recast run audit <repo> --range A..B` (blocking is the default; `--report-only` is the opt-out, see below) |
| `summary.json` under `$HPC_DEVSECOPS_AUDIT_ROOT/<repo>/<timestamp>/` | `--gate-summary PATH` writes that file, same shape, plus `"schema": 1` |
| `git push` with `hooks/pre-push` installed | `git push` with `tools/pre-push` installed: same range per ref, then `recast run audit --range … --gate-summary …` |
| `tools/install-hooks.sh <repo>` | `tools/install-hooks.sh [--force] <repo>` |
| `tools/install-config.sh <repo>` | `tools/install-config.sh [--force] <repo>`, copying `templates/` |
| `--no-ai` | no flag: the LLM audit is a stage of the extension's `audit-cesm` recipe, not of `audit`; leave it out of the recipe config |
| `tools/asan.sh`, `hpc/asan-cam.pbs` | the extension's `dynamic.asan` scanner and its PBS job; `audit-cesm` runs the scanner only when `config["asan"]` is given |

Two scans, each its own gate, both running before either blocks so the
operator gets the whole list: `secret` (gitleaks over history, any finding
blocks) and `composition` (syft, then grype with `.vex/openvex.json` when the
repository has one; a Critical CVE blocks, High is counted). The exit status
is the script's: 2 incomplete, 1 findings, 0 clean. The one inversion is
deliberate and recorded in `docs/roadmap.md`: the script defaults to
report-only and blocks on `--block`; `recast run` blocks by default and offers
`--report-only`, because `run` is what CI and the hook call and both need the
exit status to mean something.

## Where each piece went

| `hpc-devsecops` | now | file |
|---|---|---|
| `tools/devsecops-local.sh` | the `audit` recipe: two scanner stages gating at their own `blocks_on`, findings to a store | `src/recast/recipes/__init__.py` (`AuditRecipe`), `src/recast/scan/secret.py`, `src/recast/scan/composition.py`; on `main` |
| `summary.json` | `RecipeRun.gate_summary()`, written by `--gate-summary`; your state vocabulary and your roll-up, unchanged | `src/recast/run.py`; on `main` (PR #119) |
| `hooks/pre-push`, `tools/install-hooks.sh` | `tools/pre-push`, `tools/install-hooks.sh` | on `main` (PR #121) |
| `tools/install-config.sh`, `templates/.gitleaks.toml`, `templates/.vex/openvex.json` | `tools/install-config.sh`, `templates/` (plus `.recast-audit.json`, the run config the hook hands to `recast`) | on `main` (PR #121) |
| `templates/.github/scripts/ai_audit.py` | the extension's `llm-audit` scanner, a gating stage of `audit-cesm` before the adversarial adjudicator | extension PR #1 |
| `tools/asan.sh`, `hpc/asan-cam.pbs` | the extension's `dynamic.asan` scanner (build a reproducer with ifx/icx under the Executor and run it, or read the `asan.<pid>` logs a full-model run left) and `scripts/pbs/asan_cam.sh` | extension PR #1 |
| the findings | never in the summary; see below | `src/recast/store/filesystem.py`, `SECURITY.md` |

What the two engine scanners kept from the script is written at the top of
each module: `gitleaks git` over history with `--exit-code 0` and findings
read from the SARIF; syft to SPDX, grype over the SBOM with `--add-cpes-if-none`
and the VEX document if present; a tool that is not installed is a check that
did not complete, never a clean scan. `src/recast/scan/__init__.py` records
the maintainer's rule (2026-08-21) that put the other two families in the
extension, and why the public `audit` recipe does not even name a slot for
them.

The hook (`tools/pre-push`) is your range-per-ref logic: `remote..local` for
an existing branch, merge-base with the remote HEAD for a new one, the empty
tree when there is no remote HEAD, deletions skipped. It passes `--config
<repo>/.recast-audit.json` only when that file exists, and leaves each check's
gate summary under `RECAST_AUDIT_ROOT` (default `~/audits/recast`) as
`<repo>/<timestamp>-<pid>/<branch>/summary.json`, printing the path. A missing
`recast` blocks the push (exit 2), as a missing tool blocked yours. The recipe
is `audit` unless `RECAST_AUDIT_RECIPE` names another: CC-Test's wrapper sets
`audit-cesm` when the extension is installed, which is how the LLM audit plane
reaches a push.

`templates/.gitleaks.toml` is your file made site-neutral on purpose: the
hygiene check forbids the site's filesystem prefix and the allocation-account
example in engine source, so the path allowlist regex for that prefix and the
account example are not in it. The template says where they belong: a
CESM-specific copy in the extension, carrying those two lines with `# hygiene:
allow` markers. That copy does not exist yet.

`llm-audit` kept your prompt, your findings schema, the per-file batching
under the character limit, "one failed batch fails the review", and "high
blocks only when the model is also confident". The diff is taken through the
Executor, the model through the extension's `ClaudeProvider`, and the output
is `Finding`s rather than a SARIF file. `dynamic.asan` is the middle of
`asan.sh`; the compiler choice is domain knowledge and the PBS job is a site
script, which is why both are in the extension and the job is unchanged where
ASan matters.

## Deliberately not carried over

- `--staged` and `--worktree` for the secret scan. Both are a diff on
  gitleaks' stdin, and the Executor contract has no stdin; growing it for one
  scanner's second mode waits for something else to need it (`docs/roadmap.md`,
  and the docstring of `src/recast/scan/secret.py`). `llm-audit` takes its own
  diff and does have `mode: staged | worktree | vs-remote`.
- `--no-ai`. The LLM audit is a recipe stage; drop it from the recipe config.
- `ai-audit-report.md`. The findings are records now, in the store below.
- `--binary` on the LLM audit's diff, and the server-side-fallback beta it
  requested.
- The `HPC_DEVSECOPS_*` environment names. Now: `RECAST_BIN` (the executable,
  if not on PATH), `RECAST_AUDIT_ROOT` (where the hook leaves summaries),
  `RECAST_AUDIT_RECIPE` (the hook's recipe, default `audit`),
  `RECAST_FINDINGS_HOME` (where findings go).

## Where the findings go

Not into the summary, not into CC-Test. Each scanner yields `Finding`s at
`Disclosure.PLAUSIBLE` and `Access.EMBARGOED`, and the recipe's `fs-findings`
store (`src/recast/store/filesystem.py`) writes them under
`~/.recast/findings/<project>-<hash>/` (or `RECAST_FINDINGS_HOME`). That store
is a local stand-in for Sec-Track and refuses a root inside any git checkout,
so a finding cannot be committed by accident. A finding reaches Sec-Track only
after adjudication and a human's disclosure decision; `SECURITY.md` is the
rule and `FindingStore.guard()` is the mechanism.

## The summary, and CC-Test

`--gate-summary PATH` writes what `make_manifest.py --security-summary`
already reads: planes `secrets`, `cve`, `ai_audit` (and `dynamic` when that
scanner ran), each with your states (`passed`/`reviewed`, `findings`,
`unavailable` for a tool not on PATH, `error` for a run that failed,
`skipped` for a declared scanner that never ran, `not_configured` for one the
recipe does not have) and your roll-up: any `unavailable` or `error` makes
`INCOMPLETE`, else any `findings` makes `FINDINGS`, else `PASS`. `"schema": 1`
is the only added field; `AGENTS.md` ("Evidence format") says a change to the
keys, vocabulary or roll-up bumps it and tells CC-Test.

One reconciliation, and it is in CC-Test, not here. The script's `PASS` means
"nothing blocking among the checks that ran" and does not count a
`not_configured` plane against it; CC-Test's manifest schema defines `PASS` as
every plane ran. So `make_manifest.py` records a gate `PASS` with an unrun
plane as `INCOMPLETE` and warns (CC-Test PR #6). The engine mirrors the script,
and a run of the public `audit` recipe always has `ai_audit: not_configured`.

## What remains

1. Done: PR #119 (`--gate-summary`) and PR #121 (the hook, the templates,
   this document) are on `main`.
2. Merge the extension's PR #1 (`llm-audit`, `dynamic.asan`, the PBS job).
3. A CESM-specific `.gitleaks.toml` in the extension, with the site prefix
   and account example under `# hygiene: allow` markers. After 1.
4. Run `dynamic.asan` with a real `ifx`, and submit `scripts/pbs/asan_cam.sh`
   on Derecho. After 2. Neither has happened; the tests use fake compilers.
5. Run `llm-audit` against the real API. After 2; the tests script the model.
6. CC-Test step 9d (`docs/CORRECTNESS-ORGANIZATION.md` there). First half
   done 2026-09-25 (CC-Test PR #7): `tools/devsecops-local.sh`,
   `tools/install-config.sh` and `hooks/pre-push` are wrappers over this
   engine's recipe, installer and hook, keeping hpc-devsecops's flags, report
   directory and exit contract; `tools/engine.sh` there finds the engine
   (`RECAST_BIN`, `RECAST_HOME`) and chooses the recipe (`RECAST_AUDIT_RECIPE`,
   PR #122; else `audit-cesm` for a repository carrying the old opt-in
   `.github/scripts/ai_audit.py`, fail-closed when that recipe is missing;
   else `audit`). Second half, after 2: `tools/asan.sh`, `tools/test-ai-audit.py`
   and `hpc/asan-cam.pbs` over `dynamic.asan`, `llm-audit` and the PBS job;
   the `--staged` / `--worktree` patch modes if the `secret` scanner gains
   them (the wrapper refuses them today). Then a run in place on Derecho.
7. CC-Test step 9e: delete the wrappers, `hpc/`, `templates/`, and the
   `summary.json` reader; keep CC-Test's `SECURITY.md` for its own disclosure
   policy. After 6 has run in place.

## Running the tests

From this checkout:

    .venv/bin/python -m pytest -q tests/test_pre_push_hook.py tests/test_install_config.py

`test_pre_push_hook.py` (8 tests) drives a real `git push` into a bare remote
with a fake `recast` on PATH: the range for an existing branch, a new branch,
a deletion; findings and an incomplete audit both block; a missing `recast`
blocks; the installer refuses a foreign hook; and the hook leaves the gate
summary and passes `--config` only when the file exists.
`test_install_config.py` (4) installs the templates, keeps an existing file
unless forced, refuses a non-repository, and checks the templates carry no
site-shaped content. The gate summary's own tests are in `tests/test_run.py`.
The engine suite has pre-existing failures unrelated to this work:
`test_jax_reconstruction_*` (needs Python 3.12 or 3.14) and three in
`test_jax_pointwise_extended`.

The extension's suite runs with this checkout's venv, from its own checkout:

    <this checkout>/.venv/bin/python -m pytest -q -o addopts="" tests

It is installed editable into that venv; after changing its `pyproject.toml`
entry points, reinstall with `uv pip install --python <that python> -e . --no-deps`.

`tools/check_hygiene.py .` must print nothing before any of this is pushed.
