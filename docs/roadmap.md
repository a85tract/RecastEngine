# Roadmap

Phases, not dates. Each one has a check that says whether it is done.

## P0 — decisions (settled)

| Decision | Options | Status |
|---|---|---|
| Contributor agreement | DCO (lightweight) vs CLA (allows relicensing later) | **decided**: DCO, and enforced |
| PyPI name | `recast-engine` — `recast` is taken by an unrelated 2021 package | **decided**, not yet uploaded |
| What never goes public | one upfront list vs case by case | **decided**: case by case, into `docs/disclosure-ledger.md` |

The DCO is at `DCO`, the terms are in `CONTRIBUTING.md`, and the check is
`tools/check_signoff.py` — because an agreement nothing verifies is not an
agreement. It reads a range rather than the whole history: sign-off is required
from adoption forward, and the commits made before it are left as they are.

The third one resolves into a process rather than a list, because a list of what
must never go public, written before the material is in front of you, is a
guess. The cases arrive with the work — P3 moves 662 scripts out of a private
collection, P6 flips two private repositories — and each is ruled on as it turns
up. What that costs is a real record: `docs/disclosure-ledger.md` carries the
case, the reason, and *which* mechanism holds it, and P6 does not flip while a
case in it is still open. Eight cases are already settled there, five of them
enforced by `tools/check_hygiene.py`.

## P1 — scaffold (this commit)

Plugin contract, model types, registry, `local` executor, filesystem stores, the
four recipes as declarations, CLI introspection, CI, hygiene gate, licence.

**Done when:** `recast doctor`, `recast recipes`, and `recast plan` run with zero
optional dependencies installed, and `test_contract.py` passes.

## P2 — migrate the translator

Move CESM-language-translator's `pipeline/` (22 modules, ~10k lines) in with
history via `git filter-repo` path rewrite, refactoring as it lands:

- `translate.py` (2,883 lines) splits into `rules/` + `backend/numpy`. It was
  parser, rule library, and emitter at once, which is why nothing else could
  reuse any part of it. **Landed**: the split is complete
  (`docs/splitting-the-translator.md`), ending in `translate.numpy` -- the
  Transform the `translate` recipe names -- with `tools/emit_diff.py` holding
  every emitted byte to the pipeline across 27 modules.
- `extract_interface` / `extract_constants` / `chunk` / `resolve_use` become
  `recast-fortran`'s `Frontend`. **Landed** as `recast.fortran`, behind the
  `fortran` extra: the analysis is unchanged, but the rendering it used to do
  went with the Transform that has a target language, and the kind table it
  reached into as a module global is now an argument.
- `notary` / `highprec_verify` / `rwset` become `Verifier`s with honest
  `provides` levels. **Landed**: `static.rwset` (SAMPLED, the first gate),
  `symbolic.notary` (SYMBOLIC, over recorded rewrites), and
  `differential.bitexact` (BIT_EXACT, with `TOLERANCED` only when the
  operator asks), reporting in the ULP vocabulary of `recast.verify.ulp`.
- `gen_wrapper` + the f2py build become the `f2py-golden` `Oracle`.
  **Landed**: flat wrappers over the public API, the compile through the
  executor, the cache key folding source digest, compiler version and flags.

**Done when:** the `translate` recipe runs end to end on `examples/` and
reproduces the existing bit-exact result for one scheme, with 408 files' worth
of `/glade` paths gone (`tools/check_hygiene.py` is the check).

**Met** for the stage chain on 2026-08-18: `wv_sat_methods` -- the scheme the
pipeline bootstrapped on -- runs frontend → `translate.numpy` (zero deferred
blocks) → `static.rwset` (50 blocks match) → `f2py-golden` (gfortran 16, the
reference's own flags) → `differential.bitexact`: 400/400 points bit-exact
across the seven public API functions, under the golden set's init constants.
`tests/test_f2py_oracle.py` keeps a compiler-gated copy of that spine, and
breaks the candidate on purpose to prove the gate can fail.

And the chain is one command: `recast.run` walks a recipe's stages -- order,
fail-fast gates, the oracle cache, optional-stage downgrades -- and writes
one CC-Test evidence manifest per Verdict through the store, because a
Candidate without Evidence is a draft regardless of how good it looks.
`recast run translate examples/toy_physics --config .../recast.json` is the
public form, runs in CI's spine job, and ends with three manifests on disk:
sampled, bit_exact, symbolic. P2's remaining obligation is the standing one
the two differential tools carry: the translator keeps being developed, and
`golden_diff` and `emit_diff` re-run against whatever it looks like next.

Two checks, because "no site paths" and "same answers" are different claims.
`tools/golden_diff.py` runs a migrated stage over the sources the original
pipeline ran over and diffs it against the JSON that pipeline left behind,
sorting differences into additive (a key the old output did not have, which
nothing reading it can notice) and behaviour changes (which have to be
defended one at a time, in its `ACCEPTED` table, with a reason).

This one does not retire when P2 does. The translator keeps being developed,
so migrating from it is a standing job rather than a finished one, and the
check has to be re-runnable against whatever the sources look like next.
Accepted divergences pin the exact values they excuse, so a change upstream
brings them back for re-confirmation instead of staying quietly excused.

## P3 — triage the 664 agent scripts

CESM-Agent-Produced-Scripts, into four buckets:

| Bucket | Destination | Rough count |
|---|---|---|
| reusable, domain-independent | rewritten into engine modules, with tests | 40–60 |
| HPC execution | interface here, implementation in an executor plugin | ~170 |
| kernel implementations (`15_kernel_impl/`) | Product Layer repos — they are ports, not tooling | 83 |
| CESM-specific and one-shot | stay in the archived repository | ~350 |

**Done when:** every promoted script has a test and a home; the rest is tagged
read-only. Nothing is copied in to make the repository look fuller. And every
`EXCLUDE` decision has a row in `docs/disclosure-ledger.md` — the bucket that
stays behind is the one nobody reviews again, so the reason it stays behind is
written down while it is still fresh, not reconstructed at P6.

## P4 — empty out the domain

Build the CESM extension as its own repository and move every CESM-specific
rule, catalog, and golden set into it. Bring freeCAM and CESM-jax-kernels onto the
`refactor` and `port` recipes.

A separate repository rather than a directory under SciRecast's `cesm/`,
because this phase's check is that the engine passes with that extension
*uninstalled* — which is only a real check if it is a separately installable
distribution.

**The translate side (PyCAM5 track) landed 2026-08-18.** The extension ships
the framework stub tables, CAM's kind and constant conventions, and the
physical sampling ranges, delivered by three entry-point plugins: the `cesm`
frontend, the `translate.cam` transform, and the `translate-cam` recipe. A
real CAM module now runs `recast run translate-cam` to a bit-exact verdict
from a config that names the unit, the gate's subprograms and the scheme's
init call — kinds, stubs, and Kelvin never appear in it. The engine took no
CESM branch; the one engine change the attachment surfaced was contract
completion (the CLI resolving plugin recipes, and the frontend recording
Fortran accessibility so the oracle wraps only public symbols — a fact any
domain needs).

**The refactor and port sides (freeCAM, JaxCAM6) are deliberately deferred**
until their student tooling is ready; their entry-point slots are sketched in
the extension's pyproject and nothing is declared before it exists.

**Done when:** the engine passes its tests with the CESM extension
uninstalled, and freeCAM's validation gate runs through `Verifier` rather than its own
`validate_*` scripts. This phase is the only real proof that the engine is
domain-independent.

## P5 — prove the contract out of tree

Build the first extension that lives outside this repository and depends only on
the published contract: PBS/Slurm executors, relay/resume of multi-day runs, and
a restricted `FindingStore` for Sec-Track. It passes `conformance/`.

**Done when:** the engine works without it, and it needed no engine patches. Any
patch it did need is a hole in the contract, and the hole is the finding.

## P6 — public

Scrub → security review → archive the two source repositories read-only → flip
visibility → check that the SciRecast site's links resolve.

There is no pointer to repoint. SciRecast is a Jekyll site rather than a
submodule umbrella, and its `index.md`, `engine.md` and `contribute.md` already
link `a85tract/RecastEngine` by URL — links that 404 for everyone outside the
org until the flip and start working the moment it happens. So the last step is
a verification, not an edit, and it has to include the deep ones:
`contribute.md` points into `src/recast/plugins/`, `docs/writing-a-plugin.md`
and `conformance/`. A deep link is how a rename gets discovered, by someone
else, after the repository is already public.

**Done when:** `docs/disclosure-ledger.md` has no open case, and every settled
one names a mechanism that is actually in place — the pattern in
`check_hygiene.py`, the path off the migration manifest, the record class
guarded. A ledger row whose mechanism is still prose does not count.

**Irreversible.** Both source repositories are private and carry NCAR paths, a
username, an allocation account, PBS vulnerability research, and CPG audit
entries. Filtering at migration time (P2/P3) rather than before the flip is what
makes this safe, because git history keeps whatever was ever committed — which
is also why the ledger is written as the cases turn up rather than assembled
here. By the time this phase runs, the material that would populate it has
already been moved or left behind, and the decisions are months old.
