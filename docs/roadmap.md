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

P3 found five more that are the pipeline's but postdate the translator's last
commit — `jaxize`, `jax_shim`, `coverage_sweep`, `intel_math`,
`extract_build_flags`. Their own usage strings say `pipeline/`, and they exist
only in CESM-Agent-Produced-Scripts, so for those five the collection is the
source and P2 has to take them from there before it is archived.

Four have a slot waiting here already. `PortRecipe` declares a `jax` backend and
`plugins/transform.py` names `recast.port.kernel-to-jax`, neither implemented;
the first three files are that implementation, and `jax_shim` in particular is
the JAX counterpart of the `_f_*` anchors `transform/numpy/runtime.py` already
ships. `intel_math` is a third libm beside the glibc-versus-SIMD difference the
same runtime already models, which puts it next to `transform/profiles.py`.

The fifth splits, and the seam is worth landing on rather than across.
`extract_build_flags` knows which flags can change numerics — compiler knowledge,
belonging beside `profiles.py` — and where the compile line hides in a CESM build
log, which is the domain extension's. `oracle/f2py.py` takes `fflags` from config
with a default today; lifting them verbatim out of the production build log,
never hand-copied, is the capability still missing.

Ten more share a filename with something already in `pipeline/` or `tests/` but
differ in content; the triage flags each and decides none.

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

## P3 — triage the 662 agent scripts (done, bar the archiving)

CESM-Agent-Produced-Scripts, every file given a destination. The plan said four
buckets; executing it needed eight, and the extra four are the findings.

| Disposition | Count | Where it goes |
|---|---|---|
| `PRODUCT` | 177 | port outputs and their unit tests → Product Layer repos |
| `HPC-EXEC` | 170 | interface here, implementation in an executor plugin |
| `ARCHIVE` | 141 | one-shot: bound to a run, a bug, or a date that will not recur |
| `P2` | 69 | already in CESM-language-translator — P2 migrates it, with history |
| `EXCLUDE` | 50 | stays behind: disclosure-ledger rows 6 and 7 |
| `PROMOTED` | 28 | landed in the domain extension, de-site-ified, tested |
| `POOL` | 22 | re-runnable tooling with no current need; promoted when one names it |
| `META` | 5 | tooling about the agent transcripts themselves |

The table is generated from rules rather than hand-written, and lives with its
reasoning in the domain extension's `migration/` directory. Four things the plan
did not anticipate:

**69 of the "reusable" files are already in the translator.** P2 migrates those
with history; copying them in P3 would land them twice and throw away exactly
the provenance the `git filter-repo` pass exists to keep. Whole planned areas —
golden-set generation, reference builds — belong to P2 on this finding alone.

**"Reusable, domain-independent" was two claims, and most survivors were
neither.** What remained after the overlap was mostly unit tests for ported
kernels, which belong with the ports by the same argument that sends
`15_kernel_impl/` to the Product Layer, and one-shot forensics. Of the rest,
seven files were promoted into the domain extension and then moved back out for
not encoding anything about CESM, and six more were nearly sent *here* on the
strength of not being about CESM — which is only half a reason. They parse
Claude Code transcripts and PBS accounting CSV: formats, and not this engine's
subject either. That is what `META` is for, and the finding is that the
engine/domain/product/executor split has no place for tooling about the agentic
process itself, in a project whose engine is an agentic one.

**Content identity is proof; a shared filename is not.** Overlap with the
translator was detected two ways, and only one of them is evidence. Ten files
share a filename with something in `pipeline/` or `tests/` while differing in
content, and treating that as proof marked `08_cpg_tools/gen_stubs.py` — which
generates Fortran stubs so a static-analysis IR will build, and shares nothing
with `pipeline/gen_stubs.py` but its name — as P2's to migrate *here*. That is
precisely the hole the disclosure ledger warns about: a directory kept out of
the migration is not protected by a regex that was never written for it. The
two `EXCLUDE` buckets are sealed now — nothing leaves them by inference, only by
a decision written into the ledger — and a shared filename annotates a row
instead of deciding it.

**So the reusable bucket is 28, not 40–60, and it is not "engine modules".** The
gap is the overlap plus those audits, not a shortfall. Nothing was copied in to
close it.

**Done when:** every promoted script has a test and a home; the rest is tagged
read-only. Nothing is copied in to make the repository look fuller. And every
`EXCLUDE` decision has a row in `docs/disclosure-ledger.md` — the bucket that
stays behind is the one nobody reviews again, so the reason it stays behind is
written down while it is still fresh, not reconstructed at P6.

Homes and tests: **done.** 28 promoted files, each carrying a header naming the
collection file it came from, covered by a suite that runs by discovery so the
next one is covered the moment it lands. It proves the properties that survive a
migration — no site strings, parses, resolves its site through the shared helper
— rather than the science, which needs a run directory, a GPU, or a compiled
oracle.

Ledger rows: **done**, and they reconcile. Rows 6 and 7 account for all 50
`EXCLUDE` files, 42 and 8, which is the check that caught the `gen_stubs.py`
escape above — the triage said 49 and the ledger said 50.

Tagging the collection read-only: **deferred to P6**, with the rest of the
archiving. Safe to defer because the filtering happened file by file at
migration time rather than as a cleanup pass at the end, and cheaper because P2
still has files to take out of the collection.

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
