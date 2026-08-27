# Roadmap

Phases, not dates. Each one has a check that says whether it is done.

## Where it stands

**Pre-alpha.** The plugin contract, the CLI, and the whole `translate` recipe
exist and run end to end — Fortran in, verified NumPy and evidence manifests
out, gated bit-exact against the compiled original.

The other three are not one story. `port` and `audit` plan clean on the
plugins shipped here. `refactor` declares four slots nothing fills —
`refactor.carve`, `static.no-numerics-moved`, `pinned-run`,
`fullmodel.bitwise` — and is what P3–P4 are for. Planning clean is a weaker
claim than the one `translate` can make, for the reason *The other two
translate targets* below sets out: it says a plugin is registered, not that a
run reaches a verdict.

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
case in it is still open. Ten cases are already settled there, five of them
enforced by `tools/check_hygiene.py`. (This sentence said "eight" while the
ledger held nine, which is what a hand-kept count does; it is corrected rather
than made automatic, because the number is prose here and the ledger is the
record.)

## P1 — scaffold (this commit)

Plugin contract, model types, registry, `local` executor, filesystem stores, the
four recipes as declarations, CLI introspection, CI, hygiene gate, licence.

**Done when:** `recast doctor`, `recast recipes`, and `recast plan` run with zero
optional dependencies installed, and `test_contract.py` passes.

## P2 — migrate the translator (done)

Move the translator's `pipeline/` (22 modules, ~10k lines) in,
refactoring as it lands. The plan said "with history, via a `git filter-repo`
path rewrite", and that is **not what happened** — see "The history that was
not carried" at the end of this phase:

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

Met. All four bullets landed, and `recast run translate examples/toy_physics`
walks frontend → transform → `static.rwset` → `f2py-golden` →
`differential.bitexact` → store, failing closed on a machine with no Fortran
compiler rather than passing anything it could not check.

P3 turned up five files in CESM-Agent-Produced-Scripts whose usage strings say
`pipeline/` but which postdate the translator's last commit, so no path rewrite
reaches them. Two of them would extend what P2 landed rather than complete it,
and are recorded here so they are not lost — not as work this phase is waiting
on:

- `intel_math` is a third libm beside the glibc-versus-SIMD difference
  `transform/numpy/runtime.py` already models in its `_f_*` anchors, which puts
  it beside `transform/profiles.py`.
- `extract_build_flags` splits. Which flags can change numerics — the ifort/ifx
  two-token table, the drop list — is compiler knowledge and belongs beside
  `profiles.py`; finding the compile line inside a CESM build log is the domain
  extension's. `oracle/f2py.py` takes `fflags` from config with a default today,
  and lifting them verbatim out of the production build log, never hand-copied,
  is the capability neither half supplies yet.

  Sharper than "a capability neither half supplies", and worth stating that
  way: the tool's own docstring makes it a rule — *everything that can change
  numerics must come verbatim from the production build log, never hand-copied
  or chosen for harness convenience*. `DEFAULT_FLAGS` in `oracle/f2py.py` is
  hand-picked, conservatively and with its reasoning written down, but
  hand-picked. So this is not a feature the engine lacks; it is a rule the
  engine's own source pipeline wrote down and the engine does not yet keep.

  **Decided 2026-08-21: this arrives as a relay**, like the JAX backend and for
  the same reason — it is developed in its author's repository and merged
  across, not rewritten here from the outside. Which fixes the shape of the
  eventual change: the compiler half lands beside `profiles.py`, the log-
  reading half goes to the domain extension, and `NOTICE` gains the entry
  before either does. Not work this repository starts.

The other three are P4's, below. Ten more files share a filename with something
in `pipeline/` or `tests/` while differing in content; P3's triage flags each
and decides none.


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
the differential tools carry: the translator keeps being developed, and
`golden_diff`, `emit_diff`, `numba_diff` and `cuda_diff` re-run against
whatever it looks like next. There are four of them rather than the two this
paragraph named until the Numba and CUDA targets landed, and the fourth one's
denominator is itself a finding -- see "The other two translate targets" below.

**Every one of them now prints the revision it compared against.** The line
was added after a re-run reported three tools drifting at once — `emit_diff`
5 → 110, `numba_diff` 0 → 62, `cuda_diff` 3 → 18 — which turned out to be
neither repository moving but `RECAST_INTRINSICS` being unset, so every
transcendental the `ifx` profile respells counted as a difference. Chasing
that exposed the gap this line closes: the numbers below were recorded with no
note of *which* upstream they were measured against, and upstream had in the
meantime replaced a squashed single commit with 249 real ones. A count from a
differential is a claim about two trees, and only one of them is in this
repository.

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

### The other two translate targets, and how far their evidence reaches

`translate.numba` and `translate.cuda` were the two slots the `translate`
recipe declared and nothing filled -- `recast plan translate --config
'{"target":"numba"}'` answered `[MISS]`. Both answer `[ok]` now, and both are
relays rather than rewrites: the backend is upstream's, and the check is a
third and fourth emission differential beside `emit_diff.py`.

They landed the same way and their evidence is not comparable, which is the
part worth writing down rather than leaving in two commit messages.

Both re-confirmed 2026-08-27 against the translator at `6486e104d`,
with `RECAST_INTRINSICS` pointing at the domain extension's intrinsics table.
So were `emit_diff` (`different=2`) and `golden_diff --live` (no unexplained
differences).

| | compared | result |
|---|---|---|
| `tools/numba_diff.py` | 27 modules, 177 kernels, 10,193 lines | `different=0 error=0` |
| `tools/cuda_diff.py` | 27 modules, 44 device functions, 1,504 lines | `different=4 crashed=145` |

**`emit_diff` is at 2 rather than the 5 it sat at for three revisions, and
that is the same finding arriving in a third place.** Generic dispatch gained
a declared-dtype axis upstream and then here; all three of
`coordinate_systems_mod`'s differences were the two `distance` overloads,
which rank and integer-ness cannot separate and the argument's derived type
can. The two that remain are `vertical_diffusion`'s `p%finalize` and
`p_dry%finalize`, are older than any of this, and are [#6]. `numba_diff` went `1` to `0` for the same reason and gained the kernel
it had been delegating.

**`cuda_diff` stayed at 4, and the fourth changed character rather than
closing.** It was "the pipeline emitted a device function and the engine
delegated"; it is now "both emit it, and the engine calls
`_distance_cart2d_kp` where the pipeline calls `distance_cart2d`".

**Every difference these three tools still report is a tracked upstream
issue, and the engine is on the correct side of all of them.** That is worth
stating plainly rather than leaving as four counts, because a raw count reads
as "the port is four wrong" and it is the other way round:

| | count | upstream issue | what it is |
|---|---|---|---|
| `emit_diff` | 2 | #6 | a type-bound call in statement position becomes `pass` and the block still scores `mechanical`; this engine refuses it |
| `numba_diff` | 0 | -- | |
| `cuda_diff` | 4 | #14 | a generic call in a device function is resolved correctly and then spelled in the *host* naming scheme, so the emitted name is one the generated file never defines |
| `cuda_diff` crashes | 145 | #13 | `emit_kernel_variant` omits the per-subprogram state its base sets up, so the pipeline raises before emitting an expression |

The numbers are the translator's own tracker and are not linked, for the
reason `tools/check_hygiene.py` gives: that repository is private, so a URL
here is one nobody outside can follow. Each row says what the issue says, so
the table stands on its own for a reader who cannot open them.

All four `cuda_diff` differences are #14 and nothing else: the three
`rising_factorial` sites in `mg_utils` the issue names, and the `distance`
site in `coordinate_systems_mod` that the dispatch axis above made reachable
for the first time. The engine's `_<name>_k`/`_kp`/`_ka` spelling is what a
generated CUDA module actually defines -- `transform/cuda/emitter.py` gives
the reason -- and the pipeline's bare `<name>` is a `NameError` waiting on the
first launch that reaches it. So this pair of counts cannot reach zero from
this side: closing them needs the upstream fix, not a change here, and
adopting the pipeline's spelling to make a number go down would be adopting
the defect.

**Numba is what a finished relay looks like.** Every kernel and host wrapper
the pipeline emits over the corpus, byte for byte, with headers carved out for
the reason `transform.numpy.modules` already gives.

**CUDA's denominator is a quarter of the surface it emits, and that is not a
detail of the count.** `cudaize.py` raises on 146 of the corpus's subprograms
before emitting an expression, so for those there is no upstream answer to
compare against. The engine emits 127 of them anyway -- **5,725 lines that
nothing has checked, against 1,504 that something has, or 79% of the target's
output resting on no evidence at all.** It is not that those lines are known
wrong; it is that the relay standard is "the same bytes as upstream", and where
upstream produces nothing that standard is vacuous rather than satisfied.

Two things make the gap easy to under-read, and both were under-read here
first:

* `cuda_diff` counts a crash apart from a difference, correctly -- saying the
  engine disagrees with a traceback would be false. But `crashed=146` reads as
  a fact about upstream when it is also a fact about this repository's
  coverage, and the harness never renders the engine's side on that path, so
  the 5,725 lines do not appear in its output at all. They were measured with a
  probe copy, not by the harness.
* Nothing else covers the backend. `tests/test_numba_backend.py` carries two
  CUDA unit tests over a toy fixture -- the decorator line and the signature
  line -- and `differential.bitexact` needs a GPU, so CI's `gpu` marker skips
  it. Outside the differential the target is close to bare.

The commit that landed it said the crash fires "on any module with an
`allocate`". **That is wrong and the correction matters for the size of the
gap**: `alloc_lb` is read at the top of `array_ref`, which is the general
subscript emitter reached from seven call sites, so the trigger is any array
subscript at all. `mo_airmas.F90` and `mo_util.F90` contain no `allocate` and
both crash; 26 of the 27 modules are affected rather than the subset with
allocatables.

Both defects are upstream's and are filed there rather than fixed here --
upstream issue #13 for the crash and #14 for the three differences, which are one defect at three sites: a
generic call inside a device function emits the resolved specific under
`Translator`'s naming scheme, so `rising_factorial_r8` is called where only
`_rising_factorial_r8_k` is defined. Patching upstream to widen the comparison
is not available to this repository and would not be wanted if it were: the
baseline the differentials compare against is upstream *as the bit-exact CESM
gates ran it*, and editing it to make more of it comparable destroys the thing
being compared to.

So the denominator moves when #13 does, not before. Until then `translate.cuda`
is honestly described as *planning clean and relayed, on evidence covering 21%
of what it emits* -- which is a different claim from the one `[ok]` makes, and
the reason this section exists.

**And `[ok]` at plan time is a weaker claim than it reads for both of them, in
a way that has nothing to do with the differentials.** `recast plan` tests
`stage.plugin in REGISTRY.names(stage.kind)` and nothing else: it is a
registration check, so a backend whose Python dependency is not installed
still reads `[ok]`. Only `translate.numpy` records the rwset protocol in
`Candidate.notes`, and `static.rwset` fails closed on a transform that records
none -- so a `translate` run retargeted to `numba` or `cuda` stops at stage 4,
the recipe's first gate, whatever `plan` said about stage 3. Whether a relayed
backend should record spans upstream never produces is open, and is why the
evidence for these two is an emission differential rather than the recipe.

### The replay oracle, and the direction it made the gate run

`dump-replay` was the last declared slot the translator had source for.
`recast plan port --config '{"oracle":"dump-replay","dumps":...}'` answered
`[MISS]`; it answers `[ok]` now, and `recast run port` reaches a bit-exact
verdict over the material shipped in `examples/toy_physics/dumps/`.

Less of `dump_verify.py` came across than its 385 lines suggest, and the
measurement is worth keeping because it is the shape of every one of these:

| Lines | What | Where it went |
|---:|---|---|
| 83 | `parse_dump_file` -- the probe format | **here**, `recast.oracle.dump_replay` |
| 187 | `verify_from_dumps` -- load, init, call, compare | already here, `differential.bitexact` |
| 36 | `_try_init_module` -- call `*_init` by name-matching | domain extension |
| 22 | `CAM_INIT_CONSTANTS` -- gravit, rair, cpair, MG2 defaults | domain extension |
| 23 | `main` | the recipe runner |

**The migration hit a real contract question rather than a porting one.** Every
oracle the engine had answers a question the harness asks: `differential.
bitexact` generates inputs from `_SIGNATURES` and calls both sides. A replay
cannot be asked anything it was not already asked -- the inputs are whatever
the recorded run used, and the reference's outputs are recorded rather than
computed, so there is nothing to call. So the reference supplies the inputs,
and the gate had to learn to let it.

It learned it the way it learned the three f2py conventions before it: as a
declaration on the handle, `input_source: "recorded"`, rather than as a
detection. The generated path is untouched and the translate spine is
byte-identical. Three consequences are worth naming because none is obvious:

* **`trials` does not apply.** A recording holds the points it holds; asking
  for ten against a three-sample recording would be seven invented ones or
  seven copies.
* **`_PREPARE_INPUTS` is skipped.** The hook exists to drag *generated* inputs
  into the physical domain, and recorded ones are already there. It also ships
  inside the artifact under test, so running it on a replay would let the
  candidate edit the production run's own numbers before being judged on them.
* **Reference-side `setup` is skipped.** A replayed reference has no state to
  set: whatever the run's module state was is folded into what it recorded.
  An operator whose `setup` does not match the run's own initialization gets a
  difference rather than a silent pass, and that is the one thing about a
  replay this repository cannot check from here.

**One place the migration deliberately does not relay, and it is the binding.**
`dump_verify.py` matched dump names to arguments fuzzily -- exact, then with
`in`/`out` stripped, then any substring either way -- and filled whatever was
left with zeros. In a one-shot investigation that is a reasonable convenience;
in a gate it is a way to produce numbers that compare cleanly and mean nothing,
because a substring match binds `t` to `theta` and a zero fill invents an input
the run never had. The engine binds by exact name and refuses a required
argument the recording does not carry, which is what a verifier that fails
closed owes its reader.

It can refuse instead of guessing because it reads a line upstream's parser
drops. The probes write `# PROBE <module>.<sub>: call=N` first; neither of
`parse_dump_file`'s two regexes matches it, so the subprogram's identity is
lost and the script that consumes the dump has to try every subprogram in the
module. Reading it is additive in `golden_diff`'s sense -- the inputs and
outputs parse identically either way -- and it retires the guessing rather than
relaying it.

**The evidence here is the weakest of the four differentials, and the reason is
not fixable from this side.** `tools/dump_diff.py` runs both parsers over the
same dumps and compares every name, shape, dtype and value bit for bit: 12
cases, 0 differences. But the cases are *constructed*, because *no production
dump is committed in either repository* -- the recordings `dump_verify.py` was
written against were written by probes inside a CESM run and live on scratch
storage. So a green run says the two parsers agree on everything that file
knows to ask about, and nothing about a real recording. The cases are listed
rather than generated, so the count means what it says; a random-case
generator would raise the number without raising the confidence.

`examples/toy_physics/dumps/` is synthetic for the same reason and says so in
every file. Its values are `settle` evaluated in float64, which makes it a
fixture a correct candidate reproduces to the bit rather than something merely
plausible -- 3 samples, 18 points, 0 ULP, pinned by a test and checked by the
conformance suite's third `OracleCase`.

### The history that was not carried

Found 2026-08-21, while looking at what `gitleaks` had scanned: **the
`git filter-repo` pass this phase describes never ran.** All 71 commits in this
repository are the maintainer's, the earliest is its own `Initial commit`, and
no file under `src/recast/` has a commit older than the day it was written
here. "Migrated with history" was written into this document and into
`CONTRIBUTING.md` before anyone checked whether the history existed.

It did not, in any useful sense. The translator was **one commit** at the time
— `4743491`, "Initial commit: Deterministic Fortran-to-Python translation
pipeline", 2026-07-06, by Qinrun Dai (as second5t) — so a path rewrite would
have carried that single commit and nothing else. And the material was
decomposed into the plugin contract as it landed — one `main()` became a
Frontend, a Transform and three Verifiers — so no module crossed intact for a
commit to be about.

**That premise expired on 2026-08-21, and the conclusion survives it.** The
translator has since published its real history: 249 commits reaching back to
2026-07-06, `pipeline/translate.py` grown from 2,883 lines to 4,669. A rewrite
today would carry something. It still would not carry anything useful *here*,
because the second half of the argument never depended on the first: the
decomposition is what stops a commit from being about a file in this
repository, and no amount of upstream history changes that. What the published
history did change is the relay, and that is tracked where it belongs — the
differentials re-run against whatever the translator looks like next, and now
print the revision they compared against.

So the plan was wrong rather than skipped, and what replaces it is the rule
`CONTRIBUTING.md` already gives for that case: name the source in the file, and
name the author in `NOTICE`. The first half P2 shipped —
`oracle/f2py.py` and its neighbours say where they came from. **The second half
was missing until this was found**, and `NOTICE` named only the JAX backend's
author while ~10k lines of the same person's work sat unattributed beside it.
That is now a `NOTICE` entry, which is the thing that had to land before P6
makes the omission public and permanent.

Nothing about the code is in question — `emit_diff` holds the emitter to the
pipeline byte for byte across 27 modules, which is a stronger claim than any
commit graph. What was wrong was the record of whose work it is.

This also surfaced that the translator carries **no licence file** —
nor does CESM-Agent-Produced-Scripts. (Both still true, re-checked 2026-08-26.) Settled the same day rather than left to
P6: both are the maintainer's to license and all of it is Apache-2.0, the same
as here. Writing the `LICENSE` file into each goes with the archiving; the
reasoning is under P6.

## P3 — triage the 662 agent scripts (done, bar the archiving)

CESM-Agent-Produced-Scripts, every file given a destination. The plan said four
buckets; executing it needed eight, and the extra four are the findings.

| Disposition | Count | Where it goes |
|---|---|---|
| `PRODUCT` | 177 | port outputs and their unit tests → Product Layer repos |
| `HPC-EXEC` | 170 | interface here, implementation in an executor plugin |
| `ARCHIVE` | 141 | one-shot: bound to a run, a bug, or a date that will not recur |
| `P2` | 69 | already in the translator — P2 migrates it from there |
| `EXCLUDE` | 50 | stays behind: disclosure-ledger rows 6 and 7 |
| `PROMOTED` | 28 | landed in the domain extension, de-site-ified, tested |
| `POOL` | 22 | re-runnable tooling with no current need; promoted when one names it |
| `META` | 5 | tooling about the agent transcripts themselves |

The table is generated from rules rather than hand-written, and lives with its
reasoning in the domain extension's `migration/` directory. Four things the plan
did not anticipate:

**69 of the "reusable" files are already in the translator.** P2 migrates
those, so copying them in P3 would land them twice, from two directions, with
nothing saying which copy is the one under test. Whole planned areas —
golden-set generation, reference builds — belong to P2 on this finding alone.
(This argument was originally written as being about the provenance a
`git filter-repo` pass keeps. That pass did not happen; the duplication is
reason enough on its own, and is the reason that actually applied.)

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
archiving, and **coordinated with its author rather than done unilaterally** —
CESM-Agent-Produced-Scripts and the translator are a student's
repositories, not this project's. Safe to defer because the filtering happened
file by file at migration time rather than as a cleanup pass at the end, and
cheaper because P4 still has three files to take out of the collection.

Nothing downstream depends on the archiving landing on time. Work reaches this
repository only through a maintainer, so what keeps the sealed buckets sealed is
that nobody migrates them, not that the repository holding them is frozen.

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

**The refactor side (freeCAM) is deliberately deferred** until its student
tooling is ready; its entry-point slots are sketched in the extension's
pyproject and nothing is declared before it exists.

**The port side started 2026-08-20**, its tooling being ready enough to move.
Its gate landed first, because nothing the backend produces is worth reading
before something can judge it: `differential.tolerance` is a two-tier gate --
a ULP bound on the elements that dominate a row, a relative bound on all of
them -- and it is a subclass of the bit-exact gate rather than a second
harness, sharing input generation, the calls, and the ULP counting, differing
only in the policy that reads the numbers.

The tiering is not a loosening. Its own tests run one perturbation at two
indices: the same `1 + 1e-13` is `TOLERANCED` in the tail and `FAILED` in a
dominant element, and `max_rel` is `1.0e-13` in *both* -- so a plain relative
gate at 1e-12, which is what a single-tier version of this would have been,
passes the defect. What the ULP tier buys is exactly that case.

**The port side runs end to end as of 2026-08-20.** `recast run port` walks
frontend → `port.jax` → `numpy-anchor` → `differential.tolerance` → store and
reaches a verdict; on a kernel with no transcendentals that verdict is
`ULP_BOUNDED` at 1 ULP over 85 points, 76 of them bit-exact.

**And it runs somewhere other than the machine that wrote it, as of
2026-08-21.** Until then no CI job installed `jax` — every extra in `ci.yml` was
`fortran`/`translate`/`verify` — so `tests/test_port_spine.py` skipped in CI and
the port side had only ever run on one laptop, which is the same gap the `spine`
job was created to close for the f2py chain. The `port-spine` job closes it,
over `examples/toy_physics/port.json` as well as the test. It installs no
compiler, and that is the anchoring decision showing through rather than a
saving: `numpy-anchor` re-derives its reference, so nothing in the port run
builds Fortran — and the link it therefore cannot check is exactly the one the
`spine` job checks.

It gates on the verdict and does *not* diff its summary, which is where it
parts company with the `spine` job one line above it. Bit-exactness is a
device-independent claim, so `verification.json` is held byte for byte. A ULP
count is not one: XLA's CPU backend does not promise the same last bit on x86 as
on arm64. `port-verification.json` is committed for a reader without JAX, and
`candidate_device`/`reference_device` in it say which machine's answer it is.

The oracle decision went to `numpy-anchor`: the reference is the validated
NumPy translation of the same unit, re-derived from the same Facts rather than
read off the Candidate, because an Oracle that saw the artifact would stop
being independent of it. That makes the port's claim a chain — NumPy bit-exact
against the Fortran, JAX ULP-bounded against the NumPy — and the honest part is
that this oracle cannot check the first link. It records what it derived and
carries `config["anchor_evidence"]` into the Verdict when an operator has the
translate evidence to point at; absent is a legitimate answer and an
informative one. `dump-replay` stays selectable for a unit with no translation
to anchor on, and only that oracle now demands `dumps`.

**One link of the chain is missing and is deferred on purpose.** The ULP gate
measures how far two implementations land apart in float64; it cannot say
whether the lowering changed the mathematics. That is `symbolic.notary`'s
question, answered at fifty significant digits where float64's own 1e-16 noise
cannot reach, with `1e-45` as the bar for "the same function" — a different
coordinate system from ULP entirely, not a stricter setting of the same dial.
And `jaxize` rewrites. Nothing currently asks the notary about any of them,
because `port.jax` records no `Candidate.notes["rewrites"]` for it to read.

How much of the lowering the notary could cover is worth stating before anyone
starts, because it is less than the whole. The notary takes an expression and
its replacement — one formula against another over the free symbols' physical
ranges — so it reaches the expression-level rewrites: `and` and `or` becoming
`jnp.logical_and`, `math.exp` becoming `jnp.exp`, a subscript store becoming
`x.at[i].set(v)`. It does not reach the structural ones, and those are the
larger half: a `for` loop becoming `lax.fori_loop` with an explicit carry
tuple, an `if` becoming `lax.cond` over the union of both branches' assigned
names. Proving a control-flow rewrite preserves meaning is a different and
bigger question than proving two formulas are one function.

So the third link, when it is built, closes part of the gap and should say
which part. The translate recipe carries the notary as an optional stage and
the port recipe should too. Deferred, not forgotten.

Running it for real found what reading could not: the differential harness had
**three f2py conventions baked in** while claiming to be oracle-agnostic. How a
reference spells an argument (lowercased, or the emitted spelling), what it
returns for out-intent arguments (f2py's split between returned `intent(out)`
and mutated `inout`, or every out argument in declaration order), and whether
it needs `w_` wrappers at all. Each is now something the reference declares on
its handle, defaulting to what f2py does so nothing existing changed — the
translate spine is byte-identical and `verification.json` did not move. A
fourth thing was missing rather than assumed: the emitted JAX module carried no
`_SIGNATURES` table, because the script it came from was driven by hand-written
tests that already knew the interface, and the engine's gate generates its
inputs from the table. The Transform lifts it from the anchor, which keeps
`tools/jax_diff.py` green because the backend's own emission is unchanged.

**Two comparisons, and they must not be confused, because their standards are
opposite.** One is scientific: the Fortran's numbers against the JAX port's,
which cannot be bit-identical and never will be — that is what
`differential.tolerance` gates, at the ULP tier, and it needs JAX installed to
run. The other is about the migration: the code the collection's `jaxize.py`
emits against the code the migrated `port.jax` emits, over the same inputs.
That one has **no tolerance at all** — byte for byte, exactly as
`tools/emit_diff.py` holds the translator's emitter — and it needs no JAX,
because `build_module` is a pure AST transformation whose corpus is 109 suite
directories already on disk, discovered rather than listed.

The order follows from that: the migration diff comes first. If the migrated
backend emits the same bytes, the scientific comparison is preserved for free,
because it is the same code computing the same numbers — and 630 lines of
migrated AST surgery is not reviewable any other way.

**`tools/jax_diff.py` is that harness, and its baseline is measured.** Over the
109 suite directories, the collection's backend builds all 109 without raising,
emitting 203 JAX kernels and host-delegating 607 subprograms — and the reasons
divide into three families that a reviewer has to keep apart:

| | | |
|---|---|---|
| `[elig]` | 468 (77%) | not eligible: derived types, module-state writes, string arguments. By design — kernel eligibility mirrors the Numba backend's, and none of it is a limit of the JAX emitter |
| `[emit]` | 93 (15%) | the emitter's own subset: `while`, `Raise`, `Try`, a return inside a loop. This is the number that should fall as the backend grows — **upstream**, in the author's own repository, reaching here as a relay like the backend itself did. Not work this repository picks up |
| `[anchor]` | 46 (8%) | the NumPy module it was handed still carries `AGENT_QUEUE` placeholders. The anchor is incomplete, so there is nothing to port |

Only the middle row is the JAX backend's to improve, which is worth knowing
before anyone reads "607 delegated" as a coverage problem. Whose improvement it
is, is worth knowing too: the backend is developed in its author's repository
and relayed here, so that number falls when a widened `jaxize.py` is merged
across, not when someone edits `recast/transform/jax/` directly. Which is what
keeps `tools/jax_diff.py` useful rather than what retires it — see the note on
the mypy override in `pyproject.toml`. The harness compares
emitted pieces byte for byte, kernel sets and delegation placement exactly, and
delegation *reasons* only as far as their category, because the tag is a
decision and the tail is a diagnostic. It needs no JAX — verified, the survey
runs with JAX not installed — and its own tests plant one difference at a time,
including one that must not be reported. That last one is a
decision, not a port: the tooling anchors on the validated `*_numpy.py`
module, while `PortRecipe` declares `dump-replay`. The NumPy module is itself
bit-exact against the f2py oracle, so anchoring on it is a chain of trust
rather than a weaker one -- but it is a different oracle from the one the
recipe names, and the recipe or the anchor has to give.

P3 located the port side's tooling. `jaxize`, `jax_shim` and `coverage_sweep`
sit in CESM-Agent-Produced-Scripts and nowhere else, and they are what fills the
slot this engine has already declared and not implemented — `PortRecipe`'s `jax`
backend and the `recast.port.kernel-to-jax` named in `plugins/transform.py`.
`jax_shim` is specifically the JAX counterpart of the `_f_*` anchors
`transform/numpy/runtime.py` ships, so the two backends can be held to the same
anchors rather than drifting into separate notions of correct.

### Three things the PRODUCT pass handed back, 2026-08-22

The pass over the 177 files P3 filed as port *outputs*
(the extension's `migration/product-177.tsv`) found 143 already in the kernel
repositories and nine that were tooling; the rest of what it found is the
engine's, and none of it is a file to copy. Each is an item here until it is
code.

1. **Intel's libimf as a third libm -- relayed 2026-08-22.** The translator
   had already done it: under its ``ifx`` profile every transcendental is
   spelled ``intel_math.*``, a ctypes binding to libimf kept beside the
   output, because ``ifx`` links that library and its ``exp``/``log``/``pow``
   are an ULP from glibc's on some arguments. ``Profile`` now carries
   ``intel_math``, the vocabulary carries the override tables, and
   ``transform/numpy/intel_math.py`` is the binding -- loaded at first call
   rather than at import, which is the one departure, so an emitted module
   can be read anywhere and refuses, naming ``RECAST_LIBIMF`` and the
   ``gfortran`` profile, only when asked for a number. The ifx output needs
   a libimf to run; that is a library dependency by design, and the
   gfortran profile is for machines without one. The two dump-bound
   scripts stay with the translator.

2. **One test per lowering rule.** Seven tests migrated to the
   extension's `tests/port_jax/` exercise rules that live in
   `transform/jax/backend.py`: `while` to `lax.while_loop`, `break` to a
   `_brk` flag with in-list guards, a valued early return beside a terminal
   one, INOUT rebinding where NumPy mutated in place, `endrun` as an error
   channel, statement functions as closures, log-only-branch stripping. They
   exercise them on CESM modules, so they skip without a suite on disk.
   `tests/test_jax_transform.py` has seven tests and none is about a
   construct: they check that one `apply` yields both halves, that
   delegation is not deferral, that the artifact reproduces. Each rule
   above wants a synthetic Fortran fixture here, the way
   `tests/test_numpy_*` are built, so the backend's coverage is checked in
   the engine's own tests rather than inferred from a private suite.

3. **Sequence association is mistranslated.** Recorded in the docstring of
   the migrated `test_pkg_cld_sediment_jax.py`: the Fortran actual
   `xxk(1,k)` -- an array element passed as the start address of a
   `pcols`-long dummy -- is emitted as the scalar `xxk[0, k - 1]`, and the
   callee indexes it. The NumPy translation is wrong before JAX is reached,
   so both backends crash on `cfint2`, and the test covers the two
   subprograms it can. This is `transform/numpy`'s: the frontend knows the
   dummy's rank and the actual's, and an element actual against an array
   dummy is a slice from that element, `xxk[0:, k - 1]` in column order,
   not an element. A fixture with exactly that call is the test.

### The public corpus, 2026-08-22

The done-when below now has a public form. `corpus/` pins twelve
open-source Fortran libraries as submodules -- least squares, quadrature,
special functions, roots, splines, an optimizer, FFTs, a cloud-microphysics
kernel -- and `tools/corpus.py` walks the `translate` recipe over each with
no extension installed, recording per unit what the rules refused and why.
`corpus/baseline.json` is the record and the work list, and it settles a
question the relay kept raising: a rule goes into the engine when code
nobody here wrote needs it, and into the extension when only CAM does.

The first baseline said two things above everything else, and one of them
has since turned out to be wrong about this repository. Files of bare
subprograms are not units at all -- fifty-eight of fftpack's fifty-nine
files, and CLOUDSC's kernel itself -- which stands, and is not a CAM
question, so it was never asked.

The other said calls between sibling modules of one tree are refused
"because the recipe takes companions only from config, where the
translator's `auto_translate` derives them from USE". **That is not what the
code does.** `FortranFrontend._companions` walks this unit's USE statements
against a module index of the tree, transitively through a bare `use`, and
hands what it finds to the transform; `fortran-utils`' `linalg` emits
`import constants_numpy as _constants` and `import types_numpy as _types`
and both sit beside it. The mechanism is there and it works, so there was
never anything to relay here. What is left is smaller and different, and
naming it wrongly cost a re-derivation:

Of the eight units whose emitted Python still does not import, **seven fail
on an import of a module nothing in the file binds to** -- the alias appears
on its import line and nowhere else. Three separate reasons put the module
out of reach, and the last step is the same in all three: a `use` that
resolves to no companion emits `import <mod>_numpy as _<mod>` whether or not
anything needed it.

| units | the module | why it is out of reach |
|---|---|---|
| cloudsc's four, `fortran-utils:special`, `splines` | `file_io_mod`, `amos`, `lapack` | not in the case's file set in `cases.json`, so not in the tree |
| `fftpack` | `fftpack_kind` | its source is in the tree and does not parse: `rk.f90:3` is `implicit none(type, external)`, which is Fortran 2018, and `_parse.STD` is `f2008`. The companion walk recorded exactly that in `unresolved` |
| `fortran-utils:linalg` | `lapack` | nothing is wrong with it: `_lapack.ilaenv(...)` is a real call into LAPACK, which is a library and not a translation |

**The read/write check was also wrong about two things, and both were the
verifier rather than the translation.** `scope_for` keyed its subprogram
table by `subprogram_key`, which qualifies an internal procedure with its
host, so a call spelled with the bare name looked like an array element being
read -- the pipeline this was migrated from keys the same table by the bare
name, and the divergence came in with the port. And the emitter's own
temporaries were being counted as data: `_out`, which holds a multi-output
call's tuple for one statement before it is unpacked, and the `except ... as`
bindings `_g`/`_be`/`_lc`/`_le`. Together those were 29 of the 450 disagreeing
blocks, and they were enough to take `numfor`'s `sorting` from stopped to
passing every stage -- 80 points, all bit-exact. The corpus now has two units
that reach the differential rather than one. Nothing about the emitted bytes
changed; `emit_diff` is unmoved at 2. What changed is that the check is no
longer wrong about a translation that was already right, which is the failure
mode a fail-closed verifier has to be watched for.

Only the last is a dependency in any meaningful sense. The unconditional
import is shared with the reference pipeline -- `translate.py` emits one for
every `auto_stub_modules` entry with no test of whether the alias is used --
so it is an upstream finding to report rather than a rule to write here.

**And it did what it was written to do the first time a relay was measured
against it.** The five translation defects relayed from the translator move
`imports` from 37 of the 59 units to 51, and `parses` from 58 to 59, with
`mechanical`, `rwset`, `oracle`, `bitexact` and the deferred-block total all
unchanged: fourteen units that could not be loaded now load, and nothing that
worked stopped working. That is the whole of the evidence for those five,
because the three emission differentials cannot see them -- the defects fire
on the corpus libraries and not on CAM, which is exactly the case the corpus
was pinned for. The sixth, the dispatch axis, is the reverse case and shows
the two gates are not redundant: it moves all three CAM differentials *and*
takes the corpus's deferred-block total from 377 to 366, because a generic
that cannot be resolved is a refusal wherever it appears. It also caught the one thing the relay got wrong on the way
in, and caught it as a *drop*: binding `use, intrinsic :: iso_fortran_env`
correctly changed `stdout` from a bare name the module never defined into
`_iso_fortran_env.output_unit`, and the read/write check reported numfor's
`basic` as disagreeing until the alias was declared to it. A verifier that
does not know about a new binding reports the translation wrong rather than
saying it cannot tell, which is the right direction to fail in.

**Done when:** the engine passes its tests with the CESM extension
uninstalled, and freeCAM's validation gate runs through `Verifier` rather than its own
`validate_*` scripts. This phase is the only real proof that the engine is
domain-independent.

## P5 — prove the contract out of tree

Build the first extension that lives outside this repository and depends only on
the published contract: PBS/Slurm executors, relay/resume of multi-day runs, and
a restricted `FindingStore` for Sec-Track. It passes `conformance/`.

That last sentence needed a `conformance/` that runs, and until now there was
none: the directory held a specification whose own status line said the harness
would land in P2, and P2 closed without it. It runs now, over seven of the ten kinds --
including the two P5's extension is made of, `Executor` and `FindingStore` --
plus the recipe-level rules and the runner's no-retry guarantee. An extension
names its plugins to the suite by declaring a `PluginSet`, so a check that needs
a candidate, a source tree, or an oracle can be handed one;
`recast/conformance/builtin.py` is the set the engine holds itself to, and CI
runs it. `Scanner`, `Adjudicator` and `AgentProvider` are what remain, and none
of the three has an implementation anywhere yet, so their checks wait for the
first one of each rather than being written against nothing.

It has already earned its place twice. The rule that an executor's refusal must
reach the runner as a `RecastError` turned out not to hold for `f2py-golden`: it
let a bare `RuntimeError` out, which `run.py` does not catch, so one build that
could not be scheduled ended the whole run and cost every other unit its
verdict. `OracleUnavailable` had been declared since P1 and raised nowhere. And
the rule that a Frontend reads source rather than the engine's own output turned
out not to hold either: `run_recipe` writes its workspace into the tree it was
pointed at, the f2py oracle leaves compilable wrappers there, and `SKIP_DIRS`
did not list that directory -- so a second run over a tree discovered the first
run's scaffolding as a unit to translate. Both are the shape of finding this
phase is for, arriving one phase early because the checks were written before
their subjects were examined.

One rule stays deliberately unexercised: no Verifier in this repository submits
a job, so nothing here can route around a refusing executor. `f2py-golden` is
the plugin that compiles through one, which is where that rule is checked today,
and P5's batch-backed Verifier is what will meet the other half. The check skips
by name rather than passing, which is the arrangement that makes it useful on
the day something does execute.

### Started 2026-08-21, on the cyber half rather than the batch half

The phase was read as blocked on HPC access, because its first sentence names
PBS/Slurm. It is not: `conformance/test_executor.py` genuinely submits a probe
job and waits on it twice, so an executor needs a scheduler to be checked — but
the *other* kinds P5 names need nothing but a scratch directory, and the phase's
question is about the contract, not about which kind asks it.

So the first out-of-tree extension is `recast-sec`: the cyber half of CC-Test
(`a85tract/CESM-CC-Test`, by Chien-Wei Huang, already in production on
Derecho as `hpc-devsecops`) wired to the plugin contract. A real
`secret` Scanner over gitleaks, SARIF to `Finding`, plus two stubs that exist
only so the `audit` recipe reaches the stages nobody had reached before. It
registered through `recast.scanners` and `recast.adjudicators` with **no engine
change at all** — which is the half of the contract that works, and worth saying
before the half that does not.

`recast run audit` then produced this, and it is the reason the phase exists:

    [ok ]  frontend    fortran
    [skip] scanner     secret         kind 'scanner' not walked
    [skip] scanner     composition    kind 'scanner' not walked
    [skip] adjudicator adversarial    kind 'adjudicator' not walked
    [ok ]  store       fs-findings    0 verdict(s) recorded
    1 unit(s), 0 verdict(s), all passed

A security audit that scanned nothing, reporting **all passed**. The stubs raise
`NotImplementedError` on entry and neither raised, which is the proof that the
runner never called them rather than an inference that it did not.

Seven findings, five of them holes:

1. **`run_recipe` demands an executor stage from every recipe**, so `audit` is
   unrunnable as shipped. `Stage`'s own docstring says only "a recipe that
   materializes an oracle or awards a verdict has to declare one", and
   `conformance/test_recipes.py` skips its executor checks for `audit` on
   exactly that reading. The runner is the one out of step, in two places — the
   guard, and a later use of `executor_stage.plugin` for the evidence record.
2. **`scanner` and `adjudicator` stages fall through `_walk_stage` to
   `"skipped"`**, the same status an uninstalled optional plugin gets. Installed
   and absent are indistinguishable in the output.
3. **A `gate=True` stage is skipped rather than enforced.** The `audit` recipe
   gates on its adjudicator; the run passed without it. A gate that can be
   skipped is not a gate, and nothing in the runner ties `gate` to the kinds it
   does not walk.
4. **`fs-findings` is walked as an `EvidenceStore`.** The store branch iterates
   `unit_run.verdicts` and calls `put(evidence)`; `UnitRun` has no findings
   field at all. The `audit` recipe's terminal stage is structurally wrong, not
   merely unimplemented — Findings have nowhere to accumulate.
5. **A Scanner cannot report that it could not run.** `scan` returns an
   iterable; gitleaks missing yields an empty one, which is what a clean scan
   yields. `OracleUnavailable` exists for exactly this and has no counterpart
   here. `hpc-devsecops` distinguishes `PASS` from `INCOMPLETE` and the
   contract cannot carry the difference.
6. **A Scanner's subject is not always a Unit** — nor does it receive an
   `Executor`, so one that shells out to gitleaks has to bypass the seam the
   engine enforces everywhere else. The same question from the other side:
   the contract says nothing about where or on what a scanner runs.
   Originally stated as: a Scanner's subject is not always a Unit. `scan(unit, facts, ...)`
   assumes a defect belongs to an addressable piece of software. gitleaks' value
   is in *history* — a credential deleted in a later commit is still in the pack
   — and `syft`/`grype` describe *whole-repository* state, which the CC-Test
   script comments on deliberately. Per-Unit is the weakest reading and the only
   one available; scanning the repository once per Unit is worse. The contract
   gives a scanner no way to say what it is the scanner *of*.
7. Paper cut: the `Adjudicator` ABC ships in `recast/plugins/scanner.py`. Its
   kind is `adjudicator` and its entry-point group is `recast.adjudicators`, so
   the obvious import is the one that fails.

None of this was found by reading. Findings 1 through 4 needed a throwaway patch
to `run.py` to get past each previous one, reverted rather than committed,
because the patch is not the deliverable — the list is. Nothing in this
repository changed to produce it.

### Findings 1–4 closed, 2026-08-21

All four were in `run.py`, and closing them is one commit. The runner now walks
`scanner` and `adjudicator` stages, picks a store's branch from the store's own
kind rather than from the stage's position, and asks for an executor only from a
recipe that materializes an oracle or awards a verdict — the condition
`Stage`'s docstring already stated and `conformance/test_recipes.py` already
read. `recast run audit` reaches every stage it declares.

Four decisions in that are not forced by the findings, and are the part worth
arguing with:

- **A confirmed finding fails an adjudicator declared as a gate.** The
  alternative — report and pass — is what the hole produced, and an audit that
  says what it found and passes anyway is a report. `hpc-devsecops` exits
  non-zero on findings; this is the same answer.
- **A stage kind the runner does not walk is refused before the run starts.**
  Not skipped during it: `skipped` is the word an uninstalled optional plugin
  gets, and reusing it is what made installed-and-unwalked invisible in the
  first place. The refusal names the kind, and runs before the
  plugins-are-registered check, because a kind nothing has heard of has no
  registered plugins either and would otherwise be reported as a missing one.
- **`agent` and `recipe` are refused as stages rather than ignored.** They are
  registered kinds, so a recipe can name one; neither is a step. An `agent` is
  consulted by a non-deterministic Transform, and a `recipe` is what the thing
  *is*. Declaring either is a misunderstanding worth a message.
- **The run summary still says nothing about findings, including how many.**
  That file is written to be committed. A `Finding` defaults to
  `Access.EMBARGOED`, and a count is a statement about embargoed material. The
  `FindingStore` holds them, at `0700`.

Walking the store stage for real also turned up something that was not on the
list, because nothing had ever reached that stage to turn it up: the runner's
default put embargoed findings at `<project>/.recast/findings`, and `.recast/`
was not in this repository's `.gitignore`. The store's permission check passes
on a `0700` directory inside a checkout, so the one control in front of the
accident its own docstring names did not cover it. The default now resolves
outside the project and the store refuses a root inside a working tree whoever
chose it — see row 5 of the disclosure ledger, amended rather than merely
satisfied. `.recast/` is ignored now too, but that is housekeeping and the row
says so: an ignore rule is one line anyone can delete.

**All seven are closed as of 2026-08-21.** 1–4 were a runner that did not
implement a contract everything else had already agreed on. 5, 6 and 7 were the
contract being wrong, which is a different kind of work and the kind P5 exists
to find; what each one took is below.

### Finding 5 closed, 2026-08-21: a third run state

A Scanner can now say it could not run. `ScannerUnavailable` is the counterpart
to `OracleUnavailable`, raised by a `Scanner` or an `Adjudicator` whose tool is
not on PATH or whose model has no key, and the stage it produces is
`incomplete` — a fourth word beside `ok`, `failed` and `skipped`, kept distinct
from `skipped` because an absent optional plugin is a declaration the operator
made and this is a plugin that was installed, asked, and could not answer.

A run therefore reports one of three states rather than a boolean, and the
choice between the available designs is the part worth recording. Folding
`incomplete` into `failed` was rejected: the `audit` recipe's LLM scanner is
optional by declaration, and failing every run on a machine without an API key
teaches operators to delete the stage, which makes the recipe lie instead of the
run. Folding it into `passed` is the defect itself. So: three states, `passed`
False for all but one of them — anything already gating on `passed` keeps gating
without being told this enum exists — and two distinct non-zero exits from
`recast run`, so a caller can tell "fix the code" from "fix the machine" without
parsing output.

Waivers are `config["allow_incomplete"]`, and every restriction on them is
there because the unrestricted version reintroduces the bug. The stage still
reports `incomplete (waived)`, since a waiver that edited the record would be
indistinguishable from the stage having run. A name no stage declares is
refused, because a waiver matching nothing reads as coverage — the failure the
disclosure ledger warns about for hygiene patterns, one system over. And a
`gate` may not be waived, because a gate that can be absent from a passing run
is not a gate.

The verification summary says nothing about any of it, deliberately. Whether
gitleaks is installed is a fact about the machine, and that file is the one
place that leaves the machine out so its diffs mean something. Incompleteness
belongs to the status, the exit code, and the Evidence manifest.

### Findings 6 and 7 closed, 2026-08-21: what a scanner is of, and where it runs

`Scanner.subject` is the answer to 6. `"unit"` is the old behaviour; a scanner
that declares `"repository"` is walked once per run against a Unit the runner
synthesizes for the tree, and that Unit goes through the recipe's adjudicator
and store stages like any other. The alternative — a run-level findings list
threaded through every later stage — was rejected because it would have made
repository findings a second kind of thing everywhere downstream, and the
engine's status, gating, storage and summary logic are all per-Unit. Making the
tree a Unit costs one synthesized `Unit` and buys the whole pipeline unchanged.
The in-tree gitleaks scanner is the first user: `gitleaks git <root>` over
history when the root is a repository, `dir` when it is an export, and a
subdirectory of a repository gets `dir` on purpose so that pointing the engine
at a subtree does not quietly scan the enclosing history.

The other half of 6 — a scanner has to shell out and the contract gave it no
`Executor` — is closed by giving it one. `scan` and `adjudicate` take the same
executor Oracles and Verifiers take, the `audit` recipe declares one (`local`
by default, from config like every other), and the runner's executor
requirement now reads "any stage kind that is handed one", which is the
contract's list rather than the runner's guess. The `subprocess` call that was
the only sanctioned one outside `executors/local.py` is gone.

`composition` followed the same day, and is the second user of `subject`:
syft to an SBOM, grype over it with `--add-cpes-if-none` and the repository's
`.vex/openvex.json` when there is one, Critical and High matches as Findings --
exactly the three steps and the two counted severities of `hpc-devsecops`
lines 120–140, which also carry the comment that settled the scope question:
"dependency analysis intentionally describes the resulting repository state,
rather than only the patch". A match sets `Finding.upstream` to the dependency,
since the defect is theirs to fix and ours to ship. Run for real the same day
with syft 1.x and grype 0.117.0 over this repository: 84 packages in the SBOM
(56 from the tree, 28 from `.venv`, which `dir:<root>` includes exactly as
`hpc-devsecops` would), 0 matches at any severity, so the scanner's `0
finding(s)` was checked against the tools' own output rather than taken on
trust. The `audit` recipe now runs
from the engine alone up to its gate, which is the adversarial adjudicator, and
that is not CC-Test's to port: its original is a refute-prompt and a verdict
schema in Sec-Track's discovery-loop scripts, which makes it the engine's first
real `AgentProvider` consumer and a decision of its own.

The range and the hook came after, and they are how `hpc-devsecops` is
actually used: not a full-history scan on demand but a pre-push hook that
computes `<remote-sha>..<local-sha>` per ref and scans exactly what the push
would publish, blocking on a finding and on an incomplete check alike.
`config["range"]` reaches every scanner as a fact about the invocation;
`secret` turns it into `--log-opts`, `composition` ignores it because the
dependency state is whole-repository regardless, by the script's own comment.
`tools/pre-push` and `tools/install-hooks.sh` are the hook and its installer,
checked by driving a real `git push` into a bare remote with `recast` faked on
PATH. One divergence is deliberate and documented on the flag: `hpc-devsecops`
defaults to report-only and blocks on `--block`, while `recast run` defaults to
blocking and offers `--report-only`, because `run` is what CI and the hook call
and both need the exit status to mean something. Its `--staged` and
`--worktree` modes — a diff on gitleaks' stdin — are not here; the executor
contract has no stdin, and growing it for one scanner's second mode waits for
something else to need it.

**Placement of the other two families, decided by the maintainer 2026-08-21.**
`dynamic.asan` goes to the domain extension: `CC-Test`'s `tools/asan.sh`
hardcodes `ifx`/`icx` and a module load, and which compiler a CAM build expects
is domain knowledge, so only the extension can say. `audit.llm` goes there too,
for a different reason — it is an advanced capability that stays out of the
public repository; and `CC-Test` could not have supplied it anyway, since the
script only invokes `$REPO/.github/scripts/ai_audit.py`, which lives in the
audited repository. Neither is named by the public `audit` recipe, not even as an
optional slot — the maintainer's follow-up the same day, on seeing `audit.llm`
in `recast plan`: a stage name is part of the repository, and a public recipe
advertising a slot for a capability it does not ship is the thing the rule is
about. The extension carries its own recipe for them, as it does for
`translate-cam`.

**The `audit` recipe is CC-Test's shape now, decided by the maintainer
2026-08-21.** It had gated on an `adversarial` adjudicator, which turned out to
be two repositories spliced: the scanners are `hpc-devsecops`'s daily gate, the
adjudication is Sec-Track's research loop — LLM agents told to refute each
finding, with a CONFIRMED / PLAUSIBLE / DOWNGRADED / REFUTED schema — and the
daily gate never used it. It is LLM-driven, so by the same rule as `audit.llm`
it belongs in the domain extension, and a public recipe gating on a plugin the
public cannot install is a recipe the public cannot run. So the scanners are
the gates, at each one's own `blocks_on` (`secret` on anything, `composition`
on Critical, exactly the script's two bars), every check runs before anything
blocks, and the engine keeps the `Adjudicator` contract with no implementation.
`recast run audit` runs end to end from the engine alone for the first time.

7 is `recast/plugins/adjudicator.py`. `from recast.plugins.adjudicator import
Adjudicator` works, which is the whole fix.

And the preflight, owed since finding 5: a scanner declares `tool`, and
`recast plan` reports a binary that is not on PATH beside the stage, marked
`????` and counted as unavailable, before the run has read anything.

`recast-sec`'s gitleaks scanner is the consumer, and its old `return []` carried
a comment saying there was no way to report this and that raising was not in the
contract either. Both halves of that are now false, which is what closing a
contract finding is supposed to look like from the outside.

Not done here, and worth doing next to it: nothing preflights. `recast plan`
could say gitleaks is missing before the run rather than three stages in, which
is this repository's stated preference everywhere else.

### What reading the original said about all of it, 2026-08-21

The three states were designed here from the symmetry with `OracleUnavailable`,
and then `hpc-devsecops` turned out to have had them all along: `passed` /
`findings` / `incomplete`, exit `0` / `1` / `2`, written down as a contract in
its `SECURITY.md`. The design was not new; it was **lost in the port**, which is
the second time this phase that the shell prototype turned out more honest than
the abstraction built from it.

It disagreed in one place, and the engine was changed to match. `incomplete`
outranks findings there and did not here — see "Passed, incomplete, failed" in
`architecture.md` for the argument. The same reading found that
`recast-sec`'s SARIF converter returned `[]` for a report that would not parse,
which that same contract puts in the exit class of a missing tool.

**Standing rule, set by the maintainer 2026-08-21: where migrated code disagrees
with a source repository, the source is right.** Stated first in 2026-08 about
the translator, and now general — it covers
`CESM-Agent-Produced-Scripts` and `CC-Test` too, and so anything built from
their designs. Their answers have been run against real gates: bit-exact CESM
cases for the pipeline, production use on Derecho for the security gate. Nothing
in this repository has. A difference is a bug in the migration until shown
otherwise.

Two things came in from `CC-Test` as a result, neither of them by copying its
code — the shell was never ported, and the pieces below are Python that already
existed here or was written for this:

- **`recast/sarif.py`.** SARIF is what security tools already speak, and
  translating it is identical for every scanner that wraps one, so it was going
  to be rewritten once per plugin. It moved out of `recast-sec` and gained the
  two refusals above.
A boundary was drawn at the same time, on the maintainer's instruction that
anything domain-specific in the security distribution belongs in the domain
extension. Checked rather than assumed: `check_hygiene.py` passed clean over
that tree, and its only mentions of CESM or a site were attribution. Nothing
CESM-specific was there to move — and once that was established, the
maintainer's next question was the right one: then why is it a separate
distribution at all? It is not, any more. **`recast-sec` is gone, folded in as
`recast/scan/` the same day**, on the argument `plugins/scanner.py` had been
making since it was written — a check that runs against any git repository is
engine territory, the way `recast/fortran/` is. What it had been for, the P5
probe, it had finished: the findings are above. The `audit` recipe it was the
only way to run is now runnable from the engine alone, up to the two plugins
nobody has written, which `recast plan` refuses by name. The stubs did not come
in; a registered plugin that raises on entry is worse than an absence the
runner can report.

The line itself survives the move, at the top of `recast/scan/__init__.py`,
three ways matching "Engine, extension, product": what runs against any git
repository is `recast/scan/`; what needs to know it is a climate model goes to
the domain extension; what needs to know which machine it runs on is an
`Executor` and belongs to neither.

The case that will test it is `dynamic.asan`, unwritten. `CC-Test`'s
`tools/asan.sh` hardcodes `ifx`/`icx` and a module load, and `hpc/asan-cam.pbs`
is PBS plus CAM. Only the middle is a Scanner — build with `-fsanitize=address`,
run, turn the ASan report into Findings — and the other two thirds have homes
already.

- **`recast/conformance/fake_tool.py`.** `CC-Test`'s `tests/run.sh` fakes
  gitleaks, syft and grype *on PATH* and asserts on the gate's exit codes. That
  is a better test than faking the plugin, which replaces the code under test:
  faking the tool leaves the argv, the subprocess and the report parsing in the
  run. The technique is theirs; the pytest fixture is new. `ScannerCase` and
  `conformance/test_scanner.py` are what make it reachable. With the scanner
  in-tree the builtin plugin set declares it, so those four checks run in this
  repository's own suite against the real gitleaks wrapper, no longer skipped
  as unexercised.

**Done when:** the engine works without it, and it needed no engine patches. Any
patch it did need is a hole in the contract, and the hole is the finding.

## P6 — public

Scrub → security review → archive the two source repositories read-only, with
their author, since both are a student's → flip
visibility → check that the SciRecast site's links resolve.

**The licensing of the relayed work is settled, 2026-08-21.** Neither source
repository carries a licence file, which was raised here as something to ask
the author about. It is not a question: both repositories are the maintainer's
to license, and the answer for all of it is Apache-2.0, the same as this
repository. Recorded rather than left implicit, because "no `LICENSE` file in
the upstream" is exactly the observation that looks like an open problem to
whoever notices it next, and it should cost them one paragraph rather than
another round of asking.

**Extended 2026-08-21 to `CC-Test`**, on the same terms and for the same
reason. It has no `LICENSE` file either, it is the maintainer's to license, and
the answer is Apache-2.0. Recorded here rather than left to be noticed again by
whoever reads that repository next. The scanners built from its design now live
in this repository as `recast/scan/`, so the attribution is owed here and
`NOTICE` carries it — a design relay rather than a code one, and the entry says
which.

What remains is clerical and belongs with the archiving: put the `LICENSE` file
into the two source repositories that will be public — the translation pipeline
and `CC-Test` — before they go read-only, so the record does not depend on
anyone remembering this note. The agent-produced scripts collection is not
going public (maintainer, 2026-08-21), so it needs no licence file; what was
promoted out of it is licensed where it landed. The attribution half is already done —
see "The history that was not carried" under P2, and the `NOTICE` entries it
produced.

There is no pointer to repoint. SciRecast is a Jekyll site rather than a
submodule umbrella, and its `index.md`, `engine.md` and `contribute.md` already
link `a85tract/RecastEngine` by URL — links that 404 for everyone outside the
org until the flip and start working the moment it happens. So the last step is
a verification, not an edit, and it has to include the deep ones:
`contribute.md` points into `src/recast/plugins/`, `docs/writing-a-plugin.md`
and `conformance/`. A deep link is how a rename gets discovered, by someone
else, after the repository is already public.

### What the security review is

Named in the arrow above since this phase was written, and defined nowhere.
`SECURITY.md` is about vulnerabilities reported *in* this engine and
vulnerabilities found *by* it; neither is a review of this code before it goes
out. A gate that is one word in an arrow chain, in a repository whose P0 says
an agreement nothing verifies is not an agreement, is the defect the ledger
exists to prevent, on the phase where it costs the most.

It is not a generic audit, because the generic finding would be a false one:
this engine compiles and runs other people's code **on purpose**. `oracle/f2py`
shells out to a Fortran compiler, `verify/bitexact` imports the module under
test, `transform/` writes the Python that then gets imported. "It executes
untrusted code" is the product, not the bug.

So the review is about *boundaries*, and its scope is where one is
load-bearing:

- **Operator config reaching a process argument.** `oracle/f2py.py` puts
  `config["fflags"]` into `--f90flags=` and invokes meson/ninja. What else in
  a config reaches an argv, and what a value that is not a flag does there.
- **Generated code entering this process.** `verify/bitexact.py` imports
  candidate and reference by path. Those the engine wrote; an oracle's
  compiled artifact is a build product of operator-supplied source.
- **Codegen itself.** `transform/` emits Python from Fortran, `transform/jax/`
  by AST surgery. A source construct that escapes its emitted representation
  is the injection to look for.
- **`FindingStore.guard()`.** Ledger row 5 rests on it. A guard the ledger
  cites is one the review runs at rather than reads.
- **The agentic seam.** A `deterministic=False` Transform consults an
  `AgentProvider`. Nothing implements one, so the question is what the
  *contract* permits, not what an implementation does.
- **Executors.** `executors/local` starts processes; P5's will submit jobs to a
  scheduler under someone's allocation.

**Passing is not "no findings".** It is that every surface above has either a
boundary stated in the code and a test that holds it, or a written decision
that it is deliberately unbounded and that the operator is the trust boundary.
A surface nobody reached a conclusion about fails. Recording a conclusion *not*
to act is the point, exactly as with the ledger's cleared rows. Defects found
are fixed before the flip; `SECURITY.md`'s private-advisory route is for after
it.

**Done when:** five clauses, one per step of the arrow above. The check on this
phase used to be the first of them alone — which is a precondition, not the
work, and it was already satisfied while the repository was still private,
unscrubbed and unreviewed. A phase whose check can pass before its steps run is
not checked.

1. **The ledger is clear.** `docs/disclosure-ledger.md` has no open case, and
   every settled one that claims protection names a mechanism that is actually
   in place — the pattern in `check_hygiene.py`, the path off the migration
   manifest, the record class guarded. A row whose mechanism is still prose
   does not count. A row that was examined and *cleared* names no mechanism,
   and that is the point: it holds nothing because there is nothing left to
   hold, and the record of what was considered and dropped is what makes the
   ledger auditable rather than long.
2. **The scrub has run on a runner, not on a laptop.** Both jobs of the
   `hygiene` workflow green on GitHub: `check_hygiene.py` over the tree, and
   `gitleaks` over full history at the version the workflow pins.
   `tools/ci_local.sh` is a rehearsal and says so — it reports the local
   scanner's version precisely because it is not the pinned one.
3. **The security review has happened and is written down**, to the standard
   above: every surface concluded on, in a document that outlives whoever ran
   it.
4. **Both source repositories are archived**, read-only, each carrying the
   `LICENSE` file it does not have today.
5. **The flip, and then the links.** Every link in SciRecast's `index.md`,
   `engine.md` and `contribute.md` that points at this repository resolves —
   including the deep ones into `src/recast/plugins/`,
   `docs/writing-a-plugin.md` and `conformance/`.

Clause 5 is the one still backed by prose, and is marked rather than hidden:
there is no `tools/check_links.py` and the check is somebody remembering to
click. Clause 2 is not satisfiable today for a reason outside this repository —
both workflows are `workflow_dispatch` only while the Actions allowance is
exhausted, so nothing has run on a runner since 2026-08-19.

**Clause 3 is met: the review is written down, as of 2026-08-21.**
`docs/security-review.md`, to the standard above — every surface concluded
on, four defects fixed with a test each, the operator-is-the-boundary
decisions recorded per surface so nobody re-finds them. Three of the four
defects were the code under examination reaching a place only the operator
should: a finding's uid used as a filename, a Fortran character initializer
emitted as Python source, a declared extent handed to `eval`. All three were
in paths written or first exercised this week, which the review takes as its
own argument for being repeated whenever a new kind of input starts flowing.
Two influences of the audited repository on its own audit — its VEX and its
`.gitleaks.toml` — are kept as `hpc-devsecops` has them and recorded as
decisions rather than defects.

**Clause 1 is met: no case is open, as of 2026-08-20.** The one that was — the
extension's name in git history, which would have needed a rewrite of every
hash in the repository — is cleared rather than executed, because the fact it
would have hidden is published on the project's own public site. The reasoning
is ledger row 9, and it is worth reading before this phase runs, since it is
the only place a decision was made *not* to act. Clauses 2 through 5 are not
met, and none of them was being asked about until now.

**Irreversible.** Both source repositories are private and carry NCAR paths, a
username, an allocation account, PBS vulnerability research, and CPG audit
entries. Filtering at migration time (P2/P3) rather than before the flip is what
makes this safe, because git history keeps whatever was ever committed — which
is also why the ledger is written as the cases turn up rather than assembled
here. By the time this phase runs, the material that would populate it has
already been moved or left behind, and the decisions are months old.
