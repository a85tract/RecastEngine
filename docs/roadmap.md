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

## P2 — migrate the translator (done)

Move CESM-language-translator's `pipeline/` (22 modules, ~10k lines) in,
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

### The history that was not carried

Found 2026-08-21, while looking at what `gitleaks` had scanned: **the
`git filter-repo` pass this phase describes never ran.** All 71 commits in this
repository are the maintainer's, the earliest is its own `Initial commit`, and
no file under `src/recast/` has a commit older than the day it was written
here. "Migrated with history" was written into this document and into
`CONTRIBUTING.md` before anyone checked whether the history existed.

It did not, in any useful sense. CESM-language-translator is **one commit** —
`4743491`, "Initial commit: Deterministic Fortran-to-Python translation
pipeline", 2026-07-06, by Qinrun Dai (as second5t). A path rewrite would
have carried that
single commit and nothing else. And the material was decomposed into the plugin
contract as it landed — one `main()` became a Frontend, a Transform and three
Verifiers — so no module crossed intact for a commit to be about.

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

This also surfaced that CESM-language-translator carries **no licence file** —
nor does CESM-Agent-Produced-Scripts. Settled the same day rather than left to
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
| `P2` | 69 | already in CESM-language-translator — P2 migrates it from there |
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
CESM-Agent-Produced-Scripts and CESM-language-translator are a student's
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
6. **A Scanner's subject is not always a Unit.** `scan(unit, facts, ...)`
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
is not in this repository's `.gitignore`. The store's permission check passes on
a `0700` directory inside a checkout, so the one control in front of the
accident its own docstring names did not cover it. The default now resolves
outside the project and the store refuses a root inside a working tree whoever
chose it — see row 5 of the disclosure ledger, amended rather than merely
satisfied.

**Findings 5, 6 and 7 are open, and they are the ones the phase is actually
about.** 1–4 were a runner that did not implement a contract everything else had
already agreed on; the remainder are the contract being wrong. A Scanner still
cannot distinguish a clean scan from one that could not run, still cannot say
what it is the scanner *of* when the subject is a repository rather than a Unit,
and `Adjudicator` still ships in `plugins/scanner.py`. None of those is fixable
in `run.py`, which is why closing four of seven does not close the phase.

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

What remains is clerical and belongs with the archiving: put the `LICENSE` file
into both source repositories before they go read-only, so the record does not
depend on anyone remembering this note. The attribution half is already done —
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
