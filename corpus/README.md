# corpus/

Twelve open-source Fortran libraries, as pinned git submodules, that the
engine **alone** -- no domain extension installed -- is held to translating,
and two small trees shipped in-tree (`toy_physics`, `probe_kernel`; the last
section).
They are the public form of the question the roadmap's P4 asks ("does the
engine pass with the CESM extension uninstalled?"), asked of code nobody here
wrote: nonlinear least squares, quadrature, special functions, root finding,
splines, FFTs, an optimizer, a cloud-microphysics kernel.

| case | upstream | what |
|---|---|---|
| minpack | fortran-lang/minpack | nonlinear least squares, one module |
| quadpack | jacobwilliams/quadpack | adaptive quadrature (`quadpack_double`, cpp-expanded) |
| specfun | jacobwilliams/specfun | special functions, a 12.6k-line module |
| polyroots | jacobwilliams/polyroots-fortran | polynomial roots |
| pchip | jacobwilliams/pchip | SLATEC PCHIP interpolation |
| slsqp | jacobwilliams/slsqp | SLSQP + a BLAS subset + BVLS |
| bspline | jacobwilliams/bspline-fortran | B-splines without the OO layer |
| roots | jacobwilliams/roots-fortran | scalar roots, object-oriented -- a stress test |
| fortran-utils | certik/fortran-utils | sorting, splines, special functions, linear algebra |
| fftpack | fortran-lang/fftpack | 58 files of bare subprograms, no module |
| numfor | numericfor/numfor | integration, FITPACK-style goto code, random numbers |
| cloudsc | ecmwf-ifs/dwarf-p-cloudsc | the IFS CLOUDSC kernel -- the closest thing here to CAM physics |

`cases.json` says which files of each submodule make one case, what has
to happen to them first (cpp, for the `.F90` ones), and, for a case whose
routines take inputs no uniform draw lands on, an input profile under
`profiles/` -- MINPACK's packed triangular workspaces, `lr = n(n+1)/2` --
that `stage` puts beside the sources as `recast_inputs.py` for the
bit-exact gate to shape its draws by. That is an operator's statement about
the source's domain, the kind the engine cannot read off the source without
lying about it, and not a domain extension. Nothing is vendored: the
sources stay in their own repositories under their own licences, at the
commits the submodules pin.

## Running

```bash
git submodule update --init --depth 1
python tools/corpus.py run            # every case -> corpus/baseline.json
python tools/corpus.py run minpack    # one case
python tools/corpus.py report         # the table, from the recorded baseline
```

Each case is staged under `output/<case>/staged/` and walked by the
`translate` recipe with every module unit selected. One directory holds
everything a case produces:

| | |
|---|---|
| `output/<case>/staged/` | the flattened Fortran the engine reads, and a `recast.json` pinning the rest here |
| `output/<case>/translate/` | the run's per-unit workspace: candidates, and the f2py build of the same source |
| `output/<case>/translated/` | every unit's emitted Python in one flat directory, which is what the import probe below needs |
| `output/<case>/evidence/` | one manifest per verdict |

All four are the same thing -- generated, disposable, one place to delete --
and none of them is inside `corpus/`, whose other contents are submodules
this tool must never write to and the two shipped trees below.

The record per unit is how many blocks the rules refused and why
(normalised, so one missing rule counts once however often it fires),
whether the static read/write check agreed with the translation, and which
stage stopped the unit.

## One case, end to end

`run` walks every unit of every case and writes a table. To watch a single
unit go through the whole `translate` recipe instead -- the same eight stages
the shipped example runs, on code nobody here wrote:

```bash
git submodule update --init --depth 1 corpus/numfor
python tools/corpus.py stage numfor
recast run translate output/numfor/staged \
    --config output/numfor/staged/recast.json --unit fortran:basic
```

`stage` reads `cases.json` for what belongs to the case and lays it out
somewhere the engine is free to write. For `numfor` that is the 133 `.f90`
and `.inc` files under `src/`, its test tree left out, flattened into a fresh
`output/numfor/staged/` -- flat because an `#include "qtrs1d.inc"` names no
directory. A case carrying `.F90` sources goes through `gfortran -E -P -cpp`
on the way. The submodule is only ever read. The `recast.json` it leaves
behind pins the run's output to `output/numfor/`: the engine names that
directory after the source tree's basename, and every case's staged tree is
called `staged`. `stage` prints the command with the flag already on it.

```console
fortran:basic
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             5 deferred block(s)
  [ok ] verifier   static.rwset                sampled: 55 blocks match
  [ok ] oracle     f2py-golden                 f2py:basic:f4038505...
  [ok ] verifier   differential.bitexact       bit_exact: 10 points across 1 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

`basic` is numfor's 354-line utility module -- kinds, timers, a date stamp,
`is_inf` -- at the commit the submodule pins. The run writes nothing into the
staged tree; two things to open under `output/numfor/`:

| | |
|---|---|
| `translate/fortran_basic/candidate/basic_numpy.py` | the generated Python, every block carrying the source lines it came from |
| `evidence/fortran_basic/*.json` | one manifest per verdict -- artifact digest, oracle key, metrics |

The `5 deferred block(s)` are the rules declining to guess: two `cpu_time`
calls, a `date_and_time`, and two formatted internal writes, each left
standing as a `raise NotImplementedError` for a human to answer. Everything
else in the module is translated, and checked.

`basic` is also not typical, which is why `baseline.json` and not this
section is the general picture: of the 59 units the twelve cases hold, it is
the only one that currently reaches the bit-exact gate.

[`docs/corpus-numfor-example.md`](../docs/corpus-numfor-example.md) stays
with this same unit at length -- the refusals, the evidence manifest, a unit
of the same case that fails beside it, and what the passing run does not
establish.
[`docs/corpus-lapack-example.md`](../docs/corpus-lapack-example.md) does the
same for the one unit whose Python does not import, `fortran-utils`'
`linalg`: a library over LAPACK, whose import is a real dependency the
header rule rightly keeps, and what an externals shim would change.

## What the record is for

`baseline.json` is committed. It is the engine's claim about itself, and the
work list: a rule relayed from the translator, or written here, either
moves a number in it or was not needed. Files of bare subprograms -- no
module, no program -- are counted as `bare` and not yet attempted; that
count is a gap in the recipe, not in the corpus.

A refusal is a block the rules would not guess at; it is not a wrong
translation. The numbers that would say a translation is *right* -- the
bit-exact gate against an f2py build of the same source -- come after the
static check passes, and for most cases it does not yet.

## The trees shipped in-tree

Beside the submodules, four source trees live here directly, small enough
to read in one sitting and needing nothing checked out: `toy_physics/`, a
Fortran module the shipped recipes run over end to end with the operator
config beside it; `elm_leaf/`, four modules written the way the E3SM Land
Model writes its biogeophysics and the framework modules under it;
`clubb_solve/`, three modules in the shape of CLUBB's variance step and
the tridiagonal solver under it; and `probe_kernel/`,
two scripts standing in for an instrumented C kernel (its own README says
how). They are the public form of the check the roadmap names: the recipe
has to work *here*, on sources anyone can read, not only on the private
corpus it was migrated against.

`elm_leaf/` exists because the engine was green while every ELM unit was
red. Its `shr_kind_mod`, `shr_const_mod` and `elm_varcon` carry the
constant shapes ELM's tree has -- `selected_real_kind(12)` and the `p=12`
form, a kind renamed through an integer parameter, a default-real literal
stored in a double, a negative one, an integer parameter set from a real,
character names -- and `leaf_layers` reads them the way a biogeophysics
routine does, truncates a REAL into an INTEGER scalar, takes a log and an
integer power inside its loop, and takes its extent from a dummy called
`np`. Each of those refused, misfolded or reached the emitted kernel
unresolved at some engine commit that passed its own suite. Both recipes
run over it (`target: tree`, `backend: tree-jax`, the engines the
extensions stand on), with the constants modules declared in the config
the way the ELM extension's conventions declare them:

    recast run translate corpus/elm_leaf --config corpus/elm_leaf/recast.json \
        --summary corpus/elm_leaf/verification.json
    recast run port corpus/elm_leaf --config corpus/elm_leaf/port.json \
        --summary corpus/elm_leaf/port-verification.json

`clubb_solve/` exists for the same reason, on the cloud side. Its
`xp2_solve` is written the way CLUBB's `advance_xp2_xpyp_module` is: a
public driver, a private routine that assembles a tridiagonal system into
`(ngrdcol, nzm)` locals, and a solver whose dummies are one rank higher --
`rhs(ngrdcol, nzm, nrhs)`, the solution likewise -- so the call hands a
rank-2 array to a rank-3 dummy by sequence association, INOUT in and OUT
back, with the band layout (`ndiags3`, the diagonal positions) read from a
constants module and sizing an OUT array. The NumPy translation spells the
actual as the first `ngrdcol * nzm * nrhs` cells of the array flattened in
column-major order; the port to JAX once could not read that spelling
back, dropped the private routine's kernel without a note, left its caller
calling a host function that does not exist, and sized the OUT array by
its own unbound shape -- each while the engine's suite and the ELM tree
stayed green. Its scalar loop sits in one branch of a PDF switch the trace
resolves (`ipdf_type`, a static under jit), after an error return, the way
CLUBB's does: the loop's held bounds are set in that branch and carried by
the return flag's cond around it, and the port once stored them as Python
ints against their int32 start, so the whole step would not trace on the
one case with passive scalars while every case without them passed. Both
recipes run over it in CI:

    recast run translate corpus/clubb_solve --config corpus/clubb_solve/recast.json \
        --summary corpus/clubb_solve/verification.json
    recast run port corpus/clubb_solve --config corpus/clubb_solve/port.json \
        --summary corpus/clubb_solve/port-verification.json

    recast run translate corpus/toy_physics --config corpus/toy_physics/recast.json \
        --summary corpus/toy_physics/verification.json

`toy_physics` needs a Fortran compiler and the f2py build backend
(`pip install 'recast-engine[fortran,translate,verify]'` and a `gfortran` on
PATH). The run translates the module, cross-checks its dataflow, compiles
the untouched Fortran as the reference, compares every output bit for bit,
and writes the evidence manifests under `output/toy_physics/evidence/`.

Two records come out of that, and they are for different readers.

`output/toy_physics/evidence/` holds the **manifests**: one immutable, content-addressed
CC-Test document per verdict per run, append-only, carrying the full metrics
and the environment. That is the audit trail, and it accumulates a file per
attempt -- including the attempts that failed, which is the point of an audit
trail and the reason it is not committed.

`verification.json` is the **current state**: one entry per unit and verifier,
regenerated rather than appended. It carries the confidence, the artifact
digest, the oracle's *name*, and the countable metrics, and deliberately omits
wall-clock time and paths -- so two runs over the same revisions produce the
same bytes. That is what makes it worth committing: it diffs like a lockfile,
and a change in it is a change in what has been verified.

The oracle's **cache key** is deliberately not among those, and the reason is
what makes the file diffable at all. The key folds the compiler's version, so
recording it would make the summary a fact about the machine: this example's
committed bytes come from gfortran 16 and CI's come from whatever the runner's
distribution ships, and the byte comparison would fail there while nothing was
wrong. The key belongs in the evidence manifest, which is a record of one run
and is not compared to anything. What survives into the summary is only what
two correct runs agree on.
[`toy_physics/verification.json`](toy_physics/verification.json) is the one
this example produces, checked in so that a reader can see the claim without
owning a Fortran compiler.

## The same module, ported

`toy_physics` also runs the `port` recipe, over the same sources and the same
sampling config:

    recast run port corpus/toy_physics --config corpus/toy_physics/port.json \
        --summary corpus/toy_physics/port-verification.json

This one needs `jax` (`pip install 'recast-engine[fortran,translate,jax]'`) and
**no Fortran compiler** — which is the anchoring decision showing through rather
than a convenience. The reference is `numpy-anchor`: the NumPy translation of
the same unit, re-derived from the same Facts, so nothing in this run builds
Fortran. That makes the port's claim a chain — NumPy bit-exact against the
Fortran above, JAX ULP-bounded against the NumPy here — and the honest part is
that this run cannot check the first link. The run above is what checks it.

The verdict is `ulp_bounded` rather than `bit_exact`, and that is the ceiling
rather than a shortfall: XLA's transcendentals are not libm's. On this module —
which has none — 76 of 85 points land bit-exact and the remaining nine within
1 ULP, against a gate of 32.

[`toy_physics/port-verification.json`](toy_physics/port-verification.json) is
checked in for the same reason as its bit-exact sibling, with one difference in
how much it is allowed to prove. A ULP count is not device-independent — XLA's
CPU backend does not promise the same last bit on x86 as on arm64, and the
summary records `candidate_device` and `reference_device` so a reader can see
which machine produced it. So CI runs this example and gates on the verdict, but
does not diff the file the way it diffs `verification.json`. Treat a changed ULP
count here as a question, not an alarm.
