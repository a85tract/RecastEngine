# corpus/

Twelve open-source Fortran libraries, as pinned git submodules, that the
engine **alone** -- no domain extension installed -- is held to translating.
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

`cases.json` says which files of each submodule make one case and what has
to happen to them first (cpp, for the `.F90` ones). Nothing is vendored: the
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
and none of them is inside `corpus/`, whose only other contents are
submodules this tool must never write to.

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
