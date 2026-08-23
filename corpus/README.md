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

Each case is staged under `corpus/.build/<case>/` and walked by the
`translate` recipe with every module unit selected. The record per unit is
how many blocks the rules refused and why (normalised, so one missing rule
counts once however often it fires), whether the static read/write check
agreed with the translation, and which stage stopped the unit.

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
