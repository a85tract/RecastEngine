# A worked case: a canopy model, Fortran to NumPy and JAX, held to the original

[`getting-started.md`](getting-started.md) translates a thirty-line module.
This page is what the same engine does on a real model: the multilayer
canopy of the Community Land Model (CLM-ml v2, Bonan's Fortran, 76
modules), taken to NumPy and then to JAX, and held at every step against
the untouched Fortran. It is written for the reader of the first two
pages: someone who runs models and wants to know what "the translation is
right" means here, what it took, and where the claim stops.

The case reproduces a paper. Lahlou, Hawkins and Gentine (2026,
[arXiv:2606.07681](https://arxiv.org/abs/2606.07681)) translated the same
model to JAX with LLM agents in a five-phase pipeline and reported
module-level agreement within a relative 1e-4, a 31-day column within 1%,
gradients, a calibration and a throughput figure. The record here holds
the engine to the same targets, with one difference in kind: every number
below was produced by a gate that compared against compiled Fortran, not
by a reading of the code.

Two things are not public yet, and this page names them without linking:
the CLM-ml domain extension for the engine (`recast-clm-ml`) and the case
directory that drives it (`clm-ml-jax`, with its `REPRODUCTION.md`, the
record this page condenses). The upstream Fortran is public:
[`gbonan/CLM-ml_v2.CHATS`](https://github.com/gbonan/CLM-ml_v2.CHATS) at
`8d1cc40`, read and never modified. Everything ran on one laptop (Apple
silicon, gfortran 16) between 2026-08-28 and 2026-08-31.

## 1. Why this is harder than a module

The satvap module of the getting-started page has three properties that
made it easy: it depends on nothing, its arguments are plain numbers and
arrays, and any input is a valid input. A model has none of them.

| the model does this | what it means for a checker |
|---|---|
| 76 modules `use` each other in a graph | a module cannot be translated or compiled alone |
| the physics takes one big object, `mlcanopy_inst`, with a few hundred array components, and reads and writes it through `associate` aliases | f2py cannot wrap a derived type, so there is nothing for the compiled reference to be called with |
| it calls a framework: `endrun`, a log unit, netCDF, history files | none of that exists in a standalone Python run |
| routines check their own energy balance and abort if it does not close | random inputs never close it, so the routine aborts on both sides and nothing is compared |
| solvers take a procedure as an argument (`hybrid`, `zbrent`) | a wrapper cannot spell a procedure dummy |

The first attempt, the shipped `translate` recipe with no extension, is the
honest baseline: **0 of 76 units passed.** 48 stopped at the read/write
check, on blocks that touch `mlcanopy_inst%…`; 26 stopped at the oracle,
which had nothing it could wrap. That is the same picture the engine's own
corpus of third-party libraries gives, and it is where the paper's Tier-2
and Tier-3 modules begin.

## 2. What it took

Two kinds of thing, and the split is the point of the engine's design.

**A small extension that knows CLM-ml.** Which modules are constants,
which are framework stubs and what the framework answers, that `r8` is
64-bit, that an `intent(out)` object is really `inout` here (the model
reads through it), where `netcdf.inc` is. Four plugins, all tables, no
translation rules of their own.

**Machinery in the engine that knows nothing about CLM,** and had to be
built or extended on the way:

| the engine now | in plain words |
|---|---|
| resolves use-constants and bundles companions | a unit's translation carries the sibling translations it calls into, so it imports and runs on its own (the `target: tree` of the getting-started page, in its full form) |
| writes stand-ins for stub modules | `endrun` raises, history and netCDF calls are `pass`, module constants keep their values |
| **flattens** a derived-type interface | from the components a routine actually touches, it generates one adapter per side: a Fortran `<name>_flat` that allocates the object, copies flat arrays in, calls the original and copies the written components out; and a Python `<name>_flat` that does the same to the translation. f2py sees only flat arrays. |
| **records** a real run | from the same plan, a Fortran recorder module and a probed copy of the tree; the model runs its own May-2007 case and every call of every probed routine is captured, inputs and outputs, in the engine's dump format. The gate then replays those instead of random inputs. |
| ports a flat function to JAX | the JAX backend, untouched, lowers a flat function derived from the verified NumPy one; root finders are specialized per callback and given implicit-function derivatives |

The findings in section 5 are the reason most of that exists: each
piece was added when a gate failed for a reason the engine could not yet
express, not designed in advance.

## 3. The numbers, in the order they were earned

### The first module, and the first seven

`MLWaterVaporMod`, the paper's simplest module, through all eight stages
under the extension's recipe:

```console
$ recast run translate-clm-ml output/staged --config output/staged/recast.json --unit fortran:mlwatervapormod
fortran:mlwatervapormod
  [ok ] frontend   clm
  [ok ] transform  translate.clm-ml
  [ok ] verifier   static.rwset                sampled: 8 blocks match
  [ok ] oracle     f2py-golden-clm-ml          f2py:mlwatervapormod:444f824826717b73
  [ok ] verifier   differential.bitexact       bit_exact: 30 points across 2 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded
```

Thirty points, temperature sampled in 200 to 330 K, maximum difference 0
ULP. The paper's bar for a module was a relative 1e-4. With the
derived-type flattening in place, seven units went through on generated
inputs, the largest `LeafBoundaryLayer` at 48,000 points. The rest of the
physics could not be gated that way: `LeafFluxes` computes its fluxes and
aborts if the energy balance is off by more than 1e-3, which random
inputs guarantee.

### On the model's own state: all 15 canopy physics modules

The recorder answers that. The model ran one day of its own case, 40 calls
per routine were captured, and the gate replayed them. Every physics module
of `multilayer_canopy/` matched the Fortran bit for bit on the model's own
state:

| module | what it is | points, all bit-exact |
|---|---|---|
| LeafPhotosynthesis | Farquhar photosynthesis with stomatal optimization through a root finder | 160,200 |
| RungeKuttaUpdate | the time integrator | 140,200 |
| FluxProfileSolution | the implicit solver, calling LeafFluxes, SoilFluxes and tridiagonal solves | 84,400 |
| SolarRadiation | Norman and two-stream radiative transfer | 68,680 |
| LeafFluxes | leaf energy balance | 48,000 |
| CanopyNitrogenProfile | | 48,040 |
| LeafBoundaryLayer | | 24,000 |
| CanopyTurbulence | roughness-sublayer turbulence, Obukhov length by secant through a callback | 16,480 |
| LongwaveRadiation | | 16,200 |
| CanopyWater | 3 subprograms | 16,120 |
| PlantHydraulics | 3 subprograms, soil resistance over the column | 13,080 |
| SoilTemperature | the soil-column heat solver, 2 subprograms | 8,400 |
| LeafHeatCapacity | | 4,000 |
| InitVertical | 3 subprograms | 1,412 |
| SoilFluxes | | 240 |

Bit-exact means what it did on the first page: not one bit of one output
differs, across every call replayed. One lesson from the recording is worth
carrying to any model: **the build that produced the recording is part of
the reference.** Recorded under `-O2`, `LeafFluxes` differed from the
translation by up to 4,301 ULP; that was the compiler's fused
multiply-add, not the translation. Recorded under the engine's reference
flags, 0 ULP.

### The whole time step, and the whole month

The paper's Phase 5 needs the orchestrator, not just its physics: forcing
interpolation, the six sub-steps with five Runge-Kutta passes each, the
flux integration, the diagnostics. Rather than wire fifteen translated
kernels together by hand, the same record-then-translate path was pointed
at `MLCanopyFluxes` whole. Its flat plan carries 19 objects and 331
components; one recorded call is one half-hourly CLM step, 445 inputs and
259 outputs. The NumPy translation matched the Fortran on all 48 steps of
the day: **902,544 points, bit-exact.**

The recording also yields the driver's contract: of the step's inputs, 262
are the previous step's own outputs, value for value, and 194 are external
(tower forcing and the soil state). A driver was written on that split,
carrying the state from the translation's own outputs and taking only the
external inputs from the recording. Closed loop, over the full May-2007
window:

| driver | steps | NumPy translation vs Fortran |
|---|---|---|
| canopy closed loop | 1,488 | bit-for-bit at every output of every step |
| canopy + soil-thermal closed loop (SoilThermProp, MLCanopyFluxes, SoilTemperature chained per step) | 1,488 | bit-for-bit at every output of every step |

The paper's Fig. 5 claim, a 31-day column within 1%, is met with zero
tolerance on the NumPy side, within the boundary that soil hydrology and the
tower forcing are taken from the recording (the paper's scope does not
include hydrology either).

### JAX: what "within tolerance" looks like when bit-exact is out of reach

JAX is where the reader should expect the claim to weaken, and it does, for
a stated reason. XLA compiles arithmetic with fusion and its own
transcendental functions, so a JAX kernel is not expected to match the
compiler's `libm` bit for bit. The engine's gate for a port is a ULP bound
on the dominant outputs, 32 by default.

- Per module, under jit: 12 of 15 pass that gate. The three that do not
  (FluxProfileSolution, Longwave, SoilFluxes) are bit-exact with jit
  disabled, so the residual is fusion amplified by an iterative solve or a
  cancellation. That is a documented property of the backend, not a
  translation defect, and the record says so rather than widening the gate.
- The whole step as one JAX kernel, driven closed loop for the full month,
  measured the paper's way (daily-mean column fluxes against the Fortran,
  tolerance 1%):

| quantity | relative drift, canopy loop | canopy + soil-thermal loop |
|---|---|---|
| GPP | 1.04e-4 | 1.01e-4 |
| sensible heat | 9.6e-4 | 8.7e-4 |
| latent heat | 1.76e-4 | 1.77e-4 |
| soil temperature | 9.5e-8 | (t_soisno max 2.4e-6) |

One to four orders inside the paper's band, and flat from day 1 to day
31: the trajectory does not diverge.

### Gradients, calibration, throughput

- **Fig. 6, Jacobian rows.** Reverse mode (`jax.grad`) through the whole
  step, 30 solver passes and three root-finder specializations, against
  forward mode, on a leafed-out day at noon: sensible heat to forcing
  temperature agrees to 2.8e-12 relative, latent heat to 6.0e-11, ground
  temperature to 9.9e-14. Getting there needed one rule the per-module
  gradients never met: `(h2ocan/h2ocanmx)**0.67` on a dry layer has an
  infinite derivative at zero, and a single one seeded NaN into 124
  outputs. The emitted power now takes the subgradient 0 there, and is
  bit-identical for any positive base.
- **Fig. 8, calibration.** The stomatal efficiency parameter perturbed by
  1.5x and recovered from GPP and latent heat over a day, open loop: back to
  0.9995 of the truth in 7 iterations. The loss at the true value is 1.9e-6
  rather than 0, because a root finder stopped on a tolerance makes the
  model a staircase in its parameters; the implicit-function derivative
  is the derivative of the smooth limit, which is the paper's own position.
- **Fig. 9, throughput.** One CLM step of the whole-step kernel on this
  laptop's CPU: 43.7 ms jitted after a 25.5 s one-time compile, against
  377 ms for the NumPy translation and about 4.9 ms for the Fortran. The
  paper's number is a GPU batch on a supercomputer and is neither
  confirmed nor contradicted by a single-point CPU timing.

## 4. What a scientist can take from the numbers

- **Bit-exact is a real bar, and a model can clear it.** Not a tolerance
  chosen to pass, but the compiled Fortran's own bits, on its own state,
  for a month. When the NumPy side drifts from that later, the drift is a
  change, and the summary file records it as one.
- **The reference has to be recorded under known flags**, and the recording
  is part of the evidence. A model's inputs are correlated and its routines
  guard themselves, so random inputs are the wrong instrument past the
  leaf level.
- **JAX carries a residual that is XLA's, not the translation's**, and the
  way to know which is to switch jit off: same code, no fusion. The record
  keeps that check beside every failed ULP gate.
- **Gate on a day when the model is doing something.** The sharpest
  regression of the case (section 5) was invisible on day 1, when the
  canopy was leafing out and the root finders sat at a bound.

## 5. What the gates found that reading would not

Every item here was a wrong number before it was a diagnosis. In the
engine, each was fixed and the fix is on `main`; in the upstream model,
each was reported and nothing was changed here, because a reference that
has been edited to agree is not a reference.

**In the engine's translation**

- `pftcon%slatop` is allocated `(0:mxpft)`; the translation read one plant
  functional type off. 102 of 8,000 points differed; the engine did not
  know a component's allocated lower bound. It does now.
- `case (0, -1)` was emitted as `== 0 or == 1`: the unary minus is a node
  above the literal. The solver took the wrong branch and aborted.
- `tbi_profile(begp:endp, 0:nlevmlcan)` read one layer off;
  `col%dz(begc:endc, -nlevsno+1:nlevgrnd)` one snow layer off, twice;
  `tair(p,:)` clobbered above the canopy by a whole-array return; `nrk =
  runge_kutta_type/10` rendered as 4.1; two parameters started from 0
  instead of their declared initializers.
- The one that matters most for anyone building a gate: a loop `do irk =
  1, nrk_steps+1` (five passes, the last one indexing nothing) had its
  bound rewritten to the four-stage axis it appeared to index. On day 1
  nothing changed. On day 15, with the canopy alive, GPP was off by up to
  15%. The calibration scan caught it, the fix is one condition, and the
  standing policy since is that every kernel gate runs active-canopy steps
  beside day 1.

**In the upstream Fortran** (both reported to the model's repository)

- The Runge-Kutta tableau (`ark, brk, crk`) lives in un-`SAVE`d locals,
  filled on the first call and read on every later one. Standard Fortran
  leaves them undefined after return; it works only when the compiler
  happens not to reuse the stack. Builds with different compilers or flags
  disagree by up to 63 W/m² in the fluxes; with `save` added on a scratch
  copy, 99.4% of values are bit-identical to the model's shipped output.
  [Issue #1](https://github.com/gbonan/CLM-ml_v2.CHATS/issues/1). The case
  stages the tree with the tableau initialized every step, marked as a
  deviation, so that a reference exists to compare against at all.
- Three routines declare the canopy object `intent(out)` and read through
  it on the next line. [Issue #2](https://github.com/gbonan/CLM-ml_v2.CHATS/issues/2).
  The extension treats such a dummy as `inout` and records the assumption.

## 6. Where the claim stops

- The paper's own JAX code was not public when this was done, so nothing
  here is compared to the authors' artifact, only to the Fortran and to
  the paper's reported tolerances.
- Soil hydrology (`SoilWater`) and the tower forcing stay recorded; the
  canopy and the soil-thermal column are closed loops, the water column is
  not. The paper's scope stops at the same place.
- The per-module gates replay the first day of the recording (40 calls per
  routine); the whole-step gate covers every call of the month.
- The JAX ULP gate is not passed by the whole-step kernel and is not
  expected to be; the paper-level criterion is the 1% band, which it
  passes by orders of magnitude.
- Throughput is a single-point CPU number.

## 7. Running it

The case needs the engine with the `fortran`, `translate`, `verify` and
`jax` extras, the CLM-ml extension installed beside it, gfortran, and
netCDF-Fortran for the model's stub modules. The month-long recording is
4.7 GB. The case directory's own `REPRODUCTION.md` is the full record,
with every command and every intermediate number; the shape of a run is:

```bash
python stage.py                       # cpp-flatten the 76 modules into output/staged/, with the marked deviation
python record.py <units>              # build the probed model, run May 2007, capture the calls
recast run translate-clm-ml output/staged --config output/recorded/<unit>.json   # replay, bit-exact gate
python run_port.py                    # the JAX ports, ULP gate
python column.py --mode numpy         # the closed-loop day
python month_jax.py                   # the closed-loop month, Fig. 5
```

Until the extension and the case are public, the way to run this is to ask
(see the engine's README for the contact). What is public is the engine
that did it, the upstream model, the two issues, and this record.

## 8. Where to go next

| | |
|---|---|
| [`tree-units.md`](tree-units.md) | the engine's side of section 2 in full: use-constants, stand-ins, bundling, flattening, recording |
| [`reading-the-evidence.md`](reading-the-evidence.md) | how to read the manifests the gates above wrote |
| [`roadmap.md`](roadmap.md) | where each recipe stands, and what evidence each claim rests on |
