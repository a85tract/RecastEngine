# Getting started: your first translation

This page is for a scientist who has a Fortran file and wants it in Python,
and who does not spend much time with developer tools. It walks from an empty
machine to a translated, verified module of your own, and it shows what every
command prints so you can tell whether you are on track. Nothing here needs
more than a terminal and about twenty minutes.

What the engine does, in one sentence: it translates your Fortran to Python,
then compiles your *original* Fortran, runs both on the same random inputs,
and refuses to call the translation correct unless every output agrees bit
for bit. The translation is the easy half. The checking is the point.

## 1. What you need on your machine

Four things. Check each with the command on the right; if it prints a
version, you have it.

| | | check with |
|---|---|---|
| a terminal | Terminal.app on macOS, any shell on Linux | |
| `git` | to fetch the engine | `git --version` |
| `python3`, 3.11 or newer | the engine is Python | `python3 --version` |
| `gfortran` | to compile *your* Fortran as the reference | `gfortran --version` |

Installing what is missing:

```bash
# macOS (with Homebrew, https://brew.sh)
brew install gfortran git

# Ubuntu / Debian
sudo apt install gfortran git python3 python3-venv

# any platform with conda
conda install -c conda-forge gfortran git python=3.11
```

`gfortran` is not optional. The engine's correctness check compiles your
untouched Fortran and compares against it, so without a compiler there is
nothing to compare against. Section 6 shows what that looks like if you
forget.

Windows: the engine is developed and tested on macOS and Linux. The least
painful route on Windows is WSL (Windows Subsystem for Linux), and then the
Ubuntu lines above.

## 2. Install the engine

Copy these lines one at a time. Each one is explained below.

```bash
git clone https://github.com/a85tract/RecastEngine.git
cd RecastEngine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[fortran,translate,verify]"
```

Line by line:

1. `git clone` downloads the engine into a folder called `RecastEngine`.
2. `cd` moves you into it. **Every command on this page is run from inside
   this folder.**
3. `python3 -m venv .venv` creates a private Python environment in a hidden
   folder called `.venv`, so the engine's packages do not touch anything else
   on your machine.
4. `source .venv/bin/activate` switches your terminal to that environment.
   Your prompt gains a `(.venv)` prefix. **You have to do this again in every
   new terminal window**; if a later command says `recast: command not
   found`, this is the line you skipped.
5. `pip install` installs the engine plus the three extras a translation
   needs: the Fortran reader, the Python writer, and the checker. This takes
   a minute or two.

(If you have [`uv`](https://docs.astral.sh/uv/), `uv venv --python 3.11`
and `uv pip install -e ".[fortran,translate,verify]"` do the same thing
faster, and fetch a Python 3.11 for you if you have none.)

Now check the installation:

```console
$ recast doctor
recast 0.0.1.dev0  python 3.11.16
24 plugin(s) registered across 10 kinds
```

The version and the plugin count may differ from these; what matters is that
the command runs and reports plugins. If it does, you are installed.

## 3. Run the shipped example

The engine ships a tiny Fortran module, `examples/toy_physics/toy_physics.f90`,
that integrates pressure down a column. Open it if you like; it is thirty
lines and looks like any Fortran you have written. Run the engine over it:

```console
$ recast run translate examples/toy_physics --config examples/toy_physics/recast.json
fortran:toy_physics
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy
  [ok ] verifier   static.rwset                sampled: 4 blocks match
  [ok ] oracle     f2py-golden                 f2py:toy_physics:18afeebe0a929004
  [ok ] verifier   differential.bitexact       bit_exact: 85 points across 2 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded
  evidence: file:///.../output/toy_physics/evidence/fortran_toy_physics/2640ab94....json
  evidence: file:///.../output/toy_physics/evidence/fortran_toy_physics/9fd94a4d....json
  evidence: file:///.../output/toy_physics/evidence/fortran_toy_physics/7c55abd6....json

1 unit(s), 3 verdict(s), all passed
```

The first time takes longer, because the compiler runs. The command has
three parts: `run translate` says what to do, `examples/toy_physics` is the
folder holding the Fortran, and `--config ...` points at a small settings
file we will come back to.

### Reading the output

Each line is one stage, and `[ok ]` at the left means that stage succeeded.
In plain words:

| stage | what happened |
|---|---|
| `frontend fortran` | your Fortran was read and understood |
| `transform translate.numpy` | Python was written from it |
| `verifier static.rwset` | a first check: every block of the Python reads and writes the same variables as the Fortran block it came from |
| `oracle f2py-golden` | your **original** Fortran was compiled with gfortran, to serve as the reference |
| `verifier differential.bitexact` | the main check: both versions were run on the same random inputs, 85 of them, and every output matched to the last bit |
| `verifier symbolic.notary` | a check that the translation did not reorder any arithmetic (here there was nothing to check) |
| `store fs-evidence` | the three verdicts were written to disk as JSON, so the claim is on record |

The last line is the summary: one Fortran module was processed, three checks
ran, all passed.

### Where the Python went

Nothing was written next to your Fortran. Everything lands in a folder
called `output/`, named after the folder you gave:

```
output/toy_physics/
├── translate/fortran_toy_physics/candidate/
│   ├── toy_physics_numpy.py        <- the translation
│   └── toy_physics_constants.py    <- the module's parameters (r8, gravity)
└── evidence/fortran_toy_physics/
    └── *.json                      <- one file per verdict
```

The Python file is long, because its top half is a fixed block of helper
functions that reproduce Fortran arithmetic exactly (Fortran's `MOD`,
`NINT`, `SIGN` and `MIN` do not behave like Python's). Your code is at the
bottom. For `toy_physics` it ends like this:

```python
def settle(n, rho, dz, w):
    """L11-L23 subroutine (machine-translated)."""
    p = np.empty((n,), dtype=np.float64)
    i = 0
    # B001 <- L18-L18
    p[0] = ((rho[0] * GRAVITY) * dz[0])
    # B002 <- L19-L22
    for i in range(2, n + 1):
        p[i - 1] = (p[(i - 1) - 1] + ((rho[i - 1] * GRAVITY) * dz[i - 1]))
        w[i - 1] = (w[i - 1] - (dz[i - 1] / ((1.0 + rho[i - 1]))))
    return w, p
```

Two things to notice. Every block is labelled with the Fortran lines it came
from (`B002 <- L19-L22`), so you can always look back. And the calling
convention changed the way Python needs it to: Fortran's `intent(out)`
argument `p` is no longer passed in but **returned**, together with the
`intent(inout)` argument `w`. So where the Fortran was
`call settle(n, rho, dz, w, p)`, the Python is `w, p = settle(n, rho, dz, w)`.

### Using it

The translation is an ordinary Python module. From the folder it sits in:

```console
$ cd output/toy_physics/translate/fortran_toy_physics/candidate
$ python3
>>> import numpy as np
>>> import toy_physics_numpy as tp
>>> rho = np.array([1.2, 1.1, 1.0, 0.9])
>>> dz  = np.array([100.0, 100.0, 100.0, 100.0])
>>> w   = np.zeros(4)
>>> w, p = tp.settle(4, rho, dz, w)
>>> p
array([1176.7392 , 2255.4168 , 3236.0328 , 4118.5872 ])
>>> tp.column_mass(4, rho, dz)
420.0
```

Then `cd ../../../../..` to get back to the `RecastEngine` folder for the
rest of this page. Or copy the two `.py` files anywhere you like; they depend
only on NumPy.

## 4. Translate your own Fortran

The engine reads one Fortran **module** per file, and treats each module as
a unit to translate and check. So the shape it wants is what most scientific
code already has:

```fortran
module <name>
  implicit none
  ! parameters
contains
  ! subroutines and functions, with intent on every argument
end module <name>
```

Put your file in a folder of its own. The folder's name becomes the name of
the run's output. Here is a real one; make a folder called `satvap` next to
`examples` and save this as `satvap/satvap.f90`:

```fortran
! Saturation vapor pressure over water (Bolton 1980) and the mixing ratio
! it implies -- the kind of small kernel an analysis script carries around.
module satvap
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: eps = 0.622_r8

contains

  function esat(t) result(es)
    ! t in kelvin, es in hPa
    real(r8), intent(in) :: t
    real(r8) :: es, tc
    tc = t - 273.15_r8
    es = 6.112_r8 * exp(17.67_r8 * tc / (tc + 243.5_r8))
  end function esat

  subroutine mixing_ratio(n, t, p, rh, q)
    integer, intent(in) :: n
    real(r8), intent(in) :: t(n)    ! kelvin
    real(r8), intent(in) :: p(n)    ! hPa
    real(r8), intent(in) :: rh(n)   ! 0..1
    real(r8), intent(out) :: q(n)   ! kg/kg
    integer :: i
    real(r8) :: e
    do i = 1, n
      e = rh(i) * esat(t(i))
      q(i) = eps * e / (p(i) - e)
    end do
  end subroutine mixing_ratio
end module satvap
```

No settings file this time. Just point the engine at the folder:

```console
$ recast run translate satvap
fortran:satvap
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy
  [ok ] verifier   static.rwset                sampled: 3 blocks match
  [ok ] oracle     f2py-golden                 f2py:satvap:8c6c49ccd27d2248
  [ok ] verifier   differential.bitexact       bit_exact: 90 points across 2 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

The translation is in `output/satvap/translate/fortran_satvap/candidate/satvap_numpy.py`,
and it was checked against your compiled Fortran on 90 random inputs. The
`exp` became `math.exp`, deliberately: NumPy's vectorised `exp` can differ
from the compiler's by one bit in the last place, and the check would have
caught that.

### Telling the checker what realistic inputs look like

Where did the 90 inputs come from? With no settings, the checker draws every
real argument uniformly from -1000 to 1000, every array is 8 long, and it
does this 10 times per procedure. That is enough to catch a wrong
translation, but -1000 kelvin is not a temperature, and you will trust the
verdict more if it was reached on inputs your code is meant for.

This is what the settings file is for. Save this as `satvap/recast.json`:

```json
{
  "stages": {
    "differential.bitexact": {
      "trials": 20,
      "dims": {"n": 16},
      "ranges": {"t": [220.0, 320.0], "p": [300.0, 1050.0], "rh": [0.0, 1.0]}
    }
  }
}
```

`trials` is how many random inputs per procedure, `dims` fixes any integer
that sizes an array (here `n`), and `ranges` gives each argument its physical
range, by name. Arguments you do not list keep the default. Run again, with
the file:

```console
$ recast run translate satvap --config satvap/recast.json
...
  [ok ] verifier   differential.bitexact       bit_exact: 340 points across 2 subprogram(s), all bit-exact
...
1 unit(s), 3 verdict(s), all passed
```

340 points: 20 calls of `esat`, and 20 calls of `mixing_ratio` with 16
elements each. The `examples/toy_physics/recast.json` you passed in section
3 is the same file for that module.

## 5. When it is not all green

Three things you will run into. Each one is the engine being honest rather
than broken, and each has a short answer.

### A block the engine refused to translate

The engine translates by rules, and a Fortran feature it has no rule for is
left in the Python as a marked refusal rather than a guess. Timers are a
common one. This module wraps a loop in `cpu_time`:

```fortran
module timed
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine sum_sq(n, x, s, secs)
    integer, intent(in) :: n
    real(r8), intent(in) :: x(n)
    real(r8), intent(out) :: s, secs
    real(r8) :: t0, t1
    integer :: i
    call cpu_time(t0)
    s = 0.0_r8
    do i = 1, n
      s = s + x(i)**2
    end do
    call cpu_time(t1)
    secs = t1 - t0
  end subroutine sum_sq
end module timed
```

```console
$ recast run translate timed
fortran:timed
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             2 deferred block(s)
  [ok ] verifier   static.rwset                sampled: 3 blocks match
  [ok ] oracle     f2py-golden                 f2py:timed:e21d0bd034c1f0a5
  [FAIL] verifier   differential.bitexact       failed: nothing was compared; that is not a pass
  [ok ] store      fs-evidence                 2 verdict(s) recorded

1 unit(s), 2 verdict(s), FAILED
```

`2 deferred block(s)` are the two `cpu_time` calls. Open the Python and you
find, where each of them was:

```python
    # B001 <- L11-L11 AGENT_QUEUE: intrinsic subroutine 'cpu_time' has no rule
    raise NotImplementedError("intrinsic subroutine 'cpu_time' has no rule")  # B001
```

The loop itself was translated correctly. But a procedure that raises cannot
be run, so the checker skipped `sum_sq`, and since that was the only
procedure, it had nothing to compare and said so. A check that compared
nothing is not a pass, and the run reports `FAILED`.

What to do: keep timing, printing and file I/O out of the numerical routine
you want translated, and translate the numerics. Here, `sum_sq` without the
two `cpu_time` lines and the `secs` argument goes through bit-exact. If the
refused feature is one your science genuinely needs, the refusal message is
the thing to paste into an issue on the engine's repository, so that a rule
can be added.

### The Fortran does not parse

```console
$ recast run translate broken
fortran:broken
  [FAIL] frontend   fortran                     fortran:broken: .../broken/broken.f90 did not parse -- FortranSyntaxError: at line 9
>>>  ! missing: end module

1 unit(s), 0 verdict(s), FAILED
```

The first stage failed, and nothing after it ran. The message names the file
and the line. A quick way to find such problems before the engine does is
to compile the file on its own, since gfortran's messages are more detailed:

```bash
gfortran -fsyntax-only satvap/satvap.f90
```

(It leaves a small `satvap.mod` file behind, which you can delete.)

### The file is not a module

A file holding bare subroutines, with no `module ... end module` around
them, is read but not translated:

```console
$ recast run translate bare
fortran:bare
  [ok ] frontend   fortran
  [skip] transform  translate.numpy             not applicable to this unit
  [FAIL] verifier   static.rwset                no candidate to verify; transform never ran
  [ok ] store      fs-evidence                 0 verdict(s) recorded

1 unit(s), 0 verdict(s), FAILED
```

Wrap the procedures in a module, as in section 4, and run again.

### What the last line means

| the run ends with | meaning | exit code |
|---|---|---|
| `all passed` | every check ran and every check passed | 0 |
| `FAILED` | a check ran and found a difference, or had nothing to compare | 1 |
| `INCOMPLETE -- something could not run, ...` | a stage could not run at all, so nothing was decided | 2 |

The exit code is for scripts and CI; the word is for you. Note that
`INCOMPLETE` is not a pass either: a check that could not run has not
checked anything, and the run lists which stage could not run.

## 6. Common problems

**`recast: command not found`.** The environment is not active in this
terminal. Run `source .venv/bin/activate` from inside the `RecastEngine`
folder.

**`Fortran compiler 'gfortran' is not runnable`.** The oracle stage fails
with this when no `gfortran` is on your PATH:

```console
  [ok ] verifier   static.rwset                sampled: 3 blocks match
  [FAIL] oracle     f2py-golden                 Fortran compiler 'gfortran' is not runnable ([Errno 2] No such file or directory: 'gfortran'); install gfortran or point config['fc'] at one
```

Install it as in section 1 and open a new terminal.

**The run says `all passed` but the Python looks stale.** Each run writes
under `output/<folder name>/`. If you edited the Fortran and are unsure
which run you are looking at, delete `output/satvap` and run again; it is
safe to delete all of `output/` at any time. Your Fortran is never written
to.

**One module that `use`s another.** Put both files in the same folder,
and say `"target": "tree"` in the settings file. Without it, each module is
translated and checked on its own, and the one that imports its sibling
fails its check with `candidate does not import: No module named
'<sibling>_numpy'`; with it, the sibling's translation is bundled into the
candidate so the check can run. A `dewpoint.f90` that begins with
`use satvap, only: esat, r8`, beside the `satvap.f90` above, with this
`recast.json`:

```json
{
  "target": "tree",
  "stages": {
    "differential.bitexact": {
      "dims": {"n": 16},
      "ranges": {"t": [220.0, 320.0], "p": [300.0, 1050.0], "rh": [0.05, 1.0]}
    }
  }
}
```

```console
$ recast run translate pair --config pair/recast.json
fortran:dewpoint
  [ok ] frontend   fortran
  [ok ] transform  translate.tree
  [ok ] verifier   static.rwset                sampled: 3 blocks match
  [ok ] oracle     f2py-golden                 f2py:dewpoint:6cd05706f7b310c3
  [ok ] verifier   differential.bitexact       bit_exact: 10 points across 1 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded
fortran:satvap
  [ok ] frontend   fortran
  [ok ] transform  translate.tree
  ...
  [ok ] verifier   differential.bitexact       bit_exact: 170 points across 2 subprogram(s), all bit-exact
  ...

2 unit(s), 6 verdict(s), all passed
```

Each module is still checked as its own unit, against its own compiled
Fortran; `--unit fortran:dewpoint` picks one. The `rh` range starts above
zero on purpose: `dewpt` takes a logarithm of `rh`, and where Fortran's
`log` of a negative number quietly returns NaN, Python's raises, so a range
that reaches below zero makes the check stop with `math domain error`
instead of comparing. Ranges are where you say what the code is for. For a
real model tree, with constants modules and framework calls, read
[`tree-units.md`](tree-units.md).

## 7. Where to go next

| | |
|---|---|
| [`cli.md`](cli.md) | every command and every key the settings file accepts |
| [`corpus-numfor-example.md`](corpus-numfor-example.md) | the same walk over a real third-party library, and a careful reading of how much a passing run actually proves |
| [`tree-units.md`](tree-units.md) | what changes when the Fortran is not one module but a model tree |
| [`roadmap.md`](roadmap.md) | how far each of the four recipes reaches today |
