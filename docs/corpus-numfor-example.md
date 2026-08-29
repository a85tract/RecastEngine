# A worked example: numfor

This page runs the same eight stages over a module from the corpus — `basic`, from
[numfor](https://github.com/numericfor/numfor) — and stays with it long
enough to say what the passing run does and does not establish.

Every console block below is the shipped `translate` recipe, with no domain
extension installed and no stage configured — the single config file
involved sets the output directory and nothing else.

## Getting the source in place

What the tool wants is the upstream repository at `corpus/numfor`, at the
commit the corpus pins. Two ways to put it there. Clone it:

```bash
git clone https://github.com/numericfor/numfor.git corpus/numfor
git -C corpus/numfor checkout 65ee8b75b22ad300b54c87d632b3ebcd87de4b7c
```

Or, if you already have it — you are in a clone of this repository, where
`corpus/numfor` is registered as a submodule:

```bash
git submodule update --init --depth 1 corpus/numfor
```

Then stage the case:

```bash
python tools/corpus.py stage numfor
```

`stage` copies what `cases.json` lists — for `numfor`, the 133 `.f90` and
`.inc` files under `src/`, its test tree left out — flat into a fresh
`output/numfor/staged/`. Flat because an `#include "qtrs1d.inc"` names no
directory. `corpus/numfor` is only ever read from.

`stage` also leaves a `recast.json` beside the sources, pinning the run's
output to `output/numfor/`. The command below passes it.

## A successful run

```console
$ recast run translate output/numfor/staged \
      --config output/numfor/staged/recast.json --unit fortran:basic
fortran:basic
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             3 deferred block(s)
  [ok ] verifier   static.rwset                sampled: 57 blocks match
  [ok ] oracle     f2py-golden                 f2py:basic:f4038505...
  [ok ] verifier   differential.bitexact       bit_exact: 10 points across 1 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

`basic` is numfor's 354-line utility module — kinds, a timer derived type,
a date stamp, `is_inf`. Two things to open under `output/numfor/`:

| | |
|---|---|
| `translate/fortran_basic/candidate/basic_numpy.py` | the generated Python, every block carrying the source lines it came from |
| `evidence/fortran_basic/*.json` | one manifest per verdict — artifact digest, oracle key, metrics |

### Five blocks the rules would not guess

The `3 deferred block(s)` are the rules declining to guess: two `cpu_time`
calls and one `date_and_time`. A refusal is
left standing in the output as a raise, tagged for whoever answers it:

```text
    # B006 <- L217-L217 AGENT_QUEUE: intrinsic subroutine 'cpu_time' has no rule
    raise NotImplementedError("intrinsic subroutine 'cpu_time' has no rule")  # B006
```

That is not a wrong translation, and it is not a silent one. The other 57
blocks in the module are translated, and checked.

### How far the passing run actually reaches

`all passed` is a claim about three verifiers, and they do not cover the same
ground:

| verifier | what it covered |
|---|---|
| `static.rwset` | 57 of the module's 60 blocks — reads and writes agree with the source's |
| `differential.bitexact` | **one** subprogram, `is_inf`, 10 points, `max_ulp: 0` |
| `symbolic.notary` | 0 rewrites to notarize — the translation reorders no output |

The differential gate is the one that says a translation is *right*, and here
it saw one of the module's thirteen procedures. The reason is visibility, not
sampling: the reference is an f2py build of the untouched Fortran, f2py wraps
what the module makes public, and `basic.f90:74` declares `private` and then
exports exactly two procedures — `is_inf` and `print_msg`. `print_msg` holds
one of the three deferred blocks, so the gate skips it and says so
(`"skipped": ["print_msg"]`). The other eleven — the timer type's bound
procedures and the helpers around them — are private, and never reach the
oracle at all.

So: the module imports, its dataflow agrees with the source's, and the one
piece of it that could be executed against compiled Fortran matches bit for
bit. It does not say the timer procedures are correct. Nothing ran them.

### The evidence

Each verdict lands as its own content-addressed manifest. The differential
one, trimmed:

```json
{
  "artifact": {
    "digest": "d0cb59d70368a5ba9e04e8e6edad2c300b0b1ee0a5acc2cd99f57141d09477d2",
    "name": "fortran:basic",
    "transform": "recast.translate.fortran-to-numpy"
  },
  "reference": { "key": "f2py:basic:f40385059eedf8e4", "oracle": "f2py-golden" },
  "result": {
    "verdict": "bit_exact",
    "verifier": "differential.bitexact",
    "passed": true,
    "metrics": {
      "points": 10, "bit_exact": 10, "max_ulp": 0, "nan_mismatch": 0,
      "subprograms": { "is_inf": { "points": 10, "bit_exact": 10 } },
      "skipped": ["print_msg"]
    }
  }
}
```

The digest is over the generated files, so the manifest names the artifact it
judged rather than the path it sat at. The oracle key folds the compiler's
version — which is why a manifest is a record of one run and is never
diffed against another's ([`examples/README.md`](../examples/) has the
long form of that argument).

## A failed run

`array_utils`, from the same case, translates and then does not get past
the first check:

```console
$ recast run translate output/numfor/staged \
      --config output/numfor/staged/recast.json --unit fortran:array_utils
fortran:array_utils
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             2 deferred block(s)
  [FAIL] verifier   static.rwset                failed: 4/68 blocks disagree: save_array1d/B005, save_array1d/B021, save_array2d/B016, save_array2d/B017
  [ok ] store      fs-evidence                 1 verdict(s) recorded

1 unit(s), 1 verdict(s), FAILED
```

The run exits 1, and the stages after the failed verifier never run: no f2py
build, no differential, one verdict recorded instead of three. The manifest
names what disagreed, per block and per symbol:

```json
{
  "block": "save_array1d/B021",
  "reads_source_only": ["unit_"],
  "reads_target_only": [],
  "writes_source_only": [],
  "writes_target_only": []
}
```

The source block reads `unit_`; the translated one does not. The gate does
not say which side is wrong — a defect in the translation, or a limit of the
read/write analysis — and does not need to. It fails closed, and 64 of the
68 blocks matching does not buy the other four.
