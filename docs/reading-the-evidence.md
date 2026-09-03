# Reading the evidence: what "all passed" actually claims

[`getting-started.md`](getting-started.md) ends with a run that says
`all passed`. Sooner or later someone asks what that means: a co-author, a
reviewer, or you six months from now with a Python file and no memory of
how it was checked. This page is the answer. It shows the two records a run
leaves behind, what each field in them says in plain words, which of them
to keep with your code, and, just as important, what a passing run does
*not* claim.

It uses the `satvap` module from the getting-started page, with its
`recast.json`. If you have not made that, make it now; every block below
is the engine's real output over it.

## 1. Two records, for two readers

The console output is gone when you close the terminal. What stays is:

| | where | what it is for |
|---|---|---|
| the **evidence manifests** | `output/satvap/evidence/fortran_satvap/*.json` | one file per check per run, never overwritten, with the full detail. The audit trail. |
| the **summary** | wherever you ask for it with `--summary` | one file, regenerated each run, holding only what two correct runs agree on. The thing you commit next to your code. |

The first accumulates; the second replaces. The rest of this page takes
them in the order you will want them: the summary first, because it is
short and it is the one you keep, then the manifests for when you need the
detail.

## 2. The summary file

Ask for it with one more flag:

```console
$ recast run translate satvap --config satvap/recast.json --summary satvap/verification.json
...
1 unit(s), 3 verdict(s), all passed
```

`satvap/verification.json` now holds this (one check trimmed to save
space):

```json
{
  "recipe": "translate",
  "schema": 1,
  "units": [
    {
      "unit": "fortran:satvap",
      "transform": "recast.translate.fortran-to-numpy",
      "candidate": "09093158b3e3553b1465378c7aa64c5119295e17ed343f1d9efcfecc2fa12985",
      "deferred": 0,
      "oracle": "f2py-golden",
      "stopped_by": null,
      "verdicts": [
        {
          "verifier": "static.rwset",
          "passed": true,
          "confidence": "sampled",
          "detail": "3 blocks match",
          "metrics": {"blocks_checked": 3, "blocks_matched": 3, "blocks_deferred": 0, "blocks_waived": 0}
        },
        {
          "verifier": "differential.bitexact",
          "passed": true,
          "confidence": "bit_exact",
          "detail": "340 points across 2 subprogram(s), all bit-exact",
          "metrics": {"points": 340, "bit_exact": 340, "max_ulp": 0, "max_rel": 0.0, "nan_mismatch": 0, "trials": 20}
        },
        { "verifier": "symbolic.notary", "passed": true, "confidence": "symbolic", "...": "..." }
      ]
    }
  ]
}
```

### The unit

| field | in plain words |
|---|---|
| `unit` | which Fortran module this entry is about |
| `candidate` | a fingerprint (SHA-256) of the generated Python. It covers the unit's name, the transform's name, and every generated file's name and content, so it names *exactly which translation* the verdicts below are about. Change one character of the Fortran and this changes. |
| `deferred` | how many blocks the engine refused to translate. 0 is what you want; anything else means a `raise NotImplementedError` is sitting in the Python (see the getting-started page, section 5). |
| `oracle` | what the translation was compared against. `f2py-golden` means your own Fortran, compiled. |
| `stopped_by` | `null` when every stage ran. Otherwise the name of the check that stopped the unit. |

### The verdicts

Each check writes one verdict with the same four fields: `verifier` (which
check), `passed` (true or false), `confidence` (what kind of claim it
makes), `detail` (the sentence the console printed) and `metrics` (the
numbers behind the sentence).

`confidence` is the word to read first. It says how strong the claim is,
and each check has its own ceiling:

| verifier | confidence when it passes | what it means |
|---|---|---|
| `static.rwset` | `sampled` | each translated block reads and writes the same variables as the Fortran block it came from. Says nothing about the arithmetic. |
| `differential.bitexact` | `bit_exact` | on the inputs it tried, Fortran and Python produced identical bits |
| `symbolic.notary` | `symbolic` | any arithmetic the translation reordered was checked to be equivalent (here, none was) |

The differential check is the one that says the translation is *right*, so
its metrics deserve a closer look:

| metric | in plain words |
|---|---|
| `trials` | how many random input sets were tried per procedure (from your `recast.json`, or 10 by default) |
| `points` | how many output values were compared in total. Here 340: `esat` returns one number and was called 20 times, `mixing_ratio` returns an array of 16 and was called 20 times. |
| `bit_exact` | how many of those were identical to the last bit. Equal to `points` when all passed. |
| `max_ulp` | the largest difference found, in units of the last place of the floating-point number. 0 means no difference at all. |
| `max_rel` | the same, as a relative error |
| `nan_mismatch` | how many points where one side produced NaN and the other did not. A non-zero value here usually means a variable read before it was assigned. |
| `integer_points` | how many of the compared values were integers (an `integer` result or out-argument). They are compared for equality, and never contribute to `max_ulp` or `max_rel`. |
| `integer_mismatch` | how many of those integer values differed. Any non-zero value fails the check. |
| `skipped` | public procedures the oracle offered that the check did not try: one the harness cannot generate inputs for (a `character` argument), or one left out of `subprograms` in your `recast.json`. Not a pass -- see `uncovered`. |
| `uncovered` | translated public procedures nobody compared, by name. Any entry here fails the check: a procedure that was translated and never checked is a claim without evidence. Private helpers do not appear -- they run inside the public procedures that call them -- and a procedure the oracle listed as `ungated` (with its reason) is reported on the verdict rather than counted here. |

For `static.rwset`: `blocks_checked` is how many blocks were compared,
`blocks_matched` how many agreed, `blocks_deferred` how many were refused
translations and so had nothing to compare, `blocks_waived` how many were
excused by configuration (normally 0).

### What a failing summary looks like

The `timed` module from the getting-started page, the one with `cpu_time`
in it, summarizes as:

```json
{
  "unit": "fortran:timed",
  "deferred": 2,
  "stopped_by": "differential.bitexact",
  "verdicts": [
    { "verifier": "static.rwset", "passed": true, "confidence": "sampled",
      "metrics": {"blocks_checked": 3, "blocks_matched": 3, "blocks_deferred": 2, "blocks_waived": 0} },
    { "verifier": "differential.bitexact", "passed": false, "confidence": "failed",
      "detail": "nothing was compared; that is not a pass",
      "metrics": {"points": 0, "bit_exact": 0, "trials": 10} }
  ]
}
```

`deferred: 2` and `stopped_by` name the problem, `points: 0` says how much
was actually checked, and `symbolic.notary` is absent because the run
stopped before it. A summary is honest about failure in the same shape it
reports success.

## 3. Why the summary is the file to keep

Run the same command again and compare:

```console
$ cp satvap/verification.json before.json
$ recast run translate satvap --config satvap/recast.json --summary satvap/verification.json
1 unit(s), 3 verdict(s), all passed
$ diff before.json satvap/verification.json
$
```

No difference: the file is byte-for-byte the same. That is by design. The
summary leaves out everything that varies between two correct runs, such as
timestamps, file paths, the machine, and the compiler version, and keeps
only what any correct run must agree on.

Now change the Fortran. Edit `243.5_r8` to `243.0_r8` in `satvap.f90`, run
again, and diff:

```console
$ diff before.json satvap/verification.json
6c6
<       "candidate": "09093158b3e3553b1465378c7aa64c5119295e17ed343f1d9efcfecc2fa12985",
---
>       "candidate": "466eb2026558033a726167172915459948318cf0da905c46324b11127330fd7c",
```

Only the fingerprint moved. The metrics are the same, because the changed
module also translates bit-exactly. But the file now says, correctly, that
a *different* translation was verified. (Change it back before going on.)

This is what makes the summary worth committing alongside your code. Keep
these four things together, in version control:

| | |
|---|---|
| `satvap.f90` | the Fortran |
| `recast.json` | the input ranges the check used |
| `verification.json` | the summary |
| `satvap_numpy.py`, `satvap_constants.py` | the Python you are shipping, copied out of `output/` |

Someone who doubts the Python can install the engine, run the same command,
and diff the summary. If it matches, they have reproduced the claim on
their own machine, with their own compiler. A changed `candidate` line
tells them the Python they were given is not the one that was verified.

## 4. The evidence manifests

Each run also writes one JSON file per verdict into
`output/satvap/evidence/fortran_satvap/`, named by a hash of its content
and never overwritten. After the runs above there are a dozen or so files
there. The run prints the paths of the ones it just wrote:

```console
  evidence: file:///.../output/satvap/evidence/fortran_satvap/ce3b69a9....json
```

or, to see the newest three:

```bash
ls -t output/satvap/evidence/fortran_satvap | head -3
```

A manifest carries everything the summary has, plus the things the summary
deliberately leaves out. The differential one for `satvap`:

```json
{
  "schema_version": 1,
  "evidence_class": "complete",
  "artifact": {
    "name": "fortran:satvap",
    "transform": "recast.translate.fortran-to-numpy",
    "digest": "09093158b3e3553b1465378c7aa64c5119295e17ed343f1d9efcfecc2fa12985",
    "files": ["satvap_constants.py", "satvap_numpy.py"]
  },
  "reference": {"oracle": "f2py-golden", "key": "f2py:satvap:8c6c49ccd27d2248"},
  "environment": {"engine": "recast 0.0.1.dev0", "platform": "macOS-26.5.2-arm64-arm-64bit", "python": "3.11.16"},
  "cc_test": {"commit": "unknown", "version": "unknown"},
  "cases": [],
  "result": {
    "verifier": "differential.bitexact",
    "verdict": "bit_exact",
    "passed": true,
    "detail": "340 points across 2 subprogram(s), all bit-exact",
    "metrics": {
      "points": 340, "bit_exact": 340, "max_ulp": 0, "max_rel": 0.0, "nan_mismatch": 0, "trials": 20,
      "skipped": [],
      "subprograms": {
        "esat":         {"points": 20,  "bit_exact": 20,  "max_ulp": 0, "max_rel": 0.0, "nan_mismatch": 0},
        "mixing_ratio": {"points": 320, "bit_exact": 320, "max_ulp": 0, "max_rel": 0.0, "nan_mismatch": 0}
      }
    }
  },
  "timestamp": "2026-09-02T02:38:29.298123+00:00"
}
```

What is here and not in the summary:

| field | in plain words |
|---|---|
| `artifact.files` | which generated files the fingerprint covers |
| `reference.key` | the compiled reference's identity. It folds in the compiler version, which is why it is *not* in the summary: your gfortran and a colleague's differ, and the key would differ while nothing was wrong. |
| `environment` | engine version, operating system, Python |
| `timestamp` | when |
| `metrics.subprograms` | the same numbers, **per procedure** |
| `metrics.skipped` | procedures the check could not run and so did not judge |
| `evidence_class` | `complete` when the engine drove every step itself, which is always the case for a local run |
| `cc_test` | filled in when the run happens inside the CC-Test pipeline; `unknown` on your laptop |

The per-procedure breakdown and the `skipped` list are the two things to
open a manifest for. The summary says 340 points passed; the manifest says
which procedures they came from, and names any procedure that was left
out. A module with three procedures whose manifest shows points for two of
them and one name under `skipped` has been checked less than its summary
line suggests, and the manifest is where you find that out.

Two evidence folders are worth knowing about besides this one.
`output/satvap/evidence/` accumulates across runs, including the failed
attempts, because an audit trail that forgets its failures is not one. It
is safe to delete when you want a clean slate. And the manifests are never
compared to each other: two runs on two machines produce two different
manifests of the same verdict, and that is expected.

## 5. What a passing run does and does not claim

Reading `all passed` correctly means reading it narrowly. A run over
`satvap` with the `recast.json` above establishes:

- For 20 random input sets with `t` in 220 to 320 K, `p` in 300 to 1050
  hPa, `rh` in 0 to 1, and `n` equal to 16, the Python and the compiled
  Fortran produce identical bits for every output of `esat` and
  `mixing_ratio`.
- Every translated block reads and writes the same variables as its
  Fortran block.
- No arithmetic was reordered in translation.
- No block was refused.

It does not establish:

- Anything about inputs outside those ranges. `esat` has a pole at
  `tc = -243.5`; the check never went there, and a translation that was
  wrong only there would pass. Widen the ranges if that matters to you.
- Anything about a procedure listed under `skipped`, or about a procedure
  that is `private` in the module. The reference is built with f2py, which
  wraps only what the module makes public, so a private helper is exercised
  only through the public procedures that call it.
  [`corpus-numfor-example.md`](corpus-numfor-example.md) has a real case of
  this, where one public procedure out of thirteen is all the check saw.
- Anything about a block that was `deferred`. The Python raises there.
- Anything about a different compiler or different flags. The reference is
  gfortran with f2py's defaults, and the summary records only that it was
  `f2py-golden`.
- Anything about speed. This is the NumPy translation, written for
  agreement with the compiler rather than throughput.

Three things strengthen the claim, all in `recast.json`: more `trials`,
`ranges` that cover what your science will actually feed the code, and more
than one array size if the code branches on it. The cost is only run time.

## 6. Where to go next

| | |
|---|---|
| [`getting-started.md`](getting-started.md) | the page before this one |
| [`cli.md`](cli.md) | every flag on `run`, including `--report-only` for when you want the report without the exit code |
| [`corpus-numfor-example.md`](corpus-numfor-example.md) | a manifest read carefully on a real library, where the passing run reaches less far than its summary line |
| [`../examples/README.md`](../examples/README.md) | the long form of why the summary omits what it omits, and why CI diffs it |
