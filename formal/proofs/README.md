# Lean model of the NumPy emitter's rules

A Lean 4 / Mathlib development that gives the Fortran fragment and the emitted
Python/NumPy subset each an exact-arithmetic semantics, states the NumPy
emitter's translation rules (`FNP-Rxxxx`) as theorems between them, and proves
or refutes them. `witnesses/` holds one Lean check per recorded emitter defect
(`FNP-Dxxxx`): the declarations that exhibit it, and an axiom audit of them.

## What it does and does not establish

- The claims are about the *models*. [SEMANTICS.md](SEMANTICS.md) defines them
  and [TRUSTED_BASE.md](TRUSTED_BASE.md) lists what must be believed beyond
  Lean's kernel. Both are carried as written in the development this library
  comes from; the ledger files and validation script they mention (`RULES.md`,
  `rules.json`, `proofs/tools/validate.py`) are not part of this directory.
  `check_witnesses.py` is the axiom audit here.
- The models were written against an earlier revision of the emitter. A
  compiled theorem does not show that today's `src/recast/transform/numpy/`
  behaves as modelled, and it is not a runtime certificate for a translation.
  The engine's guarantee for a translated unit is still its verifier's
  `Verdict` (`docs/reading-the-evidence.md`).
- A refutation (`FNP_Rxxxx_refuted`) refutes the rule as the model states it,
  with a concrete counterexample; `witnesses.json` gives each defect's
  counterexample in source terms. A `related_only` witness checks declarations
  related to the defect, not the defect itself.
- Several rules are stated without a proof (`Rules/Obligations.lean`).

The Python tests that pin these defects in the emitter
(`tests/test_fnp_*.py`) name the same `FNP-Dxxxx` records.

## Build and check

Needs [elan](https://github.com/leanprover/elan); `lean-toolchain` pins Lean
4.34.0 and `lake-manifest.json` pins Mathlib.

```sh
cd formal/proofs
lake exe cache get        # Mathlib's prebuilt cache
lake build                # the RecastProofs library
python3 check_witnesses.py
```

`check_witnesses.py` fails on a Lean error, on a witness missing from or
unlisted in `witnesses.json`, on any `sorry`, `admit`, `native_decide` or
`axiom` in `RecastProofs/`, and on a witness theorem that depends on an axiom
other than `propext`, `Classical.choice` and `Quot.sound`. CI runs it in the
`formal` job, which no engine job depends on.

## Layout

| Path | What |
|---|---|
| `RecastProofs/Syntax/`, `Semantics/` | the two languages and their semantics |
| `RecastProofs/Translation/Expr.lean` | the modelled emitter |
| `RecastProofs/Rules/` | one theorem (`_sound`, `_refuted`, ...) per rule |
| `RecastProofs.lean` | the aggregate |
| `witnesses/FNP-Dxxxx.lean` | one defect's declarations and axiom audit |
| `witnesses.json` | each witness's title, rules, kind and counterexample |

Lean sources by Qinrun Dai (second5t) and Yueqi Chen.
