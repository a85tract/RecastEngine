# Trusted base

What must be believed, beyond Lean's kernel, for a `proved` status in
`RULES.md` to mean "the emitted Python computes the same mathematical value
as the Fortran source".

## 1. Lean, Mathlib, and the axiom audit

* Lean 4 `v4.34.0`, Mathlib `v4.34.0` (pinned in `proofs/lean/lean-toolchain`
  and `lake-manifest.json`). `proofs/tools/validate.py` rebuilds the library,
  greps for `sorry`, and runs `#print axioms` on every theorem the ledger
  cites; only `propext`, `Classical.choice`, `Quot.sound` are accepted. No
  `native_decide`, no `axiom`, no `opaque` in the artifact.

## 2. The two semantics (SEMANTICS.md)

The formal claims are about the *models*. Model fidelity is checked, not
proved:

* Fortran: definitions cite F2018 clauses (integer division 10.1.5.2.2,
  intrinsic assignment 10.2.1.3, sections 9.5.3.3.2, DO 11.1.7.4). gfortran
  15.2.0 at `-O0 -fcheck=all` is used as the concrete reference in every
  probe; the exact model and gfortran agree on every probe where the source
  is defined and the values are representable.
* Python/NumPy: `Py.adjust`/`Py.sliceLen`/`Py.sliceIndices` transcribe
  `PySlice_AdjustIndices` and were checked against CPython 3.12 on the
  boundary cases (negative edges, empty slices, clipping); `P.arith` is
  CPython `int`/`float` arithmetic in exact form; NumPy contracts assumed:
  `a[...] = b` copies through a temporary on overlap, integer-dtype stores
  truncate (not modelled: the Python array model has no dtype, see
  SEMANTICS.md idealization 9), `np.reshape(order='F')` and
  `ravel(order='F')` are the column-major storage relation.
* A model can be wrong, and one was: Semantics v1 encoded Python's `& | ^`
  on ints with `toNat`, which is only right for non-negative operands. It was
  found while extending the R0090 proof to negative operands and corrected in
  v1.1 (SEMANTICS.md change log). Every theorem that depended on it was
  re-checked by the rebuild; the correction is the kind of error the
  validation roadmap (executable semantics, differential tests) is meant to
  catch systematically rather than by inspection.

## 3. Transcription of the runtime helpers

`P.helper` (Semantics/Python.lean) transcribes the bodies of the runtime
shims from `src/recast/transform/numpy/runtime.py` at the baseline commit
(`_f_int_div = int(a / b)`, `_f_int` with int32 saturation, `_f_mod`,
`_f_modulo`, `_f_sign`, `_f_dim`, `_f_max/_f_min`, `_f_vsum`, `_f_vdot`,
`_f_powi`, `_f_lbound`, `_f_kind`, `_f_huge`, `_f_tiny`, `_f_epsilon`,
`_f_precision`, `_f_iand/_f_ior/_f_ieor/_f_ishft`, `_f_rstep_lb`,
`_f_rstep_any`, `_f_sqrt`, `_f_log`, `_f_log10`, `_f_cfold`; v1.1:
`_f_trim`, `_f_adjustl`, `_f_index`, `_f_scan`, `_f_eoshift`, `_f_mvbits`,
and `_f_seq_tail` in `Rules/SeqAssoc2.lean`). Each row cites the runtime
line span in the ledger record that uses it, and the span's SHA-256 at the
baseline commit is stored in `rules.json`. The transcription is exact
arithmetic: `int(a / b)`'s float intermediate is *not* modelled (recorded as
concrete-execution defect `FNP-D0035`).

Library calls the emitter spells directly are modelled from their documented
behaviour (NumPy 2.5 / CPython 3.12 reference): `np.reshape(order='F')`,
`np.roll`, `np.argmax`/`np.argmin` + `np.unravel_index` (row-major first
extremum), boolean-mask indexing `a[m]` and `a[m] = v` (row-major
enumeration), `np.asfortranarray(a.T)`, `np.matmul`, `np.zeros`, `np.array`,
`np.shape`, `chr`, `ord`, `str` slicing and `+`. The two behaviours that
decide new defects — row-major enumeration in boolean indexing and in
`argmax` — were confirmed end-to-end (`probes/packorder`).

The constants pipeline (a second, token-level translator) is modelled in
`Rules/Constants.lean` (`Tr.tokExpr`, `Tr.transParam`, `Tr.transUseConst`,
`Tr.stateInit`) from `transform/numpy/constants.py`, `fortran/expr.py`,
`fortran/constants.py` and `transform/numpy/modules.py`, with the same
reading-based correspondence as §4.

## 4. The modelled emitter

`Translation/Expr.lean` (`Tr.trans`, `Tr.transPow`, `Tr.transCall`,
`Tr.transSubs`, …) and `Rules/StmtSoundness.lean` (`transStmt`) reproduce
the *decisions* of `expressions.py` and `statements.py`: which spelling is
chosen under which static condition (`isInt (tyOf …)`, `rankOf`, profile
switches, literal folds). The correspondence between the model and the code
is established by reading (locations cited per rule, hashes pinned) and by
running the baseline engine on every probe/example and comparing the emitted
text with the modelled output shape (`FNP-kernel`: `saxpyP` in
`Rules/Kernel.lean` is the emitted statement). The emitter's *static typing
oracle* `tyOf` is the true Fortran typing under the declaration environment;
where the engine's analysis deviates (companion parameters, `FNP-D0010`),
that is an applicability defect, recorded, not modelled.

## 5. Name protocol and scaffolding

Names are translated by `names.py` (`pysafe`, keyword avoidance, hoisted
literal names). The model treats the protocol as the identity; injectivity
of the actual protocol on a module's names is an assumption recorded with
FNP-R0010/R0187. Module scaffolding (constants module, prologue zeroing,
copy-out conventions, deferred blocks) is outside the statement fragment and
carried as `specified`/`external_contract` rules with their own probes
(`copyout`, `save`, `companion_div`, `keywords`).

## 6. Frontend

`fparser` 0.2.5 parsing and the engine's `FortranFrontend`/`analyze` facts
are trusted to reproduce the Fortran syntax tree; the Lean syntax is an
abstract syntax at the level of the engine's own AST walk.

## 7. Oracle congruence

`erf`, `erfc`, `gamma`, `atan2` are uninterpreted on both sides
(`Oracles` structure, FNP-R0076/R0075): the theorems say the two sides call
the *same* function on the *same* arguments; that `math.erf` and
`libgfortran`'s `erf` compute the same real is a libm question outside the
model.

## 8. The verifier-evidence defect

The engine's symbolic notary (`verify/notary.py`) is **not** part of the
trusted base and is not used as evidence here: it samples with mpmath and
accepts relative errors below 1e-45 (`FNP-D0034`, reproduced in
`proofs/defects/FNP-D0034/reproduction.log`). Exactness claims in this
artifact rest on the Lean theorems only.
