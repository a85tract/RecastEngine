# Semantics version 1.2 — exact-arithmetic models of the Fortran source fragment and the Python/NumPy target

Everything in `proofs/lean/RecastProofs/Semantics/` and `Syntax/`. The two
languages are modelled **independently**: no definition in a Fortran file
mentions Python or vice versa; the only shared objects are the mathematical
carriers (ℝ, ℤ, lists) and the uninterpreted primitive oracle
(`Oracle.prim : String → List ℝ → Option ℝ`) for functions neither Mathlib nor
the model defines (`erf`, `erfc`, `gamma`, `atan2`).

## Value domains (`Semantics/Values.lean`, `Semantics/Literal.lean`)

| Concept | Fortran side | Python side |
| --- | --- | --- |
| Integers | `SVal.int (k : IKind) (v : ℤ)` — unbounded ℤ, kind `i4`/`i8` retained as metadata | `PVal.int (v : ℤ)` — unbounded (Python `int`) |
| Reals | `SVal.real (k : RKind) (v : ℝ)` — exact real, kind `r4`/`r8` retained | `PVal.float (v : ℝ)` — exact real (no rounding) |
| Logical / bool | `SVal.bool` | `PVal.bool` |
| Character | `SVal.str (List Char)`; declared length `Decl.clen`; assignment pads/truncates (`convertTo`) | `PVal.str (List Char)`; no padding |
| Arrays | `FArr`: `elemTy`, `dims : List (ℤ × ℕ)` (lower bound, extent) per axis, `get : List ℤ → SVal` on 1-based-by-declaration coordinates; `inBounds` | `PArr`: `shape : List ℕ`, `get : List ℕ → PVal` on 0-based coordinates |
| Literals | `Decimal ⟨mant, exp10⟩` with `toReal = mant · 10^exp10` — the **exact decimal policy**: a literal denotes its decimal, never a binary rounding; kinds on integer literals (`_8`) retained | The same `Decimal` for `float` tokens; `np.float64('text')` denotes the decimal of the text |

Division, remainders and rounding are the two languages' own definitions:

| Operation | Fortran | Python |
| --- | --- | --- |
| Integer `/` | `tquot = Int.tdiv` (truncation, F2018 10.1.5.2.2) | `//` is `fquot = Int.fdiv` (floor); `/` is true division on ℝ |
| `MOD` / `%` | `tmod` (sign of dividend); `MODULO` = `fmod` | `%` = `fmod` (sign of divisor) |
| `INT`, `int()` | `truncR` (toward zero) | `truncR` |
| `NINT` | `nintR`: half away from zero | `np.round`: `roundHalfEvenR` |
| Powers | `i**n`: `ipowNeg` (integer quotient, `1/(a**|n|)`), `x**n` real base: `zpowF` (undefined at `0**n, n ≤ 0`), `x**y`: `rpowF` (base > 0, or 0 with positive exponent) | `pyPow` on ints (negative exponent gives a float), `zpow`/`rpow` on floats |
| Bit intrinsics | `bitAndK/bitOrK/bitXorK/ishftK/mvbitsK` on the **kind's width** with two's-complement wrap (`wrapSigned`) | `_f_iand/_f_ior/_f_ieor` on plain ints are Python's infinite two's complement (`Int.land/lor/xor`, corrected in v1.1); `_f_ishft` is **unbounded** (`ishftUnbounded`) |
| Inquiries | `HUGE/TINY/EPSILON/PRECISION/DIGITS/KIND` are functions of the value's **kind tag** | `_f_huge` etc. dispatch on the Python **type** (`float` → binary64 values, else int32) |

Precision casts (`REAL`, `DBLE`, `np.float64`, `np.float32`) are identities on
values. Overflow does not exist in the model: `INT`'s int32 saturation in
`_f_int` is transcribed (it is a *helper* behaviour, not a rounding), and
any bound that only matters because of finite precision is a
concrete-execution (`FNP-D0035`-class) question outside the exact model.

## Evaluation (`Semantics/Fortran.lean`, `Semantics/Python.lean`)

Both evaluators are partial (`Option`), structurally recursive over the mutual
syntax (`Expr`/`Exprs`/`Sub`/`Subs`/`Bound`), and **total on their own terms**:
`none` means the source (or target) program is undefined at that point
(division by zero, `sqrt` of a negative, out-of-bounds subscript, type error).
The soundness obligations are one-directional: whenever the *source* is
defined, the target is defined and related. A defined target with an
undefined source carries no obligation (Fortran undefined behaviour).

Fortran specifics: `arith` implements kind promotion (`IKind.join`,
`RKind.join`); `compare` on characters blank-pads the shorter operand
(`strEq`); `lift1`/`lift2` are the elemental lifts with scalar broadcast;
`sectionOf` resolves `a(lo:hi:st)` and vector subscripts `a(v)` against the
array's **own declared bounds** (`resolveAxis`: omitted edges are the declared
bounds, F2018 9.5.3.3.2, every coordinate must be in bounds; `Axis.vector`,
9.5.3.3.3); `intrinsic` is the table of intrinsic functions actually cited by
rules (`intrinsicCore`, the v1 table, plus `intrinsicNew`, the v1.1
additions: character operations, RESHAPE/PACK/CSHIFT/EOSHIFT/MAXLOC/MINLOC/
TRANSPOSE/MATMUL/SHAPE/MVBITS); array constructors (`arrayCtor`) build a
unit-origin rank-1 array of homogeneous scalars; `convertTo` is intrinsic
assignment conversion (F2018 10.2.1.3); `storeArr` writes a scalar or a
conforming array into a selection; `assign` dispatches on the target's
binding.

Python specifics: `arith` is CPython/NumPy arithmetic on `int`/`float` with
promotion to float (and `str + str` concatenation); `compare` compares by
real value; `indexArr` applies `PySlice_AdjustIndices` semantics
(`Py.adjust`, `Py.sliceLen`, `Py.sliceIndices`, validated against CPython)
including **negative-index wrap and clipping**, and integer-array indices
(`PIdx.gather`, advanced indexing along one axis with negative wrap);
boolean-mask indexing `(a)[(m)]` (`maskSelect`) and masked stores `a[m] = v`
(`storeMask`) enumerate the selected elements in **row-major order**, as
NumPy does whatever the memory layout (confirmed end-to-end by probe
`packorder`); `helper` transcribes the bodies of the runtime shims
(`_f_int_div`, `_f_int`, `_f_mod`, `_f_vsum`, `_f_rstep_lb`, `_f_rstep_any`,
`_f_huge`, `_f_trim`, `_f_adjustl`, `_f_index`, `_f_scan`, `_f_eoshift`,
`_f_mvbits`, …) from `runtime.py` at the baseline commit and the library
calls the emitter spells (`np.reshape(order='F')`, `np.roll`,
`np.argmax`/`np.argmin` with `np.unravel_index` — **row-major first
extremum**, `np.asfortranarray(a.T)`, `np.matmul`, `np.zeros`, `np.array`,
`np.shape`, `chr`, `ord`, string slicing); `storeArr` performs NumPy
assignment into a selection.

Procedures (v1.2): `F.Proc` carries the subprogram's own declaration
environment, its dummy list with intents, its local scalars and its body;
`F.Exec.call` associates the actuals with the dummies positionally
(`bindDummies`, F2018 15.5.2: IN/INOUT/UNKNOWN take the actual's value, an OUT
dummy the zero of its declared type), zero-initialises the locals
(`initLocals`), runs the body under the callee's own declarations, and writes
the OUT/INOUT/UNKNOWN dummies back into the caller's variable actuals
(`copyBack`).  The declaration environment is therefore an *index* of `F.Exec`,
not a parameter.  `P.Proc` carries the emitted `def`'s parameters, the
prologue's initialisations and the names it returns; `P.Exec.callAssign`
evaluates the arguments in the caller's state, runs the body in a fresh scope,
and assigns the returned names to the targets.

Statements: big-step relations `F.Exec`/`F.Loop` (counted `DO` with trip
count `Fort.tripCount = max((hi - lo + st)/st, 0)` truncating, F2018
11.1.7.4.1, index set before each iteration and left one step past the last
on normal completion; `EXIT`/`CYCLE` outcomes) and `P.Exec`/`P.ForLoop`
(`for v in range(lo, hi, st)` over `pyRange`, with the optional `else`
clause that runs only on normal completion; `break`/`continue`). Version
1.1 adds, on the Fortran side, `FORALL` (`execForall`: every right-hand side
and subscript evaluated with the index bound *before* any store, F2018
10.2.4; the index is local to the construct), `WHERE`/`ELSEWHERE`
(`execWhere`: the mask evaluated once, masked whole-array assignments in
sequence, then the complement), `SELECT CASE` (`caseArm`: the arm's condition
as an `IF`, `caseCond`), `ASSOCIATE` with a *variable* selector (`renameS`:
the body with the name replaced by the selector) and `ALLOCATE`
(`execAllocate`: an array of the requested bounds); on the Python side the
masked store `a[m] = v` (`assignMask`). Divergence is not modelled: the
theorems speak about terminating source executions.

## The state relation (`Relations/State.lean`)

* `RelS`: scalars are related iff they denote the same mathematical value.
  A Fortran INTEGER relates to the Python `int` of the same value; a Fortran
  REAL relates to **any** Python number denoting the same real
  (`P.toR w = some v`) — a `float` or an `int` — because `x = 0` leaves a
  Python `int` in a REAL variable. LOGICAL/CHARACTER relate to equal
  `bool`/`str` (no padding on the Python side).
* `RelA`: `p.shape = a.extents` and, for every in-bounds Fortran coordinate,
  `RelS (a.get idx) (p.get (rebase a.lbs idx))` — the element at declared
  coordinates `idx` sits at 0-based coordinates `idx - lbs`.
* `RelState Γ σ τ`: every declared name is bound on both sides to related
  values.
* `Conforms o Γ σ` (source well-formedness): declared scalars hold scalars of
  the declared type; declared arrays hold homogeneous (`FArr.WellTyped`)
  arrays of the declared element type and rank whose lower bounds are the
  values of the declared bound expressions.

## Idealizations (recorded, not hidden)

1. Exact arithmetic on both sides: floating-point rounding, overflow,
   NaN/Inf and libm differences are outside the model. Equalities proved here
   are equalities of the *mathematical* functions the two programs compute;
   bit-exactness is a separate, concrete question (see `TRUSTED_BASE.md`).
2. The exact literal policy (a literal denotes its decimal) is what makes
   `0.1` mean the same on both sides; `np.float64('0.1')` is modelled as that
   decimal.
3. `MAXVAL`/`MINVAL` of an empty array are undefined in the model (Fortran
   defines them as `-HUGE`/`+HUGE`); `_f_vsum`/`np.prod` seeds are chosen from
   the first element's runtime type (NumPy uses the dtype) — both are
   documented simplifications with premises in the affected rules.
4. `_f_cfold` (compile-time MPFR folding) is modelled as the exact function.
5. Unstructured control (GOTO, named EXIT across loops via exceptions),
   derived types, pointers, I/O, procedures and modules are **not** in the
   Lean fragment; the ledger records them as `specified`, `semantic_gap` or
   `external_contract`.
6. (v1.1) `ALLOCATE` zero-fills the new array: Fortran leaves the values
   undefined, so a program that reads them has no obligation anyway; the zero
   makes both sides defined and related.
7. (v1.1) `WHERE` evaluates the right-hand side on every element before
   masking (Fortran evaluates only the masked elements, F2018 10.2.3.3); this
   is the definedness gap FNP-D0038, recorded on the WHERE rules.
8. (v1.1) `CSHIFT`, `EOSHIFT`, `MAXLOC`, `MINLOC`, `np.roll`, `_f_eoshift`,
   `np.argmax`/`argmin` are modelled on rank-1 arrays (or `DIM = 1`).
9. (v1.1) Python arrays carry no dtype, so a store of a REAL value into an
   INTEGER array is not cast (NumPy would truncate); the affected store rules
   carry an element-compatibility premise of origin `model_idealization`.
10. (v1.1) `ASSOCIATE` is modelled for scalar and array *variable* selectors by
    renaming on the Fortran side; NumPy's view aliasing (`n = x` shares
    storage for arrays) is not in the value model, so only the scalar
    refutation FNP-D0015 is claimed.

11. (v1.2) An OUT dummy is undefined on entry in Fortran; the model gives it
    the zero of its declared type, which is exactly what the emitted prologue
    assigns, so a callee that reads an OUT dummy before writing it is outside
    every obligation.  Only scalar OUT dummies and scalar locals are modelled.
12. (v1.2) A `RETURN` inside a subprogram is the body's `.exit` outcome; the
    Python side requires the callee body to finish normally.

## Change log

* **v1.1 (2026-09-18).** Correction: the Python model of `_f_iand`,
  `_f_ior`, `_f_ieor` on plain ints used `Int.ofNat (a.toNat &&& b.toNat)`,
  which is wrong for a negative operand; it is now Python's infinite two's
  complement (`Int.land`/`Int.lor`/`Int.xor`, Mathlib `Data.Int.Bitwise`).
  The v1 theorem for non-negative operands was re-proved and the signed
  range added (`FNP_R0090_sound_signed`). Additions: vector subscripts, array
  constructors, character intrinsics and substrings, the array transforms,
  MVBITS, and the statements FORALL, WHERE, SELECT CASE, ASSOCIATE, ALLOCATE
  (both languages, still defined independently); boolean-mask indexing and
  masked stores; the v1 intrinsic table is unchanged (`intrinsicCore`) and
  the additions are dispatched to `intrinsicNew`, so every v1 equation
  reduces exactly as before. The three probes `packorder`, `substr`,
  `seqtail` were written from the v1.1 definitions' predictions and
  confirmed them against gfortran/NumPy end-to-end (FNP-D0041 .. D0045).
* **v1.2 (2026-09-18).** Procedure calls: argument association, the prologue
  and copy-back on both sides (`F.Proc`, `P.Proc`, `bindDummies`, `initLocals`,
  `copyBack`, `bindParams`, `readReturns`, `assignTargets`).  The declaration
  environment became an index of `F.Exec` so that a callee runs under its own
  declarations.  The asymmetry between the engine's two tables -- `_passes`
  makes an UNKNOWN-intent dummy a parameter, `returned_value` leaves it out of
  the return -- is refuted in the model (`FNP_R0141_refuted_unknown`) and
  confirmed end to end by `probes/intentchar` (FNP-D0047).
* **v1 (2026-09-16).** Initial version.
