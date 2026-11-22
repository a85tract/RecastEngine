import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# Exact interpretation of decimal literal text

Both languages carry numeric literals as *decimal text*.  The engine never
converts a hoisted real literal through a host binary float: the constants
module receives `np.float64('<digits>')` with the original digits
(`fortran/constants.py` `hoist`, `transform/numpy/constants.py` `_real`).

The literal policy of `SEMANTICS.md` §2: a decimal token with mantissa digits
`m` (an integer, sign included) and decimal exponent `e` denotes the rational
`m · 10^e`.  `1.d-3` is `(1, -3)`; `273.15` is `(27315, -2)`; `3600.0_r8` is
`(36000, -1)`.  The kind suffix is metadata (`RKind`), not part of the value.

The Python side interprets `np.float64('<text>')` under the *same* exact
policy: the string constructor denotes the rational the text spells.  That
choice is what makes "no floating-point rounding" a consistent idealization
on both sides; it is recorded as an idealization in `TRUSTED_BASE.md`.  The
`np.float32(...)` wrapper the engine adds for default-kind literals is a
precision cast and is the identity in this model (`SEMANTICS.md` §2, casts).
-/

namespace Recast

/-- A decimal literal: mantissa digits as an integer and a decimal exponent. -/
structure Decimal where
  mant : ℤ
  exp10 : ℤ
  deriving DecidableEq, Repr

namespace Decimal

/-- The exact rational a decimal token denotes. -/
noncomputable def toReal (d : Decimal) : ℝ := (d.mant : ℝ) * (10 : ℝ) ^ d.exp10

/-- Two spellings of the same decimal denote the same real (`1.d-3` vs `0.001`). -/
theorem toReal_shift (m : ℤ) (e : ℤ) :
    toReal ⟨m * 10, e - 1⟩ = toReal ⟨m, e⟩ := by
  unfold toReal
  push_cast
  rw [zpow_sub₀ (by norm_num : (10 : ℝ) ≠ 0), zpow_one]
  field_simp

/-- Whitelisted reals the emitter writes inline (`WHITELIST_REAL = {0.0, 1.0, 0.5}`,
`fortran/constants.py:39`).  Each is a dyadic rational, so the `repr(float(text))`
the emitter writes (`names.py:123-125`) spells the same exact value. -/
def whitelistReal : List Decimal := [⟨0, 0⟩, ⟨1, 0⟩, ⟨5, -1⟩]

theorem whitelist_values :
    (whitelistReal.map toReal) = [0, 1, 1/2] := by
  simp [whitelistReal, toReal]
  norm_num

end Decimal

end Recast
