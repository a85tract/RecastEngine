import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Data.Int.Basic
import Mathlib.Tactic

/-!
# Exact-arithmetic value model shared by the Fortran and Python semantics

This file fixes the *mathematical carriers* of both languages.  It is the
"common typed mathematical IR" of `proofs/SEMANTICS.md`.  Nothing here says how
a Fortran or a Python expression is *evaluated*; each language has its own
evaluation function in `RecastProofs.Semantics.Fortran` and
`RecastProofs.Semantics.Python`.  The two languages are related only through
the state relation in `RecastProofs.Relations.State`.

Design decisions recorded in `SEMANTICS.md` (semantics version 1):

* Reals are Mathlib's `ℝ`.  There is no rounding anywhere in the model.
* Integers are `ℤ`, unbounded.  Fortran kinds are retained as metadata on the
  *Fortran* side only (`IKind`, `RKind`), because kind inquiries (`KIND`,
  `HUGE`, `PRECISION`) and bit-width-dependent intrinsics need them.  The
  Python side has no kinds: a Python `int` or `float` carries none, which is
  exactly the information the emitted `_f_kind`/`_f_huge` helpers do not have.
* Integer division is `Int.tdiv` (truncation toward zero).  It is *not* Lean's
  `/` on `ℤ` (`Int.ediv`) and not Python's `//` (`Int.fdiv`, floor).
* Real-to-integer conversion truncates toward zero (`Recast.Exact.truncR`).
* Integer-to-real conversion is the cast `((n : ℤ) : ℝ)`.
* Decimal literal text is interpreted exactly as a rational
  (`RecastProofs.Semantics.Literal`), never through a host binary float.
-/

namespace Recast

/-- Fortran integer kinds retained as metadata. -/
inductive IKind where
  | i4
  | i8
  deriving DecidableEq, Repr

/-- Fortran real kinds retained as metadata. -/
inductive RKind where
  | r4
  | r8
  deriving DecidableEq, Repr

/-- Bit width of an integer kind. -/
def IKind.bits : IKind → ℕ
  | .i4 => 32
  | .i8 => 64

/-- `HUGE` of an integer kind: `2^(bits-1) - 1`. -/
def IKind.huge (k : IKind) : ℤ := 2 ^ (k.bits - 1) - 1

/-- `KIND` inquiry value of an integer kind (the byte count). -/
def IKind.kindValue : IKind → ℤ
  | .i4 => 4
  | .i8 => 8

/-- `KIND` inquiry value of a real kind. -/
def RKind.kindValue : RKind → ℤ
  | .r4 => 4
  | .r8 => 8

/-- `PRECISION` inquiry value of a real kind (IEEE binary32 / binary64). -/
def RKind.precision : RKind → ℤ
  | .r4 => 6
  | .r8 => 15

/-- `DIGITS` inquiry value of a real kind (significand bits). -/
def RKind.digits : RKind → ℤ
  | .r4 => 24
  | .r8 => 53

/-- `DIGITS` inquiry value of an integer kind (`bits - 1`). -/
def IKind.digits (k : IKind) : ℤ := (k.bits : ℤ) - 1

/-- Fortran scalar types (with kinds). -/
inductive FTy where
  | int (k : IKind)
  | real (k : RKind)
  | bool
  | str
  deriving DecidableEq, Repr

/-- Fortran scalar values in the exact model. -/
inductive SVal where
  | int (k : IKind) (v : ℤ)
  | real (k : RKind) (v : ℝ)
  | bool (b : Bool)
  | str (s : List Char)

/-- Python scalar values in the exact model.  No kinds. -/
inductive PVal where
  | int (v : ℤ)
  | float (v : ℝ)
  | bool (b : Bool)
  | str (s : List Char)
  | none

namespace Exact

/-- Truncating integer quotient: the Fortran `/` on integers (`Int.tdiv`). -/
def tquot (a b : ℤ) : ℤ := Int.tdiv a b

/-- Flooring integer quotient: Python's `//` (`Int.fdiv`). -/
def fquot (a b : ℤ) : ℤ := Int.fdiv a b

/-- Truncation of a real toward zero: Fortran `INT`, Python `int`. -/
noncomputable def truncR (x : ℝ) : ℤ := if 0 ≤ x then ⌊x⌋ else ⌈x⌉

/-- Fortran `NINT`: round half away from zero. -/
noncomputable def nintR (x : ℝ) : ℤ := if 0 ≤ x then ⌊x + 1/2⌋ else ⌈x - 1/2⌉

/-- Round half to even, as `numpy.round` / Python `round` do. Needed to state
the ANINT discrepancy (defect FNP-D0004). -/
noncomputable def roundHalfEvenR (x : ℝ) : ℤ :=
  let f := ⌊x⌋
  let frac := x - (f : ℝ)
  if frac < 1/2 then f
  else if frac > 1/2 then f + 1
  else if f % 2 = 0 then f else f + 1

/-- Fortran `MOD(a,p) = a - INT(a/p)*p` on reals. -/
noncomputable def modR (a p : ℝ) : ℝ := a - (truncR (a / p) : ℝ) * p

/-- Fortran `MODULO(a,p) = a - FLOOR(a/p)*p` on reals; also Python `%` on floats. -/
noncomputable def moduloR (a p : ℝ) : ℝ := a - (⌊a / p⌋ : ℝ) * p

/-- Fortran `MOD` on integers: remainder with the sign of `a` (`Int.tmod`). -/
def tmod (a p : ℤ) : ℤ := Int.tmod a p

/-- Fortran `MODULO` on integers / Python `%`: remainder with the sign of `p`. -/
def fmod (a p : ℤ) : ℤ := Int.fmod a p

/-- Two's-complement wrap of an integer into the signed `bits`-wide range. -/
def wrapSigned (bits : ℕ) (v : ℤ) : ℤ :=
  let m : ℤ := 2 ^ bits
  let r := v % m
  if r ≥ m / 2 then r - m else r

/-- The unsigned `bits`-wide two's-complement image of an integer. -/
def unsignedK (bits : ℕ) (v : ℤ) : ℕ := (v % (2 : ℤ) ^ bits).toNat

/-- Fortran `IAND`/`IOR`/`IEOR` within a kind of `bits` bits. -/
def bitAndK (bits : ℕ) (x y : ℤ) : ℤ := wrapSigned bits ((unsignedK bits x &&& unsignedK bits y : ℕ) : ℤ)
def bitOrK (bits : ℕ) (x y : ℤ) : ℤ := wrapSigned bits ((unsignedK bits x ||| unsignedK bits y : ℕ) : ℤ)
def bitXorK (bits : ℕ) (x y : ℤ) : ℤ := wrapSigned bits ((unsignedK bits x ^^^ unsignedK bits y : ℕ) : ℤ)

/-- Fortran `ISHFT` within a kind's width: a logical shift on the unsigned view. -/
def ishftK (bits : ℕ) (i s : ℤ) : ℤ :=
  let u : ℤ := unsignedK bits i
  wrapSigned bits (if 0 ≤ s then u * 2 ^ s.toNat else u / 2 ^ (-s).toNat)

/-- Unbounded shift, as Python performs it on a plain `int` (the emitted
`_f_ishft` when no operand carries a NumPy dtype). -/
def ishftUnbounded (i s : ℤ) : ℤ := if 0 ≤ s then i * 2 ^ s.toNat else i / 2 ^ (-s).toNat

/-- Witness for defect FNP-D0009: `ISHFT(1, 31)` in a 32-bit kind is the sign bit. -/
theorem ishftK_example : ishftK 32 1 31 = -2147483648 := by decide
theorem ishftUnbounded_example : ishftUnbounded 1 31 = 2147483648 := by decide

theorem tquot_eq_tdiv (a b : ℤ) : tquot a b = Int.tdiv a b := rfl

/-- Truncating and flooring division agree for non-negative operands. -/
theorem tquot_eq_fquot_of_nonneg {a b : ℤ} (ha : 0 ≤ a) (hb : 0 < b) :
    tquot a b = fquot a b := by
  unfold tquot fquot
  rw [Int.tdiv_eq_ediv_of_nonneg ha, Int.fdiv_eq_ediv_of_nonneg a hb.le]

/-- The witness that the two integer divisions differ: `-3 / 2`. -/
theorem tquot_ne_fquot_example : tquot (-3) 2 ≠ fquot (-3) 2 := by decide

theorem truncR_of_nonneg {x : ℝ} (h : 0 ≤ x) : truncR x = ⌊x⌋ := by
  unfold truncR; rw [if_pos h]

theorem truncR_of_neg {x : ℝ} (h : x < 0) : truncR x = ⌈x⌉ := by
  unfold truncR; rw [if_neg (not_le.mpr h)]

/-- Truncation is odd. -/
theorem truncR_neg (x : ℝ) : truncR (-x) = -truncR x := by
  unfold truncR
  rcases lt_trichotomy x 0 with h | h | h
  · have h1 : 0 ≤ -x := by linarith
    rw [if_pos h1, if_neg (not_le.mpr h), Int.floor_neg]
  · subst h; simp
  · have h1 : ¬ (0 ≤ -x) := by intro h'; linarith
    rw [if_neg h1, if_pos h.le, Int.ceil_neg]

/-- `truncR` of an integer cast is the integer itself. -/
theorem truncR_intCast (n : ℤ) : truncR (n : ℝ) = n := by
  unfold truncR
  split_ifs <;> simp

/-- Non-negative numerator, positive denominator: truncation of the real
quotient is the truncating integer quotient. -/
theorem truncR_div_intCast_of_nonneg {a b : ℤ} (ha : 0 ≤ a) (hb : 0 < b) :
    truncR ((a : ℝ) / (b : ℝ)) = tquot a b := by
  have hq : (0 : ℝ) ≤ (a : ℝ) / (b : ℝ) := by
    apply div_nonneg <;> exact_mod_cast (by omega)
  rw [truncR_of_nonneg hq]
  unfold tquot
  rw [Int.tdiv_eq_ediv_of_nonneg ha]
  obtain ⟨d, hd⟩ : ∃ d : ℕ, b = (d : ℤ) := ⟨b.toNat, (Int.toNat_of_nonneg hb.le).symm⟩
  subst hd
  have : ((d : ℤ) : ℝ) = (d : ℝ) := by norm_cast
  rw [this, Int.floor_div_natCast, Int.floor_intCast]

/-- Truncation of a real quotient of integers is the truncating integer
quotient, for every non-zero divisor.  This is the lemma that connects the
emitted `int(a / b)` (exact real division, then truncation) to Fortran's
integer `/`. -/
theorem truncR_div_intCast (a b : ℤ) (hb : b ≠ 0) :
    truncR ((a : ℝ) / (b : ℝ)) = tquot a b := by
  rcases le_or_gt 0 a with ha | ha <;> rcases lt_or_gt_of_ne hb with hbneg | hbpos
  · -- a ≥ 0, b < 0
    have h1 : ((a : ℝ) / (b : ℝ)) = -((a : ℝ) / ((-b : ℤ) : ℝ)) := by
      push_cast; rw [div_neg, neg_neg]
    rw [h1, truncR_neg, truncR_div_intCast_of_nonneg ha (by omega)]
    unfold tquot; rw [Int.tdiv_neg, neg_neg]
  · exact truncR_div_intCast_of_nonneg ha hbpos
  · -- a < 0, b < 0
    have h1 : ((a : ℝ) / (b : ℝ)) = (((-a : ℤ) : ℝ) / ((-b : ℤ) : ℝ)) := by
      push_cast; rw [neg_div_neg_eq]
    rw [h1, truncR_div_intCast_of_nonneg (by omega) (by omega)]
    unfold tquot; rw [Int.neg_tdiv, Int.tdiv_neg, neg_neg]
  · -- a < 0, b > 0
    have h1 : ((a : ℝ) / (b : ℝ)) = -(((-a : ℤ) : ℝ) / (b : ℝ)) := by
      push_cast; rw [neg_div, neg_neg]
    rw [h1, truncR_neg, truncR_div_intCast_of_nonneg (by omega) hbpos]
    unfold tquot; rw [Int.neg_tdiv, neg_neg]

/-- Floor of a real quotient of integers is Python's `//` (`Int.fdiv`). -/
theorem floor_div_intCast (a b : ℤ) (hb : b ≠ 0) :
    ⌊(a : ℝ) / (b : ℝ)⌋ = fquot a b := by
  unfold fquot
  rcases lt_or_gt_of_ne hb with hbneg | hbpos
  · obtain ⟨d, hd⟩ : ∃ d : ℕ, -b = (d : ℤ) := ⟨(-b).toNat, (Int.toNat_of_nonneg (by omega)).symm⟩
    have h1 : ((a : ℝ) / (b : ℝ)) = (((-a : ℤ) : ℝ) / ((d : ℤ) : ℝ)) := by
      rw [← hd]; push_cast; rw [neg_div_neg_eq]
    have hc : (((d : ℤ)) : ℝ) = (d : ℝ) := by norm_cast
    rw [h1, hc, Int.floor_div_natCast, Int.floor_intCast, ← Int.neg_fdiv_neg a b, hd,
      Int.fdiv_eq_ediv_of_nonneg (-a) (by exact_mod_cast Nat.zero_le d)]
  · obtain ⟨d, hd⟩ : ∃ d : ℕ, b = (d : ℤ) := ⟨b.toNat, (Int.toNat_of_nonneg hbpos.le).symm⟩
    subst hd
    rw [Int.fdiv_eq_ediv_of_nonneg a (by exact_mod_cast Nat.zero_le d)]
    have hc : (((d : ℤ)) : ℝ) = (d : ℝ) := by norm_cast
    rw [hc, Int.floor_div_natCast, Int.floor_intCast]

/-- ANINT and numpy.round differ at a half: `2.5`. -/
theorem nintR_ne_roundHalfEven_example : nintR (5/2 : ℝ) ≠ roundHalfEvenR (5/2 : ℝ) := by
  have h1 : nintR (5/2 : ℝ) = 3 := by
    unfold nintR
    rw [if_pos (by norm_num)]
    rw [show (5/2 : ℝ) + 1/2 = ((3 : ℤ) : ℝ) by norm_num, Int.floor_intCast]
  have h2 : roundHalfEvenR (5/2 : ℝ) = 2 := by
    unfold roundHalfEvenR
    have hf : ⌊(5/2 : ℝ)⌋ = 2 := by
      rw [Int.floor_eq_iff]; norm_num
    simp only [hf]
    norm_num
  rw [h1, h2]; decide

end Exact

end Recast
