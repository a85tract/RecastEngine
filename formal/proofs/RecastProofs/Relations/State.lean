import RecastProofs.Semantics.Fortran
import RecastProofs.Semantics.Python

/-!
# The source/target state relation `R_Γ`

Scalars: equal mathematical values (kinds are dropped on the Python side).
Arrays: the Python shape is the Fortran extents, and
`A_F(i₁, …, iᵣ) = A_P[i₁ - l₁, …, iᵣ - lᵣ]` on every valid Fortran coordinate.
States: every declared name is related.
-/

namespace Recast.Rel

open Recast

/-- Scalar value relation: equal *mathematical* values.  A Fortran REAL is
related to any Python number denoting the same real -- a `float`, or an `int`
when the value is integral (`x = 0` leaves a Python `int` in a variable that
Fortran types REAL). -/
inductive RelS : SVal → PVal → Prop where
  | int {k v} : RelS (.int k v) (.int v)
  | real {k v w} : P.toR w = some v → RelS (.real k v) w
  | bool {b} : RelS (.bool b) (.bool b)
  | str {s} : RelS (.str s) (.str s)

theorem RelS.realF {k : RKind} {v : ℝ} : RelS (.real k v) (.float v) := RelS.real rfl
theorem RelS.realI {k : RKind} {n : ℤ} : RelS (.real k (n : ℝ)) (.int n) := RelS.real rfl

/-- The two shapes a Python value related to a Fortran real can take. -/
theorem RelS.real_cases {k : RKind} {v : ℝ} {w : PVal} (h : RelS (.real k v) w) :
    (∃ n : ℤ, w = .int n ∧ v = n) ∨ w = .float v := by
  cases h with
  | real ht =>
    cases w with
    | int n => left; refine ⟨n, rfl, ?_⟩; simp [P.toR] at ht; exact ht.symm
    | float r => right; simp [P.toR] at ht; rw [ht]
    | bool _ => simp [P.toR] at ht
    | str _ => simp [P.toR] at ht
    | none => simp [P.toR] at ht

theorem RelS.real_toR {k : RKind} {v : ℝ} {w : PVal} (h : RelS (.real k v) w) : P.toR w = some v := by
  cases h; assumption

/-- The rebased coordinates of a Fortran index. -/
def rebase (lbs : List ℤ) (idx : List ℤ) : List ℕ :=
  List.zipWith (fun (lb : ℤ) (i : ℤ) => (i - lb).toNat) lbs idx

/-- Array relation: shape is the extents, elements agree after rebasing. -/
structure RelA (a : FArr) (p : PArr) : Prop where
  shape : p.shape = a.extents
  elems : ∀ idx, a.inBounds idx → RelS (a.get idx) (p.get (rebase a.lbs idx))

/-- Value relation. -/
inductive RelV : F.Val → P.Val → Prop where
  | s {v w} : RelS v w → RelV (.s v) (.s w)
  | arr {a p} : RelA a p → RelV (.arr a) (.arr p)

/-- State relation over a declaration environment: every declared name is
bound on both sides to related values. -/
def RelState (Γ : F.Env) (σ : F.State) (τ : P.State) : Prop :=
  ∀ x d, Γ x = some d → ∃ v w, σ x = some v ∧ τ x = some w ∧ RelV v w

/-- Elements carry the array's element type (a homogeneous array). -/
def _root_.Recast.FArr.WellTyped (a : FArr) : Prop := ∀ idx, a.inBounds idx → (a.get idx).ty = a.elemTy

/-- A Fortran state conforms to its declarations: every declared scalar is
bound to a scalar of the declared type; every declared array is bound to an
array of the declared element type and rank whose lower bounds are the values
of the declared bound expressions.  This is the *source well-formedness*
premise of every rule. -/
def Conforms (o : F.Oracle) (Γ : F.Env) (σ : F.State) : Prop :=
  ∀ x d, Γ x = some d →
    (d.lbs = [] → ∃ sv, σ x = some (.s sv) ∧ sv.ty = d.ty) ∧
    (d.lbs ≠ [] → ∃ a : FArr, σ x = some (.arr a) ∧ a.elemTy = d.ty ∧
        a.dims.length = d.lbs.length ∧
        (∀ i (h : i < d.lbs.length), F.eval o σ (d.lbs[i]) = some (.s (.int .i4 (a.lbs.getD i 0)))) ∧
        FArr.WellTyped a)

theorem RelS.toR {v : SVal} {w : PVal} (h : RelS v w) :
    ∀ r, F.toRealS v = some r → P.toR w = some r := by
  intro r hr
  cases h with
  | int => simp [F.toRealS] at hr; subst hr; rfl
  | real ht => simp [F.toRealS] at hr; subst hr; exact ht
  | bool => simp [F.toRealS] at hr
  | str => simp [F.toRealS] at hr

end Recast.Rel
