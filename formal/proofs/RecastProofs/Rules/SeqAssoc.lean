import RecastProofs.Rules.Stmt

/-!
# FNP-R0055: sequence association through the column-major storage relation

The rule passes `arr(i, j)` to a dummy `a(m, n)` as
`np.reshape(arr.ravel(order='F')[off:off + m*n], (m, n), order='F')` with
`off` the rebased column-major offset of the element.  The storage lemma:
the element at Fortran coordinates `idx` of a related pair sits at Python
position `CM.offset shape (rebase lbs idx)` of the column-major ravel, and
`CM.coords` inverts that offset.  The copy semantics of the reshape itself
(a fresh array, so callee writes do not reach the caller) is FNP-R0060.
-/

namespace Recast.Rules

open Recast Rel Exact

/-- Rebased coordinates of an in-bounds Fortran index are in bounds for the Python shape. -/
theorem rebase_inBounds :
    ∀ (dims : List (ℤ × ℕ)) (idx : List ℤ), FArr.inBoundsB dims idx = true →
      PArr.inBoundsB (dims.map Prod.snd) (List.zipWith (fun (lb : ℤ) (i : ℤ) => (i - lb).toNat) (dims.map Prod.fst) idx) = true
  | [], [], _ => rfl
  | [], _ :: _, hin => by simp [FArr.inBoundsB] at hin
  | _ :: _, [], hin => by simp [FArr.inBoundsB] at hin
  | (lb, n) :: ds, i :: is, hin => by
      simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hin
      simp only [List.map_cons, List.zipWith_cons_cons, PArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨by omega, rebase_inBounds ds is hin.2⟩

def FNP_R0055_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ idx, a.inBounds idx →
    RelS (a.get idx) (p.get (CM.coords p.shape (CM.offset p.shape (rebase a.lbs idx))))

/-- The column-major position of a related element is the element itself. -/
theorem FNP_R0055_offset_sound : FNP_R0055_obligation := by
  intro a p h idx hin
  have hb : PArr.inBoundsB p.shape (rebase a.lbs idx) = true := by
    rw [h.shape]
    exact rebase_inBounds a.dims idx hin
  rw [CM.coords_offset p.shape (rebase a.lbs idx) hb]
  exact h.elems idx hin

end Recast.Rules
