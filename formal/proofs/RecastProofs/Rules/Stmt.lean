import RecastProofs.Rules.Sections

/-!
# Statement-level soundness: definitions and state-update lemmas

`StmtSound` is a forward simulation: from related, conforming states, every
terminating source execution is matched by a target execution ending in
related states with the corresponding control outcome.  Divergence is not
claimed (big-step semantics).
-/

namespace Recast.Rules

open Recast Rel Exact

/-- Control outcomes correspond: `EXIT ↔ break`, `CYCLE ↔ continue`. -/
def outMap : F.Outcome → P.Outcome
  | .normal => .normal
  | .exit => .brk
  | .cycle => .cont

theorem outMap_ne_brk {out : F.Outcome} (h : out ≠ .exit) : outMap out ≠ .brk := by
  cases out <;> simp [outMap] at h ⊢

theorem outMap_ne_normal {out : F.Outcome} (h : out ≠ .normal) : outMap out ≠ .normal := by
  cases out <;> simp [outMap] at h ⊢

/-- Declared lower bounds are literals, so `Conforms` is insensitive to assignments
(the model's declarations; run-time bounds are FNP-R0049's business). -/
def StaticBounds (Γ : F.Env) : Prop :=
  ∀ x d, Γ x = some d → ∀ e ∈ d.lbs, ∃ k v, e = F.Expr.intLit k v

/-- Statement-level soundness (forward simulation). -/
def StmtSound (o : Oracles) (Γ : F.Env) (s : F.Stmt) (s' : P.Stmt) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ σ' out, F.Exec o.F Γ s σ σ' out →
    ∃ τ', P.Exec o.P s' τ τ' (outMap out) ∧ RelState Γ σ' τ' ∧ Conforms o.F Γ σ'

theorem relState_update {Γ : F.Env} {σ : F.State} {τ : P.State} (h : RelState Γ σ τ) (x : String)
    {v : F.Val} {w : P.Val} (hvw : RelV v w) :
    RelState Γ (Function.update σ x (some v)) (Function.update τ x (some w)) := by
  intro y d hy
  by_cases hxy : y = x
  · subst hxy; exact ⟨v, w, by simp, by simp, hvw⟩
  · obtain ⟨v', w', hv', hw', hr⟩ := h y d hy
    exact ⟨v', w', by simp [Function.update, hxy, hv'], by simp [Function.update, hxy, hw'], hr⟩

/-- A scalar store of a value of the declared type preserves conformance (static bounds). -/
theorem conforms_update_scalar {o : F.Oracle} {Γ : F.Env} {σ : F.State} (hC : Conforms o Γ σ)
    (hS : StaticBounds Γ) (x : String) (d : F.Decl) (hx : Γ x = some d) (hd : d.lbs = [])
    (sv : SVal) (hty : sv.ty = d.ty) : Conforms o Γ (Function.update σ x (some (.s sv))) := by
  intro y d' hy
  by_cases hxy : y = x
  · subst hxy; rw [hx] at hy; cases hy
    exact ⟨fun _ => ⟨sv, by simp, hty⟩, fun h => absurd hd h⟩
  · obtain ⟨h1, h2⟩ := hC y d' hy
    refine ⟨fun hl => ?_, fun hl => ?_⟩
    · obtain ⟨sv', hs, ht⟩ := h1 hl
      exact ⟨sv', by simp [Function.update, hxy, hs], ht⟩
    · obtain ⟨a, ha, hty', hlen, hlbs, hwt⟩ := h2 hl
      refine ⟨a, by simp [Function.update, hxy, ha], hty', hlen, fun i hi => ?_, hwt⟩
      have := hlbs i hi
      obtain ⟨k, v, he⟩ := hS y d' hy (d'.lbs[i]) (List.getElem_mem hi)
      rw [he, F_eval_intLit] at this ⊢
      exact this

/-- An array store that keeps the element type and the dims preserves conformance. -/
theorem conforms_update_array {o : F.Oracle} {Γ : F.Env} {σ : F.State} (hC : Conforms o Γ σ)
    (hS : StaticBounds Γ) (x : String) (d : F.Decl) (hx : Γ x = some d) (a a' : FArr)
    (hσ : σ x = some (.arr a)) (hty : a'.elemTy = a.elemTy) (hdims : a'.dims = a.dims) (hwt : FArr.WellTyped a') :
    Conforms o Γ (Function.update σ x (some (.arr a'))) := by
  intro y d' hy
  by_cases hxy : y = x
  · subst hxy; rw [hx] at hy; cases hy
    obtain ⟨h1, h2⟩ := hC y d hx
    refine ⟨fun hl => ?_, fun hl => ?_⟩
    · obtain ⟨sv, hs, _⟩ := h1 hl; rw [hσ] at hs; cases hs
    · obtain ⟨a₀, ha₀, hty₀, hlen₀, hlbs₀, _⟩ := h2 hl
      rw [hσ] at ha₀; cases ha₀
      refine ⟨a', by simp, by rw [hty, hty₀], by rw [hdims]; exact hlen₀, fun i hi => ?_, hwt⟩
      have := hlbs₀ i hi
      obtain ⟨k, v, he⟩ := hS y d hx (d.lbs[i]) (List.getElem_mem hi)
      rw [he, F_eval_intLit] at this ⊢
      simp only [FArr.lbs, hdims]
      exact this
  · obtain ⟨h1, h2⟩ := hC y d' hy
    refine ⟨fun hl => ?_, fun hl => ?_⟩
    · obtain ⟨sv', hs, ht⟩ := h1 hl
      exact ⟨sv', by simp [Function.update, hxy, hs], ht⟩
    · obtain ⟨a₀, ha₀, hty₀, hlen₀, hlbs₀, hwt₀⟩ := h2 hl
      refine ⟨a₀, by simp [Function.update, hxy, ha₀], hty₀, hlen₀, fun i hi => ?_, hwt₀⟩
      have := hlbs₀ i hi
      obtain ⟨k, v, he⟩ := hS y d' hy (d'.lbs[i]) (List.getElem_mem hi)
      rw [he, F_eval_intLit] at this ⊢
      exact this

/-- Scalar assignment unfolds to a conversion and a store. -/
theorem F_assign_scalar (o : F.Oracle) (Γ : F.Env) (σ : F.State) (x : String) (d : F.Decl) (old : SVal)
    (sv : SVal) (hΓ : Γ x = some d) (hσ : σ x = some (.s old)) :
    F.assign o Γ σ x .nil (.s sv) = (F.convertTo d sv).map fun cv => Function.update σ x (some (.s cv)) := by
  simp only [F.assign, hΓ, hσ, Option.bind_some]
  rfl

/-- A declared scalar is bound to a scalar in a conforming state. -/
theorem conforms_scalar {o : F.Oracle} {Γ : F.Env} {σ : F.State} (hC : Conforms o Γ σ) (x : String) (d : F.Decl)
    (hx : Γ x = some d) (hd : d.lbs = []) : ∃ sv : SVal, σ x = some (.s sv) ∧ sv.ty = d.ty :=
  (hC x d hx).1 hd

/-- A declared array is bound to an array in a conforming state. -/
theorem conforms_array {o : F.Oracle} {Γ : F.Env} {σ : F.State} (hC : Conforms o Γ σ) (x : String) (d : F.Decl)
    (hx : Γ x = some d) (hd : d.lbs ≠ []) : ∃ a : FArr, σ x = some (.arr a) ∧ a.elemTy = d.ty ∧
      a.dims.length = d.lbs.length ∧
      (∀ i (h : i < d.lbs.length), F.eval o σ (d.lbs[i]) = some (.s (.int .i4 (a.lbs.getD i 0)))) ∧
      FArr.WellTyped a :=
  (hC x d hx).2 hd

/-- Well-typedness of a rank-1 array from a per-element check. -/
theorem wellTyped_rank1_of (a' a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (hdims : a'.dims = a.dims)
    (hty : a'.elemTy = a.elemTy)
    (h : ∀ j : ℤ, lb ≤ j ∧ j < lb + n → (a'.get [j]).ty = a.elemTy) : FArr.WellTyped a' := by
  intro idx hin
  rw [FArr.inBounds, hdims, hd] at hin
  match idx, hin with
  | [j], hin =>
    have hj : lb ≤ j ∧ j < lb + n := by simpa [FArr.inBoundsB] using hin
    rw [hty]; exact h j hj
  | [], hin => simp [FArr.inBoundsB] at hin
  | _ :: _ :: _, hin => simp [FArr.inBoundsB] at hin

theorem inBounds_rank1 (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (j : ℤ) (hj : lb ≤ j ∧ j < lb + n) :
    a.inBounds [j] := by
  simp [FArr.inBounds, FArr.inBoundsB, hd, hj]

end Recast.Rules
