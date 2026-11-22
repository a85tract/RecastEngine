import RecastProofs.Rules.Control

/-!
# Rules FNP-R0111 (scalar assignment), FNP-R0112 (whole-array assignment),
FNP-R0113 (element assignment)
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Conversion to the declared type -/

/-- The base type (kinds erased). -/
inductive Base where
  | int | real | bool | str
  deriving DecidableEq, Repr

def _root_.Recast.FTy.base : FTy → Base
  | .int _ => .int
  | .real _ => .real
  | .bool => .bool
  | .str => .str

/-- Loss-free compatibility of a value with a declared scalar type: the same
base type (not CHARACTER), or an INTEGER value into a REAL target. -/
def Compat (ty : FTy) (v : SVal) : Prop :=
  (v.ty.base = ty.base ∧ ty ≠ .str) ∨ ((∃ k, ty = .real k) ∧ (∃ k' n, v = .int k' n))

theorem convertTo_compat_rel (d : F.Decl) (v : SVal) (w : PVal) (hrel : RelS v w) (hc : Compat d.ty v) :
    ∃ cv, F.convertTo d v = some cv ∧ RelS cv w ∧ cv.ty = d.ty := by
  rcases hc with ⟨hty, hstr⟩ | ⟨⟨k, hk⟩, ⟨k', n, rfl⟩⟩
  · cases v with
    | int k n =>
      rw [RelS.int_eq hrel]
      cases hd : d.ty with
      | int k₂ => exact ⟨.int k₂ n, by simp [F.convertTo, hd], RelS.int, by simp [SVal.ty]⟩
      | real _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | bool => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | str => exact absurd hd hstr
    | real k r =>
      cases hd : d.ty with
      | real k₂ => exact ⟨.real k₂ r, by simp [F.convertTo, hd], RelS.real (RelS.real_toR hrel), by simp [SVal.ty]⟩
      | int _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | bool => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | str => exact absurd hd hstr
    | bool b =>
      rw [RelS.bool_eq hrel]
      cases hd : d.ty with
      | bool => exact ⟨.bool b, by simp [F.convertTo, hd], RelS.bool, by simp [SVal.ty]⟩
      | int _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | real _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | str => exact absurd hd hstr
    | str s =>
      cases hd : d.ty with
      | str => exact absurd hd hstr
      | int _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | real _ => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
      | bool => rw [hd] at hty; simp [SVal.ty, FTy.base] at hty
  · rw [RelS.int_eq hrel]
    refine ⟨.real k (n : ℝ), ?_, RelS.realI, by simp [SVal.ty, hk]⟩
    simp [F.convertTo, hk]

/-! ### FNP-R0111: scalar assignment -/

def ScalarCompat (o : Oracles) (Γ : F.Env) (d : F.Decl) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) → Compat d.ty v

def FNP_R0111_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (x : String) (e : F.Expr) (e' : P.Expr) (d : F.Decl),
    Γ x = some d → d.lbs = [] → StaticBounds Γ → SoundScalar o Γ e e' →
    StmtSound o Γ (.assign x .nil e) (.assign x e')

theorem FNP_R0111_sound_numeric (o : Oracles) (Γ : F.Env) (x : String) (e : F.Expr) (e' : P.Expr) (d : F.Decl)
    (hx : Γ x = some d) (hd : d.lbs = []) (hS : StaticBounds Γ) (he : SoundScalar o Γ e e')
    (hcompat : ScalarCompat o Γ d e) :
    StmtSound o Γ (.assign x .nil e) (.assign x e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨old, hσ, _⟩ := conforms_scalar hC x d hx hd
    cases v with
    | arr _ => simp [F.assign, hx, hσ] at hassign
    | s sv =>
      rw [F_assign_scalar o.F Γ σ x d old sv hx hσ] at hassign
      obtain ⟨w, hw, hrw⟩ := he σ τ hR hC sv hv
      obtain ⟨cv, hcv, hrel, hty⟩ := convertTo_compat_rel d sv w hrw (hcompat σ τ hR hC sv hv)
      rw [hcv] at hassign
      simp only [Option.map_some, Option.some.injEq] at hassign
      subst hassign
      exact ⟨_, P.Exec.assign hw, relState_update hR x (RelV.s hrel), conforms_update_scalar hC hS x d hx hd cv hty⟩

/-- INTEGER target from a REAL expression: `x = _f_int(e')` on the representable range. -/
theorem FNP_R0111_sound_int_of_real (o : Oracles) (Γ : F.Env) (x : String) (e : F.Expr) (e' : P.Expr)
    (k : IKind) (cl : ℕ) (hx : Γ x = some ⟨.int k, [], cl⟩) (hS : StaticBounds Γ) (he : SoundScalar o Γ e e')
    (hreal : RealTyped o Γ e)
    (hrange : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k r, F.eval o.F σ e = some (.s (.real k r)) → |r| < 2147483648) :
    StmtSound o Γ (.assign x .nil e) (.assign x (Tr.fold1 .f_int e')) := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨old, hσ, _⟩ := conforms_scalar hC x _ hx rfl
    cases v with
    | arr _ => simp [F.assign, hx, hσ] at hassign
    | s sv =>
      obtain ⟨kr, r, rfl⟩ := hreal σ τ hR hC sv hv
      rw [F_assign_scalar o.F Γ σ x _ old _ hx hσ] at hassign
      simp only [F.convertTo, Option.map_some, Option.some.injEq] at hassign
      subst hassign
      obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hv
      have ht := RelS.real_toR hrw
      have hr := hrange σ τ hR hC kr r hv
      have hpy : P.eval o.P τ (Tr.fold1 .f_int e') = some (.s (.int (truncR r))) := by
        rw [P_eval_call1, hw, Option.bind_some]
        rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
        · rw [truncR_intCast]; rfl
        · rw [helper_f_int_float]
          have hb : -(2147483648 : ℝ) ≤ r ∧ r < 2147483648 := by
            constructor
            · linarith [neg_abs_le r, hr]
            · linarith [le_abs_self r, hr]
          rw [if_pos hb]
      exact ⟨_, P.Exec.assign hpy, relState_update hR x (RelV.s RelS.int),
        conforms_update_scalar hC hS x _ hx rfl _ rfl⟩

/-! FNP-D0021: a CHARACTER target is blank-padded in Fortran and not in Python. -/

def declS : F.Decl := ⟨.str, [], 10⟩
def envS : F.Env := fun x => if x = "s" then some declS else none
def stateS : F.State := fun x => if x = "s" then some (.s (.str [])) else none
def pstateS : P.State := fun x => if x = "s" then some (.s (.str [])) else none

theorem relStateS : RelState envS stateS pstateS := by
  intro x d hx
  simp only [envS] at hx
  split_ifs at hx with hxa
  subst hxa
  exact ⟨.s (.str []), .s (.str []), by simp [stateS], by simp [pstateS], RelV.s RelS.str⟩

theorem conformsS : Conforms noPrim.F envS stateS := by
  intro x d hx
  simp only [envS] at hx
  split_ifs at hx with hxa
  subst hxa
  cases hx
  exact ⟨fun _ => ⟨.str [], by simp [stateS], rfl⟩, fun h => absurd rfl h⟩

theorem staticBoundsS : StaticBounds envS := by
  intro x d hx
  simp only [envS] at hx
  split_ifs at hx with hxa
  subst hxa; cases hx
  intro e he; simp [declS] at he

theorem strLit_sound (o : Oracles) (Γ : F.Env) (s : List Char) : SoundScalar o Γ (.strLit s) (.strLit s) := by
  intro σ τ _ _ v hv
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv
  subst hv
  exact ⟨_, rfl, RelS.str⟩

/-- FNP-D0021: `s = 'ab'` into `character(10) :: s`. -/
theorem FNP_R0111_refuted : ¬ FNP_R0111_obligation := by
  intro h
  have hs := h noPrim envS "s" (.strLit ['a', 'b']) (.strLit ['a', 'b']) declS (by simp [envS]) rfl staticBoundsS
    (strLit_sound _ _ _)
  have hF : F.Exec noPrim.F envS (.assign "s" .nil (.strLit ['a', 'b'])) stateS
      (Function.update stateS "s" (some (.s (.str (F.padTo 10 ['a', 'b']))))) .normal := by
    refine F.Exec.assign (v := .s (.str ['a', 'b'])) (by simp [F.eval]) ?_
    rw [F_assign_scalar noPrim.F envS stateS "s" declS (.str []) _ (by simp [envS]) (by simp [stateS])]
    rfl
  obtain ⟨τ', hex, hR', _⟩ := hs stateS pstateS relStateS conformsS _ _ hF
  cases hex with
  | assign hv =>
    rename_i v
    simp only [P.eval, Option.some.injEq] at hv
    subst hv
    obtain ⟨v, w, hv, hw, hrel⟩ := hR' "s" declS (by simp [envS])
    simp only [Function.update_self, Option.some.injEq] at hv hw
    subst hv; subst hw
    cases hrel with
    | s hs =>
      have h1 := PVal.str.inj (RelS.str_eq hs)
      have := congrArg List.length h1
      simp [F.padTo] at this

/-- Array assignment with subscripts unfolds to the subscript evaluation and a store. -/
theorem F_assign_array_subs (o : F.Oracle) (Γ : F.Env) (σ : F.State) (x : String) (d : F.Decl) (a : FArr)
    (s : F.Sub) (rest : F.Subs) (rhs : F.Val) (hΓ : Γ x = some d) (hσ : σ x = some (.arr a)) :
    F.assign o Γ σ x (.cons s rest) rhs = (F.evalSubs o σ (.cons s rest)).bind fun axes =>
      (F.storeArr d a axes rhs).map fun a' => Function.update σ x (some (.arr a')) := by
  unfold F.assign
  rw [hΓ, hσ]
  rfl

/-- Whole-array assignment unfolds to the full-range store. -/
theorem F_assign_array_nil (o : F.Oracle) (Γ : F.Env) (σ : F.State) (x : String) (d : F.Decl) (a : FArr)
    (rhs : F.Val) (hΓ : Γ x = some d) (hσ : σ x = some (.arr a)) :
    F.assign o Γ σ x .nil rhs = (F.storeArr d a (a.dims.map fun _ => F.Axis.range none none none) rhs).map
      fun a' => Function.update σ x (some (.arr a')) := by
  unfold F.assign
  rw [hΓ, hσ]
  rfl

/-! ### FNP-R0113: element assignment on a unit-origin rank-1 array (scalar RHS) -/

theorem F_storeArr_fixed (d : F.Decl) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (iv : ℤ)
    (hin : lb ≤ iv ∧ iv < lb + n) (v cv : SVal) (hcv : F.convertTo d v = some cv) :
    ∃ a' : FArr, F.storeArr d a [.fixed iv] (.s v) = some a' ∧ a'.elemTy = a.elemTy ∧ a'.dims = a.dims ∧
      a'.get [iv] = cv ∧ ∀ i : ℤ, i ≠ iv → a'.get [i] = a.get [i] := by
  cases hst : F.storeArr d a [.fixed iv] (.s v) with
  | none => exfalso; simp [F.storeArr, F.resolveAxis, hd, hin, hcv] at hst
  | some a' =>
    simp [F.storeArr, F.resolveAxis, hd, hin, hcv] at hst
    subst hst
    refine ⟨_, rfl, rfl, hd.symm, by simp, fun i hi => ?_⟩
    simp [hi]

theorem P_storeArr_at (p : PArr) (n : ℕ) (hs : p.shape = [n]) (j : ℤ) (hj : 0 ≤ j ∧ j < n) (w : PVal) :
    ∃ p' : PArr, P.storeArr p [.at j] (.s w) = some p' ∧ p'.shape = p.shape ∧
      p'.get [j.toNat] = w ∧ ∀ k : ℕ, k ≠ j.toNat → p'.get [k] = p.get [k] := by
  have h1 : ¬ (j < 0) := by omega
  cases hst : P.storeArr p [.at j] (.s w) with
  | none => exfalso; simp [P.storeArr, P.selection, Py.wrapIndex, hs, h1, hj.1, hj.2] at hst
  | some p' =>
    simp [P.storeArr, P.selection, Py.wrapIndex, hs, h1, hj.1, hj.2] at hst
    subst hst
    refine ⟨_, rfl, hs.symm, by simp, fun k hk => ?_⟩
    simp [hk]

/-- Element assignment through the rank-1 index rules: any shifted index pair. -/
theorem element_assign_sound (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (hx : Γ a = some d)
    (lbE : F.Expr) (hlbs : d.lbs = [lbE]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a lbE)
    (i : F.Expr) (sh : P.Expr) (e : F.Expr) (e' : P.Expr)
    (hsh : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ iv l k, F.eval o.F σ i = some (.s (.int k iv)) →
      F.eval o.F σ lbE = some (.s (.int .i4 l)) → P.eval o.P τ sh = some (.s (.int (iv - l))))
    (hint : IntTyped o Γ i) (he : SoundScalar o Γ e e') (hcompat : ScalarCompat o Γ d e) :
    StmtSound o Γ (.assign a (.cons (.scalar i) .nil) e) (.assignIndex a (.cons (.scalar sh) .nil) e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    rw [F_assign_array_subs o.F Γ σ a d av _ _ _ hx hσ, F_evalSubs_scalar1] at hassign
    simp only [Option.bind_eq_some_iff] at hassign
    obtain ⟨axes, ⟨vi, hvi, hm⟩, hrest⟩ := hassign
    cases vi with
    | arr _ => simp at hm
    | s sv =>
      obtain ⟨k, iv, rfl⟩ := hint σ τ hR hC sv hvi
      simp only [Option.some.injEq] at hm
      subst hm
      simp only [Option.map_eq_some_iff] at hrest
      obtain ⟨a', hst, hσ'⟩ := hrest
      subst hσ'
      cases v with
      | arr _ =>
        exfalso
        by_cases hin : l ≤ iv ∧ iv < l + n <;> simp [F.storeArr, F.resolveAxis, hd, hin] at hst
      | s v =>
        obtain ⟨w, hw, hrw⟩ := he σ τ hR hC v hv
        obtain ⟨cv, hcv, hrelc, hty⟩ := convertTo_compat_rel d v w hrw (hcompat σ τ hR hC v hv)
        by_cases hin : l ≤ iv ∧ iv < l + n
        · obtain ⟨a'', hst', hty', hdims', hget', hother'⟩ := F_storeArr_fixed d av l n hd iv hin v cv hcv
          rw [hst'] at hst
          cases hst
          have hshape := RelA_dims1 hrel l n hd
          obtain ⟨p', hpst, hpshape, hpget, hpother⟩ := P_storeArr_at p n hshape (iv - l) (by omega) w
          refine ⟨Function.update τ a (some (.arr p')), ?_, ?_, ?_⟩
          · refine P.Exec.assignIndex hw hτ ?_ hpst
            rw [P_evalIdxs_scalar1, hsh σ τ hR hC iv l k hvi hl]
            rfl
          · apply relState_update hR a
            refine RelV.arr ⟨by rw [hpshape, hrel.shape]; simp [FArr.extents, hdims'], ?_⟩
            intro idx hidx
            obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ l ≤ j ∧ j < l + n := by
              simp only [FArr.inBounds, FArr.inBoundsB, hdims', hd] at hidx
              match idx, hidx with
              | [j], hidx =>
                simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
                exact ⟨j, rfl, hidx.1⟩
              | [], hidx => simp [FArr.inBoundsB] at hidx
              | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
            simp only [rebase, FArr.lbs, hdims', hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
              List.zipWith_nil_right]
            by_cases hji : j = iv
            · subst hji
              rw [hget', hpget]
              exact hrelc
            · rw [hother' j hji, hpother _ (by omega)]
              exact RelA_get1 hrel l n hd j hj
          · obtain ⟨av₀, hσ₀, hty_av, _, _, hwt_av⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
            rw [hσ] at hσ₀; cases hσ₀
            refine conforms_update_array hC hS a d hx av _ hσ hty' hdims' ?_
            refine wellTyped_rank1_of _ av l n hd hdims' hty' (fun j hj => ?_)
            by_cases hji : j = iv
            · subst hji; rw [hget', hty, hty_av]
            · rw [hother' j hji]; exact hwt_av [j] (inBounds_rank1 av l n hd j hj)
        · exfalso
          simp [F.storeArr, F.resolveAxis, hd, hin] at hst

/-- FNP-R0113 on a unit-origin axis: `A(i) = e` ↦ `A[i' - 1] = e'`. -/
theorem FNP_R0113_sound (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (k : IKind) (hx : Γ a = some d)
    (hlbs : d.lbs = [.intLit k 1]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a (.intLit k 1))
    (i : F.Expr) (i' : P.Expr) (e : F.Expr) (e' : P.Expr) (hi : SoundScalar o Γ i i') (hint : IntTyped o Γ i)
    (he : SoundScalar o Γ e e') (hcompat : ScalarCompat o Γ d e) :
    StmtSound o Γ (.assign a (.cons (.scalar i) .nil) e)
      (.assignIndex a (.cons (.scalar (.bin .sub i' (.intLit 1))) .nil) e') := by
  refine element_assign_sound o Γ a d hx (.intLit k 1) hlbs hS hdecl i _ e e' ?_ hint he hcompat
  intro σ τ hR hC iv l k' hiv hl
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  obtain ⟨w, hw, hrw⟩ := hi σ τ hR hC _ hiv
  rw [RelS.int_eq hrw] at hw
  rw [P_eval_bin, hw]
  simp [P.eval, P.lift2, P.arith]

/-! ### FNP-R0112: whole-array assignment with a scalar (broadcast) RHS -/

theorem sectionCoords_all_inBounds (lb : ℤ) (n : ℕ) :
    ((Fort.sectionCoords lb (lb + n - 1) 1).all fun i => decide (lb ≤ i ∧ i < lb + n)) = true := by
  rw [List.all_eq_true]
  intro i hi
  rw [sectionCoords_mem] at hi
  simp only [decide_eq_true_eq]
  omega

theorem F_storeArr_all_scalar (d : F.Decl) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)])
    (v cv : SVal) (hcv : F.convertTo d v = some cv) :
    ∃ a' : FArr, F.storeArr d a [.range none none none] (.s v) = some a' ∧ a'.elemTy = a.elemTy ∧
      a'.dims = a.dims ∧ ∀ i : ℤ, lb ≤ i ∧ i < lb + n → a'.get [i] = cv := by
  have hall := sectionCoords_all_inBounds lb n
  have hall' : (∀ x ∈ Fort.sectionCoords lb (lb + n - 1) 1, lb ≤ x ∧ x < lb + (n : ℤ)) = True := by
    rw [List.all_eq_true] at hall
    exact eq_true (fun x hx => by simpa using hall x hx)
  cases hst : F.storeArr d a [.range none none none] (.s v) with
  | none => exfalso; simp [F.storeArr, F.resolveAxis, hd, hall', hcv] at hst
  | some a' =>
    simp [F.storeArr, F.resolveAxis, hd, hall', hcv] at hst
    subst hst
    refine ⟨_, rfl, rfl, hd.symm, fun i hi => ?_⟩
    have hmem : i ∈ Fort.sectionCoords lb (lb + n - 1) 1 := by
      rw [sectionCoords_mem]; omega
    simp [hmem]

theorem P_storeArr_all_scalar (p : PArr) (n : ℕ) (hs : p.shape = [n]) (w : PVal) :
    ∃ p' : PArr, P.storeArr p [.sl none none 1] (.s w) = some p' ∧ p'.shape = p.shape ∧
      ∀ k : ℕ, k < n → p'.get [k] = w := by
  have hL : Py.sliceIndices n none none 1 = List.range n := by
    have := sliceIndices_pos n none none 1 (by norm_num) (fun _ h => by cases h) (fun _ h => by cases h)
    rw [this]
    simp only [Option.getD_none]
    have hlen : Py.sliceLen 0 n 1 = n := by
      simp only [Py.sliceLen]
      split_ifs <;> omega
    rw [hlen]
    conv_rhs => rw [← List.map_id (List.range n)]
    apply List.map_congr_left
    intro k _; simp
  cases hst : P.storeArr p [.sl none none 1] (.s w) with
  | none => exfalso; simp [P.storeArr, P.selection, hs, hL] at hst
  | some p' =>
    simp [P.storeArr, P.selection, hs, hL] at hst
    subst hst
    refine ⟨_, rfl, hs.symm, fun k hk => ?_⟩
    simp [hk]

/-- Whole-array assignment of a scalar (broadcast) on a rank-1 array. -/
theorem FNP_R0112_sound_scalar (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (hx : Γ a = some d)
    (lbE : F.Expr) (hlbs : d.lbs = [lbE]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a lbE)
    (e : F.Expr) (e' : P.Expr) (he : SoundScalar o Γ e e') (hcompat : ScalarCompat o Γ d e)
    (hscalar : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ sv, v = .s sv) :
    StmtSound o Γ (.assign a .nil e) (.assignAll a e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    obtain ⟨sv, rfl⟩ := hscalar σ τ hR hC v hv
    rw [F_assign_array_nil o.F Γ σ a d av _ hx hσ, hd] at hassign
    simp only [List.map_cons, List.map_nil, Option.map_eq_some_iff] at hassign
    obtain ⟨a', hst, hσ'⟩ := hassign
    subst hσ'
    obtain ⟨w, hw, hrw⟩ := he σ τ hR hC sv hv
    obtain ⟨cv, hcv, hrelc, hty⟩ := convertTo_compat_rel d sv w hrw (hcompat σ τ hR hC sv hv)
    obtain ⟨a'', hst', hty', hdims', hget'⟩ := F_storeArr_all_scalar d av l n hd sv cv hcv
    rw [hst'] at hst
    cases hst
    have hshape := RelA_dims1 hrel l n hd
    obtain ⟨p', hpst, hpshape, hpget⟩ := P_storeArr_all_scalar p n hshape w
    refine ⟨Function.update τ a (some (.arr p')), ?_, ?_, ?_⟩
    · refine P.Exec.assignAll hw hτ ?_
      rw [hshape]; simpa using hpst
    · apply relState_update hR a
      refine RelV.arr ⟨by rw [hpshape, hrel.shape]; simp [FArr.extents, hdims'], ?_⟩
      intro idx hidx
      obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ l ≤ j ∧ j < l + n := by
        simp only [FArr.inBounds, FArr.inBoundsB, hdims', hd] at hidx
        match idx, hidx with
        | [j], hidx =>
          simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
          exact ⟨j, rfl, hidx.1⟩
        | [], hidx => simp [FArr.inBoundsB] at hidx
        | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
      simp only [rebase, FArr.lbs, hdims', hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
        List.zipWith_nil_right]
      rw [hget' j hj, hpget _ (by omega)]
      exact hrelc
    · obtain ⟨av₀, hσ₀, hty_av, _, _, hwt_av⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
      rw [hσ] at hσ₀; cases hσ₀
      refine conforms_update_array hC hS a d hx av _ hσ hty' hdims' ?_
      refine wellTyped_rank1_of _ av l n hd hdims' hty' (fun j hj => ?_)
      rw [hget' j hj, hty, hty_av]

end Recast.Rules
