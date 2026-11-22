import RecastProofs.Rules.Index

/-!
# Rules FNP-R0101 (SUM), FNP-R0108 (PRODUCT), FNP-R0109 (SIZE), FNP-R0110 (LBOUND)
on rank-1 arrays
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Rank-1 array facts -/

theorem FArr_size_rank1 (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) : a.size = n := by
  simp [FArr.size, FArr.extents, hd]

theorem PArr_size_rank1 (p : PArr) (n : ℕ) (hs : p.shape = [n]) : p.size = n := by
  simp [PArr.size, hs]

theorem allCoords_rank1 (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) :
    F.allCoords a = List.map (fun (k : ℕ) => [lb + (k : ℤ)]) (List.range n) := by
  unfold F.allCoords
  rw [FArr_size_rank1 a lb n hd]
  apply List.map_congr_left
  intro k hk
  rw [List.mem_range] at hk
  simp [FArr.lbs, FArr.extents, hd, CM.coords, Nat.mod_eq_of_lt hk]

theorem ravelF_rank1 (p : PArr) (n : ℕ) (hs : p.shape = [n]) :
    P.ravelF p = List.map (fun (k : ℕ) => p.get [k]) (List.range n) := by
  unfold P.ravelF
  rw [PArr_size_rank1 p n hs]
  apply List.map_congr_left
  intro k hk
  rw [List.mem_range] at hk
  simp [hs, CM.coords, Nat.mod_eq_of_lt hk]

/-! ### Folds over related element sequences -/

theorem foldl_bind_none {α β : Type} (g : β → α → Option α) (l : List β) :
    l.foldl (fun acc k => acc.bind (g k)) none = none := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, Option.bind_none]; exact ih

theorem fold_rel_range (opF : F.BinOp) (hop : opF = .add ∨ opF = .sub ∨ opF = .mul)
    (fF : ℕ → SVal) (fP : ℕ → PVal) :
    ∀ (l : List ℕ), (∀ k ∈ l, RelS (fF k) (fP k)) → ∀ (z : SVal) (z' : PVal), RelS z z' → ∀ r,
      l.foldl (fun acc k => acc.bind fun x => F.arith opF x (fF k)) (some z) = some r →
      ∃ r', l.foldl (fun acc k => acc.bind fun s => P.arith (toP opF) s (fP k)) (some z') = some r' ∧ RelS r r' := by
  intro l
  induction l with
  | nil => intro _ z z' hz r hr; simp only [List.foldl_nil, Option.some.injEq] at hr; subst hr; exact ⟨z', rfl, hz⟩
  | cons k ks ih =>
    intro hrel z z' hz r hr
    simp only [List.foldl_cons, Option.bind_some] at hr
    cases h1 : F.arith opF z (fF k) with
    | none => rw [h1] at hr; rw [foldl_bind_none (fun k x => F.arith opF x (fF k))] at hr; cases hr
    | some z₁ =>
      rw [h1] at hr
      obtain ⟨z₁', hz₁', hrel₁⟩ := arith_addsubmul_rel hop hz (hrel k (List.mem_cons_self ..)) h1
      obtain ⟨r', hr', hrr'⟩ := ih (fun k hk => hrel k (List.mem_cons_of_mem _ hk)) z₁ z₁' hrel₁ r hr
      refine ⟨r', ?_, hrr'⟩
      simp only [List.foldl_cons, Option.bind_some, hz₁']
      exact hr'

/-- The whole-array fold of a rank-1 pair of related arrays. -/
theorem reduction_rank1 (opF : F.BinOp) (hop : opF = .add ∨ opF = .sub ∨ opF = .mul)
    (av : FArr) (p : PArr) (lb : ℤ) (n : ℕ) (hd : av.dims = [(lb, n)]) (hA : RelA av p)
    (z : SVal) (z' : PVal) (hz : RelS z z') (r : SVal)
    (hF : F.foldElems av (F.arith opF) z = some r) :
    ∃ r', (P.ravelF p).foldl (fun acc x => acc.bind fun s => P.arith (toP opF) s x) (some z') = some r' ∧ RelS r r' := by
  have hshape := RelA_dims1 hA lb n hd
  unfold F.foldElems at hF
  rw [allCoords_rank1 av lb n hd, List.foldl_map] at hF
  obtain ⟨r', hr', hrr'⟩ := fold_rel_range opF hop (fun k => av.get [lb + (k : ℤ)]) (fun k => p.get [k])
    (List.range n) (fun k hk => by
      rw [List.mem_range] at hk
      have := RelA_get1 hA lb n hd (lb + k) ⟨by omega, by omega⟩
      simpa using this) z z' hz r hF
  refine ⟨r', ?_, hrr'⟩
  rw [ravelF_rank1 p n hshape, List.foldl_map]
  exact hr'

/-- Python's seed (`_f_vsum`/`np.prod` model): chosen from the first element's runtime type. -/
def pySeed (cI : ℤ) (cR : ℝ) : PVal → PVal
  | .int _ => .int cI
  | _ => .float cR

/-- Fortran's typed seed (`zeroOf`/`oneOf`). -/
def fSeed (cI : ℤ) (cR : ℝ) : FTy → Option SVal
  | .int k => some (.int k cI)
  | .real k => some (.real k cR)
  | _ => none

theorem seed_rel_int (cI : ℤ) (cR : ℝ) (k : IKind) (m : ℤ) (w : PVal) (hrel : RelS (.int k m) w) :
    RelS (.int k cI) (pySeed cI cR w) := by
  rw [RelS.int_eq hrel]; exact RelS.int

theorem seed_rel_real (cI : ℤ) (cR : ℝ) (hc : (cI : ℝ) = cR) (k : RKind) (x : ℝ) (w : PVal)
    (hrel : RelS (.real k x) w) : RelS (.real k cR) (pySeed cI cR w) := by
  rcases RelS.real_cases hrel with ⟨m, rfl, _⟩ | rfl
  · exact RelS.real (by simp [pySeed, P.toR, hc])
  · exact RelS.realF

/-! ### FNP-R0101: `sum(a)` -> `_f_vsum(a')` -/

/-- Premise: the argument is a homogeneous non-empty rank-1 array. -/
def Rank1Arg (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av, F.eval o.F σ e = some (.arr av) →
    (∃ lb n, av.dims = [(lb, n)] ∧ 0 < n) ∧ FArr.WellTyped av

def FNP_R0101_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr), SoundPair o Γ e e' → Rank1Arg o Γ e →
    SoundScalar o Γ (.call .sum (.cons e .nil) none) (Tr.fold1 .f_vsum e')

def FNP_R0108_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr), SoundPair o Γ e e' → Rank1Arg o Γ e →
    SoundScalar o Γ (.call .product (.cons e .nil) none) (Tr.fold1 .np_prod e')

/-- Shared proof of SUM and PRODUCT. -/
theorem reduction_sound (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (he : SoundPair o Γ e e')
    (hr : Rank1Arg o Γ e) (f : F.Intr) (h : P.Helper) (opF : F.BinOp) (cI : ℤ) (cR : ℝ) (hc : (cI : ℝ) = cR)
    (hop : opF = .add ∨ opF = .mul)
    (hF : ∀ av : FArr, F.intrinsic o.F f [.arr av] none =
      (fSeed cI cR av.elemTy).bind fun z => (F.foldElems av (F.arith opF) z).map F.Val.s)
    (hP : ∀ p : PArr, P.helper o.P h [.arr p] =
      ((P.ravelF p).foldl (fun acc x => acc.bind fun s => P.arith (toP opF) s x)
        (some (pySeed cI cR (p.get (p.shape.map fun _ => 0))))).map P.Val.s)
    (hFs : ∀ sv : SVal, F.intrinsic o.F f [.s sv] none = none) :
    SoundScalar o Γ (.call f (.cons e .nil) none) (Tr.fold1 h e') := by
  intro σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | s sv => rw [hFs] at hint; cases hint
  | arr av =>
    obtain ⟨⟨lb, n, hd, hn⟩, hwt⟩ := hr σ τ hR hC av hva
    obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hva
    cases hrw with
    | arr hA =>
      rename_i p
      rw [hF] at hint
      simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at hint
      obtain ⟨z, hz, r, hfold, hcz⟩ := hint
      cases hcz
      rw [P_eval_call1, hw, Option.bind_some, hP]
      have hshape := RelA_dims1 hA lb n hd
      have hel : RelS (av.get [lb]) (p.get [0]) := by
        have := RelA_get1 hA lb n hd lb ⟨le_refl _, by omega⟩; simpa using this
      have hty : (av.get [lb]).ty = av.elemTy := hwt [lb] (by simp [FArr.inBounds, FArr.inBoundsB, hd]; omega)
      rw [hshape]
      simp only [List.map_cons, List.map_nil]
      have hop' : opF = .add ∨ opF = .sub ∨ opF = .mul := by rcases hop with rfl | rfl <;> simp
      have hseed : RelS z (pySeed cI cR (p.get [0])) := by
        cases hg : av.get [lb] with
        | int k m =>
          rw [hg] at hel hty
          simp only [SVal.ty] at hty
          rw [← hty] at hz
          simp only [fSeed, Option.some.injEq] at hz
          subst hz
          exact seed_rel_int cI cR k m _ hel
        | real k x =>
          rw [hg] at hel hty
          simp only [SVal.ty] at hty
          rw [← hty] at hz
          simp only [fSeed, Option.some.injEq] at hz
          subst hz
          exact seed_rel_real cI cR hc k x _ hel
        | bool _ => rw [hg] at hty; simp only [SVal.ty] at hty; rw [← hty] at hz; simp [fSeed] at hz
        | str _ => rw [hg] at hty; simp only [SVal.ty] at hty; rw [← hty] at hz; simp [fSeed] at hz
      obtain ⟨r', hr', hrr'⟩ := reduction_rank1 opF hop' av p lb n hd hA z _ hseed _ hfold
      exact ⟨r', by rw [hr']; rfl, hrr'⟩

theorem FNP_R0101_sound : FNP_R0101_obligation := by
  intro o Γ e e' he hr
  refine reduction_sound o Γ e e' he hr .sum .f_vsum .add 0 0 (by simp) (Or.inl rfl) ?_ ?_ ?_
  · intro av
    show (F.zeroOf av.elemTy).bind _ = _
    cases av.elemTy <;> rfl
  · intro p; rfl
  · intro sv; rfl

theorem FNP_R0108_prod_sound : FNP_R0108_obligation := by
  intro o Γ e e' he hr
  refine reduction_sound o Γ e e' he hr .product .np_prod .mul 1 1 (by simp) (Or.inr rfl) ?_ ?_ ?_
  · intro av
    show (F.oneOf av.elemTy).bind _ = _
    cases av.elemTy <;> rfl
  · intro p; rfl
  · intro sv; rfl

/-! ### FNP-R0109: `size(a)` -> `np.size(a')` -/

def FNP_R0109_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr), SoundPair o Γ e e' →
    SoundScalar o Γ (.call .size (.cons e .nil) none) (Tr.fold1 .np_size e')

theorem FNP_R0109_sound : FNP_R0109_obligation := by
  intro o Γ e e' he σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | s sv => rw [show F.intrinsic o.F .size [.s sv] none = none from rfl] at hint; cases hint
  | arr av =>
    rw [show F.intrinsic o.F .size [.arr av] none = some (.s (.int .i4 av.size)) from rfl] at hint
    cases hint
    obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hva
    cases hrw with
    | arr hA =>
      rename_i p
      have hsz : p.size = av.size := by simp [PArr.size, FArr.size, hA.shape]
      refine ⟨.int p.size, ?_, ?_⟩
      · rw [P_eval_call1, hw]; rfl
      · rw [hsz]; exact RelS.int

/-! ### FNP-R0110: `lbound(a, d)` -> `_f_lbound(a', d')` -/

def FNP_R0110_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e d : F.Expr) (e' d' : P.Expr),
    SoundPair o Γ e e' → SoundScalar o Γ d d' → IntTyped o Γ d →
    SoundScalar o Γ (.call .lbound (.cons e (.cons d .nil)) none) (Tr.call2 .f_lbound e' d')

/-- Sound when the array has unit lower bounds (rank 1). -/
theorem FNP_R0110_sound_unit (o : Oracles) (Γ : F.Env) (a : String) (d : F.Expr) (d' : P.Expr)
    (hu : UnitRank1 o Γ a) (hd : SoundScalar o Γ d d') (hint : IntTyped o Γ d) :
    SoundScalar o Γ (.call .lbound (.cons (.var a) (.cons d .nil)) none) (Tr.call2 .f_lbound (.name a) d') := by
  intro σ τ hR hC v hv
  obtain ⟨av, p, n, hσ, hτ, hrel, hdims⟩ := hu σ τ hR hC
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint'⟩ := hv
  simp only [F.eval, hσ, Option.some.injEq] at hva
  subst hva
  cases vb with
  | arr _ => rw [show F.intrinsic o.F .lbound [.arr av, .arr _] none = none from rfl] at hint'; cases hint'
  | s dv =>
    obtain ⟨k, dvi, rfl⟩ := hint σ τ hR hC dv hvb
    obtain ⟨w, hw, hrw⟩ := hd σ τ hR hC _ hvb
    rw [RelS.int_eq hrw] at hw
    have e : F.intrinsic o.F .lbound [.arr av, .s (.int k dvi)] none =
        (if 1 ≤ dvi ∧ dvi ≤ av.rank then some (.s (.int .i4 (av.lbs.getD (dvi - 1).toNat 0))) else none) := rfl
    rw [e] at hint'
    split_ifs at hint' with hrange
    · cases hint'
      have hrank : av.rank = 1 := by simp [FArr.rank, hdims]
      rw [hrank] at hrange
      have hd1 : dvi = 1 := by omega
      subst hd1
      refine ⟨.int 1, ?_, ?_⟩
      · rw [P_eval_call2, hw]
        simp only [P.eval, hτ, Option.bind_some]
        rfl
      · simp [FArr.lbs, hdims]; exact RelS.int

/-! Concrete zero-based two-element array for the refutation. -/

def twoF0 : FArr := { elemTy := .real .r8, dims := [(0, 2)], get := fun _ => .real .r8 0 }
theorem twoRel0 : RelA twoF0 twoP := ⟨rfl, fun _ _ => RelS.realF⟩

def declB : F.Decl := ⟨.real .r8, [.intLit .i4 0], 0⟩
def envB : F.Env := fun x => if x = "a" then some declB else none
def stateB : F.State := fun x => if x = "a" then some (.arr twoF0) else none
def pstateB : P.State := fun x => if x = "a" then some (.arr twoP) else none

theorem relStateB : RelState envB stateB pstateB := by
  intro x d hx
  simp only [envB] at hx
  split_ifs at hx with hxa
  subst hxa
  exact ⟨.arr twoF0, .arr twoP, by simp [stateB], by simp [pstateB], RelV.arr twoRel0⟩

theorem conformsB : Conforms noPrim.F envB stateB := by
  intro x d hx
  simp only [envB] at hx
  split_ifs at hx with hxa
  subst hxa
  cases hx
  refine ⟨fun h => by simp [declB] at h, fun _ => ⟨twoF0, by simp [stateB], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
  intro i hi
  simp only [declB] at hi
  match i, hi with
  | 0, _ => simp [declB, F_eval_intLit, twoF0, FArr.lbs]

theorem soundPair_varB : SoundPair noPrim envB (.var "a") (.name "a") :=
  FNP_R0010_sound noPrim envB "a" declB (by simp [envB])

/-- FNP-D0006: `lbound(a, 1)` with `a(0:1)` is `0`; `_f_lbound` answers `1`. -/
theorem FNP_R0110_refuted : ¬ FNP_R0110_obligation := by
  intro h
  have := h noPrim envB (.var "a") (.intLit .i4 1) (.name "a") (.intLit 1) soundPair_varB
    (FNP_R0001_sound _ _ _ _) (intLit_intTyped _ _ .i4 1) stateB pstateB relStateB conformsB (.int .i4 0) (by
      rw [F_eval_call2, F_eval_intLit]
      simp only [F.eval, stateB, if_true, Option.bind_some]
      rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call2] at hw
  simp only [P.eval, pstateB, if_true, Option.bind_some] at hw
  rw [show P.helper noPrim.P .f_lbound [.arr twoP, .s (.int 1)] = some (.s (.int 1)) from rfl] at hw
  cases hw
  cases hrel

end Recast.Rules
