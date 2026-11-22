import RecastProofs.Rules.Obligations
import RecastProofs.Rules.Reductions
import RecastProofs.Rules.Reductions2
import RecastProofs.Rules.SeqAssoc2

/-!
# Array transforms: FNP-R0102 (RESHAPE), FNP-R0104 (PACK), FNP-R0106 (CSHIFT /
EOSHIFT), FNP-R0107 (MAXLOC / MINLOC), FNP-R0103 (TRANSPOSE / MATMUL),
FNP-R0098 (SHAPE), FNP-R0094 (MVBITS)

The Fortran side is the v1.1 table `F.intrinsicNew` (reached through the
dispatcher `F.intrinsic`); the Python side is the helper table.  Each rule is
stated in the `SoundPair`/`SoundScalar` shape.  Sound sub-cases are proved;
the rules as the emitter accepts them (keyword arguments dropped, NumPy's
row-major enumeration, exact-size `np.reshape`) are refuted by closed
counterexamples.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Unfolding lemmas -/

theorem P_eval_call3T (o : P.Oracle) (τ : P.State) (h : P.Helper) (a b c : P.Expr) :
    P.eval o τ (.call h (.cons a (.cons b (.cons c .nil)))) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => (P.eval o τ c).bind fun vc =>
        P.helper o h [va, vb, vc] := by
  cases h1 : P.eval o τ a <;> cases h2 : P.eval o τ b <;> cases h3 : P.eval o τ c <;>
    simp only [P.eval, P.evalExprs, h1, h2, h3] <;> rfl

theorem P_eval_maskIndex (o : P.Oracle) (τ : P.State) (b m : P.Expr) :
    P.eval o τ (.maskIndex b m) = (P.eval o τ b).bind fun vb => (P.eval o τ m).bind fun vm =>
      match vb, vm with
      | .arr a, .arr ma => P.maskSelect a ma
      | _, _ => none := by
  cases h1 : P.eval o τ b <;> cases h2 : P.eval o τ m <;> simp only [P.eval, h1, h2] <;> rfl

theorem F_eval_arrayCtor2 (o : F.Oracle) (σ : F.State) (k₁ k₂ : IKind) (v₁ v₂ : ℤ) :
    F.eval o σ (.arrayCtor (.cons (.intLit k₁ v₁) (.cons (.intLit k₂ v₂) .nil))) =
      (if [SVal.int k₁ v₁, SVal.int k₂ v₂].all (fun x => x.ty = (SVal.int k₁ v₁).ty)
       then some (.arr (F.listArr (.int k₁) [.int k₁ v₁, .int k₂ v₂])) else none) := by
  simp only [F.eval, F.evalExprs, Option.bind_some, List.mapM, List.mapM.loop]
  rfl

theorem P_eval_npArray2 (o : P.Oracle) (τ : P.State) (v₁ v₂ : ℤ) :
    P.eval o τ (.call .np_array (.cons (.intLit v₁) (.cons (.intLit v₂) .nil))) =
      some (.arr (P.listArr [.int v₁, .int v₂])) := by
  simp only [P.eval, P.evalExprs, Option.bind_some]
  rfl

/-! ### Rank-1 / rank-2 array facts -/

theorem inBounds1 (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (idx : List ℤ) (h : a.inBounds idx) :
    ∃ j, idx = [j] ∧ lb ≤ j ∧ j < lb + n := by
  simp only [FArr.inBounds, FArr.inBoundsB, hd] at h
  match idx, h with
  | [j], h =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨j, rfl, h.1⟩
  | [], h => simp [FArr.inBoundsB] at h
  | _ :: _ :: _, h => simp [FArr.inBoundsB] at h

theorem inBounds2 (a : FArr) (l1 l2 : ℤ) (n1 n2 : ℕ) (hd : a.dims = [(l1, n1), (l2, n2)]) (idx : List ℤ)
    (h : a.inBounds idx) : ∃ i j, idx = [i, j] ∧ l1 ≤ i ∧ i < l1 + n1 ∧ l2 ≤ j ∧ j < l2 + n2 := by
  simp only [FArr.inBounds, FArr.inBoundsB, hd] at h
  match idx, h with
  | [i, j], h =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨i, j, rfl, h.1.1, h.1.2, h.2.1.1, h.2.1.2⟩
  | [], h => simp [FArr.inBoundsB] at h
  | [_], h => simp [FArr.inBoundsB] at h
  | _ :: _ :: _ :: _, h => simp [FArr.inBoundsB] at h

theorem RelA_dims2 {a : FArr} {p : PArr} (h : RelA a p) (l1 l2 : ℤ) (n1 n2 : ℕ)
    (hd : a.dims = [(l1, n1), (l2, n2)]) : p.shape = [n1, n2] := by
  rw [h.shape]; simp [FArr.extents, hd]

theorem RelA_get2 {a : FArr} {p : PArr} (h : RelA a p) (l1 l2 : ℤ) (n1 n2 : ℕ)
    (hd : a.dims = [(l1, n1), (l2, n2)]) (i j : ℤ) (hi : l1 ≤ i ∧ i < l1 + n1) (hj : l2 ≤ j ∧ j < l2 + n2) :
    RelS (a.get [i, j]) (p.get [(i - l1).toNat, (j - l2).toNat]) := by
  have hb : a.inBounds [i, j] := by
    simp [FArr.inBounds, FArr.inBoundsB, hd, hi, hj]
  have := h.elems [i, j] hb
  simpa [rebase, FArr.lbs, hd] using this

theorem getD_map_lt {α β : Type} (f : α → β) (l : List α) (k : ℕ) (hk : k < l.length) (d : β) :
    (l.map f).getD k d = f l[k] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hk]; rfl

theorem getD_lt {α : Type} (l : List α) (k : ℕ) (hk : k < l.length) (d : α) : l.getD k d = l[k] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk]; rfl

/-- Two rank-1 lists whose elements are pairwise related give related unit-origin arrays. -/
theorem relA_listArr (ty : FTy) (l : List SVal) (l' : List PVal) (hlen : l.length = l'.length)
    (hrel : ∀ k (hk : k < l.length), RelS l[k] (l'[k]'(hlen ▸ hk))) : RelA (F.listArr ty l) (P.listArr l') := by
  refine ⟨by simp [F.listArr, P.listArr, FArr.extents, hlen], ?_⟩
  intro idx hin
  obtain ⟨j, rfl, hj1, hj2⟩ := inBounds1 (F.listArr ty l) 1 l.length rfl idx hin
  simp only [F.listArr, P.listArr, rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.headD_cons]
  have hk : (j - 1).toNat < l.length := by omega
  rw [getD_lt l _ hk, getD_lt l' _ (hlen ▸ hk)]
  exact hrel _ hk

/-! ### FNP-R0098: SHAPE -/

def FNP_R0098_shape_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr), SoundPair o Γ a a' →
    SoundPair o Γ (.call .shape (.cons a .nil) none) (Tr.fold1 .np_shape a')

theorem FNP_R0098_shape_sound : FNP_R0098_shape_obligation := by
  intro o Γ a a' ha σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | s x => rw [show F.intrinsic o.F .shape [.s x] none = none from rfl] at hint; cases hint
  | arr av =>
    rw [show F.intrinsic o.F .shape [.arr av] none =
        some (.arr (F.listArr (.int .i4) (av.extents.map fun n => .int .i4 n))) from rfl] at hint
    cases hint
    obtain ⟨w, hw, hrw⟩ := ha σ τ hR hC _ hva
    cases hrw with
    | arr hA =>
      rename_i p
      refine ⟨.arr (P.listArr (p.shape.map fun n => .int n)), ?_, RelV.arr ?_⟩
      · rw [P_eval_call1, hw]; rfl
      · rw [hA.shape]
        refine relA_listArr _ _ _ (by simp) ?_
        intro k hk
        simp only [List.length_map] at hk
        simp only [List.getElem_map]
        exact RelS.int

/-! ### FNP-R0103 (TRANSPOSE part): `transpose(a)` ↦ `np.asfortranarray(a.T)` -/

/-- The argument evaluates to a rank-2 array. -/
def Rank2Arg (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av, F.eval o.F σ e = some (.arr av) →
    ∃ l1 l2 n1 n2, av.dims = [(l1, n1), (l2, n2)]

def FNP_R0103_transpose_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr), SoundPair o Γ a a' → Rank2Arg o Γ a →
    SoundPair o Γ (.call .transpose (.cons a .nil) none) (Tr.fold1 .np_transpose a')

theorem FNP_R0103_transpose_sound : FNP_R0103_transpose_obligation := by
  intro o Γ a a' ha hr σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | s x => rw [show F.intrinsic o.F .transpose [.s x] none = none from rfl] at hint; cases hint
  | arr av =>
    obtain ⟨l1, l2, n1, n2, hd⟩ := hr σ τ hR hC av hva
    rw [show F.intrinsic o.F .transpose [.arr av] none = F.transposeF av from rfl] at hint
    simp only [F.transposeF, hd, Option.some.injEq] at hint
    subst hint
    obtain ⟨w, hw, hrw⟩ := ha σ τ hR hC _ hva
    cases hrw with
    | arr hA =>
      rename_i p
      have hs := RelA_dims2 hA l1 l2 n1 n2 hd
      refine ⟨.arr { shape := [n2, n1], get := fun idx => p.get [idx.getD 1 0, idx.getD 0 0] }, ?_, RelV.arr ?_⟩
      · rw [P_eval_call1, hw, Option.bind_some]
        have e : P.helper o.P .np_transpose [.arr p] = (match p.shape with
          | [n1, n2] => some (.arr { shape := [n2, n1], get := fun idx => p.get [idx.getD 1 0, idx.getD 0 0] })
          | _ => none) := rfl
        rw [e, hs]
      · refine ⟨by simp [FArr.extents], ?_⟩
        intro idx hin
        obtain ⟨i, j, rfl, hi1, hi2, hj1, hj2⟩ := inBounds2 _ 1 1 n2 n1 rfl idx hin
        simp only [rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons, List.zipWith_nil_right,
          List.getD_cons_zero, List.getD_cons_succ]
        have := RelA_get2 hA l1 l2 n1 n2 hd (l1 + j - 1) (l2 + i - 1) ⟨by omega, by omega⟩ ⟨by omega, by omega⟩
        have e1 : l1 + j - 1 - l1 = j - 1 := by omega
        have e2 : l2 + i - 1 - l2 = i - 1 := by omega
        rw [e1, e2] at this
        exact this

/-! ### FNP-R0106: CSHIFT / EOSHIFT on rank-1 arrays -/

/-- The emitted axis argument `(1) - 1`. -/
def axis0 : P.Expr := .bin .sub (.paren (.intLit 1)) (.intLit 1)

theorem P_eval_axis0 (o : P.Oracle) (τ : P.State) : P.eval o τ axis0 = some (.s (.int 0)) := by
  simp [axis0, P.eval, P.lift2, P.arith]

theorem P_eval_negInt (o : P.Oracle) (τ : P.State) (e : P.Expr) (n : ℤ) (h : P.eval o τ e = some (.s (.int n))) :
    P.eval o τ (.paren (.neg e)) = some (.s (.int (-n))) := by
  simp [P.eval, h, P.lift1]

/-- `F.intrinsic` on a scalar first argument is undefined for the two-argument transforms. -/
theorem cshift_scalar_none (o : F.Oracle) (x : SVal) (vb : F.Val) :
    F.intrinsic o (.cshift false) [.s x, vb] none = none := by
  cases vb with
  | s sv => cases sv <;> rfl
  | arr _ => rfl

theorem eoshift_scalar_none (o : F.Oracle) (x : SVal) (vb : F.Val) :
    F.intrinsic o (.eoshift false false) [.s x, vb] none = none := by
  cases vb with
  | s sv => cases sv <;> rfl
  | arr _ => rfl

def FNP_R0106_cshift_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a sh : F.Expr) (a' sh' : P.Expr),
    SoundPair o Γ a a' → SoundScalar o Γ sh sh' → IntTyped o Γ sh → Rank1Arg o Γ a →
    SoundPair o Γ (.call (.cshift false) (.cons a (.cons sh .nil)) none)
      (.call .np_roll (.cons a' (.cons (.paren (.neg sh')) (.cons axis0 .nil))))

theorem FNP_R0106_cshift_sound : FNP_R0106_cshift_obligation := by
  intro o Γ a sh a' sh' ha hsh hint hr σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint'⟩ := hv
  cases va with
  | s x => rw [cshift_scalar_none] at hint'; cases hint'
  | arr av =>
    cases vb with
    | arr _ => rw [show F.intrinsic o.F (.cshift false) [.arr av, .arr _] none = none from rfl] at hint'; cases hint'
    | s sv =>
      obtain ⟨k, s, rfl⟩ := hint σ τ hR hC sv hvb
      obtain ⟨⟨lb, n, hd, hn⟩, _⟩ := hr σ τ hR hC av hva
      rw [show F.intrinsic o.F (.cshift false) [.arr av, .s (.int k s)] none = F.cshift1 av s from rfl] at hint'
      simp only [F.cshift1, hd, if_neg hn.ne', Option.some.injEq] at hint'
      subst hint'
      obtain ⟨w, hw, hrw⟩ := ha σ τ hR hC _ hva
      obtain ⟨ws, hws, hrws⟩ := hsh σ τ hR hC _ hvb
      rw [RelS.int_eq hrws] at hws
      cases hrw with
      | arr hA =>
        rename_i p
        have hs := RelA_dims1 hA lb n hd
        refine ⟨.arr { shape := [n], get := fun idx => p.get [(((idx.headD 0 : ℕ) : ℤ) - (-s)) % (n : ℤ) |>.toNat] },
          ?_, RelV.arr ?_⟩
        · rw [P_eval_call3T, hw, P_eval_negInt o.P τ sh' s hws, P_eval_axis0]
          simp only [Option.bind_some]
          have e : P.helper o.P .np_roll [.arr p, .s (.int (-s)), .s (.int 0)] = (match p.shape with
            | [n] => if (0 : ℤ) ≠ 0 then none
                else if n = 0 then some (.arr p)
                else some (.arr { shape := [n], get := fun idx => p.get [(((idx.headD 0 : ℕ) : ℤ) - (-s)) % (n : ℤ) |>.toNat] })
            | _ => none) := rfl
          rw [e, hs]
          simp [hn.ne']
        · refine ⟨by simp [FArr.extents], ?_⟩
          intro idx hin
          obtain ⟨j, rfl, hj1, hj2⟩ := inBounds1 _ 1 n rfl idx hin
          simp only [rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons, List.zipWith_nil_right,
            List.headD_cons]
          have hj' : (((j - 1).toNat : ℕ) : ℤ) = j - 1 := Int.toNat_of_nonneg (by omega)
          rw [hj']
          have hn' : (0 : ℤ) < n := by exact_mod_cast hn
          have hr0 : 0 ≤ (j - 1 + s) % (n : ℤ) := Int.emod_nonneg _ hn'.ne'
          have hr1 : (j - 1 + s) % (n : ℤ) < n := Int.emod_lt_of_pos _ hn'
          have := RelA_get1 hA lb n hd (lb + (j - 1 + s) % (n : ℤ)) ⟨by omega, by omega⟩
          have e1 : lb + (j - 1 + s) % (n : ℤ) - lb = (j - 1 + s) % (n : ℤ) := by ring
          have e2 : (j - 1 - -s) = j - 1 + s := by ring
          rw [e1] at this
          rw [e2]
          exact this

/-- The element type has a zero that both sides agree on (INTEGER, REAL, LOGICAL). -/
def ZeroTyped (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av, F.eval o.F σ e = some (.arr av) → av.elemTy ≠ .str

def FNP_R0106_eoshift_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a sh : F.Expr) (a' sh' : P.Expr),
    SoundPair o Γ a a' → SoundScalar o Γ sh sh' → IntTyped o Γ sh → Rank1Arg o Γ a → ZeroTyped o Γ a →
    SoundPair o Γ (.call (.eoshift false false) (.cons a (.cons sh .nil)) none)
      (.call .f_eoshift (.cons a' (.cons sh' (.cons axis0 .nil))))

theorem FNP_R0106_eoshift_sound : FNP_R0106_eoshift_obligation := by
  intro o Γ a sh a' sh' ha hsh hint hr hz σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint'⟩ := hv
  cases va with
  | s x => rw [eoshift_scalar_none] at hint'; cases hint'
  | arr av =>
    cases vb with
    | arr _ => rw [show F.intrinsic o.F (.eoshift false false) [.arr av, .arr _] none = none from rfl] at hint'; cases hint'
    | s sv =>
      obtain ⟨k, s, rfl⟩ := hint σ τ hR hC sv hvb
      obtain ⟨⟨lb, n, hd, hn⟩, hwt⟩ := hr σ τ hR hC av hva
      have hstr := hz σ τ hR hC av hva
      rw [show F.intrinsic o.F (.eoshift false false) [.arr av, .s (.int k s)] none = F.eoshift1 av s none from rfl] at hint'
      simp only [F.eoshift1, hd, Option.getD_none, Option.some.injEq] at hint'
      subst hint'
      obtain ⟨w, hw, hrw⟩ := ha σ τ hR hC _ hva
      obtain ⟨ws, hws, hrws⟩ := hsh σ τ hR hC _ hvb
      rw [RelS.int_eq hrws] at hws
      cases hrw with
      | arr hA =>
        rename_i p
        have hs := RelA_dims1 hA lb n hd
        refine ⟨.arr { shape := [n], get := fun idx =>
            let j := ((idx.headD 0 : ℕ) : ℤ) + s
            if 0 ≤ j ∧ j < n then p.get [j.toNat] else P.zeroLike (p.get [0]) }, ?_, RelV.arr ?_⟩
        · rw [P_eval_call3T, hw, hws, P_eval_axis0]
          simp only [Option.bind_some]
          have e : P.helper o.P .f_eoshift [.arr p, .s (.int s), .s (.int 0)] = (match p.shape with
            | [n] => if (0 : ℤ) ≠ 0 then none
                else
                  let z := if n = 0 then PVal.none else P.zeroLike (p.get [0])
                  some (.arr { shape := [n], get := fun idx =>
                    let j := ((idx.headD 0 : ℕ) : ℤ) + s
                    if 0 ≤ j ∧ j < n then p.get [j.toNat] else z })
            | _ => none) := rfl
          rw [e, hs]
          simp [hn.ne']
        · refine ⟨by simp [FArr.extents], ?_⟩
          intro idx hin
          obtain ⟨j, rfl, hj1, hj2⟩ := inBounds1 _ 1 n rfl idx hin
          simp only [rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons, List.zipWith_nil_right,
            List.headD_cons]
          have hj' : (((j - 1).toNat : ℕ) : ℤ) = j - 1 := Int.toNat_of_nonneg (by omega)
          rw [hj']
          by_cases hin' : 0 ≤ j - 1 + s ∧ j - 1 + s < n
          · simp only [eq_true hin', if_true]
            have := RelA_get1 hA lb n hd (lb + (j - 1 + s)) ⟨by omega, by omega⟩
            have e1 : lb + (j - 1 + s) - lb = j - 1 + s := by ring
            rw [e1] at this
            exact this
          · simp only [eq_false hin', if_false]
            -- the boundary: the zero of the element type on both sides
            have h0 := RelA_get1 hA lb n hd lb ⟨le_refl _, by omega⟩
            simp only [sub_self, Int.toNat_zero] at h0
            have hty : (av.get [lb]).ty = av.elemTy := hwt [lb] (inBounds_rank1 av lb n hd lb ⟨le_refl _, by omega⟩)
            cases hg : av.get [lb] with
            | int k' m =>
              rw [hg] at h0 hty
              rw [RelS.int_eq h0]
              simp only [SVal.ty] at hty
              rw [← hty]
              exact RelS.int
            | real k' x =>
              rw [hg] at h0 hty
              simp only [SVal.ty] at hty
              rw [← hty]
              rcases RelS.real_cases h0 with ⟨m, hm, _⟩ | hm
              · rw [hm]; exact RelS.real (by simp [P.zeroLike, P.toR])
              · rw [hm]; exact RelS.real (by simp [P.zeroLike, P.toR])
            | bool b =>
              rw [hg] at h0 hty
              rw [RelS.bool_eq h0]
              simp only [SVal.ty] at hty
              rw [← hty]
              exact RelS.bool
            | str t =>
              rw [hg] at hty
              simp only [SVal.ty] at hty
              exact absurd hty.symm hstr

/-! ### Closed array arguments for the counterexamples

Array constructors of literals (`[v₁, …, vₙ]` ↦ `np.array([v₁, …, vₙ])`) are the
smallest closed argument pairs the transform refutations need: they evaluate in
any state, so the counterexamples use `emptyEnv`/`emptyF`/`emptyP`. -/

def ictor1 (a : ℤ) : F.Expr := .arrayCtor (.cons (.intLit .i4 a) .nil)
def ictor1P (a : ℤ) : P.Expr := .call .np_array (.cons (.intLit a) .nil)

def ictor2 (a b : ℤ) : F.Expr := .arrayCtor (.cons (.intLit .i4 a) (.cons (.intLit .i4 b) .nil))
def ictor2P (a b : ℤ) : P.Expr := .call .np_array (.cons (.intLit a) (.cons (.intLit b) .nil))

def ictor4 (a b c d : ℤ) : F.Expr :=
  .arrayCtor (.cons (.intLit .i4 a) (.cons (.intLit .i4 b) (.cons (.intLit .i4 c) (.cons (.intLit .i4 d) .nil))))
def ictor4P (a b c d : ℤ) : P.Expr :=
  .call .np_array (.cons (.intLit a) (.cons (.intLit b) (.cons (.intLit c) (.cons (.intLit d) .nil))))

def ictor6 (a b c d e f : ℤ) : F.Expr :=
  .arrayCtor (.cons (.intLit .i4 a) (.cons (.intLit .i4 b) (.cons (.intLit .i4 c)
    (.cons (.intLit .i4 d) (.cons (.intLit .i4 e) (.cons (.intLit .i4 f) .nil))))))
def ictor6P (a b c d e f : ℤ) : P.Expr :=
  .call .np_array (.cons (.intLit a) (.cons (.intLit b) (.cons (.intLit c)
    (.cons (.intLit d) (.cons (.intLit e) (.cons (.intLit f) .nil))))))

def bctor2 (a b : Bool) : F.Expr := .arrayCtor (.cons (.boolLit a) (.cons (.boolLit b) .nil))
def bctor2P (a b : Bool) : P.Expr := .call .np_array (.cons (.boolLit a) (.cons (.boolLit b) .nil))

def bctor4 (a b c d : Bool) : F.Expr :=
  .arrayCtor (.cons (.boolLit a) (.cons (.boolLit b) (.cons (.boolLit c) (.cons (.boolLit d) .nil))))
def bctor4P (a b c d : Bool) : P.Expr :=
  .call .np_array (.cons (.boolLit a) (.cons (.boolLit b) (.cons (.boolLit c) (.cons (.boolLit d) .nil))))

theorem ictor1_sound (o : Oracles) (Γ : F.Env) (a : ℤ) : SoundPair o Γ (ictor1 a) (ictor1P a) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (ictor1 a) = some (.arr (F.listArr (.int .i4) [.int .i4 a])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.int a]), rfl, RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.int

theorem ictor2_sound (o : Oracles) (Γ : F.Env) (a b : ℤ) : SoundPair o Γ (ictor2 a b) (ictor2P a b) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (ictor2 a b) = some (.arr (F.listArr (.int .i4) [.int .i4 a, .int .i4 b])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.int a, .int b]), rfl, RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.int
  | 1, _ => exact RelS.int

theorem ictor4_sound (o : Oracles) (Γ : F.Env) (a b c d : ℤ) :
    SoundPair o Γ (ictor4 a b c d) (ictor4P a b c d) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (ictor4 a b c d)
      = some (.arr (F.listArr (.int .i4) [.int .i4 a, .int .i4 b, .int .i4 c, .int .i4 d])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.int a, .int b, .int c, .int d]), rfl, RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.int
  | 1, _ => exact RelS.int
  | 2, _ => exact RelS.int
  | 3, _ => exact RelS.int

theorem ictor6_sound (o : Oracles) (Γ : F.Env) (a b c d e f : ℤ) :
    SoundPair o Γ (ictor6 a b c d e f) (ictor6P a b c d e f) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (ictor6 a b c d e f)
      = some (.arr (F.listArr (.int .i4)
          [.int .i4 a, .int .i4 b, .int .i4 c, .int .i4 d, .int .i4 e, .int .i4 f])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.int a, .int b, .int c, .int d, .int e, .int f]), rfl,
    RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.int
  | 1, _ => exact RelS.int
  | 2, _ => exact RelS.int
  | 3, _ => exact RelS.int
  | 4, _ => exact RelS.int
  | 5, _ => exact RelS.int

theorem bctor2_sound (o : Oracles) (Γ : F.Env) (a b : Bool) : SoundPair o Γ (bctor2 a b) (bctor2P a b) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (bctor2 a b) = some (.arr (F.listArr .bool [.bool a, .bool b])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.bool a, .bool b]), rfl, RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.bool
  | 1, _ => exact RelS.bool

theorem bctor4_sound (o : Oracles) (Γ : F.Env) (a b c d : Bool) :
    SoundPair o Γ (bctor4 a b c d) (bctor4P a b c d) := by
  intro σ τ _ _ v hv
  rw [show F.eval o.F σ (bctor4 a b c d)
      = some (.arr (F.listArr .bool [.bool a, .bool b, .bool c, .bool d])) from rfl] at hv
  cases hv
  refine ⟨.arr (P.listArr [.bool a, .bool b, .bool c, .bool d]), rfl,
    RelV.arr (relA_listArr _ _ _ rfl ?_)⟩
  intro k hk
  match k, hk with
  | 0, _ => exact RelS.bool
  | 1, _ => exact RelS.bool
  | 2, _ => exact RelS.bool
  | 3, _ => exact RelS.bool

/-! ### FNP-R0102: RESHAPE

`Expressions._array_transform` renders every `reshape` as
`np.reshape(a, shape, order='F')` and drops `ORDER=` and `PAD=`
(expressions.py:1573-1577).  Three things separate the two sides:

* `ORDER=` permutes the subscript order the result is filled in (F2018
  16.9.163); the emitted call always fills in column-major order (FNP-D0017).
* `PAD=` extends a SOURCE that is too short; `np.reshape` raises instead.
* `PRODUCT(SHAPE)` may be *smaller* than `SIZE(SOURCE)` with `PAD` absent
  (F2018 16.9.163 requires only `≥`); `np.reshape` requires equality. -/

/-- The rule as the emitter accepts it: any shape, no keyword arguments. -/
def FNP_R0102_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a shp : F.Expr) (a' shp' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ shp shp' →
    SoundPair o Γ (.call (.reshape false false) (.cons a (.cons shp .nil)) none)
      (Tr.call2 .np_reshapeF a' shp')

/-- With `PAD=` present (the argument is dropped). -/
def FNP_R0102_pad_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a shp pad : F.Expr) (a' shp' pad' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ shp shp' → SoundPair o Γ pad pad' →
    SoundPair o Γ (.call (.reshape true false) (.cons a (.cons shp (.cons pad .nil))) none)
      (Tr.call2 .np_reshapeF a' shp')

/-- With `ORDER=` present (the argument is dropped). -/
def FNP_R0102_order_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a shp ord : F.Expr) (a' shp' ord' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ shp shp' → SoundPair o Γ ord ord' →
    SoundPair o Γ (.call (.reshape false true) (.cons a (.cons shp (.cons ord .nil))) none)
      (Tr.call2 .np_reshapeF a' shp')

/-- FNP-D0017 (ORDER=): `reshape([1,…,6], [2,3], order=[2,1])` fills the result
with the second subscript varying fastest, so its element `(2,1)` is `4`; the
emitted `np.reshape(…, order='F')` fills column-major and puts `2` there.
(gfortran `1 4 2 5 3 6` vs the translation's `1 2 3 4 5 6`, `probes/reshape_order`.) -/
theorem FNP_R0102_refuted_order : ¬ FNP_R0102_order_obligation := by
  intro h
  have hs := h noPrim emptyEnv (ictor6 1 2 3 4 5 6) (ictor2 2 3) (ictor2 2 1)
    (ictor6P 1 2 3 4 5 6) (ictor2P 2 3) (ictor2P 2 1)
    (ictor6_sound _ _ _ _ _ _ _ _) (ictor2_sound _ _ _ _) (ictor2_sound _ _ _ _)
  obtain ⟨r, hF, hget, hdims⟩ : ∃ r : FArr,
      F.eval noPrim.F emptyF (.call (.reshape false true)
          (.cons (ictor6 1 2 3 4 5 6) (.cons (ictor2 2 3) (.cons (ictor2 2 1) .nil))) none) = some (.arr r) ∧
      r.get [2, 1] = .int .i4 4 ∧ r.dims = [(1, 2), (1, 3)] := ⟨_, rfl, rfl, rfl⟩
  obtain ⟨w, hw, hrel⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  obtain ⟨q, hP, hqget, _⟩ : ∃ q : PArr,
      P.eval noPrim.P emptyP (Tr.call2 .np_reshapeF (ictor6P 1 2 3 4 5 6) (ictor2P 2 3)) = some (.arr q) ∧
      q.get [1, 0] = .int 2 ∧ q.shape = [2, 3] := ⟨_, rfl, rfl, rfl⟩
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have hb : r.inBounds [2, 1] := by simp [FArr.inBounds, FArr.inBoundsB, hdims]
    have hre := hA.elems [2, 1] hb
    rw [hget] at hre
    have he : rebase r.lbs [2, 1] = [1, 0] := by simp [rebase, FArr.lbs, hdims]
    rw [he, hqget] at hre
    exact absurd (PVal.int.inj (RelS.int_eq hre)) (by decide)

/-- FNP-D0017 (PAD=): `reshape([1,2,3,4], [2,3], pad=[0])` is defined in Fortran
(the result is padded); `np.reshape` of a 4-element array into `(2,3)` raises,
so the target is undefined where the source is defined. -/
theorem FNP_R0102_refuted_pad : ¬ FNP_R0102_pad_obligation := by
  intro h
  have hs := h noPrim emptyEnv (ictor4 1 2 3 4) (ictor2 2 3) (ictor1 0)
    (ictor4P 1 2 3 4) (ictor2P 2 3) (ictor1P 0)
    (ictor4_sound _ _ _ _ _ _) (ictor2_sound _ _ _ _) (ictor1_sound _ _ _)
  obtain ⟨r, hF⟩ : ∃ r : F.Val,
      F.eval noPrim.F emptyF (.call (.reshape true false)
        (.cons (ictor4 1 2 3 4) (.cons (ictor2 2 3) (.cons (ictor1 0) .nil))) none) = some r := ⟨_, rfl⟩
  obtain ⟨w, hw, _⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  rw [show P.eval noPrim.P emptyP (Tr.call2 .np_reshapeF (ictor4P 1 2 3 4) (ictor2P 2 3)) = none from rfl] at hw
  cases hw

/-- The plain rule is refuted as well: `PRODUCT(SHAPE)` smaller than
`SIZE(SOURCE)` is legal Fortran (F2018 16.9.163 requires only `≥` when `PAD` is
absent) and `np.reshape` requires equality.  The sound statement needs the
premise that the two agree (`FNP_R0102_sound_exact`). -/
theorem FNP_R0102_refuted_shrink : ¬ FNP_R0102_obligation := by
  intro h
  have hs := h noPrim emptyEnv (ictor4 1 2 3 4) (ictor1 2) (ictor4P 1 2 3 4) (ictor1P 2)
    (ictor4_sound _ _ _ _ _ _) (ictor1_sound _ _ _)
  obtain ⟨r, hF⟩ : ∃ r : F.Val,
      F.eval noPrim.F emptyF (.call (.reshape false false)
        (.cons (ictor4 1 2 3 4) (.cons (ictor1 2) .nil)) none) = some r := ⟨_, rfl⟩
  obtain ⟨w, hw, _⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  rw [show P.eval noPrim.P emptyP (Tr.call2 .np_reshapeF (ictor4P 1 2 3 4) (ictor1P 2)) = none from rfl] at hw
  cases hw

/-! ### Closed rank-2 arguments (through RESHAPE)

The only closed rank-2 expressions the fragment has are `reshape` of an array
constructor; the pair is sound at these concrete instances, which is all the
rank-2 counterexamples need. -/

def resh22 (a b c d : ℤ) : F.Expr :=
  .call (.reshape false false) (.cons (ictor4 a b c d) (.cons (ictor2 2 2) .nil)) none
def resh22P (a b c d : ℤ) : P.Expr := Tr.call2 .np_reshapeF (ictor4P a b c d) (ictor2P 2 2)

def bresh22 (a b c d : Bool) : F.Expr :=
  .call (.reshape false false) (.cons (bctor4 a b c d) (.cons (ictor2 2 2) .nil)) none
def bresh22P (a b c d : Bool) : P.Expr := Tr.call2 .np_reshapeF (bctor4P a b c d) (ictor2P 2 2)

theorem resh22_sound (o : Oracles) (Γ : F.Env) (a b c d : ℤ) :
    SoundPair o Γ (resh22 a b c d) (resh22P a b c d) := by
  intro σ τ _ _ v hv
  obtain ⟨r, hF, hdims, h11, h21, h12, h22⟩ : ∃ r : FArr, F.eval o.F σ (resh22 a b c d) = some (.arr r) ∧
      r.dims = [(1, 2), (1, 2)] ∧ r.get [1, 1] = .int .i4 a ∧ r.get [2, 1] = .int .i4 b ∧
      r.get [1, 2] = .int .i4 c ∧ r.get [2, 2] = .int .i4 d := ⟨_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  obtain ⟨q, hP, hqs, q00, q10, q01, q11⟩ : ∃ q : PArr, P.eval o.P τ (resh22P a b c d) = some (.arr q) ∧
      q.shape = [2, 2] ∧ q.get [0, 0] = .int a ∧ q.get [1, 0] = .int b ∧
      q.get [0, 1] = .int c ∧ q.get [1, 1] = .int d := ⟨_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  obtain rfl : v = .arr r := (Option.some.inj (hF.symm.trans hv)).symm
  refine ⟨.arr q, hP, RelV.arr ⟨by rw [hqs]; simp [FArr.extents, hdims], ?_⟩⟩
  intro idx hin
  obtain ⟨i, j, rfl, hi1, hi2, hj1, hj2⟩ := inBounds2 _ 1 1 2 2 hdims idx hin
  simp only [rebase, FArr.lbs, hdims, List.map_cons, List.map_nil, List.zipWith_cons_cons,
    List.zipWith_nil_right]
  interval_cases i <;> interval_cases j <;>
    simp only [h11, h21, h12, h22, q00, q10, q01, q11, show ((1:ℤ) - 1).toNat = 0 from rfl,
      show ((2:ℤ) - 1).toNat = 1 from rfl] <;> exact RelS.int

theorem bresh22_sound (o : Oracles) (Γ : F.Env) (a b c d : Bool) :
    SoundPair o Γ (bresh22 a b c d) (bresh22P a b c d) := by
  intro σ τ _ _ v hv
  obtain ⟨r, hF, hdims, h11, h21, h12, h22⟩ : ∃ r : FArr, F.eval o.F σ (bresh22 a b c d) = some (.arr r) ∧
      r.dims = [(1, 2), (1, 2)] ∧ r.get [1, 1] = .bool a ∧ r.get [2, 1] = .bool b ∧
      r.get [1, 2] = .bool c ∧ r.get [2, 2] = .bool d := ⟨_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  obtain ⟨q, hP, hqs, q00, q10, q01, q11⟩ : ∃ q : PArr, P.eval o.P τ (bresh22P a b c d) = some (.arr q) ∧
      q.shape = [2, 2] ∧ q.get [0, 0] = .bool a ∧ q.get [1, 0] = .bool b ∧
      q.get [0, 1] = .bool c ∧ q.get [1, 1] = .bool d := ⟨_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  obtain rfl : v = .arr r := (Option.some.inj (hF.symm.trans hv)).symm
  refine ⟨.arr q, hP, RelV.arr ⟨by rw [hqs]; simp [FArr.extents, hdims], ?_⟩⟩
  intro idx hin
  obtain ⟨i, j, rfl, hi1, hi2, hj1, hj2⟩ := inBounds2 _ 1 1 2 2 hdims idx hin
  simp only [rebase, FArr.lbs, hdims, List.map_cons, List.map_nil, List.zipWith_cons_cons,
    List.zipWith_nil_right]
  interval_cases i <;> interval_cases j <;>
    simp only [h11, h21, h12, h22, q00, q10, q01, q11, show ((1:ℤ) - 1).toNat = 0 from rfl,
      show ((2:ℤ) - 1).toNat = 1 from rfl] <;> exact RelS.bool

/-! ### FNP-R0104: PACK

`pack(a, m)` ↦ `(a)[(m)]` (expressions.py:1583-1587); `VECTOR=` is dropped.
Fortran takes the selected elements in *array element order* (column-major,
F2018 16.9.145); NumPy's boolean indexing enumerates them in row-major order,
so the two agree only on rank 1 (FNP-D0041, `probes/packorder`). -/

def FNP_R0104_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a m : F.Expr) (a' m' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ m m' →
    SoundPair o Γ (.call (.pack false) (.cons a (.cons m .nil)) none)
      (.maskIndex (.paren a') (.paren m'))

def FNP_R0104_vector_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a m vec : F.Expr) (a' m' vec' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ m m' → SoundPair o Γ vec vec' →
    SoundPair o Γ (.call (.pack true) (.cons a (.cons m (.cons vec .nil))) none)
      (.maskIndex (.paren a') (.paren m'))

/-- FNP-D0018: `pack([7,9], [.true., .false.], vector=[0,0,0,0])` has the
VECTOR's size (4) in Fortran; the emitted `(a)[(m)]` has the number of selected
elements (1).  The shapes already disagree. -/
theorem FNP_R0104_refuted_vector : ¬ FNP_R0104_vector_obligation := by
  intro h
  have hs := h noPrim emptyEnv (ictor2 7 9) (bctor2 true false) (ictor4 0 0 0 0)
    (ictor2P 7 9) (bctor2P true false) (ictor4P 0 0 0 0)
    (ictor2_sound _ _ _ _) (bctor2_sound _ _ _ _) (ictor4_sound _ _ _ _ _ _)
  obtain ⟨r, hF, hdims⟩ : ∃ r : FArr,
      F.eval noPrim.F emptyF (.call (.pack true)
          (.cons (ictor2 7 9) (.cons (bctor2 true false) (.cons (ictor4 0 0 0 0) .nil))) none) = some (.arr r) ∧
      r.dims = [(1, 4)] := ⟨_, rfl, rfl⟩
  obtain ⟨w, hw, hrel⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  obtain ⟨q, hP, hqs⟩ : ∃ q : PArr,
      P.eval noPrim.P emptyP (.maskIndex (.paren (ictor2P 7 9)) (.paren (bctor2P true false))) = some (.arr q) ∧
      q.shape = [1] := ⟨_, rfl, rfl⟩
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have := hA.shape
    rw [hqs] at this
    simp [FArr.extents, hdims] at this

/-- FNP-D0041: on a rank-2 array PACK takes the elements in array element
(column-major) order; NumPy's boolean indexing takes them row-major.  With
`a(1,1)=1, a(2,1)=2, a(1,2)=3, a(2,2)=4` and an all-true mask, Fortran gives
`[1,2,3,4]` and the translation `[1,3,2,4]`; they differ at the second element. -/
theorem FNP_R0104_refuted_rank2 : ¬ FNP_R0104_obligation := by
  intro h
  have hs := h noPrim emptyEnv (resh22 1 2 3 4) (bresh22 true true true true)
    (resh22P 1 2 3 4) (bresh22P true true true true)
    (resh22_sound _ _ _ _ _ _) (bresh22_sound _ _ _ _ _ _)
  obtain ⟨r, hF, hdims, hget⟩ : ∃ r : FArr,
      F.eval noPrim.F emptyF (.call (.pack false)
          (.cons (resh22 1 2 3 4) (.cons (bresh22 true true true true) .nil)) none) = some (.arr r) ∧
      r.dims = [(1, 4)] ∧ r.get [2] = .int .i4 2 := ⟨_, rfl, rfl, rfl⟩
  obtain ⟨w, hw, hrel⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  obtain ⟨q, hP, hqget⟩ : ∃ q : PArr,
      P.eval noPrim.P emptyP
          (.maskIndex (.paren (resh22P 1 2 3 4)) (.paren (bresh22P true true true true))) = some (.arr q) ∧
      q.get [1] = .int 3 := ⟨_, rfl, rfl⟩
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have hb : r.inBounds [2] := by simp [FArr.inBounds, FArr.inBoundsB, hdims]
    have hre := hA.elems [2] hb
    rw [hget] at hre
    have he : rebase r.lbs [2] = [1] := by simp [rebase, FArr.lbs, hdims]
    rw [he, hqget] at hre
    exact absurd (PVal.int.inj (RelS.int_eq hre)) (by decide)


/-! ### FNP-R0107: MAXLOC / MINLOC

`maxloc(a[, mask])` ↦ `(np.array(np.unravel_index(np.argmax(a), np.shape(a))) + 1)`
(expressions.py:1609-1617); `MASK=` is dropped. -/

def FNP_R0107_mask_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a m : F.Expr) (a' m' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ m m' →
    SoundPair o Γ (.call (.maxloc true) (.cons a (.cons m .nil)) none) (Tr.fold1 .np_argmax1 a')

/-- FNP-D0020: with an all-false `MASK=` nothing competes, so `MAXLOC` is `0`
(F2018 16.9.126); the emitted `np.argmax` ignores the mask and its result is a
position, hence at least `1`.  The counterexample needs no comparison to be
decided: the two answers differ by sign alone. -/
theorem FNP_R0107_refuted_mask : ¬ FNP_R0107_mask_obligation := by
  intro h
  have hs := h noPrim emptyEnv (ictor2 1 5) (bctor2 false false) (ictor2P 1 5) (bctor2P false false)
    (ictor2_sound _ _ _ _) (bctor2_sound _ _ _ _)
  obtain ⟨r, hF, hdims, hget⟩ : ∃ r : FArr,
      F.eval noPrim.F emptyF (.call (.maxloc true) (.cons (ictor2 1 5) (.cons (bctor2 false false) .nil)) none)
        = some (.arr r) ∧ r.dims = [(1, 1)] ∧ r.get [1] = .int .i4 0 := ⟨_, rfl, rfl, rfl⟩
  obtain ⟨w, hw, hrel⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  obtain ⟨m, hP⟩ : ∃ m : ℕ, P.eval noPrim.P emptyP (Tr.fold1 .np_argmax1 (ictor2P 1 5))
      = some (.arr (P.listArr [.int ((m : ℤ) + 1)])) := ⟨_, rfl⟩
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have hb : r.inBounds [1] := by simp [FArr.inBounds, FArr.inBoundsB, hdims]
    have hre := hA.elems [1] hb
    rw [hget] at hre
    have he : rebase r.lbs [1] = [0] := by simp [rebase, FArr.lbs, hdims]
    rw [he, show (P.listArr [PVal.int ((m : ℤ) + 1)]).get [0] = .int ((m : ℤ) + 1) from rfl] at hre
    have := PVal.int.inj (RelS.int_eq hre)
    omega

/-! #### PACK on rank-1 arrays is sound

On rank 1 the two enumeration orders coincide, so the rule holds. -/

theorem filterMap_congr' {α β : Type} {f g : α → Option β} :
    ∀ l : List α, (∀ a ∈ l, f a = g a) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | x :: xs, h => by
      rw [List.filterMap_cons, List.filterMap_cons, h x (List.mem_cons_self ..),
        filterMap_congr' xs (fun a ha => h a (List.mem_cons_of_mem _ ha))]

theorem filterMap_ite {α β : Type} (P : α → Bool) (f : α → β) :
    ∀ l : List α, l.filterMap (fun k => if P k then some (f k) else none) = (l.filter P).map f
  | [] => rfl
  | x :: xs => by
      by_cases h : P x <;> simp [List.filterMap_cons, List.filter_cons, h, filterMap_ite P f xs]

theorem filter_map_comm {α β : Type} (p : β → Bool) (f : α → β) :
    ∀ l : List α, (l.map f).filter p = (l.filter (fun a => p (f a))).map f
  | [] => rfl
  | x :: xs => by
      by_cases h : p (f x) <;> simp [List.map_cons, List.filter_cons, h, filter_map_comm p f xs]

theorem rowMajor_rank1T (n : ℕ) : P.rowMajor [n] = (List.range n).map (fun (k : ℕ) => [k]) := by
  unfold P.rowMajor
  simp only [List.foldr_cons, List.foldr_nil, Nat.mul_one]
  apply List.map_congr_left
  intro k hk
  rw [List.mem_range] at hk
  simp [P.cCoords, CM.coords, Nat.mod_eq_of_lt hk]

/-- The 0-based positions a rank-1 Python mask selects (`a[m]`, row-major). -/
def maskBitsP (mp : PArr) (k : ℕ) : Bool :=
  match mp.get [k] with | .bool true => true | _ => false

/-- The 0-based positions a rank-1 Fortran mask selects (array element order). -/
def maskBitsF (mv : FArr) (lm : ℤ) (k : ℕ) : Bool :=
  match mv.get [lm + (k : ℤ)] with | .bool true => true | _ => false

theorem maskCoords_rank1 (mp : PArr) (n : ℕ) (hs : mp.shape = [n]) :
    P.maskCoords mp = ((List.range n).filter (maskBitsP mp)).map (fun (k : ℕ) => [k]) := by
  unfold P.maskCoords
  rw [hs, rowMajor_rank1T, filter_map_comm]
  rfl

theorem svalBoolMatch (x : SVal) (c : List ℤ) :
    (match x with | .bool true => some c | _ => none)
      = if (match x with | .bool true => true | _ => false) = true then some c else none := by
  cases x with
  | bool b => cases b <;> rfl
  | int _ _ => rfl
  | real _ _ => rfl
  | str _ => rfl

theorem maskedCoords_rank1 (av mv : FArr) (lb lm : ℤ) (n : ℕ)
    (hda : av.dims = [(lb, n)]) (hdm : mv.dims = [(lm, n)]) :
    F.maskedCoords av mv =
      some (((List.range n).filter (maskBitsF mv lm)).map (fun (k : ℕ) => [lb + (k : ℤ)])) := by
  have hext : av.extents = mv.extents := by simp [FArr.extents, hda, hdm]
  have hsz : av.size = n := FArr_size_rank1 av lb n hda
  unfold F.maskedCoords
  rw [if_neg (by simp [hext]), hsz]
  congr 1
  rw [← filterMap_ite (maskBitsF mv lm) (fun (k : ℕ) => [lb + (k : ℤ)])]
  apply filterMap_congr'
  intro k hk
  rw [List.mem_range] at hk
  have hlen : k < (List.range n).length := by simpa using hk
  have ha : (F.allCoords av).getD k [] = [lb + (k : ℤ)] := by
    rw [allCoords_rank1 av lb n hda,
      getD_map_lt (fun (j : ℕ) => [lb + (j : ℤ)]) (List.range n) k hlen []]
    simp
  have hm : (F.allCoords mv).getD k [] = [lm + (k : ℤ)] := by
    rw [allCoords_rank1 mv lm n hdm,
      getD_map_lt (fun (j : ℕ) => [lm + (j : ℤ)]) (List.range n) k hlen []]
    simp
  simp only [ha, hm, maskBitsF]
  exact svalBoolMatch _ _

/-- Related rank-1 masks select the same positions. -/
theorem maskBits_agree {mv : FArr} {mp : PArr} (h : RelA mv mp) (lm : ℤ) (n : ℕ)
    (hdm : mv.dims = [(lm, n)]) (k : ℕ) (hk : k < n) : maskBitsF mv lm k = maskBitsP mp k := by
  have hrel := RelA_get1 h lm n hdm (lm + (k : ℤ)) ⟨by omega, by omega⟩
  have he : (lm + (k : ℤ) - lm).toNat = k := by omega
  rw [he] at hrel
  unfold maskBitsF maskBitsP
  cases hg : mv.get [lm + (k : ℤ)] with
  | bool b => rw [hg] at hrel; rw [RelS.bool_eq hrel]; cases b <;> rfl
  | int k' vv => rw [hg] at hrel; rw [RelS.int_eq hrel]
  | real k' r =>
      rw [hg] at hrel
      rcases RelS.real_cases hrel with ⟨m', hm', _⟩ | hm' <;> rw [hm']
  | str s => rw [hg] at hrel; rw [RelS.str_eq hrel]

/-- The rule restricted to rank-1 arguments (the case the two enumeration
orders agree on). -/
def FNP_R0104_rank1_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a m : F.Expr) (a' m' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ m m' → Rank1Arg o Γ a → Rank1Arg o Γ m →
    SoundPair o Γ (.call (.pack false) (.cons a (.cons m .nil)) none)
      (.maskIndex (.paren a') (.paren m'))

theorem FNP_R0104_sound_rank1 : FNP_R0104_rank1_obligation := by
  intro o Γ a m a' m' ha hm hra hrm σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | s x =>
      rw [show F.intrinsic o.F (.pack false) [.s x, vb] none = none from by cases vb <;> rfl] at hint
      cases hint
  | arr av =>
    cases vb with
    | s y => rw [show F.intrinsic o.F (.pack false) [.arr av, .s y] none = none from rfl] at hint; cases hint
    | arr mv =>
      obtain ⟨⟨lb, n, hda, hn⟩, hwta⟩ := hra σ τ hR hC av hva
      obtain ⟨⟨lm, nm, hdm, hnm⟩, hwtm⟩ := hrm σ τ hR hC mv hvb
      rw [show F.intrinsic o.F (.pack false) [.arr av, .arr mv] none = F.packF av mv none from rfl] at hint
      -- the shapes must conform, or the source is undefined
      by_cases hext : av.extents = mv.extents
      · have hnn : nm = n := by
          simp only [FArr.extents, hda, hdm, List.map_cons, List.map_nil, List.cons.injEq] at hext
          omega
        subst hnn
        rw [F.packF, maskedCoords_rank1 av mv lb lm nm hda hdm] at hint
        simp only [Option.bind_some, Option.some.injEq] at hint
        subst hint
        -- the Python side
        obtain ⟨wa, hwa, hrwa⟩ := ha σ τ hR hC _ hva
        obtain ⟨wm, hwm, hrwm⟩ := hm σ τ hR hC _ hvb
        cases hrwa with
        | arr hAa =>
          rename_i p
          cases hrwm with
          | arr hAm =>
            rename_i mp
            have hps : p.shape = [nm] := RelA_dims1 hAa lb nm hda
            have hms : mp.shape = [nm] := RelA_dims1 hAm lm nm hdm
            -- the two masks select the same positions
            have hfilter : (List.range nm).filter (maskBitsF mv lm)
                = (List.range nm).filter (maskBitsP mp) :=
              List.filter_congr (fun x hx => maskBits_agree hAm lm nm hdm x (List.mem_range.mp hx))
            have hmc : P.maskCoords mp
                = ((List.range nm).filter (maskBitsF mv lm)).map (fun (k : ℕ) => [k]) := by
              rw [maskCoords_rank1 mp nm hms, ← hfilter]
            have hSn : ∀ s ∈ (List.range nm).filter (maskBitsF mv lm), s < nm := by
              intro s hs
              rw [List.mem_filter, List.mem_range] at hs
              exact hs.1
            refine ⟨.arr ⟨[(P.maskCoords mp).length],
              fun idx => p.get ((P.maskCoords mp).getD (idx.headD 0) [])⟩, ?_, RelV.arr ?_⟩
            · rw [P_eval_maskIndex, P_eval_paren, P_eval_paren, hwa, hwm]
              simp only [Option.bind_some, P.maskSelect]
              rw [if_neg (by rw [hps, hms]; simp)]
            · refine ⟨?_, ?_⟩
              · simp [FArr.extents, F.listArr, hmc]
              · intro idx hin
                obtain ⟨j, rfl, hj1, hj2⟩ := inBounds1 _ 1 _ rfl idx hin
                have hlen : (j - 1).toNat < ((List.range nm).filter (maskBitsF mv lm)).length := by
                  simp only [F.listArr, List.length_map] at hj2 ⊢
                  omega
                have hmem : ((List.range nm).filter (maskBitsF mv lm))[(j - 1).toNat] ∈
                    (List.range nm).filter (maskBitsF mv lm) := List.getElem_mem hlen
                have hslt := hSn _ hmem
                simp only [rebase, F.listArr, FArr.lbs, List.map_cons, List.map_nil,
                  List.zipWith_cons_cons, List.zipWith_nil_right, List.headD_cons, List.map_map]
                rw [getD_lt _ _ (by simpa using hlen) _, hmc,
                  getD_map_lt (fun (k : ℕ) => [k]) _ _ hlen []]
                simp only [List.getElem_map, Function.comp]
                have hg := RelA_get1 hAa lb nm hda
                  (lb + (((List.range nm).filter (maskBitsF mv lm))[(j - 1).toNat] : ℤ))
                  ⟨by omega, by omega⟩
                have he : (lb + (((List.range nm).filter (maskBitsF mv lm))[(j - 1).toNat] : ℤ) - lb).toNat
                    = ((List.range nm).filter (maskBitsF mv lm))[(j - 1).toNat] := by omega
                rw [he] at hg
                exact hg
      · simp [F.packF, F.maskedCoords, hext] at hint

/-! #### RESHAPE without keyword arguments, on an exact shape

With `PRODUCT(SHAPE) = SIZE(SOURCE)` (the case `np.reshape` also accepts) the
rule is sound: both sides lay the source's *storage sequence* (column-major on
both sides) into the new extents, and the offsets agree. -/

theorem mapM_congr_option {α γ : Type} {h₁ h₂ : α → Option γ} :
    ∀ l : List α, (∀ a ∈ l, h₁ a = h₂ a) → l.mapM h₁ = l.mapM h₂
  | [], _ => rfl
  | x :: xs, h => by
      rw [List.mapM_cons, List.mapM_cons, h x (List.mem_cons_self ..),
        mapM_congr_option xs (fun a ha => h a (List.mem_cons_of_mem _ ha))]

theorem map_range_getD {α : Type} (l : List α) (d : α) :
    (List.range l.length).map (fun k => l.getD k d) = l := by
  apply List.ext_getElem
  · simp
  · intro n h1 h2
    simp only [List.getElem_map, List.getElem_range, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem h2, Option.getD_some]

theorem elems_length (a : FArr) : (F.elems a).length = a.size := by
  simp [F.elems, F.allCoords]

theorem elems_getD (a : FArr) (k : ℕ) (hk : k < a.size) (d : SVal) :
    (F.elems a).getD k d = a.get (F.coordAt a k) := by
  have hlen : k < (List.range a.size).length := by simpa using hk
  unfold F.elems F.allCoords
  rw [List.map_map, getD_map_lt _ (List.range a.size) k hlen d]
  simp [F.coordAt, Function.comp]

theorem rebase_ones : ∀ (ns : List ℕ) (idx : List ℤ), idx.length = ns.length →
    rebase (ns.map fun _ => (1 : ℤ)) idx = idx.map (fun i => (i - 1).toNat)
  | [], [], _ => rfl
  | _ :: _, [], h => by simp at h
  | [], _ :: _, h => by simp at h
  | n :: ns, i :: idx, h => by
      have ih := rebase_ones ns idx (by simpa using h)
      simp only [List.map_cons, rebase, List.zipWith_cons_cons] at ih ⊢
      rw [ih]

theorem inBoundsB_unit : ∀ (ns : List ℕ) (idx : List ℤ),
    FArr.inBoundsB (ns.map fun n => ((1 : ℤ), n)) idx = true →
    PArr.inBoundsB ns (idx.map fun i => (i - 1).toNat) = true
  | [], [], _ => rfl
  | _ :: _, [], h => by simp [FArr.inBoundsB] at h
  | [], _ :: _, h => by simp [FArr.inBoundsB] at h
  | n :: ns, i :: idx, h => by
      simp only [List.map_cons, FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
      simp only [List.map_cons, PArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨by omega, inBoundsB_unit ns idx h.2⟩

theorem mapM_map_option {α β γ : Type} (f : α → β) (g : β → Option γ) :
    ∀ l : List α, (l.map f).mapM g = l.mapM (fun a => g (f a))
  | [] => rfl
  | x :: xs => by rw [List.map_cons, List.mapM_cons, List.mapM_cons, mapM_map_option f g xs]

theorem sval_int_of_ty {v : SVal} {k : IKind} (h : v.ty = .int k) : ∃ n, v = .int k n := by
  cases v with
  | int k' n => simp only [SVal.ty, FTy.int.injEq] at h; exact ⟨n, by rw [h]⟩
  | real _ _ => simp [SVal.ty] at h
  | bool _ => simp [SVal.ty] at h
  | str _ => simp [SVal.ty] at h

/-- Reading a shape from a related pair of rank-1 INTEGER arrays gives the same
extents on both sides. -/
theorem shape_agree (sv : FArr) (sp : PArr) (lbs : ℤ) (l : List ℕ)
    (hrel : ∀ k ∈ l, RelS (sv.get [lbs + (k : ℤ)]) (sp.get [k]))
    (hint : ∀ k ∈ l, ∃ k' n, sv.get [lbs + (k : ℤ)] = .int k' n) :
    (l.map (fun (k : ℕ) => sv.get [lbs + (k : ℤ)])).mapM
        (fun v => match v with | .int _ n => if 0 ≤ n then some n.toNat else none | _ => none)
      = l.mapM (fun (k : ℕ) => match sp.get [k] with
          | .int n => if 0 ≤ n then some n.toNat else none | _ => none) := by
  rw [mapM_map_option]
  apply mapM_congr_option
  intro k hk
  obtain ⟨k', n, hg⟩ := hint k hk
  have hx := hrel k hk
  rw [hg] at hx
  rw [hg, RelS.int_eq hx]

theorem inBoundsB_length : ∀ (dims : List (ℤ × ℕ)) (idx : List ℤ),
    FArr.inBoundsB dims idx = true → idx.length = dims.length
  | [], [], _ => rfl
  | _ :: _, [], h => by simp [FArr.inBoundsB] at h
  | [], _ :: _, h => by simp [FArr.inBoundsB] at h
  | _ :: ds, _ :: is, h => by
      simp only [FArr.inBoundsB, Bool.and_eq_true] at h
      simp only [List.length_cons, inBoundsB_length ds is h.2]

/-- The shape argument is a rank-1 INTEGER array whose product is the source's
size -- the case NumPy's `reshape` accepts too. -/
def ReshapeExact (o : Oracles) (Γ : F.Env) (a shp : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av sv, F.eval o.F σ a = some (.arr av) →
    F.eval o.F σ shp = some (.arr sv) → (∃ k, sv.elemTy = .int k) ∧
      ∀ ns, F.shapeOf sv = some ns → CM.prod ns = av.size

def FNP_R0102_exact_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a shp : F.Expr) (a' shp' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ shp shp' → Rank1Arg o Γ shp → ReshapeExact o Γ a shp →
    SoundPair o Γ (.call (.reshape false false) (.cons a (.cons shp .nil)) none)
      (Tr.call2 .np_reshapeF a' shp')

set_option maxHeartbeats 1000000 in
theorem FNP_R0102_sound_exact : FNP_R0102_exact_obligation := by
  intro o Γ a shp a' shp' ha hshp hr1 hex σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | s x =>
      rw [show F.intrinsic o.F (.reshape false false) [.s x, vb] none = none from by cases vb <;> rfl] at hint
      cases hint
  | arr av =>
    cases vb with
    | s y =>
        rw [show F.intrinsic o.F (.reshape false false) [.arr av, .s y] none = none from rfl] at hint
        cases hint
    | arr sv =>
      obtain ⟨⟨lbs, r, hds, hr0⟩, hwts⟩ := hr1 σ τ hR hC sv hvb
      obtain ⟨⟨ke, hke⟩, hprod⟩ := hex σ τ hR hC av sv hva hvb
      rw [show F.intrinsic o.F (.reshape false false) [.arr av, .arr sv] none
          = F.reshapeF av sv none none from rfl] at hint
      cases hsh : F.shapeOf sv with
      | none => rw [F.reshapeF, hsh] at hint; simp at hint
      | some ns =>
        have htot : CM.prod ns = av.size := hprod ns hsh
        have hlen : (F.elems av).length = av.size := elems_length av
        have hcond : List.foldr (fun x1 x2 => x1 * x2) 1 ns ≤ (F.elems av).length := by
          rw [hlen]; exact le_of_eq htot
        simp only [F.reshapeF, hsh, Option.bind_some,
          show F.orderOf ns.length none = some ((List.range ns.length).map (· + 1)) from rfl,
          if_pos hcond, Option.map_some, Option.some.injEq] at hint
        subst hint
        -- the Python side
        obtain ⟨wa, hwa, hrwa⟩ := ha σ τ hR hC _ hva
        obtain ⟨ws, hws, hrws⟩ := hshp σ τ hR hC _ hvb
        cases hrwa with
        | arr hAa =>
          rename_i p
          cases hrws with
          | arr hAs =>
            rename_i sp
            have hps : sp.shape = [r] := RelA_dims1 hAs lbs r hds
            have hsvsize : sv.size = r := FArr_size_rank1 sv lbs r hds
            -- the shape reads the same on both sides
            have helems : F.elems sv = (List.range r).map (fun (k : ℕ) => sv.get [lbs + (k : ℤ)]) := by
              unfold F.elems
              rw [allCoords_rank1 sv lbs r hds, List.map_map]
              rfl
            have hsa : P.shapeArg sp = some ns := by
              have h1 : F.shapeOf sv = (F.elems sv).mapM
                  (fun v => match v with | .int _ n => if 0 ≤ n then some n.toNat else none | _ => none) := by
                unfold F.shapeOf
                rw [if_neg (by simp [FArr.rank, hds])]
                rfl
              rw [h1, helems, shape_agree sv sp lbs (List.range r)
                (fun k hk => by
                  rw [List.mem_range] at hk
                  have := RelA_get1 hAs lbs r hds (lbs + (k : ℤ)) ⟨by omega, by omega⟩
                  have he : (lbs + (k : ℤ) - lbs).toNat = k := by omega
                  rwa [he] at this)
                (fun k hk => by
                  rw [List.mem_range] at hk
                  have hb : sv.inBounds [lbs + (k : ℤ)] :=
                    inBounds_rank1 sv lbs r hds _ ⟨by omega, by omega⟩
                  obtain ⟨n, hn⟩ := sval_int_of_ty (by rw [hwts _ hb, hke])
                  exact ⟨ke, n, hn⟩)] at hsh
              rw [P.shapeArg, hps]
              exact hsh
            have hpsize : p.size = av.size := by
              simp [PArr.size, FArr.size, hAa.shape]
            refine ⟨.arr ⟨ns, fun idx => (P.ravelF p).getD (CM.offset ns idx) .none⟩, ?_, RelV.arr ?_⟩
            · rw [P_eval_call2, hwa, hws]
              simp only [Option.bind_some]
              rw [show P.helper o.P .np_reshapeF [.arr p, .arr sp]
                  = (P.shapeArg sp).bind (fun ns => if ns.foldr (· * ·) 1 ≠ p.size then none
                      else some (.arr ⟨ns, fun idx => (P.ravelF p).getD (CM.offset ns idx) .none⟩)) from rfl,
                hsa, Option.bind_some, if_neg (by rw [hpsize, ← htot]; simp [CM.prod])]
            · refine ⟨by simp [FArr.extents, List.map_map, Function.comp], ?_⟩
              intro idx hin
              have hidxlen : idx.length = ns.length := by
                have := inBoundsB_length _ _ hin
                simpa using this
              have hin0 : PArr.inBoundsB ns (idx.map fun i => (i - 1).toNat) = true :=
                inBoundsB_unit ns idx hin
              have hoff : CM.offset ns (idx.map fun i => (i - 1).toNat) < CM.prod ns :=
                CM.offset_lt_prod ns _ hin0
              have hidx0len : (idx.map fun i => (i - 1).toNat).length = ns.length := by
                simpa using hidxlen
              -- the rebased coordinates are the 0-based ones
              simp only [FArr.lbs, List.map_map]
              rw [show List.map (Prod.fst ∘ fun (n : ℕ) => ((1 : ℤ), n)) ns = ns.map (fun _ => (1 : ℤ)) from
            List.map_congr_left (fun x _ => rfl), rebase_ones ns idx hidxlen]
              -- the permuted extents and subscripts are the identity ones
              have hpermE : List.map ((fun (a : ℕ) => ns.getD (a - 1) 0) ∘ fun (x : ℕ) => x + 1)
                  (List.range ns.length) = ns := by
                rw [show ((fun (a : ℕ) => ns.getD (a - 1) 0) ∘ fun (x : ℕ) => x + 1)
                    = (fun k => ns.getD k 0) from rfl]
                exact map_range_getD ns 0
              have hpermI : List.map
                  ((fun (a : ℕ) => (idx.map fun i => (i - 1).toNat).getD (a - 1) 0) ∘ fun (x : ℕ) => x + 1)
                  (List.range ns.length) = idx.map fun i => (i - 1).toNat := by
                rw [show ((fun (a : ℕ) => (idx.map fun i => (i - 1).toNat).getD (a - 1) 0) ∘ fun (x : ℕ) => x + 1)
                    = (fun k => (idx.map fun i => (i - 1).toNat).getD k 0) from rfl, ← hidx0len]
                exact map_range_getD _ 0
              rw [hpermE, hpermI]
              rw [show List.take (List.foldr (fun (x1 x2 : ℕ) => x1 * x2) 1 ns) (F.elems av) = F.elems av from by
                rw [show List.foldr (fun (x1 x2 : ℕ) => x1 * x2) 1 ns = (F.elems av).length from by
                  rw [hlen]; exact htot]
                exact List.take_length]
              rw [elems_getD av _ (by omega) _, P.ravelF_getD p _ (by omega) _]
              exact RelA.storage hAa _ (by omega)

end Recast.Rules
