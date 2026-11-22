import RecastProofs.Rules.Obligations
import RecastProofs.Rules.Reductions

/-!
# Rules FNP-R0103 (DOT_PRODUCT), FNP-R0108 (MAXVAL/MINVAL) and FNP-R0090 on the
signed range

* FNP-R0103: refuted for INTEGER operands (FNP-D0040: `_f_vdot` seeds its
  accumulator with the float `0.0`); proved for REAL rank-1 operands.
* FNP-R0108: MAXVAL/MINVAL on non-empty rank-1 arrays, through a generic fold
  lemma over related element sequences with a value-based accumulator relation.
* FNP-R0090: `IAND`/`IOR`/`IEOR` agree with Python's unbounded two's-complement
  `& | ^` for every operand in the kind's *signed* range, not only the
  non-negative half proved in `Intrinsics2.lean`.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### A generic fold lemma over related element sequences -/

/-- Sequential folds on both sides stay related when every step does.  The
element type is abstract so that the same lemma serves single sequences
(`MAXVAL`) and zipped pairs (`DOT_PRODUCT`). -/
theorem fold_rel_gen {α β : Type} (RelE : α → β → Prop) (fS : SVal → α → Option SVal)
    (gS : PVal → β → Option PVal)
    (hrel : ∀ (x : SVal) (x' : PVal) (y : α) (y' : β) (c : SVal), RelS x x' → RelE y y' →
      fS x y = some c → ∃ c', gS x' y' = some c' ∧ RelS c c')
    (fF : ℕ → α) (fP : ℕ → β) :
    ∀ (l : List ℕ), (∀ k ∈ l, RelE (fF k) (fP k)) → ∀ (z : SVal) (z' : PVal), RelS z z' → ∀ r,
      l.foldl (fun acc k => acc.bind fun x => fS x (fF k)) (some z) = some r →
      ∃ r', l.foldl (fun acc k => acc.bind fun s => gS s (fP k)) (some z') = some r' ∧ RelS r r' := by
  intro l
  induction l with
  | nil => intro _ z z' hz r hr; simp only [List.foldl_nil, Option.some.injEq] at hr; subst hr; exact ⟨z', rfl, hz⟩
  | cons k ks ih =>
    intro hrelk z z' hz r hr
    simp only [List.foldl_cons, Option.bind_some] at hr
    cases h1 : fS z (fF k) with
    | none => rw [h1] at hr; rw [foldl_bind_none (fun k x => fS x (fF k))] at hr; cases hr
    | some z₁ =>
      rw [h1] at hr
      obtain ⟨z₁', hz₁', hrel₁⟩ := hrel z z' (fF k) (fP k) z₁ hz (hrelk k (List.mem_cons_self ..)) h1
      obtain ⟨r', hr', hrr'⟩ := ih (fun k hk => hrelk k (List.mem_cons_of_mem _ hk)) z₁ z₁' hrel₁ r hr
      refine ⟨r', ?_, hrr'⟩
      simp only [List.foldl_cons, Option.bind_some, hz₁']
      exact hr'

/-- An element of a homogeneous REAL array is a REAL value. -/
theorem SVal_real_of_ty (v : SVal) (k : RKind) (h : v.ty = .real k) : ∃ x, v = .real k x := by
  cases v <;> simp [SVal.ty] at h
  subst h; exact ⟨_, rfl⟩

theorem F_arith_mul_real (k₁ k₂ : RKind) (x y : ℝ) :
    F.arith .mul (.real k₁ x) (.real k₂ y) = some (.real (RKind.join k₁ k₂) (x * y)) := rfl

/-! ### FNP-R0103: DOT_PRODUCT -/

/-- The product array of two related rank-1 arrays of extent `n > 0`, when the
first pair of elements has a defined product. -/
theorem F_lift2_arr_rank1 (f : SVal → SVal → Option SVal) (a b : FArr) (la lb : ℤ) (n : ℕ)
    (hda : a.dims = [(la, n)]) (hdb : b.dims = [(lb, n)]) (hn : 0 < n) (sample : SVal)
    (hs : f (a.get [la]) (b.get [lb]) = some sample) :
    ∃ p : FArr, F.lift2 f (.arr a) (.arr b) = some (.arr p) ∧ p.elemTy = sample.ty ∧ p.dims = [(1, n)] ∧
      ∀ j : ℤ, p.get [j] = (f (a.get [la + j - 1]) (b.get [lb + j - 1])).getD sample := by
  cases hl : F.lift2 f (.arr a) (.arr b) with
  | none => exfalso; simp [F.lift2, FArr.extents, FArr.size, FArr.lbs, hda, hdb, hn.ne', hs] at hl
  | some v =>
    cases v with
    | s _ => exfalso; simp [F.lift2, FArr.extents, FArr.size, FArr.lbs, hda, hdb, hn.ne', hs] at hl
    | arr p =>
      simp [F.lift2, FArr.extents, FArr.size, FArr.lbs, hda, hdb, hn.ne', hs] at hl
      subst hl
      refine ⟨_, rfl, rfl, by simp, fun j => by simp⟩

/-- INTEGER rank-1 arrays for the counterexample. -/
def oneFI : FArr := { elemTy := .int .i4, dims := [(1, 1)], get := fun _ => .int .i4 1 }
def onePI : PArr := { shape := [1], get := fun _ => .int 1 }

theorem oneRelI : RelA oneFI onePI := ⟨rfl, fun _ _ => RelS.int⟩

def declI : F.Decl := ⟨.int .i4, [.intLit .i4 1], 0⟩
def envI : F.Env := fun x => if x = "a" then some declI else none
def stateI : F.State := fun x => if x = "a" then some (.arr oneFI) else none
def pstateI : P.State := fun x => if x = "a" then some (.arr onePI) else none

theorem relStateI : RelState envI stateI pstateI := by
  intro x d hx
  simp only [envI] at hx
  split_ifs at hx with hxa
  subst hxa
  exact ⟨.arr oneFI, .arr onePI, by simp [stateI], by simp [pstateI], RelV.arr oneRelI⟩

theorem conformsI : Conforms noPrim.F envI stateI := by
  intro x d hx
  simp only [envI] at hx
  split_ifs at hx with hxa
  subst hxa
  cases hx
  refine ⟨fun h => by simp [declI] at h, fun _ => ⟨oneFI, by simp [stateI], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
  intro i hi
  simp only [declI] at hi
  match i, hi with
  | 0, _ => simp [declI, F_eval_intLit, oneFI, FArr.lbs]

theorem soundPair_varI : SoundPair noPrim envI (.var "a") (.name "a") :=
  FNP_R0010_sound noPrim envI "a" declI (by simp [envI])

/-- FNP-D0040: `dot_product(a, a)` with `integer :: a(1) = 1` is the INTEGER `1`
in Fortran; `_f_vdot` starts from `0.0` and answers the float `1.0`. -/
theorem FNP_R0103_refuted : ¬ FNP_R0103_obligation := by
  intro h
  have := h noPrim envI (.var "a") (.var "a") (.name "a") (.name "a") soundPair_varI soundPair_varI
    stateI pstateI relStateI conformsI (.int .i4 1) (by
      rw [F_eval_call2]
      simp only [F.eval, stateI, if_true, Option.bind_some]
      rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call2] at hw
  simp only [P.eval, pstateI, if_true, Option.bind_some] at hw
  have e : ∃ r : ℝ, P.helper noPrim.P .f_vdot [.arr onePI, .arr onePI] = some (.s (.float r)) := ⟨_, rfl⟩
  obtain ⟨r, hr⟩ := e
  rw [hr] at hw
  cases hw
  exact RelS.int_ne_float hrel

/-- Premise: every array value the expression takes has REAL elements. -/
def RealElems (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av, F.eval o.F σ e = some (.arr av) → ∃ k, av.elemTy = .real k

/-- The dot-product step relation: the Fortran element is the (defined) product
of two values related to the Python pair. -/
def DotRelE (c : SVal) (uv : PVal × PVal) : Prop :=
  ∃ a b : SVal, RelS a uv.1 ∧ RelS b uv.2 ∧ F.arith .mul a b = some c

theorem dot_step_rel (x : SVal) (x' : PVal) (c : SVal) (uv : PVal × PVal) (r : SVal)
    (hx : RelS x x') (hc : DotRelE c uv) (hr : F.arith .add x c = some r) :
    ∃ r', ((P.arith .mul uv.1 uv.2).bind fun m => P.arith .add x' m) = some r' ∧ RelS r r' := by
  obtain ⟨a, b, hau, hbv, hab⟩ := hc
  obtain ⟨m, hm, hcm⟩ := arith_addsubmul_rel (Or.inr (Or.inr rfl)) hau hbv hab
  obtain ⟨r', hr', hrr⟩ := arith_addsubmul_rel (Or.inl rfl) hx hcm hr
  have hm' : P.arith .mul uv.1 uv.2 = some m := hm
  have hr'' : P.arith .add x' m = some r' := hr'
  exact ⟨r', by rw [hm', Option.bind_some, hr''], hrr⟩

/-- FNP-R0103 for REAL rank-1 operands: `dot_product(a, b)` ↦ `_f_vdot(a', b')`. -/
theorem FNP_R0103_sound_real (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundPair o Γ a a') (hb : SoundPair o Γ b b')
    (hra : Rank1Arg o Γ a) (hrb : Rank1Arg o Γ b) (hea : RealElems o Γ a) (heb : RealElems o Γ b) :
    SoundScalar o Γ (.call .dot_product (.cons a (.cons b .nil)) none) (Tr.call2 .f_vdot a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | s x =>
    cases vb with
    | s y => rw [show F.intrinsic o.F .dot_product [.s x, .s y] none = none from rfl] at hint; cases hint
    | arr bv => rw [show F.intrinsic o.F .dot_product [.s x, .arr bv] none = none from rfl] at hint; cases hint
  | arr av =>
    cases vb with
    | s y => rw [show F.intrinsic o.F .dot_product [.arr av, .s y] none = none from rfl] at hint; cases hint
    | arr bv =>
      obtain ⟨⟨la, n, hda, hn⟩, hwta⟩ := hra σ τ hR hC av hva
      obtain ⟨⟨lb, m, hdb, _⟩, hwtb⟩ := hrb σ τ hR hC bv hvb
      obtain ⟨ka, hka⟩ := hea σ τ hR hC av hva
      obtain ⟨kb, hkb⟩ := heb σ τ hR hC bv hvb
      obtain ⟨wa, hwa, hrwa⟩ := ha σ τ hR hC _ hva
      obtain ⟨wb, hwb, hrwb⟩ := hb σ τ hR hC _ hvb
      cases hrwa with
      | arr hA =>
        rename_i pa
        cases hrwb with
        | arr hB =>
          rename_i pb
          have e : F.intrinsic o.F .dot_product [.arr av, .arr bv] none =
              (if av.rank ≠ 1 ∨ av.extents ≠ bv.extents then none
               else match F.lift2 (F.arith .mul) (.arr av) (.arr bv) with
                 | some (.arr p) => (F.zeroOf p.elemTy).bind fun z => (F.foldElems p (F.arith .add) z).map F.Val.s
                 | _ => none) := rfl
          rw [e] at hint
          by_cases hcond : av.rank ≠ 1 ∨ av.extents ≠ bv.extents
          · rw [if_pos hcond] at hint; cases hint
          · rw [if_neg hcond] at hint
            have hnm : n = m := by
              have := hcond
              simp [FArr.rank, FArr.extents, hda, hdb] at this
              exact this
            subst hnm
            -- the elements are reals
            have hreal_a : ∀ k : ℕ, k < n → ∃ x, av.get [la + k] = .real ka x := by
              intro k hk
              apply SVal_real_of_ty _ ka
              rw [← hka]
              exact hwta [la + k] (inBounds_rank1 av la n hda _ ⟨by omega, by omega⟩)
            have hreal_b : ∀ k : ℕ, k < n → ∃ y, bv.get [lb + k] = .real kb y := by
              intro k hk
              apply SVal_real_of_ty _ kb
              rw [← hkb]
              exact hwtb [lb + k] (inBounds_rank1 bv lb n hdb _ ⟨by omega, by omega⟩)
            obtain ⟨x0, hx0⟩ := hreal_a 0 hn
            obtain ⟨y0, hy0⟩ := hreal_b 0 hn
            simp only [Nat.cast_zero, add_zero] at hx0 hy0
            obtain ⟨p, hp, hpty, hpd, hpget⟩ := F_lift2_arr_rank1 (F.arith .mul) av bv la lb n hda hdb hn
              (.real (RKind.join ka kb) (x0 * y0)) (by rw [hx0, hy0]; rfl)
            rw [hp] at hint
            simp only at hint
            rw [hpty] at hint
            simp only [SVal.ty] at hint
            rw [show F.zeroOf (.real (RKind.join ka kb)) = some (.real (RKind.join ka kb) 0) from rfl,
              Option.bind_some] at hint
            simp only [Option.map_eq_some_iff] at hint
            obtain ⟨r, hfold, hcr⟩ := hint
            cases hcr
            -- the Python side
            have hsa := RelA_dims1 hA la n hda
            have hsb := RelA_dims1 hB lb n hdb
            rw [P_eval_call2, hwa, hwb, Option.bind_some, Option.bind_some]
            have eP : P.helper o.P .f_vdot [.arr pa, .arr pb] = (if pa.size ≠ pb.size then none else
                ((List.zip (P.ravelF pa) (P.ravelF pb)).foldl
                  (fun acc q => acc.bind fun s => (P.arith .mul q.1 q.2).bind fun m => P.arith .add s m)
                  (some (.float 0))).map P.Val.s) := rfl
            rw [eP, if_neg (by rw [PArr_size_rank1 pa n hsa, PArr_size_rank1 pb n hsb]; simp)]
            rw [ravelF_rank1 pa n hsa, ravelF_rank1 pb n hsb, List.zip_map', List.foldl_map]
            -- the Fortran fold
            unfold F.foldElems at hfold
            rw [allCoords_rank1 p 1 n hpd, List.foldl_map] at hfold
            obtain ⟨r', hr', hrr'⟩ := fold_rel_gen DotRelE (fun x c => F.arith .add x c)
              (fun s uv => (P.arith .mul uv.1 uv.2).bind fun m => P.arith .add s m)
              (fun x x' c uv r hx hc hr => dot_step_rel x x' c uv r hx hc hr)
              (fun k => p.get [1 + (k : ℤ)]) (fun k => (pa.get [k], pb.get [k])) (List.range n)
              (by
                intro k hk
                rw [List.mem_range] at hk
                obtain ⟨x, hx⟩ := hreal_a k hk
                obtain ⟨y, hy⟩ := hreal_b k hk
                refine ⟨av.get [la + k], bv.get [lb + k], ?_, ?_, ?_⟩
                · have := RelA_get1 hA la n hda (la + k) ⟨by omega, by omega⟩
                  simpa using this
                · have := RelA_get1 hB lb n hdb (lb + k) ⟨by omega, by omega⟩
                  simpa using this
                · rw [hpget (1 + k)]
                  have e1 : la + (1 + (k : ℤ)) - 1 = la + k := by ring
                  have e2 : lb + (1 + (k : ℤ)) - 1 = lb + k := by ring
                  rw [e1, e2, hx, hy]
                  rfl)
              (.real (RKind.join ka kb) 0) (.float 0) RelS.realF _ hfold
            exact ⟨r', by rw [hr']; rfl, hrr'⟩

/-! ### FNP-R0108: MAXVAL / MINVAL -/

/-- One step of `np.max`'s fold: keep the larger value (`m` on ties). -/
noncomputable def pyMaxStep (m x : PVal) : Option PVal :=
  (P.toR m).bind fun p => (P.toR x).map fun q => if p < q then x else m

/-- One step of `np.min`'s fold: keep the smaller value (`m` on ties). -/
noncomputable def pyMinStep (m x : PVal) : Option PVal :=
  (P.toR m).bind fun p => (P.toR x).map fun q => if q < p then x else m

theorem maxS_rel (x y : SVal) (x' y' : PVal) (c : SVal) (hx : RelS x x') (hy : RelS y y')
    (hc : F.maxS x y = some c) : ∃ c', pyMaxStep x' y' = some c' ∧ RelS c c' := by
  cases x with
  | int k₁ a =>
    cases y with
    | int k₂ b =>
      rw [RelS.int_eq hx, RelS.int_eq hy]
      simp only [F.maxS, Option.some.injEq] at hc
      subst hc
      refine ⟨if (a : ℝ) < (b : ℝ) then .int b else .int a, rfl, ?_⟩
      split_ifs with h
      · rw [max_eq_right (le_of_lt (by exact_mod_cast h))]; exact RelS.int
      · rw [max_eq_left (by exact_mod_cast (not_lt.mp h))]; exact RelS.int
    | real _ _ => simp [F.maxS] at hc
    | bool _ => simp [F.maxS] at hc
    | str _ => simp [F.maxS] at hc
  | real k₁ a =>
    cases y with
    | real k₂ b =>
      have ta := RelS.real_toR hx
      have tb := RelS.real_toR hy
      simp only [F.maxS, Option.some.injEq] at hc
      subst hc
      refine ⟨if a < b then y' else x', ?_, ?_⟩
      · simp only [pyMaxStep, ta, tb, Option.bind_some, Option.map_some]
      · split_ifs with h
        · rw [max_eq_right h.le]; exact RelS.real tb
        · rw [max_eq_left (not_lt.mp h)]; exact RelS.real ta
    | int _ _ => simp [F.maxS] at hc
    | bool _ => simp [F.maxS] at hc
    | str _ => simp [F.maxS] at hc
  | bool _ => cases y <;> simp [F.maxS] at hc
  | str _ => cases y <;> simp [F.maxS] at hc

theorem minS_rel (x y : SVal) (x' y' : PVal) (c : SVal) (hx : RelS x x') (hy : RelS y y')
    (hc : F.minS x y = some c) : ∃ c', pyMinStep x' y' = some c' ∧ RelS c c' := by
  cases x with
  | int k₁ a =>
    cases y with
    | int k₂ b =>
      rw [RelS.int_eq hx, RelS.int_eq hy]
      simp only [F.minS, Option.some.injEq] at hc
      subst hc
      refine ⟨if (b : ℝ) < (a : ℝ) then .int b else .int a, rfl, ?_⟩
      split_ifs with h
      · rw [min_eq_right (le_of_lt (by exact_mod_cast h))]; exact RelS.int
      · rw [min_eq_left (by exact_mod_cast (not_lt.mp h))]; exact RelS.int
    | real _ _ => simp [F.minS] at hc
    | bool _ => simp [F.minS] at hc
    | str _ => simp [F.minS] at hc
  | real k₁ a =>
    cases y with
    | real k₂ b =>
      have ta := RelS.real_toR hx
      have tb := RelS.real_toR hy
      simp only [F.minS, Option.some.injEq] at hc
      subst hc
      refine ⟨if b < a then y' else x', ?_, ?_⟩
      · simp only [pyMinStep, ta, tb, Option.bind_some, Option.map_some]
      · split_ifs with h
        · rw [min_eq_right h.le]; exact RelS.real tb
        · rw [min_eq_left (not_lt.mp h)]; exact RelS.real ta
    | int _ _ => simp [F.minS] at hc
    | bool _ => simp [F.minS] at hc
    | str _ => simp [F.minS] at hc
  | bool _ => cases y <;> simp [F.minS] at hc
  | str _ => cases y <;> simp [F.minS] at hc

def FNP_R0108_maxval_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr), SoundPair o Γ e e' → Rank1Arg o Γ e →
    SoundScalar o Γ (.call .maxval (.cons e .nil) none) (Tr.fold1 .np_max e')

def FNP_R0108_minval_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr), SoundPair o Γ e e' → Rank1Arg o Γ e →
    SoundScalar o Γ (.call .minval (.cons e .nil) none) (Tr.fold1 .np_min e')

/-- Shared proof of MAXVAL and MINVAL: a seeded fold on both sides. -/
theorem extremum_sound (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (he : SoundPair o Γ e e')
    (hr : Rank1Arg o Γ e) (f : F.Intr) (h : P.Helper) (fS : SVal → SVal → Option SVal)
    (gS : PVal → PVal → Option PVal)
    (hrel : ∀ (x y : SVal) (x' y' : PVal) (c : SVal), RelS x x' → RelS y y' → fS x y = some c →
      ∃ c', gS x' y' = some c' ∧ RelS c c')
    (hF : ∀ av : FArr, F.intrinsic o.F f [.arr av] none =
      if av.size = 0 then none else (F.foldElems av fS (av.get av.lbs)).map F.Val.s)
    (hP : ∀ p : PArr, P.helper o.P h [.arr p] =
      if p.size = 0 then none else
        ((P.ravelF p).foldl (fun acc x => acc.bind fun m => gS m x) (some (p.get (p.shape.map fun _ => 0)))).map P.Val.s)
    (hFs : ∀ sv : SVal, F.intrinsic o.F f [.s sv] none = none) :
    SoundScalar o Γ (.call f (.cons e .nil) none) (Tr.fold1 h e') := by
  intro σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | s sv => rw [hFs] at hint; cases hint
  | arr av =>
    obtain ⟨⟨lb, n, hd, hn⟩, _⟩ := hr σ τ hR hC av hva
    obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hva
    cases hrw with
    | arr hA =>
      rename_i p
      have hshape := RelA_dims1 hA lb n hd
      rw [hF, if_neg (by rw [FArr_size_rank1 av lb n hd]; omega)] at hint
      simp only [Option.map_eq_some_iff] at hint
      obtain ⟨r, hfold, hcr⟩ := hint
      cases hcr
      rw [P_eval_call1, hw, Option.bind_some, hP, if_neg (by rw [PArr_size_rank1 p n hshape]; omega)]
      rw [hshape]
      simp only [List.map_cons, List.map_nil]
      have hlbs : av.lbs = [lb] := by simp [FArr.lbs, hd]
      rw [hlbs] at hfold
      have hseed : RelS (av.get [lb]) (p.get [0]) := by
        have := RelA_get1 hA lb n hd lb ⟨le_refl _, by omega⟩; simpa using this
      unfold F.foldElems at hfold
      rw [allCoords_rank1 av lb n hd, List.foldl_map] at hfold
      rw [ravelF_rank1 p n hshape, List.foldl_map]
      obtain ⟨r', hr', hrr'⟩ := fold_rel_gen RelS fS gS
        (fun x x' y y' c hx hy hc => hrel x y x' y' c hx hy hc)
        (fun k => av.get [lb + (k : ℤ)]) (fun k => p.get [k]) (List.range n)
        (fun k hk => by
          rw [List.mem_range] at hk
          have := RelA_get1 hA lb n hd (lb + k) ⟨by omega, by omega⟩
          simpa using this)
        _ _ hseed _ hfold
      exact ⟨r', by rw [hr']; rfl, hrr'⟩

theorem FNP_R0108_maxval_sound : FNP_R0108_maxval_obligation := by
  intro o Γ e e' he hr
  refine extremum_sound o Γ e e' he hr .maxval .np_max F.maxS pyMaxStep maxS_rel ?_ ?_ ?_
  · intro av; rfl
  · intro p; rfl
  · intro sv; rfl

theorem FNP_R0108_minval_sound : FNP_R0108_minval_obligation := by
  intro o Γ e e' he hr
  refine extremum_sound o Γ e e' he hr .minval .np_min F.minS pyMinStep minS_rel ?_ ?_ ?_
  · intro av; rfl
  · intro p; rfl
  · intro sv; rfl

/-! ### FNP-R0090 on the signed range -/

/-- The `k`-bit complement of `n < 2^k`. -/
theorem testBit_cpl (k n i : ℕ) (hn : n < 2 ^ k) :
    (2 ^ k - (n + 1)).testBit i = (decide (i < k) && !n.testBit i) :=
  Nat.testBit_two_pow_sub_succ hn i

theorem testBit_high (k m i : ℕ) (hm : m < 2 ^ k) (hi : k ≤ i) : m.testBit i = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le hm (Nat.pow_le_pow_right (by norm_num) hi))

theorem ldiff_lt_two_pow (k m n : ℕ) (hm : m < 2 ^ k) : Nat.ldiff m n < 2 ^ k := by
  apply Nat.lt_pow_two_of_testBit
  intro i hi
  rw [Nat.testBit_ldiff, testBit_high k m i hm hi]
  rfl

theorem and_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    m &&& (2 ^ k - (n + 1)) = Nat.ldiff m n := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, Nat.testBit_ldiff, testBit_cpl k n i hn]
  by_cases hi : i < k
  · simp [hi]
  · simp [hi, testBit_high k m i hm (not_lt.mp hi)]

theorem cpl_and_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    (2 ^ k - (m + 1)) &&& (2 ^ k - (n + 1)) = 2 ^ k - ((m ||| n) + 1) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, testBit_cpl k m i hm, testBit_cpl k n i hn,
    testBit_cpl k (m ||| n) i (Nat.or_lt_two_pow hm hn), Nat.testBit_or]
  by_cases hi : i < k
  · simp only [hi, decide_true, Bool.true_and]
    cases m.testBit i <;> cases n.testBit i <;> rfl
  · simp [hi]

theorem or_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    m ||| (2 ^ k - (n + 1)) = 2 ^ k - (Nat.ldiff n m + 1) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_or, testBit_cpl k n i hn, testBit_cpl k (Nat.ldiff n m) i (ldiff_lt_two_pow k n m hn),
    Nat.testBit_ldiff]
  by_cases hi : i < k
  · simp only [hi, decide_true, Bool.true_and]
    cases m.testBit i <;> cases n.testBit i <;> rfl
  · simp [hi, testBit_high k m i hm (not_lt.mp hi)]

theorem cpl_or_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    (2 ^ k - (m + 1)) ||| (2 ^ k - (n + 1)) = 2 ^ k - ((m &&& n) + 1) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_or, testBit_cpl k m i hm, testBit_cpl k n i hn,
    testBit_cpl k (m &&& n) i (Nat.and_lt_two_pow m hn), Nat.testBit_and]
  by_cases hi : i < k
  · simp only [hi, decide_true, Bool.true_and]
    cases m.testBit i <;> cases n.testBit i <;> rfl
  · simp [hi]

theorem xor_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    m ^^^ (2 ^ k - (n + 1)) = 2 ^ k - ((m ^^^ n) + 1) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, testBit_cpl k n i hn, testBit_cpl k (m ^^^ n) i (Nat.xor_lt_two_pow hm hn),
    Nat.testBit_xor]
  by_cases hi : i < k
  · simp only [hi, decide_true, Bool.true_and]
    cases m.testBit i <;> cases n.testBit i <;> rfl
  · simp [hi, testBit_high k m i hm (not_lt.mp hi)]

theorem cpl_xor_cpl (k m n : ℕ) (hm : m < 2 ^ k) (hn : n < 2 ^ k) :
    (2 ^ k - (m + 1)) ^^^ (2 ^ k - (n + 1)) = m ^^^ n := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, testBit_cpl k m i hm, testBit_cpl k n i hn, Nat.testBit_xor]
  by_cases hi : i < k
  · simp only [hi, decide_true, Bool.true_and]
    cases m.testBit i <;> cases n.testBit i <;> rfl
  · simp [hi, testBit_high k m i hm (not_lt.mp hi), testBit_high k n i hn (not_lt.mp hi)]

/-- The unsigned image of a non-negative in-range integer. -/
theorem unsignedK_natCast (k m : ℕ) (hm : m < 2 ^ k) : unsignedK k (m : ℤ) = m := by
  unfold unsignedK
  rw [Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hm)]
  simp

/-- The unsigned image of a negative in-range integer is the `k`-bit complement. -/
theorem unsignedK_negSucc (k m : ℕ) (hm : m < 2 ^ k) : unsignedK k (Int.negSucc m) = 2 ^ k - (m + 1) := by
  unfold unsignedK
  rw [Int.negSucc_emod m (by positivity), Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hm)]
  have hle : m + 1 ≤ 2 ^ k := hm
  have : ((2 : ℤ) ^ k - 1 - (m : ℤ)) = ((2 ^ k - (m + 1) : ℕ) : ℤ) := by
    push_cast [Nat.cast_sub hle]; ring
  rw [this, Int.toNat_natCast]

theorem two_pow_pred_lt (k : ℕ) (hb : 1 ≤ k) : 2 ^ (k - 1) < 2 ^ k :=
  Nat.pow_lt_pow_right (by norm_num) (by omega)

theorem two_pow_eq_two_mul_pred (k : ℕ) (hb : 1 ≤ k) : (2 : ℤ) ^ k = 2 * 2 ^ (k - 1) := by
  rw [← pow_succ']; congr 1; omega

/-- Wrapping a value below `2^(k-1)` is the identity. -/
theorem wrap_lt (k N : ℕ) (hb : 1 ≤ k) (hN : N < 2 ^ (k - 1)) : wrapSigned k (N : ℤ) = N :=
  wrapSigned_of_lt k N (by positivity) (by exact_mod_cast hN) hb

/-- Wrapping the `k`-bit complement of `N < 2^(k-1)` gives `-(N+1)`. -/
theorem wrap_ge (k N : ℕ) (hb : 1 ≤ k) (hN : N < 2 ^ (k - 1)) :
    wrapSigned k ((2 ^ k - (N + 1) : ℕ) : ℤ) = Int.negSucc N := by
  have hlt := two_pow_pred_lt k hb
  have hle : N + 1 ≤ 2 ^ k := by omega
  have hv : ((2 ^ k - (N + 1) : ℕ) : ℤ) = 2 ^ k - ((N : ℤ) + 1) := by push_cast [Nat.cast_sub hle]; ring
  have hHz : ((2 ^ (k - 1) : ℕ) : ℤ) = (2 : ℤ) ^ (k - 1) := by push_cast; rfl
  have hNz : (N : ℤ) < 2 ^ (k - 1) := by exact_mod_cast hN
  have h2k := two_pow_eq_two_mul_pred k hb
  have hpos : (0 : ℤ) < 2 ^ (k - 1) := by positivity
  unfold wrapSigned
  rw [hv]
  have hmod : ((2 : ℤ) ^ k - ((N : ℤ) + 1)) % 2 ^ k = 2 ^ k - ((N : ℤ) + 1) :=
    Int.emod_eq_of_lt (by rw [h2k]; linarith) (by linarith)
  simp only [hmod]
  rw [h2k, show (2 * (2 : ℤ) ^ (k - 1)) / 2 = (2 : ℤ) ^ (k - 1) by
    rw [mul_comm]; exact Int.mul_ediv_cancel _ (by norm_num)]
  rw [if_pos (by linarith), Int.negSucc_eq]
  ring

/-- Fortran's kind-width bit operations agree with Python's unbounded
two's-complement operations for every operand in the kind's signed range. -/
theorem bitK_eq_int_ops (bits : ℕ) (x y : ℤ) (hb : 1 ≤ bits)
    (hx : -2 ^ (bits - 1) ≤ x) (hx' : x < 2 ^ (bits - 1))
    (hy : -2 ^ (bits - 1) ≤ y) (hy' : y < 2 ^ (bits - 1)) :
    bitAndK bits x y = Int.land x y ∧ bitOrK bits x y = Int.lor x y ∧ bitXorK bits x y = Int.xor x y := by
  have hlt := two_pow_pred_lt bits hb
  -- represent each operand as a natural or a negative successor below 2^(bits-1)
  have repr : ∀ z : ℤ, -2 ^ (bits - 1) ≤ z → z < 2 ^ (bits - 1) →
      (∃ m : ℕ, z = (m : ℤ) ∧ m < 2 ^ (bits - 1)) ∨ (∃ m : ℕ, z = Int.negSucc m ∧ m < 2 ^ (bits - 1)) := by
    intro z hz hz'
    rcases le_or_gt 0 z with h0 | h0
    · obtain ⟨m, rfl⟩ := Int.eq_ofNat_of_zero_le h0
      exact Or.inl ⟨m, rfl, by exact_mod_cast hz'⟩
    · obtain ⟨m, rfl⟩ := Int.eq_negSucc_of_lt_zero h0
      refine Or.inr ⟨m, rfl, ?_⟩
      rw [Int.negSucc_eq] at hz
      have : (m : ℤ) + 1 ≤ 2 ^ (bits - 1) := by linarith
      have : m + 1 ≤ 2 ^ (bits - 1) := by exact_mod_cast this
      omega
  rcases repr x hx hx' with ⟨m, rfl, hm⟩ | ⟨m, rfl, hm⟩ <;>
  rcases repr y hy hy' with ⟨n, rfl, hn⟩ | ⟨n, rfl, hn⟩
  · -- both non-negative
    have hm' : m < 2 ^ bits := lt_trans hm hlt
    have hn' : n < 2 ^ bits := lt_trans hn hlt
    refine ⟨?_, ?_, ?_⟩
    · show wrapSigned bits ((unsignedK bits (m : ℤ) &&& unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_natCast bits n hn']
      exact wrap_lt bits _ hb (Nat.and_lt_two_pow m hn)
    · show wrapSigned bits ((unsignedK bits (m : ℤ) ||| unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_natCast bits n hn']
      exact wrap_lt bits _ hb (Nat.or_lt_two_pow hm hn)
    · show wrapSigned bits ((unsignedK bits (m : ℤ) ^^^ unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_natCast bits n hn']
      exact wrap_lt bits _ hb (Nat.xor_lt_two_pow hm hn)
  · -- x ≥ 0, y < 0
    have hm' : m < 2 ^ bits := lt_trans hm hlt
    have hn' : n < 2 ^ bits := lt_trans hn hlt
    refine ⟨?_, ?_, ?_⟩
    · show wrapSigned bits ((unsignedK bits (m : ℤ) &&& unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_negSucc bits n hn', and_cpl bits m n hm' hn']
      exact wrap_lt bits _ hb (ldiff_lt_two_pow _ m n hm)
    · show wrapSigned bits ((unsignedK bits (m : ℤ) ||| unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_negSucc bits n hn', or_cpl bits m n hm' hn']
      exact wrap_ge bits _ hb (ldiff_lt_two_pow _ n m hn)
    · show wrapSigned bits ((unsignedK bits (m : ℤ) ^^^ unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_natCast bits m hm', unsignedK_negSucc bits n hn', xor_cpl bits m n hm' hn']
      exact wrap_ge bits _ hb (Nat.xor_lt_two_pow hm hn)
  · -- x < 0, y ≥ 0
    have hm' : m < 2 ^ bits := lt_trans hm hlt
    have hn' : n < 2 ^ bits := lt_trans hn hlt
    refine ⟨?_, ?_, ?_⟩
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) &&& unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_natCast bits n hn', Nat.and_comm, and_cpl bits n m hn' hm']
      exact wrap_lt bits _ hb (ldiff_lt_two_pow _ n m hn)
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) ||| unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_natCast bits n hn', Nat.or_comm, or_cpl bits n m hn' hm']
      exact wrap_ge bits _ hb (ldiff_lt_two_pow _ m n hm)
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) ^^^ unsignedK bits (n : ℤ) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_natCast bits n hn', Nat.xor_comm, xor_cpl bits n m hn' hm',
        Nat.xor_comm n m]
      exact wrap_ge bits _ hb (Nat.xor_lt_two_pow hm hn)
  · -- both negative
    have hm' : m < 2 ^ bits := lt_trans hm hlt
    have hn' : n < 2 ^ bits := lt_trans hn hlt
    refine ⟨?_, ?_, ?_⟩
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) &&& unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_negSucc bits n hn', cpl_and_cpl bits m n hm' hn']
      exact wrap_ge bits _ hb (Nat.or_lt_two_pow hm hn)
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) ||| unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_negSucc bits n hn', cpl_or_cpl bits m n hm' hn']
      exact wrap_ge bits _ hb (Nat.and_lt_two_pow m hn)
    · show wrapSigned bits ((unsignedK bits (Int.negSucc m) ^^^ unsignedK bits (Int.negSucc n) : ℕ) : ℤ) = _
      rw [unsignedK_negSucc bits m hm', unsignedK_negSucc bits n hn', cpl_xor_cpl bits m n hm' hn']
      exact wrap_lt bits _ hb (Nat.xor_lt_two_pow hm hn)

/-- In-range premise on the *signed* range of the joined kind. -/
def BitOpsInRangeSigned (o : Oracles) (Γ : F.Env) (a b : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k₁ x k₂ y,
    F.eval o.F σ a = some (.s (.int k₁ x)) → F.eval o.F σ b = some (.s (.int k₂ y)) →
    -2 ^ ((IKind.join k₁ k₂).bits - 1) ≤ x ∧ x < 2 ^ ((IKind.join k₁ k₂).bits - 1) ∧
    -2 ^ ((IKind.join k₁ k₂).bits - 1) ≤ y ∧ y < 2 ^ ((IKind.join k₁ k₂).bits - 1)

theorem bitOpsInRangeSigned_of_nonneg (o : Oracles) (Γ : F.Env) (a b : F.Expr) (h : BitOpsInRange o Γ a b) :
    BitOpsInRangeSigned o Γ a b := by
  intro σ τ hR hC k₁ x k₂ y ha hb
  obtain ⟨hx0, hx, hy0, hy⟩ := h σ τ hR hC k₁ x k₂ y ha hb
  have : (0 : ℤ) ≤ 2 ^ ((IKind.join k₁ k₂).bits - 1) := by positivity
  exact ⟨by linarith, hx, by linarith, hy⟩

def FNP_R0090_signed_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → IntTyped o Γ a → IntTyped o Γ b →
    BitOpsInRangeSigned o Γ a b →
    SoundScalar o Γ (.call .iand (.cons a (.cons b .nil)) none) (Tr.call2 .f_iand a' b') ∧
    SoundScalar o Γ (.call .ior (.cons a (.cons b .nil)) none) (Tr.call2 .f_ior a' b') ∧
    SoundScalar o Γ (.call .ieor (.cons a (.cons b .nil)) none) (Tr.call2 .f_ieor a' b')

theorem bitop_sound_signed (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr) (f : F.Intr) (h : P.Helper)
    (hf : (f = .iand ∧ h = .f_iand) ∨ (f = .ior ∧ h = .f_ior) ∨ (f = .ieor ∧ h = .f_ieor))
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b)
    (hrange : BitOpsInRangeSigned o Γ a b) :
    SoundScalar o Γ (.call f (.cons a (.cons b .nil)) none) (Tr.call2 h a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  have hf' : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨
      f = .atan2 ∨ f = .iand ∨ f = .ior ∨ f = .ieor ∨ f = .ishft := by
    rcases hf with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp
  cases va with
  | arr av => exact absurd hint (intrinsic2_arr_ne o.F f _ _ _ (Or.inl ⟨av, rfl⟩) hf')
  | s x =>
    cases vb with
    | arr bv => exact absurd hint (intrinsic2_arr_ne o.F f _ _ _ (Or.inr ⟨bv, rfl⟩) hf')
    | s y =>
      obtain ⟨k₁, xv, rfl⟩ := hia σ τ hR hC x hva
      obtain ⟨k₂, yv, rfl⟩ := hib σ τ hR hC y hvb
      obtain ⟨hx0, hxlt, hy0, hylt⟩ := hrange σ τ hR hC k₁ xv k₂ yv hva hvb
      obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
      obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
      rw [RelS.int_eq hrx] at hx'
      rw [RelS.int_eq hry] at hy'
      obtain ⟨hand, hor, hxor⟩ :=
        bitK_eq_int_ops (IKind.join k₁ k₂).bits xv yv (IKind.one_le_bits _) hx0 hxlt hy0 hylt
      rw [P_eval_call2, hx', hy', Option.bind_some, Option.bind_some]
      rcases hf with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · rw [show F.intrinsic o.F .iand [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitAndK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.land xv yv), rfl, ?_⟩
        rw [hand]; exact RelS.int
      · rw [show F.intrinsic o.F .ior [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitOrK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.lor xv yv), rfl, ?_⟩
        rw [hor]; exact RelS.int
      · rw [show F.intrinsic o.F .ieor [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitXorK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.xor xv yv), rfl, ?_⟩
        rw [hxor]; exact RelS.int

theorem FNP_R0090_sound_signed : FNP_R0090_signed_obligation := by
  intro o Γ a b a' b' ha hb hia hib hrange
  exact ⟨bitop_sound_signed o Γ a b a' b' _ _ (Or.inl ⟨rfl, rfl⟩) ha hb hia hib hrange,
    bitop_sound_signed o Γ a b a' b' _ _ (Or.inr (Or.inl ⟨rfl, rfl⟩)) ha hb hia hib hrange,
    bitop_sound_signed o Γ a b a' b' _ _ (Or.inr (Or.inr ⟨rfl, rfl⟩)) ha hb hia hib hrange⟩

/-- The non-negative statement of `Intrinsics2.lean` is the special case. -/
theorem FNP_R0090_sound_of_signed : FNP_R0090_obligation := by
  intro o Γ a b a' b' ha hb hia hib hrange
  exact FNP_R0090_sound_signed o Γ a b a' b' ha hb hia hib (bitOpsInRangeSigned_of_nonneg o Γ a b hrange)

end Recast.Rules
