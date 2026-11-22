import RecastProofs.Rules.Intrinsics

/-!
# Rules FNP-R0038 .. FNP-R0046: subscripts and rank-1 sections

Element access is proved for any declared origin (unit or not).  Unit-step
sections are proved under an edge condition (`1 ≤ lo ≤ n + 1`, `0 ≤ hi ≤ n`)
that covers every non-empty in-bounds section and the empty sections whose
rendered edges stay in `[0, n]`; the rule *as accepted by the engine* (any
edges) is refuted by `a(1:-1)`, which CPython's negative-index wrap turns into
a non-empty slice (FNP-D0037).  The descending-section rule with an implied
edge is refuted (FNP-D0001).
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Unfolding lemmas -/

theorem F_eval_ref (o : F.Oracle) (σ : F.State) (a : String) (subs : F.Subs) :
    F.eval o σ (.ref a subs) = (σ a).bind fun arr => match arr with
      | .arr av => (F.evalSubs o σ subs).bind fun axes => F.sectionOf av axes
      | .s _ => none := by
  cases h : σ a <;> simp only [F.eval, h] <;> rfl

theorem P_eval_index (o : P.Oracle) (τ : P.State) (a : String) (idx : P.Idxs) :
    P.eval o τ (.index a idx) = (τ a).bind fun base => match base with
      | .arr arr => (P.evalIdxs o τ idx).bind fun ps => P.indexArr arr ps
      | _ => none := by
  cases h : τ a <;> simp only [P.eval, h] <;> rfl

theorem F_evalSubs_scalar1 (o : F.Oracle) (σ : F.State) (i : F.Expr) :
    F.evalSubs o σ (.cons (.scalar i) .nil) = (F.eval o σ i).bind fun v => match v with
      | .s (.int _ iv) => some [.fixed iv]
      | _ => none := by
  cases h : F.eval o σ i <;> simp only [F.evalSubs, F.evalSub, h] <;> (try rfl)
  rename_i v; cases v with
  | s sv => cases sv <;> rfl
  | arr _ => rfl

theorem F_evalSubs_triplet1 (o : F.Oracle) (σ : F.State) (lo hi : F.Bound) (st : F.Bound) :
    F.evalSubs o σ (.cons (.triplet lo hi st) .nil) =
      (F.evalBound o σ lo).bind fun l => (F.evalBound o σ hi).bind fun h =>
        (F.evalBound o σ st).bind fun s => some [.range l h s] := by
  cases h1 : F.evalBound o σ lo <;> cases h2 : F.evalBound o σ hi <;> cases h3 : F.evalBound o σ st <;>
    simp only [F.evalSubs, F.evalSub, h1, h2, h3] <;> rfl

theorem F_evalBound_given_int (o : F.Oracle) (σ : F.State) (e : F.Expr) (k : IKind) (v : ℤ)
    (h : F.eval o σ e = some (.s (.int k v))) : F.evalBound o σ (.given e) = some (some v) := by
  simp [F.evalBound, h]

theorem F_evalBound_omitted (o : F.Oracle) (σ : F.State) : F.evalBound o σ .omitted = some none := rfl

/-- A defined given bound is an integer scalar. -/
theorem F_evalBound_given_eq_some (o : F.Oracle) (σ : F.State) (e : F.Expr) (b : Option ℤ)
    (h : F.evalBound o σ (.given e) = some b) : ∃ k v, F.eval o σ e = some (.s (.int k v)) ∧ b = some v := by
  cases he : F.eval o σ e with
  | none => simp [F.evalBound, he] at h
  | some v =>
    cases v with
    | arr _ => simp [F.evalBound, he] at h
    | s sv =>
      cases sv <;> simp [F.evalBound, he] at h
      first | exact ⟨_, _, rfl, h.symm⟩ | exact ⟨_, _, rfl, h⟩

theorem P_evalIdxs_scalar1 (o : P.Oracle) (τ : P.State) (e : P.Expr) :
    P.evalIdxs o τ (.cons (.scalar e) .nil) = (P.eval o τ e).bind fun v => match v with
      | .s (.int j) => some [.at j]
      | .slice s e st => some [.sl s e (st.getD 1)]
      | _ => none := by
  cases h : P.eval o τ e <;> simp only [P.evalIdxs, P.evalIdx, h] <;> (try rfl)
  rename_i v; cases v with
  | s sv => cases sv <;> rfl
  | arr _ => rfl
  | slice _ _ _ => rfl

theorem P_evalIdxs_slice1 (o : P.Oracle) (τ : P.State) (s e : P.Bound) (st : P.Bound) :
    P.evalIdxs o τ (.cons (.slice s e st) .nil) =
      (P.evalBound o τ s).bind fun vs => (P.evalBound o τ e).bind fun ve =>
        (P.evalBound o τ st).bind fun vst => some [.sl vs ve (vst.getD 1)] := by
  cases h1 : P.evalBound o τ s <;> cases h2 : P.evalBound o τ e <;> cases h3 : P.evalBound o τ st <;>
    simp only [P.evalIdxs, P.evalIdx, h1, h2, h3] <;> rfl

theorem P_evalBound_some_int (o : P.Oracle) (τ : P.State) (e : P.Expr) (v : ℤ)
    (h : P.eval o τ e = some (.s (.int v))) : P.evalBound o τ (.some e) = some (some v) := by
  simp [P.evalBound, h]

theorem P_evalBound_none (o : P.Oracle) (τ : P.State) : P.evalBound o τ .none = some none := rfl

/-! ### Rank-1 array facts -/

theorem sectionOf_fixed (a : FArr) (lb : ℤ) (n : ℕ) (i : ℤ) (hd : a.dims = [(lb, n)])
    (hi : lb ≤ i ∧ i < lb + n) : F.sectionOf a [.fixed i] = some (.s (a.get [i])) := by
  simp [F.sectionOf, F.resolveAxis, hd, hi]

theorem sectionOf_fixed_oob (a : FArr) (lb : ℤ) (n : ℕ) (i : ℤ) (hd : a.dims = [(lb, n)])
    (hi : ¬ (lb ≤ i ∧ i < lb + n)) : F.sectionOf a [.fixed i] = none := by
  simp [F.sectionOf, F.resolveAxis, hd, hi]

theorem indexArr_at (p : PArr) (n : ℕ) (j : ℤ) (hs : p.shape = [n]) (hj : 0 ≤ j ∧ j < n) :
    P.indexArr p [.at j] = some (.s (p.get [j.toNat])) := by
  have h1 : ¬ (j < 0) := by omega
  simp [P.indexArr, Py.wrapIndex, hs, h1, hj.1, hj.2]

theorem RelA_dims1 {a : FArr} {p : PArr} (h : RelA a p) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) :
    p.shape = [n] := by
  rw [h.shape]; simp [FArr.extents, hd]

theorem RelA_get1 {a : FArr} {p : PArr} (h : RelA a p) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)])
    (i : ℤ) (hi : lb ≤ i ∧ i < lb + n) : RelS (a.get [i]) (p.get [(i - lb).toNat]) := by
  have hb : a.inBounds [i] := by
    simp [FArr.inBounds, FArr.inBoundsB, hd, hi]
  have := h.elems [i] hb
  simpa [rebase, FArr.lbs, hd] using this

/-! ### Element access (FNP-R0039 unit origin, FNP-R0040 general origin) -/

/-- The declared array is bound on both sides to related rank-1 arrays whose
lower bound is the value of the declared bound expression. -/
def Rank1Decl (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ →
    ∃ av p l n, σ a = some (.arr av) ∧ τ a = some (.arr p) ∧ RelA av p ∧
      av.dims = [(l, n)] ∧ F.eval o.F σ lbE = some (.s (.int .i4 l))

/-- Generic element-access lemma: any shifted index expression evaluating to
`i - lb` is sound. -/
theorem element_sound (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (i : F.Expr) (sh : P.Expr)
    (hdecl : Rank1Decl o Γ a lbE)
    (hsh : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ iv l k, F.eval o.F σ i = some (.s (.int k iv)) →
      F.eval o.F σ lbE = some (.s (.int .i4 l)) → P.eval o.P τ sh = some (.s (.int (iv - l))))
    (hint : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ i = some (.s v) → ∃ k iv, v = .int k iv) :
    SoundPair o Γ (.ref a (.cons (.scalar i) .nil)) (.index a (.cons (.scalar sh) .nil)) := by
  intro σ τ hR hC v hv
  obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
  rw [F_eval_ref, hσ, Option.bind_some] at hv
  simp only at hv
  rw [F_evalSubs_scalar1] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨axes, ⟨vi, hvi, hm⟩, hrest⟩ := hv
  cases vi with
  | arr _ => simp at hm
  | s sv =>
    obtain ⟨k, iv, rfl⟩ := hint σ τ hR hC sv hvi
    simp only [Option.some.injEq] at hm
    subst hm
    by_cases hb : (l ≤ iv ∧ iv < l + n)
    · rw [sectionOf_fixed av l n iv hd hb] at hrest
      cases hrest
      refine ⟨.s (p.get [(iv - l).toNat]), ?_, RelV.s (RelA_get1 hrel l n hd iv hb)⟩
      rw [P_eval_index, hτ, Option.bind_some]
      simp only
      rw [P_evalIdxs_scalar1, hsh σ τ hR hC iv l k hvi hl, Option.bind_some]
      simp only
      exact indexArr_at p n (iv - l) (RelA_dims1 hrel l n hd) (by omega)
    · rw [sectionOf_fixed_oob av l n iv hd hb] at hrest
      cases hrest

def FNP_R0039_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (i : F.Expr) (i' : P.Expr) (k : IKind),
    Rank1Decl o Γ a (.intLit k 1) → SoundScalar o Γ i i' → IntTyped o Γ i →
    SoundPair o Γ (.ref a (.cons (.scalar i) .nil))
      (.index a (.cons (.scalar (.bin .sub i' (.intLit 1))) .nil))

theorem FNP_R0039_sound : FNP_R0039_obligation := by
  intro o Γ a i i' k hdecl hi hint
  refine element_sound o Γ a (.intLit k 1) i (.bin .sub i' (.intLit 1)) hdecl ?_ hint
  intro σ τ hR hC iv l k' hiv hl
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  obtain ⟨w, hw, hrw⟩ := hi σ τ hR hC _ hiv
  cases hrw
  rw [P_eval_bin, hw]
  simp [P.eval, P.lift2, P.arith]

def FNP_R0040_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lb' : P.Expr) (i : F.Expr) (i' : P.Expr),
    Rank1Decl o Γ a lbE → SoundScalar o Γ i i' → SoundScalar o Γ lbE lb' → IntTyped o Γ i →
    SoundPair o Γ (.ref a (.cons (.scalar i) .nil))
      (.index a (.cons (.scalar (.bin .sub (.paren i') (.paren lb'))) .nil))

theorem FNP_R0040_sound : FNP_R0040_obligation := by
  intro o Γ a lbE lb' i i' hdecl hi hlb hint
  refine element_sound o Γ a lbE i (.bin .sub (.paren i') (.paren lb')) hdecl ?_ hint
  intro σ τ hR hC iv l k hiv hl
  obtain ⟨w, hw, hrw⟩ := hi σ τ hR hC _ hiv
  obtain ⟨wl, hwl, hrwl⟩ := hlb σ τ hR hC _ hl
  cases hrw; cases hrwl
  rw [P_eval_bin, P_eval_paren, P_eval_paren, hw, hwl]
  simp [P.lift2, P.arith]

/-- FNP-R0038: the folded literal index is the same rule with `sh = intLit (v - 1)`. -/
def FNP_R0038_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (k k' : IKind) (v : ℤ),
    Rank1Decl o Γ a (.intLit k 1) →
    SoundPair o Γ (.ref a (.cons (.scalar (.intLit k' v)) .nil))
      (.index a (.cons (.scalar (.intLit (v - 1))) .nil))

theorem FNP_R0038_sound : FNP_R0038_obligation := by
  intro o Γ a k k' v hdecl
  refine element_sound o Γ a (.intLit k 1) (.intLit k' v) (.intLit (v - 1)) hdecl ?_ ?_
  · intro σ τ hR hC iv l k'' hiv hl
    rw [F_eval_intLit] at hiv hl
    simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hiv hl
    obtain ⟨_, rfl⟩ := hiv
    obtain ⟨_, rfl⟩ := hl
    rfl
  · intro σ τ _ _ w hw
    rw [F_eval_intLit] at hw
    simp only [Option.some.injEq, F.Val.s.injEq] at hw
    subst hw
    exact ⟨k', v, rfl⟩

/-! ### Unit-step sections on a unit-origin rank-1 array (FNP-R0042) -/

theorem sectionCoords_unit (lo hi : ℤ) :
    Fort.sectionCoords lo hi 1 = List.map (fun (k : ℕ) => lo + (k : ℤ)) (List.range (max (hi - lo + 1) 0).toNat) := by
  simp [Fort.sectionCoords, Fort.tripCount_unit]

theorem sectionCoords_unit_length (lo hi : ℤ) :
    (Fort.sectionCoords lo hi 1).length = (max (hi - lo + 1) 0).toNat := by
  simp [sectionCoords_unit]

theorem sectionCoords_unit_getD (lo hi : ℤ) (j : ℕ) (hj : j < (max (hi - lo + 1) 0).toNat) :
    (Fort.sectionCoords lo hi 1).getD j 0 = lo + j := by
  rw [sectionCoords_unit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj]
  rfl

theorem sectionCoords_mem (lo hi i : ℤ) :
    i ∈ Fort.sectionCoords lo hi 1 ↔ lo ≤ i ∧ i ≤ hi := by
  rw [sectionCoords_unit]
  simp only [List.mem_map, List.mem_range]
  constructor
  · rintro ⟨k, hk, rfl⟩
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨(i - lo).toNat, by omega, by omega⟩

/-- The Fortran side: a unit-step section whose coordinates are in bounds. -/
theorem sectionOf_range_unit (a : FArr) (n : ℕ) (lo hi : ℤ) (hd : a.dims = [(1, n)])
    (hin : ∀ i ∈ Fort.sectionCoords lo hi 1, 1 ≤ i ∧ i < 1 + n) :
    ∃ r : FArr, F.sectionOf a [.range (some lo) (some hi) none] = some (.arr r) ∧
      r.dims = [(1, (Fort.sectionCoords lo hi 1).length)] ∧
      ∀ j : ℤ, r.get [j] = a.get [(Fort.sectionCoords lo hi 1).getD (j - 1).toNat 0] := by
  have hin' : (∀ x ∈ Fort.sectionCoords lo hi 1, 1 ≤ x ∧ x < 1 + (n : ℤ)) = True := eq_true hin
  cases hsec : F.sectionOf a [.range (some lo) (some hi) none] with
  | none => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
  | some v =>
    cases v with
    | s _ => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
    | arr r =>
      simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
      subst hsec
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

theorem sectionOf_range_unit_oob (a : FArr) (n : ℕ) (lo hi : ℤ) (hd : a.dims = [(1, n)])
    (hout : ¬ ∀ i ∈ Fort.sectionCoords lo hi 1, 1 ≤ i ∧ i < 1 + n) :
    F.sectionOf a [.range (some lo) (some hi) none] = none := by
  simp [F.sectionOf, F.resolveAxis, hd, hout]

/-- CPython `a[s:e]` for `0 ≤ s ≤ n`, `0 ≤ e ≤ n`: no wrap, no clipping. -/
theorem sliceIndices_unit (n : ℕ) (s e : ℤ) (hs : 0 ≤ s) (hsn : s ≤ n) (he : 0 ≤ e) (hen : e ≤ n) :
    Py.sliceIndices n (some s) (some e) 1 =
      List.map (fun (k : ℕ) => (s + (k : ℤ)).toNat) (List.range (if s < e then (e - s).toNat else 0)) := by
  have hadj : Py.adjust n (some s) (some e) 1 = (s, e) := by
    simp only [Py.adjust, Prod.mk.injEq]
    constructor <;> (split_ifs <;> omega)
  have hlen : Py.sliceLen s e 1 = if s < e then (e - s).toNat else 0 := by
    simp only [Py.sliceLen]
    split_ifs <;> omega
  unfold Py.sliceIndices
  rw [hadj]
  simp only [hlen, mul_one]

/-- The Python side: `p[s:e]` with in-range edges. -/
theorem indexArr_slice_unit (p : PArr) (n : ℕ) (s e : ℤ) (hs : p.shape = [n])
    (h0 : 0 ≤ s) (hsn : s ≤ n) (he : 0 ≤ e) (hen : e ≤ n) :
    ∃ r : PArr, P.indexArr p [.sl (some s) (some e) 1] = some (.arr r) ∧
      r.shape = [if s < e then (e - s).toNat else 0] ∧
      ∀ j : ℕ, r.get [j] = p.get [(List.map (fun (k : ℕ) => (s + (k : ℤ)).toNat)
        (List.range (if s < e then (e - s).toNat else 0))).getD j 0] := by
  have hL := sliceIndices_unit n s e h0 hsn he hen
  cases hidx : P.indexArr p [.sl (some s) (some e) 1] with
  | none => exfalso; simp [P.indexArr, hs] at hidx
  | some v =>
    cases v with
    | s _ => exfalso; simp [P.indexArr, hs] at hidx
    | slice _ _ _ => exfalso; simp [P.indexArr, hs] at hidx
    | arr r =>
      simp [P.indexArr, hs, hL] at hidx
      subst hidx
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

/-- Section declaration premise: unit origin, rank 1. -/
def UnitRank1 (o : Oracles) (Γ : F.Env) (a : String) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ →
    ∃ av p n, σ a = some (.arr av) ∧ τ a = some (.arr p) ∧ RelA av p ∧ av.dims = [(1, n)]

/-- Obligation shape of FNP-R0042 on a unit-origin rank-1 array: `a(lo:hi)` ↦ `a[lo' - 1:hi']`. -/
def Section42 (o : Oracles) (Γ : F.Env) (a : String) (lo hi : F.Expr) (lo' hi' : P.Expr) : Prop :=
  SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil))
    (.index a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') .none) .nil))

/-- FNP-R0042 under the edge condition `1 ≤ lo ≤ n + 1 ∧ 0 ≤ hi ≤ n`, which
covers every non-empty in-bounds section and the empty sections whose rendered
edges stay within `[0, n]`. -/
theorem FNP_R0042_sound_edges (o : Oracles) (Γ : F.Env) (a : String) (lo hi : F.Expr) (lo' hi' : P.Expr)
    (hdecl : UnitRank1 o Γ a) (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hlint : IntTyped o Γ lo) (hhint : IntTyped o Γ hi)
    (hcond : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av l h k₁ k₂ n, σ a = some (.arr av) → av.dims = [(1, n)] →
      F.eval o.F σ lo = some (.s (.int k₁ l)) → F.eval o.F σ hi = some (.s (.int k₂ h)) →
      1 ≤ l ∧ l ≤ n + 1 ∧ 0 ≤ h ∧ h ≤ n) :
    Section42 o Γ a lo hi lo' hi' := by
  intro σ τ hR hC v hv
  obtain ⟨av, p, n, hσ, hτ, hrel, hd⟩ := hdecl σ τ hR hC
  rw [F_eval_ref, hσ, Option.bind_some] at hv
  simp only at hv
  rw [F_evalSubs_triplet1] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨axes, ⟨bl, hbl, bh, hbh, bs, hbs, hax⟩, hrest⟩ := hv
  simp only [Option.some.injEq] at hax
  subst hax
  -- the edges are integers
  obtain ⟨k₁, l, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
  obtain ⟨k₂, h, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
  rw [F_evalBound_omitted] at hbs
  cases hbs
  obtain ⟨hl1, hln, hh0, hhn⟩ := hcond σ τ hR hC av l h k₁ k₂ n hσ hd hvl hvh
  by_cases hin : ∀ i ∈ Fort.sectionCoords l h 1, 1 ≤ i ∧ i < 1 + n
  · obtain ⟨r, hr, hrd, hrg⟩ := sectionOf_range_unit av n l h hd hin
    rw [hr] at hrest
    cases hrest
    obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
    obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
    cases hrwl; cases hrwh
    have hshape := RelA_dims1 hrel 1 n hd
    obtain ⟨r', hr', hrs', hrg'⟩ := indexArr_slice_unit p n (l - 1) h hshape (by omega) (by omega) hh0 hhn
    refine ⟨.arr r', ?_, RelV.arr ?_⟩
    · rw [P_eval_index, hτ, Option.bind_some]
      simp only
      rw [P_evalIdxs_slice1, P_evalBound_none]
      have hs : P.eval o.P τ (.bin .sub lo' (.intLit 1)) = some (.s (.int (l - 1))) := by
        rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
      rw [P_evalBound_some_int o.P τ _ _ hs, P_evalBound_some_int o.P τ _ _ hwh]
      simpa using hr'
    · -- RelA r r'
      have hlen : (if l - 1 < h then (h - (l - 1)).toNat else 0) = (max (h - l + 1) 0).toNat := by
        split_ifs with hlt <;> omega
      refine ⟨?_, ?_⟩
      · rw [hrs', hlen]; simp [FArr.extents, hrd, sectionCoords_unit_length]
      · intro idx hidx
        -- idx = [j] with 1 ≤ j ≤ len
        obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ 1 ≤ j ∧ j < 1 + (max (h - l + 1) 0).toNat := by
          simp only [FArr.inBounds, FArr.inBoundsB, hrd] at hidx
          match idx, hidx with
          | [j], hidx =>
            simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
            exact ⟨j, rfl, by simpa [sectionCoords_unit_length] using hidx.1⟩
          | [], hidx => simp [FArr.inBoundsB] at hidx
          | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
        have hjn : (j - 1).toNat < (max (h - l + 1) 0).toNat := by omega
        rw [hrg j, sectionCoords_unit_getD l h (j - 1).toNat hjn]
        simp only [rebase, FArr.lbs, hrd, List.map_cons, List.map_nil, List.zipWith_cons_cons, List.zipWith_nil_right]
        rw [hrg' (j - 1).toNat]
        have hidx' : (List.map (fun (k : ℕ) => (l - 1 + (k : ℤ)).toNat)
            (List.range (if l - 1 < h then (h - (l - 1)).toNat else 0))).getD (j - 1).toNat 0
            = (l - 1 + ((j - 1).toNat : ℤ)).toNat := by
          rw [hlen, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hjn]
          rfl
        rw [hidx']
        have hcoord : 1 ≤ l + ((j - 1).toNat : ℤ) ∧ l + ((j - 1).toNat : ℤ) < 1 + n := by
          have := hin (l + ((j - 1).toNat : ℤ)) ((sectionCoords_mem l h _).mpr ⟨by omega, by omega⟩)
          exact this
        have := RelA_get1 hrel 1 n hd (l + ((j - 1).toNat : ℤ)) hcoord
        convert this using 3
        omega
  · rw [sectionOf_range_unit_oob av n l h hd hin] at hrest
    cases hrest

/-- The rule as the engine accepts it: any integer edges. -/
def FNP_R0042_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (lo hi : F.Expr) (lo' hi' : P.Expr),
    UnitRank1 o Γ a → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' → IntTyped o Γ lo → IntTyped o Γ hi →
    Section42 o Γ a lo hi lo' hi'

/-- A declared unit-origin rank-1 array is a related pair of arrays in every
conforming related state. -/
theorem unitRank1_of_decl (o : Oracles) (Γ : F.Env) (a : String) (ty : FTy) (k : IKind) (c : ℕ)
    (hΓ : Γ a = some ⟨ty, [.intLit k 1], c⟩) : UnitRank1 o Γ a := by
  intro σ τ hR hC
  obtain ⟨_, harr⟩ := hC a _ hΓ
  obtain ⟨av, hσ, _, hlen, hlbs, _⟩ := harr (by simp)
  obtain ⟨v, w, hσ', hτ, hrel⟩ := hR a _ hΓ
  rw [hσ] at hσ'; cases hσ'
  cases hrel with
  | arr hA =>
    rename_i p
    have h0 := hlbs 0 (by simp)
    obtain ⟨d, n, hd⟩ : ∃ d n, av.dims = [(d, n)] := by
      match hd : av.dims, hlen with
      | [(d, n)], _ => exact ⟨d, n, rfl⟩
      | [], h => simp [hd] at h
      | _ :: _ :: _, h => simp [hd] at h
    have hlb0 : av.lbs.getD 0 0 = d := by simp [FArr.lbs, hd]
    rw [hlb0] at h0
    simp only [List.getElem_cons_zero, F_eval_intLit, Option.some.injEq, F.Val.s.injEq,
      SVal.int.injEq] at h0
    have hl : d = 1 := by omega
    subst hl
    exact ⟨av, p, n, hσ, hτ, hA, hd⟩

/-- A declared rank-1 array with lower bound `lbE`. -/
theorem rank1Decl_of_decl (o : Oracles) (Γ : F.Env) (a : String) (ty : FTy) (lbE : F.Expr) (c : ℕ)
    (hΓ : Γ a = some ⟨ty, [lbE], c⟩) : Rank1Decl o Γ a lbE := by
  intro σ τ hR hC
  obtain ⟨_, harr⟩ := hC a _ hΓ
  obtain ⟨av, hσ, _, hlen, hlbs, _⟩ := harr (by simp)
  obtain ⟨v, w, hσ', hτ, hrel⟩ := hR a _ hΓ
  rw [hσ] at hσ'; cases hσ'
  cases hrel with
  | arr hA =>
    rename_i p
    have h0 := hlbs 0 (by simp)
    simp only [List.getElem_cons_zero] at h0
    obtain ⟨d, n, hd⟩ : ∃ d n, av.dims = [(d, n)] := by
      match hd : av.dims, hlen with
      | [(d, n)], _ => exact ⟨d, n, rfl⟩
      | [], h => simp [hd] at h
      | _ :: _ :: _, h => simp [hd] at h
    refine ⟨av, p, d, n, hσ, hτ, hA, hd, ?_⟩
    have : av.lbs.getD 0 0 = d := by simp [FArr.lbs, hd]
    rw [this] at h0; exact h0

/-! ### Concrete two-element arrays for the counterexamples -/

def twoF : FArr := { elemTy := .real .r8, dims := [(1, 2)], get := fun _ => .real .r8 0 }
def twoP : PArr := { shape := [2], get := fun _ => .float 0 }

theorem twoRel : RelA twoF twoP := ⟨rfl, fun _ _ => RelS.realF⟩

def declA : F.Decl := ⟨.real .r8, [.intLit .i4 1], 0⟩
def envA : F.Env := fun x => if x = "a" then some declA else none
def stateA : F.State := fun x => if x = "a" then some (.arr twoF) else none
def pstateA : P.State := fun x => if x = "a" then some (.arr twoP) else none

theorem envA_a : envA "a" = some declA := by simp [envA]

theorem relStateA : RelState envA stateA pstateA := by
  intro x d hx
  simp only [envA] at hx
  split_ifs at hx with hxa
  · subst hxa
    exact ⟨.arr twoF, .arr twoP, by simp [stateA], by simp [pstateA], RelV.arr twoRel⟩

theorem conformsA : Conforms noPrim.F envA stateA := by
  intro x d hx
  simp only [envA] at hx
  split_ifs at hx with hxa
  · subst hxa
    cases hx
    refine ⟨fun h => by simp [declA] at h, fun _ => ⟨twoF, by simp [stateA], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
    intro i hi
    simp only [declA] at hi
    match i, hi with
    | 0, _ => simp [declA, F_eval_intLit, twoF, FArr.lbs]

/-- FNP-D0037: `a(1:-1)` is empty in Fortran; `a[0:-1]` has `n - 1` elements. -/
theorem FNP_R0042_refuted : ¬ FNP_R0042_obligation := by
  intro h
  have hu : UnitRank1 noPrim envA "a" := unitRank1_of_decl noPrim envA "a" _ _ _ envA_a
  have hs := h noPrim envA "a" (.intLit .i4 1) (.intLit .i4 (-1)) (.intLit 1) (.intLit (-1)) hu
    (FNP_R0001_sound _ _ _ _) (FNP_R0001_sound _ _ _ _) (intLit_intTyped _ _ .i4 1) (intLit_intTyped _ _ .i4 (-1))
  -- the source: an empty section
  obtain ⟨r, hr, hrd, _⟩ := sectionOf_range_unit twoF 2 1 (-1) rfl (by
    intro i hi; rw [sectionCoords_mem] at hi; omega)
  have hF : F.eval noPrim.F stateA (.ref "a" (.cons (.triplet (.given (.intLit .i4 1)) (.given (.intLit .i4 (-1))) .omitted) .nil))
      = some (.arr r) := by
    rw [F_eval_ref]
    simp only [stateA, if_true, Option.bind_some]
    rw [F_evalSubs_triplet1, F_evalBound_given_int _ _ _ _ _ (F_eval_intLit _ _ .i4 1),
      F_evalBound_given_int _ _ _ _ _ (F_eval_intLit _ _ .i4 (-1)), F_evalBound_omitted]
    simpa using hr
  obtain ⟨w, hw, hrel⟩ := hs stateA pstateA relStateA conformsA (.arr r) hF
  -- the target: one element
  have hP : P.eval noPrim.P pstateA (.index "a" (.cons (.slice (.some (.bin .sub (.intLit 1) (.intLit 1))) (.some (.intLit (-1))) .none) .nil))
      = some (.arr { shape := [1], get := fun idx => twoP.get [(Py.sliceIndices 2 (some 0) (some (-1)) 1).getD (idx.headD 0) 0] }) := by
    rw [P_eval_index]
    simp only [pstateA, if_true, Option.bind_some]
    rw [P_evalIdxs_slice1, P_evalBound_none]
    have hs0 : P.eval noPrim.P pstateA (.bin .sub (.intLit 1) (.intLit 1)) = some (.s (.int 0)) := by
      rw [P_eval_bin]; simp [P.eval, P.lift2, P.arith]
    rw [P_evalBound_some_int _ _ _ _ hs0, P_evalBound_some_int _ _ _ _ rfl]
    simp [P.indexArr, twoP, Py.sliceIndices, Py.adjust, Py.sliceLen]
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have := hA.shape
    simp [FArr.extents, hrd, sectionCoords_unit_length] at this

/-! ### Descending section with an implied edge (FNP-R0046) -/

/-- Obligation shape of FNP-R0046 with a literal step `-1`: `a(lo:hi:-1)` ↦
`a[_f_rstep_lb(lo', hi', (-1), 1)]`, edges possibly omitted (`None`). -/
def Section46 (o : Oracles) (Γ : F.Env) (a : String) (lo hi : F.Bound) (lo' hi' : P.Expr) : Prop :=
  SoundPair o Γ (.ref a (.cons (.triplet lo hi (.given (.neg (.intLit .i4 1)))) .nil))
    (.index a (.cons (.scalar (Tr.call4 .f_rstep_lb lo' hi' (.paren (.neg (.intLit 1))) (.intLit 1))) .nil))

/-- The rule as accepted: an omitted edge is rendered `None`. -/
def FNP_R0046_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (hi : F.Expr) (hi' : P.Expr),
    UnitRank1 o Γ a → SoundScalar o Γ hi hi' → IntTyped o Γ hi →
    Section46 o Γ a .omitted (.given hi) .none hi'

/-- FNP-D0001: `a(:1:-1)` is `a(1)` in Fortran; `a[slice(None, None, -1)]` is the whole array reversed. -/
theorem FNP_R0046_refuted : ¬ FNP_R0046_obligation := by
  intro h
  have hu : UnitRank1 noPrim envA "a" := unitRank1_of_decl noPrim envA "a" _ _ _ envA_a
  have hs := h noPrim envA "a" (.intLit .i4 1) (.intLit 1) hu (FNP_R0001_sound _ _ _ _) (intLit_intTyped _ _ .i4 1)
  -- source: the single element a(1)
  have hcoords : Fort.sectionCoords 1 1 (-1) = [1] := by decide
  have hF : F.eval noPrim.F stateA (.ref "a" (.cons (.triplet .omitted (.given (.intLit .i4 1)) (.given (.neg (.intLit .i4 1)))) .nil))
      = some (.arr { elemTy := twoF.elemTy, dims := [(1, 1)],
                     get := fun idx => twoF.get [[(1 : ℤ)].getD ((idx.headD 1) - 1).toNat 0] }) := by
    rw [F_eval_ref]
    simp only [stateA, if_true, Option.bind_some]
    have hneg : F.eval noPrim.F stateA (.neg (.intLit .i4 1)) = some (.s (.int .i4 (-1))) := by
      simp [F.eval, F.lift1]
    rw [F_evalSubs_triplet1, F_evalBound_omitted, F_evalBound_given_int _ _ _ _ _ (F_eval_intLit _ _ .i4 1),
      F_evalBound_given_int _ _ _ _ _ hneg]
    simp [F.sectionOf, F.resolveAxis, twoF, hcoords]
  obtain ⟨w, hw, hrel⟩ := hs stateA pstateA relStateA conformsA _ hF
  -- target: the whole array reversed (two elements)
  have hP : P.eval noPrim.P pstateA (.index "a" (.cons (.scalar (Tr.call4 .f_rstep_lb .none (.intLit 1) (.paren (.neg (.intLit 1))) (.intLit 1))) .nil))
      = some (.arr { shape := [2], get := fun idx => twoP.get [(Py.sliceIndices 2 none none (-1)).getD (idx.headD 0) 0] }) := by
    rw [P_eval_index]
    simp only [pstateA, if_true, Option.bind_some]
    rw [P_evalIdxs_scalar1]
    have hcall : P.eval noPrim.P pstateA (Tr.call4 .f_rstep_lb .none (.intLit 1) (.paren (.neg (.intLit 1))) (.intLit 1))
        = some (.slice none none (some (-1))) := by
      have e1 : P.eval noPrim.P pstateA (.paren (.neg (.intLit 1))) = some (.s (.int (-1))) := by
        simp [P.eval, P.lift1]
      simp only [Tr.call4, P.eval, P.evalExprs, e1, Option.bind_some, Option.pure_def]
      rfl
    rw [hcall]
    simp [P.indexArr, twoP, Py.sliceIndices, Py.adjust, Py.sliceLen]
  rw [hP] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have := hA.shape
    simp [FArr.extents] at this

end Recast.Rules
