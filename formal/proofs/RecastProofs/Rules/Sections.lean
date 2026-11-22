import RecastProofs.Rules.Index

/-!
# Rules FNP-R0042 .. FNP-R0045 and FNP-R0047 (positive step): rank-1 sections

One generic theorem (`section_pos_sound`) covers every ascending section of a
rank-1 array whose rendered Python index evaluates to the slice
`slice(lo - lb, hi - lb + 1, st)` (an omitted edge stays `None`).  Each rule is
an instance: it only has to show that *its* emitted expression shape evaluates
to that slice, and that the edges are in range (the CPython clipping/wrap
premise, cf. FNP-D0037).
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Trip counts and slice lengths -/

theorem tripCount_eq_sliceLen (lo hi lb st : ℤ) (hst : 0 < st) :
    Py.sliceLen (lo - lb) (hi - lb + 1) st = Fort.tripCount lo hi st := by
  unfold Py.sliceLen Fort.tripCount Exact.tquot
  rw [if_pos hst]
  by_cases hle : lo ≤ hi
  · rw [if_pos (by omega)]
    have h1 : hi - lb + 1 - (lo - lb) - 1 = hi - lo := by ring
    rw [h1]
    have h2 : (hi - lo + st).tdiv st = (hi - lo) / st + 1 := by
      rw [Int.tdiv_eq_ediv_of_nonneg (by omega)]
      rw [show hi - lo + st = (hi - lo) + 1 * st by ring, Int.add_mul_ediv_right _ _ (ne_of_gt hst)]
    rw [h2, max_eq_left (by have := Int.ediv_nonneg (show 0 ≤ hi - lo by omega) hst.le; omega)]
  · rw [if_neg (by omega)]
    have hle0 : (hi - lo + st).tdiv st ≤ 0 := by
      by_cases h0 : 0 ≤ hi - lo + st
      · rw [Int.tdiv_eq_ediv_of_nonneg h0]
        exact le_of_eq (Int.ediv_eq_zero_of_lt h0 (by omega))
      · rw [show hi - lo + st = -(-(hi - lo + st)) by ring, Int.neg_tdiv]
        have := Int.tdiv_nonneg (show 0 ≤ -(hi - lo + st) by omega) hst.le
        omega
    rw [max_eq_right hle0]

theorem sectionCoords_length (lo hi st : ℤ) :
    (Fort.sectionCoords lo hi st).length = Fort.tripCount lo hi st := by
  simp [Fort.sectionCoords]

theorem sectionCoords_getD (lo hi st : ℤ) (k : ℕ) (hk : k < Fort.tripCount lo hi st) :
    (Fort.sectionCoords lo hi st).getD k 0 = lo + k * st := by
  unfold Fort.sectionCoords
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk]
  rfl

theorem sectionCoords_mem_of_lt (lo hi st : ℤ) (k : ℕ) (hk : k < Fort.tripCount lo hi st) :
    lo + k * st ∈ Fort.sectionCoords lo hi st := by
  unfold Fort.sectionCoords
  exact List.mem_map.mpr ⟨k, List.mem_range.mpr hk, rfl⟩

/-! ### The Fortran side -/

theorem sectionOf_range_pos (a : FArr) (lb : ℤ) (n : ℕ) (lo hi : Option ℤ) (st : Option ℤ)
    (hst : 0 < st.getD 1) (hd : a.dims = [(lb, n)])
    (hin : ∀ i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) (st.getD 1), lb ≤ i ∧ i < lb + n) :
    ∃ r : FArr, F.sectionOf a [.range lo hi st] = some (.arr r) ∧
      r.dims = [(1, (Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) (st.getD 1)).length)] ∧
      ∀ j : ℤ, r.get [j] =
        a.get [(Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) (st.getD 1)).getD (j - 1).toNat 0] := by
  have hin' : (∀ x ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) (st.getD 1),
      lb ≤ x ∧ x < lb + (n : ℤ)) = True := eq_true hin
  have hst0 : st.getD 1 ≠ 0 := ne_of_gt hst
  cases hsec : F.sectionOf a [.range lo hi st] with
  | none => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin', hst0] at hsec
  | some v =>
    cases v with
    | s _ => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin', hst0] at hsec
    | arr r =>
      simp [F.sectionOf, F.resolveAxis, hd, hin', hst0] at hsec
      subst hsec
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

theorem sectionOf_range_pos_oob (a : FArr) (lb : ℤ) (n : ℕ) (lo hi : Option ℤ) (st : Option ℤ)
    (hst : 0 < st.getD 1) (hd : a.dims = [(lb, n)])
    (hout : ¬ ∀ i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) (st.getD 1), lb ≤ i ∧ i < lb + n) :
    F.sectionOf a [.range lo hi st] = none := by
  have hst0 : st.getD 1 ≠ 0 := ne_of_gt hst
  simp [F.sectionOf, F.resolveAxis, hd, hout, hst0]

/-! ### The Python side -/

theorem adjust_pos (n : ℕ) (start stop : Option ℤ) (st : ℤ) (hst : 0 < st)
    (hs : ∀ s, start = some s → 0 ≤ s ∧ s ≤ n) (he : ∀ e, stop = some e → 0 ≤ e ∧ e ≤ n) :
    Py.adjust n start stop st = (start.getD 0, stop.getD n) := by
  cases start with
  | none =>
    cases stop with
    | none => simp only [Py.adjust, Prod.mk.injEq, Option.getD_none]; constructor <;> (split_ifs <;> omega)
    | some e =>
      obtain ⟨he1, he2⟩ := he e rfl
      simp only [Py.adjust, Prod.mk.injEq, Option.getD_none, Option.getD_some]
      constructor <;> (split_ifs <;> omega)
  | some s =>
    obtain ⟨hs1, hs2⟩ := hs s rfl
    cases stop with
    | none => simp only [Py.adjust, Prod.mk.injEq, Option.getD_none, Option.getD_some]; constructor <;> (split_ifs <;> omega)
    | some e =>
      obtain ⟨he1, he2⟩ := he e rfl
      simp only [Py.adjust, Prod.mk.injEq, Option.getD_some]
      constructor <;> (split_ifs <;> omega)

theorem sliceIndices_pos (n : ℕ) (start stop : Option ℤ) (st : ℤ) (hst : 0 < st)
    (hs : ∀ s, start = some s → 0 ≤ s ∧ s ≤ n) (he : ∀ e, stop = some e → 0 ≤ e ∧ e ≤ n) :
    Py.sliceIndices n start stop st =
      List.map (fun (k : ℕ) => (start.getD 0 + (k : ℤ) * st).toNat)
        (List.range (Py.sliceLen (start.getD 0) (stop.getD n) st)) := by
  unfold Py.sliceIndices
  rw [adjust_pos n start stop st hst hs he]

theorem indexArr_slice_pos (p : PArr) (n : ℕ) (start stop : Option ℤ) (st : ℤ) (hst : 0 < st)
    (hs : p.shape = [n])
    (hs' : ∀ s, start = some s → 0 ≤ s ∧ s ≤ n) (he' : ∀ e, stop = some e → 0 ≤ e ∧ e ≤ n) :
    ∃ r : PArr, P.indexArr p [.sl start stop st] = some (.arr r) ∧
      r.shape = [Py.sliceLen (start.getD 0) (stop.getD n) st] ∧
      ∀ j : ℕ, r.get [j] = p.get [(List.map (fun (k : ℕ) => (start.getD 0 + (k : ℤ) * st).toNat)
        (List.range (Py.sliceLen (start.getD 0) (stop.getD n) st))).getD j 0] := by
  have hL := sliceIndices_pos n start stop st hst hs' he'
  have hst0 : st ≠ 0 := ne_of_gt hst
  cases hidx : P.indexArr p [.sl start stop st] with
  | none => exfalso; simp [P.indexArr, hs, hst0] at hidx
  | some v =>
    cases v with
    | s _ => exfalso; simp [P.indexArr, hs, hst0] at hidx
    | slice _ _ _ => exfalso; simp [P.indexArr, hs, hst0] at hidx
    | arr r =>
      simp [P.indexArr, hs, hst0, hL] at hidx
      subst hidx
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

/-! ### The generic section theorem -/

/-- The Python index list evaluates to the rebased slice of the Fortran section,
with a positive step. -/
def SectionIdx (o : Oracles) (Γ : F.Env) (lbE : F.Expr) (lo hi st : F.Bound) (idxs : P.Idxs) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ l, F.eval o.F σ lbE = some (.s (.int .i4 l)) →
    ∀ bl bh bs, F.evalBound o.F σ lo = some bl → F.evalBound o.F σ hi = some bh →
      F.evalBound o.F σ st = some bs →
      0 < bs.getD 1 ∧
      P.evalIdxs o.P τ idxs = some [.sl (bl.map (· - l)) (bh.map (· - l + 1)) (bs.getD 1)]

/-- In-range edges: the CPython clipping / negative-wrap premise. -/
def SectionInRange (o : Oracles) (Γ : F.Env) (a : String) (lo hi : F.Bound) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av l n, σ a = some (.arr av) → av.dims = [(l, n)] →
    ∀ bl bh, F.evalBound o.F σ lo = some bl → F.evalBound o.F σ hi = some bh →
      (∀ l0, bl = some l0 → l ≤ l0 ∧ l0 ≤ l + n) ∧ (∀ h0, bh = some h0 → l - 1 ≤ h0 ∧ h0 ≤ l + n - 1)

theorem section_pos_sound (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lo hi st : F.Bound)
    (idxs : P.Idxs) (hdecl : Rank1Decl o Γ a lbE) (hidx : SectionIdx o Γ lbE lo hi st idxs)
    (hrange : SectionInRange o Γ a lo hi) :
    SoundPair o Γ (.ref a (.cons (.triplet lo hi st) .nil)) (.index a idxs) := by
  intro σ τ hR hC v hv
  obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
  rw [F_eval_ref, hσ, Option.bind_some] at hv
  simp only at hv
  rw [F_evalSubs_triplet1] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨axes, ⟨bl, hbl, bh, hbh, bs, hbs, hax⟩, hrest⟩ := hv
  simp only [Option.some.injEq] at hax
  subst hax
  obtain ⟨hpos, hpy⟩ := hidx σ τ hR hC l hl bl bh bs hbl hbh hbs
  obtain ⟨hlo_rng, hhi_rng⟩ := hrange σ τ hR hC av l n hσ hd bl bh hbl hbh
  by_cases hin : ∀ i ∈ Fort.sectionCoords (bl.getD l) (bh.getD (l + n - 1)) (bs.getD 1), l ≤ i ∧ i < l + n
  · obtain ⟨r, hr, hrd, hrg⟩ := sectionOf_range_pos av l n bl bh bs hpos hd hin
    rw [hr] at hrest
    cases hrest
    have hshape := RelA_dims1 hrel l n hd
    have hs' : ∀ s, bl.map (· - l) = some s → 0 ≤ s ∧ s ≤ n := by
      intro s hs
      cases bl with
      | none => cases hs
      | some l0 => simp only [Option.map_some, Option.some.injEq] at hs; have := hlo_rng l0 rfl; omega
    have he' : ∀ e, bh.map (· - l + 1) = some e → 0 ≤ e ∧ e ≤ n := by
      intro e he
      cases bh with
      | none => cases he
      | some h0 => simp only [Option.map_some, Option.some.injEq] at he; have := hhi_rng h0 rfl; omega
    obtain ⟨r', hr', hrs', hrg'⟩ :=
      indexArr_slice_pos p n (bl.map (· - l)) (bh.map (· - l + 1)) (bs.getD 1) hpos hshape hs' he'
    have e1 : (bl.map (· - l)).getD 0 = bl.getD l - l := by cases bl <;> simp
    have e2 : (bh.map (· - l + 1)).getD n = bh.getD (l + n - 1) - l + 1 := by cases bh <;> simp <;> omega
    have hlen : Py.sliceLen ((bl.map (· - l)).getD 0) ((bh.map (· - l + 1)).getD n) (bs.getD 1) =
        Fort.tripCount (bl.getD l) (bh.getD (l + n - 1)) (bs.getD 1) := by
      rw [e1, e2]; exact tripCount_eq_sliceLen _ _ l _ hpos
    refine ⟨.arr r', ?_, RelV.arr ?_⟩
    · rw [P_eval_index, hτ, Option.bind_some]
      simp only
      rw [hpy, Option.bind_some, hr']
    · refine ⟨?_, ?_⟩
      · rw [hrs', hlen]; simp [FArr.extents, hrd, sectionCoords_length]
      · intro idx hidx
        obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ 1 ≤ j ∧
            j < 1 + Fort.tripCount (bl.getD l) (bh.getD (l + n - 1)) (bs.getD 1) := by
          simp only [FArr.inBounds, FArr.inBoundsB, hrd] at hidx
          match idx, hidx with
          | [j], hidx =>
            simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
            exact ⟨j, rfl, by simpa [sectionCoords_length] using hidx.1⟩
          | [], hidx => simp [FArr.inBoundsB] at hidx
          | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
        have hjn : (j - 1).toNat < Fort.tripCount (bl.getD l) (bh.getD (l + n - 1)) (bs.getD 1) := by omega
        rw [hrg j, sectionCoords_getD _ _ _ _ hjn]
        simp only [rebase, FArr.lbs, hrd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
          List.zipWith_nil_right]
        rw [hrg' (j - 1).toNat]
        have hidx' : (List.map (fun (k : ℕ) => ((bl.map (· - l)).getD 0 + (k : ℤ) * bs.getD 1).toNat)
            (List.range (Py.sliceLen ((bl.map (· - l)).getD 0) ((bh.map (· - l + 1)).getD n) (bs.getD 1)))).getD
              (j - 1).toNat 0 = (bl.getD l - l + ((j - 1).toNat : ℤ) * bs.getD 1).toNat := by
          rw [hlen, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hjn, e1]
          rfl
        rw [hidx']
        have hcoord := hin _ (sectionCoords_mem_of_lt _ _ _ _ hjn)
        have := RelA_get1 hrel l n hd _ hcoord
        convert this using 3
        congr 1; ring
  · rw [sectionOf_range_pos_oob av l n bl bh bs hpos hd hin] at hrest
    cases hrest

theorem P_eval_call4 (o : P.Oracle) (τ : P.State) (h : P.Helper) (a b c d : P.Expr) :
    P.eval o τ (Tr.call4 h a b c d) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => (P.eval o τ c).bind fun vc =>
        (P.eval o τ d).bind fun vd => P.helper o h [va, vb, vc, vd] := by
  cases h1 : P.eval o τ a <;> cases h2 : P.eval o τ b <;> cases h3 : P.eval o τ c <;> cases h4 : P.eval o τ d <;>
    simp only [Tr.call4, P.eval, P.evalExprs, h1, h2, h3, h4] <;> rfl

/-- An optional edge as the Python argument value (`None` or the int). -/
def optIntVal : Option ℤ → PVal
  | none => .none
  | some l0 => .int l0

/-- `_f_rstep_any` with a non-negative step: `slice(lo - lb, hi - lb + 1, st)`, `None` edges kept. -/
theorem helper_rstep_any_pos (o : P.Oracle) (bl bh : Option ℤ) (st lb : ℤ) (hst : ¬ st < 0) :
    P.helper o .f_rstep_any [.s (optIntVal bl), .s (optIntVal bh), .s (.int st), .s (.int lb)] =
      some (.slice (bl.map (· - lb)) (bh.map (· - lb + 1)) (some st)) := by
  cases bl with
  | none =>
    cases bh with
    | none =>
      have e : P.helper o .f_rstep_any [.s .none, .s .none, .s (.int st), .s (.int lb)] =
          if st < 0 then P.helper.helperRstep (.s .none) (.s .none) st lb
          else some (.slice none none (some st)) := rfl
      simp only [optIntVal]
      rw [e, if_neg hst]; rfl
    | some h0 =>
      have e : P.helper o .f_rstep_any [.s .none, .s (.int h0), .s (.int st), .s (.int lb)] =
          if st < 0 then P.helper.helperRstep (.s .none) (.s (.int h0)) st lb
          else some (.slice none (some (h0 - lb + 1)) (some st)) := rfl
      simp only [optIntVal]
      rw [e, if_neg hst]; rfl
  | some l0 =>
    cases bh with
    | none =>
      have e : P.helper o .f_rstep_any [.s (.int l0), .s .none, .s (.int st), .s (.int lb)] =
          if st < 0 then P.helper.helperRstep (.s (.int l0)) (.s .none) st lb
          else some (.slice (some (l0 - lb)) none (some st)) := rfl
      simp only [optIntVal]
      rw [e, if_neg hst]; rfl
    | some h0 =>
      have e : P.helper o .f_rstep_any [.s (.int l0), .s (.int h0), .s (.int st), .s (.int lb)] =
          if st < 0 then P.helper.helperRstep (.s (.int l0)) (.s (.int h0)) st lb
          else some (.slice (some (l0 - lb)) (some (h0 - lb + 1)) (some st)) := rfl
      simp only [optIntVal]
      rw [e, if_neg hst]; rfl

/-! ### Instances -/

theorem P_eval_intLit (o : P.Oracle) (τ : P.State) (n : ℤ) : P.eval o τ (.intLit n) = some (.s (.int n)) := rfl

/-- FNP-R0042: unit origin, unit step, explicit edges: `a(lo:hi)` ↦ `a[lo' - 1:hi']`. -/
theorem FNP_R0042_sound_gen (o : Oracles) (Γ : F.Env) (a : String) (k : IKind) (lo hi : F.Expr) (lo' hi' : P.Expr)
    (hdecl : Rank1Decl o Γ a (.intLit k 1)) (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hrange : SectionInRange o Γ a (.given lo) (.given hi)) :
    SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil))
      (.index a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') .none) .nil)) := by
  refine section_pos_sound o Γ a (.intLit k 1) _ _ _ _ hdecl ?_ hrange
  intro σ τ hR hC l hl bl bh bs hbl hbh hbs
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  rw [F_evalBound_omitted] at hbs
  cases hbs
  obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
  obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
  obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
  rw [RelS.int_eq hrwl] at hwl
  rw [RelS.int_eq hrwh] at hwh
  refine ⟨by simp, ?_⟩
  have hs : P.eval o.P τ (.bin .sub lo' (.intLit 1)) = some (.s (.int (l0 - 1))) := by
    rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
  rw [P_evalIdxs_slice1, P_evalBound_none, P_evalBound_some_int _ _ _ _ hs, P_evalBound_some_int _ _ _ _ hwh]
  simp

/-- FNP-R0043: non-unit origin, unit step: `a(lo:hi)` ↦ `a[(lo') - (lb'):(hi') - (lb') + 1]`. -/
theorem FNP_R0043_sound (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lb' : P.Expr) (lo hi : F.Expr)
    (lo' hi' : P.Expr) (hdecl : Rank1Decl o Γ a lbE) (hlb : SoundScalar o Γ lbE lb')
    (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hrange : SectionInRange o Γ a (.given lo) (.given hi)) :
    SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil))
      (.index a (.cons (.slice (.some (.bin .sub (.paren lo') (.paren lb')))
        (.some (.bin .add (.bin .sub (.paren hi') (.paren lb')) (.intLit 1))) .none) .nil)) := by
  refine section_pos_sound o Γ a lbE _ _ _ _ hdecl ?_ hrange
  intro σ τ hR hC l hl bl bh bs hbl hbh hbs
  rw [F_evalBound_omitted] at hbs
  cases hbs
  obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
  obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
  obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
  obtain ⟨wb, hwb, hrwb⟩ := hlb σ τ hR hC _ hl
  rw [RelS.int_eq hrwl] at hwl
  rw [RelS.int_eq hrwh] at hwh
  rw [RelS.int_eq hrwb] at hwb
  refine ⟨by simp, ?_⟩
  have hs : P.eval o.P τ (.bin .sub (.paren lo') (.paren lb')) = some (.s (.int (l0 - l))) := by
    rw [P_eval_bin, P_eval_paren, P_eval_paren, hwl, hwb]; simp [P.lift2, P.arith]
  have he : P.eval o.P τ (.bin .add (.bin .sub (.paren hi') (.paren lb')) (.intLit 1)) = some (.s (.int (h0 - l + 1))) := by
    rw [P_eval_bin, P_eval_bin, P_eval_paren, P_eval_paren, hwh, hwb]; simp [P.eval, P.lift2, P.arith]
  rw [P_evalIdxs_slice1, P_evalBound_none, P_evalBound_some_int _ _ _ _ hs, P_evalBound_some_int _ _ _ _ he]
  simp

/-- FNP-R0044: omitted edges on a unit-origin axis: `a(:hi)`, `a(lo:)`, `a(:)`. -/
theorem FNP_R0044_sound (o : Oracles) (Γ : F.Env) (a : String) (k : IKind) (lo hi : Option F.Expr)
    (lo' hi' : Option P.Expr) (hdecl : Rank1Decl o Γ a (.intLit k 1))
    (hlo : ∀ e, lo = some e → ∃ e', lo' = some e' ∧ SoundScalar o Γ e e') (hlo' : lo = none → lo' = none)
    (hhi : ∀ e, hi = some e → ∃ e', hi' = some e' ∧ SoundScalar o Γ e e') (hhi' : hi = none → hi' = none)
    (hrange : SectionInRange o Γ a (lo.elim .omitted .given) (hi.elim .omitted .given)) :
    SoundPair o Γ (.ref a (.cons (.triplet (lo.elim .omitted .given) (hi.elim .omitted .given) .omitted) .nil))
      (.index a (.cons (.slice (lo'.elim .none fun e' => .some (.bin .sub e' (.intLit 1)))
        (hi'.elim .none .some) .none) .nil)) := by
  refine section_pos_sound o Γ a (.intLit k 1) _ _ _ _ hdecl ?_ hrange
  intro σ τ hR hC l hl bl bh bs hbl hbh hbs
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  rw [F_evalBound_omitted] at hbs
  cases hbs
  refine ⟨by simp, ?_⟩
  rw [P_evalIdxs_slice1, P_evalBound_none]
  -- start
  have hstart : P.evalBound o.P τ (lo'.elim .none fun e' => .some (.bin .sub e' (.intLit 1))) =
      some (bl.map (· - 1)) := by
    cases lo with
    | none =>
      rw [hlo' rfl]
      simp only [Option.elim] at hbl ⊢
      rw [F_evalBound_omitted] at hbl; cases hbl; rfl
    | some e =>
      obtain ⟨e', he', hse⟩ := hlo e rfl
      subst he'
      simp only [Option.elim] at hbl ⊢
      obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
      obtain ⟨wl, hwl, hrwl⟩ := hse σ τ hR hC _ hvl
      rw [RelS.int_eq hrwl] at hwl
      have hs : P.eval o.P τ (.bin .sub e' (.intLit 1)) = some (.s (.int (l0 - 1))) := by
        rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
      rw [P_evalBound_some_int _ _ _ _ hs]; rfl
  have hstop : P.evalBound o.P τ (hi'.elim .none .some) = some (bh.map (· - 1 + 1)) := by
    cases hi with
    | none =>
      rw [hhi' rfl]
      simp only [Option.elim] at hbh ⊢
      rw [F_evalBound_omitted] at hbh; cases hbh; rfl
    | some e =>
      obtain ⟨e', he', hse⟩ := hhi e rfl
      subst he'
      simp only [Option.elim] at hbh ⊢
      obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
      obtain ⟨wh, hwh, hrwh⟩ := hse σ τ hR hC _ hvh
      rw [RelS.int_eq hrwh] at hwh
      rw [P_evalBound_some_int _ _ _ _ hwh]; simp
  rw [hstart, hstop]
  rfl

/-- FNP-R0045: positive literal step on a unit-origin axis: `a(lo:hi:s)` ↦ `a[lo' - 1:hi':s]`. -/
theorem FNP_R0045_sound (o : Oracles) (Γ : F.Env) (a : String) (k ks : IKind) (lo hi : F.Expr) (lo' hi' : P.Expr)
    (s : ℤ) (hs : 0 < s)
    (hdecl : Rank1Decl o Γ a (.intLit k 1)) (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hrange : SectionInRange o Γ a (.given lo) (.given hi)) :
    SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) (.given (.intLit ks s))) .nil))
      (.index a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') (.some (.intLit s))) .nil)) := by
  refine section_pos_sound o Γ a (.intLit k 1) _ _ _ _ hdecl ?_ hrange
  intro σ τ hR hC l hl bl bh bs hbl hbh hbs
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  rw [F_evalBound_given_int _ _ _ _ _ (F_eval_intLit _ _ ks s)] at hbs
  cases hbs
  obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
  obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
  obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
  rw [RelS.int_eq hrwl] at hwl
  rw [RelS.int_eq hrwh] at hwh
  refine ⟨by simpa using hs, ?_⟩
  have hst : P.eval o.P τ (.bin .sub lo' (.intLit 1)) = some (.s (.int (l0 - 1))) := by
    rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
  rw [P_evalIdxs_slice1, P_evalBound_some_int _ _ _ _ hst, P_evalBound_some_int _ _ _ _ hwh,
    P_evalBound_some_int _ _ _ _ (P_eval_intLit _ _ s)]
  simp

/-- FNP-R0047 with a positive run-time step: `a(lo:hi:st)` ↦ `a[_f_rstep_any(lo', hi', st', lb')]`
(edges may be omitted). -/
theorem FNP_R0047_sound_pos (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lb' : P.Expr)
    (lo hi : Option F.Expr) (lo' hi' : Option P.Expr) (st : F.Expr) (st' : P.Expr)
    (hdecl : Rank1Decl o Γ a lbE) (hlb : SoundScalar o Γ lbE lb')
    (hlo : ∀ e, lo = some e → ∃ e', lo' = some e' ∧ SoundScalar o Γ e e') (hlo' : lo = none → lo' = none)
    (hhi : ∀ e, hi = some e → ∃ e', hi' = some e' ∧ SoundScalar o Γ e e') (hhi' : hi = none → hi' = none)
    (hst : SoundScalar o Γ st st') (hsti : IntTyped o Γ st)
    (hpos : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k s, F.eval o.F σ st = some (.s (.int k s)) → 0 < s)
    (hrange : SectionInRange o Γ a (lo.elim .omitted .given) (hi.elim .omitted .given)) :
    SoundPair o Γ (.ref a (.cons (.triplet (lo.elim .omitted .given) (hi.elim .omitted .given) (.given st)) .nil))
      (.index a (.cons (.scalar (Tr.call4 .f_rstep_any (lo'.elim .none id) (hi'.elim .none id) st' lb')) .nil)) := by
  refine section_pos_sound o Γ a lbE _ _ _ _ hdecl ?_ hrange
  intro σ τ hR hC l hl bl bh bs hbl hbh hbs
  obtain ⟨ks, sv, hvs, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbs
  have hspos := hpos σ τ hR hC ks sv hvs
  obtain ⟨ws, hws, hrws⟩ := hst σ τ hR hC _ hvs
  rw [RelS.int_eq hrws] at hws
  obtain ⟨wb, hwb, hrwb⟩ := hlb σ τ hR hC _ hl
  rw [RelS.int_eq hrwb] at hwb
  refine ⟨by simpa using hspos, ?_⟩
  -- the edge arguments evaluate to `None` or the int
  have hlo_eval : P.eval o.P τ (lo'.elim .none id) = some (.s (optIntVal bl)) := by
    cases lo with
    | none =>
      rw [hlo' rfl]; simp only [Option.elim] at hbl ⊢
      rw [F_evalBound_omitted] at hbl; cases hbl; rfl
    | some e =>
      obtain ⟨e', he', hse⟩ := hlo e rfl
      subst he'; simp only [Option.elim, id] at hbl ⊢
      obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
      obtain ⟨wl, hwl, hrwl⟩ := hse σ τ hR hC _ hvl
      rw [RelS.int_eq hrwl] at hwl
      exact hwl
  have hhi_eval : P.eval o.P τ (hi'.elim .none id) = some (.s (optIntVal bh)) := by
    cases hi with
    | none =>
      rw [hhi' rfl]; simp only [Option.elim] at hbh ⊢
      rw [F_evalBound_omitted] at hbh; cases hbh; rfl
    | some e =>
      obtain ⟨e', he', hse⟩ := hhi e rfl
      subst he'; simp only [Option.elim, id] at hbh ⊢
      obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
      obtain ⟨wh, hwh, hrwh⟩ := hse σ τ hR hC _ hvh
      rw [RelS.int_eq hrwh] at hwh
      exact hwh
  have hcall : P.eval o.P τ (Tr.call4 .f_rstep_any (lo'.elim .none id) (hi'.elim .none id) st' lb') =
      some (.slice (bl.map (· - l)) (bh.map (· - l + 1)) (some sv)) := by
    have hneg : ¬ (sv < 0) := by omega
    rw [P_eval_call4, hlo_eval, hhi_eval, hws, hwb]
    simp only [Option.bind_some]
    exact helper_rstep_any_pos o.P bl bh sv l hneg
  rw [P_evalIdxs_scalar1, hcall]
  rfl

end Recast.Rules
