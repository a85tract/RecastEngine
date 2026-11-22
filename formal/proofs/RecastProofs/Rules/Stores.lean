import RecastProofs.Rules.Obligations
import RecastProofs.Rules.Reductions

/-!
# Rules FNP-R0114 (section assignment) and FNP-R0112 (whole-array assignment
with an array RHS) on unit-origin rank-1 arrays

The store rules mirror the read rules of `Sections.lean`: the Fortran section
`A(lo:hi)` selects `Fort.sectionCoords lo hi 1`, the emitted `A[lo' - 1:hi']`
selects `Py.sliceIndices n (some (lo - 1)) (some hi) 1`; both store the k-th
RHS element at the k-th selected position (`F.storeArr` pairs `targets` with
`allCoords b`, `P.storeArr` pairs `coords` with `rs_shape_coords`).
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### List facts -/

theorem flatten_map_singleton_singleton {α : Type} (l : List α) :
    (l.map fun x => [[x]]).flatten = l.map fun x => [x] := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [ih]

/-- `find?` over an injectively keyed enumeration of `range' s m` finds position `k`. -/
theorem find?_range'_key {β γ : Type} [DecidableEq β] (key : ℕ → β) (val : ℕ → γ) :
    ∀ (m s k : ℕ), s ≤ k → k < s + m →
    (∀ i j, s ≤ i → i < s + m → s ≤ j → j < s + m → key i = key j → i = j) →
    ((List.range' s m).map fun i => (key i, val i)).find? (fun q => q.1 = key k) = some (key k, val k) := by
  intro m
  induction m with
  | zero => intro s k h1 h2; omega
  | succ m ih =>
    intro s k h1 h2 hinj
    rw [List.range'_succ, List.map_cons]
    by_cases hsk : s = k
    · subst hsk
      rw [List.find?_cons_of_pos]
      simp
    · rw [List.find?_cons_of_neg]
      · exact ih (s + 1) k (by omega) (by omega) (fun i j hi hi' hj hj' h => hinj i j (by omega) (by omega) (by omega) (by omega) h)
      · simp only [decide_eq_true_eq]
        intro h
        exact hsk (hinj s k (le_refl _) (by omega) h1 h2 h)

theorem find?_range_key {β γ : Type} [DecidableEq β] (m k : ℕ) (hk : k < m) (key : ℕ → β) (val : ℕ → γ)
    (hinj : ∀ i j, i < m → j < m → key i = key j → i = j) :
    ((List.range m).map fun i => (key i, val i)).find? (fun q => q.1 = key k) = some (key k, val k) := by
  rw [List.range_eq_range']
  exact find?_range'_key key val m 0 k (Nat.zero_le _) (by omega)
    (fun i j _ hi _ hj h => hinj i j (by omega) (by omega) h)

theorem find?_range_none {β γ : Type} [DecidableEq β] (m : ℕ) (key : ℕ → β) (val : ℕ → γ) (x : β)
    (hx : ∀ i, i < m → key i ≠ x) :
    ((List.range m).map fun i => (key i, val i)).find? (fun q => q.1 = x) = none := by
  rw [List.find?_eq_none]
  intro q hq
  rw [List.mem_map] at hq
  obtain ⟨i, hi, rfl⟩ := hq
  rw [List.mem_range] at hi
  simpa using hx i hi

/-! ### Slice membership -/

theorem sliceIndices_unit_mem (n : ℕ) (s e : ℤ) (h0 : 0 ≤ s) (hsn : s ≤ n) (he : 0 ≤ e) (hen : e ≤ n) (j : ℕ) :
    j ∈ Py.sliceIndices n (some s) (some e) 1 ↔ s ≤ (j : ℤ) ∧ (j : ℤ) < e := by
  rw [sliceIndices_unit n s e h0 hsn he hen]
  simp only [List.mem_map, List.mem_range]
  constructor
  · rintro ⟨k, hk, rfl⟩
    split_ifs at hk with hlt <;> omega
  · rintro ⟨h1, h2⟩
    refine ⟨(j - s).toNat, ?_, ?_⟩
    · split_ifs <;> omega
    · omega

/-! ### Rank-1 array facts -/

theorem extents_singleton (a : FArr) (m : ℕ) (h : a.extents = [m]) : ∃ lb, a.dims = [(lb, m)] := by
  rcases hd : a.dims with _ | ⟨⟨lb, m'⟩, _ | _⟩ <;> simp [FArr.extents, hd] at h
  exact ⟨lb, by simp [hd, h]⟩

theorem sectionCoords_unit' (lo hi : ℤ) :
    Fort.sectionCoords lo hi 1 = List.map (fun (k : ℕ) => lo + (k : ℤ)) (List.range (Fort.tripCount lo hi 1)) := by
  unfold Fort.sectionCoords; simp

theorem find?_section_pairs (lo lbb : ℤ) (tc k : ℕ) (hk : k < tc) :
    ((List.range tc).map fun (i : ℕ) => ([lo + (i : ℤ)], [lbb + (i : ℤ)])).find?
      (fun q => decide (q.1 = [lo + (k : ℤ)])) = some ([lo + (k : ℤ)], [lbb + (k : ℤ)]) :=
  find?_range_key tc k hk (fun i => [lo + (i : ℤ)]) (fun i => [lbb + (i : ℤ)])
    (by intro i j _ _ h; simp only [List.cons.injEq, and_true] at h; omega)

theorem find?_section_pairs_none (lo lbb : ℤ) (tc : ℕ) (i : ℤ) (hi : ¬ ∃ k : ℕ, k < tc ∧ i = lo + k) :
    ((List.range tc).map fun (i : ℕ) => ([lo + (i : ℤ)], [lbb + (i : ℤ)])).find?
      (fun q => decide (q.1 = [i])) = none :=
  find?_range_none tc (fun i => [lo + (i : ℤ)]) (fun i => [lbb + (i : ℤ)]) [i]
    (by intro k hk h; simp only [List.cons.injEq, and_true] at h; exact hi ⟨k, hk, h.symm⟩)

theorem find?_slice_pairs (s : ℤ) (h0 : 0 ≤ s) (m k : ℕ) (hk : k < m) :
    ((List.range m).map fun (i : ℕ) => ([(s + (i : ℤ)).toNat], [i])).find?
      (fun q => decide (q.1 = [(s + (k : ℤ)).toNat])) = some ([(s + (k : ℤ)).toNat], [k]) :=
  find?_range_key m k hk (fun i => [(s + (i : ℤ)).toNat]) (fun i => [i])
    (by intro i j _ _ h; simp only [List.cons.injEq, and_true] at h; omega)

theorem find?_slice_pairs_none (s : ℤ) (h0 : 0 ≤ s) (m : ℕ) (j : ℕ) (hj : ¬ ∃ k : ℕ, k < m ∧ j = (s + k).toNat) :
    ((List.range m).map fun (i : ℕ) => ([(s + (i : ℤ)).toNat], [i])).find?
      (fun q => decide (q.1 = [j])) = none :=
  find?_range_none m (fun i => [(s + (i : ℤ)).toNat]) (fun i => [i]) [j]
    (by intro k hk h; simp only [List.cons.injEq, and_true] at h; exact hj ⟨k, hk, h.symm⟩)

theorem rs_shape_coords_single (m : ℕ) :
    P.storeArr.rs_shape_coords [m] = (List.range m).map fun i => [i] := by
  simp [P.storeArr.rs_shape_coords, flatten_map_singleton_singleton]

theorem sliceIndices_none_none (n : ℕ) : Py.sliceIndices n none none 1 = List.range n := by
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

/-! ### Fortran stores through a unit-step section (edges possibly omitted) -/

theorem F_storeArr_range_oob (d : F.Decl) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (lo hi : Option ℤ)
    (hout : ¬ ∀ i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1, lb ≤ i ∧ i < lb + n) (rhs : F.Val) :
    F.storeArr d a [.range lo hi none] rhs = none := by
  simp [F.storeArr, F.resolveAxis, hd, hout]

theorem F_storeArr_range_scalar (d : F.Decl) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (lo hi : Option ℤ)
    (hin : ∀ i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1, lb ≤ i ∧ i < lb + n)
    (v cv : SVal) (hcv : F.convertTo d v = some cv) :
    ∃ a' : FArr, F.storeArr d a [.range lo hi none] (.s v) = some a' ∧ a'.elemTy = a.elemTy ∧
      a'.dims = a.dims ∧
      (∀ i : ℤ, i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1 → a'.get [i] = cv) ∧
      (∀ i : ℤ, i ∉ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1 → a'.get [i] = a.get [i]) := by
  have hin' : (∀ x ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1, lb ≤ x ∧ x < lb + (n : ℤ)) = True :=
    eq_true hin
  cases hst : F.storeArr d a [.range lo hi none] (.s v) with
  | none => exfalso; simp [F.storeArr, F.resolveAxis, hd, hin', hcv] at hst
  | some a' =>
    simp [F.storeArr, F.resolveAxis, hd, hin', hcv] at hst
    subst hst
    refine ⟨_, rfl, rfl, hd.symm, fun i hmem => ?_, fun i hmem => ?_⟩
    · simp [hmem]
    · simp [hmem]

/-- Array RHS: from a defined store, the RHS is a rank-1 array of the section's
length, every element converts, and the k-th converted element lands at the k-th
selected position. -/
theorem F_storeArr_range_array (d : F.Decl) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (lo hi : Option ℤ)
    (hin : ∀ i ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1, lb ≤ i ∧ i < lb + n)
    (b a' : FArr) (hst : F.storeArr d a [.range lo hi none] (.arr b) = some a') :
    ∃ lbb : ℤ, b.dims = [(lbb, Fort.tripCount (lo.getD lb) (hi.getD (lb + n - 1)) 1)] ∧
      a'.elemTy = a.elemTy ∧ a'.dims = a.dims ∧
      (∀ k : ℕ, k < Fort.tripCount (lo.getD lb) (hi.getD (lb + n - 1)) 1 →
        ∃ cv, F.convertTo d (b.get [lbb + k]) = some cv ∧ a'.get [lo.getD lb + k] = cv) ∧
      (∀ i : ℤ, i ∉ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1 → a'.get [i] = a.get [i]) := by
  have hin' : (∀ x ∈ Fort.sectionCoords (lo.getD lb) (hi.getD (lb + n - 1)) 1, lb ≤ x ∧ x < lb + (n : ℤ)) = True :=
    eq_true hin
  simp [F.storeArr, F.resolveAxis, hd, hin'] at hst
  obtain ⟨hext, hdef, hst⟩ := hst
  rw [sectionCoords_length] at hext
  obtain ⟨lbb, hb⟩ := extents_singleton b _ hext.symm
  rw [allCoords_rank1 b lbb _ hb, sectionCoords_unit', flatten_map_singleton_singleton, List.map_map,
    List.zip_map'] at hdef hst
  simp only [Function.comp_def] at hdef hst
  subst hst
  refine ⟨lbb, hb, rfl, hd.symm, fun k hk => ?_, fun i hi => ?_⟩
  · have hmem : ([lo.getD lb + (k : ℤ)], [lbb + (k : ℤ)]) ∈
        (List.range (Fort.tripCount (lo.getD lb) (hi.getD (lb + n - 1)) 1)).map
          (fun (i : ℕ) => ([lo.getD lb + (i : ℤ)], [lbb + (i : ℤ)])) :=
      List.mem_map.mpr ⟨k, List.mem_range.mpr hk, rfl⟩
    have hne := hdef _ _ hmem
    obtain ⟨cv, hcv⟩ := Option.ne_none_iff_exists'.mp hne
    refine ⟨cv, hcv, ?_⟩
    dsimp only
    rw [find?_section_pairs _ _ _ _ hk, Option.map_some, hcv]
  · dsimp only
    rw [find?_section_pairs_none, Option.map_none]
    rw [sectionCoords_unit'] at hi
    intro ⟨k, hk, hik⟩
    exact hi (List.mem_map.mpr ⟨k, List.mem_range.mpr hk, hik.symm⟩)

/-! ### Python stores through a unit slice (edges possibly `None`) -/

theorem P_storeArr_slice_scalar (p : PArr) (n : ℕ) (hs : p.shape = [n]) (s e : Option ℤ)
    (hs' : ∀ x, s = some x → 0 ≤ x ∧ x ≤ n) (he' : ∀ x, e = some x → 0 ≤ x ∧ x ≤ n) (w : PVal) :
    ∃ p' : PArr, P.storeArr p [.sl s e 1] (.s w) = some p' ∧ p'.shape = p.shape ∧
      (∀ j : ℕ, j ∈ Py.sliceIndices n s e 1 → p'.get [j] = w) ∧
      (∀ j : ℕ, j ∉ Py.sliceIndices n s e 1 → p'.get [j] = p.get [j]) := by
  cases hst : P.storeArr p [.sl s e 1] (.s w) with
  | none => exfalso; simp [P.storeArr, P.selection, hs] at hst
  | some p' =>
    simp [P.storeArr, P.selection, hs] at hst
    subst hst
    refine ⟨_, rfl, hs.symm, fun j hmem => ?_, fun j hmem => ?_⟩
    · simp [hmem]
    · simp [hmem]

theorem P_storeArr_slice_array (p : PArr) (n : ℕ) (hs : p.shape = [n]) (s e : Option ℤ)
    (hs' : ∀ x, s = some x → 0 ≤ x ∧ x ≤ n) (he' : ∀ x, e = some x → 0 ≤ x ∧ x ≤ n) (b : PArr)
    (hb : b.shape = [Py.sliceLen (s.getD 0) (e.getD n) 1]) :
    ∃ p' : PArr, P.storeArr p [.sl s e 1] (.arr b) = some p' ∧ p'.shape = p.shape ∧
      (∀ k : ℕ, k < Py.sliceLen (s.getD 0) (e.getD n) 1 → p'.get [(s.getD 0 + k).toNat] = b.get [k]) ∧
      (∀ j : ℕ, j ∉ Py.sliceIndices n s e 1 → p'.get [j] = p.get [j]) := by
  have hL := sliceIndices_pos n s e 1 (by norm_num) hs' he'
  simp only [mul_one] at hL
  have h0 : 0 ≤ s.getD 0 := by cases s with | none => simp | some x => exact (hs' x rfl).1
  cases hst : P.storeArr p [.sl s e 1] (.arr b) with
  | none => exfalso; simp [P.storeArr, P.selection, hs, hL, hb] at hst
  | some p' =>
    simp [P.storeArr, P.selection, hs, hL, hb] at hst
    subst hst
    refine ⟨_, rfl, hs.symm, fun k hk => ?_, fun j hmem => ?_⟩
    · dsimp only
      rw [rs_shape_coords_single, ← List.map_map, flatten_map_singleton_singleton, List.map_map, List.zip_map']
      simp only [Function.comp_def]
      rw [find?_slice_pairs _ h0 _ _ hk]
    · dsimp only
      rw [rs_shape_coords_single, ← List.map_map, flatten_map_singleton_singleton, List.map_map, List.zip_map']
      simp only [Function.comp_def]
      rw [find?_slice_pairs_none _ h0]
      rw [hL] at hmem
      intro ⟨k, hk, hjk⟩
      exact hmem (List.mem_map.mpr ⟨k, List.mem_range.mpr hk, hjk.symm⟩)

/-! ### Auxiliary facts -/

theorem convertTo_ty (d : F.Decl) (v cv : SVal) (h : F.convertTo d v = some cv) : cv.ty = d.ty := by
  cases v <;> cases hd : d.ty <;> simp [F.convertTo, hd] at h <;> (try subst h) <;> simp [SVal.ty]

theorem sliceLen_zero_n (n : ℕ) : Py.sliceLen 0 n 1 = n := by
  simp only [Py.sliceLen]
  split_ifs <;> omega

theorem sliceLen_unit_eq_tripCount (l0 h0 : ℤ) : Py.sliceLen (l0 - 1) h0 1 = Fort.tripCount l0 h0 1 := by
  have := tripCount_eq_sliceLen l0 h0 1 1 one_pos
  simpa using this

theorem tripCount_full (l : ℤ) (n : ℕ) : Fort.tripCount l (l + n - 1) 1 = n := by
  rw [Fort.tripCount_unit]; omega

/-- The rank-1 index of a conforming array as a singleton coordinate. -/
theorem inBounds_rank1_elim (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (idx : List ℤ)
    (hidx : a.inBounds idx) : ∃ j, idx = [j] ∧ lb ≤ j ∧ j < lb + n := by
  simp only [FArr.inBounds, FArr.inBoundsB, hd] at hidx
  match idx, hidx with
  | [j], hidx =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
    exact ⟨j, rfl, hidx.1⟩
  | [], hidx => simp [FArr.inBoundsB] at hidx
  | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx

/-! ### FNP-R0114: section assignment `A(lo:hi) = e` ↦ `A[lo' - 1:hi'] = e'` (unit origin, rank 1) -/

/-- Scalar (broadcast) RHS. -/
theorem FNP_R0114_sound_scalar (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (k : IKind) (hx : Γ a = some d)
    (hlbs : d.lbs = [.intLit k 1]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a (.intLit k 1))
    (lo hi : F.Expr) (lo' hi' : P.Expr) (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hrange : SectionInRange o Γ a (.given lo) (.given hi))
    (e : F.Expr) (e' : P.Expr) (he : SoundScalar o Γ e e') (hcompat : ScalarCompat o Γ d e)
    (hscalar : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ sv, v = .s sv) :
    StmtSound o Γ (.assign a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil) e)
      (.assignIndex a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') .none) .nil) e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    rw [F_eval_intLit] at hl
    simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
    obtain ⟨_, rfl⟩ := hl
    obtain ⟨sv, rfl⟩ := hscalar σ τ hR hC v hv
    rw [F_assign_array_subs o.F Γ σ a d av _ _ _ hx hσ, F_evalSubs_triplet1] at hassign
    simp only [Option.bind_eq_some_iff] at hassign
    obtain ⟨axes, ⟨bl, hbl, bh, hbh, bs, hbs, hax⟩, hrest⟩ := hassign
    simp only [Option.some.injEq] at hax
    subst hax
    obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
    obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
    rw [F_evalBound_omitted] at hbs
    cases hbs
    simp only [Option.map_eq_some_iff] at hrest
    obtain ⟨a', hst, hσ'⟩ := hrest
    subst hσ'
    obtain ⟨hlo_rng, hhi_rng⟩ := hrange σ τ hR hC av 1 n hσ hd (some l0) (some h0) hbl hbh
    have hl0 := hlo_rng l0 rfl
    have hh0 := hhi_rng h0 rfl
    by_cases hin : ∀ i ∈ Fort.sectionCoords l0 h0 1, 1 ≤ i ∧ i < 1 + n
    · obtain ⟨w, hw, hrw⟩ := he σ τ hR hC sv hv
      obtain ⟨cv, hcv, hrelc, hty⟩ := convertTo_compat_rel d sv w hrw (hcompat σ τ hR hC sv hv)
      obtain ⟨a'', hst', hty', hdims', hget', hother'⟩ :=
        F_storeArr_range_scalar d av 1 n hd (some l0) (some h0) hin sv cv hcv
      rw [hst'] at hst
      cases hst
      simp only [Option.getD_some] at hget' hother'
      have hshape := RelA_dims1 hrel 1 n hd
      have hs1 : ∀ x, some (l0 - 1) = some x → 0 ≤ x ∧ x ≤ n := by intro x hx; cases hx; omega
      have he1 : ∀ x, some h0 = some x → 0 ≤ x ∧ x ≤ n := by intro x hx; cases hx; omega
      obtain ⟨p', hpst, hpshape, hpget, hpother⟩ :=
        P_storeArr_slice_scalar p n hshape (some (l0 - 1)) (some h0) hs1 he1 w
      obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
      obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
      rw [RelS.int_eq hrwl] at hwl
      rw [RelS.int_eq hrwh] at hwh
      have hmemP : ∀ j : ℕ, j ∈ Py.sliceIndices n (some (l0 - 1)) (some h0) 1 ↔ l0 - 1 ≤ (j : ℤ) ∧ (j : ℤ) < h0 :=
        fun j => sliceIndices_unit_mem n (l0 - 1) h0 (by omega) (by omega) (by omega) (by omega) j
      refine ⟨Function.update τ a (some (.arr p')), ?_, ?_, ?_⟩
      · refine P.Exec.assignIndex hw hτ ?_ hpst
        have hs : P.eval o.P τ (.bin .sub lo' (.intLit 1)) = some (.s (.int (l0 - 1))) := by
          rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
        rw [P_evalIdxs_slice1, P_evalBound_none, P_evalBound_some_int _ _ _ _ hs, P_evalBound_some_int _ _ _ _ hwh]
        rfl
      · apply relState_update hR a
        refine RelV.arr ⟨by rw [hpshape, hrel.shape]; simp [FArr.extents, hdims'], ?_⟩
        intro idx hidx
        obtain ⟨j, rfl, hj⟩ := inBounds_rank1_elim _ 1 n (hdims'.trans hd) idx hidx
        simp only [rebase, FArr.lbs, hdims', hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
          List.zipWith_nil_right]
        by_cases hjin : l0 ≤ j ∧ j ≤ h0
        · rw [hget' j ((sectionCoords_mem l0 h0 j).mpr hjin), hpget (j - 1).toNat ((hmemP _).mpr (by omega))]
          exact hrelc
        · rw [hother' j (fun hm => hjin ((sectionCoords_mem l0 h0 j).mp hm)),
            hpother (j - 1).toNat (fun hm => hjin (by have := (hmemP _).mp hm; omega))]
          exact RelA_get1 hrel 1 n hd j hj
      · obtain ⟨av₀, hσ₀, hty_av, _, _, hwt_av⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
        rw [hσ] at hσ₀; cases hσ₀
        refine conforms_update_array hC hS a d hx av _ hσ hty' hdims' ?_
        refine wellTyped_rank1_of _ av 1 n hd hdims' hty' (fun j hj => ?_)
        by_cases hjin : l0 ≤ j ∧ j ≤ h0
        · rw [hget' j ((sectionCoords_mem l0 h0 j).mpr hjin), hty, hty_av]
        · rw [hother' j (fun hm => hjin ((sectionCoords_mem l0 h0 j).mp hm))]
          exact hwt_av [j] (inBounds_rank1 av 1 n hd j hj)
    · exfalso
      rw [F_storeArr_range_oob d av 1 n hd (some l0) (some h0) hin] at hst
      cases hst

/-- Array RHS of the section's extent. -/
theorem FNP_R0114_sound_array (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (k : IKind) (hx : Γ a = some d)
    (hlbs : d.lbs = [.intLit k 1]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a (.intLit k 1))
    (lo hi : F.Expr) (lo' hi' : P.Expr) (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hrange : SectionInRange o Γ a (.given lo) (.given hi))
    (e : F.Expr) (e' : P.Expr) (he : SoundPair o Γ e e')
    (harr : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ bv, v = .arr bv)
    (hcompat : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ bv, F.eval o.F σ e = some (.arr bv) →
      ∀ idx, bv.inBounds idx → Compat d.ty (bv.get idx)) :
    StmtSound o Γ (.assign a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil) e)
      (.assignIndex a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') .none) .nil) e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    rw [F_eval_intLit] at hl
    simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
    obtain ⟨_, rfl⟩ := hl
    obtain ⟨bv, rfl⟩ := harr σ τ hR hC v hv
    rw [F_assign_array_subs o.F Γ σ a d av _ _ _ hx hσ, F_evalSubs_triplet1] at hassign
    simp only [Option.bind_eq_some_iff] at hassign
    obtain ⟨axes, ⟨bl, hbl, bh, hbh, bs, hbs, hax⟩, hrest⟩ := hassign
    simp only [Option.some.injEq] at hax
    subst hax
    obtain ⟨k₁, l0, hvl, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbl
    obtain ⟨k₂, h0, hvh, rfl⟩ := F_evalBound_given_eq_some _ _ _ _ hbh
    rw [F_evalBound_omitted] at hbs
    cases hbs
    simp only [Option.map_eq_some_iff] at hrest
    obtain ⟨a', hst, hσ'⟩ := hrest
    subst hσ'
    obtain ⟨hlo_rng, hhi_rng⟩ := hrange σ τ hR hC av 1 n hσ hd (some l0) (some h0) hbl hbh
    have hl0 := hlo_rng l0 rfl
    have hh0 := hhi_rng h0 rfl
    by_cases hin : ∀ i ∈ Fort.sectionCoords l0 h0 1, 1 ≤ i ∧ i < 1 + n
    · obtain ⟨lbb, hbd, hty', hdims', hget', hother'⟩ :=
        F_storeArr_range_array d av 1 n hd (some l0) (some h0) hin bv a' hst
      simp only [Option.getD_some] at hbd hget' hother'
      obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hv
      cases hrw with
      | arr hrelb =>
        rename_i pb
        have hshape := RelA_dims1 hrel 1 n hd
        have hpbshape := RelA_dims1 hrelb lbb _ hbd
        have hs1 : ∀ x, some (l0 - 1) = some x → 0 ≤ x ∧ x ≤ n := by intro x hx; cases hx; omega
        have he1 : ∀ x, some h0 = some x → 0 ≤ x ∧ x ≤ n := by intro x hx; cases hx; omega
        obtain ⟨p', hpst, hpshape, hpget, hpother⟩ :=
          P_storeArr_slice_array p n hshape (some (l0 - 1)) (some h0) hs1 he1 pb
            (by rw [hpbshape]; simp only [Option.getD_some]; rw [sliceLen_unit_eq_tripCount])
        simp only [Option.getD_some] at hpget
        obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hvl
        obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hvh
        rw [RelS.int_eq hrwl] at hwl
        rw [RelS.int_eq hrwh] at hwh
        have hmemP : ∀ j : ℕ, j ∈ Py.sliceIndices n (some (l0 - 1)) (some h0) 1 ↔ l0 - 1 ≤ (j : ℤ) ∧ (j : ℤ) < h0 :=
          fun j => sliceIndices_unit_mem n (l0 - 1) h0 (by omega) (by omega) (by omega) (by omega) j
        -- the k-th element of the section is related to the k-th element of the RHS
        have hsec : ∀ kk : ℕ, kk < Fort.tripCount l0 h0 1 →
            ∃ cv, a'.get [l0 + kk] = cv ∧ cv.ty = d.ty ∧ RelS cv (pb.get [kk]) := by
          intro kk hkk
          obtain ⟨cv, hcv, hget⟩ := hget' kk hkk
          have hb_in : lbb ≤ lbb + kk ∧ lbb + kk < lbb + Fort.tripCount l0 h0 1 := ⟨by omega, by omega⟩
          have hrel_el := RelA_get1 hrelb lbb _ hbd (lbb + kk) hb_in
          have hidx2 : (lbb + kk - lbb).toNat = kk := by omega
          rw [hidx2] at hrel_el
          have hcomp := hcompat σ τ hR hC bv hv [lbb + kk] (inBounds_rank1 bv lbb _ hbd _ hb_in)
          obtain ⟨cv', hcv', hrelc, hty⟩ := convertTo_compat_rel d _ _ hrel_el hcomp
          rw [hcv] at hcv'
          cases hcv'
          exact ⟨cv, hget, hty, hrelc⟩
        refine ⟨Function.update τ a (some (.arr p')), ?_, ?_, ?_⟩
        · refine P.Exec.assignIndex hw hτ ?_ hpst
          have hs : P.eval o.P τ (.bin .sub lo' (.intLit 1)) = some (.s (.int (l0 - 1))) := by
            rw [P_eval_bin, hwl]; simp [P.eval, P.lift2, P.arith]
          rw [P_evalIdxs_slice1, P_evalBound_none, P_evalBound_some_int _ _ _ _ hs, P_evalBound_some_int _ _ _ _ hwh]
          rfl
        · apply relState_update hR a
          refine RelV.arr ⟨by rw [hpshape, hrel.shape]; simp [FArr.extents, hdims'], ?_⟩
          intro idx hidx
          obtain ⟨j, rfl, hj⟩ := inBounds_rank1_elim _ 1 n (hdims'.trans hd) idx hidx
          simp only [rebase, FArr.lbs, hdims', hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
            List.zipWith_nil_right]
          by_cases hjin : l0 ≤ j ∧ j ≤ h0
          · obtain ⟨kk, hkk, rfl⟩ : ∃ kk : ℕ, kk < Fort.tripCount l0 h0 1 ∧ j = l0 + kk :=
              ⟨(j - l0).toNat, by rw [Fort.tripCount_unit]; omega, by omega⟩
            obtain ⟨cv, hget, _, hrelc⟩ := hsec kk hkk
            rw [hget]
            have hidx : (l0 + (kk : ℤ) - 1).toNat = (l0 - 1 + (kk : ℤ)).toNat := by omega
            rw [hidx, hpget kk (by rw [sliceLen_unit_eq_tripCount]; exact hkk)]
            exact hrelc
          · rw [hother' j (fun hm => hjin ((sectionCoords_mem l0 h0 j).mp hm)),
              hpother (j - 1).toNat (fun hm => hjin (by have := (hmemP _).mp hm; omega))]
            exact RelA_get1 hrel 1 n hd j hj
        · obtain ⟨av₀, hσ₀, hty_av, _, _, hwt_av⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
          rw [hσ] at hσ₀; cases hσ₀
          refine conforms_update_array hC hS a d hx av _ hσ hty' hdims' ?_
          refine wellTyped_rank1_of _ av 1 n hd hdims' hty' (fun j hj => ?_)
          by_cases hjin : l0 ≤ j ∧ j ≤ h0
          · obtain ⟨kk, hkk, rfl⟩ : ∃ kk : ℕ, kk < Fort.tripCount l0 h0 1 ∧ j = l0 + kk :=
              ⟨(j - l0).toNat, by rw [Fort.tripCount_unit]; omega, by omega⟩
            obtain ⟨cv, hget, hty, _⟩ := hsec kk hkk
            rw [hget, hty, hty_av]
          · rw [hother' j (fun hm => hjin ((sectionCoords_mem l0 h0 j).mp hm))]
            exact hwt_av [j] (inBounds_rank1 av 1 n hd j hj)
    · exfalso
      rw [F_storeArr_range_oob d av 1 n hd (some l0) (some h0) hin] at hst
      cases hst

/-! ### FNP-R0112: whole-array assignment with a same-shape rank-1 array RHS -/

theorem FNP_R0112_sound_array (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (hx : Γ a = some d)
    (lbE : F.Expr) (hlbs : d.lbs = [lbE]) (hS : StaticBounds Γ) (hdecl : Rank1Decl o Γ a lbE)
    (e : F.Expr) (e' : P.Expr) (he : SoundPair o Γ e e')
    (harr : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ bv, v = .arr bv)
    (hcompat : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ bv, F.eval o.F σ e = some (.arr bv) →
      ∀ idx, bv.inBounds idx → Compat d.ty (bv.get idx)) :
    StmtSound o Γ (.assign a .nil e) (.assignAll a e') := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    obtain ⟨bv, rfl⟩ := harr σ τ hR hC v hv
    rw [F_assign_array_nil o.F Γ σ a d av _ hx hσ, hd] at hassign
    simp only [List.map_cons, List.map_nil, Option.map_eq_some_iff] at hassign
    obtain ⟨a', hst, hσ'⟩ := hassign
    subst hσ'
    have hin : ∀ i ∈ Fort.sectionCoords ((none : Option ℤ).getD l) ((none : Option ℤ).getD (l + n - 1)) 1,
        l ≤ i ∧ i < l + n := by
      intro i hi
      simp only [Option.getD_none] at hi
      rw [sectionCoords_mem] at hi
      omega
    obtain ⟨lbb, hbd, hty', hdims', hget', hother'⟩ := F_storeArr_range_array d av l n hd none none hin bv a' hst
    simp only [Option.getD_none, tripCount_full] at hbd hget'
    obtain ⟨w, hw, hrw⟩ := he σ τ hR hC _ hv
    cases hrw with
    | arr hrelb =>
      rename_i pb
      have hshape := RelA_dims1 hrel l n hd
      have hpbshape := RelA_dims1 hrelb lbb _ hbd
      obtain ⟨p', hpst, hpshape, hpget, hpother⟩ :=
        P_storeArr_slice_array p n hshape none none (fun _ h => by cases h) (fun _ h => by cases h) pb
          (by rw [hpbshape]; simp only [Option.getD_none]; rw [sliceLen_zero_n])
      simp only [Option.getD_none, sliceLen_zero_n, zero_add, Int.toNat_natCast] at hpget
      have hsec : ∀ kk : ℕ, kk < n → ∃ cv, a'.get [l + kk] = cv ∧ cv.ty = d.ty ∧ RelS cv (pb.get [kk]) := by
        intro kk hkk
        obtain ⟨cv, hcv, hget⟩ := hget' kk hkk
        have hb_in : lbb ≤ lbb + kk ∧ lbb + kk < lbb + n := ⟨by omega, by omega⟩
        have hrel_el := RelA_get1 hrelb lbb _ hbd (lbb + kk) hb_in
        have hidx2 : (lbb + kk - lbb).toNat = kk := by omega
        rw [hidx2] at hrel_el
        have hcomp := hcompat σ τ hR hC bv hv [lbb + kk] (inBounds_rank1 bv lbb _ hbd _ hb_in)
        obtain ⟨cv', hcv', hrelc, hty⟩ := convertTo_compat_rel d _ _ hrel_el hcomp
        rw [hcv] at hcv'
        cases hcv'
        exact ⟨cv, hget, hty, hrelc⟩
      refine ⟨Function.update τ a (some (.arr p')), ?_, ?_, ?_⟩
      · refine P.Exec.assignAll hw hτ ?_
        rw [hshape]; simpa using hpst
      · apply relState_update hR a
        refine RelV.arr ⟨by rw [hpshape, hrel.shape]; simp [FArr.extents, hdims'], ?_⟩
        intro idx hidx
        obtain ⟨j, rfl, hj⟩ := inBounds_rank1_elim _ l n (hdims'.trans hd) idx hidx
        simp only [rebase, FArr.lbs, hdims', hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
          List.zipWith_nil_right]
        obtain ⟨kk, hkk, rfl⟩ : ∃ kk : ℕ, kk < n ∧ j = l + kk := ⟨(j - l).toNat, by omega, by omega⟩
        obtain ⟨cv, hget, _, hrelc⟩ := hsec kk hkk
        rw [hget]
        have hidx : (l + (kk : ℤ) - l).toNat = kk := by omega
        rw [hidx, hpget kk hkk]
        exact hrelc
      · obtain ⟨av₀, hσ₀, hty_av, _, _, hwt_av⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
        rw [hσ] at hσ₀; cases hσ₀
        refine conforms_update_array hC hS a d hx av _ hσ hty' hdims' ?_
        refine wellTyped_rank1_of _ av l n hd hdims' hty' (fun j hj => ?_)
        obtain ⟨kk, hkk, rfl⟩ : ∃ kk : ℕ, kk < n ∧ j = l + kk := ⟨(j - l).toNat, by omega, by omega⟩
        obtain ⟨cv, hget, hty, _⟩ := hsec kk hkk
        rw [hget, hty, hty_av]

end Recast.Rules
