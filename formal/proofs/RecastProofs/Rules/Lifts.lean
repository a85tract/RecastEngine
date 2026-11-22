import RecastProofs.Rules.Obligations
import RecastProofs.Rules.Reductions

/-!
# Elementwise lifts on rank-1 arrays: FNP-R0100 (elemental intrinsics over
arrays), FNP-R0025 (array `**` -> `_f_vpow`), FNP-R0035 (mask `&` / `|`)

`lift1_rel` / `lift2_rel_*`: `RelA` is preserved by `F.lift1`/`P.lift1` and
`F.lift2`/`P.lift2` when the scalar functions are related pointwise and every
source element is in the function's domain (the Fortran lift substitutes the
first element's value where an element fails, so source definedness is a
genuine premise, cf. `F.allDefined`).

FNP-R0100 as stated (every `Tr.arrayHelper` row) is refuted: the fallback rows
apply a scalar-only Python spelling to an array (`tan -> math.tan`, defect
FNP-D0036's class) and `anint -> np.round` inherits FNP-D0004 elementwise.
The genuinely vectorised rows (`VectorRow`) are proved sound.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Unfolding lemmas for the lifts -/

theorem F_lift1_arr_empty (f : SVal → Option SVal) (a : FArr) (h0 : a.size = 0) :
    F.lift1 f (.arr a) = some (.arr a) := by
  simp [F.lift1, h0]

theorem P_lift1_arr_empty (g : PVal → Option PVal) (p : PArr) (h0 : p.size = 0) :
    P.lift1 g (.arr p) = some (.arr p) := by
  simp [P.lift1, h0]

/-- A non-empty rank-1 array: the lift is the pointwise map (with the first
element's image as the fallback the model uses for failing elements). -/
theorem F_lift1_arr_pos (f : SVal → Option SVal) (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)])
    (hn : n ≠ 0) (sample : SVal) (hs : f (a.get [lb]) = some sample) :
    ∃ r : FArr, F.lift1 f (.arr a) = some (.arr r) ∧ r.elemTy = sample.ty ∧ r.dims = a.dims ∧
      ∀ idx, r.get idx = (f (a.get idx)).getD sample := by
  have h0 : a.size ≠ 0 := by rw [FArr_size_rank1 a lb n hd]; exact hn
  have hlbs : a.lbs = [lb] := by simp [FArr.lbs, hd]
  refine ⟨{ elemTy := sample.ty, dims := a.dims, get := fun idx => (f (a.get idx)).getD sample }, ?_, rfl, rfl,
    fun _ => rfl⟩
  simp [F.lift1, h0, hlbs, hs]

theorem P_lift1_arr_pos (g : PVal → Option PVal) (p : PArr) (n : ℕ) (hs : p.shape = [n]) (hn : n ≠ 0)
    (sample : PVal) (hg : g (p.get [0]) = some sample) :
    ∃ r : PArr, P.lift1 g (.arr p) = some (.arr r) ∧ r.shape = p.shape ∧
      ∀ i, r.get i = (g (p.get i)).getD sample := by
  have h0 : p.size ≠ 0 := by rw [PArr_size_rank1 p n hs]; exact hn
  have hz : p.shape.map (fun _ => (0 : ℕ)) = [0] := by rw [hs]; rfl
  refine ⟨{ shape := p.shape, get := fun i => (g (p.get i)).getD sample }, ?_, rfl, fun _ => rfl⟩
  simp [P.lift1, h0, hz, hg]

theorem F_lift1_arr_not_scalar (f : SVal → Option SVal) (a : FArr) (sv : SVal) :
    F.lift1 f (.arr a) ≠ some (.s sv) := by
  intro h
  by_cases h0 : a.size = 0
  · simp [F.lift1, h0] at h
  · cases hf : f (a.get a.lbs) <;> simp [F.lift1, h0, hf] at h

/-- Every in-bounds coordinate of a rank-1 array is `[j]` with `lb ≤ j < lb + n`. -/
theorem inBounds_rank1_iff (a : FArr) (lb : ℤ) (n : ℕ) (hd : a.dims = [(lb, n)]) (idx : List ℤ)
    (hidx : a.inBounds idx) : ∃ j, idx = [j] ∧ lb ≤ j ∧ j < lb + n := by
  simp only [FArr.inBounds, hd] at hidx
  match idx, hidx with
  | [j], hidx =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
    exact ⟨j, rfl, hidx.1⟩
  | [], hidx => simp [FArr.inBoundsB] at hidx
  | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx

/-! ### `RelA` is preserved by pointwise-related unary lifts (rank 1) -/

/-- Pointwise relation of two scalar functions. -/
def PointRel (f : SVal → Option SVal) (g : PVal → Option PVal) : Prop :=
  ∀ v w, RelS v w → ∀ v', f v = some v' → ∃ w', g w = some w' ∧ RelS v' w'

theorem lift1_rel (f : SVal → Option SVal) (g : PVal → Option PVal) (av : FArr) (p : PArr) (lb : ℤ) (n : ℕ)
    (hd : av.dims = [(lb, n)]) (hA : RelA av p) (hpt : PointRel f g)
    (hdef : ∀ idx, av.inBounds idx → (f (av.get idx)).isSome)
    (r : FArr) (hF : F.lift1 f (.arr av) = some (.arr r)) :
    ∃ r' : PArr, P.lift1 g (.arr p) = some (.arr r') ∧ RelA r r' := by
  have hsz := FArr_size_rank1 av lb n hd
  have hshape := RelA_dims1 hA lb n hd
  have hpsz := PArr_size_rank1 p n hshape
  by_cases hn : n = 0
  · subst hn
    rw [F_lift1_arr_empty f av hsz] at hF
    cases hF
    exact ⟨p, P_lift1_arr_empty g p hpsz, hA⟩
  · have hin : av.inBounds [lb] := inBounds_rank1 av lb n hd lb ⟨le_refl _, by omega⟩
    obtain ⟨sample, hs⟩ := Option.isSome_iff_exists.mp (hdef [lb] hin)
    obtain ⟨r₀, hF₀, hty₀, hdims₀, hget₀⟩ := F_lift1_arr_pos f av lb n hd hn sample hs
    rw [hF₀] at hF
    cases hF
    have hel : RelS (av.get [lb]) (p.get [0]) := by
      have := RelA_get1 hA lb n hd lb ⟨le_refl _, by omega⟩
      simpa using this
    obtain ⟨sample', hs', hrel⟩ := hpt _ _ hel sample hs
    obtain ⟨r', hP, hshape', hget'⟩ := P_lift1_arr_pos g p n hshape hn sample' hs'
    refine ⟨r', hP, ⟨by rw [hshape', hA.shape]; simp [FArr.extents, hdims₀], ?_⟩⟩
    intro idx hidx
    obtain ⟨j, rfl, hj⟩ := inBounds_rank1_iff r lb n (by rw [hdims₀, hd]) idx hidx
    have hjin : av.inBounds [j] := inBounds_rank1 av lb n hd j hj
    obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hdef [j] hjin)
    obtain ⟨w', hw', hrel'⟩ := hpt _ _ (RelA_get1 hA lb n hd j hj) v' hv'
    have hreb : rebase r.lbs [j] = [(j - lb).toNat] := by simp [rebase, FArr.lbs, hdims₀, hd]
    rw [hreb]
    simp only [hget₀, hget', hv', hw', Option.getD_some]
    exact hrel'

/-! ### Unfolding lemmas for the binary lifts (rank 1) -/

theorem F_lift2_arr_arr_extents {f : SVal → SVal → Option SVal} {a b : FArr} {v : F.Val}
    (hF : F.lift2 f (.arr a) (.arr b) = some v) : a.extents = b.extents := by
  by_contra h; simp [F.lift2, h] at hF

theorem F_lift2_arr_arr_empty (f : SVal → SVal → Option SVal) (a b : FArr) (heq : a.extents = b.extents)
    (h0 : a.size = 0) :
    F.lift2 f (.arr a) (.arr b) =
      some (.arr { elemTy := a.elemTy, dims := a.dims.map fun d => (1, d.2), get := a.get }) := by
  simp [F.lift2, heq, h0]

theorem F_lift2_arr_arr_pos (f : SVal → SVal → Option SVal) (a b : FArr) (la lb : ℤ) (n : ℕ)
    (hda : a.dims = [(la, n)]) (hdb : b.dims = [(lb, n)]) (hn : n ≠ 0) (sample : SVal)
    (hs : f (a.get [la]) (b.get [lb]) = some sample) :
    ∃ r : FArr, F.lift2 f (.arr a) (.arr b) = some (.arr r) ∧ r.elemTy = sample.ty ∧ r.dims = [(1, n)] ∧
      ∀ i : ℤ, r.get [i] = (f (a.get [la + i - 1]) (b.get [lb + i - 1])).getD sample := by
  have heq : a.extents = b.extents := by simp [FArr.extents, hda, hdb]
  have h0 : ¬ (a.size = 0) := by rw [FArr_size_rank1 a la n hda]; exact hn
  have hla : a.lbs = [la] := by simp [FArr.lbs, hda]
  have hlb : b.lbs = [lb] := by simp [FArr.lbs, hdb]
  refine ⟨{ elemTy := sample.ty, dims := a.dims.map fun d => (1, d.2), get := fun idx =>
      (f (a.get (List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) a.lbs idx))
         (b.get (List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) b.lbs idx))).getD sample },
    ?_, rfl, by simp [hda], fun i => by simp [hla, hlb]⟩
  simp [F.lift2, heq, h0, hla, hlb, hs]

theorem F_lift2_arr_s_none (f : SVal → SVal → Option SVal) (a : FArr) (b : SVal) (la : ℤ) (n : ℕ)
    (hd : a.dims = [(la, n)]) (hs : f (a.get [la]) b = none) : F.lift2 f (.arr a) (.s b) = none := by
  have hla : a.lbs = [la] := by simp [FArr.lbs, hd]
  simp [F.lift2, hla, hs]

theorem F_lift2_arr_s_some (f : SVal → SVal → Option SVal) (a : FArr) (b : SVal) (la : ℤ) (n : ℕ)
    (hd : a.dims = [(la, n)]) (sample : SVal) (hs : f (a.get [la]) b = some sample) :
    ∃ r : FArr, F.lift2 f (.arr a) (.s b) = some (.arr r) ∧ r.elemTy = sample.ty ∧ r.dims = [(1, n)] ∧
      ∀ i : ℤ, r.get [i] = (f (a.get [la + i - 1]) b).getD sample := by
  have hla : a.lbs = [la] := by simp [FArr.lbs, hd]
  refine ⟨{ elemTy := sample.ty, dims := a.dims.map fun d => (1, d.2), get := fun idx =>
      (f (a.get (List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) a.lbs idx)) b).getD sample },
    ?_, rfl, by simp [hd], fun i => by simp [hla]⟩
  simp [F.lift2, hla, hs]

theorem F_lift2_s_arr_none (f : SVal → SVal → Option SVal) (a : SVal) (b : FArr) (lb : ℤ) (n : ℕ)
    (hd : b.dims = [(lb, n)]) (hs : f a (b.get [lb]) = none) : F.lift2 f (.s a) (.arr b) = none := by
  have hlb : b.lbs = [lb] := by simp [FArr.lbs, hd]
  simp [F.lift2, hlb, hs]

theorem F_lift2_s_arr_some (f : SVal → SVal → Option SVal) (a : SVal) (b : FArr) (lb : ℤ) (n : ℕ)
    (hd : b.dims = [(lb, n)]) (sample : SVal) (hs : f a (b.get [lb]) = some sample) :
    ∃ r : FArr, F.lift2 f (.s a) (.arr b) = some (.arr r) ∧ r.elemTy = sample.ty ∧ r.dims = [(1, n)] ∧
      ∀ i : ℤ, r.get [i] = (f a (b.get [lb + i - 1])).getD sample := by
  have hlb : b.lbs = [lb] := by simp [FArr.lbs, hd]
  refine ⟨{ elemTy := sample.ty, dims := b.dims.map fun d => (1, d.2), get := fun idx =>
      (f a (b.get (List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) b.lbs idx))).getD sample },
    ?_, rfl, by simp [hd], fun i => by simp [hlb]⟩
  simp [F.lift2, hlb, hs]

theorem F_lift2_s_s (f : SVal → SVal → Option SVal) (a b : SVal) : F.lift2 f (.s a) (.s b) = (f a b).map .s := by
  simp [F.lift2]

theorem P_lift2_arr_arr_empty (g : PVal → PVal → Option PVal) (p q : PArr) (heq : p.shape = q.shape)
    (h0 : p.size = 0) : P.lift2 g (.arr p) (.arr q) = some (.arr p) := by
  simp [P.lift2, heq, h0]

theorem P_lift2_arr_arr_pos (g : PVal → PVal → Option PVal) (p q : PArr) (n : ℕ) (hp : p.shape = [n])
    (hq : q.shape = [n]) (hn : n ≠ 0) (sample : PVal) (hs : g (p.get [0]) (q.get [0]) = some sample) :
    ∃ r : PArr, P.lift2 g (.arr p) (.arr q) = some (.arr r) ∧ r.shape = [n] ∧
      ∀ i, r.get i = (g (p.get i) (q.get i)).getD sample := by
  have heq : p.shape = q.shape := by rw [hp, hq]
  have h0 : ¬ (p.size = 0) := by rw [PArr_size_rank1 p n hp]; exact hn
  have hz : p.shape.map (fun _ => (0 : ℕ)) = [0] := by rw [hp]; rfl
  have hz' : q.shape.map (fun _ => (0 : ℕ)) = [0] := by rw [hq]; rfl
  refine ⟨{ shape := p.shape, get := fun i => (g (p.get i) (q.get i)).getD sample }, ?_, hp, fun _ => rfl⟩
  simp [P.lift2, heq, h0, hz, hz', hs]

theorem P_lift2_arr_s_empty (g : PVal → PVal → Option PVal) (p : PArr) (w : PVal) (h0 : p.size = 0) :
    P.lift2 g (.arr p) (.s w) = some (.arr p) := by
  simp [P.lift2, h0]

theorem P_lift2_arr_s_pos (g : PVal → PVal → Option PVal) (p : PArr) (w : PVal) (n : ℕ) (hp : p.shape = [n])
    (hn : n ≠ 0) (sample : PVal) (hs : g (p.get [0]) w = some sample) :
    ∃ r : PArr, P.lift2 g (.arr p) (.s w) = some (.arr r) ∧ r.shape = [n] ∧
      ∀ i, r.get i = (g (p.get i) w).getD sample := by
  have h0 : ¬ (p.size = 0) := by rw [PArr_size_rank1 p n hp]; exact hn
  have hz : p.shape.map (fun _ => (0 : ℕ)) = [0] := by rw [hp]; rfl
  refine ⟨{ shape := p.shape, get := fun i => (g (p.get i) w).getD sample }, ?_, hp, fun _ => rfl⟩
  simp [P.lift2, h0, hz, hs]

theorem P_lift2_s_arr_empty (g : PVal → PVal → Option PVal) (w : PVal) (q : PArr) (h0 : q.size = 0) :
    P.lift2 g (.s w) (.arr q) = some (.arr q) := by
  simp [P.lift2, h0]

theorem P_lift2_s_arr_pos (g : PVal → PVal → Option PVal) (w : PVal) (q : PArr) (n : ℕ) (hq : q.shape = [n])
    (hn : n ≠ 0) (sample : PVal) (hs : g w (q.get [0]) = some sample) :
    ∃ r : PArr, P.lift2 g (.s w) (.arr q) = some (.arr r) ∧ r.shape = [n] ∧
      ∀ i, r.get i = (g w (q.get i)).getD sample := by
  have h0 : ¬ (q.size = 0) := by rw [PArr_size_rank1 q n hq]; exact hn
  have hz : q.shape.map (fun _ => (0 : ℕ)) = [0] := by rw [hq]; rfl
  refine ⟨{ shape := q.shape, get := fun i => (g w (q.get i)).getD sample }, ?_, hq, fun _ => rfl⟩
  simp [P.lift2, h0, hz, hs]

/-- A rank-1 result array with lower bound 1 is related to a Python array of the
same length whose elements are related index by index. -/
theorem relA_of_get (r : FArr) (r' : PArr) (n : ℕ) (hd : r.dims = [(1, n)]) (hs : r'.shape = [n])
    (h : ∀ i : ℕ, i < n → RelS (r.get [1 + (i : ℤ)]) (r'.get [i])) : RelA r r' := by
  refine ⟨by rw [hs]; simp [FArr.extents, hd], ?_⟩
  intro idx hidx
  obtain ⟨j, rfl, hj⟩ := inBounds_rank1_iff r 1 n hd idx hidx
  have hreb : rebase r.lbs [j] = [(j - 1).toNat] := by simp [rebase, FArr.lbs, hd]
  rw [hreb]
  have := h (j - 1).toNat (by omega)
  rwa [show (1 : ℤ) + ((j - 1).toNat : ℤ) = j by omega] at this

/-! ### `RelA` is preserved by pointwise-related binary lifts (rank 1, scalar broadcast included) -/

def Rank1Val : F.Val → Prop
  | .s _ => True
  | .arr a => ∃ lb n, a.dims = [(lb, n)]

/-- `Q` holds of every pair of corresponding elements (scalar broadcast included). -/
def Pairwise2 (Q : SVal → SVal → Prop) : F.Val → F.Val → Prop
  | .s a, .s b => Q a b
  | .arr a, .s b => ∀ idx, a.inBounds idx → Q (a.get idx) b
  | .s a, .arr b => ∀ idx, b.inBounds idx → Q a (b.get idx)
  | .arr a, .arr b => ∀ i : ℕ, i < a.size → Q (a.get [a.lbs.headD 0 + i]) (b.get [b.lbs.headD 0 + i])

/-- Pointwise relation of two binary scalar functions on the pairs satisfying `Q`. -/
def PointRel2 (Q : SVal → SVal → Prop) (f : SVal → SVal → Option SVal) (g : PVal → PVal → Option PVal) : Prop :=
  ∀ v₁ w₁ v₂ w₂, Q v₁ v₂ → RelS v₁ w₁ → RelS v₂ w₂ → ∀ v', f v₁ v₂ = some v' → ∃ w', g w₁ w₂ = some w' ∧ RelS v' w'

theorem lift2_rel (Q : SVal → SVal → Prop) (f : SVal → SVal → Option SVal) (g : PVal → PVal → Option PVal)
    (hpt : PointRel2 Q f g) (hQdef : ∀ v₁ v₂, Q v₁ v₂ → (f v₁ v₂).isSome)
    (x y : F.Val) (x' y' : P.Val) (hx : RelV x x') (hy : RelV y y') (hrx : Rank1Val x) (hry : Rank1Val y)
    (hQ : Pairwise2 Q x y) (v : F.Val) (hF : F.lift2 f x y = some v) :
    ∃ w, P.lift2 g x' y' = some w ∧ RelV v w := by
  cases hx with
  | s hsa =>
    rename_i a wa
    cases hy with
    | s hsb =>
      rename_i b wb
      rw [F_lift2_s_s] at hF
      simp only [Option.map_eq_some_iff] at hF
      obtain ⟨v', hv', rfl⟩ := hF
      obtain ⟨w', hw', hrel⟩ := hpt a wa b wb hQ hsa hsb v' hv'
      exact ⟨.s w', by rw [P_lift2_scalar, hw']; rfl, RelV.s hrel⟩
    | arr hB =>
      rename_i bv q
      obtain ⟨lb, n, hdb⟩ := hry
      cases hs : f a (bv.get [lb]) with
      | none => rw [F_lift2_s_arr_none f a bv lb n hdb hs] at hF; cases hF
      | some sample =>
        obtain ⟨r, hr, _, hdims, hget⟩ := F_lift2_s_arr_some f a bv lb n hdb sample hs
        rw [hr] at hF; cases hF
        have hshape := RelA_dims1 hB lb n hdb
        by_cases hn : n = 0
        · subst hn
          refine ⟨.arr q, P_lift2_s_arr_empty g wa q (PArr_size_rank1 q 0 hshape), RelV.arr ?_⟩
          exact relA_of_get r q 0 hdims hshape (fun i hi => absurd hi (Nat.not_lt_zero _))
        · have hel0 : RelS (bv.get [lb]) (q.get [0]) := by
            have := RelA_get1 hB lb n hdb lb ⟨le_refl _, by omega⟩; simpa using this
          have hQ0 : Q a (bv.get [lb]) := hQ [lb] (inBounds_rank1 bv lb n hdb lb ⟨le_refl _, by omega⟩)
          obtain ⟨sample', hs', _⟩ := hpt a wa _ _ hQ0 hsa hel0 sample hs
          obtain ⟨r', hr', hshape', hget'⟩ := P_lift2_s_arr_pos g wa q n hshape hn sample' hs'
          refine ⟨.arr r', hr', RelV.arr (relA_of_get r r' n hdims hshape' fun i hi => ?_)⟩
          have hjin := inBounds_rank1 bv lb n hdb (lb + i) ⟨by omega, by omega⟩
          have hQi := hQ [lb + i] hjin
          obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hQdef _ _ hQi)
          have hel : RelS (bv.get [lb + i]) (q.get [i]) := by
            have := RelA_get1 hB lb n hdb (lb + i) ⟨by omega, by omega⟩; simpa using this
          obtain ⟨w', hw', hrel⟩ := hpt a wa _ _ hQi hsa hel v' hv'
          rw [hget, hget', show lb + (1 + (i : ℤ)) - 1 = lb + i by ring, hv', hw']
          exact hrel
  | arr hA =>
    rename_i av p
    obtain ⟨la, n, hda⟩ := hrx
    cases hy with
    | s hsb =>
      rename_i b wb
      cases hs : f (av.get [la]) b with
      | none => rw [F_lift2_arr_s_none f av b la n hda hs] at hF; cases hF
      | some sample =>
        obtain ⟨r, hr, _, hdims, hget⟩ := F_lift2_arr_s_some f av b la n hda sample hs
        rw [hr] at hF; cases hF
        have hshape := RelA_dims1 hA la n hda
        by_cases hn : n = 0
        · subst hn
          refine ⟨.arr p, P_lift2_arr_s_empty g p wb (PArr_size_rank1 p 0 hshape), RelV.arr ?_⟩
          exact relA_of_get r p 0 hdims hshape (fun i hi => absurd hi (Nat.not_lt_zero _))
        · have hel0 : RelS (av.get [la]) (p.get [0]) := by
            have := RelA_get1 hA la n hda la ⟨le_refl _, by omega⟩; simpa using this
          have hQ0 : Q (av.get [la]) b := hQ [la] (inBounds_rank1 av la n hda la ⟨le_refl _, by omega⟩)
          obtain ⟨sample', hs', _⟩ := hpt _ _ b wb hQ0 hel0 hsb sample hs
          obtain ⟨r', hr', hshape', hget'⟩ := P_lift2_arr_s_pos g p wb n hshape hn sample' hs'
          refine ⟨.arr r', hr', RelV.arr (relA_of_get r r' n hdims hshape' fun i hi => ?_)⟩
          have hjin := inBounds_rank1 av la n hda (la + i) ⟨by omega, by omega⟩
          have hQi := hQ [la + i] hjin
          obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hQdef _ _ hQi)
          have hel : RelS (av.get [la + i]) (p.get [i]) := by
            have := RelA_get1 hA la n hda (la + i) ⟨by omega, by omega⟩; simpa using this
          obtain ⟨w', hw', hrel⟩ := hpt _ _ b wb hQi hel hsb v' hv'
          rw [hget, hget', show la + (1 + (i : ℤ)) - 1 = la + i by ring, hv', hw']
          exact hrel
    | arr hB =>
      rename_i bv q
      obtain ⟨lb, m, hdb⟩ := hry
      have hext := F_lift2_arr_arr_extents hF
      have hnm : m = n := by simp [FArr.extents, hda, hdb] at hext; exact hext.symm
      subst hnm
      have hshape := RelA_dims1 hA la m hda
      have hshapeq := RelA_dims1 hB lb m hdb
      have hsz := FArr_size_rank1 av la m hda
      by_cases hn : m = 0
      · subst hn
        rw [F_lift2_arr_arr_empty f av bv hext hsz] at hF
        cases hF
        refine ⟨.arr p, P_lift2_arr_arr_empty g p q (by rw [hshape, hshapeq]) (PArr_size_rank1 p 0 hshape),
          RelV.arr ?_⟩
        exact relA_of_get _ p 0 (by simp [hda]) hshape (fun i hi => absurd hi (Nat.not_lt_zero _))
      · have hQ0 : Q (av.get [la]) (bv.get [lb]) := by
          have := hQ 0 (by omega)
          simpa [FArr.lbs, hda, hdb] using this
        obtain ⟨sample, hs⟩ := Option.isSome_iff_exists.mp (hQdef _ _ hQ0)
        obtain ⟨r, hr, _, hdims, hget⟩ := F_lift2_arr_arr_pos f av bv la lb m hda hdb hn sample hs
        rw [hr] at hF; cases hF
        have hela : RelS (av.get [la]) (p.get [0]) := by
          have := RelA_get1 hA la m hda la ⟨le_refl _, by omega⟩; simpa using this
        have helb : RelS (bv.get [lb]) (q.get [0]) := by
          have := RelA_get1 hB lb m hdb lb ⟨le_refl _, by omega⟩; simpa using this
        obtain ⟨sample', hs', _⟩ := hpt _ _ _ _ hQ0 hela helb sample hs
        obtain ⟨r', hr', hshape', hget'⟩ := P_lift2_arr_arr_pos g p q m hshape hshapeq hn sample' hs'
        refine ⟨.arr r', hr', RelV.arr (relA_of_get r r' m hdims hshape' fun i hi => ?_)⟩
        have hQi : Q (av.get [la + i]) (bv.get [lb + i]) := by
          have := hQ i (by omega)
          simpa [FArr.lbs, hda, hdb] using this
        obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.mp (hQdef _ _ hQi)
        have hela' : RelS (av.get [la + i]) (p.get [i]) := by
          have := RelA_get1 hA la m hda (la + i) ⟨by omega, by omega⟩; simpa using this
        have helb' : RelS (bv.get [lb + i]) (q.get [i]) := by
          have := RelA_get1 hB lb m hdb (lb + i) ⟨by omega, by omega⟩; simpa using this
        obtain ⟨w', hw', hrel⟩ := hpt _ _ _ _ hQi hela' helb' v' hv'
        rw [hget, hget', show la + (1 + (i : ℤ)) - 1 = la + i by ring,
          show lb + (1 + (i : ℤ)) - 1 = lb + i by ring, hv', hw']
        exact hrel

/-! ### FNP-R0100: the vectorised rows of `Tr.arrayHelper` -/

/-- The rows of `ELEMENTAL_ARRAY` whose Python spelling is elementwise on arrays. -/
inductive VectorRow : F.Intr → P.Helper → Prop where
  | abs : VectorRow .abs .np_abs
  | sqrt : VectorRow .sqrt .np_sqrt
  | exp : VectorRow .exp .f_vexp
  | log : VectorRow .log .f_vlog
  | log10 : VectorRow .log10 .f_vlog10
  | sin : VectorRow .sin .np_sin
  | cos : VectorRow .cos .np_cos
  | tanh : VectorRow .tanh .np_tanh
  | aint : VectorRow .aint .np_trunc
  | floor : VectorRow .floor .f_vfloor
  | ceiling : VectorRow .ceiling .f_vceil
  | real : VectorRow .real .np_float64
  | dble : VectorRow .dble .np_float64

theorem vectorRow_arrayHelper {f : F.Intr} {h : P.Helper} (hrow : VectorRow f h) : Tr.arrayHelper f = some h := by
  cases hrow <;> rfl

/-- The scalar function each row applies on the Fortran side (`F.intrinsic`, no KIND argument). -/
noncomputable def rowF : F.Intr → SVal → Option SVal
  | .abs => fun v => match v with
      | .int k a => some (.int k |a|)
      | .real k a => some (.real k |a|)
      | _ => none
  | .sqrt => F.realFn fun a => if 0 ≤ a then some (Real.sqrt a) else none
  | .exp => F.realFn fun a => some (Real.exp a)
  | .log => F.realFn fun a => if 0 < a then some (Real.log a) else none
  | .log10 => F.realFn fun a => if 0 < a then some (Real.logb 10 a) else none
  | .sin => F.realFn fun a => some (Real.sin a)
  | .cos => F.realFn fun a => some (Real.cos a)
  | .tanh => F.realFn fun a => some (Real.tanh a)
  | .aint => fun v => match v with
      | .real k a => some (.real k (truncR a : ℝ))
      | _ => none
  | .floor => fun v => match v with
      | .real _ a => some (.int .i4 ⌊a⌋)
      | _ => none
  | .ceiling => fun v => match v with
      | .real _ a => some (.int .i4 ⌈a⌉)
      | _ => none
  | .real => fun v => match v with
      | .int _ a => some (.real .r4 (a : ℝ))
      | .real _ a => some (.real .r4 a)
      | _ => none
  | .dble => fun v => match v with
      | .int _ a => some (.real .r8 (a : ℝ))
      | .real _ a => some (.real .r8 a)
      | _ => none
  | _ => fun _ => none

/-- The scalar function each row applies on the Python side (`P.helper`). -/
noncomputable def rowP : P.Helper → PVal → Option PVal
  | .np_abs => fun v => match v with
      | .int a => some (.int |a|)
      | .float a => some (.float |a|)
      | _ => none
  | .np_sqrt => P.realFn fun a => if 0 ≤ a then some (Real.sqrt a) else none
  | .f_vexp => P.realFn fun a => some (Real.exp a)
  | .f_vlog => P.realFn fun a => if 0 < a then some (Real.log a) else none
  | .f_vlog10 => P.realFn fun a => if 0 < a then some (Real.logb 10 a) else none
  | .np_sin => P.realFn fun a => some (Real.sin a)
  | .np_cos => P.realFn fun a => some (Real.cos a)
  | .np_tanh => P.realFn fun a => some (Real.tanh a)
  | .np_trunc => fun v => match v with
      | .float a => some (.float (truncR a : ℝ))
      | .int a => some (.int a)
      | _ => none
  | .np_round => fun v => match v with
      | .float a => some (.float (roundHalfEvenR a : ℝ))
      | .int a => some (.int a)
      | _ => none
  | .f_vfloor => fun v => (P.toR v).map fun a => .int ⌊a⌋
  | .f_vceil => fun v => (P.toR v).map fun a => .int ⌈a⌉
  | .np_float64 => fun v => (P.toR v).map .float
  | _ => fun _ => none

theorem intrinsic_row {f : F.Intr} {h : P.Helper} (hrow : VectorRow f h) (o : F.Oracle) (x : F.Val) :
    F.intrinsic o f [x] none = F.lift1 (rowF f) x := by
  cases hrow <;> rfl

theorem helper_row {f : F.Intr} {h : P.Helper} (hrow : VectorRow f h) (o : P.Oracle) (x : P.Val) :
    P.helper o h [x] = P.lift1 (rowP h) x := by
  cases hrow <;> rfl

/-- A real function applied through `F.realFn` / `P.realFn` is pointwise related
(value-based: the Python operand may be an `int`). -/
theorem realFn_pointRel (g : ℝ → Option ℝ) : PointRel (F.realFn g) (P.realFn g) := by
  intro v w hrel v' hv
  cases v with
  | real k a =>
    have ht := RelS.real_toR hrel
    simp only [F.realFn, Option.map_eq_some_iff] at hv
    obtain ⟨r, hr, rfl⟩ := hv
    refine ⟨.float r, ?_, RelS.realF⟩
    simp [P.realFn, ht, hr]
  | int _ _ => simp [F.realFn] at hv
  | bool _ => simp [F.realFn] at hv
  | str _ => simp [F.realFn] at hv

theorem row_rel {f : F.Intr} {h : P.Helper} (hrow : VectorRow f h) : PointRel (rowF f) (rowP h) := by
  cases hrow
  case abs =>
    intro v w hrel v' hv
    cases v with
    | int k a =>
      rw [RelS.int_eq hrel]
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      exact ⟨.int |a|, rfl, RelS.int⟩
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      rcases RelS.real_cases hrel with ⟨n, rfl, rfl⟩ | rfl
      · exact ⟨.int |n|, rfl, RelS.real (by simp [P.toR, Int.cast_abs])⟩
      · exact ⟨.float |a|, rfl, RelS.realF⟩
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv
  case sqrt => exact realFn_pointRel _
  case exp => exact realFn_pointRel _
  case log => exact realFn_pointRel _
  case log10 => exact realFn_pointRel _
  case sin => exact realFn_pointRel _
  case cos => exact realFn_pointRel _
  case tanh => exact realFn_pointRel _
  case aint =>
    intro v w hrel v' hv
    cases v with
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      rcases RelS.real_cases hrel with ⟨n, rfl, rfl⟩ | rfl
      · exact ⟨.int n, rfl, RelS.real (by simp [P.toR, truncR_intCast])⟩
      · exact ⟨.float (truncR a : ℝ), rfl, RelS.realF⟩
    | int _ _ => simp [rowF] at hv
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv
  case floor =>
    intro v w hrel v' hv
    cases v with
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      have ht := RelS.real_toR hrel
      exact ⟨.int ⌊a⌋, by simp [rowP, ht], RelS.int⟩
    | int _ _ => simp [rowF] at hv
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv
  case ceiling =>
    intro v w hrel v' hv
    cases v with
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      have ht := RelS.real_toR hrel
      exact ⟨.int ⌈a⌉, by simp [rowP, ht], RelS.int⟩
    | int _ _ => simp [rowF] at hv
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv
  case real =>
    intro v w hrel v' hv
    cases v with
    | int k a =>
      rw [RelS.int_eq hrel]
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      exact ⟨.float (a : ℝ), rfl, RelS.realF⟩
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      have ht := RelS.real_toR hrel
      exact ⟨.float a, by simp [rowP, ht], RelS.realF⟩
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv
  case dble =>
    intro v w hrel v' hv
    cases v with
    | int k a =>
      rw [RelS.int_eq hrel]
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      exact ⟨.float (a : ℝ), rfl, RelS.realF⟩
    | real k a =>
      simp only [rowF, Option.some.injEq] at hv
      subst hv
      have ht := RelS.real_toR hrel
      exact ⟨.float a, by simp [rowP, ht], RelS.realF⟩
    | bool _ => simp [rowF] at hv
    | str _ => simp [rowF] at hv

/-- FNP-R0100 restricted to the vectorised rows: the argument is a scalar or a
rank-1 array (`hrank`) whose every element is in the intrinsic's domain (`hdef`,
source definedness). -/
theorem FNP_R0100_sound_row (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper) (hrow : VectorRow f h)
    (a : F.Expr) (a' : P.Expr) (ha : SoundPair o Γ a a')
    (hrank : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ a = some v → Rank1Val v)
    (hdef : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ av, F.eval o.F σ a = some (.arr av) →
      ∀ idx, av.inBounds idx → (rowF f (av.get idx)).isSome) :
    SoundPair o Γ (.call f (.cons a .nil) none) (Tr.fold1 h a') := by
  intro σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  rw [intrinsic_row hrow] at hint
  obtain ⟨w, hw, hrw⟩ := ha σ τ hR hC _ hva
  rw [P_eval_call1, hw, Option.bind_some, helper_row hrow]
  cases hrw with
  | s hs =>
    rename_i sv pv
    simp only [F.lift1, Option.map_eq_some_iff] at hint
    obtain ⟨v', hv', rfl⟩ := hint
    obtain ⟨w', hw', hrel'⟩ := row_rel hrow sv pv hs v' hv'
    exact ⟨.s w', by rw [P_lift1_scalar, hw']; rfl, RelV.s hrel'⟩
  | arr hA =>
    rename_i av p
    obtain ⟨lb, n, hd⟩ := hrank σ τ hR hC _ hva
    cases v with
    | s sv => exact absurd hint (F_lift1_arr_not_scalar _ av sv)
    | arr r =>
      obtain ⟨r', hr', hrel'⟩ :=
        lift1_rel (rowF f) (rowP h) av p lb n hd hA (row_rel hrow) (hdef σ τ hR hC av hva) r hint
      exact ⟨.arr r', hr', RelV.arr hrel'⟩

/-! ### FNP-R0100 as stated is refuted -/

/-- FNP-D0036 class: `Tr.arrayHelper .tan` falls back to the scalar spelling
`math.tan`, which is undefined on an array; Fortran's `TAN` of a REAL array is
defined. -/
theorem FNP_R0100_refuted : ¬ FNP_R0100_obligation := by
  intro h
  have hs := h noPrim envA .tan .math_tan rfl (.var "a") (.name "a") (FNP_R0010_sound noPrim envA "a" declA envA_a)
  obtain ⟨r, hr, _, _, _⟩ := F_lift1_arr_pos (F.realFn fun a => some (Real.tan a)) twoF 1 2 rfl (by decide)
    (.real .r8 (Real.tan 0)) rfl
  have hF : F.eval noPrim.F stateA (.call .tan (.cons (.var "a") .nil) none) = some (.arr r) := by
    rw [F_eval_call1]
    simp only [F.eval, stateA, if_true, Option.bind_some, Option.map_none]
    exact hr
  obtain ⟨w, hw, _⟩ := hs stateA pstateA relStateA conformsA _ hF
  rw [P_eval_call1] at hw
  simp only [P.eval, pstateA, if_true, Option.bind_some] at hw
  rw [show P.helper noPrim.P .math_tan [.arr twoP] = none from rfl] at hw
  cases hw

/-! A one-element REAL(8) array holding `2.5`, for the elementwise ANINT counterexample. -/

noncomputable def halfF : FArr := { elemTy := .real .r8, dims := [(1, 1)], get := fun _ => .real .r8 (5/2) }
noncomputable def halfP : PArr := { shape := [1], get := fun _ => .float (5/2) }
noncomputable def stateH : F.State := fun x => if x = "a" then some (.arr halfF) else none
noncomputable def pstateH : P.State := fun x => if x = "a" then some (.arr halfP) else none

theorem relStateH : RelState envA stateH pstateH := by
  intro x d hx
  simp only [envA] at hx
  split_ifs at hx with hxa
  · subst hxa
    exact ⟨.arr halfF, .arr halfP, by simp [stateH], by simp [pstateH], RelV.arr ⟨rfl, fun _ _ => RelS.realF⟩⟩

theorem conformsH : Conforms noPrim.F envA stateH := by
  intro x d hx
  simp only [envA] at hx
  split_ifs at hx with hxa
  · subst hxa
    cases hx
    refine ⟨fun h => by simp [declA] at h, fun _ => ⟨halfF, by simp [stateH], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
    intro i hi
    simp only [declA] at hi
    match i, hi with
    | 0, _ => simp [declA, F_eval_intLit, halfF, FArr.lbs]

/-- The scalar function of `ANINT` (`F.intrinsic .anint`). -/
noncomputable def rowAnint : SVal → Option SVal := fun v => match v with
  | .real k a => some (.real k (nintR a : ℝ))
  | _ => none

/-- FNP-D0004 elementwise: `ANINT` of `[2.5]` is `[3.0]`; `np.round` gives `[2.0]`. -/
theorem FNP_R0100_refuted_anint : ¬ FNP_R0100_obligation := by
  intro h
  have hs := h noPrim envA .anint .np_round rfl (.var "a") (.name "a") (FNP_R0010_sound noPrim envA "a" declA envA_a)
  obtain ⟨r, hr, _, hdims, hget⟩ := F_lift1_arr_pos rowAnint halfF 1 1 rfl (by decide)
    (.real .r8 (nintR (5/2 : ℝ) : ℝ)) (by simp [rowAnint, halfF])
  have hF : F.eval noPrim.F stateH (.call .anint (.cons (.var "a") .nil) none) = some (.arr r) := by
    rw [F_eval_call1]
    simp only [F.eval, stateH, if_true, Option.bind_some, Option.map_none]
    exact hr
  obtain ⟨w, hw, hrel⟩ := hs stateH pstateH relStateH conformsH _ hF
  obtain ⟨r', hr', _, hget'⟩ := P_lift1_arr_pos (rowP .np_round) halfP 1 rfl (by decide)
    (.float (roundHalfEvenR (5/2 : ℝ) : ℝ)) (by simp [rowP, halfP])
  rw [P_eval_call1] at hw
  simp only [P.eval, pstateH, if_true, Option.bind_some] at hw
  have e : ∀ x : P.Val, P.helper noPrim.P .np_round [x] = P.lift1 (rowP .np_round) x := fun _ => rfl
  rw [e, hr'] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have hin : r.inBounds [1] := by simp [FArr.inBounds, FArr.inBoundsB, hdims, halfF]
    have := hA.elems [1] hin
    have hreb : rebase r.lbs [1] = [0] := by simp [rebase, FArr.lbs, hdims, halfF]
    rw [hreb, hget, hget'] at this
    simp only [halfF, halfP, rowAnint, rowP, Option.getD_some, nintR_five_halves, roundHalfEven_five_halves] at this
    have ht := RelS.real_toR this
    simp only [P.toR, Option.some.injEq] at ht
    norm_num at ht

/-! ### FNP-R0025: array `**` -> `_f_vpow` -/

theorem P_arith_pow_eq_pyPow (x y : PVal) : P.arith .pow x y = P.pyPow x y := by
  cases x <;> cases y <;> rfl

/-- Element premise of the REAL-base power: a REAL base and a defined power. -/
def PowOk (v₁ v₂ : SVal) : Prop := (∃ k r, v₁ = .real k r) ∧ (F.arith .pow v₁ v₂).isSome

theorem pow_pointRel : PointRel2 PowOk (F.arith .pow) P.pyPow := by
  intro v₁ w₁ v₂ w₂ hQ h1 h2 v' hv
  obtain ⟨w', hw', hrel⟩ := arith_pow_real_rel h1 h2 hQ.1 hv
  exact ⟨w', by rw [← P_arith_pow_eq_pyPow]; exact hw', hrel⟩

/-- Both operands evaluate to scalars or rank-1 arrays whose corresponding
elements satisfy `Q`. -/
def Rank1Pair (o : Oracles) (Γ : F.Env) (a b : F.Expr) (Q : SVal → SVal → Prop) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ va vb, F.eval o.F σ a = some va → F.eval o.F σ b = some vb →
    Rank1Val va ∧ Rank1Val vb ∧ Pairwise2 Q va vb

/-- FNP-R0025 for REAL bases (scalar or rank-1, scalar broadcast allowed), every
element power defined in the source. -/
theorem FNP_R0025_sound_real (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundPair o Γ a a') (hb : SoundPair o Γ b b') (hok : Rank1Pair o Γ a b PowOk) :
    SoundPair o Γ (.bin .pow a b) (Tr.call2 .f_vpow a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨wa, hwa, hrwa⟩ := ha σ τ hR hC _ hva
  obtain ⟨wb, hwb, hrwb⟩ := hb σ τ hR hC _ hvb
  obtain ⟨hra, hrb, hQ⟩ := hok σ τ hR hC va vb hva hvb
  rw [P_eval_call2, hwa, hwb, Option.bind_some, Option.bind_some]
  rw [show P.helper o.P .f_vpow [wa, wb] = P.lift2 P.pyPow wa wb from rfl]
  exact lift2_rel PowOk _ _ pow_pointRel (fun _ _ hq => hq.2) va vb wa wb hrwa hrwb hra hrb hQ v hlift

/-! A one-element INTEGER array holding `-1`, for the integer-power counterexample. -/

def negF : FArr := { elemTy := .int .i4, dims := [(1, 1)], get := fun _ => .int .i4 (-1) }
def negP : PArr := { shape := [1], get := fun _ => .int (-1) }
def declIL : F.Decl := ⟨.int .i4, [.intLit .i4 1], 0⟩
def envIL : F.Env := fun x => if x = "a" then some declIL else none
def stateIL : F.State := fun x => if x = "a" then some (.arr negF) else none
def pstateIL : P.State := fun x => if x = "a" then some (.arr negP) else none

theorem relStateIL : RelState envIL stateIL pstateIL := by
  intro x d hx
  simp only [envIL] at hx
  split_ifs at hx with hxa
  · subst hxa
    exact ⟨.arr negF, .arr negP, by simp [stateIL], by simp [pstateIL], RelV.arr ⟨rfl, fun _ _ => RelS.int⟩⟩

theorem conformsIL : Conforms noPrim.F envIL stateIL := by
  intro x d hx
  simp only [envIL] at hx
  split_ifs at hx with hxa
  · subst hxa
    cases hx
    refine ⟨fun h => by simp [declIL] at h, fun _ => ⟨negF, by simp [stateIL], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
    intro i hi
    simp only [declIL] at hi
    match i, hi with
    | 0, _ => simp [declIL, F_eval_intLit, negF, FArr.lbs]

/-- FNP-D0012 elementwise: `[-1] ** [-1]` is the INTEGER `[-1]` in Fortran and
the float `[-1.0]` through `_f_vpow`. -/
theorem FNP_R0025_refuted : ¬ FNP_R0025_obligation := by
  intro h
  have hva : SoundPair noPrim envIL (.var "a") (.name "a") := FNP_R0010_sound noPrim envIL "a" declIL (by simp [envIL])
  have hs := h noPrim envIL (.var "a") (.var "a") (.name "a") (.name "a") hva hva
  have hpow : F.arith .pow (.int .i4 (-1)) (.int .i4 (-1)) = some (.int .i4 (-1)) := by
    simp [F.arith, F.ipowNeg, tquot, IKind.join]
  obtain ⟨r, hr, _, hdims, hget⟩ := F_lift2_arr_arr_pos (F.arith .pow) negF negF 1 1 1 rfl rfl (by decide) _ hpow
  have hF : F.eval noPrim.F stateIL (.bin .pow (.var "a") (.var "a")) = some (.arr r) := by
    rw [F_eval_bin]
    simp only [F.eval, stateIL, if_true, Option.bind_some]
    exact hr
  obtain ⟨w, hw, hrel⟩ := hs stateIL pstateIL relStateIL conformsIL _ hF
  have hpy : P.pyPow (.int (-1)) (.int (-1)) = some (.float ((-1 : ℝ) ^ (-1 : ℤ))) := by
    simp [P.pyPow]
  obtain ⟨r', hr', _, hget'⟩ := P_lift2_arr_arr_pos P.pyPow negP negP 1 rfl rfl (by decide) _ hpy
  rw [P_eval_call2] at hw
  simp only [P.eval, pstateIL, if_true, Option.bind_some] at hw
  rw [show P.helper noPrim.P .f_vpow [.arr negP, .arr negP] = P.lift2 P.pyPow (.arr negP) (.arr negP) from rfl,
    hr'] at hw
  cases hw
  cases hrel with
  | arr hA =>
    have hin : r.inBounds [1] := by simp [FArr.inBounds, FArr.inBoundsB, hdims]
    have := hA.elems [1] hin
    have hreb : rebase r.lbs [1] = [0] := by simp [rebase, FArr.lbs, hdims]
    rw [hreb, hget, hget'] at this
    simp only [negF, negP, hpow, hpy, Option.getD_some] at this
    exact RelS.int_ne_float this

/-! ### FNP-R0035: array `.AND.` / `.OR.` -> `&` / `|` -/

def BoolPair (v₁ v₂ : SVal) : Prop := (∃ p, v₁ = .bool p) ∧ (∃ q, v₂ = .bool q)

/-- The elementwise functions of `P.eval` for `.bitand` / `.bitor`. -/
def pyAnd : PVal → PVal → Option PVal := fun x y => match x, y with
  | .bool p, .bool q => some (.bool (p && q))
  | _, _ => none
def pyOr : PVal → PVal → Option PVal := fun x y => match x, y with
  | .bool p, .bool q => some (.bool (p || q))
  | _, _ => none

theorem P_eval_bitand (o : P.Oracle) (τ : P.State) (a b : P.Expr) :
    P.eval o τ (.bitand a b) = (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.lift2 pyAnd va vb := by
  simp only [P.eval]; rfl

theorem P_eval_bitor (o : P.Oracle) (τ : P.State) (a b : P.Expr) :
    P.eval o τ (.bitor a b) = (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.lift2 pyOr va vb := by
  simp only [P.eval]; rfl

theorem logic_defined (op : F.LogOp) (v₁ v₂ : SVal) (h : BoolPair v₁ v₂) : (F.logic op v₁ v₂).isSome := by
  obtain ⟨⟨p, rfl⟩, ⟨q, rfl⟩⟩ := h
  cases op <;> rfl

theorem and_pointRel : PointRel2 BoolPair (F.logic .and) pyAnd := by
  rintro v₁ w₁ v₂ w₂ ⟨⟨p, rfl⟩, ⟨q, rfl⟩⟩ h1 h2 v' hv
  rw [RelS.bool_eq h1, RelS.bool_eq h2]
  simp only [F.logic, Option.some.injEq] at hv
  subst hv
  exact ⟨.bool (p && q), rfl, RelS.bool⟩

theorem or_pointRel : PointRel2 BoolPair (F.logic .or) pyOr := by
  rintro v₁ w₁ v₂ w₂ ⟨⟨p, rfl⟩, ⟨q, rfl⟩⟩ h1 h2 v' hv
  rw [RelS.bool_eq h1, RelS.bool_eq h2]
  simp only [F.logic, Option.some.injEq] at hv
  subst hv
  exact ⟨.bool (p || q), rfl, RelS.bool⟩

/-- FNP-R0035 on LOGICAL operands (scalars or rank-1 arrays, scalar broadcast allowed). -/
theorem FNP_R0035_sound_logical (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundPair o Γ a a') (hb : SoundPair o Γ b b') (hok : Rank1Pair o Γ a b BoolPair) :
    SoundPair o Γ (.logic .and a b) (.paren (.bitand a' b')) ∧
    SoundPair o Γ (.logic .or a b) (.paren (.bitor a' b')) := by
  constructor <;> intro σ τ hR hC v hv <;> rw [F_eval_logic] at hv <;>
    simp only [Option.bind_eq_some_iff] at hv <;> obtain ⟨va, hva, vb, hvb, hlift⟩ := hv <;>
    obtain ⟨wa, hwa, hrwa⟩ := ha σ τ hR hC _ hva <;> obtain ⟨wb, hwb, hrwb⟩ := hb σ τ hR hC _ hvb <;>
    obtain ⟨hra, hrb, hQ⟩ := hok σ τ hR hC va vb hva hvb
  · rw [P_eval_paren, P_eval_bitand, hwa, hwb, Option.bind_some, Option.bind_some]
    exact lift2_rel BoolPair _ _ and_pointRel (logic_defined .and) va vb wa wb hrwa hrwb hra hrb hQ v hlift
  · rw [P_eval_paren, P_eval_bitor, hwa, hwb, Option.bind_some, Option.bind_some]
    exact lift2_rel BoolPair _ _ or_pointRel (logic_defined .or) va vb wa wb hrwa hrwb hra hrb hQ v hlift

end Recast.Rules
