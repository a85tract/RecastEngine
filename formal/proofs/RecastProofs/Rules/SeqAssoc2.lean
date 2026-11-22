import RecastProofs.Rules.Obligations

/-!
# Sequence association, value level: FNP-R0054, R0056, R0057, R0058, R0059

Fortran passes an array element (or a whole array of another rank) to a dummy
argument by *sequence association* (F2018 15.5.2.11): the dummy's element
sequence is the actual's storage sequence, in array element (column-major)
order, starting at the element passed.  The emitter spells each case as a
NumPy view or reshape.  This file states the value content of those rules:

* the Fortran side is the storage sequence `F.storageFrom a off` and, for an
  explicit-shape dummy, the array `F.seqDummy a off extents` laid over it;
* the Python side is the emitted view / reshape / `_f_seq_tail` as a function
  `PArr → Option PArr` (`P.viewColumn`, `P.tail`, `P.reshapeTake`,
  `P.reshapeNeg1`, `P.seqTail`);
* the obligation is `RelA (F.seqDummy …) q` (explicit shape) or the elementwise
  relation of the two sequences `SeqRel (F.storageFrom a off) (P.ravelF q)`
  (assumed size: the dummy *is* the remaining storage, F2018 8.5.8.5).

Results: R0054, R0056, R0058 sound; R0059 sound for `c(*)`, conditional on
divisibility for `vl(ldvl, *)`; R0057 refuted -- the rank-2 "view" special case
of `_f_seq_tail` and the whole-column truncation of its general case both
drop elements of the dummy's storage sequence.
-/

namespace Recast

open Rel

/-! ### Column-major coordinates: products and the two inverses -/

namespace CM

/-- Product of extents (the size). -/
def prod (ns : List ℕ) : ℕ := ns.foldr (· * ·) 1

theorem prod_nil : prod [] = 1 := rfl
theorem prod_cons (n : ℕ) (ns : List ℕ) : prod (n :: ns) = n * prod ns := rfl

theorem prod_append_singleton (l : List ℕ) (m : ℕ) : prod (l ++ [m]) = prod l * m := by
  induction l with
  | nil => simp [prod]
  | cons n ns ih => simp only [List.cons_append, prod_cons, ih]; ring

theorem coords_length : ∀ (ns : List ℕ) (k : ℕ), (coords ns k).length = ns.length
  | [], _ => rfl
  | _ :: ns, k => by simp [coords, coords_length ns]

theorem coords_inBounds : ∀ (ns : List ℕ) (k : ℕ), k < prod ns → PArr.inBoundsB ns (coords ns k) = true
  | [], _, _ => rfl
  | n :: ns, k, h => by
      rw [prod_cons] at h
      have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp at h)
      simp only [coords, PArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq]
      refine ⟨Nat.mod_lt k hn, coords_inBounds ns (k / n) ?_⟩
      rw [Nat.div_lt_iff_lt_mul hn, Nat.mul_comm]; exact h

theorem offset_lt_prod : ∀ (ns is : List ℕ), PArr.inBoundsB ns is = true → offset ns is < prod ns
  | [], [], _ => by simp [offset, prod]
  | [], _ :: _, h => by simp [PArr.inBoundsB] at h
  | _ :: _, [], h => by simp [PArr.inBoundsB] at h
  | n :: ns, i :: is, h => by
      simp only [PArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
      have ho := offset_lt_prod ns is h.2
      simp only [offset, prod_cons]
      calc i + n * offset ns is < n + n * offset ns is := by omega
        _ = n * (offset ns is + 1) := by ring
        _ ≤ n * prod ns := Nat.mul_le_mul_left n ho

theorem offset_coords : ∀ (ns : List ℕ) (k : ℕ), k < prod ns → offset ns (coords ns k) = k
  | [], k, h => by simp [prod] at h; simp [offset, coords, h]
  | n :: ns, k, h => by
      rw [prod_cons] at h
      have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp at h)
      simp only [coords, offset]
      rw [offset_coords ns (k / n) (by rw [Nat.div_lt_iff_lt_mul hn, Nat.mul_comm]; exact h)]
      exact Nat.mod_add_div k n

end CM

/-! ### The Fortran storage sequence -/

namespace F

/-- The Fortran coordinates of column-major position `k` (the `k`-th element
in array element order). -/
def coordAt (a : FArr) (k : ℕ) : List ℤ :=
  List.zipWith (fun (lb : ℤ) (i : ℕ) => lb + i) a.lbs (CM.coords a.extents k)

/-- The storage sequence of `a` from 0-based column-major position `off`
(F2018 15.5.2.11: the dummy's element sequence). -/
def storageFrom (a : FArr) (off : ℕ) : List SVal :=
  (List.range (a.size - off)).map fun k => a.get (coordAt a (off + k))

/-- An explicit-shape dummy of the given extents (unit lower bounds) laid over
the storage sequence from `off`. -/
def seqDummy (a : FArr) (off : ℕ) (extents : List ℕ) : FArr :=
  { elemTy := a.elemTy,
    dims := extents.map fun n => ((1 : ℤ), n),
    get := fun idx =>
      (storageFrom a off).getD (CM.offset extents (rebase (extents.map fun _ => (1 : ℤ)) idx)) (.int .i4 0) }

theorem size_eq_prod (a : FArr) : a.size = CM.prod a.extents := rfl

theorem storageFrom_length (a : FArr) (off : ℕ) : (storageFrom a off).length = a.size - off := by
  simp [storageFrom]

theorem storageFrom_getD (a : FArr) (off k : ℕ) (hk : k < a.size - off) (d : SVal) :
    (storageFrom a off).getD k d = a.get (coordAt a (off + k)) := by
  unfold storageFrom
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk]
  rfl

theorem seqDummy_lbs (a : FArr) (off : ℕ) (extents : List ℕ) :
    (seqDummy a off extents).lbs = extents.map fun _ => (1 : ℤ) := by
  simp [seqDummy, FArr.lbs, List.map_map, Function.comp_def]

theorem seqDummy_extents (a : FArr) (off : ℕ) (extents : List ℕ) :
    (seqDummy a off extents).extents = extents := by
  simp [seqDummy, FArr.extents, List.map_map, Function.comp_def]

/-- In-bounds coordinates of the dummy rebase to in-bounds 0-based coordinates. -/
theorem seqDummy_inBounds (a : FArr) (off : ℕ) (extents : List ℕ) (idx : List ℤ)
    (h : (seqDummy a off extents).inBounds idx) :
    PArr.inBoundsB extents (rebase (extents.map fun _ => (1 : ℤ)) idx) = true := by
  have := Rules.rebase_inBounds (seqDummy a off extents).dims idx h
  simp only [seqDummy, List.map_map, Function.comp_def] at this
  simpa [rebase] using this

/-- The column-major coordinates of a position below the size are in bounds. -/
theorem coordAt_inBounds : ∀ (dims : List (ℤ × ℕ)) (k : ℕ), k < CM.prod (dims.map Prod.snd) →
    FArr.inBoundsB dims (List.zipWith (fun (lb : ℤ) (i : ℕ) => lb + i) (dims.map Prod.fst)
      (CM.coords (dims.map Prod.snd) k)) = true
  | [], _, _ => rfl
  | (lb, n) :: ds, k, h => by
      simp only [List.map_cons, CM.prod_cons] at h
      have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp at h)
      simp only [List.map_cons, CM.coords, List.zipWith_cons_cons, FArr.inBoundsB, Bool.and_eq_true,
        decide_eq_true_eq]
      refine ⟨⟨by omega, ?_⟩, coordAt_inBounds ds (k / n) ?_⟩
      · have := Nat.mod_lt k hn; omega
      · rw [Nat.div_lt_iff_lt_mul hn, Nat.mul_comm]; exact h

theorem coordAt_inBounds' (a : FArr) (k : ℕ) (hk : k < a.size) : a.inBounds (coordAt a k) :=
  coordAt_inBounds a.dims k hk

end F

namespace Rel

theorem rebase_add : ∀ (lbs : List ℤ) (cs : List ℕ), lbs.length = cs.length →
    rebase lbs (List.zipWith (fun (lb : ℤ) (i : ℕ) => lb + i) lbs cs) = cs
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | lb :: lbs, c :: cs, h => by
      simp only [List.length_cons, Nat.add_right_cancel_iff] at h
      simp only [rebase, List.zipWith_cons_cons, List.cons.injEq]
      exact ⟨by simp, by simpa [rebase] using rebase_add lbs cs h⟩

/-- A related pair agrees along the storage sequence: the `k`-th Fortran element
is the `k`-th element of the column-major ravel. -/
theorem RelA.storage {a : FArr} {p : PArr} (h : RelA a p) (k : ℕ) (hk : k < a.size) :
    RelS (a.get (F.coordAt a k)) (p.get (CM.coords p.shape k)) := by
  have hin := F.coordAt_inBounds' a k hk
  have := h.elems _ hin
  have hlen : a.lbs.length = (CM.coords a.extents k).length := by
    simp [FArr.lbs, FArr.extents, CM.coords_length]
  rw [F.coordAt, rebase_add a.lbs _ hlen] at this
  rw [h.shape]; exact this

end Rel

/-! ### The Python side: ravel positions, views, reshapes -/

namespace P

theorem ravelF_length (p : PArr) : (ravelF p).length = p.size := by simp [ravelF]

theorem ravelF_getD (p : PArr) (k : ℕ) (hk : k < p.size) (d : PVal) :
    (ravelF p).getD k d = p.get (CM.coords p.shape k) := by
  unfold ravelF
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk]
  rfl

theorem size_eq_prod (p : PArr) : p.size = CM.prod p.shape := rfl

/-- `arr[:, k]` on a rank-2 array: the whole column `k`. -/
def viewColumn (p : PArr) (k : ℕ) : Option PArr :=
  match p.shape with
  | [n0, n1] => if k < n1 then some { shape := [n0], get := fun idx => p.get [idx.headD 0, k] } else none
  | _ => none

/-- `x[start:]` on a rank-1 array (`start ≤ n`). -/
def tail (p : PArr) (start : ℕ) : Option PArr :=
  match p.shape with
  | [n] => if start ≤ n then some { shape := [n - start], get := fun idx => p.get [start + idx.headD 0] } else none
  | _ => none

/-- `np.reshape(flat, extents, order='F')` of a flat sequence whose length is the product. -/
def reshapeF (flat : List PVal) (extents : List ℕ) : PArr :=
  { shape := extents, get := fun idx => flat.getD (CM.offset extents idx) .none }

/-- `np.reshape(np.ravel(a, order='F')[:span], extents, order='F')`: NumPy raises unless
`span` elements are available and `prod extents = span`. -/
def reshapeTake (p : PArr) (span : ℕ) (extents : List ℕ) : Option PArr :=
  if span ≤ p.size ∧ CM.prod extents = span then some (reshapeF ((ravelF p).take span) extents) else none

/-- `np.reshape(a, (leading..., -1), order='F')`: NumPy infers the last extent and raises
unless the size is a multiple of the leading block. -/
def reshapeNeg1 (p : PArr) (leading : List ℕ) : Option PArr :=
  let block := CM.prod leading
  if 0 < block ∧ block ∣ p.size then some (reshapeF (ravelF p) (leading ++ [p.size / block])) else none

/-- `arr[r:, c:]` on a rank-2 array (NumPy clips the empty case). -/
def viewBlock (p : PArr) (n0 n1 r c : ℕ) : PArr :=
  { shape := [n0 - r, n1 - c], get := fun idx => p.get [r + idx.getD 0 0, c + idx.getD 1 0] }

/-- `_f_seq_tail(arr, off, *leading)` (runtime.py:210-238): the rank-2 special case
returns the view `arr[off % e0:, off // e0:]` when the single leading extent equals
the actual's first extent; otherwise the column-major tail, reshaped onto whole
columns (`whole = (size // block) * block`) when leading extents are given. -/
def seqTail (p : PArr) (off : ℕ) (leading : List ℕ) : Option PArr :=
  match leading, p.shape with
  | [e0], [n0, n1] =>
    if e0 = n0 ∧ 0 < n0 then some (viewBlock p n0 n1 (off % e0) (off / e0))
    else seqTailGeneral p off leading
  | _, _ => seqTailGeneral p off leading
where
  seqTailGeneral (p : PArr) (off : ℕ) (leading : List ℕ) : Option PArr :=
    let flat := (ravelF p).drop off
    match leading with
    | [] => some { shape := [flat.length], get := fun idx => flat.getD (idx.headD 0) .none }
    | _ =>
      let block := CM.prod leading
      if block = 0 then none
      else
        let whole := (flat.length / block) * block
        some (reshapeF (flat.take whole) (leading ++ [whole / block]))

end P

namespace Rules

open Recast Rel

/-- Elementwise relation of two sequences (same length, related elements). -/
def SeqRel (l : List SVal) (l' : List PVal) : Prop :=
  l.length = l'.length ∧ ∀ (k : ℕ) (d : SVal) (d' : PVal), k < l.length → RelS (l.getD k d) (l'.getD k d')

/-! ### FNP-R0054: `arr(1, k)` to a rank-1 dummy -> `arr[:, k - 1]` -/

/-- The whole column `k` (0-based) of a rank-2 array is the storage sequence
from offset `k * n0` laid over extents `[n0]`. -/
def FNP_R0054_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ (l0 l1 : ℤ) (n0 n1 : ℕ), a.dims = [(l0, n0), (l1, n1)] →
    ∀ k, k < n1 → ∃ q, P.viewColumn p k = some q ∧ RelA (F.seqDummy a (k * n0) [n0]) q

theorem coords_two (n0 n1 j k : ℕ) (hj : j < n0) (hk : k < n1) :
    CM.coords [n0, n1] (k * n0 + j) = [j, k] := by
  have hn0 : 0 < n0 := by omega
  simp only [CM.coords, List.cons.injEq, and_true]
  constructor
  · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj]
  · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ hn0, Nat.div_eq_of_lt hj, Nat.zero_add,
      Nat.mod_eq_of_lt hk]

theorem FNP_R0054_sound : FNP_R0054_obligation := by
  intro a p h l0 l1 n0 n1 hd k hk
  have hshape : p.shape = [n0, n1] := by rw [h.shape]; simp [FArr.extents, hd]
  have hsize : a.size = n0 * n1 := by simp [FArr.size, FArr.extents, hd]
  refine ⟨{ shape := [n0], get := fun idx => p.get [idx.headD 0, k] }, by simp [P.viewColumn, hshape, hk], ?_⟩
  refine ⟨by rw [F.seqDummy_extents], ?_⟩
  intro idx hin
  have hb := F.seqDummy_inBounds a (k * n0) [n0] idx hin
  rw [F.seqDummy_lbs]
  obtain ⟨j, hj, hidx⟩ : ∃ j, j < n0 ∧ rebase (List.map (fun _ => (1 : ℤ)) [n0]) idx = [j] := by
    match hr : rebase (List.map (fun _ => (1 : ℤ)) [n0]) idx, hb with
    | [j], hb => exact ⟨j, by simpa [hr, PArr.inBoundsB] using hb, rfl⟩
    | [], hb => simp [hr, PArr.inBoundsB] at hb
    | _ :: _ :: _, hb => simp [hr, PArr.inBoundsB] at hb
  show RelS ((F.storageFrom a (k * n0)).getD (CM.offset [n0] (rebase _ idx)) _) _
  rw [hidx]
  simp only [CM.offset, Nat.mul_zero, Nat.add_zero]
  have h1 : n0 * k + n0 ≤ n0 * n1 := by
    have := Nat.mul_le_mul_left n0 hk; rwa [Nat.mul_succ] at this
  have hlt : j < a.size - k * n0 := by rw [hsize, Nat.mul_comm k n0]; omega
  rw [F.storageFrom_getD a (k * n0) j hlt]
  have := RelA.storage h (k * n0 + j) (by rw [hsize, Nat.mul_comm k n0]; omega)
  rw [hshape, coords_two n0 n1 j k hj hk] at this
  simpa using this

/-! ### FNP-R0056: `x(i)` to `x(*)` -> `x[(i - lb):]` -/

/-- The tail from `start` is the storage sequence from `start` laid over `[n - start]`. -/
def FNP_R0056_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ (lb : ℤ) (n : ℕ), a.dims = [(lb, n)] →
    ∀ start, start ≤ n → ∃ q, P.tail p start = some q ∧ RelA (F.seqDummy a start [n - start]) q

theorem FNP_R0056_sound : FNP_R0056_obligation := by
  intro a p h lb n hd start hs
  have hshape : p.shape = [n] := by rw [h.shape]; simp [FArr.extents, hd]
  have hsize : a.size = n := by simp [FArr.size, FArr.extents, hd]
  refine ⟨{ shape := [n - start], get := fun idx => p.get [start + idx.headD 0] }, by simp [P.tail, hshape, hs], ?_⟩
  refine ⟨by rw [F.seqDummy_extents], ?_⟩
  intro idx hin
  have hb := F.seqDummy_inBounds a start [n - start] idx hin
  rw [F.seqDummy_lbs]
  obtain ⟨j, hj, hidx⟩ : ∃ j, j < n - start ∧ rebase (List.map (fun _ => (1 : ℤ)) [n - start]) idx = [j] := by
    match hr : rebase (List.map (fun _ => (1 : ℤ)) [n - start]) idx, hb with
    | [j], hb => exact ⟨j, by simpa [hr, PArr.inBoundsB] using hb, rfl⟩
    | [], hb => simp [hr, PArr.inBoundsB] at hb
    | _ :: _ :: _, hb => simp [hr, PArr.inBoundsB] at hb
  show RelS ((F.storageFrom a start).getD (CM.offset [n - start] (rebase _ idx)) _) _
  rw [hidx]
  simp only [CM.offset, Nat.mul_zero, Nat.add_zero]
  rw [F.storageFrom_getD a start j (by omega)]
  have := RelA.storage h (start + j) (by omega)
  rw [hshape] at this
  simp only [CM.coords] at this
  rw [Nat.mod_eq_of_lt (by omega : start + j < n)] at this
  simpa using this

/-! ### FNP-R0058: rank-1 `w` to `a(mda, n)` -> `np.reshape(w[:mda*n], (mda, n), order='F')` -/

/-- The first `prod extents` elements of the storage laid over `extents`. -/
def FNP_R0058_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ (extents : List ℕ), CM.prod extents ≤ a.size →
    ∃ q, P.reshapeTake p (CM.prod extents) extents = some q ∧ RelA (F.seqDummy a 0 extents) q

theorem FNP_R0058_sound : FNP_R0058_obligation := by
  intro a p h extents hle
  have hpsize : p.size = a.size := by simp [PArr.size, FArr.size, h.shape]
  refine ⟨P.reshapeF ((P.ravelF p).take (CM.prod extents)) extents, by simp [P.reshapeTake, hpsize, hle], ?_⟩
  refine ⟨by rw [F.seqDummy_extents]; rfl, ?_⟩
  intro idx hin
  have hb := F.seqDummy_inBounds a 0 extents idx hin
  rw [F.seqDummy_lbs]
  have hoff := CM.offset_lt_prod extents _ hb
  show RelS ((F.storageFrom a 0).getD (CM.offset extents (rebase _ idx)) _)
    (((P.ravelF p).take (CM.prod extents)).getD (CM.offset extents (rebase _ idx)) .none)
  set m := CM.offset extents (rebase (extents.map fun _ => (1 : ℤ)) idx) with hm
  have hm_size : m < a.size := lt_of_lt_of_le hoff hle
  rw [F.storageFrom_getD a 0 m (by omega)]
  rw [List.getD_eq_getElem?_getD, List.getElem?_take, if_pos hoff, ← List.getD_eq_getElem?_getD,
    P.ravelF_getD p m (by rw [hpsize]; exact hm_size)]
  simpa using RelA.storage h m hm_size

/-! ### The assumed-size cases: the dummy is the remaining storage sequence -/

/-- `_f_seq_tail(arr, 0)` with no leading extents: the whole storage as a rank-1 array
(FNP-R0059, `call f(a)` with `c(*)`). -/
def FNP_R0059_flat_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p →
    ∃ q, P.seqTail p 0 [] = some q ∧ SeqRel (F.storageFrom a 0) (P.ravelF q)

theorem seqTail_nil (p : PArr) (off : ℕ) : P.seqTail p off [] =
    some { shape := [((P.ravelF p).drop off).length], get := fun idx => ((P.ravelF p).drop off).getD (idx.headD 0) .none } := by
  unfold P.seqTail
  cases hs : p.shape with
  | nil => rfl
  | cons n0 rest => cases rest with
    | nil => rfl
    | cons n1 rest' => cases rest' <;> rfl

theorem ravelF_rank1_getD (q : PArr) (L : ℕ) (hs : q.shape = [L]) (k : ℕ) (hk : k < L) (d : PVal) :
    (P.ravelF q).getD k d = q.get [k] := by
  rw [P.ravelF_getD q k (by simp [PArr.size, hs]; exact hk), hs]
  simp [CM.coords, Nat.mod_eq_of_lt hk]

theorem FNP_R0059_flat_sound : FNP_R0059_flat_obligation := by
  intro a p h
  have hpsize : p.size = a.size := by simp [PArr.size, FArr.size, h.shape]
  refine ⟨_, seqTail_nil p 0, ?_⟩
  simp only [List.drop_zero]
  constructor
  · rw [F.storageFrom_length, P.ravelF_length]
    show a.size - 0 = [(P.ravelF p).length].foldr (· * ·) 1
    simp [P.ravelF_length, hpsize]
  · intro k d d' hk
    rw [F.storageFrom_length] at hk
    rw [ravelF_rank1_getD _ (P.ravelF p).length rfl k (by rw [P.ravelF_length, hpsize]; omega)]
    simp only [List.headD_cons]
    rw [F.storageFrom_getD a 0 k hk, P.ravelF_getD p k (by rw [hpsize]; omega)]
    simpa using RelA.storage h k (by omega)

/-- `np.reshape(vl, (ldvl, -1), order='F')` for `vl(ldvl, *)` (FNP-R0059): the whole
storage over `ldvl` rows and as many columns as fit -- NumPy raises unless the size is
a multiple of `ldvl`. -/
def FNP_R0059_reshape_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ (leading : List ℕ), 0 < CM.prod leading → CM.prod leading ∣ a.size →
    ∃ q, P.reshapeNeg1 p leading = some q ∧ SeqRel (F.storageFrom a 0) (P.ravelF q)

/-- Ravel of a Fortran-order reshape of a flat sequence of the right length is the sequence. -/
theorem ravelF_reshapeF_getD (flat : List PVal) (extents : List ℕ) (hlen : flat.length = CM.prod extents)
    (k : ℕ) (hk : k < CM.prod extents) (d : PVal) :
    (P.ravelF (P.reshapeF flat extents)).getD k d = flat.getD k .none := by
  rw [P.ravelF_getD _ k (by simp [PArr.size, P.reshapeF]; exact hk)]
  simp only [P.reshapeF]
  rw [CM.offset_coords extents k hk]

theorem FNP_R0059_reshape_sound : FNP_R0059_reshape_obligation := by
  intro a p h leading hpos hdvd
  have hpsize : p.size = a.size := by simp [PArr.size, FArr.size, h.shape]
  refine ⟨_, by simp only [P.reshapeNeg1]; rw [if_pos ⟨hpos, by rw [hpsize]; exact hdvd⟩], ?_⟩
  have hprod : CM.prod (leading ++ [p.size / CM.prod leading]) = p.size := by
    rw [CM.prod_append_singleton, Nat.mul_div_cancel' (by rw [hpsize]; exact hdvd)]
  constructor
  · rw [F.storageFrom_length, P.ravelF_length]
    show a.size - 0 = CM.prod (leading ++ [p.size / CM.prod leading])
    rw [hprod, hpsize]; simp
  · intro k d d' hk
    rw [F.storageFrom_length] at hk
    rw [ravelF_reshapeF_getD _ _ (by rw [P.ravelF_length, hprod]) k (by rw [hprod, hpsize]; omega)]
    rw [F.storageFrom_getD a 0 k hk, P.ravelF_getD p k (by rw [hpsize]; omega)]
    simpa using RelA.storage h k (by omega)

/-- Without divisibility the reshape is undefined on the Python side while the Fortran
dummy (the remaining storage, F2018 8.5.8.5, with a partial last column) exists:
`vl(5)` passed to `vl(2, *)`. -/
theorem FNP_R0059_reshape_undefined (p : PArr) (hs : p.shape = [5]) : P.reshapeNeg1 p [2] = none := by
  simp [P.reshapeNeg1, PArr.size, hs, CM.prod]

/-! ### FNP-R0057: `_f_seq_tail(arr, start, *leading)` -/

/-- The obligation as the rule records it: the result's element sequence is the
storage from `start`. -/
def FNP_R0057_obligation : Prop :=
  ∀ (a : FArr) (p : PArr), RelA a p → ∀ (off : ℕ) (leading : List ℕ), 0 < CM.prod leading → off ≤ a.size →
    ∃ q, P.seqTail p off leading = some q ∧ SeqRel (F.storageFrom a off) (P.ravelF q)

/-- A 3×2 INTEGER array with `a(i, j) = i + 3 (j - 1)` (storage `1, 2, 3, 4, 5, 6`). -/
def threeTwoF : FArr :=
  { elemTy := .int .i4, dims := [(1, 3), (1, 2)],
    get := fun idx => .int .i4 (idx.getD 0 0 + 3 * (idx.getD 1 0 - 1)) }

def threeTwoP : PArr := { shape := [3, 2], get := fun idx => .int ((idx.getD 0 0 : ℕ) + 1 + 3 * (idx.getD 1 0 : ℕ)) }

theorem threeTwoRel : RelA threeTwoF threeTwoP := by
  refine ⟨rfl, ?_⟩
  intro idx hin
  simp only [FArr.inBounds, FArr.inBoundsB, threeTwoF] at hin
  match idx, hin with
  | [i, j], hin =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hin
    simp only [threeTwoF, threeTwoP, rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
      List.zipWith_nil_right, List.getD_cons_zero, List.getD_cons_succ]
    have : (i + 3 * (j - 1) : ℤ) = (((i - 1).toNat : ℕ) : ℤ) + 1 + 3 * (((j - 1).toNat : ℕ) : ℤ) := by omega
    rw [this]
    exact_mod_cast RelS.int
  | [], hin => simp [FArr.inBoundsB] at hin
  | [_], hin => simp [FArr.inBoundsB] at hin
  | _ :: _ :: _ :: _, hin => simp [FArr.inBoundsB] at hin

/-- FNP-D0041: `a(2, 1)` passed to `u(3, *)`: the dummy's sequence is `2, 3, 4, 5, 6`
(five elements, `u(3, 1) = a(1, 2) = 4`); `_f_seq_tail(a, 1, 3)` returns the view
`a[1:, 0:]` = `[[2, 5], [3, 6]]` (four elements, no `4`). -/
theorem FNP_R0057_refuted : ¬ FNP_R0057_obligation := by
  intro h
  obtain ⟨q, hq, hrel⟩ := h threeTwoF threeTwoP threeTwoRel 1 [3] (by decide) (by decide)
  have hq' : q = P.viewBlock threeTwoP 3 2 (1 % 3) (1 / 3) := by
    simp [P.seqTail, threeTwoP] at hq
    exact hq.symm
  have hlen := hrel.1
  rw [F.storageFrom_length, P.ravelF_length, hq'] at hlen
  simp [PArr.size, P.viewBlock, FArr.size, FArr.extents, threeTwoF] at hlen

/-- The view special case is sound when the element sits at the top of a column
(`off % n0 = 0`, i.e. `a(1, j)`): the view `arr[0:, c:]` is the storage from `c * n0`. -/
theorem FNP_R0057_view_sound_aligned (a : FArr) (p : PArr) (h : RelA a p) (l0 l1 : ℤ) (n0 n1 : ℕ)
    (hd : a.dims = [(l0, n0), (l1, n1)]) (hn0 : 0 < n0) (c : ℕ) (hc : c ≤ n1) :
    ∃ q, P.seqTail p (c * n0) [n0] = some q ∧ SeqRel (F.storageFrom a (c * n0)) (P.ravelF q) := by
  have hshape : p.shape = [n0, n1] := by rw [h.shape]; simp [FArr.extents, hd]
  have hsize : a.size = n0 * n1 := by simp [FArr.size, FArr.extents, hd]
  have hmod : c * n0 % n0 = 0 := by simp
  have hdiv : c * n0 / n0 = c := Nat.mul_div_cancel c hn0
  have hcn : c * n0 = n0 * c := Nat.mul_comm c n0
  refine ⟨P.viewBlock p n0 n1 0 c, ?_, ?_⟩
  · simp [P.seqTail, hshape, hmod, hdiv, hn0]
  · constructor
    · simp only [F.storageFrom_length, P.ravelF_length, PArr.size, P.viewBlock, List.foldr_cons,
        List.foldr_nil, hsize, Nat.sub_zero, Nat.mul_one]
      rw [Nat.mul_sub, Nat.mul_comm n0 c]
    · intro k d d' hk
      rw [F.storageFrom_length, hsize, hcn] at hk
      have hk' : k < n0 * (n1 - c) := by rw [Nat.mul_sub]; omega
      have hq : (P.ravelF (P.viewBlock p n0 n1 0 c)).getD k d' = p.get [k % n0, c + k / n0 % (n1 - c)] := by
        rw [P.ravelF_getD _ k (by simp [PArr.size, P.viewBlock]; exact hk')]
        simp [P.viewBlock, CM.coords]
      rw [hq, F.storageFrom_getD a (c * n0) k (by omega)]
      have hkd : k / n0 < n1 - c := by rw [Nat.div_lt_iff_lt_mul hn0, Nat.mul_comm]; exact hk'
      have hcoords : CM.coords [n0, n1] (c * n0 + k) = [k % n0, c + k / n0] := by
        simp only [CM.coords, List.cons.injEq, and_true]
        constructor
        · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
        · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ hn0, Nat.add_comm,
            Nat.mod_eq_of_lt (by omega)]
      have := RelA.storage h (c * n0 + k) (by rw [hsize, hcn]; rw [Nat.mul_sub] at hk'; omega)
      rw [hshape, hcoords] at this
      rw [Nat.mod_eq_of_lt hkd]
      simpa using this

/-- Sub-block soundness of the view: every element the view *has* is the right one
(`u(i, j)` with `i ≤ n0 - off % n0`, `j ≤ n1 - off / n0`); the dummy's remaining elements
(the rest of each column, e.g. `u(3, 1)` above) are simply not present. This is the
premise under which a BLAS-style callee that walks one row (`h12(..., a(i, 1), mda, ...)`)
is served correctly. -/
theorem FNP_R0057_view_subblock (a : FArr) (p : PArr) (h : RelA a p) (l0 l1 : ℤ) (n0 n1 : ℕ)
    (hd : a.dims = [(l0, n0), (l1, n1)]) (hn0 : 0 < n0) (off : ℕ) (i j : ℕ)
    (hi : i < n0 - off % n0) (hj : j < n1 - off / n0) (d : SVal) :
    RelS ((F.storageFrom a off).getD (i + j * n0) d) ((P.viewBlock p n0 n1 (off % n0) (off / n0)).get [i, j]) := by
  have hshape : p.shape = [n0, n1] := by rw [h.shape]; simp [FArr.extents, hd]
  have hsize : a.size = n0 * n1 := by simp [FArr.size, FArr.extents, hd]
  have hr := Nat.mod_lt off hn0
  have hat : off = off % n0 + n0 * (off / n0) := (Nat.mod_add_div off n0).symm
  have hlt : off + (i + j * n0) < n0 * n1 := by
    have h1 : off % n0 + i < n0 := by omega
    have h2 : off / n0 + j < n1 := by omega
    calc off + (i + j * n0) = (off % n0 + i) + n0 * (off / n0 + j) := by nth_rw 1 [hat]; ring
      _ < n0 + n0 * (off / n0 + j) := by omega
      _ = n0 * (off / n0 + j + 1) := by ring
      _ ≤ n0 * n1 := Nat.mul_le_mul_left n0 h2
  rw [F.storageFrom_getD a off (i + j * n0) (by omega)]
  have hcoords : CM.coords [n0, n1] (off + (i + j * n0)) = [off % n0 + i, off / n0 + j] := by
    have e : off + (i + j * n0) = (off % n0 + i) + n0 * (off / n0 + j) := by nth_rw 1 [hat]; ring
    rw [e]
    simp only [CM.coords, List.cons.injEq, and_true]
    constructor
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (by omega)]
    · rw [Nat.add_mul_div_left _ _ hn0, Nat.div_eq_of_lt (by omega), Nat.zero_add, Nat.mod_eq_of_lt (by omega)]
  have := RelA.storage h (off + (i + j * n0)) (by rw [hsize]; exact hlt)
  rw [hshape, hcoords] at this
  simpa [P.viewBlock] using this

/-- The general branch (`ravel(F)[off:]` reshaped onto whole columns) is sound exactly when
the remaining storage is a multiple of the leading block. -/
theorem FNP_R0057_general_sound (a : FArr) (p : PArr) (h : RelA a p) (off : ℕ) (leading : List ℕ)
    (hne : leading ≠ []) (hpos : 0 < CM.prod leading) (hat : off ≤ a.size)
    (hdvd : CM.prod leading ∣ (a.size - off))
    (hnotview : ∀ e0 n0 n1, leading = [e0] → p.shape = [n0, n1] → ¬ (e0 = n0 ∧ 0 < n0)) :
    ∃ q, P.seqTail p off leading = some q ∧ SeqRel (F.storageFrom a off) (P.ravelF q) := by
  have hpsize : p.size = a.size := by simp [PArr.size, FArr.size, h.shape]
  have hflat : ((P.ravelF p).drop off).length = a.size - off := by rw [List.length_drop, P.ravelF_length, hpsize]
  have hwhole : ((a.size - off) / CM.prod leading) * CM.prod leading = a.size - off := Nat.div_mul_cancel hdvd
  have hgen : P.seqTail p off leading = P.seqTail.seqTailGeneral p off leading := by
    unfold P.seqTail
    match hl : leading, hs : p.shape with
    | [], _ => rfl
    | [e0], [] => rfl
    | [e0], [_] => rfl
    | [e0], [n0, n1] =>
      dsimp only
      rw [if_neg (hnotview e0 n0 n1 rfl hs)]
    | [e0], _ :: _ :: _ :: _ => rfl
    | _ :: _ :: _, _ => rfl
  have hgen' : P.seqTail.seqTailGeneral p off leading =
      some (P.reshapeF (((P.ravelF p).drop off).take (a.size - off)) (leading ++ [(a.size - off) / CM.prod leading])) := by
    cases leading with
    | nil => exact absurd rfl hne
    | cons l ls =>
      have hne0 : CM.prod (l :: ls) ≠ 0 := Nat.pos_iff_ne_zero.mp hpos
      simp only [P.seqTail.seqTailGeneral]
      rw [if_neg hne0, hflat, hwhole]
  rw [hgen, hgen']
  refine ⟨_, rfl, ?_⟩
  have hprod : CM.prod (leading ++ [(a.size - off) / CM.prod leading]) = a.size - off := by
    rw [CM.prod_append_singleton, Nat.mul_comm, hwhole]
  have htake : (((P.ravelF p).drop off).take (a.size - off)).length = a.size - off := by
    rw [List.length_take, hflat, Nat.min_self]
  constructor
  · rw [F.storageFrom_length, P.ravelF_length]
    show a.size - off = CM.prod (leading ++ [(a.size - off) / CM.prod leading])
    rw [hprod]
  · intro k d d' hk
    rw [F.storageFrom_length] at hk
    rw [ravelF_reshapeF_getD _ _ (by rw [htake, hprod]) k (by rw [hprod]; exact hk)]
    rw [F.storageFrom_getD a off k hk]
    rw [List.getD_eq_getElem?_getD, List.getElem?_take, if_pos hk, List.getElem?_drop,
      ← List.getD_eq_getElem?_getD, P.ravelF_getD p (off + k) (by rw [hpsize]; omega)]
    simpa using RelA.storage h (off + k) (by omega)

/-- FNP-D0042: the whole-column truncation. `x(5)` (rank 1) passed as `x(1)` to
`u(2, *)`: the dummy's sequence has five elements (`u(1, 3) = x(5)` is addressable,
F2018 9.5.3.2); `_f_seq_tail(x, 0, 2)` keeps `(5 // 2) * 2 = 4` of them. -/
def fiveF : FArr := { elemTy := .int .i4, dims := [(1, 5)], get := fun idx => .int .i4 (idx.getD 0 0) }
def fiveP : PArr := { shape := [5], get := fun idx => .int ((idx.getD 0 0 : ℕ) + 1) }

theorem fiveRel : RelA fiveF fiveP := by
  refine ⟨rfl, ?_⟩
  intro idx hin
  simp only [FArr.inBounds, FArr.inBoundsB, fiveF] at hin
  match idx, hin with
  | [i], hin =>
    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hin
    simp only [fiveF, fiveP, rebase, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
      List.zipWith_nil_right, List.getD_cons_zero]
    have e : (((i - 1).toNat : ℕ) : ℤ) + 1 = i := by omega
    rw [e]
    exact RelS.int
  | [], hin => simp [FArr.inBoundsB] at hin
  | _ :: _ :: _, hin => simp [FArr.inBoundsB] at hin

theorem FNP_R0057_trunc_refuted : ¬ FNP_R0057_obligation := by
  intro h
  obtain ⟨q, hq, hrel⟩ := h fiveF fiveP fiveRel 0 [2] (by decide) (by decide)
  have hlen := hrel.1
  rw [F.storageFrom_length, P.ravelF_length] at hlen
  simp [P.seqTail, P.seqTail.seqTailGeneral, fiveP, P.ravelF, PArr.size, CM.prod] at hq
  rw [← hq] at hlen
  simp [PArr.size, P.reshapeF, FArr.size, FArr.extents, fiveF] at hlen

end Rules

end Recast
