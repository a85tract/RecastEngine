import RecastProofs.Semantics.Values

/-!
# Arrays: Fortran arrays with lower bounds, Python (NumPy) arrays, and the
coordinate/storage relation

* A Fortran array carries per-axis `(lower bound, extent)` and an element
  function on integer coordinates.  Only in-bounds coordinates are meaningful.
* A Python array carries a `shape` and an element function on natural
  (0-based) coordinates.
* Column-major linearisation (`Recast.CM.offset`) is the storage relation used
  by the sequence-association rules and by `ravel(order='F')`.
* `Recast.Py.sliceIndices` is CPython's slice semantics (`PySlice_AdjustIndices`),
  including negative-index wrap, clipping, and the `None` defaults that
  depend on the sign of the step.  It is *Python's* meaning of `a[lo:hi:st]`,
  defined independently of anything Fortran says.
-/

namespace Recast

/-- A Fortran array value. -/
structure FArr where
  elemTy : FTy
  /-- `(lower bound, extent)` per axis. -/
  dims : List (ℤ × ℕ)
  get : List ℤ → SVal

namespace FArr

def rank (a : FArr) : ℕ := a.dims.length

def extents (a : FArr) : List ℕ := a.dims.map Prod.snd

def lbs (a : FArr) : List ℤ := a.dims.map Prod.fst

/-- `UBOUND` per axis: `lb + extent - 1`. -/
def ubs (a : FArr) : List ℤ := a.dims.map fun d => d.1 + d.2 - 1

def size (a : FArr) : ℕ := a.extents.foldr (· * ·) 1

def inBoundsB : List (ℤ × ℕ) → List ℤ → Bool
  | [], [] => true
  | (lb, n) :: ds, i :: is => decide (lb ≤ i ∧ i < lb + n) && inBoundsB ds is
  | _, _ => false

def inBounds (a : FArr) (idx : List ℤ) : Prop := FArr.inBoundsB a.dims idx = true

end FArr

/-- A Python/NumPy array value (0-based). -/
structure PArr where
  shape : List ℕ
  get : List ℕ → PVal

namespace PArr

def rank (a : PArr) : ℕ := a.shape.length

def size (a : PArr) : ℕ := a.shape.foldr (· * ·) 1

def inBoundsB : List ℕ → List ℕ → Bool
  | [], [] => true
  | n :: ns, i :: is => decide (i < n) && inBoundsB ns is
  | _, _ => false

def inBounds (a : PArr) (idx : List ℕ) : Prop := inBoundsB a.shape idx = true

end PArr

namespace CM

/-- Column-major offset of 0-based coordinates `idx` in an array of `extents`:
`i₀ + n₀ * (i₁ + n₁ * (i₂ + …))`. -/
def offset : List ℕ → List ℕ → ℕ
  | [], _ => 0
  | _, [] => 0
  | n :: ns, i :: is => i + n * offset ns is

/-- Inverse of `offset`: 0-based coordinates of column-major position `k`. -/
def coords : List ℕ → ℕ → List ℕ
  | [], _ => []
  | n :: ns, k => (k % n) :: coords ns (k / n)

theorem offset_nil (is : List ℕ) : offset [] is = 0 := by cases is <;> rfl

/-- `coords` inverts `offset` on in-bounds coordinates. -/
theorem coords_offset : ∀ (ns is : List ℕ), PArr.inBoundsB ns is = true →
    coords ns (offset ns is) = is
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [PArr.inBoundsB] at h
  | _ :: _, [], h => by simp [PArr.inBoundsB] at h
  | n :: ns, i :: is, h => by
    simp only [PArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hi, hrest⟩ := h
    have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le i) hi
    simp only [coords, offset, List.cons.injEq]
    refine ⟨?_, ?_⟩
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hi]
    · rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hi, Nat.zero_add]
      exact coords_offset ns is hrest

end CM

namespace Py

/-- CPython `PySlice_AdjustIndices` followed by the index enumeration of
`range(start, stop, step)`.  `n` is the length of the axis.  `step ≠ 0` is
assumed (Python raises otherwise; the caller returns `none`). -/
def adjust (n : ℕ) (start stop : Option ℤ) (step : ℤ) : ℤ × ℤ :=
  let len : ℤ := n
  let s0 : ℤ :=
    match start with
    | none => if step < 0 then len - 1 else 0
    | some s =>
      let s' := if s < 0 then s + len else s
      if s' < 0 then (if step < 0 then -1 else 0)
      else if s' ≥ len then (if step < 0 then len - 1 else len)
      else s'
  let e0 : ℤ :=
    match stop with
    | none => if step < 0 then -1 else len
    | some e =>
      let e' := if e < 0 then e + len else e
      if e' < 0 then (if step < 0 then -1 else 0)
      else if e' ≥ len then (if step < 0 then len - 1 else len)
      else e'
  (s0, e0)

/-- Number of indices `range(start, stop, step)` produces (CPython `_PySlice_GetLongIndices`). -/
def sliceLen (start stop step : ℤ) : ℕ :=
  if step > 0 then
    (if start < stop then ((stop - start - 1) / step + 1) else 0).toNat
  else
    (if stop < start then ((start - stop - 1) / (-step) + 1) else 0).toNat

/-- The 0-based indices selected by `a[start:stop:step]` on an axis of length `n`. -/
def sliceIndices (n : ℕ) (start stop : Option ℤ) (step : ℤ) : List ℕ :=
  let (s, e) := adjust n start stop step
  List.map (fun (k : ℕ) => (s + (k : ℤ) * step).toNat) (List.range (sliceLen s e step))

/-- A Python scalar index with negative wrap: `a[i]` reads position `i + n` for `i < 0`. -/
def wrapIndex (n : ℕ) (i : ℤ) : Option ℕ :=
  let i' := if i < 0 then i + n else i
  if 0 ≤ i' ∧ i' < n then some i'.toNat else none

end Py

namespace Fort

/-- Fortran trip count of `lo, hi, st` (`MAX((hi - lo + st) / st, 0)` with
truncating division; F2018 11.1.7.4.1).  `st ≠ 0` assumed. -/
def tripCount (lo hi st : ℤ) : ℕ := (max (Exact.tquot (hi - lo + st) st) 0).toNat

/-- The 1-based-in-the-array's-own-coordinates positions a section triplet selects. -/
def sectionCoords (lo hi st : ℤ) : List ℤ :=
  List.map (fun (k : ℕ) => lo + (k : ℤ) * st) (List.range (tripCount lo hi st))

theorem tripCount_unit (lo hi : ℤ) : tripCount lo hi 1 = (max (hi - lo + 1) 0).toNat := by
  simp [tripCount, Exact.tquot, Int.tdiv_one]

end Fort

end Recast
