import RecastProofs.Syntax.Fortran
import RecastProofs.Semantics.Arrays
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Exact-arithmetic semantics of the Fortran fragment (`F_math`)

An evaluation *function* `eval` from expressions to `Option Val`: `none` is
"undefined" (division by zero, out-of-bounds subscript, non-conforming
shapes, a domain error of an intrinsic, an ill-typed operand).  Partiality is
therefore explicit and is never totalised the way Lean's `/` on `ℝ` is.

Statement execution is a big-step relation `Exec` with control outcomes
`normal`, `exit`, `cycle`; a `DO` loop evaluates its bounds once, runs the
Fortran trip count of iterations, and leaves the index one step past the
last value on normal completion (F2018 11.1.7.4).

Kinds are retained on values.  Default `INTEGER` is `i4`; default `REAL` is
`r4` (single precision, as the engine's frontend also assumes for an
unsuffixed literal).  In the exact model the kind never changes a numeric
value; it only answers inquiries (`KIND`, `HUGE`, ...) and fixes the width
of the bit intrinsics.
-/

namespace Recast

def SVal.ty : SVal → FTy
  | .int k _ => .int k
  | .real k _ => .real k
  | .bool _ => .bool
  | .str _ => .str

/-- `HUGE` of a real kind: the largest finite binary32/binary64 value, as a real. -/
noncomputable def RKind.hugeR : RKind → ℝ
  | .r4 => (2 - 2 ^ (-(23 : ℤ))) * 2 ^ (127 : ℤ)
  | .r8 => (2 - 2 ^ (-(52 : ℤ))) * 2 ^ (1023 : ℤ)

noncomputable def RKind.tinyR : RKind → ℝ
  | .r4 => 2 ^ (-(126 : ℤ))
  | .r8 => 2 ^ (-(1022 : ℤ))

noncomputable def RKind.epsR : RKind → ℝ
  | .r4 => 2 ^ (-(23 : ℤ))
  | .r8 => 2 ^ (-(52 : ℤ))

def IKind.join : IKind → IKind → IKind
  | .i8, _ => .i8
  | _, .i8 => .i8
  | _, _ => .i4

def RKind.join : RKind → RKind → RKind
  | .r8, _ => .r8
  | _, .r8 => .r8
  | _, _ => .r4

end Recast

namespace Recast.F

open Recast Exact

noncomputable section

inductive Val where
  | s (v : SVal)
  | arr (a : FArr)

abbrev State := String → Option Val

/-- Uninterpreted pure real primitives (erf, gamma, ...), shared by both
semantics and used by congruence. -/
structure Oracle where
  prim : String → List ℝ → Option ℝ
  /-- The subprograms a `CALL` can reach (F2018 15.5.2). -/
  procs : ProcEnv := fun _ => none

/-- Fortran integer power with a negative exponent (F2018 10.1.5.2.1: the
result is `1 / (a ** |b|)` with integer division). Undefined for `a = 0`. -/
def ipowNeg (a : ℤ) (b : ℤ) : Option ℤ :=
  if a = 0 then none
  else some (tquot 1 (a ^ (-b).toNat))

/-- Real power. Defined for a positive base, or a zero base with a positive
exponent (F2018 10.1.5.2.4: a negative real base is a domain error). -/
noncomputable def rpowF (a b : ℝ) : Option ℝ :=
  if 0 < a then some (Real.rpow a b)
  else if a = 0 ∧ 0 < b then some 0
  else none

/-- Real base, integer exponent: `zpow`, undefined for `0 ** n` with `n ≤ 0`. -/
noncomputable def zpowF (a : ℝ) (n : ℤ) : Option ℝ :=
  if a = 0 ∧ n ≤ 0 then none else some (a ^ n)

/-- Scalar binary arithmetic with Fortran's type promotion. -/
noncomputable def arith (op : BinOp) : SVal → SVal → Option SVal
  | .int k₁ a, .int k₂ b =>
    let k := IKind.join k₁ k₂
    match op with
    | .add => some (.int k (a + b))
    | .sub => some (.int k (a - b))
    | .mul => some (.int k (a * b))
    | .div => if b = 0 then none else some (.int k (tquot a b))
    | .pow => if 0 ≤ b then some (.int k (a ^ b.toNat)) else (ipowNeg a b).map (.int k)
  | .real k₁ a, .real k₂ b =>
    let k := RKind.join k₁ k₂
    match op with
    | .add => some (.real k (a + b))
    | .sub => some (.real k (a - b))
    | .mul => some (.real k (a * b))
    | .div => if b = 0 then none else some (.real k (a / b))
    | .pow => (rpowF a b).map (.real k)
  | .int _ a, .real k b =>
    match op with
    | .add => some (.real k ((a : ℝ) + b))
    | .sub => some (.real k ((a : ℝ) - b))
    | .mul => some (.real k ((a : ℝ) * b))
    | .div => if b = 0 then none else some (.real k ((a : ℝ) / b))
    | .pow => (rpowF (a : ℝ) b).map (.real k)
  | .real k a, .int _ b =>
    match op with
    | .add => some (.real k (a + (b : ℝ)))
    | .sub => some (.real k (a - (b : ℝ)))
    | .mul => some (.real k (a * (b : ℝ)))
    | .div => if b = 0 then none else some (.real k (a / (b : ℝ)))
    | .pow => (zpowF a b).map (.real k)
  | _, _ => none

def cmpR (op : CmpOp) (a b : ℝ) : Bool :=
  match op with
  | .eq => decide (a = b) | .ne => decide (a ≠ b)
  | .lt => decide (a < b) | .le => decide (a ≤ b)
  | .gt => decide (b < a) | .ge => decide (b ≤ a)

/-- Fortran character comparison: the shorter operand is blank-padded. -/
def padTo (n : ℕ) (s : List Char) : List Char := s ++ List.replicate (n - s.length) ' '

def strEq (a b : List Char) : Bool :=
  let n := max a.length b.length
  decide (padTo n a = padTo n b)

noncomputable def compare (op : CmpOp) : SVal → SVal → Option SVal
  | .int _ a, .int _ b => some (.bool (cmpR op a b))
  | .real _ a, .real _ b => some (.bool (cmpR op a b))
  | .int _ a, .real _ b => some (.bool (cmpR op a b))
  | .real _ a, .int _ b => some (.bool (cmpR op a b))
  | .str a, .str b =>
    match op with
    | .eq => some (.bool (strEq a b))
    | .ne => some (.bool (!strEq a b))
    | _ => none      -- collating-sequence order: outside the model
  | _, _ => none

def logic (op : LogOp) : SVal → SVal → Option SVal
  | .bool a, .bool b =>
    match op with
    | .and => some (.bool (a && b))
    | .or => some (.bool (a || b))
    | .eqv => some (.bool (a == b))
    | .neqv => some (.bool (a != b))
  | _, _ => none

/-- Column-major enumeration of all coordinates of an array. -/
def allCoords (a : FArr) : List (List ℤ) :=
  (List.range a.size).map fun k =>
    List.zipWith (fun (lb : ℤ) (i : ℕ) => lb + i) a.lbs (CM.coords a.extents k)

/-- Lift a scalar binary operation elementwise (with scalar broadcast). -/
noncomputable def lift2 (f : SVal → SVal → Option SVal) : Val → Val → Option Val
  | .s a, .s b => (f a b).map .s
  | .arr a, .s b =>
    match f (a.get a.lbs) b with
    | none => none
    | some sample =>
      some (.arr { elemTy := sample.ty, dims := a.dims.map fun d => (1, d.2),
                   get := fun idx =>
                     let src := List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) a.lbs idx
                   (f (a.get src) b).getD sample })
  | .s a, .arr b =>
    match f a (b.get b.lbs) with
    | none => none
    | some sample =>
      some (.arr { elemTy := sample.ty, dims := b.dims.map fun d => (1, d.2),
                   get := fun idx =>
                     let src := List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) b.lbs idx
                   (f a (b.get src)).getD sample })
  | .arr a, .arr b =>
    if a.extents ≠ b.extents then none
    else if a.size = 0 then
      some (.arr { elemTy := a.elemTy, dims := a.dims.map fun d => (1, d.2), get := a.get })
    else
      match f (a.get a.lbs) (b.get b.lbs) with
      | none => none
      | some sample =>
        some (.arr { elemTy := sample.ty, dims := a.dims.map fun d => (1, d.2),
                     get := fun idx =>
                       let sa := List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) a.lbs idx
                     let sb := List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i - 1) b.lbs idx
                     (f (a.get sa) (b.get sb)).getD sample })

/-- Elementwise operations are only *defined* when every element is: this
predicate is part of the definedness obligation of array expressions. -/
def allDefined (f : SVal → SVal → Option SVal) (a : FArr) (b : SVal) : Prop :=
  ∀ idx, a.inBounds idx → (f (a.get idx) b).isSome

noncomputable def lift1 (f : SVal → Option SVal) : Val → Option Val
  | .s a => (f a).map .s
  | .arr a =>
    if a.size = 0 then some (.arr a)
    else match f (a.get a.lbs) with
    | none => none
    | some sample =>
      some (.arr { elemTy := sample.ty, dims := a.dims, get := fun idx => (f (a.get idx)).getD sample })

/-- One resolved subscript position. -/
inductive Axis where
  | fixed (i : ℤ)
  | range (lo hi : Option ℤ) (st : Option ℤ)
  /-- A vector subscript: the selected coordinates, in order (F2018 9.5.3.3.3). -/
  | vector (cs : List ℤ)

/-- Resolve a section against the array's own bounds: omitted edges take the
declared bounds (F2018 9.5.3.3.2), an omitted stride is 1. -/
def resolveAxis (d : ℤ × ℕ) : Axis → Option (List ℤ ⊕ ℤ)
  | .fixed i => if d.1 ≤ i ∧ i < d.1 + d.2 then some (Sum.inr i) else none
  | .range lo hi st =>
    let st' := st.getD 1
    if st' = 0 then none
    else
      let lo' := lo.getD d.1
      let hi' := hi.getD (d.1 + d.2 - 1)
      let cs := Fort.sectionCoords lo' hi' st'
      if cs.all (fun i => decide (d.1 ≤ i ∧ i < d.1 + d.2)) then some (Sum.inl cs) else none
  | .vector cs =>
    if cs.all (fun i => decide (d.1 ≤ i ∧ i < d.1 + d.2)) then some (Sum.inl cs) else none

/-- Apply resolved axes: the section's dims (each ranging from 1) and the map
from section coordinates back to the parent's coordinates. -/
def sectionOf (a : FArr) (axes : List Axis) : Option Val :=
  if axes.length ≠ a.dims.length then none
  else
    let resolved := List.zipWith resolveAxis a.dims axes
    if resolved.any Option.isNone then none
    else
      let rs := resolved.filterMap id
      let ranges := rs.filterMap fun r => match r with | Sum.inl cs => some cs | Sum.inr _ => none
      if ranges.isEmpty then
        -- every position is a scalar index: an element
        some (.s (a.get (rs.map fun r => match r with | Sum.inl _ => 0 | Sum.inr i => i)))
      else
        let back (idx : List ℤ) : List ℤ :=
          (rs.foldl (fun (acc : List ℤ × List ℤ) r =>
              match r with
              | Sum.inr i => (acc.1 ++ [i], acc.2)
              | Sum.inl cs =>
                match acc.2 with
                | [] => (acc.1 ++ [0], [])
                | j :: rest => (acc.1 ++ [cs.getD (j - 1).toNat 0], rest)) ([], idx)).1
        some (.arr { elemTy := a.elemTy,
                     dims := ranges.map fun cs => (1, cs.length),
                     get := fun idx => a.get (back idx) })

/-- Sequential fold over all elements in column-major order (the Fortran
`SUM`/`PRODUCT` element order is unspecified; any order gives the same exact
result, and column-major is the storage order). -/
noncomputable def foldElems (a : FArr) (f : SVal → SVal → Option SVal) (init : SVal) : Option SVal :=
  (allCoords a).foldl (fun acc c => acc.bind fun x => f x (a.get c)) (some init)

def zeroOf : FTy → Option SVal
  | .int k => some (.int k 0)
  | .real k => some (.real k 0)
  | _ => none

def oneOf : FTy → Option SVal
  | .int k => some (.int k 1)
  | .real k => some (.real k 1)
  | _ => none

noncomputable def maxS : SVal → SVal → Option SVal
  | .int k₁ a, .int k₂ b => some (.int (IKind.join k₁ k₂) (max a b))
  | .real k₁ a, .real k₂ b => some (.real (RKind.join k₁ k₂) (max a b))
  | _, _ => none

noncomputable def minS : SVal → SVal → Option SVal
  | .int k₁ a, .int k₂ b => some (.int (IKind.join k₁ k₂) (min a b))
  | .real k₁ a, .real k₂ b => some (.real (RKind.join k₁ k₂) (min a b))
  | _, _ => none

def toRealS : SVal → Option ℝ
  | .int _ a => some (a : ℝ)
  | .real _ a => some a
  | _ => none

/-- A real-valued function applied to one real argument, result in the
argument's real kind (an integer argument is not allowed by Fortran for
these intrinsics, except where noted). -/
noncomputable def realFn (f : ℝ → Option ℝ) : SVal → Option SVal
  | .real k a => (f a).map (.real k)
  | _ => none

/-! ### Character helpers (F2018 16.9.95 INDEX, 16.9.166 SCAN, 16.9.192 TRIM, 16.9.5 ADJUSTL) -/

/-- 1-based position of the first occurrence of `t` in `s`, `0` if none. -/
def strIndex (s t : List Char) : ℕ :=
  let rec go : List Char → ℕ → ℕ
    | [], p => if t = [] then p else 0
    | c :: rest, p => if t.isPrefixOf (c :: rest) then p else go rest (p + 1)
  go s 1

/-- 1-based position of the first character of `s` that is in `set`, `0` if none. -/
def strScan (s set : List Char) : ℕ :=
  let rec go : List Char → ℕ → ℕ
    | [], _ => 0
    | c :: rest, p => if set.contains c then p else go rest (p + 1)
  go s 1

def strTrim (s : List Char) : List Char := (s.reverse.dropWhile (· = ' ')).reverse

def strAdjustl (s : List Char) : List Char :=
  let t := s.dropWhile (· = ' ')
  t ++ List.replicate (s.length - t.length) ' '

/-! ### Array transforms (F2018 16.9.163 RESHAPE, 16.9.145 PACK, 16.9.56 CSHIFT,
16.9.71 EOSHIFT, 16.9.126 MAXLOC, 16.9.131 MINLOC, 16.9.196 TRANSPOSE, 16.9.124 MATMUL) -/

/-- Elements in array element order (column-major). -/
def elems (a : FArr) : List SVal := (allCoords a).map a.get

/-- The extents named by a rank-1 INTEGER shape array, if all are non-negative. -/
def shapeOf (shp : FArr) : Option (List ℕ) :=
  if shp.rank ≠ 1 then none
  else (elems shp).mapM fun v => match v with
    | .int _ n => if 0 ≤ n then some n.toNat else none
    | _ => none

/-- An ORDER argument as a permutation of `1..r` (1-based axis numbers). -/
def orderOf (r : ℕ) : Option FArr → Option (List ℕ)
  | none => some ((List.range r).map (· + 1))
  | some ord =>
    (shapeOf ord).bind fun l =>
      if l.length = r ∧ (List.range r).all (fun k => l.contains (k + 1)) then some l else none

/-- RESHAPE: the result's elements in the ORDER-permuted subscript order are
SOURCE's elements followed by cyclic copies of PAD. -/
noncomputable def reshapeF (src shp : FArr) (pad ord : Option FArr) : Option Val :=
  (shapeOf shp).bind fun ns =>
  (orderOf ns.length ord).bind fun perm =>
  let total := ns.foldr (· * ·) 1
  let srcSeq := elems src
  let seq : Option (List SVal) :=
    if total ≤ srcSeq.length then some (srcSeq.take total)
    else match pad with
      | none => none
      | some p =>
        let ps := elems p
        if ps.length = 0 then none
        else some (srcSeq ++ (List.range (total - srcSeq.length)).map fun k => ps.getD (k % ps.length) (.int .i4 0))
  seq.map fun sq =>
    let getR (idx : List ℤ) : SVal :=
      let idx0 := idx.map fun i => (i - 1).toNat
      let permIdx := perm.map fun a => idx0.getD (a - 1) 0
      let permExt := perm.map fun a => ns.getD (a - 1) 0
      sq.getD (CM.offset permExt permIdx) (.int .i4 0)
    .arr { elemTy := src.elemTy, dims := ns.map fun n => (1, n), get := getR }

/-- The coordinates of an array in array element order where a conforming LOGICAL mask is true. -/
def maskedCoords (a m : FArr) : Option (List (List ℤ)) :=
  if a.extents ≠ m.extents then none
  else some ((List.range a.size).filterMap fun k =>
    let ca := (allCoords a).getD k []
    let cm := (allCoords m).getD k []
    match m.get cm with
    | .bool true => some ca
    | _ => none)

/-- A rank-1 unit-origin array over a list of elements. -/
def listArr (ty : FTy) (l : List SVal) : FArr :=
  { elemTy := ty, dims := [(1, l.length)], get := fun idx => l.getD ((idx.headD 1) - 1).toNat (.int .i4 0) }

/-- PACK(ARRAY, MASK [, VECTOR]). -/
def packF (a m : FArr) (vec : Option FArr) : Option Val :=
  (maskedCoords a m).bind fun cs =>
  let picked := cs.map a.get
  match vec with
  | none => some (.arr (listArr a.elemTy picked))
  | some v =>
    if v.rank ≠ 1 ∨ v.size < picked.length then none
    else some (.arr (listArr a.elemTy (picked ++ (elems v).drop picked.length)))

/-- CSHIFT on a rank-1 array: `result(i) = a(lb + (i - 1 + s) mod n)`. -/
def cshift1 (a : FArr) (sh : ℤ) : Option Val :=
  match a.dims with
  | [(lb, n)] =>
    if n = 0 then some (.arr { a with dims := [(1, 0)] })
    else
      let getR (idx : List ℤ) : SVal := a.get [lb + ((idx.headD 1) - 1 + sh) % (n : ℤ)]
      some (.arr { elemTy := a.elemTy, dims := [(1, n)], get := getR })
  | _ => none

/-- Default boundary of EOSHIFT: the zero of the element type. -/
def eoshiftDefault : FTy → SVal
  | .int k => .int k 0
  | .real k => .real k 0
  | .bool => .bool false
  | .str => .str []

/-- EOSHIFT on a rank-1 array with an explicit or default (zero) boundary. -/
def eoshift1 (a : FArr) (sh : ℤ) (bnd : Option SVal) : Option Val :=
  match a.dims with
  | [(lb, n)] =>
    let b := bnd.getD (eoshiftDefault a.elemTy)
    let getR (idx : List ℤ) : SVal :=
      let j := (idx.headD 1) - 1 + sh
      if 0 ≤ j ∧ j < n then a.get [lb + j] else b
    some (.arr { elemTy := a.elemTy, dims := [(1, n)], get := getR })
  | _ => none

/-- Whether position `k` (0-based) of a rank-1 array passes an optional conforming LOGICAL mask. -/
def maskOk (n : ℕ) (m : Option FArr) (k : ℕ) : Bool :=
  match m with
  | none => true
  | some mk =>
    match mk.dims with
    | [(lm, nm)] => nm = n && (match mk.get [lm + k] with | .bool b => b | _ => false)
    | _ => false

/-- First position (1-based, array element order) of the extremum of a rank-1
array under an optional mask; `0` when nothing is selected. `better x y` is
"`x` strictly beats `y`". -/
noncomputable def locate1 (a : FArr) (m : Option FArr) (better : SVal → SVal → Bool) : Option Val :=
  match a.dims with
  | [(lb, n)] =>
    let cands := (List.range n).filter (maskOk n m)
    let best := cands.foldl (fun (acc : Option ℕ) k => match acc with
      | none => some k
      | some j => if better (a.get [lb + k]) (a.get [lb + j]) then some k else some j) none
    let pos : ℤ := match best with | none => 0 | some k => (k : ℤ) + 1
    some (.arr { elemTy := .int .i4, dims := [(1, 1)], get := fun _ => .int .i4 pos })
  | _ => none

noncomputable def gtS (x y : SVal) : Bool := match x, y with
  | .int _ p, .int _ q => decide (q < p)
  | .real _ p, .real _ q => decide (q < p)
  | _, _ => false

noncomputable def ltS (x y : SVal) : Bool := match x, y with
  | .int _ p, .int _ q => decide (p < q)
  | .real _ p, .real _ q => decide (p < q)
  | _, _ => false

def transposeF (a : FArr) : Option Val :=
  match a.dims with
  | [(l1, n1), (l2, n2)] =>
    let getR (idx : List ℤ) : SVal := a.get [l1 + (idx.getD 1 1) - 1, l2 + (idx.getD 0 1) - 1]
    some (.arr { elemTy := a.elemTy, dims := [(1, n2), (1, n1)], get := getR })
  | _ => none

/-- Σₖ a(i,k) b(k,j) as a sequential fold with the scalar arithmetic. -/
noncomputable def dotS (xs ys : List SVal) (z : SVal) : Option SVal :=
  (List.zip xs ys).foldl (fun acc p => acc.bind fun s => (arith .mul p.1 p.2).bind fun m => arith .add s m) (some z)

noncomputable def matmulF (a b : FArr) : Option Val :=
  match a.dims, b.dims with
  | [(la1, n), (la2, k1)], [(lb1, k2), (lb2, m)] =>
    if k1 ≠ k2 ∨ k1 = 0 then none else
    (zeroOf a.elemTy).bind fun z =>
    let cell (i j : ℕ) : Option SVal :=
      dotS ((List.range k1).map fun k => a.get [la1 + i, la2 + k]) ((List.range k1).map fun k => b.get [lb1 + k, lb2 + j]) z
    match cell 0 0 with
    | none => none
    | some sample =>
      let getR (idx : List ℤ) : SVal := (cell ((idx.getD 0 1) - 1).toNat ((idx.getD 1 1) - 1).toNat).getD sample
      some (.arr { elemTy := sample.ty, dims := [(1, n), (1, m)], get := getR })
  | [(la1, n), (la2, k1)], [(lb1, k2)] =>
    if k1 ≠ k2 ∨ k1 = 0 then none else
    (zeroOf a.elemTy).bind fun z =>
    let cell (i : ℕ) : Option SVal :=
      dotS ((List.range k1).map fun k => a.get [la1 + i, la2 + k]) ((List.range k1).map fun k => b.get [lb1 + k]) z
    match cell 0 with
    | none => none
    | some sample => some (.arr { elemTy := sample.ty, dims := [(1, n)],
                                  get := fun idx => (cell ((idx.headD 1) - 1).toNat).getD sample })
  | _, _ => none

/-- MVBITS as a function of the kind's width: the new TO value. -/
def mvbitsK (bits : ℕ) (from_ frompos len to_ topos : ℤ) : ℤ :=
  let u := unsignedK bits from_
  let t := unsignedK bits to_
  let mask : ℕ := 2 ^ len.toNat - 1
  let field : ℕ := (u / 2 ^ frompos.toNat) &&& mask
  let cleared : ℕ := t &&& ((2 ^ bits - 1) ^^^ (mask * 2 ^ topos.toNat))
  wrapSigned bits ((cleared ||| (field * 2 ^ topos.toNat) : ℕ) : ℤ)

/-- The intrinsic table of Semantics v1 (unchanged; `intrinsic` dispatches to it). -/
noncomputable def intrinsicCore (o : Oracle) (f : Intr) (args : List Val) (kArg : Option ℤ) : Option Val :=
  match f, args with
  -- elemental real functions
  | .sqrt, [x] => lift1 (realFn fun a => if 0 ≤ a then some (Real.sqrt a) else none) x
  | .exp, [x] => lift1 (realFn fun a => some (Real.exp a)) x
  | .log, [x] => lift1 (realFn fun a => if 0 < a then some (Real.log a) else none) x
  | .log10, [x] => lift1 (realFn fun a => if 0 < a then some (Real.logb 10 a) else none) x
  | .sin, [x] => lift1 (realFn fun a => some (Real.sin a)) x
  | .cos, [x] => lift1 (realFn fun a => some (Real.cos a)) x
  | .tan, [x] => lift1 (realFn fun a => some (Real.tan a)) x
  | .asin, [x] => lift1 (realFn fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arcsin a) else none) x
  | .acos, [x] => lift1 (realFn fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arccos a) else none) x
  | .atan, [x] => lift1 (realFn fun a => some (Real.arctan a)) x
  | .sinh, [x] => lift1 (realFn fun a => some (Real.sinh a)) x
  | .cosh, [x] => lift1 (realFn fun a => some (Real.cosh a)) x
  | .tanh, [x] => lift1 (realFn fun a => some (Real.tanh a)) x
  | .atan2, [y, x] => lift2 (fun a b => match a, b with
      | .real k₁ p, .real k₂ q => (o.prim "atan2" [p, q]).map (.real (RKind.join k₁ k₂))
      | _, _ => none) y x
  | .prim name, [x] => lift1 (realFn fun a => o.prim name [a]) x
  | .abs, [x] => lift1 (fun v => match v with
      | .int k a => some (.int k |a|)
      | .real k a => some (.real k |a|)
      | _ => none) x
  -- conversions
  | .int, [x] =>
    let k := if kArg = some 8 then IKind.i8 else IKind.i4
    lift1 (fun v => match v with
      | .int _ a => some (.int k a)
      | .real _ a => some (.int k (truncR a))
      | _ => none) x
  | .nint, [x] =>
    let k := if kArg = some 8 then IKind.i8 else IKind.i4
    lift1 (fun v => match v with
      | .real _ a => some (.int k (nintR a))
      | _ => none) x
  | .aint, [x] => lift1 (fun v => match v with
      | .real k a => some (.real k (truncR a : ℝ))
      | _ => none) x
  | .anint, [x] => lift1 (fun v => match v with
      | .real k a => some (.real k (nintR a : ℝ))
      | _ => none) x
  | .floor, [x] =>
    let k := if kArg = some 8 then IKind.i8 else IKind.i4
    lift1 (fun v => match v with
      | .real _ a => some (.int k ⌊a⌋)
      | _ => none) x
  | .ceiling, [x] =>
    let k := if kArg = some 8 then IKind.i8 else IKind.i4
    lift1 (fun v => match v with
      | .real _ a => some (.int k ⌈a⌉)
      | _ => none) x
  | .real, [x] =>
    let k := if kArg = some 8 then RKind.r8 else RKind.r4
    lift1 (fun v => match v with
      | .int _ a => some (.real k (a : ℝ))
      | .real _ a => some (.real k a)
      | _ => none) x
  | .dble, [x] => lift1 (fun v => match v with
      | .int _ a => some (.real .r8 (a : ℝ))
      | .real _ a => some (.real .r8 a)
      | _ => none) x
  -- two-argument numeric
  | .mod, [a, p] => lift2 (fun u v => match u, v with
      | .int k₁ x, .int k₂ y => if y = 0 then none else some (.int (IKind.join k₁ k₂) (tmod x y))
      | .real k₁ x, .real k₂ y => if y = 0 then none else some (.real (RKind.join k₁ k₂) (modR x y))
      | _, _ => none) a p
  | .modulo, [a, p] => lift2 (fun u v => match u, v with
      | .int k₁ x, .int k₂ y => if y = 0 then none else some (.int (IKind.join k₁ k₂) (fmod x y))
      | .real k₁ x, .real k₂ y => if y = 0 then none else some (.real (RKind.join k₁ k₂) (moduloR x y))
      | _, _ => none) a p
  | .sign, [a, b] => lift2 (fun u v => match u, v with
      | .int k x, .int _ y => some (.int k (if 0 ≤ y then |x| else -|x|))
      | .real k x, .real _ y => some (.real k (if 0 ≤ y then |x| else -|x|))
      | _, _ => none) a b
  | .dim, [a, b] => lift2 (fun u v => match u, v with
      | .int k x, .int _ y => some (.int k (max (x - y) 0))
      | .real k x, .real _ y => some (.real k (max (x - y) 0))
      | _, _ => none) a b
  | .max, x :: rest => rest.foldl (fun acc y => acc.bind fun a => lift2 maxS a y) (some x)
  | .min, x :: rest => rest.foldl (fun acc y => acc.bind fun a => lift2 minS a y) (some x)
  | .merge, [t, f', m] =>
    match t, f', m with
    | .s tv, .s fv, .s (.bool b) => some (.s (if b then tv else fv))
    | _, _, _ => none
  -- reductions (whole array)
  | .sum, [.arr a] => (zeroOf a.elemTy).bind fun z => (foldElems a (arith .add) z).map .s
  | .product, [.arr a] => (oneOf a.elemTy).bind fun z => (foldElems a (arith .mul) z).map .s
  | .maxval, [.arr a] => if a.size = 0 then none else (foldElems a maxS (a.get a.lbs)).map .s
  | .minval, [.arr a] => if a.size = 0 then none else (foldElems a minS (a.get a.lbs)).map .s
  | .dot_product, [.arr a, .arr b] =>
    if a.rank ≠ 1 ∨ a.extents ≠ b.extents then none
    else
      match lift2 (arith .mul) (.arr a) (.arr b) with
      | some (.arr p) => (zeroOf p.elemTy).bind fun z => (foldElems p (arith .add) z).map .s
      | _ => none
  -- inquiries
  | .size, [.arr a] => some (.s (.int .i4 a.size))
  | .size, [.arr a, .s (.int _ d)] =>
    if 1 ≤ d ∧ d ≤ a.rank then some (.s (.int .i4 (a.extents.getD (d - 1).toNat 0))) else none
  | .lbound, [.arr a, .s (.int _ d)] =>
    if 1 ≤ d ∧ d ≤ a.rank then some (.s (.int .i4 (a.lbs.getD (d - 1).toNat 0))) else none
  | .ubound, [.arr a, .s (.int _ d)] =>
    if 1 ≤ d ∧ d ≤ a.rank then some (.s (.int .i4 (a.ubs.getD (d - 1).toNat 0))) else none
  | .lbound, [.arr a] =>
    some (.arr { elemTy := .int .i4, dims := [(1, a.rank)],
                 get := fun idx => .int .i4 (a.lbs.getD ((idx.headD 1) - 1).toNat 0) })
  | .ubound, [.arr a] =>
    some (.arr { elemTy := .int .i4, dims := [(1, a.rank)],
                 get := fun idx => .int .i4 (a.ubs.getD ((idx.headD 1) - 1).toNat 0) })
  | .kind, [.s v] => match v with
      | .int k _ => some (.s (.int .i4 k.kindValue))
      | .real k _ => some (.s (.int .i4 k.kindValue))
      | .bool _ => some (.s (.int .i4 4))
      | .str _ => some (.s (.int .i4 1))
  | .huge, [.s v] => match v with
      | .int k _ => some (.s (.int k k.huge))
      | .real k _ => some (.s (.real k k.hugeR))
      | _ => none
  | .tiny, [.s (.real k _)] => some (.s (.real k k.tinyR))
  | .epsilon, [.s (.real k _)] => some (.s (.real k k.epsR))
  | .precision, [.s (.real k _)] => some (.s (.int .i4 k.precision))
  | .digits, [.s v] => match v with
      | .int k _ => some (.s (.int .i4 k.digits))
      | .real k _ => some (.s (.int .i4 k.digits))
      | _ => none
  -- bit intrinsics on the kind's width
  | .iand, [a, b] => lift2 (fun u v => match u, v with
      | .int k₁ x, .int k₂ y => some (.int (IKind.join k₁ k₂) (bitAndK (IKind.join k₁ k₂).bits x y))
      | _, _ => none) a b
  | .ior, [a, b] => lift2 (fun u v => match u, v with
      | .int k₁ x, .int k₂ y => some (.int (IKind.join k₁ k₂) (bitOrK (IKind.join k₁ k₂).bits x y))
      | _, _ => none) a b
  | .ieor, [a, b] => lift2 (fun u v => match u, v with
      | .int k₁ x, .int k₂ y => some (.int (IKind.join k₁ k₂) (bitXorK (IKind.join k₁ k₂).bits x y))
      | _, _ => none) a b
  | .ishft, [a, b] => lift2 (fun u v => match u, v with
      | .int k x, .int _ s => some (.int k (ishftK k.bits x s))
      | _, _ => none) a b
  -- character
  | .len, [.s (.str s)] => some (.s (.int .i4 s.length))
  | .len_trim, [.s (.str s)] =>
    some (.s (.int .i4 (s.reverse.dropWhile (· = ' ')).length))
  | _, _ => none

/-- The intrinsics added in Semantics v1.1 (characters, array transforms, MVBITS). -/
noncomputable def intrinsicNew (o : Oracle) (f : Intr) (args : List Val) (kArg : Option ℤ) : Option Val :=
  match f, args with
  | .concat, [.s (.str a), .s (.str b)] => some (.s (.str (a ++ b)))
  -- substring `s(i:j)` (F2018 9.4.1): empty when `j < i`, else `1 ≤ i` and `j ≤ len`
  | .substr, [.s (.str s), .s (.int _ i), .s (.int _ j)] =>
    if j < i then some (.s (.str []))
    else if 1 ≤ i ∧ j ≤ s.length then some (.s (.str ((s.drop (i - 1).toNat).take (j - i + 1).toNat)))
    else none
  | .trim, [.s (.str s)] => some (.s (.str (strTrim s)))
  | .adjustl, [.s (.str s)] => some (.s (.str (strAdjustl s)))
  | .index_, [.s (.str s), .s (.str t)] => some (.s (.int .i4 (strIndex s t)))
  | .scan, [.s (.str s), .s (.str set)] => some (.s (.int .i4 (strScan s set)))
  | .char, [.s (.int _ i)] => if 0 ≤ i ∧ i < 256 then some (.s (.str [Char.ofNat i.toNat])) else none
  | .ichar, [.s (.str [c])] => some (.s (.int .i4 c.toNat))
  -- array transforms
  | .reshape false false, [.arr src, .arr shp] => reshapeF src shp none none
  | .reshape true false, [.arr src, .arr shp, .arr pad] => reshapeF src shp (some pad) none
  | .reshape false true, [.arr src, .arr shp, .arr ord] => reshapeF src shp none (some ord)
  | .reshape true true, [.arr src, .arr shp, .arr pad, .arr ord] => reshapeF src shp (some pad) (some ord)
  | .pack false, [.arr a, .arr m] => packF a m none
  | .pack true, [.arr a, .arr m, .arr v] => packF a m (some v)
  | .cshift false, [.arr a, .s (.int _ sh)] => cshift1 a sh
  | .cshift true, [.arr a, .s (.int _ sh), .s (.int _ d)] => if d = 1 then cshift1 a sh else none
  | .eoshift false false, [.arr a, .s (.int _ sh)] => eoshift1 a sh none
  | .eoshift true false, [.arr a, .s (.int _ sh), .s b] => eoshift1 a sh (some b)
  | .eoshift false true, [.arr a, .s (.int _ sh), .s (.int _ d)] => if d = 1 then eoshift1 a sh none else none
  | .eoshift true true, [.arr a, .s (.int _ sh), .s b, .s (.int _ d)] => if d = 1 then eoshift1 a sh (some b) else none
  | .maxloc false, [.arr a] => locate1 a none gtS
  | .maxloc true, [.arr a, .arr m] => locate1 a (some m) gtS
  | .minloc false, [.arr a] => locate1 a none ltS
  | .minloc true, [.arr a, .arr m] => locate1 a (some m) ltS
  | .transpose, [.arr a] => transposeF a
  | .matmul, [.arr a, .arr b] => matmulF a b
  | .shape, [.arr a] => some (.arr (listArr (.int .i4) (a.extents.map fun n => .int .i4 n)))
  | .mvbits, [.s (.int k f'), .s (.int _ fp), .s (.int _ ln), .s (.int _ t), .s (.int _ tp)] =>
    if 0 ≤ fp ∧ 0 ≤ ln ∧ 0 ≤ tp ∧ fp + ln ≤ k.bits ∧ tp + ln ≤ k.bits
    then some (.s (.int k (mvbitsK k.bits f' fp ln t tp))) else none
  | _, _ => none

/-- Whether an intrinsic belongs to the v1.1 extension table. -/
def Intr.isNew : Intr → Bool
  | .concat | .substr | .trim | .adjustl | .index_ | .scan | .char | .ichar => true
  | .reshape _ _ | .pack _ | .cshift _ | .eoshift _ _ | .maxloc _ | .minloc _ => true
  | .transpose | .matmul | .shape | .mvbits => true
  | _ => false

/-- Intrinsic dispatch: the extension table for the new constructors, the v1
table otherwise (so that every v1 equation reduces exactly as before). -/
noncomputable def intrinsic (o : Oracle) (f : Intr) (args : List Val) (kArg : Option ℤ) : Option Val :=
  match f with
  | .concat | .substr | .trim | .adjustl | .index_ | .scan | .char | .ichar => intrinsicNew o f args kArg
  | .reshape _ _ | .pack _ | .cshift _ | .eoshift _ _ | .maxloc _ | .minloc _ => intrinsicNew o f args kArg
  | .transpose | .matmul | .shape | .mvbits => intrinsicNew o f args kArg
  | _ => intrinsicCore o f args kArg

mutual
  noncomputable def eval (o : Oracle) (σ : State) : Expr → Option Val
    | .intLit k v => some (.s (.int k v))
    | .realLit k d => some (.s (.real k d.toReal))
    | .boolLit b => some (.s (.bool b))
    | .strLit s => some (.s (.str s))
    | .var x => σ x
    | .neg e => (eval o σ e).bind (lift1 fun v => match v with
        | .int k a => some (.int k (-a))
        | .real k a => some (.real k (-a))
        | _ => none)
    | .paren e => eval o σ e
    | .bin op a b => do
        let va ← eval o σ a
        let vb ← eval o σ b
        lift2 (arith op) va vb
    | .cmp op a b => do
        let va ← eval o σ a
        let vb ← eval o σ b
        lift2 (compare op) va vb
    | .logic op a b => do
        let va ← eval o σ a
        let vb ← eval o σ b
        lift2 (logic op) va vb
    | .not e => (eval o σ e).bind (lift1 fun v => match v with
        | .bool b => some (.bool (!b))
        | _ => none)
    | .ref a subs => do
        let arr ← σ a
        match arr with
        | .arr av =>
          let axes ← evalSubs o σ subs
          sectionOf av axes
        | .s _ => none
    | .call f args k => do
        let vs ← evalExprs o σ args
        intrinsic o f vs (k.map fun kk => match kk with | .i4 => (4 : ℤ) | .i8 => 8)
    | .arrayCtor es => do
        let vs ← evalExprs o σ es
        let svs ← vs.mapM fun v => match v with | .s sv => some sv | .arr _ => none
        match svs with
        | [] => none
        | sv :: _ =>
          if svs.all (fun x => x.ty = sv.ty) then pure (.arr (listArr sv.ty svs)) else none

  noncomputable def evalExprs (o : Oracle) (σ : State) : Exprs → Option (List Val)
    | .nil => some []
    | .cons e rest => do
        let v ← eval o σ e
        let vs ← evalExprs o σ rest
        pure (v :: vs)

  noncomputable def evalBound (o : Oracle) (σ : State) : Bound → Option (Option ℤ)
    | .omitted => some none
    | .given e => do
        let v ← eval o σ e
        match v with
        | .s (.int _ i) => pure (some i)
        | _ => none

  noncomputable def evalSub (o : Oracle) (σ : State) : Sub → Option Axis
    | .scalar e => do
        let v ← eval o σ e
        match v with
        | .s (.int _ i) => pure (.fixed i)
        | _ => none
    | .triplet lo hi st => do
        let l ← evalBound o σ lo
        let h ← evalBound o σ hi
        let s ← evalBound o σ st
        pure (.range l h s)
    | .vector e => do
        let v ← eval o σ e
        match v with
        | .arr av =>
          match av.dims with
          | [(lb, n)] =>
            let cs ← (List.range n).mapM fun k => match av.get [lb + k] with
              | .int _ i => some i
              | _ => none
            pure (.vector cs)
          | _ => none
        | .s _ => none

  noncomputable def evalSubs (o : Oracle) (σ : State) : Subs → Option (List Axis)
    | .nil => some []
    | .cons s rest => do
        let a ← evalSub o σ s
        let as ← evalSubs o σ rest
        pure (a :: as)
end

/-! ## Assignment with conversion to the declared type -/

/-- Convert a scalar to the declared type of the target (F2018 10.2.1.3):
integer targets truncate reals, real targets embed integers, character
targets are blank-padded or truncated to the declared length. -/
noncomputable def convertTo (d : Decl) : SVal → Option SVal
  | .int _ a => match d.ty with
    | .int k => some (.int k a)
    | .real k => some (.real k (a : ℝ))
    | _ => none
  | .real _ a => match d.ty with
    | .int k => some (.int k (truncR a))
    | .real k => some (.real k a)
    | _ => none
  | .bool b => match d.ty with
    | .bool => some (.bool b)
    | _ => none
  | .str s => match d.ty with
    | .str => some (.str (padTo d.clen (s.take d.clen)))
    | _ => none

/-- Store a value into the whole array or the selected part of it. -/
noncomputable def storeArr (d : Decl) (a : FArr) (axes : List Axis) (rhs : Val) : Option FArr :=
  if axes.length ≠ a.dims.length then none
  else
    let resolved := List.zipWith resolveAxis a.dims axes
    if resolved.any Option.isNone then none
    else
      let rs := resolved.filterMap id
      let ranges := rs.filterMap fun r => match r with | Sum.inl cs => some cs | Sum.inr _ => none
      -- the set of parent coordinates written, in section order
      let targets : List (List ℤ) :=
        rs.foldl (fun (acc : List (List ℤ)) r =>
          match r with
          | Sum.inr i => acc.map fun c => c ++ [i]
          | Sum.inl cs => (cs.map fun i => acc.map fun c => c ++ [i]).flatten.reverse.reverse) [[]]
      match rhs with
      | .s v =>
        (convertTo d v).map fun cv =>
          { a with get := fun idx => if targets.contains idx then cv else a.get idx }
      | .arr b =>
        if ranges.isEmpty then none
        else if (ranges.map List.length) ≠ b.extents then none
        else
          -- pair each target coordinate with the corresponding rhs coordinate
          let pairs := List.zip targets (allCoords b)
          let converted := pairs.map fun (t, c) => (t, convertTo d (b.get c))
          if converted.any (fun p => p.2.isNone) then none
          else some { a with get := fun idx =>
            match converted.find? (fun p => p.1 = idx) with
            | some (_, some v) => v
            | _ => a.get idx }

/-- Assign `target(subs) = rhs`. -/
noncomputable def assign (o : Oracle) (Γ : Env) (σ : State) (target : String) (subs : Subs) (rhs : Val)
    : Option State := do
  let d ← Γ target
  let cur ← σ target
  match cur, subs with
  | .s _, .nil =>
    match rhs with
    | .s v => (convertTo d v).map fun cv => Function.update σ target (some (.s cv))
    | .arr _ => none
  | .s _, _ => none
  | .arr a, .nil =>
    -- whole-array assignment: every position is a full range
    let axes := a.dims.map fun _ => Axis.range none none none
    (storeArr d a axes rhs).map fun a' => Function.update σ target (some (.arr a'))
  | .arr a, _ => do
    let axes ← evalSubs o σ subs
    (storeArr d a axes rhs).map fun a' => Function.update σ target (some (.arr a'))

/-! ## Renaming (ASSOCIATE with a variable selector) -/

mutual
  /-- Replace the variable `n` by `x` in an expression. -/
  def renameE (n x : String) : Expr → Expr
    | .intLit k v => .intLit k v
    | .realLit k d => .realLit k d
    | .boolLit b => .boolLit b
    | .strLit s => .strLit s
    | .var y => .var (if y = n then x else y)
    | .neg e => .neg (renameE n x e)
    | .paren e => .paren (renameE n x e)
    | .bin op a b => .bin op (renameE n x a) (renameE n x b)
    | .cmp op a b => .cmp op (renameE n x a) (renameE n x b)
    | .logic op a b => .logic op (renameE n x a) (renameE n x b)
    | .not e => .not (renameE n x e)
    | .ref a subs => .ref (if a = n then x else a) (renameSubs n x subs)
    | .call f args k => .call f (renameEs n x args) k
    | .arrayCtor es => .arrayCtor (renameEs n x es)
  def renameEs (n x : String) : Exprs → Exprs
    | .nil => .nil
    | .cons e rest => .cons (renameE n x e) (renameEs n x rest)
  def renameB (n x : String) : Bound → Bound
    | .omitted => .omitted
    | .given e => .given (renameE n x e)
  def renameSub (n x : String) : Sub → Sub
    | .scalar e => .scalar (renameE n x e)
    | .triplet lo hi st => .triplet (renameB n x lo) (renameB n x hi) (renameB n x st)
    | .vector e => .vector (renameE n x e)
  def renameSubs (n x : String) : Subs → Subs
    | .nil => .nil
    | .cons s rest => .cons (renameSub n x s) (renameSubs n x rest)
end

def renameItem (n x : String) : CaseItem → CaseItem
  | .value e => .value (renameE n x e)
  | .range lo hi => .range (lo.map (renameE n x)) (hi.map (renameE n x))

/-- Replace the variable `n` by `x` in a statement (loop and FORALL indices included). -/
def renameS (n x : String) : Stmt → Stmt
  | .assign t subs e => .assign (if t = n then x else t) (renameSubs n x subs) (renameE n x e)
  | .seq a b => .seq (renameS n x a) (renameS n x b)
  | .skip => .skip
  | .ite c t e => .ite (renameE n x c) (renameS n x t) (renameS n x e)
  | .doLoop v lo hi st body =>
      .doLoop (if v = n then x else v) (renameE n x lo) (renameE n x hi) (st.map (renameE n x)) (renameS n x body)
  | .doWhile c body => .doWhile (renameE n x c) (renameS n x body)
  | .exit => .exit
  | .cycle => .cycle
  | .forall1 v lo hi st mask t subs rhs =>
      .forall1 (if v = n then x else v) (renameE n x lo) (renameE n x hi) (st.map (renameE n x))
        (mask.map (renameE n x)) (if t = n then x else t) (renameSubs n x subs) (renameE n x rhs)
  | .whereBlock m asgs els =>
      .whereBlock (renameE n x m) (asgs.map fun p => ((if p.1 = n then x else p.1), renameE n x p.2))
        (els.map fun p => ((if p.1 = n then x else p.1), renameE n x p.2))
  | .caseArm sel items body rest =>
      .caseArm (renameE n x sel) (items.map (renameItem n x)) (renameS n x body) (renameS n x rest)
  | .associate m y body => .associate m (if y = n then x else y) (if m = n then body else renameS n x body)
  | .allocate y specs => .allocate (if y = n then x else y) (specs.map fun p => (p.1.map (renameE n x), renameE n x p.2))
  | .call p acts => .call p (acts.map fun a => match a with
      | .var y => .var (if y = n then x else y)
      | .expr e => .expr (renameE n x e))

/-! ## SELECT CASE: the arm condition as an expression (F2018 11.1.9.2) -/

def caseItemCond (sel : Expr) : CaseItem → Expr
  | .value v => .cmp .eq sel v
  | .range (some lo) (some hi) => .logic .and (.cmp .le lo sel) (.cmp .le sel hi)
  | .range (some lo) none => .cmp .le lo sel
  | .range none (some hi) => .cmp .le sel hi
  | .range none none => .boolLit true

/-- `sel` matches one of the items (an empty item list never matches). -/
def caseCond (sel : Expr) : List CaseItem → Expr
  | [] => .boolLit false
  | [it] => caseItemCond sel it
  | it :: rest => .logic .or (caseItemCond sel it) (caseCond sel rest)

/-! ## FORALL and WHERE as state functions -/

/-- Read an INTEGER scalar. -/
noncomputable def evalInt (o : Oracle) (σ : State) (e : Expr) : Option ℤ :=
  (eval o σ e).bind fun v => match v with
    | .s (.int _ i) => some i
    | _ => none

/-- `FORALL (v = lo:hi:st [, mask]) target(subs) = rhs`: all index values, the
active ones by the mask, every right-hand side and subscript evaluated with
`v` bound *before* any store; then the stores in index order (the standard
requires the assigned elements to be distinct, so the order is immaterial).
The index variable is local to the construct: `v`'s outer binding is restored. -/
noncomputable def execForall (o : Oracle) (Γ : Env) (σ : State) (v : String) (lo hi : Expr) (st : Option Expr)
    (mask : Option Expr) (target : String) (subs : Subs) (rhs : Expr) : Option State := do
  let l ← evalInt o σ lo
  let h ← evalInt o σ hi
  let s ← match st with
    | none => some (1 : ℤ)
    | some e => evalInt o σ e
  if s = 0 then none else
  let vals := Fort.sectionCoords l h s
  let bind (i : ℤ) : State := Function.update σ v (some (.s (.int .i4 i)))
  let active ← vals.filterMapM fun i => match mask with
    | none => some (some i)
    | some m => (eval o (bind i) m).bind fun mv => match mv with
      | .s (.bool true) => some (some i)
      | .s (.bool false) => some none
      | _ => none
  let rhsVals ← active.mapM fun i => eval o (bind i) rhs
  let axesList ← active.mapM fun i => evalSubs o (bind i) subs
  let d ← Γ target
  let σ' ← (List.zip axesList rhsVals).foldlM (fun (acc : State) p => do
      let cur ← acc target
      match cur with
      | .arr a => (storeArr d a p.1 p.2).map fun a' => Function.update acc target (some (.arr a'))
      | .s _ => none) σ
  pure (Function.update σ' v (σ v))

/-- One masked whole-array assignment `a = e` under the LOGICAL mask array `m`
(`true` selects). A scalar `e` broadcasts; an array `e` must conform. -/
noncomputable def maskedAssign (o : Oracle) (Γ : Env) (σ : State) (m : FArr) (sense : Bool)
    (a : String) (e : Expr) : Option State := do
  let d ← Γ a
  let cur ← σ a
  let av ← match cur with | .arr av => some av | .s _ => none
  if av.extents ≠ m.extents then none else
  let rhs ← eval o σ e
  let pick (idx : List ℤ) : Bool :=
    match m.get (List.zipWith (fun (lm : ℤ) (i : ℤ) => lm + i) m.lbs (rebaseZ av.lbs idx)) with
    | .bool b => b = sense
    | _ => false
  let newVal (idx : List ℤ) : Option SVal :=
    match rhs with
    | .s sv => convertTo d sv
    | .arr bv => if bv.extents ≠ av.extents then none
        else convertTo d (bv.get (List.zipWith (fun (lb : ℤ) (i : ℤ) => lb + i) bv.lbs (rebaseZ av.lbs idx)))
  if (allCoords av).all (fun idx => !pick idx || (newVal idx).isSome) then
    pure (Function.update σ a (some (.arr { av with
      get := fun idx => if pick idx then (newVal idx).getD (av.get idx) else av.get idx })))
  else none
where
  rebaseZ (lbs : List ℤ) (idx : List ℤ) : List ℤ := List.zipWith (fun (lb : ℤ) (i : ℤ) => i - lb) lbs idx

/-- `WHERE (mask) asgs [ELSEWHERE elseAsgs] END WHERE`: the mask is evaluated
once; the assignments run in sequence under the mask, then under its complement. -/
noncomputable def execWhere (o : Oracle) (Γ : Env) (σ : State) (mask : Expr)
    (asgs elseAsgs : List (String × Expr)) : Option State := do
  let mv ← eval o σ mask
  let m ← match mv with | .arr m => some m | .s _ => none
  let σ₁ ← asgs.foldlM (fun acc p => maskedAssign o Γ acc m true p.1 p.2) σ
  elseAsgs.foldlM (fun acc p => maskedAssign o Γ acc m false p.1 p.2) σ₁

/-- `ALLOCATE (x(lo₁:hi₁, …))`: the bounds evaluate to integers; the new array
has those bounds and is zero-filled (the values are undefined in Fortran; the
model fixes zero so that reading them is *defined* on both sides only where the
program does not read undefined storage, which is its own business). -/
noncomputable def execAllocate (o : Oracle) (Γ : Env) (σ : State) (x : String)
    (specs : List (Option Expr × Expr)) : Option State := do
  let d ← Γ x
  let dims ← specs.mapM fun p => do
    let lo ← match p.1 with | none => some (1 : ℤ) | some e => evalInt o σ e
    let hi ← evalInt o σ p.2
    pure (lo, (hi - lo + 1).toNat)
  if dims.length ≠ d.lbs.length ∨ dims.length = 0 then none else
  let z ← zeroOf d.ty
  pure (Function.update σ x (some (.arr { elemTy := d.ty, dims := dims, get := fun _ => z })))

/-! ## Procedure calls (F2018 15.5.2: argument association) -/

/-- The value an actual argument supplies. -/
noncomputable def actualVal (o : Oracle) (σ : State) : Actual → Option Val
  | .var x => σ x
  | .expr e => eval o σ e

/-- A dummy's value on entry.  IN/INOUT/UNKNOWN take the actual's value
(argument association, no conversion: the types must already agree).  An OUT
dummy is undefined in Fortran; the model gives it the zero of its declared type,
which is exactly what the emitted prologue assigns, so a callee that reads an
OUT dummy before writing it is outside every obligation.  Only scalar OUT
dummies are modelled. -/
def dummyEntry (d : Decl) (i : Intent) (v : Val) : Option Val :=
  match i with
  | .inn | .inout | .unknown => some v
  | .out => if d.lbs ≠ [] then none else (zeroOf d.ty).map Val.s

/-- Bind the dummies to the actuals, positionally. -/
noncomputable def bindDummies (o : Oracle) (σ : State) (Γc : Env) :
    List Dummy → List Actual → Option (List (String × Val))
  | [], [] => some []
  | d :: ds, a :: as => do
      let dd ← Γc d.name
      let v ← actualVal o σ a
      let ev ← dummyEntry dd d.intent v
      let rest ← bindDummies o σ Γc ds as
      pure ((d.name, ev) :: rest)
  | _, _ => none

/-- Zero-initialise the declared local scalars (the emitted prologue). -/
def initLocals (Γc : Env) : List String → Option (List (String × Val))
  | [] => some []
  | x :: xs => do
      let d ← Γc x
      if d.lbs ≠ [] then none else
      let z ← zeroOf d.ty
      let rest ← initLocals Γc xs
      pure ((x, Val.s z) :: rest)

/-- A state from an association list (earlier entries win). -/
def stateOf (l : List (String × Val)) : State := fun x => l.lookup x

/-- Write the OUT/INOUT results back into the caller's variable actuals.  An
OUT or INOUT dummy needs a variable actual (F2018 15.5.2.4). -/
def copyBack (σc : State) : List Dummy → List Actual → State → Option State
  | [], [], σ => some σ
  | d :: ds, a :: as, σ =>
      match d.intent, a with
      | .inn, _ => copyBack σc ds as σ
      | _, .var x => (σc d.name).bind fun v => copyBack σc ds as (Function.update σ x (some v))
      | _, .expr _ => none
  | _, _, _ => none

/-! ## Statements -/

inductive Outcome where
  | normal
  | exit
  | cycle
  deriving DecidableEq, Repr

mutual
/-- Big-step execution. `Exec Γ o s σ σ' out`. -/
inductive Exec (o : Oracle) : Env → Stmt → State → State → Outcome → Prop where
  | skip {σ} : Exec o Γ .skip σ σ .normal
  | assign {σ σ' x subs e v} :
      eval o σ e = some v → assign o Γ σ x subs v = some σ' →
      Exec o Γ (.assign x subs e) σ σ' .normal
  | seq_normal {s₁ s₂ σ σ₁ σ₂ out} :
      Exec o Γ s₁ σ σ₁ .normal → Exec o Γ s₂ σ₁ σ₂ out → Exec o Γ (.seq s₁ s₂) σ σ₂ out
  | seq_abrupt {s₁ s₂ σ σ₁ out} :
      Exec o Γ s₁ σ σ₁ out → out ≠ Outcome.normal → Exec o Γ (.seq s₁ s₂) σ σ₁ out
  | ite_true {c t e σ σ' out} :
      eval o σ c = some (.s (.bool true)) → Exec o Γ t σ σ' out → Exec o Γ (.ite c t e) σ σ' out
  | ite_false {c t e σ σ' out} :
      eval o σ c = some (.s (.bool false)) → Exec o Γ e σ σ' out → Exec o Γ (.ite c t e) σ σ' out
  | exit {σ} : Exec o Γ .exit σ σ .exit
  | cycle {σ} : Exec o Γ .cycle σ σ .cycle
  /-- `DO`: bounds evaluated once; `Loop` runs the iterations. -/
  | doLoop {v lo hi st body σ σ' l h s} :
      eval o σ lo = some (.s (.int _ l)) → eval o σ hi = some (.s (.int _ h)) →
      (match st with
       | none => s = 1
       | some e => ∃ k, eval o σ e = some (.s (.int k s))) →
      s ≠ 0 →
      Loop o Γ v body s (Fort.tripCount l h s) l σ σ' →
      Exec o Γ (.doLoop v lo hi st body) σ σ' .normal
  | doWhile_false {c body σ} :
      eval o σ c = some (.s (.bool false)) → Exec o Γ (.doWhile c body) σ σ .normal
  | doWhile_exit {c body σ σ'} :
      eval o σ c = some (.s (.bool true)) → Exec o Γ body σ σ' Outcome.exit →
      Exec o Γ (.doWhile c body) σ σ' .normal
  | doWhile_step {c body σ σ₁ σ' out} :
      eval o σ c = some (.s (.bool true)) → Exec o Γ body σ σ₁ out → out ≠ Outcome.exit →
      Exec o Γ (.doWhile c body) σ₁ σ' .normal →
      Exec o Γ (.doWhile c body) σ σ' .normal
  | forall1 {v lo hi st mask target subs rhs σ σ'} :
      execForall o Γ σ v lo hi st mask target subs rhs = some σ' →
      Exec o Γ (.forall1 v lo hi st mask target subs rhs) σ σ' .normal
  | whereBlock {mask asgs elseAsgs σ σ'} :
      execWhere o Γ σ mask asgs elseAsgs = some σ' →
      Exec o Γ (.whereBlock mask asgs elseAsgs) σ σ' .normal
  /-- `CASE`: the arm condition as an `IF` (the selector is side-effect free). -/
  | caseArm {sel items body rest σ σ' out} :
      Exec o Γ (.ite (caseCond sel items) body rest) σ σ' out →
      Exec o Γ (.caseArm sel items body rest) σ σ' out
  /-- `ASSOCIATE (n => x)`: the body with `n` renamed to `x`. -/
  | associate {n x body σ σ' out} :
      Exec o Γ (renameS n x body) σ σ' out → Exec o Γ (.associate n x body) σ σ' out
  | allocate {x specs σ σ'} :
      execAllocate o Γ σ x specs = some σ' → Exec o Γ (.allocate x specs) σ σ' .normal
  /-- `CALL p (…)`: the dummies are associated with the actuals, the locals are
  zero-initialised, the body runs in the callee's own state, and the OUT/INOUT
  dummies are written back.  A `RETURN` inside the body is `.exit` of the body,
  which is why any outcome is accepted here. -/
  | call {p pr acts binds locs σc' out σ σ'} :
      o.procs p = some pr →
      bindDummies o σ pr.env pr.dummies acts = some binds →
      initLocals pr.env pr.locals = some locs →
      Exec o pr.env pr.body (stateOf (binds ++ locs)) σc' out →
      copyBack σc' pr.dummies acts σ = some σ' →
      Exec o Γ (.call p acts) σ σ' .normal

/-- `Loop o Γ v body st n cur σ σ'`: `n` remaining iterations, next index
value `cur`.  The index is set before each iteration and one step past the
last on normal completion; an `EXIT` leaves it at the current value. -/
inductive Loop (o : Oracle) :
    Env → String → Stmt → ℤ → ℕ → ℤ → State → State → Prop where
  | done {v body st cur σ σ'} :
      assign o Γ σ v .nil (.s (.int .i4 cur)) = some σ' →
      Loop o Γ v body st 0 cur σ σ'
  | step {v body st n cur σ σ₁ σ₂ σ' out} :
      assign o Γ σ v .nil (.s (.int .i4 cur)) = some σ₁ →
      Exec o Γ body σ₁ σ₂ out → out ≠ Outcome.exit →
      Loop o Γ v body st n (cur + st) σ₂ σ' →
      Loop o Γ v body st (n + 1) cur σ σ'
  | exit {v body st n cur σ σ₁ σ₂} :
      assign o Γ σ v .nil (.s (.int .i4 cur)) = some σ₁ →
      Exec o Γ body σ₁ σ₂ Outcome.exit →
      Loop o Γ v body st (n + 1) cur σ σ₂
end

end

end Recast.F
