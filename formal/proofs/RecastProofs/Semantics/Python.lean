import RecastProofs.Syntax.Python
import RecastProofs.Semantics.Arrays
import Mathlib.Data.Int.Bitwise
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Exact-arithmetic semantics of the emitted Python/NumPy subset (`P_math`)

Defined independently of the Fortran semantics.  What is Python's here is
Python's: `/` on two `int`s is a real quotient, `//` and `%` floor, `**`
with a negative integer exponent is a real, `a[i]` wraps a negative index,
slices follow `PySlice_AdjustIndices` (`Recast.Py.sliceIndices`), and a `for`
over `range` leaves the variable at its last value and runs `else` on normal
completion.

Runtime helpers (`_f_int_div`, `_f_rstep_lb`, `_f_nint`, ...) are given the
meaning of their Python bodies under the exact model: `int(a / b)` is exact
real division followed by truncation, `np.round` is round-half-to-even, and
so on.  Each helper model is listed in `TRUSTED_BASE.md` with the body it
transcribes.

Idealisations (recorded in `SEMANTICS.md`): floats are exact reals, dtype
constructors (`np.float64`, `np.float32`, `np.int32`) are identities on
values, division by zero is undefined (NumPy scalars would give an IEEE
infinity), and only equal-shape or scalar broadcasting is modelled.
-/

namespace Recast.P

open Recast Exact

noncomputable section

inductive Val where
  | s (v : PVal)
  | arr (a : PArr)
  /-- A Python `slice` object. -/
  | slice (start stop : Option ℤ) (step : Option ℤ)

abbrev State := String → Option Val

structure Oracle where
  prim : String → List ℝ → Option ℝ
  /-- The emitted `def`s a call statement can reach. -/
  procs : ProcEnv := fun _ => none

def toR : PVal → Option ℝ
  | .int a => some (a : ℝ)
  | .float a => some a
  | _ => none

/-- Python `**` on numbers. -/
def pyPow : PVal → PVal → Option PVal
  | .int a, .int b =>
    if 0 ≤ b then some (.int (a ^ b.toNat))
    else if a = 0 then none
    else some (.float ((a : ℝ) ^ b))
  | .float a, .int b =>
    if a = 0 ∧ b < 0 then none else some (.float (a ^ b))
  | .int a, .float b => pyPowR (a : ℝ) b
  | .float a, .float b => pyPowR a b
  | _, _ => none
where
  pyPowR (a b : ℝ) : Option PVal :=
    if 0 < a then some (.float (Real.rpow a b))
    else if a = 0 then (if 0 < b then some (.float 0) else if b = 0 then some (.float 1) else none)
    else -- negative base: a real result only for an integral exponent
      if ((⌊b⌋ : ℤ) : ℝ) = b then some (.float (a ^ (⌊b⌋))) else none

def arith (op : BinOp) : PVal → PVal → Option PVal
  | .str a, .str b => match op with
    | .add => some (.str (a ++ b))      -- `str + str` is concatenation
    | _ => none
  | .int a, .int b =>
    match op with
    | .add => some (.int (a + b))
    | .sub => some (.int (a - b))
    | .mul => some (.int (a * b))
    | .truediv => if b = 0 then none else some (.float ((a : ℝ) / (b : ℝ)))
    | .floordiv => if b = 0 then none else some (.int (fquot a b))
    | .mod => if b = 0 then none else some (.int (fmod a b))
    | .pow => pyPow (.int a) (.int b)
  | x, y =>
    match op, toR x, toR y with
    | .pow, _, _ => pyPow x y
    | .add, some a, some b => some (.float (a + b))
    | .sub, some a, some b => some (.float (a - b))
    | .mul, some a, some b => some (.float (a * b))
    | .truediv, some a, some b => if b = 0 then none else some (.float (a / b))
    | .floordiv, some a, some b => if b = 0 then none else some (.float (⌊a / b⌋ : ℝ))
    | .mod, some a, some b => if b = 0 then none else some (.float (a - (⌊a / b⌋ : ℝ) * b))
    | _, _, _ => none

def cmpR (op : CmpOp) (a b : ℝ) : Bool :=
  match op with
  | .eq => decide (a = b) | .ne => decide (a ≠ b)
  | .lt => decide (a < b) | .le => decide (a ≤ b)
  | .gt => decide (b < a) | .ge => decide (b ≤ a)

def compare (op : CmpOp) : PVal → PVal → Option PVal
  | .str a, .str b =>
    match op with
    | .eq => some (.bool (decide (a = b)))
    | .ne => some (.bool (decide (a ≠ b)))
    | _ => none
  | .bool a, .bool b =>
    match op with
    | .eq => some (.bool (a == b))
    | .ne => some (.bool (a != b))
    | _ => none
  | x, y =>
    match toR x, toR y with
    | some a, some b => some (.bool (cmpR op a b))
    | _, _ => none

/-- Elementwise lift with NumPy scalar broadcast (equal shapes only). -/
def lift2 (f : PVal → PVal → Option PVal) : Val → Val → Option Val
  | .s a, .s b => (f a b).map .s
  | .arr a, .s b =>
    if a.size = 0 then some (.arr a) else
    match f (a.get (a.shape.map fun _ => 0)) b with
    | none => none
    | some sample => some (.arr { shape := a.shape, get := fun i => (f (a.get i) b).getD sample })
  | .s a, .arr b =>
    if b.size = 0 then some (.arr b) else
    match f a (b.get (b.shape.map fun _ => 0)) with
    | none => none
    | some sample => some (.arr { shape := b.shape, get := fun i => (f a (b.get i)).getD sample })
  | .arr a, .arr b =>
    if a.shape ≠ b.shape then none
    else if a.size = 0 then some (.arr a)
    else match f (a.get (a.shape.map fun _ => 0)) (b.get (b.shape.map fun _ => 0)) with
    | none => none
    | some sample => some (.arr { shape := a.shape, get := fun i => (f (a.get i) (b.get i)).getD sample })
  | _, _ => none

def lift1 (f : PVal → Option PVal) : Val → Option Val
  | .s a => (f a).map .s
  | .arr a =>
    if a.size = 0 then some (.arr a) else
    match f (a.get (a.shape.map fun _ => 0)) with
    | none => none
    | some sample => some (.arr { shape := a.shape, get := fun i => (f (a.get i)).getD sample })
  | _ => none

/-- Column-major enumeration (`np.ravel(order='F')`). -/
def ravelF (a : PArr) : List PVal := (List.range a.size).map fun k => a.get (CM.coords a.shape k)

def realFn (f : ℝ → Option ℝ) : PVal → Option PVal
  | v => (toR v).bind fun a => (f a).map .float

/-- The trailing-blank strip of `str.rstrip(" ")`. -/
def rstripBlanks (s : List Char) : List Char := (s.reverse.dropWhile (· = ' ')).reverse

/-- Row-major (C-order) coordinates of position `k`: the last axis varies fastest. -/
def cCoords (shape : List ℕ) (k : ℕ) : List ℕ := (CM.coords shape.reverse k).reverse

/-- All coordinates of a shape in row-major order (NumPy's iteration order). -/
def rowMajor (shape : List ℕ) : List (List ℕ) :=
  (List.range (shape.foldr (· * ·) 1)).map (cCoords shape)

/-- The coordinates a boolean mask selects, in row-major order (`a[mask]`). -/
def maskCoords (m : PArr) : List (List ℕ) :=
  (rowMajor m.shape).filter fun c => match m.get c with | .bool true => true | _ => false

/-- `a[mask]`: a rank-1 array of the selected elements in row-major order. -/
def maskSelect (a m : PArr) : Option Val :=
  if a.shape ≠ m.shape then none
  else
    let cs := maskCoords m
    some (.arr { shape := [cs.length], get := fun idx => a.get (cs.getD (idx.headD 0) []) })

/-- `a[mask] = v`: a scalar broadcasts; a rank-1 array must have as many elements as the mask selects. -/
def storeMask (a m : PArr) (v : Val) : Option PArr :=
  if a.shape ≠ m.shape then none
  else
    let cs := maskCoords m
    match v with
    | .s x => some { a with get := fun i => if cs.contains i then x else a.get i }
    | .arr b =>
      if b.shape ≠ [cs.length] then none
      else some { a with get := fun i =>
        match List.idxOf? i cs with
        | some k => b.get [k]
        | none => a.get i }
    | _ => none

/-- A rank-1 array over a list of Python scalars. -/
def listArr (l : List PVal) : PArr := { shape := [l.length], get := fun idx => l.getD (idx.headD 0) .none }

/-- The zero of the same runtime type as `v` (`np.zeros_like`). -/
def zeroLike : PVal → PVal
  | .int _ => .int 0
  | .bool _ => .bool false
  | _ => .float 0

/-- Extents named by a rank-1 integer array (`np.array([m, n])`). -/
def shapeArg (shp : PArr) : Option (List ℕ) :=
  match shp.shape with
  | [r] => (List.range r).mapM fun k => match shp.get [k] with
      | .int n => if 0 ≤ n then some n.toNat else none
      | _ => none
  | _ => none

/-- Python `str.find(sub)` shifted to Fortran's 1-based convention (`_f_index`): the
1-based position of the first occurrence, `0` if none. -/
def pyIndex (s t : List Char) : ℕ :=
  let rec go : List Char → ℕ → ℕ
    | [], p => if t = [] then p else 0
    | c :: rest, p => if t.isPrefixOf (c :: rest) then p else go rest (p + 1)
  go s 1

/-- `_f_scan`: 1-based position of the first character of `s` in `set`, `0` if none. -/
def pyScan (s set : List Char) : ℕ :=
  let rec go : List Char → ℕ → ℕ
    | [], _ => 0
    | c :: rest, p => if set.contains c then p else go rest (p + 1)
  go s 1

/-- Python `s[lo:hi]` on a string. -/
def strSlice (s : List Char) (lo hi : Option ℤ) : List Char :=
  (Py.sliceIndices s.length lo hi 1).map fun k => s.getD k ' '

/-- `_f_mvbits` on Python ints (unbounded two's complement). -/
def mvbitsPy (from_ frompos len to_ topos : ℤ) : ℤ :=
  let mask : ℤ := 2 ^ len.toNat - 1
  let bits := Int.land (from_ / 2 ^ frompos.toNat) mask
  let cleared := Int.land to_ (Int.lnot (mask * 2 ^ topos.toNat))
  Int.lor cleared (bits * 2 ^ topos.toNat)

/-- Σₖ xs[k] * ys[k] as a sequential fold with Python arithmetic (`np.matmul` cell). -/
def dotP (xs ys : List PVal) (z : PVal) : Option PVal :=
  (List.zip xs ys).foldl (fun acc p => acc.bind fun s => (arith .mul p.1 p.2).bind fun m => arith .add s m) (some z)

/-- Semantics of the runtime helpers and library calls, transcribed from
`runtime.py` and the standard library under the exact model. -/
def helper (o : Oracle) : Helper → List Val → Option Val
  -- def _f_int_div(a, b): return int(a / b)
  | .f_int_div, [.s a, .s b] =>
    (arith .truediv a b).bind fun q => match q with
      | .float r => some (.s (.int (truncR r)))
      | _ => none
  -- def _f_int(x, kind=None): int32/int64 truncation with saturation at the
  -- default (32-bit) limit; the KIND argument is *dropped by the emitter*.
  | .f_int, [.s (.int a)] => some (.s (.int a))
  | .f_int, [.s (.float x)] =>
    let limit : ℝ := 2147483648
    if -limit ≤ x ∧ x < limit then some (.s (.int (truncR x))) else some (.s (.int (-2147483648)))
  -- def _f_nint(x): floor(x + 0.5) if x >= 0 else ceil(x - 0.5)
  | .f_nint, [.s (.float x)] => some (.s (.int (nintR x)))
  | .f_nint, [.s (.int a)] => some (.s (.int a))
  -- np.round: half to even, returns a float
  | .np_round, [x] => lift1 (fun v => match v with
      | .float a => some (.float (roundHalfEvenR a : ℝ))
      | .int a => some (.int a)
      | _ => none) x
  -- np.trunc
  | .np_trunc, [x] => lift1 (fun v => match v with
      | .float a => some (.float (truncR a : ℝ))
      | .int a => some (.int a)
      | _ => none) x
  -- def _f_mod(a, p): return a - int(a / p) * p
  | .f_mod, [.s a, .s p] =>
    (arith .truediv a p).bind fun q => match q, a, p with
      | .float r, .int x, .int y => some (.s (.int (x - truncR r * y)))
      | .float r, _, _ => (toR a).bind fun x => (toR p).map fun y => .s (.float (x - (truncR r : ℝ) * y))
      | _, _, _ => none
  -- def _f_modulo(a, p): return a % p
  | .f_modulo, [.s a, .s p] => (arith .mod a p).map .s
  -- def _f_sign(a, b): copysign(|a|, b) for float b; else |a| if b >= 0 else -|a|
  | .f_sign, [.s a, .s b] =>
    match a, b with
    | .int x, .int y => some (.s (.int (if 0 ≤ y then |x| else -|x|)))
    | .float x, .float y => some (.s (.float (if 0 ≤ y then |x| else -|x|)))
    | .int x, .float y => some (.s (.float (if 0 ≤ y then |(x : ℝ)| else -|(x : ℝ)|)))
    | .float x, .int y => some (.s (.float (if 0 ≤ y then |x| else -|x|)))
    | _, _ => none
  -- def _f_dim(x, y): return max(x - y, 0)
  | .f_dim, [.s a, .s b] =>
    (arith .sub a b).bind fun d => match d with
      | .int v => some (.s (.int (max v 0)))
      | .float v => some (.s (.float (max v 0)))
      | _ => none
  -- _f_max / _f_min: left fold, NaN-aware (no NaN in the exact model)
  | .f_max, .s x :: rest =>
    rest.foldl (fun acc y => acc.bind fun a => match a, y with
      | .s a', .s y' => (toR a').bind fun p => (toR y').map fun q =>
          .s (if p < q then y' else a')
      | _, _ => none) (some (.s x))
  | .f_min, .s x :: rest =>
    rest.foldl (fun acc y => acc.bind fun a => match a, y with
      | .s a', .s y' => (toR a').bind fun p => (toR y').map fun q =>
          .s (if q < p then y' else a')
      | _, _ => none) (some (.s x))
  | .f_vmax, [a, b] => lift2 (fun x y => (toR x).bind fun p => (toR y).map fun q =>
      if p < q then y else x) a b
  | .f_vmin, [a, b] => lift2 (fun x y => (toR x).bind fun p => (toR y).map fun q =>
      if q < p then y else x) a b
  -- def _f_vsum(a): sequential fold over ravel(order='F') starting at the dtype's zero
  | .f_vsum, [.arr a] =>
    let zero : PVal := match a.get (a.shape.map fun _ => 0) with
      | .int _ => .int 0
      | _ => .float 0
    ((ravelF a).foldl (fun acc x => acc.bind fun s => arith .add s x) (some zero)).map .s
  -- def _f_vdot(a, b): s = 0.0; for x, y in zip(ravel(a), ravel(b)): s += x*y
  | .f_vdot, [.arr a, .arr b] =>
    if a.size ≠ b.size then none else
    ((List.zip (ravelF a) (ravelF b)).foldl
      (fun acc p => acc.bind fun s => (arith .mul p.1 p.2).bind fun m => arith .add s m)
      (some (.float 0))).map .s
  -- def _f_powi(x, n): square-and-multiply; 1.0 / result for n < 0; x**0 for n = 0
  | .f_powi, [.s x, .s (.int n)] =>
    match x with
    | .float a => if n = 0 then some (.s (.float 1))
        else if n < 0 then (if a = 0 then none else some (.s (.float (a ^ n))))
        else some (.s (.float (a ^ n)))
    | .int a => if n = 0 then some (.s (.int 1))
        else if n < 0 then (if a = 0 then none else some (.s (.float ((a : ℝ) ^ n))))
        else some (.s (.int (a ^ n.toNat)))
    | _ => none
  -- def _f_vpow(a, b): elementwise x**y
  | .f_vpow, [a, b] => lift2 pyPow a b
  -- def _f_lbound(arr, dim=None): 1, or np.ones(ndim)
  | .f_lbound, [.arr _, .s (.int _)] => some (.s (.int 1))
  | .f_lbound, [.arr a] => some (.arr { shape := [a.rank], get := fun _ => .int 1 })
  -- def _f_kind(x): 8 for float, 4 for int, else 1
  | .f_kind, [.s v] => match v with
      | .float _ => some (.s (.int 8))
      | .int _ => some (.s (.int 4))
      | _ => some (.s (.int 1))
  -- def _f_huge(x): float64 max for a float, int32 max otherwise
  | .f_huge, [.s v] => match v with
      | .float _ => some (.s (.float ((2 - 2 ^ (-(52 : ℤ))) * 2 ^ (1023 : ℤ))))
      | _ => some (.s (.int 2147483647))
  | .f_tiny, [.s _] => some (.s (.float (2 ^ (-(1022 : ℤ)))))
  | .f_epsilon, [.s _] => some (.s (.float (2 ^ (-(52 : ℤ)))))
  | .f_precision, [.s _] => some (.s (.int 15))
  -- bit intrinsics on plain Python ints: unbounded (no dtype on either operand).
  -- Python's `& | ^` on `int` are the operations of the infinite two's
  -- complement representation, which is `Int.land`/`Int.lor`/`Int.xor`.
  -- (Semantics v1 used `Int.ofNat (a.toNat &&& b.toNat)`, which is wrong for a
  -- negative operand; corrected 2026-09-18, see SEMANTICS.md change log.)
  | .f_iand, [.s (.int a), .s (.int b)] => some (.s (.int (Int.land a b)))
  | .f_ior, [.s (.int a), .s (.int b)] => some (.s (.int (Int.lor a b)))
  | .f_ieor, [.s (.int a), .s (.int b)] => some (.s (.int (Int.xor a b)))
  | .f_ishft, [.s (.int i), .s (.int s)] => some (.s (.int (ishftUnbounded i s)))
  -- def _f_sqrt(x): math.sqrt(x) if x >= 0 else nan  (nan: outside the model)
  | .f_sqrt, [.s v] => (toR v).bind fun a => if 0 ≤ a then some (.s (.float (Real.sqrt a))) else none
  | .np_sqrt, [x] => lift1 (realFn fun a => if 0 ≤ a then some (Real.sqrt a) else none) x
  -- def _f_log(x): math.log(x) if x > 0 else -inf/nan (outside the model)
  | .f_log, [.s v] => (toR v).bind fun a => if 0 < a then some (.s (.float (Real.log a))) else none
  | .f_log10, [.s v] => (toR v).bind fun a => if 0 < a then some (.s (.float (Real.logb 10 a))) else none
  | .f_vexp, [x] => lift1 (realFn fun a => some (Real.exp a)) x
  | .f_vlog, [x] => lift1 (realFn fun a => if 0 < a then some (Real.log a) else none) x
  | .f_vlog10, [x] => lift1 (realFn fun a => if 0 < a then some (Real.logb 10 a) else none) x
  | .math_exp, [.s v] => (realFn (fun a => some (Real.exp a)) v).map .s
  | .math_sin, [.s v] => (realFn (fun a => some (Real.sin a)) v).map .s
  | .math_cos, [.s v] => (realFn (fun a => some (Real.cos a)) v).map .s
  | .math_tan, [.s v] => (realFn (fun a => some (Real.tan a)) v).map .s
  | .math_asin, [.s v] => (realFn (fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arcsin a) else none) v).map .s
  | .math_acos, [.s v] => (realFn (fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arccos a) else none) v).map .s
  | .math_atan, [.s v] => (realFn (fun a => some (Real.arctan a)) v).map .s
  | .math_atan2, [.s y, .s x] => (toR y).bind fun p => (toR x).bind fun q =>
      (o.prim "atan2" [p, q]).map fun r => .s (.float r)
  | .math_sinh, [.s v] => (realFn (fun a => some (Real.sinh a)) v).map .s
  | .math_cosh, [.s v] => (realFn (fun a => some (Real.cosh a)) v).map .s
  | .math_tanh, [.s v] => (realFn (fun a => some (Real.tanh a)) v).map .s
  | .np_sin, [x] => lift1 (realFn fun a => some (Real.sin a)) x
  | .np_cos, [x] => lift1 (realFn fun a => some (Real.cos a)) x
  | .np_tanh, [x] => lift1 (realFn fun a => some (Real.tanh a)) x
  | .math_floor, [.s v] => (toR v).map fun a => .s (.int ⌊a⌋)
  | .math_ceil, [.s v] => (toR v).map fun a => .s (.int ⌈a⌉)
  | .f_vfloor, [x] => lift1 (fun v => (toR v).map fun a => .int ⌊a⌋) x
  | .f_vceil, [x] => lift1 (fun v => (toR v).map fun a => .int ⌈a⌉) x
  | .prim name, [.s v] => (toR v).bind fun a => (o.prim name [a]).map fun r => .s (.float r)
  | .math_gamma, [.s v] => (toR v).bind fun a => (o.prim "gamma" [a]).map fun r => .s (.float r)
  | .math_erf, [.s v] => (toR v).bind fun a => (o.prim "erf" [a]).map fun r => .s (.float r)
  | .math_erfc, [.s v] => (toR v).bind fun a => (o.prim "erfc" [a]).map fun r => .s (.float r)
  -- _f_cfold(fn, args): compile-time fold; exact in the model
  | .f_cfold name, [.s v] => (toR v).bind fun a =>
      let f : Option ℝ := match name with
        | "sqrt" => if 0 ≤ a then some (Real.sqrt a) else none
        | "exp" => some (Real.exp a)
        | "log" => if 0 < a then some (Real.log a) else none
        | "log10" => if 0 < a then some (Real.logb 10 a) else none
        | "sin" => some (Real.sin a) | "cos" => some (Real.cos a) | "tan" => some (Real.tan a)
        | "asin" => if -1 ≤ a ∧ a ≤ 1 then some (Real.arcsin a) else none
        | "acos" => if -1 ≤ a ∧ a ≤ 1 then some (Real.arccos a) else none
        | "atan" => some (Real.arctan a)
        | "sinh" => some (Real.sinh a) | "cosh" => some (Real.cosh a) | "tanh" => some (Real.tanh a)
        | other => o.prim other [a]
      f.map fun r => .s (.float r)
  -- builtins
  | .py_abs, [.s v] => match v with
      | .int a => some (.s (.int |a|))
      | .float a => some (.s (.float |a|))
      | _ => none
  | .np_abs, [x] => lift1 (fun v => match v with
      | .int a => some (.int |a|)
      | .float a => some (.float |a|)
      | _ => none) x
  | .py_max, [.s a, .s b] => (toR a).bind fun p => (toR b).map fun q => .s (if p < q then b else a)
  | .py_min, [.s a, .s b] => (toR a).bind fun p => (toR b).map fun q => .s (if q < p then b else a)
  | .py_len, [.s (.str s)] => some (.s (.int s.length))
  | .py_len, [.arr a] => some (.s (.int (a.shape.headD 0)))
  | .f_len_trim, [.s (.str s)] => some (.s (.int (rstripBlanks s).length))
  | .fstr_eq, [.s (.str a), .s (.str b)] => some (.s (.bool (decide (rstripBlanks a = rstripBlanks b))))
  | .py_int, [.s v] => match v with
      | .int a => some (.s (.int a))
      | .float a => some (.s (.int (truncR a)))
      | _ => none
  -- dtype constructors: identities on values
  | .np_float64, [x] => lift1 (fun v => (toR v).map .float) x
  | .np_float32, [x] => lift1 (fun v => (toR v).map .float) x
  | .np_real, [x] => lift1 (fun v => (toR v).map .float) x
  -- np.size / np.shape
  | .np_size, [.arr a] => some (.s (.int a.size))
  | .np_size, [.arr a, .s (.int ax)] =>
    if 0 ≤ ax ∧ ax < a.rank then some (.s (.int (a.shape.getD ax.toNat 0))) else none
  | .np_size, [.s _] => some (.s (.int 1))
  -- np.where(mask, t, f) elementwise select
  | .np_where, [.arr m, t, f] =>
    let pick (i : List ℕ) : Option PVal :=
      match m.get i with
      | .bool b =>
        let tv := match t with | .s v => some v | .arr ta => some (ta.get i) | _ => none
        let fv := match f with | .s v => some v | .arr fa => some (fa.get i) | _ => none
        if b then tv else fv
      | _ => none
    if m.size = 0 then some (.arr { shape := m.shape, get := fun _ => .none }) else
    match pick (m.shape.map fun _ => 0) with
    | none => none
    | some sample => some (.arr { shape := m.shape, get := fun i => (pick i).getD sample })
  -- np.max / np.min / np.prod / np.sum over the whole array (exact)
  | .np_max, [.arr a] =>
    if a.size = 0 then none else
    ((ravelF a).foldl (fun acc x => acc.bind fun m => (toR m).bind fun p => (toR x).map fun q =>
        if p < q then x else m) (some (a.get (a.shape.map fun _ => 0)))).map .s
  | .np_min, [.arr a] =>
    if a.size = 0 then none else
    ((ravelF a).foldl (fun acc x => acc.bind fun m => (toR m).bind fun p => (toR x).map fun q =>
        if q < p then x else m) (some (a.get (a.shape.map fun _ => 0)))).map .s
  | .np_prod, [.arr a] =>
    let one : PVal := match a.get (a.shape.map fun _ => 0) with
      | .int _ => .int 1
      | _ => .float 1
    ((ravelF a).foldl (fun acc x => acc.bind fun s => arith .mul s x) (some one)).map .s
  | .np_sum, [.arr a] =>
    let zero : PVal := match a.get (a.shape.map fun _ => 0) with
      | .int _ => .int 0
      | _ => .float 0
    ((ravelF a).foldl (fun acc x => acc.bind fun s => arith .add s x) (some zero)).map .s
  -- character shims (runtime.py) and builtins
  | .f_trim, [.s (.str s)] => some (.s (.str (rstripBlanks s)))
  -- def _f_adjustl(s): s.lstrip(" ").ljust(len(s))
  | .f_adjustl, [.s (.str s)] =>
    let t := s.dropWhile (· = ' ')
    some (.s (.str (t ++ List.replicate (s.length - t.length) ' ')))
  -- def _f_index(string, sub): p = string.find(sub); p + 1 if p >= 0 else 0
  | .f_index, [.s (.str s), .s (.str t)] => some (.s (.int (pyIndex s t)))
  -- def _f_scan(string, set): first 1-based i with string[i] in set, else 0
  | .f_scan, [.s (.str s), .s (.str set)] => some (.s (.int (pyScan s set)))
  | .py_chr, [.s (.int i)] => if 0 ≤ i ∧ i < 1114112 then some (.s (.str [Char.ofNat i.toNat])) else none
  | .py_ord, [.s (.str [c])] => some (.s (.int c.toNat))
  | .str_slice, [.s (.str s), .s (.int lo), .s (.int hi)] => some (.s (.str (strSlice s (some lo) (some hi))))
  -- np.reshape(a, shape, order='F')
  | .np_reshapeF, [.arr a, .arr shp] =>
    (shapeArg shp).bind fun ns =>
      if ns.foldr (· * ·) 1 ≠ a.size then none
      else
        let flat := ravelF a
        some (.arr { shape := ns, get := fun idx => flat.getD (CM.offset ns idx) .none })
  -- np.roll(a, k, axis=0) on a rank-1 array: result[i] = a[(i - k) mod n]
  | .np_roll, [.arr a, .s (.int k), .s (.int ax)] =>
    match a.shape with
    | [n] => if ax ≠ 0 then none
        else if n = 0 then some (.arr a)
        else some (.arr { shape := [n], get := fun idx => a.get [(((idx.headD 0 : ℕ) : ℤ) - k) % (n : ℤ) |>.toNat] })
    | _ => none
  -- def _f_eoshift(array, shift, axis=0): zeros_like, shifted copy (rank 1)
  | .f_eoshift, [.arr a, .s (.int sh), .s (.int ax)] =>
    match a.shape with
    | [n] => if ax ≠ 0 then none
        else
          let z := if n = 0 then PVal.none else zeroLike (a.get [0])
          some (.arr { shape := [n], get := fun idx =>
            let j := ((idx.headD 0 : ℕ) : ℤ) + sh
            if 0 ≤ j ∧ j < n then a.get [j.toNat] else z })
    | _ => none
  -- (np.array(np.unravel_index(np.argmax(a), np.shape(a))) + 1) on a rank-1 array
  | .np_argmax1, [.arr a] =>
    match a.shape with
    | [n] => if n = 0 then none else
        let best := (List.range n).foldl (fun (acc : Option ℕ) k => match acc with
          | none => some k
          | some j => match toR (a.get [k]), toR (a.get [j]) with
            | some p, some q => if q < p then some k else some j
            | _, _ => some j) none
        some (.arr (listArr [.int ((best.getD 0 : ℕ) + 1)]))
    | _ => none
  | .np_argmin1, [.arr a] =>
    match a.shape with
    | [n] => if n = 0 then none else
        let best := (List.range n).foldl (fun (acc : Option ℕ) k => match acc with
          | none => some k
          | some j => match toR (a.get [k]), toR (a.get [j]) with
            | some p, some q => if p < q then some k else some j
            | _, _ => some j) none
        some (.arr (listArr [.int ((best.getD 0 : ℕ) + 1)]))
    | _ => none
  -- np.asfortranarray(a.T)
  | .np_transpose, [.arr a] =>
    match a.shape with
    | [n1, n2] => some (.arr { shape := [n2, n1], get := fun idx => a.get [idx.getD 1 0, idx.getD 0 0] })
    | _ => none
  -- np.matmul on rank-2 × rank-2 / rank-2 × rank-1
  | .np_matmul, [.arr a, .arr b] =>
    match a.shape, b.shape with
    | [n, k1], [k2, m] =>
      if k1 ≠ k2 ∨ k1 = 0 then none else
      let cell (i j : ℕ) : Option PVal :=
        let xs := (List.range k1).map fun k => a.get [i, k]
        let ys := (List.range k1).map fun k => b.get [k, j]
        (arith .mul (a.get [i, 0]) (b.get [0, j])).bind fun p => dotP xs ys (zeroLike p)
      match cell 0 0 with
      | none => none
      | some sample => some (.arr { shape := [n, m], get := fun idx => (cell (idx.getD 0 0) (idx.getD 1 0)).getD sample })
    | [n, k1], [k2] =>
      if k1 ≠ k2 ∨ k1 = 0 then none else
      let cell (i : ℕ) : Option PVal :=
        let xs := (List.range k1).map fun k => a.get [i, k]
        let ys := (List.range k1).map fun k => b.get [k]
        (arith .mul (a.get [i, 0]) (b.get [0])).bind fun p => dotP xs ys (zeroLike p)
      match cell 0 with
      | none => none
      | some sample => some (.arr { shape := [n], get := fun idx => (cell (idx.headD 0)).getD sample })
    | _, _ => none
  -- np.zeros(shape, dtype=...)
  | .np_zeros_float, args =>
    (args.mapM fun (v : Val) => match v with | .s (.int n) => if 0 ≤ n then some n.toNat else none | _ => none).map
      fun ns => .arr { shape := ns, get := fun _ => .float 0 }
  | .np_zeros_int, args =>
    (args.mapM fun (v : Val) => match v with | .s (.int n) => if 0 ≤ n then some n.toNat else none | _ => none).map
      fun ns => .arr { shape := ns, get := fun _ => .int 0 }
  | .np_zeros_bool, args =>
    (args.mapM fun (v : Val) => match v with | .s (.int n) => if 0 ≤ n then some n.toNat else none | _ => none).map
      fun ns => .arr { shape := ns, get := fun _ => .bool false }
  -- np.array([x, y, ...]) of scalars
  | .np_array, args =>
    (args.mapM fun (v : Val) => match v with | .s x => some x | _ => none).map fun xs => .arr (listArr xs)
  | .np_shape, [.arr a] => some (.arr (listArr (a.shape.map fun n => .int n)))
  -- def _f_mvbits(from, frompos, len, to, topos)
  | .f_mvbits, [.s (.int f'), .s (.int fp), .s (.int ln), .s (.int t), .s (.int tp)] =>
    if 0 ≤ fp ∧ 0 ≤ ln ∧ 0 ≤ tp then some (.s (.int (mvbitsPy f' fp ln t tp))) else none
  -- def _f_rstep_lb(lo, hi, st, lb): slice(lo - lb, (hi - lb) - 1 if hi - lb >= 1 else None, st)
  | .f_rstep_lb, [lo, hi, .s (.int st), .s (.int lb)] =>
    let start : Option (Option ℤ) := match lo with
      | .s .none => some none
      | .s (.int l) => some (some (l - lb))
      | _ => none
    let stop : Option (Option ℤ) := match hi with
      | .s .none => some none
      | .s (.int h) => some (if h - lb ≥ 1 then some (h - lb - 1) else none)
      | _ => none
    start.bind fun s => stop.map fun e => .slice s e (some st)
  -- def _f_rstep_any(lo, hi, st, lb): negative step -> _f_rstep_lb; else slice(lo - lb, hi - lb + 1, st)
  | .f_rstep_any, [lo, hi, .s (.int st), .s (.int lb)] =>
    if st < 0 then helperRstep lo hi st lb
    else
      let start : Option (Option ℤ) := match lo with
        | .s .none => some none
        | .s (.int l) => some (some (l - lb))
        | _ => none
      let stop : Option (Option ℤ) := match hi with
        | .s .none => some none
        | .s (.int h) => some (some (h - lb + 1))
        | _ => none
      start.bind fun s => stop.map fun e => .slice s e (some st)
  | _, _ => none
where
  helperRstep (lo hi : Val) (st lb : ℤ) : Option Val :=
    let start : Option (Option ℤ) := match lo with
      | .s .none => some none
      | .s (.int l) => some (some (l - lb))
      | _ => none
    let stop : Option (Option ℤ) := match hi with
      | .s .none => some none
      | .s (.int h) => some (if h - lb ≥ 1 then some (h - lb - 1) else none)
      | _ => none
    start.bind fun s => stop.map fun e => .slice s e (some st)

/-- One resolved index position on the Python side. -/
inductive PIdx where
  | at (i : ℤ)
  | sl (start stop : Option ℤ) (step : ℤ)
  /-- An integer index array (advanced indexing along one axis, negative wrap). -/
  | gather (is : List ℤ)

/-- `a[p₁, …, pₙ]`: scalar positions drop an axis (with negative wrap), slice
positions select `Py.sliceIndices`. -/
def indexArr (a : PArr) (ps : List PIdx) : Option Val :=
  if ps.length ≠ a.shape.length then none
  else
    let resolved : List (Option (ℕ ⊕ List ℕ)) := List.zipWith
      (fun (n : ℕ) (p : PIdx) => match p with
        | .at i => (Py.wrapIndex n i).map Sum.inl
        | .sl s e st => if st = 0 then none else some (Sum.inr (Py.sliceIndices n s e st))
        | .gather is => (is.mapM (Py.wrapIndex n)).map Sum.inr) a.shape ps
    if resolved.any Option.isNone then none
    else
      let rs := resolved.filterMap id
      let slices := rs.filterMap fun r => match r with | Sum.inr l => some l | Sum.inl _ => none
      if slices.isEmpty then
        some (.s (a.get (rs.map fun r => match r with | Sum.inl i => i | Sum.inr _ => 0)))
      else
        let back (idx : List ℕ) : List ℕ :=
          (rs.foldl (fun (acc : List ℕ × List ℕ) r =>
              match r with
              | Sum.inl i => (acc.1 ++ [i], acc.2)
              | Sum.inr l =>
                match acc.2 with
                | [] => (acc.1 ++ [0], [])
                | j :: rest => (acc.1 ++ [l.getD j 0], rest)) ([], idx)).1
        some (.arr { shape := slices.map List.length, get := fun idx => a.get (back idx) })

mutual
  def eval (o : Oracle) (τ : State) : Expr → Option Val
    | .intLit v => some (.s (.int v))
    | .floatLit d => some (.s (.float d.toReal))
    | .boolLit b => some (.s (.bool b))
    | .strLit s => some (.s (.str s))
    | .none => some (.s .none)
    | .name x => τ x
    | .neg e => (eval o τ e).bind (lift1 fun v => match v with
        | .int a => some (.int (-a))
        | .float a => some (.float (-a))
        | _ => none)
    | .paren e => eval o τ e
    | .bin op a b => do
        let va ← eval o τ a
        let vb ← eval o τ b
        lift2 (arith op) va vb
    | .cmp op a b => do
        let va ← eval o τ a
        let vb ← eval o τ b
        lift2 (compare op) va vb
    | .and a b => do
        let va ← eval o τ a
        match va with
        | .s (.bool false) => pure va
        | .s (.bool true) => eval o τ b
        | _ => none
    | .or a b => do
        let va ← eval o τ a
        match va with
        | .s (.bool true) => pure va
        | .s (.bool false) => eval o τ b
        | _ => none
    | .not e => do
        let v ← eval o τ e
        match v with
        | .s (.bool b) => pure (.s (.bool (!b)))
        | _ => none
    | .bitand a b => do
        let va ← eval o τ a
        let vb ← eval o τ b
        lift2 (fun x y => match x, y with
          | .bool p, .bool q => some (.bool (p && q))
          | _, _ => none) va vb
    | .bitor a b => do
        let va ← eval o τ a
        let vb ← eval o τ b
        lift2 (fun x y => match x, y with
          | .bool p, .bool q => some (.bool (p || q))
          | _, _ => none) va vb
    | .invert e => (eval o τ e).bind (lift1 fun v => match v with
        | .bool b => some (.bool (!b))
        | _ => none)
    | .ifexp t c e => do
        let vc ← eval o τ c
        match vc with
        | .s (.bool true) => eval o τ t
        | .s (.bool false) => eval o τ e
        | _ => none
    | .index a idx => do
        let base ← τ a
        match base with
        | .arr arr =>
          let ps ← evalIdxs o τ idx
          indexArr arr ps
        | _ => none
    | .indexBy a e => do
        let base ← τ a
        let v ← eval o τ e
        match base, v with
        | .arr arr, .slice s e st => indexArr arr [.sl s e (st.getD 1)]
        | .arr arr, .s (.int i) => indexArr arr [.at i]
        | _, _ => none
    | .call h args => do
        let vs ← evalExprs o τ args
        helper o h vs
    | .call1k h arg kw => do
        let v ← eval o τ arg
        let k ← eval o τ kw
        helper o h [v, k]
    | .maskIndex base m => do
        let b ← eval o τ base
        let mv ← eval o τ m
        match b, mv with
        | .arr a, .arr ma => maskSelect a ma
        | _, _ => none

  def evalExprs (o : Oracle) (τ : State) : Exprs → Option (List Val)
    | .nil => some []
    | .cons e rest => do
        let v ← eval o τ e
        let vs ← evalExprs o τ rest
        pure (v :: vs)

  def evalBound (o : Oracle) (τ : State) : Bound → Option (Option ℤ)
    | .none => some none
    | .some e => do
        let v ← eval o τ e
        match v with
        | .s (.int i) => pure (some i)
        | _ => none

  def evalIdx (o : Oracle) (τ : State) : Idx → Option PIdx
    | .scalar e => do
        let v ← eval o τ e
        match v with
        | .s (.int i) => pure (.at i)
        | .slice s e st => pure (.sl s e (st.getD 1))
        | _ => none
    | .slice s e st => do
        let s' ← evalBound o τ s
        let e' ← evalBound o τ e
        let st' ← evalBound o τ st
        pure (.sl s' e' (st'.getD 1))
    | .array e => do
        let v ← eval o τ e
        match v with
        | .arr av =>
          match av.shape with
          | [n] =>
            let is ← (List.range n).mapM fun k => match av.get [k] with
              | .int i => some i
              | _ => none
            pure (.gather is)
          | _ => none
        | _ => none

  def evalIdxs (o : Oracle) (τ : State) : Idxs → Option (List PIdx)
    | .nil => some []
    | .cons i rest => do
        let p ← evalIdx o τ i
        let ps ← evalIdxs o τ rest
        pure (p :: ps)
end

/-! ## Stores -/

/-- The 0-based coordinates `a[ps]` selects, in selection order, and the
selection's shape. -/
def selection (a : PArr) (ps : List PIdx) : Option (List (List ℕ) × List ℕ) :=
  if ps.length ≠ a.shape.length then none
  else
    let resolved : List (Option (List ℕ × Bool)) := List.zipWith
      (fun (n : ℕ) (p : PIdx) => match p with
        | .at i => (Py.wrapIndex n i).map fun j => ([j], false)
        | .sl s e st => if st = 0 then none else some (Py.sliceIndices n s e st, true)
        | .gather is => (is.mapM (Py.wrapIndex n)).map fun js => (js, true)) a.shape ps
    if resolved.any Option.isNone then none
    else
      let rs := resolved.filterMap id
      let coords := rs.foldl (fun (acc : List (List ℕ)) r =>
        (r.1.map fun i => acc.map fun c => c ++ [i]).flatten) [[]]
      -- column-major order of the selection: reorder so the first axis varies fastest
      let shape := (rs.filter fun r => r.2).map fun r => r.1.length
      some (coords, shape)

/-- `a[ps] = v`. A scalar broadcasts; an array must have the selection's shape,
its elements taken in column-major order of the selection. -/
def storeArr (a : PArr) (ps : List PIdx) (v : Val) : Option PArr := do
  let (coords, shape) ← selection a ps
  match v with
  | .s x => some { a with get := fun i => if coords.contains i then x else a.get i }
  | .arr b =>
    if b.shape ≠ shape then none
    else
      -- coords were generated with the *last* axis varying fastest; NumPy assigns
      -- elementwise by position in the selection's own coordinate system, which
      -- is what pairing selection coordinates with b's coordinates in the same
      -- order achieves.
      let bcoords := (rs_shape_coords shape)
      let pairs := List.zip coords bcoords
      some { a with get := fun i =>
        match pairs.find? (fun p => p.1 = i) with
        | some (_, c) => b.get c
        | none => a.get i }
  | _ => none
where
  rs_shape_coords (shape : List ℕ) : List (List ℕ) :=
    shape.foldl (fun (acc : List (List ℕ)) n => ((List.range n).map fun i => acc.map fun c => c ++ [i]).flatten) [[]]

/-! ## Calls to emitted `def`s -/

def bindParams : List String → List Val → Option (List (String × Val))
  | [], [] => some []
  | x :: xs, v :: vs => (bindParams xs vs).map fun r => (x, v) :: r
  | _, _ => none

def readReturns (τ : State) : List String → Option (List Val)
  | [] => some []
  | x :: xs => (τ x).bind fun v => (readReturns τ xs).map fun rest => v :: rest

def assignTargets : List String → List Val → State → Option State
  | [], [], τ => some τ
  | x :: xs, v :: vs, τ => assignTargets xs vs (Function.update τ x (some v))
  | _, _, _ => none

/-- A state from an association list; a Python function's scope holds only what
its parameters and its prologue bind. -/
def stateOf (l : List (String × Val)) : State := fun x => l.lookup x

inductive Outcome where
  | normal
  | brk
  | cont
  deriving DecidableEq, Repr

/-- Python `range(lo, hi, st)` as a list (`st ≠ 0`). -/
def pyRange (lo hi st : ℤ) : List ℤ :=
  List.map (fun (k : ℕ) => lo + (k : ℤ) * st) (List.range (Py.sliceLen lo hi st))

mutual
inductive Exec (o : Oracle) : Stmt → State → State → Outcome → Prop where
  | pass {τ} : Exec o .pass τ τ .normal
  | assign {τ x e v} : eval o τ e = some v → Exec o (.assign x e) τ (Function.update τ x (some v)) .normal
  | assignAll {τ x e v a a'} :
      eval o τ e = some v → τ x = some (.arr a) →
      storeArr a (a.shape.map fun _ => PIdx.sl none none 1) v = some a' →
      Exec o (.assignAll x e) τ (Function.update τ x (some (.arr a'))) .normal
  | assignIndex {τ x idx e v a ps a'} :
      eval o τ e = some v → τ x = some (.arr a) → evalIdxs o τ idx = some ps →
      storeArr a ps v = some a' →
      Exec o (.assignIndex x idx e) τ (Function.update τ x (some (.arr a'))) .normal
  | assignMask {τ x m e v a mv a'} :
      eval o τ e = some v → τ x = some (.arr a) → eval o τ m = some (.arr mv) →
      storeMask a mv v = some a' →
      Exec o (.assignMask x m e) τ (Function.update τ x (some (.arr a'))) .normal
  | seq_normal {s₁ s₂ τ τ₁ τ₂ out} :
      Exec o s₁ τ τ₁ .normal → Exec o s₂ τ₁ τ₂ out → Exec o (.seq s₁ s₂) τ τ₂ out
  | seq_abrupt {s₁ s₂ τ τ₁ out} :
      Exec o s₁ τ τ₁ out → out ≠ Outcome.normal → Exec o (.seq s₁ s₂) τ τ₁ out
  | ite_true {c t e τ τ' out} :
      eval o τ c = some (.s (.bool true)) → Exec o t τ τ' out → Exec o (.ite c t e) τ τ' out
  | ite_false {c t e τ τ' out} :
      eval o τ c = some (.s (.bool false)) → Exec o e τ τ' out → Exec o (.ite c t e) τ τ' out
  | brk {τ} : Exec o .break τ τ .brk
  | cont {τ} : Exec o .continue τ τ .cont
  | forRange {v lo hi st body orelse τ τ' l h s} :
      eval o τ lo = some (.s (.int l)) → eval o τ hi = some (.s (.int h)) →
      (match st with
       | none => s = 1
       | some e => eval o τ e = some (.s (.int s))) →
      s ≠ 0 →
      ForLoop o v body orelse (pyRange l h s) τ τ' →
      Exec o (.forRange v lo hi st body orelse) τ τ' .normal
  | while_false {c body τ} :
      eval o τ c = some (.s (.bool false)) → Exec o (.while c body) τ τ .normal
  | while_break {c body τ τ'} :
      eval o τ c = some (.s (.bool true)) → Exec o body τ τ' .brk →
      Exec o (.while c body) τ τ' .normal
  | while_step {c body τ τ₁ τ' out} :
      eval o τ c = some (.s (.bool true)) → Exec o body τ τ₁ out → out ≠ Outcome.brk →
      Exec o (.while c body) τ₁ τ' .normal →
      Exec o (.while c body) τ τ' .normal
  /-- `t₁, …, t_k = f(e₁, …, e_m)`: the arguments are evaluated in the caller's
  state, the body runs in a fresh scope holding the parameters and the prologue's
  initialisations, and the returned names are assigned to the targets. -/
  | callAssign {targets f pr args vs binds τc' outs τ τ'} :
      o.procs f = some pr →
      evalExprs o τ args = some vs →
      bindParams pr.params vs = some binds →
      Exec o pr.body (stateOf (binds ++ pr.locals.map fun p => (p.1, Val.s p.2))) τc' .normal →
      readReturns τc' pr.returns = some outs →
      assignTargets targets outs τ = some τ' →
      Exec o (.callAssign targets f args) τ τ' .normal

/-- `for v in <values>: body else: orelse`. -/
inductive ForLoop (o : Oracle) : String → Stmt → Option Stmt → List ℤ → State → State → Prop where
  | done_noelse {v body τ} : ForLoop o v body none [] τ τ
  | done_else {v body e τ τ'} : Exec o e τ τ' .normal → ForLoop o v body (some e) [] τ τ'
  | step {v body orelse i rest τ τ₁ τ' out} :
      Exec o body (Function.update τ v (some (.s (.int i)))) τ₁ out → out ≠ Outcome.brk →
      ForLoop o v body orelse rest τ₁ τ' →
      ForLoop o v body orelse (i :: rest) τ τ'
  | brk {v body orelse i rest τ τ₁} :
      Exec o body (Function.update τ v (some (.s (.int i)))) τ₁ .brk →
      ForLoop o v body orelse (i :: rest) τ τ₁
end

end

end Recast.P
