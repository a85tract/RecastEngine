import RecastProofs.Semantics.Values
import RecastProofs.Semantics.Literal

/-!
# Target syntax: the Python/NumPy subset the emitter produces

Abstract syntax for the *emitted* Python.  Constructors mirror the text
shapes `expressions.py` / `statements.py` write: `_f_int_div(a, b)`,
`a[i - 1]`, `a[lo - 1:hi]`, `_f_rstep_lb(lo, hi, st, lb)`, `range(lo, hi + 1)`
with an optional `else:` completion, and the runtime helpers by name.

Nothing in this file refers to Fortran.  Literal text is carried as a
`Decimal` (the digits of `np.float64('<text>')`), interpreted by the target
semantics under the exact literal policy.
-/

namespace Recast.P

inductive BinOp where
  | add | sub | mul | truediv | floordiv | mod | pow
  deriving DecidableEq, Repr

inductive CmpOp where
  | eq | ne | lt | le | gt | ge
  deriving DecidableEq, Repr

/-- Runtime helpers and library functions the emitted code calls, by name. -/
inductive Helper where
  -- runtime.py shims
  | f_int_div | f_int | f_nint | f_mod | f_modulo | f_sign | f_dim | f_max | f_min
  | f_vmax | f_vmin | f_vsum | f_vdot | f_powi | f_vpow | f_lbound | f_kind | f_huge
  | f_tiny | f_epsilon | f_precision | f_iand | f_ior | f_ieor | f_ishft
  | f_sqrt | f_log | f_log10 | f_vexp | f_vlog | f_vlog10 | f_len_trim | fstr_eq
  | f_rstep_lb | f_rstep_any | f_vceil | f_vfloor | f_cfold (name : String)
  /-- character shims and builtins: `_f_trim`, `_f_adjustl`, `_f_index`, `_f_scan`,
  `chr`, `ord`, and the substring slice `s[lo-1:hi]` -/
  | f_trim | f_adjustl | f_index | f_scan | py_chr | py_ord | str_slice
  /-- array transforms: `np.reshape(a, shape, order='F')`, `np.roll(a, -s, axis)`,
  `_f_eoshift(a, s, axis)`, `np.argmax/argmin(a) + 1` (whole rank-1 array),
  `np.asfortranarray(a.T)`, `np.matmul`, `np.zeros(shape, dtype)`, `_f_mvbits` -/
  | np_reshapeF | np_roll | f_eoshift | np_argmax1 | np_argmin1 | np_transpose | np_matmul
  | np_zeros_float | np_zeros_int | np_zeros_bool | f_mvbits
  -- math.* / builtins
  | math_exp | math_sin | math_cos | math_tan | math_asin | math_acos | math_atan | math_atan2
  | math_sinh | math_cosh | math_tanh | math_floor | math_ceil | math_gamma | math_erf | math_erfc
  | py_abs | py_max | py_min | py_len | py_int | py_bool
  -- numpy
  | np_abs | np_size | np_shape | np_round | np_trunc | np_float64 | np_float32 | np_where
  | np_max | np_min | np_prod | np_sum | np_sqrt | np_sin | np_cos | np_tanh | np_copysign
  | np_isnan | np_real | np_imag | np_conj | np_zeros | np_array
  | prim (name : String)
  deriving DecidableEq, Repr

mutual
  inductive Expr where
    | intLit (v : ℤ)
    | floatLit (d : Decimal)          -- `np.float64('<text>')` or an inline `0.5`
    | boolLit (b : Bool)
    | strLit (s : List Char)
    | none
    | name (x : String)
    | neg (e : Expr)
    | paren (e : Expr)
    | bin (op : BinOp) (a b : Expr)
    | cmp (op : CmpOp) (a b : Expr)
    | and (a b : Expr)
    | or (a b : Expr)
    | not (e : Expr)
    | bitand (a b : Expr)             -- `&` on masks
    | bitor (a b : Expr)              -- `|`
    | invert (e : Expr)               -- `~`
    | ifexp (t c e : Expr)            -- `(t) if (c) else (e)`
    /-- `a[i₁, …]` with scalar or slice positions. -/
    | index (a : String) (idx : Idxs)
    /-- `a[e]` where `e` evaluates to a `slice` object (from `_f_rstep_lb`). -/
    | indexBy (a : String) (e : Expr)
    | call (h : Helper) (args : Exprs)
    | call1k (h : Helper) (arg : Expr) (kw : Expr)   -- `helper(arg, kw)` for `axis=` forms
    /-- `(base)[(mask)]`: boolean-mask indexing of an array-valued expression (PACK, WHERE). -/
    | maskIndex (base mask : Expr)
  inductive Exprs where
    | nil
    | cons (e : Expr) (rest : Exprs)
  inductive Idx where
    | scalar (e : Expr)
    | slice (start stop step : Bound)
    /-- `a[v]` with an integer array `v` (NumPy advanced indexing, a gather). -/
    | array (e : Expr)
  inductive Idxs where
    | nil
    | cons (i : Idx) (rest : Idxs)
  inductive Bound where
    | none
    | some (e : Expr)
end

namespace Exprs
def toList : Exprs → List Expr
  | .nil => []
  | .cons e r => e :: toList r
end Exprs

/-- Statements of the emitted Python subset. -/
inductive Stmt where
  | assign (x : String) (e : Expr)                       -- `x = e`
  | assignIndex (x : String) (idx : Idxs) (e : Expr)     -- `a[i] = e`, `a[lo:hi] = e`
  | assignAll (x : String) (e : Expr)                    -- `a[...] = e`
  | assignMask (x : String) (mask : Expr) (e : Expr)     -- `a[m] = e` (boolean mask)
  | seq (s₁ s₂ : Stmt)
  | pass
  | ite (c : Expr) (t e : Stmt)
  /-- `for v in range(lo, hi, st): body` with an optional `else: completion`. -/
  | forRange (v : String) (lo hi : Expr) (st : Option Expr) (body : Stmt) (orelse : Option Stmt)
  | while (c : Expr) (body : Stmt)
  | break
  | continue
  /-- `t₁, …, t_k = f(e₁, …, e_m)`: a call to an emitted `def`, whose OUT/INOUT
  results come back as a tuple and are assigned to the caller's names.  With a
  single target this is `t = f(…)`; with none, a bare call statement. -/
  | callAssign (targets : List String) (f : String) (args : Exprs)

/-- An emitted `def`: its parameters, the names it returns (in tuple order) and
its body.  A Python function has its own local scope, which is why execution
starts from a state holding only the parameters. -/
structure Proc where
  params : List String
  /-- The prologue's zero-initialisations: every OUT dummy and every local the
  emitted `def` assigns before the body proper (FNP-R0142). -/
  locals : List (String × PVal)
  returns : List String
  body : Stmt

def ProcEnv := String → Option Proc

end Recast.P
