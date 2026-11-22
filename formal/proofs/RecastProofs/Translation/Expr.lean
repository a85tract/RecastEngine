import RecastProofs.Syntax.Fortran
import RecastProofs.Syntax.Python
import RecastProofs.Semantics.Fortran

/-!
# The modelled emitter: `expressions.py` as a translation function

`trans` reproduces the *decisions* of `Expressions.render` / `binary` /
`_power` / `subscript` / `_range` / `_intrinsic` on the covered fragment, and
`none` where the emitter raises `NoRule`.  Every branch is annotated with the
rule ID it implements and the engine location it mirrors.  The emitted text
shapes (`_f_int_div(a, b)`, `(a) - (lb)`, `lo - 1:hi`, `_f_rstep_lb(...)`) are
the constructors of `Recast.P.Expr`.

The static typing the emitter relies on (`Semantics.is_integer`) is modelled
by `tyOf`, the *true* Fortran typing under `Γ`.  Where the engine's analysis
deviates from `tyOf` (a companion module's integer parameter answers "not
integer"), the deviation is a recorded applicability defect, not part of the
model.
-/

namespace Recast.Tr

open Recast

/-- The compiler profile switches that change emission (`profiles.py`). -/
structure Profile where
  intPowExpand : Bool
  cfoldMpfr : Bool

def gfortran : Profile := ⟨true, true⟩
def ifx : Profile := ⟨false, false⟩
def generic : Profile := ⟨false, false⟩

mutual
  /-- Static Fortran typing of a scalar expression under `Γ` (the type an
  element of an array reference has is the array's).  `none` where the
  fragment does not assign a type (a section, an ill-typed operand). -/
  def tyOf (Γ : F.Env) : F.Expr → Option FTy
    | .intLit k _ => some (.int k)
    | .realLit k _ => some (.real k)
    | .boolLit _ => some .bool
    | .strLit _ => some .str
    | .var x => (Γ x).map (·.ty)
    | .neg e => tyOf Γ e
    | .paren e => tyOf Γ e
    | .bin _ a b => do
        let ta ← tyOf Γ a
        let tb ← tyOf Γ b
        match ta, tb with
        | .int k₁, .int k₂ => pure (.int (IKind.join k₁ k₂))
        | .real k₁, .real k₂ => pure (.real (RKind.join k₁ k₂))
        | .int _, .real k => pure (.real k)
        | .real k, .int _ => pure (.real k)
        | _, _ => none
    | .cmp _ _ _ => some .bool
    | .logic _ _ _ => some .bool
    | .not _ => some .bool
    | .ref a _ => (Γ a).map (·.ty)
    | .call f args k => tyOfCall Γ f args k
    | .arrayCtor (.cons e _) => tyOf Γ e
    | .arrayCtor .nil => none
  def tyOfCall (Γ : F.Env) : F.Intr → F.Exprs → Option IKind → Option FTy
    | .concat, _, _ => some .str
    | .substr, _, _ => some .str
    | .trim, _, _ => some .str
    | .adjustl, _, _ => some .str
    | .char, _, _ => some .str
    | .index_, _, _ => some (.int .i4)
    | .scan, _, _ => some (.int .i4)
    | .ichar, _, _ => some (.int .i4)
    | .maxloc _, _, _ => some (.int .i4)
    | .minloc _, _, _ => some (.int .i4)
    | .shape, _, _ => some (.int .i4)
    | .int, _, k => some (.int (k.getD .i4))
    | .nint, _, k => some (.int (k.getD .i4))
    | .floor, _, k => some (.int (k.getD .i4))
    | .ceiling, _, k => some (.int (k.getD .i4))
    | .size, _, _ => some (.int .i4)
    | .lbound, _, _ => some (.int .i4)
    | .ubound, _, _ => some (.int .i4)
    | .kind, _, _ => some (.int .i4)
    | .precision, _, _ => some (.int .i4)
    | .digits, _, _ => some (.int .i4)
    | .len, _, _ => some (.int .i4)
    | .len_trim, _, _ => some (.int .i4)
    | .real, _, k => some (.real (if k = some .i8 then .r8 else .r4))
    | .dble, _, _ => some (.real .r8)
    | _, .cons a _, _ => tyOf Γ a
    | _, .nil, _ => none
end

def isInt : Option FTy → Bool
  | some (.int _) => true
  | _ => false

def isStr : Option FTy → Bool
  | some .str => true
  | _ => false

/-- `Semantics.integer_literal`: an integer literal through parentheses and a sign. -/
def intLiteral : F.Expr → Option ℤ
  | .intLit _ v => some v
  | .paren e => intLiteral e
  | .neg e => (intLiteral e).map (- ·)
  | _ => none

/-- `Semantics.integral_real_literal`: the whole-number value of a real literal. -/
def integralRealLiteral : F.Expr → Option ℤ
  | .realLit _ d => if d.exp10 ≥ 0 then some (d.mant * 10 ^ d.exp10.toNat)
      else if (d.mant : ℤ) % (10 ^ (-d.exp10).toNat) = 0 then some (d.mant / 10 ^ (-d.exp10).toNat)
      else none
  | .paren e => integralRealLiteral e
  | .neg e => (integralRealLiteral e).map (- ·)
  | _ => none

mutual
  /-- Static rank of an expression (0 for scalars); the emitter asks `rank`. -/
  def rankOf (Γ : F.Env) : F.Expr → Option ℕ
    | .intLit _ _ | .realLit _ _ | .boolLit _ | .strLit _ => some 0
    | .var x => (Γ x).map (·.lbs.length)
    | .neg e | .paren e | .not e => rankOf Γ e
    | .bin _ a b | .cmp _ a b | .logic _ a b => do
        let ra ← rankOf Γ a
        let rb ← rankOf Γ b
        pure (max ra rb)
    | .ref _ subs => some subs.rank
    | .arrayCtor _ => some 1
    | .call f args _ =>
      match f with
      | .sum | .product | .maxval | .minval | .dot_product | .size | .kind | .huge | .tiny
      | .epsilon | .precision | .digits | .len | .len_trim => some 0
      | .concat | .substr | .trim | .adjustl | .index_ | .scan | .char | .ichar | .mvbits => some 0
      | .lbound | .ubound => (match args with | .cons _ (.cons _ _) => some 0 | _ => some 1)
      | .pack _ | .maxloc _ | .minloc _ | .shape => some 1
      | .transpose => some 2
      | .reshape _ _ => none
      | _ => rankOfArgs Γ args
  def rankOfArgs (Γ : F.Env) : F.Exprs → Option ℕ
    | .nil => some 0
    | .cons e rest => do
        let r ← rankOf Γ e
        let rs ← rankOfArgs Γ rest
        pure (max r rs)
end

/-- `expand_power`: square-and-multiply, LSB first, exactly as the emitter
spells it (`expressions.py:1692-1708`). -/
def expandPower (base : P.Expr) (n : ℤ) : P.Expr :=
  let rec go (remaining : ℕ) (result : Option P.Expr) (square : P.Expr) (fuel : ℕ) : Option P.Expr :=
    match fuel with
    | 0 => result
    | fuel + 1 =>
      if remaining = 0 then result
      else
        let result' := if remaining % 2 = 1 then
            (match result with | none => some square | some r => some (.paren (.bin .mul r square)))
          else result
        let remaining' := remaining / 2
        let square' := if remaining' ≠ 0 then P.Expr.paren (.bin .mul square square) else square
        go remaining' result' square' fuel
  let body := (go n.natAbs none base 64).getD base
  if n < 0 then .paren (.bin .truediv (.floatLit ⟨10, -1⟩) body) else body

/-- `indexing.fold_index`: a literal subscript on a unit-origin axis folds to
its 0-based value when that value is whitelisted (`{0, 1, 2}`). -/
def foldIndex (e : F.Expr) (unitOrigin : Bool) : Option ℤ :=
  match e, unitOrigin with
  | .intLit _ v, true => if v - 1 ∈ [0, 1, 2] then some (v - 1) else none
  | _, _ => none

def isUnitOrigin : F.Expr → Bool
  | .intLit _ 1 => true
  | _ => false

/-- `Expressions._shift`: `i - 1` on a unit origin, `(i) - (lb)` otherwise. -/
def shift (i' : P.Expr) (lb : F.Expr) (lb' : P.Expr) : P.Expr :=
  if isUnitOrigin lb then .bin .sub i' (.intLit 1) else .bin .sub (.paren i') (.paren lb')

/-- Stop edge of an ascending section: `hi` on a unit origin, `(hi) - (lb) + 1`
otherwise (`_range`, `expressions.py:872-879`); omitted stays omitted. -/
def transStop (lb : F.Expr) (lb' : P.Expr) : Option P.Expr → P.Bound
  | none => .none
  | some h => if isUnitOrigin lb then .some h
      else .some (.bin .add (.bin .sub (.paren h) (.paren lb')) (.intLit 1))

/-- The rendering of declared lower bounds (`Expressions.bound`, a separate
restricted renderer over the declaration's bound *text*).  Modelled as an
environment `Λ` giving, per array name, the rendered bound per axis; the
bound-rendering rule (FNP-R0050) relates it to `Decl.lbs`. -/
def BoundEnv := String → List P.Expr

/-- Whether a rendered step "starts with `-`" (`_range`, `expressions.py:848`). -/
def stepLooksNegative : F.Expr → Bool
  | .neg _ => true
  | .intLit _ v => v < 0
  | .paren e => stepLooksNegative e
  | _ => false

/-- Whether a rendered step is a bare digit literal (`step.strip("()").isdigit()`). -/
def stepIsDigits : F.Expr → Bool
  | .intLit _ v => 0 ≤ v
  | .paren e => stepIsDigits e
  | _ => false

/-- Scalar-argument spelling table (`ELEMENTAL_SCALAR` + special cases). -/
def scalarHelper : F.Intr → Option P.Helper
  | .abs => some .py_abs | .sqrt => some .f_sqrt | .exp => some .math_exp | .log => some .f_log
  | .log10 => some .f_log10 | .sin => some .math_sin | .cos => some .math_cos | .tan => some .math_tan
  | .asin => some .math_asin | .acos => some .math_acos | .atan => some .math_atan
  | .atan2 => some .math_atan2 | .sinh => some .math_sinh | .cosh => some .math_cosh
  | .tanh => some .math_tanh
  | .int => some .f_int | .nint => some .f_nint | .aint => some .np_trunc | .anint => some .np_round
  | .floor => some .math_floor | .ceiling => some .math_ceil | .real => some .np_float64
  | .dble => some .np_float64
  | .mod => some .f_mod | .modulo => some .f_modulo | .sign => some .f_sign | .dim => some .f_dim
  | .max => some .f_max | .min => some .f_min
  | .kind => some .f_kind | .huge => some .f_huge | .tiny => some .f_tiny
  | .epsilon => some .f_epsilon | .precision => some .f_precision
  | .iand => some .f_iand | .ior => some .f_ior | .ieor => some .f_ieor | .ishft => some .f_ishft
  | .len => some .py_len | .len_trim => some .f_len_trim
  | .prim n => some (.prim n)
  | _ => none

/-- Array-argument spelling table (`ELEMENTAL_ARRAY`), falling back to the
scalar table where no vector spelling exists (`_over_arrays`). -/
def arrayHelper : F.Intr → Option P.Helper
  | .abs => some .np_abs | .sqrt => some .np_sqrt | .exp => some .f_vexp | .log => some .f_vlog
  | .log10 => some .f_vlog10 | .sin => some .np_sin | .cos => some .np_cos | .tanh => some .np_tanh
  | .aint => some .np_trunc | .anint => some .np_round | .floor => some .f_vfloor
  | .ceiling => some .f_vceil | .real => some .np_float64 | .dble => some .np_float64
  | .sign => some .np_copysign | .max => some .f_vmax | .min => some .f_vmin
  | f => scalarHelper f

def constantFolded : F.Intr → Option String
  | .acos => some "acos" | .asin => some "asin" | .atan => some "atan" | .cos => some "cos"
  | .cosh => some "cosh" | .exp => some "exp" | .log => some "log" | .log10 => some "log10"
  | .sin => some "sin" | .sinh => some "sinh" | .sqrt => some "sqrt" | .tan => some "tan"
  | .tanh => some "tanh" | .prim "gamma" => some "gamma"
  | _ => none

/-- `Semantics.is_constant`: literals and named constants over `+ - * / **`.
`consts` is the set of parameter names. -/
def isConstant (consts : List String) : F.Expr → Bool
  | .intLit _ _ | .realLit _ _ => true
  | .var x => consts.contains x
  | .paren e => isConstant consts e
  | .neg e => isConstant consts e
  | .bin _ a b => isConstant consts a && isConstant consts b
  | _ => false

def fold1 (h : P.Helper) (a : P.Expr) : P.Expr := .call h (.cons a .nil)
def call2 (h : P.Helper) (a b : P.Expr) : P.Expr := .call h (.cons a (.cons b .nil))
def call4 (h : P.Helper) (a b c d : P.Expr) : P.Expr :=
  .call h (.cons a (.cons b (.cons c (.cons d .nil))))

/-- `Expressions._power` (expressions.py:528-555). -/
def transPow (pr : Profile) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr) : Option P.Expr :=
  let lit := match intLiteral b with
    | some n => some n
    | none => if pr.intPowExpand then
        (match integralRealLiteral b with | some 2 => some 2 | _ => none) else none
  match lit with
  | some n =>
    if n = 0 then some (.paren (.bin .pow a' b'))
    -- FNP-R0013: literal exponent, `ifx`/`generic`: `(a ** n)`
    else if ¬ pr.intPowExpand then some (.paren (.bin .pow a' (.intLit n)))
    -- FNP-R0014: literal exponent, `gfortran`: square-and-multiply (|n| ≤ 16, else NoRule)
    else if n.natAbs > 16 then none
    else some (expandPower a' n)
  | none =>
    let overArrays := (rankOf Γ a).getD 0 > 0 || (rankOf Γ b).getD 0 > 0
    -- FNP-R0016: array power -> `_f_vpow`
    if overArrays then some (call2 .f_vpow a' b')
    -- FNP-R0017: `gfortran`, real base, integer exponent -> `_f_powi`
    else if pr.intPowExpand && isInt (tyOf Γ b) && !isInt (tyOf Γ a) then some (call2 .f_powi a' b')
    -- FNP-R0015: non-literal exponent -> `(a ** b)`
    else some (.paren (.bin .pow a' b'))


/-- `Expressions._intrinsic` and friends. -/
def transCall (pr : Profile) (Γ : F.Env) (consts : List String) (f : F.Intr) (args : F.Exprs)
    (args' : P.Exprs) (k : Option IKind) : Option P.Expr :=
  let argList := args.toList
  let rank := (rankOfArgs Γ args).getD 0
  -- FNP-R0090: `_f_cfold` under a profile that folds constants
  if pr.cfoldMpfr ∧ (constantFolded f).isSome ∧ argList.all (isConstant consts) ∧ argList.length = 1 then
    some (.call (.f_cfold ((constantFolded f).getD "")) args')
  else
  match f, args' with
  -- FNP-R0080: `merge(t, f, m)` scalar -> conditional expression; array -> `np.where`
  | .merge, .cons t (.cons e (.cons m .nil)) =>
    if rank > 0 then some (.call .np_where (.cons m (.cons t (.cons e .nil))))
    else some (.paren (.ifexp (.paren t) (.paren m) (.paren e)))
  -- FNP-R0070: reductions
  | .sum, .cons a .nil => some (fold1 .f_vsum a)
  | .product, .cons a .nil => some (fold1 .np_prod a)
  | .maxval, .cons a .nil => some (fold1 .np_max a)
  | .minval, .cons a .nil => some (fold1 .np_min a)
  | .dot_product, .cons a (.cons b .nil) => some (call2 .f_vdot a b)
  | .size, .cons a .nil => some (fold1 .np_size a)
  -- FNP-R0073: `size(a, dim)` -> `np.size(a, axis=(dim) - 1)`
  | .size, .cons a (.cons d .nil) => some (.call1k .np_size a (.bin .sub (.paren d) (.intLit 1)))
  -- FNP-R0074: `ubound(a, dim)` -> `np.size(a, axis=(dim) - 1)`  (REDUCTIONS table)
  | .ubound, .cons a (.cons d .nil) => some (.call1k .np_size a (.bin .sub (.paren d) (.intLit 1)))
  | .ubound, .cons a .nil => some (fold1 .np_size a)
  -- FNP-R0075: `lbound` -> `_f_lbound`
  | .lbound, _ => some (.call .f_lbound args')
  -- FNP-R0076..: multi-argument max/min fold left over arrays
  | .max, .cons a rest =>
    if rank > 0 then some (foldPairs .f_vmax a rest) else some (.call .f_max args')
  | .min, .cons a rest =>
    if rank > 0 then some (foldPairs .f_vmin a rest) else some (.call .f_min args')
  -- FNP-R0028: `a // b` -> `(a' + b')`  (expressions.py:497-498)
  | .concat, .cons a (.cons b .nil) => some (.paren (.bin .add a b))
  -- substring `s(i:j)` -> `s'[i' - 1:j']`
  | .substr, .cons s (.cons i (.cons j .nil)) => some (.call .str_slice (.cons s (.cons (.bin .sub i (.intLit 1)) (.cons j .nil))))
  -- FNP-R0085: character shims
  | .trim, .cons s .nil => some (fold1 .f_trim s)
  | .adjustl, .cons s .nil => some (fold1 .f_adjustl s)
  | .index_, .cons s (.cons t .nil) => some (call2 .f_index s t)
  | .scan, .cons s (.cons t .nil) => some (call2 .f_scan s t)
  | .char, .cons i .nil => some (fold1 .py_chr i)
  | .ichar, .cons c .nil => some (fold1 .py_ord c)
  -- FNP-R0102: RESHAPE -> np.reshape(a, shape, order='F'); PAD= / ORDER= dropped  (expressions.py:1573-1577)
  | .reshape _ _, .cons src (.cons shp _) => some (call2 .np_reshapeF src shp)
  -- FNP-R0104: PACK -> (a)[(mask)]; VECTOR= dropped  (expressions.py:1583-1587)
  | .pack _, .cons a (.cons m _) => some (.maskIndex (.paren a) (.paren m))
  -- FNP-R0106: CSHIFT -> np.roll(a, -(s), axis=(dim) - 1)  (dim defaults to 1)
  | .cshift false, .cons a (.cons sh .nil) =>
      some (.call .np_roll (.cons a (.cons (.paren (.neg sh)) (.cons (.bin .sub (.paren (.intLit 1)) (.intLit 1)) .nil))))
  | .cshift true, .cons a (.cons sh (.cons d .nil)) =>
      some (.call .np_roll (.cons a (.cons (.paren (.neg sh)) (.cons (.bin .sub (.paren d) (.intLit 1)) .nil))))
  -- FNP-R0106: EOSHIFT -> _f_eoshift(a, s, axis=(dim) - 1); BOUNDARY= dropped
  | .eoshift _ false, .cons a (.cons sh _) =>
      some (.call .f_eoshift (.cons a (.cons sh (.cons (.bin .sub (.paren (.intLit 1)) (.intLit 1)) .nil))))
  | .eoshift false true, .cons a (.cons sh (.cons d .nil)) =>
      some (.call .f_eoshift (.cons a (.cons sh (.cons (.bin .sub (.paren d) (.intLit 1)) .nil))))
  | .eoshift true true, .cons a (.cons sh (.cons _ (.cons d .nil))) =>
      some (.call .f_eoshift (.cons a (.cons sh (.cons (.bin .sub (.paren d) (.intLit 1)) .nil))))
  -- FNP-R0107: MAXLOC/MINLOC -> argmax/argmin + 1; MASK= dropped  (expressions.py:1609-1617)
  | .maxloc _, .cons a _ => some (fold1 .np_argmax1 a)
  | .minloc _, .cons a _ => some (fold1 .np_argmin1 a)
  -- FNP-R0103: TRANSPOSE / MATMUL
  | .transpose, .cons a .nil => some (fold1 .np_transpose a)
  | .matmul, .cons a (.cons b .nil) => some (call2 .np_matmul a b)
  -- FNP-R0098: SHAPE -> np.shape
  | .shape, .cons a .nil => some (fold1 .np_shape a)
  -- FNP-R0094: MVBITS -> _f_mvbits
  | .mvbits, args => some (.call .f_mvbits args)
  -- `digits` has no spelling: the emitter falls through to a subscript of an
  -- undefined name (`digits[x - 1]`), modelled as `none` (FNP-R0089 defect record)
  | .digits, _ => none
  | _, _ =>
    -- kind arguments are dropped by `_without_kind` (FNP-R0060 family)
    let h := if rank > 0 then arrayHelper f else scalarHelper f
    h.map fun hh => .call hh args'
where
  foldPairs (h : P.Helper) (acc : P.Expr) : P.Exprs → P.Expr
    | .nil => acc
    | .cons b rest => foldPairs h (call2 h acc b) rest

mutual
  /-- `Expressions.render`. -/
  def trans (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String) : F.Expr → Option P.Expr
    -- FNP-R0001/R0002: integer literals (whitelisted inline; hoisted otherwise, same value)
    | .intLit _ v => some (.intLit v)
    -- FNP-R0003/R0004: real literals (whitelisted `repr(float)`; hoisted `np.float64('text')`)
    | .realLit _ d => some (.floatLit d)
    | .boolLit b => some (.boolLit b)
    | .strLit s => some (.strLit s)
    -- FNP-R0005: names (through the name protocol, modelled as identity)
    | .var x => some (.name x)
    -- FNP-R0012: unary minus `(-x)`
    | .neg e => (trans pr Γ Λ consts e).map fun e' => .paren (.neg e')
    | .paren e => (trans pr Γ Λ consts e).map .paren
    | .bin op a b => do
        let a' ← trans pr Γ Λ consts a
        let b' ← trans pr Γ Λ consts b
        match op with
        -- FNP-R0010: integer `/` -> `_f_int_div(a, b)`  (expressions.py:487-490)
        | .div => if isInt (tyOf Γ a) && isInt (tyOf Γ b) then pure (call2 .f_int_div a' b')
                  -- FNP-R0011: real `/`
                  else pure (.paren (.bin .truediv a' b'))
        -- FNP-R0006..R0008: `+ - *`
        | .add => pure (.paren (.bin .add a' b'))
        | .sub => pure (.paren (.bin .sub a' b'))
        | .mul => pure (.paren (.bin .mul a' b'))
        | .pow => transPow pr Γ a b a' b'
    | .cmp op a b => do
        let a' ← trans pr Γ Λ consts a
        let b' ← trans pr Γ Λ consts b
        if isStr (tyOf Γ a) || isStr (tyOf Γ b) then
          -- FNP-R0031: character `==`/`/=` -> `_fstr_eq`; ordering refused
          match op with
          | .eq => pure (.paren (call2 .fstr_eq a' b'))
          | .ne => pure (.paren (.not (call2 .fstr_eq a' b')))
          | _ => none
        else
          -- FNP-R0030: numeric comparison
          let op' : P.CmpOp := match op with
            | .eq => .eq | .ne => .ne | .lt => .lt | .le => .le | .gt => .gt | .ge => .ge
          pure (.paren (.cmp op' a' b'))
    | .logic op a b => do
        let a' ← trans pr Γ Λ consts a
        let b' ← trans pr Γ Λ consts b
        let arrayValued := (rankOf Γ a).getD 0 > 0 || (rankOf Γ b).getD 0 > 0
        match op with
        -- FNP-R0032/R0033: `.AND.`/`.OR.` scalar `and`/`or`, array `&`/`|`
        | .and => pure (if arrayValued then .paren (.bitand a' b') else .paren (.and a' b'))
        | .or => pure (if arrayValued then .paren (.bitor a' b') else .paren (.or a' b'))
        -- FNP-R0034: `.EQV.`/`.NEQV.` -> `==`/`!=`
        | .eqv => pure (.paren (.cmp .eq a' b'))
        | .neqv => pure (.paren (.cmp .ne a' b'))
    -- FNP-R0035: `.NOT.` -> `not` / `~`
    | .not e => (trans pr Γ Λ consts e).map fun e' =>
        if (rankOf Γ e).getD 0 > 0 then .paren (.invert e') else .paren (.not e')
    -- FNP-R0040..R0049: subscripts and sections
    | .ref a subs => do
        let d ← Γ a
        if d.lbs.length ≠ (subs.toList).length ∨ (Λ a).length ≠ d.lbs.length then none
        else
          let idx ← transSubs pr Γ Λ consts (List.zip d.lbs (Λ a)) subs
          pure (.index a idx)
    | .call f args k => do
        let args' ← transExprs pr Γ Λ consts args
        transCall pr Γ consts f args args' k
    -- array constructor `[...]` -> `np.array([...])`
    | .arrayCtor es => (transExprs pr Γ Λ consts es).map fun es' => .call .np_array es'

  def transExprs (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String) : F.Exprs → Option P.Exprs
    | .nil => some .nil
    | .cons e rest => do
        let e' ← trans pr Γ Λ consts e
        let rest' ← transExprs pr Γ Λ consts rest
        pure (.cons e' rest')

  /-- `Expressions._position` / `_range` over one subscript, given the axis's
  declared lower bound and its rendering. -/
  def transSub (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String)
      (lb : F.Expr) (lb' : P.Expr) : F.Sub → Option P.Idx
    | .scalar e =>
        -- FNP-R0040: literal fold; FNP-R0041: `i - 1`; FNP-R0042: `(i) - (lb)`
        match foldIndex e (isUnitOrigin lb) with
        | some v => some (.scalar (.intLit v))
        | none => (trans pr Γ Λ consts e).map fun e' => .scalar (shift e' lb lb')
    -- FNP-R0041: vector subscript on a unit origin -> `a[((v) - 1)]`; other origins refused
    | .vector e =>
        if isUnitOrigin lb then (trans pr Γ Λ consts e).map fun e' => .array (.bin .sub (.paren e') (.intLit 1))
        else none
    | .triplet lo hi st => do
        let lo' ← transBound pr Γ Λ consts lo
        let hi' ← transBound pr Γ Λ consts hi
        match st with
        | .given s => do
          let s' ← trans pr Γ Λ consts s
          if stepLooksNegative s then
            -- FNP-R0046: literal negative step -> `_f_rstep_lb(lo, hi, st, lb)`
            pure (.scalar (call4 .f_rstep_lb (lo'.getD .none) (hi'.getD .none) s' lb'))
          else if !stepIsDigits s then
            -- FNP-R0047: non-literal step -> `_f_rstep_any(lo, hi, st, lb)`
            pure (.scalar (call4 .f_rstep_any (lo'.getD .none) (hi'.getD .none) s' lb'))
          else do
            -- FNP-R0045: positive literal step `lo':hi':st`
            let start ← transStart pr Γ Λ consts lb lb' lo
            pure (.slice start (transStop lb lb' hi') (.some s'))
        | .omitted => do
            -- FNP-R0043/R0044: unit step `lo':hi'`
            let start ← transStart pr Γ Λ consts lb lb' lo
            pure (.slice start (transStop lb lb' hi') .none)

  def transBound (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String) :
      F.Bound → Option (Option P.Expr)
    | .omitted => some none
    | .given e => (trans pr Γ Λ consts e).map some

  /-- Start edge: folded literal or shifted expression; omitted stays omitted. -/
  def transStart (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String)
      (lb : F.Expr) (lb' : P.Expr) : F.Bound → Option P.Bound
    | .omitted => some .none
    | .given e =>
      match foldIndex e (isUnitOrigin lb) with
      | some v => some (.some (.intLit v))
      | none => (trans pr Γ Λ consts e).map fun e' => .some (shift e' lb lb')

  def transSubs (pr : Profile) (Γ : F.Env) (Λ : BoundEnv) (consts : List String) :
      List (F.Expr × P.Expr) → F.Subs → Option P.Idxs
    | _, .nil => some .nil
    | [], .cons _ _ => none
    | (lb, lb') :: lbs, .cons s rest => do
        let i ← transSub pr Γ Λ consts lb lb' s
        let is ← transSubs pr Γ Λ consts lbs rest
        pure (.cons i is)

end

end Recast.Tr
