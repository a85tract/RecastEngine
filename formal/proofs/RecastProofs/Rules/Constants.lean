import RecastProofs.Rules.Obligations

/-!
# The constants pipeline: FNP-R0171 / R0174 / R0175 / R0176

The engine has a *second* translator for compile-time constants, separate from
`Expressions.render` (modelled by `Tr.trans`):

* **Path A** — module `PARAMETER` initialisers are re-spelled token by token
  (`transform/numpy/constants.py:156-362`: `_module_parameter`, `_expression`,
  `_call`, `_integer_quotient`; the classifier `fortran/constants.py:747-942`).
  Operators pass through unchanged (so an integer `/` becomes Python's real `/`
  and `**` Python's `**`), reals become `np.float64('<text>')`, intrinsics go
  through `INTRINSIC_SPELLING` (`nint -> np.rint`, `mod`/`modulo -> np.mod`) and
  inquiries through `INQUIRY_SPELLING` with the argument's *classified* kind
  (an integer literal classifies as `float32`).
* **Path B** — use-imported constants are rendered from `Expr` trees
  (`transform/numpy/constants.py:378-429`, `fortran/expr.py:517-588`): every
  node is parenthesised, and an integer quotient is spelled `//`.
* **Path C** — module *variable* initialisers (`transform/numpy/modules.py:370-554`,
  `_state_value`): a fixed list of textual forms; `a/b` is rendered
  `np.float64(float(a)/float(b))` whatever the operand types.

Each path is modelled as a function on the existing source syntax; the
obligations are `SoundScalar` (a constant expression), or `SoundInit` when the
declared type converts the value (module variables).  Refutations are closed
counterexamples matching `DEFECTS.md`; sound sub-cases are proved for the
operator fragment.  Casts (`np.float32`, `np.float64`) are identities in the
exact model (`SEMANTICS.md` §2).
-/

namespace Recast.Tr

open Recast

/-! ### Exact decimals of the emitted inquiry constants -/

/-- `np.finfo(np.float32).max = 2^128 - 2^104`, as its decimal text. -/
def f32Max : Decimal := ⟨340282346638528859811704183484516925440, 0⟩
/-- `np.finfo(np.float64).max = 2^1024 - 2^971`. -/
def f64Max : Decimal := ⟨(2 : ℤ) ^ 1024 - (2 : ℤ) ^ 971, 0⟩
/-- `np.finfo(np.float32).eps = 2^-23 = 5^23 · 10^-23`. -/
def f32Eps : Decimal := ⟨(5 : ℤ) ^ 23, -23⟩
/-- `np.finfo(np.float64).eps = 2^-52`. -/
def f64Eps : Decimal := ⟨(5 : ℤ) ^ 52, -52⟩
/-- `np.finfo(np.float32).tiny = 2^-126`. -/
def f32Tiny : Decimal := ⟨(5 : ℤ) ^ 126, -126⟩
/-- `np.finfo(np.float64).tiny = 2^-1022`. -/
def f64Tiny : Decimal := ⟨(5 : ℤ) ^ 1022, -1022⟩

/-! ### Path A: the token route of module PARAMETER initialisers -/

/-- The declared scalar type of a name. -/
def declTy (Γ : F.Env) (x : String) : Option FTy := (Γ x).map (·.ty)

/-- The classified kind of an inquiry argument (`fortran/constants.py:418-` `_intrinsic_call`
with `kinds.literal` / `kinds.dtypes`; `fortran/expr.py:129-135` `literal_kind`): a
numeric literal without `_kind` or `d` is `float32` -- **an integer literal
included** (FNP-D0026) -- a `d` literal or `_r8` is `float64`, a name is its
declared real kind; anything else refuses the initialiser. -/
def inquiryDtype (Γ : F.Env) : F.Expr → Option RKind
  | .intLit _ _ => some .r4
  | .realLit k _ => some k
  | .var x => match declTy Γ x with
    | some (FTy.real k) => some k
    | _ => none
  | _ => none

/-- `INQUIRY_SPELLING` (`transform/numpy/constants.py:262-268`) with the `float32`
substitution of `_call` (`:339-342`): `digits` is the text `53` for every kind. -/
def inquiry (f : F.Intr) (k : RKind) : Option P.Expr :=
  match f, k with
  | .huge, .r4 => some (.floatLit f32Max)
  | .huge, .r8 => some (.floatLit f64Max)
  | .epsilon, .r4 => some (.floatLit f32Eps)
  | .epsilon, .r8 => some (.floatLit f64Eps)
  | .tiny, .r4 => some (.floatLit f32Tiny)
  | .tiny, .r8 => some (.floatLit f64Tiny)
  | .digits, _ => some (.intLit 53)
  | _, _ => none

/-- `INTRINSIC_SPELLING` (`transform/numpy/constants.py:236-260`) over already
re-spelled arguments (`_call`, `:336-348`). -/
def tokCall (f : F.Intr) (args' : P.Exprs) : Option P.Expr :=
  match f, args' with
  | .nint, .cons a .nil => some (fold1 .np_round a)          -- np.rint: half to even, a float
  | .mod, .cons a (.cons b .nil) => some (.bin .mod a b)     -- np.mod: floor remainder
  | .modulo, .cons a (.cons b .nil) => some (.bin .mod a b)
  | .int, .cons a .nil => some (fold1 .py_int a)
  | .real, .cons a .nil => some (fold1 .np_float64 a)
  | .dble, .cons a .nil => some (fold1 .np_float64 a)
  | .abs, .cons a .nil => some (fold1 .np_abs a)
  | .sqrt, .cons a .nil => some (fold1 .np_sqrt a)
  | .max, .cons a (.cons b .nil) => some (call2 .py_max a b)
  | .min, .cons a (.cons b .nil) => some (call2 .py_min a b)
  | _, _ => none

mutual
  /-- `_expression` (`transform/numpy/constants.py:271-298`): token-wise re-spelling.
  Refs stay names, a double literal is `np.float64('<text>')`, a default-kind
  literal `np.float64(np.float32('<text>'))`, integer literals and operators pass
  through, calls go through `tokCall`/`inquiry`. -/
  def tokExpr (Γ : F.Env) : F.Expr → Option P.Expr
    | .intLit _ v => some (.intLit v)
    | .realLit .r8 d => some (.floatLit d)
    | .realLit .r4 d => some (fold1 .np_float64 (fold1 .np_float32 (.floatLit d)))
    | .var x => some (.name x)
    | .paren e => (tokExpr Γ e).map .paren
    | .neg e => (tokExpr Γ e).map .neg
    | .bin op a b => do
        let a' ← tokExpr Γ a
        let b' ← tokExpr Γ b
        pure (.bin (Rules.toP op) a' b')
    | .call f args _ =>
        match f, args with
        | .huge, .cons a .nil => (inquiryDtype Γ a).bind (inquiry .huge)
        | .epsilon, .cons a .nil => (inquiryDtype Γ a).bind (inquiry .epsilon)
        | .tiny, .cons a .nil => (inquiryDtype Γ a).bind (inquiry .tiny)
        | .digits, .cons a .nil => (inquiryDtype Γ a).bind (inquiry .digits)
        | _, _ => (tokArgs Γ args).bind (tokCall f)
    | _ => none
  def tokArgs (Γ : F.Env) : F.Exprs → Option P.Exprs
    | .nil => some .nil
    | .cons e rest => do
        let e' ← tokExpr Γ e
        let rest' ← tokArgs Γ rest
        pure (.cons e' rest')
end

/-- Whether an expression contains a `/` operator token (`_integer_quotient`,
`transform/numpy/constants.py:301-320`: `any(t["t"] == "op" and t["v"] == "/")`). -/
def hasDiv : F.Expr → Bool
  | .bin .div _ _ => true
  | .bin _ a b => hasDiv a || hasDiv b
  | .paren e | .neg e => hasDiv e
  | _ => false

/-- A "plain" integer operand of a whole-initialiser quotient: an integer literal
or a named constant (`_integer_quotient`: token kind `int`/`ref` with dtype `int`). -/
def plainInt : F.Expr → Bool
  | .intLit _ _ => true
  | .var _ => true
  | _ => false

/-- `_module_parameter` for `kind == "expr"` (`transform/numpy/constants.py:181-187`):
an INTEGER parameter whose initialiser contains a `/` is spelled `int(A / B)` when it
is exactly one quotient of plain operands and *skipped* otherwise; every other
initialiser takes the token route. -/
def transParam (Γ : F.Env) (declInt : Bool) (e : F.Expr) : Option P.Expr :=
  if declInt && hasDiv e then
    match e with
    | .bin .div a b =>
      if plainInt a && plainInt b then
        (tokExpr Γ a).bind fun a' => (tokExpr Γ b).map fun b' => fold1 .py_int (.bin .truediv a' b')
      else none
    | _ => none
  else tokExpr Γ e

/-! ### Path B: the use-constants module (`Expr` trees) -/

/-- `typed` (`fortran/expr.py:488-516`): `some true` = int, `some false` = real,
`none` = unknown.  A real leaf anywhere makes the expression real; all-int makes
it int; `int(...)` is int, `real`/`dble`/`float`/`sqrt` and the kind inquiries real. -/
def promote2 : Option Bool → Option Bool → Option Bool
  | some false, _ => some false
  | _, some false => some false
  | some true, some true => some true
  | _, _ => none

mutual
  def useTyped (Γ : F.Env) : F.Expr → Option Bool
    | .intLit _ _ => some true
    | .realLit _ _ => some false
    | .var x => match declTy Γ x with
      | some (FTy.int _) => some true
      | some (FTy.real _) => some false
      | _ => none
    | .paren e => useTyped Γ e
    | .neg e => useTyped Γ e
    | .bin _ a b => promote2 (useTyped Γ a) (useTyped Γ b)
    | .call f args _ =>
      match f with
      | .int => some true
      | .real | .dble | .sqrt | .epsilon | .huge | .tiny => some false
      | _ => useTypedArgs Γ args
    | _ => none
  def useTypedArgs (Γ : F.Env) : F.Exprs → Option Bool
    | .nil => none
    | .cons e .nil => useTyped Γ e
    | .cons e rest => promote2 (useTyped Γ e) (useTypedArgs Γ rest)
end

/-- `with_integer_division` (`fortran/expr.py:517-541`): the quotient is spelled `//`
when both operands are typed int, or when neither is typed real and the whole
initialiser is not typed real (`default_integer`). -/
def intQuot : Option Bool → Option Bool → Bool → Bool
  | some true, some true, _ => true
  | some false, _, _ => false
  | _, some false, _ => false
  | _, _, defaultInt => defaultInt

/-- `python_call` (`fortran/expr.py:553-588`) over rendered arguments; kind inquiries
(`np.finfo(<arg>).<attr>`) are outside the modelled fragment. -/
def useCall (f : F.Intr) (args' : P.Exprs) : Option P.Expr :=
  match f, args' with
  | .max, .cons a (.cons b .nil) => some (call2 .py_max a b)
  | .min, .cons a (.cons b .nil) => some (call2 .py_min a b)
  | .abs, .cons a .nil => some (fold1 .py_abs a)
  | .int, .cons a .nil => some (fold1 .py_int a)
  | .sqrt, .cons a .nil => some (fold1 .np_sqrt a)
  | .real, .cons a .nil => some (fold1 .np_float32 a)     -- `real(x)`: the default real
  | .dble, .cons a .nil => some (fold1 .np_float64 a)
  | _, _ => none

/-- The binary node of the use-constants renderer: `//` for an integer quotient
(`with_integer_division`), the operator otherwise; always parenthesised. -/
def useBinNode (Γ : F.Env) (defaultInt : Bool) (op : F.BinOp) (a b : F.Expr) (a' b' : P.Expr) : P.Expr :=
  match op with
  | .div => .paren (.bin (if intQuot (useTyped Γ a) (useTyped Γ b) defaultInt then .floordiv else .truediv) a' b')
  | _ => .paren (.bin (Rules.toP op) a' b')

mutual
  /-- `render` with `_python`'s atom spellings (`transform/numpy/constants.py:411-429`,
  `fortran/expr.py:328-404`): every compound node is parenthesised. -/
  def useConst (Γ : F.Env) (defaultInt : Bool) : F.Expr → Option P.Expr
    | .intLit _ v => some (.intLit v)
    | .realLit .r8 d => some (.floatLit d)
    | .realLit .r4 d => some (fold1 .np_float32 (.floatLit d))
    | .var x => some (.name x)
    | .paren e => (useConst Γ defaultInt e).map .paren
    | .neg e => (useConst Γ defaultInt e).map fun e' => .paren (.neg e')
    | .bin op a b => do
        let a' ← useConst Γ defaultInt a
        let b' ← useConst Γ defaultInt b
        pure (useBinNode Γ defaultInt op a b a' b')
    | .call f args _ => (useConstArgs Γ defaultInt args).bind (useCall f)
    | _ => none
  def useConstArgs (Γ : F.Env) (defaultInt : Bool) : F.Exprs → Option P.Exprs
    | .nil => some .nil
    | .cons e rest => do
        let e' ← useConst Γ defaultInt e
        let rest' ← useConstArgs Γ defaultInt rest
        pure (.cons e' rest')
end

/-- One use-imported constant (`use_constants_module`, `transform/numpy/constants.py:378-409`):
`default_integer` is "the whole initialiser is not typed real"; an INTEGER constant
initialised by a real expression is wrapped in `int(...)`. -/
def transUseConst (Γ : F.Env) (declInt : Bool) (e : F.Expr) : Option P.Expr :=
  (useConst Γ (useTyped Γ e != some false) e).map fun v =>
    if declInt && useTyped Γ e == some false then fold1 .py_int v else v

/-! ### Path C: module variable initialisers (`_state_value`) -/

/-- `_state_value` (`transform/numpy/modules.py:479-554`): the recognised textual forms.
`huge(1)` is the default-integer constant, `huge(1.0…)` the float64 maximum,
`epsilon(<name>)` the float64 epsilon, and `a/b` over two numeric literals is
`np.float64(float(a)/float(b))` (FNP-D0029).  Anything else is refused
(`None  # TODO`). -/
def isOneLit : Decimal → Bool
  | ⟨1, 0⟩ => true
  | ⟨10, -1⟩ => true
  | _ => false

def numLit : F.Expr → Option P.Expr
  | .intLit _ v => some (.intLit v)
  | .realLit _ d => some (.floatLit d)
  | _ => none

def stateInit (Γ : F.Env) : F.Expr → Option P.Expr
  | .var x => if (Γ x).isSome then some (.name x) else none
  | .intLit _ v => some (.intLit v)
  | .boolLit b => some (.boolLit b)
  | .realLit _ d => some (.floatLit d)
  | .strLit s => some (.strLit s)
  | .call .huge (.cons (.intLit _ 1) .nil) _ => some (.intLit 2147483647)
  | .call .huge (.cons (.realLit _ d) .nil) _ => if isOneLit d then some (.floatLit f64Max) else none
  | .neg (.call .huge (.cons (.intLit _ 1) .nil) _) => some (.intLit (-2147483647))
  | .neg (.call .huge (.cons (.realLit _ d) .nil) _) => if isOneLit d then some (.neg (.floatLit f64Max)) else none
  | .call .epsilon (.cons (.var _) .nil) _ => some (.floatLit f64Eps)
  | .bin .div a b => (numLit a).bind fun a' => (numLit b).map fun b' =>
      fold1 .np_float64 (.bin .truediv (fold1 .np_float64 a') (fold1 .np_float64 b'))
  | _ => none

end Recast.Tr

namespace Recast.Rules

open Recast Rel Exact

/-! ### Small evaluation facts -/

theorem P_eval_neg (o : P.Oracle) (τ : P.State) (e : P.Expr) :
    P.eval o τ (.neg e) = (P.eval o τ e).bind (P.lift1 fun v => match v with
      | .int a => some (.int (-a))
      | .float a => some (.float (-a))
      | _ => none) := rfl

theorem F_eval_neg (o : F.Oracle) (σ : F.State) (e : F.Expr) :
    F.eval o σ (.neg e) = (F.eval o σ e).bind (F.lift1 fun v => match v with
      | .int k a => some (.int k (-a))
      | .real k a => some (.real k (-a))
      | _ => none) := rfl

/-- Parentheses on the Python side are semantically transparent. -/
theorem soundScalar_unparen {o : Oracles} {Γ : F.Env} {e : F.Expr} {e' : P.Expr}
    (h : SoundScalar o Γ e (.paren e')) : SoundScalar o Γ e e' := h

theorem soundScalar_paren {o : Oracles} {Γ : F.Env} {e : F.Expr} {e' : P.Expr}
    (h : SoundScalar o Γ e e') : SoundScalar o Γ e (.paren e') := h

theorem soundScalar_fparen {o : Oracles} {Γ : F.Env} {e : F.Expr} {e' : P.Expr}
    (h : SoundScalar o Γ e e') : SoundScalar o Γ (.paren e) e' := by
  intro σ τ hR hC v hv
  exact h σ τ hR hC v (by simpa [F.eval] using hv)

/-- Unary minus, both sides. -/
theorem neg_sound (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (h : SoundScalar o Γ e e') :
    SoundScalar o Γ (.neg e) (.neg e') := by
  intro σ τ hR hC v hv
  rw [F_eval_neg] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨u, hu, hlift⟩ := hv
  obtain ⟨x, rfl, hfx⟩ := F_lift1_eq_scalar hlift
  obtain ⟨w, hw, hrw⟩ := h σ τ hR hC x hu
  rw [P_eval_neg, hw, Option.bind_some, P_lift1_scalar]
  cases x with
  | int k a =>
    rw [RelS.int_eq hrw]
    simp only [Option.some.injEq] at hfx; subst hfx
    exact ⟨.int (-a), rfl, RelS.int⟩
  | real k r =>
    simp only [Option.some.injEq] at hfx; subst hfx
    rcases RelS.real_cases hrw with ⟨n, rfl, rfl⟩ | rfl
    · exact ⟨.int (-n), rfl, RelS.real (by simp [P.toR])⟩
    · exact ⟨.float (-r), rfl, RelS.realF⟩
  | bool _ => simp at hfx
  | str _ => simp at hfx

/-- The `np.float32` / `np.float64` casts are identities on numbers. -/
theorem helper_np_float64_toR (o : P.Oracle) (w : PVal) (r : ℝ) (h : P.toR w = some r) :
    P.helper o .np_float64 [.s w] = some (.s (.float r)) := by
  rcases P_toR_cases h with ⟨n, rfl, rfl⟩ | rfl <;> rfl

theorem helper_np_float32_toR (o : P.Oracle) (w : PVal) (r : ℝ) (h : P.toR w = some r) :
    P.helper o .np_float32 [.s w] = some (.s (.float r)) := by
  rcases P_toR_cases h with ⟨n, rfl, rfl⟩ | rfl <;> rfl

theorem cast_sound (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (h : SoundScalar o Γ e e')
    (hreal : RealTyped o Γ e) (c : P.Helper) (hc : c = .np_float64 ∨ c = .np_float32) :
    SoundScalar o Γ e (Tr.fold1 c e') := by
  intro σ τ hR hC v hv
  obtain ⟨k, r, rfl⟩ := hreal σ τ hR hC v hv
  obtain ⟨w, hw, hrw⟩ := h σ τ hR hC _ hv
  have ht := RelS.real_toR hrw
  refine ⟨.float r, ?_, RelS.realF⟩
  rw [P_eval_call1, hw, Option.bind_some]
  rcases hc with rfl | rfl
  · exact helper_np_float64_toR o.P w r ht
  · exact helper_np_float32_toR o.P w r ht

theorem realLit_r4_sound (o : Oracles) (Γ : F.Env) (d : Decimal) :
    SoundScalar o Γ (.realLit .r4 d) (Tr.fold1 .np_float64 (Tr.fold1 .np_float32 (.floatLit d))) := by
  have h0 := realLit_sound' o Γ .r4 d
  have hreal : RealTyped o Γ (.realLit .r4 d) := by
    intro σ τ _ _ v hv; simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv
    exact ⟨_, _, rfl⟩
  exact cast_sound o Γ _ _ (cast_sound o Γ _ _ h0 hreal _ (Or.inr rfl)) hreal _ (Or.inl rfl)

/-- `int(a / b)` over two integers is the truncating quotient (the body of `_f_int_div`). -/
theorem pyint_truediv_sound (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b) :
    SoundScalar o Γ (.bin .div a b) (Tr.fold1 .py_int (.bin .truediv a' b')) := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨kx, nx, rfl⟩ := hia σ τ hR hC x hva
  obtain ⟨ky, ny, rfl⟩ := hib σ τ hR hC y hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
  rw [RelS.int_eq hrx] at hx'
  rw [RelS.int_eq hry] at hy'
  simp only [F.arith] at hxy
  split at hxy
  · simp at hxy
  · rename_i hny
    simp only [Option.some.injEq] at hxy
    subst hxy
    refine ⟨.int (tquot nx ny), ?_, RelS.int⟩
    rw [P_eval_call1, P_eval_bin, hx', hy']
    simp only [Option.bind_some, P_lift2_scalar]
    have hdiv : P.arith .truediv (.int nx) (.int ny) = some (.float ((nx : ℝ) / (ny : ℝ))) := by
      simp [P.arith, hny]
    rw [hdiv]
    simp only [Option.map_some, Option.bind_some]
    rw [show P.helper o.P .py_int [.s (.float ((nx : ℝ) / (ny : ℝ)))] = some (.s (.int (truncR ((nx : ℝ) / (ny : ℝ))))) from rfl,
      truncR_div_intCast nx ny hny]

/-- Python `//` on two ints is the truncating quotient when the operands are non-negative. -/
theorem floordiv_nonneg_sound (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b)
    (hnn : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k₁ x k₂ y, F.eval o.F σ a = some (.s (.int k₁ x)) →
      F.eval o.F σ b = some (.s (.int k₂ y)) → 0 ≤ x ∧ 0 < y) :
    SoundScalar o Γ (.bin .div a b) (.bin .floordiv a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨kx, nx, rfl⟩ := hia σ τ hR hC x hva
  obtain ⟨ky, ny, rfl⟩ := hib σ τ hR hC y hvb
  obtain ⟨hx0, hy0⟩ := hnn σ τ hR hC kx nx ky ny hva hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
  rw [RelS.int_eq hrx] at hx'
  rw [RelS.int_eq hry] at hy'
  simp only [F.arith] at hxy
  split at hxy
  · omega
  · rename_i hny
    simp only [Option.some.injEq] at hxy
    subst hxy
    refine ⟨.int (fquot nx ny), ?_, ?_⟩
    · rw [P_eval_bin, hx', hy']; simp [P.lift2, P.arith, hny]
    · rw [tquot_eq_fquot_of_nonneg hx0 hy0]; exact RelS.int

/-! ### The closed environment used by every counterexample -/

/-- `x = 0` with `real(8) :: x`: not needed here; all counterexamples are closed terms. -/
theorem F_eval_call1_closed (f : F.Intr) (a : F.Expr) (va : F.Val) (h : F.eval noPrim.F emptyF a = some va) :
    F.eval noPrim.F emptyF (.call f (.cons a .nil) none) = F.intrinsic noPrim.F f [va] none := by
  rw [F_eval_call1, h]; rfl

/-! ### FNP-R0171: the token route -/

/-- The rule as the engine applies it: any classified initialiser of a REAL or
INTEGER parameter is re-spelled by `transParam` and denotes the same value. -/
def FNP_R0171_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (declInt : Bool) (e : F.Expr) (e' : P.Expr),
    Tr.transParam Γ declInt e = some e' → SoundScalar o Γ e e'

/-- FNP-D0023: `real(r8), parameter :: x = 1.0_r8 * (7/2)` — the nested integer
quotient becomes Python's real `/`: Fortran `3.0`, Python `3.5`. -/
theorem FNP_R0171_refuted_quot : ¬ FNP_R0171_obligation := by
  intro h
  have hval : Decimal.toReal ⟨10, -1⟩ = (1 : ℝ) := by norm_num [Decimal.toReal]
  have h72 : F.eval noPrim.F emptyF (.paren (.bin .div (.intLit .i4 7) (.intLit .i4 2))) = some (.s (.int .i4 3)) := by
    show F.eval noPrim.F emptyF (.bin .div (.intLit .i4 7) (.intLit .i4 2)) = _
    rw [F_eval_bin, F_eval_intLit, F_eval_intLit]
    simp [F.lift2, F.arith, tquot, IKind.join]
  have hlit : F.eval noPrim.F emptyF (.realLit .r8 ⟨10, -1⟩) = some (.s (.real .r8 (Decimal.toReal ⟨10, -1⟩))) := rfl
  have := h noPrim emptyEnv false (.bin .mul (.realLit .r8 ⟨10, -1⟩) (.paren (.bin .div (.intLit .i4 7) (.intLit .i4 2))))
    _ rfl emptyF emptyP relState_empty conforms_empty (.real .r8 (Decimal.toReal ⟨10, -1⟩ * ((3 : ℤ) : ℝ))) (by
      rw [F_eval_bin, hlit, h72]
      simp [F.lift2, F.arith])
  obtain ⟨w, hw, hrel⟩ := this
  have h72' : P.eval noPrim.P emptyP (.paren (.bin (toP .div) (.intLit 7) (.intLit 2))) = some (.s (.float ((7 : ℤ) / (2 : ℤ) : ℝ))) := by
    rw [P_eval_paren, P_eval_bin]
    simp [P.eval, P.lift2, P.arith, toP]
  rw [P_eval_bin, h72'] at hw
  simp only [P.eval, Option.bind_some, P_lift2_scalar] at hw
  simp only [toP, P.arith, P.toR, Option.map_some, Option.some.injEq, P.Val.s.injEq] at hw
  subst hw
  have ht := RelS.real_toR hrel
  simp only [P.toR, Option.some.injEq, hval] at ht
  push_cast at ht
  norm_num at ht

/-- FNP-D0024: `integer, parameter :: n = 2 ** (-1)`: Fortran `0`, Python `0.5`. -/
theorem FNP_R0171_refuted_negpow : ¬ FNP_R0171_obligation := by
  intro h
  have := h noPrim emptyEnv true (.bin .pow (.intLit .i4 2) (.paren (.neg (.intLit .i4 1)))) _ rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 0) (by
      rw [F_eval_bin]
      simp [F.eval, F.lift1, F.lift2, F.arith, F.ipowNeg, tquot, IKind.join])
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_bin] at hw
  simp [P.eval, P.lift1, P.lift2, P.arith, P.pyPow, toP] at hw
  subst hw
  exact RelS.int_ne_float hrel

/-- FNP-D0025 (NINT): `integer, parameter :: n = nint(2.5_r8)`: Fortran `3`,
Python `np.rint(np.float64('2.5'))` = `2.0`. -/
theorem FNP_R0171_refuted_nint : ¬ FNP_R0171_obligation := by
  intro h
  have hval : Decimal.toReal ⟨25, -1⟩ = (5/2 : ℝ) := by simp [Decimal.toReal]; norm_num
  have := h noPrim emptyEnv true (.call .nint (.cons (.realLit .r8 ⟨25, -1⟩) .nil) none) _ rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 3) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r8 ⟨25, -1⟩) = some (.s (.real .r8 (Decimal.toReal ⟨25, -1⟩))) from rfl,
        Option.bind_some]
      show some (F.Val.s (.int .i4 (nintR (Decimal.toReal ⟨25, -1⟩)))) = _
      rw [hval, nintR_five_halves])
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.floatLit ⟨25, -1⟩) = some (.s (.float (Decimal.toReal ⟨25, -1⟩))) from rfl,
    Option.bind_some,
    show P.helper noPrim.P .np_round [.s (.float (Decimal.toReal ⟨25, -1⟩))] =
      some (.s (.float (roundHalfEvenR (Decimal.toReal ⟨25, -1⟩) : ℝ))) from rfl] at hw
  cases hw
  exact RelS.int_ne_float hrel

/-- FNP-D0025 (MOD): `integer, parameter :: m = mod(-7, 3)`: Fortran `-1`,
Python `np.mod(-7, 3)` = `2`. -/
theorem FNP_R0171_refuted_mod : ¬ FNP_R0171_obligation := by
  intro h
  have := h noPrim emptyEnv true (.call .mod (.cons (.neg (.intLit .i4 7)) (.cons (.intLit .i4 3) .nil)) none) _ rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 (-1)) (by
      have hneg : F.eval noPrim.F emptyF (.neg (.intLit .i4 7)) = some (.s (.int .i4 (-7))) := by
        simp [F.eval, F.lift1]
      rw [F_eval_call2, hneg, F_eval_intLit]
      simp [F.intrinsic, F.intrinsicCore, F.lift2, IKind.join, tmod])
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_bin] at hw
  simp [P.eval, P.lift1, P.lift2, P.arith, fmod] at hw
  subst hw
  have := RelS.int_eq hrel
  simp only [PVal.int.injEq] at this
  first | omega | (revert this; decide)

/-- FNP-D0026: `integer, parameter :: big = huge(0)`: the literal `0` classifies as
`float32`, so the emitted constant is `np.finfo(np.float32).max`: Fortran
`2147483647`, Python a float. -/
theorem FNP_R0171_refuted_huge : ¬ FNP_R0171_obligation := by
  intro h
  have := h noPrim emptyEnv true (.call .huge (.cons (.intLit .i4 0) .nil) none) _ rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 (IKind.huge .i4)) (by
      rw [F_eval_call1, F_eval_intLit]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  simp only [P.eval, Option.some.injEq, P.Val.s.injEq] at hw
  subst hw
  exact RelS.int_ne_float hrel

/-- FNP-D0027 family: `digits(1.0_r8)` is `53` on both sides, but `digits(0)` is
the same text `53` where Fortran answers `31`. -/
theorem FNP_R0171_refuted_digits : ¬ FNP_R0171_obligation := by
  intro h
  have := h noPrim emptyEnv true (.call .digits (.cons (.intLit .i4 0) .nil) none) _ rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 (IKind.digits .i4)) (by
      rw [F_eval_call1, F_eval_intLit]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  simp only [P.eval, Option.some.injEq, P.Val.s.injEq] at hw
  subst hw
  have := RelS.int_eq hrel
  simp only [PVal.int.injEq, IKind.digits, IKind.bits] at this
  omega

/-! #### Sound sub-cases of the token route -/

/-- The operator fragment on which the token route is sound: literals, scalar
constants, parentheses, unary minus, `+ - *` over numbers, `/` with a REAL
operand, `**` with a REAL base or an INTEGER base and a non-negative literal
exponent.  Excluded (and refuted above): integer quotients, negative integer
exponents, the intrinsic spellings. -/
def TokWF (Γ : F.Env) : F.Expr → Prop
  | .intLit _ _ => True
  | .realLit _ _ => True
  | .var x => ∃ d, Γ x = some d ∧ d.lbs = []
  | .paren e => TokWF Γ e
  | .neg e => TokWF Γ e
  | .bin .div a b => TokWF Γ a ∧ TokWF Γ b ∧ numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b) ∧
      (isRealTy (Tr.tyOf Γ a) ∨ isRealTy (Tr.tyOf Γ b))
  | .bin .pow a b => TokWF Γ a ∧ TokWF Γ b ∧
      ((isRealTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)) ∨
       (isIntTy (Tr.tyOf Γ a) ∧ ∃ k n, b = .intLit k n ∧ 0 ≤ n))
  | .bin _ a b => TokWF Γ a ∧ TokWF Γ b ∧ numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)
  | _ => False

theorem tokWF_wfe (Γ : F.Env) : ∀ e, TokWF Γ e → WFE Γ e
  | .intLit _ _, _ => trivial
  | .realLit _ _, _ => trivial
  | .var _, h => h
  | .paren e, h => tokWF_wfe Γ e h
  | .neg e, h => tokWF_wfe Γ e h
  | .bin .div a b, ⟨ha, hb, hna, hnb, _⟩ => ⟨tokWF_wfe Γ a ha, tokWF_wfe Γ b hb, hna, hnb⟩
  | .bin .pow a b, ⟨ha, hb, hc⟩ => ⟨tokWF_wfe Γ a ha, tokWF_wfe Γ b hb, hc⟩
  | .bin .add a b, ⟨ha, hb, hna, hnb⟩ => ⟨tokWF_wfe Γ a ha, tokWF_wfe Γ b hb, hna, hnb⟩
  | .bin .sub a b, ⟨ha, hb, hna, hnb⟩ => ⟨tokWF_wfe Γ a ha, tokWF_wfe Γ b hb, hna, hnb⟩
  | .bin .mul a b, ⟨ha, hb, hna, hnb⟩ => ⟨tokWF_wfe Γ a ha, tokWF_wfe Γ b hb, hna, hnb⟩

theorem realTyped_of_isRealTy (o : Oracles) (Γ : F.Env) (e : F.Expr) (hwf : WFE Γ e)
    (h : isRealTy (Tr.tyOf Γ e)) : RealTyped o Γ e := by
  match ht : Tr.tyOf Γ e, h with
  | some (.real k), _ => exact realTyped_of_tyOf o Γ e hwf k ht

theorem intTyped_of_isIntTy (o : Oracles) (Γ : F.Env) (e : F.Expr) (hwf : WFE Γ e)
    (h : isIntTy (Tr.tyOf Γ e)) : IntTyped o Γ e := by
  match ht : Tr.tyOf Γ e, h with
  | some (.int k), _ => exact intTyped_of_tyOf o Γ e hwf k ht

/-- The token route is sound on the operator fragment. -/
theorem tokExpr_sound (o : Oracles) (Γ : F.Env) :
    ∀ e, TokWF Γ e → ∀ e', Tr.tokExpr Γ e = some e' → SoundScalar o Γ e e'
  | .intLit k v, _, e', h => by
      simp only [Tr.tokExpr, Option.some.injEq] at h; subst h
      exact FNP_R0001_sound o Γ k v
  | .realLit k d, _, e', h => by
      cases k with
      | r8 => simp only [Tr.tokExpr, Option.some.injEq] at h; subst h; exact realLit_sound' o Γ .r8 d
      | r4 => simp only [Tr.tokExpr, Option.some.injEq] at h; subst h; exact realLit_r4_sound o Γ d
  | .var x, ⟨d, hd, _⟩, e', h => by
      simp only [Tr.tokExpr, Option.some.injEq] at h; subst h
      exact (FNP_R0010_sound o Γ x d hd).toScalar
  | .paren e, hwf, e', h => by
      simp only [Tr.tokExpr, Option.map_eq_some_iff] at h
      obtain ⟨e₁, he₁, rfl⟩ := h
      exact soundScalar_paren (soundScalar_fparen (tokExpr_sound o Γ e hwf e₁ he₁))
  | .neg e, hwf, e', h => by
      simp only [Tr.tokExpr, Option.map_eq_some_iff] at h
      obtain ⟨e₁, he₁, rfl⟩ := h
      exact neg_sound o Γ e e₁ (tokExpr_sound o Γ e hwf e₁ he₁)
  | .bin op a b, hwf, e', h => by
      have hbin : ∃ a' b', Tr.tokExpr Γ a = some a' ∧ Tr.tokExpr Γ b = some b' ∧ e' = .bin (toP op) a' b' := by
        cases ha : Tr.tokExpr Γ a <;> cases hb : Tr.tokExpr Γ b <;> simp only [Tr.tokExpr, ha, hb] at h <;>
          simp at h
        exact ⟨_, _, rfl, rfl, h.symm⟩
      obtain ⟨a', b', ha', hb', rfl⟩ := hbin
      cases op with
      | add =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact soundScalar_unparen (FNP_R0016_sound o Γ .add a b a' b' (Or.inl rfl)
          (tokExpr_sound o Γ a hwa a' ha') (tokExpr_sound o Γ b hwb b' hb'))
      | sub =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact soundScalar_unparen (FNP_R0016_sound o Γ .sub a b a' b' (Or.inr (Or.inl rfl))
          (tokExpr_sound o Γ a hwa a' ha') (tokExpr_sound o Γ b hwb b' hb'))
      | mul =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact soundScalar_unparen (FNP_R0016_sound o Γ .mul a b a' b' (Or.inr (Or.inr rfl))
          (tokExpr_sound o Γ a hwa a' ha') (tokExpr_sound o Γ b hwb b' hb'))
      | div =>
        obtain ⟨hwa, hwb, _, _, hreal⟩ := hwf
        refine soundScalar_unparen (FNP_R0018_sound o Γ a b a' b'
          (tokExpr_sound o Γ a hwa a' ha') (tokExpr_sound o Γ b hwb b' hb') ?_)
        rcases hreal with hr | hr
        · exact Or.inl (realTyped_of_isRealTy o Γ a (tokWF_wfe Γ a hwa) hr)
        · exact Or.inr (realTyped_of_isRealTy o Γ b (tokWF_wfe Γ b hwb) hr)
      | pow =>
        obtain ⟨hwa, hwb, hc⟩ := hwf
        have ha := tokExpr_sound o Γ a hwa a' ha'
        have hb := tokExpr_sound o Γ b hwb b' hb'
        rcases hc with ⟨hra, _⟩ | ⟨hia, k, n, rfl, hn⟩
        · exact soundScalar_unparen (FNP_R0023_sound o Γ a b a' b' ha hb
            (realTyped_of_isRealTy o Γ a (tokWF_wfe Γ a hwa) hra))
        · simp only [Tr.tokExpr, Option.some.injEq] at hb'
          subst hb'
          have hint := intTyped_of_isIntTy o Γ a (tokWF_wfe Γ a hwa) hia
          rcases lt_or_eq_of_le hn with hpos | hzero
          · exact soundScalar_unparen (FNP_R0020_sound_pos o Γ a a' n hpos ha hint k)
          · subst hzero
            exact soundScalar_unparen (pow_int_zero_sound o Γ a a' k ha hint)
  | .boolLit _, h, _, _ => h.elim
  | .strLit _, h, _, _ => h.elim
  | .cmp _ _ _, h, _, _ => h.elim
  | .logic _ _ _, h, _, _ => h.elim
  | .not _, h, _, _ => h.elim
  | .ref _ _, h, _, _ => h.elim
  | .call _ _ _, h, _, _ => h.elim
termination_by e => sizeOf e
decreasing_by all_goals simp_wf <;> omega

/-- The REAL-parameter branch of FNP-R0171 on the operator fragment. -/
theorem FNP_R0171_sound_real (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (hwf : TokWF Γ e)
    (h : Tr.transParam Γ false e = some e') : SoundScalar o Γ e e' :=
  tokExpr_sound o Γ e hwf e' (by simpa [Tr.transParam] using h)

/-- The INTEGER-parameter branch without a quotient is the same token route. -/
theorem FNP_R0171_sound_int_noquot (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (hwf : TokWF Γ e)
    (hnd : Tr.hasDiv e = false) (h : Tr.transParam Γ true e = some e') : SoundScalar o Γ e e' := by
  simp only [Tr.transParam, hnd, Bool.and_false, Bool.false_eq_true, ↓reduceIte] at h
  exact tokExpr_sound o Γ e hwf e' h

/-- The INTEGER-parameter whole-initialiser quotient `A / B` -> `int(A / B)`
(`_integer_quotient`): sound for every non-zero divisor. -/
theorem FNP_R0171_sound_intquot (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b) :
    SoundScalar o Γ (.bin .div a b) (Tr.fold1 .py_int (.bin .truediv a' b')) :=
  pyint_truediv_sound o Γ a b a' b' ha hb hia hib

/-- MODULO -> `np.mod` is sound (both are the floor remainder); only MOD is refuted. -/
theorem FNP_R0171_sound_modulo (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b) :
    SoundScalar o Γ (.call .modulo (.cons a (.cons b .nil)) none) (.bin .mod a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | arr av => exact absurd hint (intrinsic2_arr_ne o.F .modulo _ _ _ (Or.inl ⟨av, rfl⟩) (by simp))
  | s x =>
    cases vb with
    | arr bv => exact absurd hint (intrinsic2_arr_ne o.F .modulo _ _ _ (Or.inr ⟨bv, rfl⟩) (by simp))
    | s y =>
      obtain ⟨kx, nx, rfl⟩ := hia σ τ hR hC x hva
      obtain ⟨ky, ny, rfl⟩ := hib σ τ hR hC y hvb
      obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
      obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
      rw [RelS.int_eq hrx] at hx'
      rw [RelS.int_eq hry] at hy'
      have e : F.intrinsic o.F .modulo [.s (.int kx nx), .s (.int ky ny)] none =
          (if ny = 0 then none else some (.s (.int (IKind.join kx ky) (fmod nx ny)))) := by
        simp only [F.intrinsic, F.intrinsicCore, F.lift2]
        split_ifs <;> rfl
      rw [e] at hint
      split_ifs at hint with hny
      cases hint
      refine ⟨.int (fmod nx ny), ?_, RelS.int⟩
      rw [P_eval_bin, hx', hy']
      simp [P.lift2, P.arith, hny]

/-! ### FNP-R0174: array parameters, per element -/

/-- Each element of `(/ 0.1_r8, 0.2_r8 /)` -> `np.array([np.float64('0.1'), …])`, and of
the `[0.1, 0.2]` route (bare Python literals, exact decimals in the model), denotes
its decimal.  The array-level statement needs an array-constructor node in the
source syntax (not in this fragment). -/
def FNP_R0174_elem_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (k : RKind) (d : Decimal), SoundScalar o Γ (.realLit k d) (.floatLit d)

theorem FNP_R0174_elem_sound : FNP_R0174_elem_obligation := fun o Γ k d => realLit_sound' o Γ k d

/-! ### FNP-R0175: the use-constants module -/

def FNP_R0175_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (declInt : Bool) (e : F.Expr) (e' : P.Expr),
    Tr.transUseConst Γ declInt e = some e' → SoundScalar o Γ e e'

/-- FNP-D0013: `integer, parameter :: m = -7/2` is rendered `((-7) // 2)`:
Fortran `-3` (truncation), Python `-4` (floor). -/
theorem FNP_R0175_refuted : ¬ FNP_R0175_obligation := by
  intro h
  have := h noPrim emptyEnv true (.bin .div (.neg (.intLit .i4 7)) (.intLit .i4 2))
    (.paren (.bin .floordiv (.paren (.neg (.intLit 7))) (.intLit 2))) rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 (tquot (-7) 2)) (by
      have hneg : F.eval noPrim.F emptyF (.neg (.intLit .i4 7)) = some (.s (.int .i4 (-7))) := by
        simp [F.eval, F.lift1]
      rw [F_eval_bin, hneg, F_eval_intLit]
      simp [F.lift2, F.arith, IKind.join])
  obtain ⟨w, hw, hrel⟩ := this
  have hneg' : P.eval noPrim.P emptyP (.paren (.neg (.intLit 7))) = some (.s (.int (-7))) := by
    simp [P.eval, P.lift1]
  rw [P_eval_paren, P_eval_bin, hneg', P_eval_intLit] at hw
  simp [P.lift2, P.arith] at hw
  subst hw
  have := RelS.int_eq hrel
  simp only [PVal.int.injEq] at this
  -- `-7 // 2 = -4` in Python, `-7 / 2 = -3` in Fortran (FNP-D0013)
  revert this; simp only [tquot, fquot]; decide

/-- The operator fragment of the use-constants route without integer quotients:
literals, constants, parentheses, unary minus, `+ - *`, `/` with a REAL operand
(typed real by `useTyped`), `**` as for the token route. -/
def UseWF (Γ : F.Env) : F.Expr → Prop
  | .intLit _ _ => True
  | .realLit _ _ => True
  | .var x => ∃ d, Γ x = some d ∧ d.lbs = []
  | .paren e => UseWF Γ e
  | .neg e => UseWF Γ e
  | .bin .div a b => UseWF Γ a ∧ UseWF Γ b ∧ numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b) ∧
      (isRealTy (Tr.tyOf Γ a) ∨ isRealTy (Tr.tyOf Γ b)) ∧
      (Tr.useTyped Γ a = some false ∨ Tr.useTyped Γ b = some false)
  | .bin .pow a b => UseWF Γ a ∧ UseWF Γ b ∧
      ((isRealTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)) ∨
       (isIntTy (Tr.tyOf Γ a) ∧ ∃ k n, b = .intLit k n ∧ 0 ≤ n))
  | .bin _ a b => UseWF Γ a ∧ UseWF Γ b ∧ numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)
  | _ => False

theorem useWF_wfe (Γ : F.Env) : ∀ e, UseWF Γ e → WFE Γ e
  | .intLit _ _, _ => trivial
  | .realLit _ _, _ => trivial
  | .var _, h => h
  | .paren e, h => useWF_wfe Γ e h
  | .neg e, h => useWF_wfe Γ e h
  | .bin .div a b, ⟨ha, hb, hna, hnb, _, _⟩ => ⟨useWF_wfe Γ a ha, useWF_wfe Γ b hb, hna, hnb⟩
  | .bin .pow a b, ⟨ha, hb, hc⟩ => ⟨useWF_wfe Γ a ha, useWF_wfe Γ b hb, hc⟩
  | .bin .add a b, ⟨ha, hb, hna, hnb⟩ => ⟨useWF_wfe Γ a ha, useWF_wfe Γ b hb, hna, hnb⟩
  | .bin .sub a b, ⟨ha, hb, hna, hnb⟩ => ⟨useWF_wfe Γ a ha, useWF_wfe Γ b hb, hna, hnb⟩
  | .bin .mul a b, ⟨ha, hb, hna, hnb⟩ => ⟨useWF_wfe Γ a ha, useWF_wfe Γ b hb, hna, hnb⟩

theorem intQuot_false_of_real {ta tb : Option Bool} (d : Bool) (h : ta = some false ∨ tb = some false) :
    Tr.intQuot ta tb d = false := by
  rcases h with rfl | rfl
  · rfl
  · rcases ta with _ | (_ | _) <;> rfl

theorem realLit_r4_use_sound (o : Oracles) (Γ : F.Env) (d : Decimal) :
    SoundScalar o Γ (.realLit .r4 d) (Tr.fold1 .np_float32 (.floatLit d)) := by
  have hreal : RealTyped o Γ (.realLit .r4 d) := by
    intro σ τ _ _ v hv; simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv
    exact ⟨_, _, rfl⟩
  exact cast_sound o Γ _ _ (realLit_sound' o Γ .r4 d) hreal _ (Or.inr rfl)

/-- The use-constants route is sound on its operator fragment (for either
value of `default_integer`). -/
theorem useConst_sound (o : Oracles) (Γ : F.Env) (dI : Bool) :
    ∀ e, UseWF Γ e → ∀ e', Tr.useConst Γ dI e = some e' → SoundScalar o Γ e e'
  | .intLit k v, _, e', h => by
      simp only [Tr.useConst, Option.some.injEq] at h; subst h
      exact FNP_R0001_sound o Γ k v
  | .realLit k d, _, e', h => by
      cases k with
      | r8 => simp only [Tr.useConst, Option.some.injEq] at h; subst h; exact realLit_sound' o Γ .r8 d
      | r4 => simp only [Tr.useConst, Option.some.injEq] at h; subst h; exact realLit_r4_use_sound o Γ d
  | .var x, ⟨d, hd, _⟩, e', h => by
      simp only [Tr.useConst, Option.some.injEq] at h; subst h
      exact (FNP_R0010_sound o Γ x d hd).toScalar
  | .paren e, hwf, e', h => by
      simp only [Tr.useConst, Option.map_eq_some_iff] at h
      obtain ⟨e₁, he₁, rfl⟩ := h
      exact soundScalar_paren (soundScalar_fparen (useConst_sound o Γ dI e hwf e₁ he₁))
  | .neg e, hwf, e', h => by
      simp only [Tr.useConst, Option.map_eq_some_iff] at h
      obtain ⟨e₁, he₁, rfl⟩ := h
      exact soundScalar_paren (neg_sound o Γ e e₁ (useConst_sound o Γ dI e hwf e₁ he₁))
  | .bin op a b, hwf, e', h => by
      have hbin : ∃ a' b', Tr.useConst Γ dI a = some a' ∧ Tr.useConst Γ dI b = some b' ∧
          e' = Tr.useBinNode Γ dI op a b a' b' := by
        cases ha : Tr.useConst Γ dI a <;> cases hb : Tr.useConst Γ dI b <;> simp only [Tr.useConst, ha, hb] at h <;>
          simp at h
        exact ⟨_, _, rfl, rfl, h.symm⟩
      obtain ⟨a', b', ha', hb', rfl⟩ := hbin
      cases op with
      | add =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact FNP_R0016_sound o Γ .add a b a' b' (Or.inl rfl)
          (useConst_sound o Γ dI a hwa a' ha') (useConst_sound o Γ dI b hwb b' hb')
      | sub =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact FNP_R0016_sound o Γ .sub a b a' b' (Or.inr (Or.inl rfl))
          (useConst_sound o Γ dI a hwa a' ha') (useConst_sound o Γ dI b hwb b' hb')
      | mul =>
        obtain ⟨hwa, hwb, _, _⟩ := hwf
        exact FNP_R0016_sound o Γ .mul a b a' b' (Or.inr (Or.inr rfl))
          (useConst_sound o Γ dI a hwa a' ha') (useConst_sound o Γ dI b hwb b' hb')
      | div =>
        obtain ⟨hwa, hwb, _, _, hreal, htyped⟩ := hwf
        simp only [Tr.useBinNode, intQuot_false_of_real dI htyped, Bool.false_eq_true, ↓reduceIte]
        refine FNP_R0018_sound o Γ a b a' b'
          (useConst_sound o Γ dI a hwa a' ha') (useConst_sound o Γ dI b hwb b' hb') ?_
        rcases hreal with hr | hr
        · exact Or.inl (realTyped_of_isRealTy o Γ a (useWF_wfe Γ a hwa) hr)
        · exact Or.inr (realTyped_of_isRealTy o Γ b (useWF_wfe Γ b hwb) hr)
      | pow =>
        obtain ⟨hwa, hwb, hc⟩ := hwf
        have ha := useConst_sound o Γ dI a hwa a' ha'
        have hb := useConst_sound o Γ dI b hwb b' hb'
        simp only [Tr.useBinNode]
        rcases hc with ⟨hra, _⟩ | ⟨hia, k, n, rfl, hn⟩
        · exact FNP_R0023_sound o Γ a b a' b' ha hb (realTyped_of_isRealTy o Γ a (useWF_wfe Γ a hwa) hra)
        · simp only [Tr.useConst, Option.some.injEq] at hb'
          subst hb'
          have hint := intTyped_of_isIntTy o Γ a (useWF_wfe Γ a hwa) hia
          rcases lt_or_eq_of_le hn with hpos | hzero
          · exact FNP_R0020_sound_pos o Γ a a' n hpos ha hint k
          · subst hzero
            exact pow_int_zero_sound o Γ a a' k ha hint
  | .boolLit _, h, _, _ => h.elim
  | .strLit _, h, _, _ => h.elim
  | .cmp _ _ _, h, _, _ => h.elim
  | .logic _ _ _, h, _, _ => h.elim
  | .not _, h, _, _ => h.elim
  | .ref _ _, h, _, _ => h.elim
  | .call _ _ _, h, _, _ => h.elim
termination_by e => sizeOf e
decreasing_by all_goals simp_wf <;> omega

/-- A REAL-typed use-imported constant on the operator fragment (no `int(...)` wrapper). -/
theorem FNP_R0175_sound_real (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) (hwf : UseWF Γ e)
    (hty : Tr.useTyped Γ e = some false) (h : Tr.transUseConst Γ false e = some e') : SoundScalar o Γ e e' := by
  simp only [Tr.transUseConst, hty, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Option.map_eq_some_iff] at h
  obtain ⟨v, hv, rfl⟩ := h
  exact useConst_sound o Γ _ e hwf v hv

/-- An integer quotient of non-negative operands is spelled `//` and is sound there
(the operands' signs are NOT checked by the engine: FNP-D0013). -/
theorem FNP_R0175_sound_intquot_nonneg (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b)
    (hnn : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k₁ x k₂ y, F.eval o.F σ a = some (.s (.int k₁ x)) →
      F.eval o.F σ b = some (.s (.int k₂ y)) → 0 ≤ x ∧ 0 < y) :
    SoundScalar o Γ (.bin .div a b) (.paren (.bin .floordiv a' b')) :=
  soundScalar_paren (floordiv_nonneg_sound o Γ a b a' b' ha hb hia hib hnn)

/-- `integer, parameter :: i = <real expression>` -> `int(<expr>)`: the declared
conversion truncates on both sides. -/
theorem FNP_R0175_sound_int_of_real (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr)
    (h : SoundScalar o Γ e e') (hreal : RealTyped o Γ e) (d : F.Decl) (k : IKind) (hd : d.ty = .int k) :
    ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v cv, F.eval o.F σ e = some (.s v) → F.convertTo d v = some cv →
      ∃ w, P.eval o.P τ (Tr.fold1 .py_int e') = some (.s w) ∧ RelS cv w := by
  intro σ τ hR hC v cv hv hcv
  obtain ⟨kr, r, rfl⟩ := hreal σ τ hR hC v hv
  simp only [F.convertTo, hd, Option.some.injEq] at hcv
  subst hcv
  obtain ⟨w, hw, hrw⟩ := h σ τ hR hC _ hv
  have ht := RelS.real_toR hrw
  refine ⟨.int (truncR r), ?_, RelS.int⟩
  rw [P_eval_call1, hw, Option.bind_some]
  rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
  · rw [truncR_intCast]; rfl
  · rfl

/-! ### FNP-R0176: module variable initialisers -/

/-- Soundness of an initialiser under the declared conversion: the Fortran value
is converted to the variable's type (F2018 10.2.1.3) and must be related to the
Python value the emitted text denotes. -/
def SoundInit (o : Oracles) (Γ : F.Env) (d : F.Decl) (e : F.Expr) (e' : P.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v cv, F.eval o.F σ e = some (.s v) → F.convertTo d v = some cv →
    ∃ w, P.eval o.P τ e' = some (.s w) ∧ RelS cv w

def FNP_R0176_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (d : F.Decl) (e : F.Expr) (e' : P.Expr), d.lbs = [] →
    Tr.stateInit Γ e = some e' → SoundInit o Γ d e e'

/-- FNP-D0029: `real(8) :: third = 1/3` is rendered `np.float64(float(1)/float(3))`:
Fortran evaluates the integer quotient (`0`) and converts it (`0.0`); Python `1/3`. -/
theorem FNP_R0176_refuted : ¬ FNP_R0176_obligation := by
  intro h
  have := h noPrim emptyEnv ⟨.real .r8, [], 0⟩ (.bin .div (.intLit .i4 1) (.intLit .i4 3)) _ rfl rfl
    emptyF emptyP relState_empty conforms_empty (.int .i4 (tquot 1 3)) (.real .r8 ((tquot 1 3 : ℤ) : ℝ))
    (by rw [F_eval_bin]; simp [F.eval, F.lift2, F.arith, IKind.join]) (by simp [F.convertTo])
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1, P_eval_bin, P_eval_call1, P_eval_call1] at hw
  simp only [P.eval, Option.bind_some] at hw
  rw [show P.helper noPrim.P .np_float64 [.s (.int 1)] = some (.s (.float ((1 : ℤ) : ℝ))) from rfl,
    show P.helper noPrim.P .np_float64 [.s (.int 3)] = some (.s (.float ((3 : ℤ) : ℝ))) from rfl] at hw
  simp only [Option.bind_some, P_lift2_scalar] at hw
  rw [P_arith_truediv_toR (P_toR_float _) (P_toR_float _) (by norm_num)] at hw
  simp only [Option.map_some, Option.bind_some] at hw
  rw [show P.helper noPrim.P .np_float64 [.s (.float (((1 : ℤ) : ℝ) / ((3 : ℤ) : ℝ)))] =
      some (.s (.float (((1 : ℤ) : ℝ) / ((3 : ℤ) : ℝ)))) from rfl] at hw
  cases hw
  have ht := RelS.real_toR hrel
  simp only [P.toR, Option.some.injEq, tquot] at ht
  norm_num at ht

/-- Literal initialisers: an integer literal into an INTEGER or REAL variable, a real
literal into a REAL variable, a logical literal into a LOGICAL variable. -/
theorem FNP_R0176_sound_intlit (o : Oracles) (Γ : F.Env) (d : F.Decl) (k : IKind) (v : ℤ)
    (hd : (∃ k', d.ty = .int k') ∨ (∃ k', d.ty = .real k')) :
    SoundInit o Γ d (.intLit k v) (.intLit v) := by
  intro σ τ _ _ x cx hx hcx
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hx; subst hx
  rcases hd with ⟨k', hk⟩ | ⟨k', hk⟩
  · simp only [F.convertTo, hk, Option.some.injEq] at hcx; subst hcx
    exact ⟨.int v, rfl, RelS.int⟩
  · simp only [F.convertTo, hk, Option.some.injEq] at hcx; subst hcx
    exact ⟨.int v, rfl, RelS.realI⟩

theorem FNP_R0176_sound_reallit (o : Oracles) (Γ : F.Env) (d : F.Decl) (k : RKind) (dec : Decimal)
    (hd : ∃ k', d.ty = .real k') : SoundInit o Γ d (.realLit k dec) (.floatLit dec) := by
  intro σ τ _ _ x cx hx hcx
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hx; subst hx
  obtain ⟨k', hk⟩ := hd
  simp only [F.convertTo, hk, Option.some.injEq] at hcx; subst hcx
  exact ⟨.float dec.toReal, rfl, RelS.realF⟩

theorem FNP_R0176_sound_boollit (o : Oracles) (Γ : F.Env) (d : F.Decl) (b : Bool) (hd : d.ty = .bool) :
    SoundInit o Γ d (.boolLit b) (.boolLit b) := by
  intro σ τ _ _ x cx hx hcx
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hx; subst hx
  simp only [F.convertTo, hd, Option.some.injEq] at hcx; subst hcx
  exact ⟨.bool b, rfl, RelS.bool⟩

/-- `integer :: n = huge(1)` -> `np.int32(2147483647)`: the default kind's HUGE. -/
theorem FNP_R0176_sound_huge_int (o : Oracles) (Γ : F.Env) (d : F.Decl) (k : IKind) (hd : d.ty = .int k) :
    SoundInit o Γ d (.call .huge (.cons (.intLit .i4 1) .nil) none) (.intLit 2147483647) := by
  intro σ τ _ _ x cx hx hcx
  rw [F_eval_call1, F_eval_intLit] at hx
  simp only [Option.bind_some, Option.map_none] at hx
  rw [show F.intrinsic o.F .huge [.s (.int .i4 1)] none = some (.s (.int .i4 (IKind.huge .i4))) from rfl] at hx
  simp only [Option.some.injEq, F.Val.s.injEq] at hx; subst hx
  simp only [F.convertTo, hd, Option.some.injEq] at hcx; subst hcx
  refine ⟨.int (IKind.huge .i4), rfl, ?_⟩
  have : IKind.huge .i4 = 2147483647 := by decide
  rw [this]; exact RelS.int

/-- `real(8) :: x = a / b` with a REAL literal operand: the folded real quotient. -/
theorem FNP_R0176_sound_realdiv (o : Oracles) (Γ : F.Env) (d : F.Decl) (a b : F.Expr) (a' b' : P.Expr)
    (hd : ∃ k', d.ty = .real k') (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b')
    (hreal : RealTyped o Γ a ∨ RealTyped o Γ b)
    (hna : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ a = some (.s v) → ∃ r, F.toRealS v = some r)
    (hnb : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ b = some (.s v) → ∃ r, F.toRealS v = some r) :
    SoundInit o Γ d (.bin .div a b)
      (Tr.fold1 .np_float64 (.bin .truediv (Tr.fold1 .np_float64 a') (Tr.fold1 .np_float64 b'))) := by
  intro σ τ hR hC v cv hv hcv
  have hs := FNP_R0018_sound o Γ a b a' b' ha hb hreal σ τ hR hC v hv
  obtain ⟨w, hw, hrw⟩ := hs
  -- the quotient is a real; convert to the declared real kind
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨ra, hra⟩ := hna σ τ hR hC x hva
  obtain ⟨rb, hrb⟩ := hnb σ τ hR hC y hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
  have htx := RelS.toR hrx ra hra
  have hty := RelS.toR hry rb hrb
  obtain ⟨k', hk⟩ := hd
  -- the Fortran quotient value
  have hval : ∃ kq, v = .real kq (ra / rb) ∧ rb ≠ 0 := by
    rcases hreal with hr | hr
    · obtain ⟨k₁, r₁, rfl⟩ := hr σ τ hR hC x hva
      simp only [F.toRealS, Option.some.injEq] at hra; subst hra
      cases y with
      | int k₂ n =>
        simp only [F.toRealS, Option.some.injEq] at hrb; subst hrb
        simp only [F.arith] at hxy
        split_ifs at hxy with hz
        simp only [Option.some.injEq] at hxy; subst hxy
        exact ⟨_, rfl, by exact_mod_cast hz⟩
      | real k₂ r₂ =>
        simp only [F.toRealS, Option.some.injEq] at hrb; subst hrb
        simp only [F.arith] at hxy
        split_ifs at hxy with hz
        simp only [Option.some.injEq] at hxy; subst hxy
        exact ⟨_, rfl, hz⟩
      | bool _ => simp [F.toRealS] at hrb
      | str _ => simp [F.toRealS] at hrb
    · obtain ⟨k₂, r₂, rfl⟩ := hr σ τ hR hC y hvb
      simp only [F.toRealS, Option.some.injEq] at hrb; subst hrb
      cases x with
      | int k₁ n =>
        simp only [F.toRealS, Option.some.injEq] at hra; subst hra
        simp only [F.arith] at hxy
        split_ifs at hxy with hz
        simp only [Option.some.injEq] at hxy; subst hxy
        exact ⟨_, rfl, hz⟩
      | real k₁ r₁ =>
        simp only [F.toRealS, Option.some.injEq] at hra; subst hra
        simp only [F.arith] at hxy
        split_ifs at hxy with hz
        simp only [Option.some.injEq] at hxy; subst hxy
        exact ⟨_, rfl, hz⟩
      | bool _ => simp [F.toRealS] at hra
      | str _ => simp [F.toRealS] at hra
  obtain ⟨kq, rfl, hrb0⟩ := hval
  simp only [F.convertTo, hk, Option.some.injEq] at hcv; subst hcv
  refine ⟨.float (ra / rb), ?_, RelS.realF⟩
  rw [P_eval_call1, P_eval_bin, P_eval_call1, P_eval_call1, hx', hy']
  simp only [Option.bind_some]
  rw [helper_np_float64_toR o.P x' ra htx, helper_np_float64_toR o.P y' rb hty]
  simp only [Option.bind_some, P_lift2_scalar]
  rw [P_arith_truediv_toR (P_toR_float ra) (P_toR_float rb) hrb0]
  simp only [Option.map_some, Option.bind_some]
  rfl

end Recast.Rules
