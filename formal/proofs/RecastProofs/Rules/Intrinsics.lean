import RecastProofs.Rules.Logic

/-!
# Rules FNP-R0061 .. FNP-R0093: scalar intrinsic spellings

Each intrinsic is a pair of *equation lemmas* (one for the Fortran
`intrinsic` table, one for the Python `helper` table, both by `rfl`) and a
soundness theorem obtained from a small number of generic lemmas
(`call1_sound_of_rel`, `real_fn_sound`, `call2_sound_of_rel`).  Refutations
are closed counterexamples on the empty environment.

A Python operand standing for a Fortran real may be an `int` with the same
value (`RelS.real_cases`); helpers that dispatch on the Python *type*
(`_f_kind`, `_f_huge`, ...) therefore need the float shape as a premise.
-/

namespace Recast.Rules

open Recast Rel Exact

theorem F_eval_call1 (o : F.Oracle) (σ : F.State) (f : F.Intr) (a : F.Expr) (k : Option IKind) :
    F.eval o σ (.call f (.cons a .nil) k) = (F.eval o σ a).bind fun va =>
      F.intrinsic o f [va] (k.map fun kk => match kk with | .i4 => (4 : ℤ) | .i8 => 8) := by
  cases h : F.eval o σ a <;> simp only [F.eval, F.evalExprs, h] <;> rfl

theorem F_eval_call2 (o : F.Oracle) (σ : F.State) (f : F.Intr) (a b : F.Expr) (k : Option IKind) :
    F.eval o σ (.call f (.cons a (.cons b .nil)) k) = (F.eval o σ a).bind fun va =>
      (F.eval o σ b).bind fun vb =>
      F.intrinsic o f [va, vb] (k.map fun kk => match kk with | .i4 => (4 : ℤ) | .i8 => 8) := by
  cases h1 : F.eval o σ a <;> cases h2 : F.eval o σ b <;> simp only [F.eval, F.evalExprs, h1, h2] <;> rfl

theorem lift1_arr_ne (f : SVal → Option SVal) (av : FArr) (c : SVal) :
    F.lift1 f (.arr av) ≠ some (.s c) := by
  intro h
  obtain ⟨a, ha, _⟩ := F_lift1_eq_scalar h
  cases ha

/-- The elemental one-argument intrinsics never turn an array into a scalar. -/
theorem intrinsic1_arr_ne (o : F.Oracle) (f : F.Intr) (av : FArr) (c : SVal)
    (hf : f = .int ∨ f = .nint ∨ f = .aint ∨ f = .anint ∨ f = .floor ∨ f = .ceiling ∨
      f = .real ∨ f = .dble ∨ f = .abs ∨ f = .sqrt ∨ f = .exp ∨ f = .log ∨ f = .log10 ∨ f = .sin ∨
      f = .cos ∨ f = .tan ∨ f = .asin ∨ f = .acos ∨ f = .atan ∨ f = .sinh ∨ f = .cosh ∨ f = .tanh) :
    F.intrinsic o f [.arr av] none ≠ some (.s c) := by
  intro h
  rcases hf with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl <;>
  first
  | exact lift1_arr_ne _ _ _ h
  | (simp only [F.intrinsic, F.intrinsicCore, F.intrinsicCore] at h; exact lift1_arr_ne _ _ _ h)

/-- The one-argument call obligation shape: `f(a)` against `helper(a')`. -/
def Call1Sound (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper) : Prop :=
  ∀ (a : F.Expr) (a' : P.Expr), SoundScalar o Γ a a' →
    SoundScalar o Γ (.call f (.cons a .nil) none) (Tr.fold1 h a')

/-- Generic: an intrinsic and a helper that agree pointwise on scalars, and an
intrinsic that never turns an array argument into a scalar. -/
theorem call1_sound_of_rel (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper)
    (harr : ∀ (av : FArr) (c : SVal), F.intrinsic o.F f [.arr av] none ≠ some (.s c))
    (hrel : ∀ (x : SVal) (x' : PVal) (c : SVal), RelS x x' → F.intrinsic o.F f [.s x] none = some (.s c) →
      ∃ c', P.helper o.P h [.s x'] = some (.s c') ∧ RelS c c') :
    Call1Sound o Γ f h := by
  intro a a' ha σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | arr av => exact absurd hint (harr av v)
  | s x =>
    obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
    obtain ⟨c', hc', hrc⟩ := hrel x x' v hrx hint
    refine ⟨c', ?_, hrc⟩
    rw [P_eval_call1, hx']
    simpa using hc'

/-- Generic two-argument version. -/
theorem call2_sound_of_rel (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper)
    (a b : F.Expr) (a' b' : P.Expr) (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b')
    (hrel : ∀ (x y : SVal) (x' y' : PVal) (c : SVal), RelS x x' → RelS y y' →
      F.intrinsic o.F f [.s x, .s y] none = some (.s c) →
      ∃ c', P.helper o.P h [.s x', .s y'] = some (.s c') ∧ RelS c c')
    (harr : ∀ (va vb : F.Val) (c : SVal), (∃ av, va = .arr av) ∨ (∃ bv, vb = .arr bv) →
      F.intrinsic o.F f [va, vb] none ≠ some (.s c)) :
    SoundScalar o Γ (.call f (.cons a (.cons b .nil)) none) (Tr.call2 h a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | arr av => exact absurd hint (harr _ _ _ (Or.inl ⟨av, rfl⟩))
  | s x =>
    cases vb with
    | arr bv => exact absurd hint (harr _ _ _ (Or.inr ⟨bv, rfl⟩))
    | s y =>
      obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
      obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb
      obtain ⟨c', hc', hrc⟩ := hrel x y x' y' v hrx hry hint
      refine ⟨c', ?_, hrc⟩
      rw [P_eval_call2, hx', hy']
      simpa using hc'

/-! ### Real elemental functions -/

/-- Soundness of a real elemental function whose two spellings agree with `g`.
`hf` is the Fortran table row (an elementwise lift of `realFn g`); `hp` the
Python helper row on any numeric value denoting the real `r`. -/
theorem real_fn_sound (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper) (g : ℝ → Option ℝ)
    (hf : ∀ v : F.Val, F.intrinsic o.F f [v] none = F.lift1 (F.realFn g) v)
    (hp : ∀ (w : PVal) (r : ℝ), P.toR w = some r →
      P.helper o.P h [.s w] = (g r).map fun y => P.Val.s (.float y)) :
    Call1Sound o Γ f h := by
  apply call1_sound_of_rel
  · intro av c; rw [hf]; exact lift1_arr_ne _ _ _
  · intro x x' c hrel hF
    rw [hf] at hF
    cases hrel with
    | real ht =>
      rename_i k r
      simp only [F.lift1, F.realFn, Option.map_eq_some_iff] at hF
      obtain ⟨y, hy, hc⟩ := hF
      obtain ⟨z, hz, hcz⟩ := hy
      subst hcz
      cases hc
      refine ⟨.float z, ?_, RelS.realF⟩
      rw [hp x' r ht, hz]; rfl
    | int => simp [F.lift1, F.realFn] at hF
    | bool => simp [F.lift1, F.realFn] at hF
    | str => simp [F.lift1, F.realFn] at hF

/-- Helpers written as `(toR v).bind (fun a => ...)` reduce on any numeric value. -/
theorem helper_toR_form (o : P.Oracle) (h : P.Helper) (g : ℝ → Option ℝ) (w : PVal) (r : ℝ) (ht : P.toR w = some r)
    (e : ∀ v : PVal, P.helper o h [.s v] = (P.toR v).bind fun a => (g a).map fun y => P.Val.s (.float y)) :
    P.helper o h [.s w] = (g r).map fun y => P.Val.s (.float y) := by
  rw [e, ht]; rfl

theorem helper_realFn_form (o : P.Oracle) (h : P.Helper) (g : ℝ → Option ℝ) (w : PVal) (r : ℝ) (ht : P.toR w = some r)
    (e : ∀ v : PVal, P.helper o h [.s v] = (P.realFn g v).map P.Val.s) :
    P.helper o h [.s w] = (g r).map fun y => P.Val.s (.float y) := by
  rw [e, P.realFn, ht]; simp [Option.map_map]; rfl

def FNP_R0069_obligation : Prop := ∀ o Γ, Call1Sound o Γ .sqrt .f_sqrt
theorem FNP_R0069_sound : FNP_R0069_obligation := fun o Γ =>
  real_fn_sound o Γ .sqrt .f_sqrt (fun a => if 0 ≤ a then some (Real.sqrt a) else none)
    (fun _ => rfl) (fun w r ht => by
      have e : P.helper o.P .f_sqrt [.s w] =
          (P.toR w).bind (fun a => if 0 ≤ a then some (P.Val.s (.float (Real.sqrt a))) else none) := rfl
      rw [e, ht]; simp only [Option.bind_some]; split_ifs <;> rfl)

def FNP_R0070_obligation : Prop := ∀ o Γ, Call1Sound o Γ .exp .math_exp
theorem FNP_R0070_sound : FNP_R0070_obligation := fun o Γ =>
  real_fn_sound o Γ .exp .math_exp (fun a => some (Real.exp a)) (fun _ => rfl)
    (fun w r ht => helper_realFn_form o.P .math_exp _ w r ht (fun _ => rfl))

def FNP_R0071_obligation : Prop := ∀ o Γ, Call1Sound o Γ .log .f_log
theorem FNP_R0071_sound : FNP_R0071_obligation := fun o Γ =>
  real_fn_sound o Γ .log .f_log (fun a => if 0 < a then some (Real.log a) else none)
    (fun _ => rfl) (fun w r ht => by
      have e : P.helper o.P .f_log [.s w] =
          (P.toR w).bind (fun a => if 0 < a then some (P.Val.s (.float (Real.log a))) else none) := rfl
      rw [e, ht]; simp only [Option.bind_some]; split_ifs <;> rfl)

def FNP_R0072_obligation : Prop := ∀ o Γ, Call1Sound o Γ .log10 .f_log10
theorem FNP_R0072_sound : FNP_R0072_obligation := fun o Γ =>
  real_fn_sound o Γ .log10 .f_log10 (fun a => if 0 < a then some (Real.logb 10 a) else none)
    (fun _ => rfl) (fun w r ht => by
      have e : P.helper o.P .f_log10 [.s w] =
          (P.toR w).bind (fun a => if 0 < a then some (P.Val.s (.float (Real.logb 10 a))) else none) := rfl
      rw [e, ht]; simp only [Option.bind_some]; split_ifs <;> rfl)

def FNP_R0073_obligation : Prop := ∀ o Γ,
  Call1Sound o Γ .sin .math_sin ∧ Call1Sound o Γ .cos .math_cos ∧ Call1Sound o Γ .tan .math_tan ∧
  Call1Sound o Γ .sinh .math_sinh ∧ Call1Sound o Γ .cosh .math_cosh ∧ Call1Sound o Γ .tanh .math_tanh
theorem FNP_R0073_sound : FNP_R0073_obligation := fun o Γ =>
  ⟨real_fn_sound o Γ .sin .math_sin (fun a => some (Real.sin a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_sin _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .cos .math_cos (fun a => some (Real.cos a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_cos _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .tan .math_tan (fun a => some (Real.tan a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_tan _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .sinh .math_sinh (fun a => some (Real.sinh a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_sinh _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .cosh .math_cosh (fun a => some (Real.cosh a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_cosh _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .tanh .math_tanh (fun a => some (Real.tanh a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_tanh _ w r ht (fun _ => rfl))⟩

def FNP_R0074_obligation : Prop := ∀ o Γ,
  Call1Sound o Γ .asin .math_asin ∧ Call1Sound o Γ .acos .math_acos ∧ Call1Sound o Γ .atan .math_atan
theorem FNP_R0074_sound : FNP_R0074_obligation := fun o Γ =>
  ⟨real_fn_sound o Γ .asin .math_asin (fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arcsin a) else none)
      (fun _ => rfl) (fun w r ht => helper_realFn_form o.P .math_asin _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .acos .math_acos (fun a => if -1 ≤ a ∧ a ≤ 1 then some (Real.arccos a) else none)
      (fun _ => rfl) (fun w r ht => helper_realFn_form o.P .math_acos _ w r ht (fun _ => rfl)),
   real_fn_sound o Γ .atan .math_atan (fun a => some (Real.arctan a)) (fun _ => rfl)
      (fun w r ht => helper_realFn_form o.P .math_atan _ w r ht (fun _ => rfl))⟩

/-- Shared uninterpreted primitives (erf, gamma, ...): congruence. -/
def FNP_R0076_obligation : Prop := ∀ o Γ (name : String), Call1Sound o Γ (.prim name) (.prim name)
theorem FNP_R0076_sound : FNP_R0076_obligation := fun o Γ name =>
  real_fn_sound o Γ (.prim name) (.prim name) (fun a => o.prim name [a]) (fun _ => rfl)
    (fun w r ht => by
      have e : P.helper o.P (.prim name) [.s w] =
          (P.toR w).bind (fun a => (o.P.prim name [a]).map fun y => P.Val.s (.float y)) := rfl
      rw [e, ht]; rfl)

/-! ### Conversions and rounding -/

theorem nintR_intCast (n : ℤ) : nintR (n : ℝ) = n := by
  unfold nintR
  have hf : ⌊(1/2 : ℝ)⌋ = 0 := by rw [Int.floor_eq_iff]; norm_num
  have hc : ⌈(-(1/2) : ℝ)⌉ = 0 := by rw [Int.ceil_eq_iff]; norm_num
  split_ifs with h
  · rw [Int.floor_intCast_add, hf, add_zero]
  · rw [sub_eq_add_neg, Int.ceil_intCast_add, hc, add_zero]

/-- Generic: a real-argument intrinsic whose value on the Fortran side is
`gF k r` and whose helper on a float is `gP r`, on an int `gI n`. -/
theorem real_arg_sound (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper)
    (harr : ∀ (av : FArr) (c : SVal), F.intrinsic o.F f [.arr av] none ≠ some (.s c))
    (hnone : ∀ x c, (∀ k r, x ≠ .real k r) → F.intrinsic o.F f [.s x] none ≠ some (.s c))
    (gF : RKind → ℝ → SVal) (gP : ℝ → PVal) (gI : ℤ → PVal)
    (hF : ∀ k r, F.intrinsic o.F f [.s (.real k r)] none = some (.s (gF k r)))
    (hP : ∀ r, P.helper o.P h [.s (.float r)] = some (.s (gP r)))
    (hI : ∀ n, P.helper o.P h [.s (.int n)] = some (.s (gI n)))
    (hrelP : ∀ k r, RelS (gF k r) (gP r)) (hrelI : ∀ k (n : ℤ), RelS (gF k (n : ℝ)) (gI n)) :
    Call1Sound o Γ f h := by
  apply call1_sound_of_rel o Γ f h harr
  intro x x' c hrx hint
  cases hrx with
  | real ht =>
    rename_i k r
    rw [hF] at hint; cases hint
    rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
    · exact ⟨gI n, hI n, hrelI k n⟩
    · exact ⟨gP r, hP r, hrelP k r⟩
  | int => exact absurd hint (hnone _ _ (fun _ _ h => by cases h))
  | bool => exact absurd hint (hnone _ _ (fun _ _ h => by cases h))
  | str => exact absurd hint (hnone _ _ (fun _ _ h => by cases h))

def FNP_R0062_obligation : Prop := ∀ o Γ, Call1Sound o Γ .nint .f_nint
theorem FNP_R0062_sound : FNP_R0062_obligation := fun o Γ =>
  real_arg_sound o Γ .nint .f_nint
    (fun av c => intrinsic1_arr_ne o.F .nint av c (by simp))
    (fun x c hx => by cases x <;> first | exact absurd rfl (hx _ _) | simp [F.intrinsic, F.intrinsicCore, F.lift1])
    (fun _ r => .int .i4 (nintR r)) (fun r => .int (nintR r)) (fun n => .int n)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ _ => RelS.int)
    (fun _ n => by rw [nintR_intCast]; exact RelS.int)

def FNP_R0063_obligation : Prop := ∀ o Γ, Call1Sound o Γ .aint .np_trunc
theorem FNP_R0063_sound : FNP_R0063_obligation := fun o Γ =>
  real_arg_sound o Γ .aint .np_trunc
    (fun av c => intrinsic1_arr_ne o.F .aint av c (by simp))
    (fun x c hx => by cases x <;> first | exact absurd rfl (hx _ _) | simp [F.intrinsic, F.intrinsicCore, F.lift1])
    (fun k r => .real k (truncR r : ℝ)) (fun r => .float (truncR r : ℝ)) (fun n => .int n)
    (fun _ _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ _ => RelS.realF)
    (fun _ n => by rw [truncR_intCast]; exact RelS.realI)

def FNP_R0065_obligation : Prop := ∀ o Γ, Call1Sound o Γ .floor .math_floor
theorem FNP_R0065_sound : FNP_R0065_obligation := fun o Γ =>
  real_arg_sound o Γ .floor .math_floor
    (fun av c => intrinsic1_arr_ne o.F .floor av c (by simp))
    (fun x c hx => by cases x <;> first | exact absurd rfl (hx _ _) | simp [F.intrinsic, F.intrinsicCore, F.lift1])
    (fun _ r => .int .i4 ⌊r⌋) (fun r => .int ⌊r⌋) (fun n => .int n)
    (fun _ _ => rfl) (fun _ => rfl) (fun n => by
      show (P.toR (.int n)).map (fun a => P.Val.s (.int ⌊a⌋)) = _
      simp [P.toR])
    (fun _ _ => RelS.int) (fun _ n => by rw [Int.floor_intCast]; exact RelS.int)

def FNP_R0066_obligation : Prop := ∀ o Γ, Call1Sound o Γ .ceiling .math_ceil
theorem FNP_R0066_sound : FNP_R0066_obligation := fun o Γ =>
  real_arg_sound o Γ .ceiling .math_ceil
    (fun av c => intrinsic1_arr_ne o.F .ceiling av c (by simp))
    (fun x c hx => by cases x <;> first | exact absurd rfl (hx _ _) | simp [F.intrinsic, F.intrinsicCore, F.lift1])
    (fun _ r => .int .i4 ⌈r⌉) (fun r => .int ⌈r⌉) (fun n => .int n)
    (fun _ _ => rfl) (fun _ => rfl) (fun n => by
      show (P.toR (.int n)).map (fun a => P.Val.s (.int ⌈a⌉)) = _
      simp [P.toR])
    (fun _ _ => RelS.int) (fun _ n => by rw [Int.ceil_intCast]; exact RelS.int)

/-- REAL/DBLE: precision casts are identities on values (both int and real arguments). -/
def FNP_R0067_obligation : Prop := ∀ o Γ, Call1Sound o Γ .real .np_float64 ∧ Call1Sound o Γ .dble .np_float64
theorem FNP_R0067_sound : FNP_R0067_obligation := by
  intro o Γ
  have hnp : ∀ (w : PVal) (r : ℝ), P.toR w = some r → P.helper o.P .np_float64 [.s w] = some (.s (.float r)) := by
    intro w r ht
    show P.lift1 (fun v => (P.toR v).map PVal.float) (.s w) = _
    simp [P.lift1, ht]
  constructor <;> apply call1_sound_of_rel
  · intro av c; exact intrinsic1_arr_ne o.F .real av c (by simp)
  · intro x x' c hrx hint
    cases hrx with
    | int =>
      rename_i k n
      rw [show F.intrinsic o.F .real [.s (.int k n)] none = some (.s (.real .r4 (n : ℝ))) from rfl] at hint
      cases hint; exact ⟨.float (n : ℝ), hnp _ _ rfl, RelS.realF⟩
    | real ht =>
      rename_i k r
      rw [show F.intrinsic o.F .real [.s (.real k r)] none = some (.s (.real .r4 r)) from rfl] at hint
      cases hint; exact ⟨.float r, hnp x' r ht, RelS.realF⟩
    | bool => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint
    | str => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint
  · intro av c; exact intrinsic1_arr_ne o.F .dble av c (by simp)
  · intro x x' c hrx hint
    cases hrx with
    | int =>
      rename_i k n
      rw [show F.intrinsic o.F .dble [.s (.int k n)] none = some (.s (.real .r8 (n : ℝ))) from rfl] at hint
      cases hint; exact ⟨.float (n : ℝ), hnp _ _ rfl, RelS.realF⟩
    | real ht =>
      rename_i k r
      rw [show F.intrinsic o.F .dble [.s (.real k r)] none = some (.s (.real .r8 r)) from rfl] at hint
      cases hint; exact ⟨.float r, hnp x' r ht, RelS.realF⟩
    | bool => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint
    | str => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint

def FNP_R0068_obligation : Prop := ∀ o Γ, Call1Sound o Γ .abs .py_abs
theorem FNP_R0068_sound : FNP_R0068_obligation := by
  intro o Γ
  apply call1_sound_of_rel
  · intro av c; exact intrinsic1_arr_ne o.F .abs av c (by simp)
  · intro x x' c hrx hint
    cases hrx with
    | int =>
      rename_i k n
      rw [show F.intrinsic o.F .abs [.s (.int k n)] none = some (.s (.int k |n|)) from rfl] at hint
      cases hint; exact ⟨.int |n|, rfl, RelS.int⟩
    | real ht =>
      rename_i k r
      rw [show F.intrinsic o.F .abs [.s (.real k r)] none = some (.s (.real k |r|)) from rfl] at hint
      cases hint
      rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
      · exact ⟨.int |n|, rfl, by rw [← Int.cast_abs]; exact RelS.realI⟩
      · exact ⟨.float |r|, rfl, RelS.realF⟩
    | bool => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint
    | str => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint

/-- INT for the default kind on the representable range. -/
def FNP_R0061_obligation_default : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr), SoundScalar o Γ a a' →
    (∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k r, F.eval o.F σ a = some (.s (.real k r)) → |r| < 2147483648) →
    SoundScalar o Γ (.call .int (.cons a .nil) none) (Tr.fold1 .f_int a')

theorem helper_f_int_float (o : P.Oracle) (r : ℝ) :
    P.helper o .f_int [.s (.float r)] =
      (if -(2147483648 : ℝ) ≤ r ∧ r < 2147483648 then some (.s (.int (truncR r)))
       else some (.s (.int (-2147483648)))) := rfl

theorem FNP_R0061_sound_default : FNP_R0061_obligation_default := by
  intro o Γ a a' ha hrange σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | arr av =>
    exact absurd hint (intrinsic1_arr_ne o.F .int av v (by simp))
  | s x =>
    obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
    cases hrx with
    | int =>
      rename_i k n
      rw [show F.intrinsic o.F .int [.s (.int k n)] none = some (.s (.int .i4 n)) from rfl] at hint
      cases hint
      refine ⟨.int n, ?_, RelS.int⟩
      rw [P_eval_call1, hx']; rfl
    | real ht =>
      rename_i k r
      rw [show F.intrinsic o.F .int [.s (.real k r)] none = some (.s (.int .i4 (truncR r))) from rfl] at hint
      cases hint
      have hr := hrange σ τ hR hC k r hva
      refine ⟨.int (truncR r), ?_, RelS.int⟩
      rw [P_eval_call1, hx', Option.bind_some]
      rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
      · rw [truncR_intCast]; rfl
      · rw [helper_f_int_float]
        have hb : -(2147483648 : ℝ) ≤ r ∧ r < 2147483648 := by
          constructor
          · linarith [neg_abs_le r, hr]
          · linarith [le_abs_self r, hr]
        rw [if_pos hb]
    | bool => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint
    | str => simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint

/-- INT with an explicit KIND=8, as the emitter translates it (kind dropped). -/
def FNP_R0061_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr), SoundScalar o Γ a a' →
    SoundScalar o Γ (.call .int (.cons a .nil) (some .i8)) (Tr.fold1 .f_int a')

theorem realLit_sound (k : RKind) (d : Decimal) : SoundScalar noPrim emptyEnv (.realLit k d) (.floatLit d) := by
  intro σ τ _ _ v hv; simp only [F.eval] at hv; cases hv; exact ⟨_, rfl, RelS.realF⟩

/-- FNP-D0005: `int(3.0d10, kind=8)`. -/
theorem FNP_R0061_refuted : ¬ FNP_R0061_obligation := by
  intro h
  have hval : Decimal.toReal ⟨3, 10⟩ = ((30000000000 : ℤ) : ℝ) := by
    simp [Decimal.toReal]; norm_num
  have := h noPrim emptyEnv _ _ (realLit_sound .r8 ⟨3, 10⟩) emptyF emptyP relState_empty conforms_empty
    (.int .i8 30000000000) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r8 ⟨3, 10⟩) = some (.s (.real .r8 (Decimal.toReal ⟨3, 10⟩))) from rfl,
        Option.bind_some]
      show some (F.Val.s (.int .i8 (truncR (Decimal.toReal ⟨3, 10⟩)))) = _
      rw [hval, truncR_intCast])
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.floatLit ⟨3, 10⟩) = some (.s (.float (Decimal.toReal ⟨3, 10⟩))) from rfl,
    Option.bind_some, helper_f_int_float, hval] at hw
  rw [if_neg (by norm_num)] at hw
  cases hw
  cases hrel

/-- FNP-D0004: ANINT(2.5). -/
def FNP_R0064_obligation : Prop := ∀ o Γ, Call1Sound o Γ .anint .np_round

theorem nintR_five_halves : nintR (5/2 : ℝ) = 3 := by
  unfold nintR
  rw [if_pos (by norm_num)]
  rw [show (5/2 : ℝ) + 1/2 = ((3 : ℤ) : ℝ) by norm_num, Int.floor_intCast]

theorem roundHalfEven_five_halves : roundHalfEvenR (5/2 : ℝ) = 2 := by
  unfold roundHalfEvenR
  have hf : ⌊(5/2 : ℝ)⌋ = 2 := by rw [Int.floor_eq_iff]; norm_num
  simp only [hf]; norm_num

theorem FNP_R0064_refuted : ¬ FNP_R0064_obligation := by
  intro h
  have hval : Decimal.toReal ⟨25, -1⟩ = (5/2 : ℝ) := by simp [Decimal.toReal]; norm_num
  have := h noPrim emptyEnv _ _ (realLit_sound .r8 ⟨25, -1⟩) emptyF emptyP relState_empty conforms_empty
    (.real .r8 3) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r8 ⟨25, -1⟩) = some (.s (.real .r8 (Decimal.toReal ⟨25, -1⟩))) from rfl,
        Option.bind_some]
      show some (F.Val.s (.real .r8 (nintR (Decimal.toReal ⟨25, -1⟩) : ℝ))) = _
      rw [hval, nintR_five_halves]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.floatLit ⟨25, -1⟩) = some (.s (.float (Decimal.toReal ⟨25, -1⟩))) from rfl,
    Option.bind_some] at hw
  rw [show P.helper noPrim.P .np_round [.s (.float (Decimal.toReal ⟨25, -1⟩))] =
      some (.s (.float (roundHalfEvenR (Decimal.toReal ⟨25, -1⟩) : ℝ))) from rfl, hval, roundHalfEven_five_halves] at hw
  cases hw
  have := RelS.real_toR hrel
  simp [P.toR] at this

/-! ### Two-argument numeric intrinsics -/

theorem intrinsic2_arr_ne (o : F.Oracle) (f : F.Intr) (va vb : F.Val) (c : SVal)
    (h : (∃ av, va = .arr av) ∨ (∃ bv, vb = .arr bv))
    (hf : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨
      f = .atan2 ∨ f = .iand ∨ f = .ior ∨ f = .ieor ∨ f = .ishft) :
    F.intrinsic o f [va, vb] none ≠ some (.s c) := by
  intro hc
  rcases hf with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  rcases h with ⟨av, rfl⟩ | ⟨bv, rfl⟩ <;>
  first
  | exact F_lift2_arr_left _ _ _ _ hc
  | (cases va with
      | arr av' => exact F_lift2_arr_left _ _ _ _ hc
      | s a => exact F_lift2_arr_right _ _ _ _ hc)
  | (simp only [F.intrinsic, F.intrinsicCore, List.foldl, Option.bind_some] at hc
     first
     | exact F_lift2_arr_left _ _ _ _ hc
     | (cases va with
         | arr av' => exact F_lift2_arr_left _ _ _ _ hc
         | s a => exact F_lift2_arr_right _ _ _ _ hc))

/-- The two-argument numeric intrinsics are undefined on mixed int/real or
non-numeric scalar operands. -/
theorem intrinsic2_mixed_none (o : F.Oracle) (f : F.Intr) (u v : SVal)
    (hf : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨ f = .atan2)
    (hmix : ¬ ((∃ k₁ x k₂ y, u = .int k₁ x ∧ v = .int k₂ y) ∨ (∃ k₁ x k₂ y, u = .real k₁ x ∧ v = .real k₂ y))) :
    F.intrinsic o f [.s u, .s v] none = none := by
  rcases hf with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> cases u <;> cases v <;>
  first
  | exact absurd (Or.inl ⟨_, _, _, _, rfl, rfl⟩) hmix
  | exact absurd (Or.inr ⟨_, _, _, _, rfl, rfl⟩) hmix
  | rfl

theorem intrinsic2_mixed_absurd {o : F.Oracle} {f : F.Intr} {u v c : SVal}
    (hf : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨ f = .atan2)
    (hF : F.intrinsic o f [.s u, .s v] none = some (.s c))
    (hmix : ¬ ((∃ k₁ x k₂ y, u = .int k₁ x ∧ v = .int k₂ y) ∨ (∃ k₁ x k₂ y, u = .real k₁ x ∧ v = .real k₂ y))) :
    False := by
  rw [intrinsic2_mixed_none o f u v hf hmix] at hF
  cases hF

/-- MOD on integers. -/
theorem helper_f_mod_int (o : P.Oracle) (u w : ℤ) (hw : w ≠ 0) :
    P.helper o .f_mod [.s (.int u), .s (.int w)] = some (.s (.int (u - truncR ((u : ℝ) / (w : ℝ)) * w))) := by
  have e : P.helper o .f_mod [.s (.int u), .s (.int w)] =
      (P.arith .truediv (.int u) (.int w)).bind fun q => match q, PVal.int u, PVal.int w with
        | .float r, .int x, .int y => some (.s (.int (x - truncR r * y)))
        | .float r, _, _ => (P.toR (.int u)).bind fun x => (P.toR (.int w)).map fun y => .s (.float (x - (truncR r : ℝ) * y))
        | _, _, _ => none := rfl
  rw [e]; simp [P.arith, hw]

def FNP_R0077_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .mod (.cons a (.cons b .nil)) none) (Tr.call2 .f_mod a' b')

theorem FNP_R0077_sound : FNP_R0077_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .mod .f_mod a b a' b' ha hb ?_
    (fun va vb c h => intrinsic2_arr_ne o.F .mod va vb c h (by simp))
  intro x y x' y' c hrx hry hF
  cases hrx with
  | int =>
    rename_i k1 u
    cases hry with
    | int =>
      rename_i k2 w
      have e : F.intrinsic o.F .mod [.s (.int k1 u), .s (.int k2 w)] none =
          (if w = 0 then none else some (SVal.int (IKind.join k1 k2) (tmod u w))).map F.Val.s := rfl
      rw [e] at hF
      split_ifs at hF with hw
      · simp at hF
      · simp only [Option.map_some, Option.some.injEq, F.Val.s.injEq] at hF; subst hF
        refine ⟨.int (tmod u w), ?_, RelS.int⟩
        rw [helper_f_mod_int o.P u w hw, truncR_div_intCast u w hw]
        congr 3
        rw [tmod, Int.tmod_def, tquot]; ring
    | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | real ht₁ =>
    rename_i k1 u
    cases hry with
    | real ht₂ =>
      rename_i k2 w
      have e : F.intrinsic o.F .mod [.s (.real k1 u), .s (.real k2 w)] none =
          (if w = 0 then none else some (SVal.real (RKind.join k1 k2) (modR u w))).map F.Val.s := rfl
      rw [e] at hF
      split_ifs at hF with hw
      · simp at hF
      · simp only [Option.map_some, Option.some.injEq, F.Val.s.injEq] at hF; subst hF
        -- Python: _f_mod on the two numeric representations
        rcases P_toR_cases ht₁ with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases ht₂ with ⟨m, rfl, rfl⟩ | rfl
        · have hm : m ≠ 0 := by exact_mod_cast hw
          refine ⟨_, helper_f_mod_int o.P n m hm, RelS.real ?_⟩
          simp [P.toR, modR]
        · refine ⟨.float (modR n w), ?_, RelS.realF⟩
          have e2 : P.helper o.P .f_mod [.s (.int n), .s (.float w)] =
              (P.arith .truediv (.int n) (.float w)).bind fun q => match q, PVal.int n, PVal.float w with
                | .float r, .int x, .int y => some (.s (.int (x - truncR r * y)))
                | .float r, _, _ => (P.toR (.int n)).bind fun x => (P.toR (.float w)).map fun y => .s (.float (x - (truncR r : ℝ) * y))
                | _, _, _ => none := rfl
          rw [e2]; simp [P.arith, P.toR, hw, modR]
        · have hm : (m : ℝ) ≠ 0 := by exact_mod_cast hw
          refine ⟨.float (modR u m), ?_, RelS.realF⟩
          have e2 : P.helper o.P .f_mod [.s (.float u), .s (.int m)] =
              (P.arith .truediv (.float u) (.int m)).bind fun q => match q, PVal.float u, PVal.int m with
                | .float r, .int x, .int y => some (.s (.int (x - truncR r * y)))
                | .float r, _, _ => (P.toR (.float u)).bind fun x => (P.toR (.int m)).map fun y => .s (.float (x - (truncR r : ℝ) * y))
                | _, _, _ => none := rfl
          rw [e2]; simp [P.arith, P.toR, hm, modR]
        · refine ⟨.float (modR u w), ?_, RelS.realF⟩
          have e2 : P.helper o.P .f_mod [.s (.float u), .s (.float w)] =
              (P.arith .truediv (.float u) (.float w)).bind fun q => match q, PVal.float u, PVal.float w with
                | .float r, .int x, .int y => some (.s (.int (x - truncR r * y)))
                | .float r, _, _ => (P.toR (.float u)).bind fun x => (P.toR (.float w)).map fun y => .s (.float (x - (truncR r : ℝ) * y))
                | _, _, _ => none := rfl
          rw [e2]; simp [P.arith, P.toR, hw, modR]
    | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim

/-- Python `%` on two numbers denoting reals: floor remainder. -/
theorem P_arith_mod_toR {w₁ w₂ : PVal} {a b : ℝ} (h1 : P.toR w₁ = some a) (h2 : P.toR w₂ = some b) (hb : b ≠ 0) :
    ∃ c, P.arith .mod w₁ w₂ = some c ∧ P.toR c = some (moduloR a b) := by
  rcases P_toR_cases h1 with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases h2 with ⟨m, rfl, rfl⟩ | rfl
  · have hm : m ≠ 0 := by exact_mod_cast hb
    refine ⟨.int (fmod n m), by simp [P.arith, hm], ?_⟩
    simp only [P.toR, Option.some.injEq, moduloR]
    rw [fmod, Int.fmod_def, floor_div_intCast n m hm, fquot]; push_cast; ring
  · exact ⟨.float (moduloR n b), by simp [P.arith, P.toR, hb, moduloR], rfl⟩
  · have hm : (m : ℝ) ≠ 0 := by exact_mod_cast hb
    exact ⟨.float (moduloR a m), by simp [P.arith, P.toR, hm, moduloR], rfl⟩
  · exact ⟨.float (moduloR a b), by simp [P.arith, P.toR, hb, moduloR], rfl⟩

def FNP_R0078_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .modulo (.cons a (.cons b .nil)) none) (Tr.call2 .f_modulo a' b')

theorem helper_f_modulo (o : P.Oracle) (u w : PVal) :
    P.helper o .f_modulo [.s u, .s w] = (P.arith .mod u w).map P.Val.s := rfl

theorem FNP_R0078_sound : FNP_R0078_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .modulo .f_modulo a b a' b' ha hb ?_
    (fun va vb c h => intrinsic2_arr_ne o.F .modulo va vb c h (by simp))
  intro x y x' y' c hrx hry hF
  cases hrx with
  | int =>
    rename_i k1 u
    cases hry with
    | int =>
      rename_i k2 w
      have e : F.intrinsic o.F .modulo [.s (.int k1 u), .s (.int k2 w)] none =
          (if w = 0 then none else some (SVal.int (IKind.join k1 k2) (fmod u w))).map F.Val.s := rfl
      rw [e] at hF
      split_ifs at hF with hw
      · simp at hF
      · simp only [Option.map_some, Option.some.injEq, F.Val.s.injEq] at hF; subst hF
        refine ⟨.int (fmod u w), ?_, RelS.int⟩
        rw [helper_f_modulo]; simp [P.arith, hw]
    | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | real ht₁ =>
    rename_i k1 u
    cases hry with
    | real ht₂ =>
      rename_i k2 w
      have e : F.intrinsic o.F .modulo [.s (.real k1 u), .s (.real k2 w)] none =
          (if w = 0 then none else some (SVal.real (RKind.join k1 k2) (moduloR u w))).map F.Val.s := rfl
      rw [e] at hF
      split_ifs at hF with hw
      · simp at hF
      · simp only [Option.map_some, Option.some.injEq, F.Val.s.injEq] at hF; subst hF
        obtain ⟨c', hc', hr'⟩ := P_arith_mod_toR ht₁ ht₂ hw
        exact ⟨c', by rw [helper_f_modulo, hc']; rfl, RelS.real hr'⟩
    | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim

def FNP_R0079_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .sign (.cons a (.cons b .nil)) none) (Tr.call2 .f_sign a' b')

theorem FNP_R0079_sound : FNP_R0079_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .sign .f_sign a b a' b' ha hb ?_
    (fun va vb c h => intrinsic2_arr_ne o.F .sign va vb c h (by simp))
  intro x y x' y' c hrx hry hF
  cases hrx with
  | int =>
    rename_i k1 u
    cases hry with
    | int =>
      rename_i k2 w
      rw [show F.intrinsic o.F .sign [.s (.int k1 u), .s (.int k2 w)] none =
          some (.s (.int k1 (if 0 ≤ w then |u| else -|u|))) from rfl] at hF
      cases hF
      exact ⟨.int _, rfl, RelS.int⟩
    | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | real ht₁ =>
    rename_i k1 u
    cases hry with
    | real ht₂ =>
      rename_i k2 w
      rw [show F.intrinsic o.F .sign [.s (.real k1 u), .s (.real k2 w)] none =
          some (.s (.real k1 (if 0 ≤ w then |u| else -|u|))) from rfl] at hF
      cases hF
      rcases P_toR_cases ht₁ with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases ht₂ with ⟨m, rfl, rfl⟩ | rfl
      · refine ⟨.int (if 0 ≤ m then |n| else -|n|), rfl, RelS.real ?_⟩
        simp only [P.toR, Option.some.injEq]
        have : (0 : ℝ) ≤ (m : ℝ) ↔ 0 ≤ m := by exact_mod_cast Iff.rfl
        split_ifs <;> simp_all
      · refine ⟨.float (if 0 ≤ w then |(n : ℝ)| else -|(n : ℝ)|), ?_, RelS.realF⟩
        show some (P.Val.s (.float (if 0 ≤ w then |(n : ℝ)| else -|(n : ℝ)|))) = _
        rfl
      · refine ⟨.float (if 0 ≤ m then |u| else -|u|), ?_, RelS.real ?_⟩
        · rfl
        · simp only [P.toR, Option.some.injEq]
          have : (0 : ℝ) ≤ (m : ℝ) ↔ 0 ≤ m := by exact_mod_cast Iff.rfl
          split_ifs <;> simp_all
      · exact ⟨.float _, rfl, RelS.realF⟩
    | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim

def FNP_R0080_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .dim (.cons a (.cons b .nil)) none) (Tr.call2 .f_dim a' b')

theorem helper_f_dim (o : P.Oracle) (u w : PVal) :
    P.helper o .f_dim [.s u, .s w] = (P.arith .sub u w).bind fun d => match d with
      | .int v => some (.s (.int (max v 0)))
      | .float v => some (.s (.float (max v 0)))
      | _ => none := rfl

theorem FNP_R0080_sound : FNP_R0080_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .dim .f_dim a b a' b' ha hb ?_
    (fun va vb c h => intrinsic2_arr_ne o.F .dim va vb c h (by simp))
  intro x y x' y' c hrx hry hF
  cases hrx with
  | int =>
    rename_i k1 u
    cases hry with
    | int =>
      rename_i k2 w
      rw [show F.intrinsic o.F .dim [.s (.int k1 u), .s (.int k2 w)] none = some (.s (.int k1 (max (u - w) 0))) from rfl] at hF
      cases hF
      exact ⟨.int (max (u - w) 0), rfl, RelS.int⟩
    | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | real ht₁ =>
    rename_i k1 u
    cases hry with
    | real ht₂ =>
      rename_i k2 w
      rw [show F.intrinsic o.F .dim [.s (.real k1 u), .s (.real k2 w)] none = some (.s (.real k1 (max (u - w) 0))) from rfl] at hF
      cases hF
      obtain ⟨d, hd, hdr⟩ := P_arith_addsubmul_toR .sub (Or.inr (Or.inl rfl)) ht₁ ht₂
      simp only [opR] at hdr
      rw [helper_f_dim, hd, Option.bind_some]
      rcases P_toR_cases hdr with ⟨n, rfl, hn⟩ | rfl
      · refine ⟨.int (max n 0), rfl, RelS.real ?_⟩
        rw [hn]; simp [P.toR]
      · exact ⟨.float (max (u - w) 0), rfl, RelS.realF⟩
    | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim

/-- MAX/MIN with two arguments. -/
def FNP_R0081_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .max (.cons a (.cons b .nil)) none) (Tr.call2 .f_max a' b') ∧
    SoundScalar o Γ (.call .min (.cons a (.cons b .nil)) none) (Tr.call2 .f_min a' b')

theorem helper_f_max2 (o : P.Oracle) (u w : PVal) :
    P.helper o .f_max [.s u, .s w] = (P.toR u).bind fun p => (P.toR w).map fun q => P.Val.s (if p < q then w else u) := rfl

theorem helper_f_min2 (o : P.Oracle) (u w : PVal) :
    P.helper o .f_min [.s u, .s w] = (P.toR u).bind fun p => (P.toR w).map fun q => P.Val.s (if q < p then w else u) := rfl

theorem FNP_R0081_sound : FNP_R0081_obligation := by
  intro o Γ a b a' b' ha hb
  constructor
  · refine call2_sound_of_rel o Γ .max .f_max a b a' b' ha hb ?_
      (fun va vb c h => intrinsic2_arr_ne o.F .max va vb c h (by simp))
    intro x y x' y' c hrx hry hF
    cases hrx with
    | int =>
      rename_i k1 u
      cases hry with
      | int =>
        rename_i k2 w
        rw [show F.intrinsic o.F .max [.s (.int k1 u), .s (.int k2 w)] none =
            some (.s (.int (IKind.join k1 k2) (max u w))) from rfl] at hF
        cases hF
        refine ⟨.int (max u w), ?_, RelS.int⟩
        rw [helper_f_max2]
        simp only [P.toR, Option.bind_some, Option.map_some, Option.some.injEq, P.Val.s.injEq]
        split_ifs with hlt
        · rw [max_eq_right (by exact_mod_cast hlt.le)]
        · rw [max_eq_left (by exact_mod_cast (not_lt.mp hlt))]
      | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | real ht₁ =>
      rename_i k1 u
      cases hry with
      | real ht₂ =>
        rename_i k2 w
        rw [show F.intrinsic o.F .max [.s (.real k1 u), .s (.real k2 w)] none =
            some (.s (.real (RKind.join k1 k2) (max u w))) from rfl] at hF
        cases hF
        refine ⟨if u < w then y' else x', ?_, RelS.real ?_⟩
        · rw [helper_f_max2, ht₁, ht₂]; rfl
        · split_ifs with hlt
          · rw [max_eq_right hlt.le]; exact ht₂
          · rw [max_eq_left (not_lt.mp hlt)]; exact ht₁
      | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
  · refine call2_sound_of_rel o Γ .min .f_min a b a' b' ha hb ?_
      (fun va vb c h => intrinsic2_arr_ne o.F .min va vb c h (by simp))
    intro x y x' y' c hrx hry hF
    cases hrx with
    | int =>
      rename_i k1 u
      cases hry with
      | int =>
        rename_i k2 w
        rw [show F.intrinsic o.F .min [.s (.int k1 u), .s (.int k2 w)] none =
            some (.s (.int (IKind.join k1 k2) (min u w))) from rfl] at hF
        cases hF
        refine ⟨.int (min u w), ?_, RelS.int⟩
        rw [helper_f_min2]
        simp only [P.toR, Option.bind_some, Option.map_some, Option.some.injEq, P.Val.s.injEq]
        split_ifs with hlt
        · rw [min_eq_right (by exact_mod_cast hlt.le)]
        · rw [min_eq_left (by exact_mod_cast (not_lt.mp hlt))]
      | real => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | real ht₁ =>
      rename_i k1 u
      cases hry with
      | real ht₂ =>
        rename_i k2 w
        rw [show F.intrinsic o.F .min [.s (.real k1 u), .s (.real k2 w)] none =
            some (.s (.real (RKind.join k1 k2) (min u w))) from rfl] at hF
        cases hF
        refine ⟨if w < u then y' else x', ?_, RelS.real ?_⟩
        · rw [helper_f_min2, ht₁, ht₂]; rfl
        · split_ifs with hlt
          · rw [min_eq_right hlt.le]; exact ht₂
          · rw [min_eq_left (not_lt.mp hlt)]; exact ht₁
      | int => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | bool => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
      | str => exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | bool => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim
    | str => cases hry <;> exact (intrinsic2_mixed_absurd (by simp) hF (by simp)).elim

/-! ### Character length -/

def FNP_R0083_obligation : Prop := ∀ o Γ, Call1Sound o Γ .len .py_len
theorem FNP_R0083_sound : FNP_R0083_obligation := by
  intro o Γ
  apply call1_sound_of_rel
  · intro av c h; exact absurd h (by rw [show F.intrinsic o.F .len [.arr av] none = none from rfl]; simp)
  · intro x x' c hrx hint
    cases hrx with
    | str =>
      rename_i s
      rw [show F.intrinsic o.F .len [.s (.str s)] none = some (.s (.int .i4 s.length)) from rfl] at hint
      cases hint; exact ⟨.int s.length, rfl, RelS.int⟩
    | int => simp [F.intrinsic, F.intrinsicCore] at hint
    | real => simp [F.intrinsic, F.intrinsicCore] at hint
    | bool => simp [F.intrinsic, F.intrinsicCore] at hint

def FNP_R0084_obligation : Prop := ∀ o Γ, Call1Sound o Γ .len_trim .f_len_trim
theorem FNP_R0084_sound : FNP_R0084_obligation := by
  intro o Γ
  apply call1_sound_of_rel
  · intro av c h; exact absurd h (by rw [show F.intrinsic o.F .len_trim [.arr av] none = none from rfl]; simp)
  · intro x x' c hrx hint
    cases hrx with
    | str =>
      rename_i s
      rw [show F.intrinsic o.F .len_trim [.s (.str s)] none =
          some (.s (.int .i4 ((s.reverse.dropWhile (· = ' ')).length))) from rfl] at hint
      cases hint
      refine ⟨.int _, ?_, RelS.int⟩
      show some (P.Val.s (.int ((P.rstripBlanks s).length : ℤ))) = _
      simp [P.rstripBlanks]
    | int => simp [F.intrinsic, F.intrinsicCore] at hint
    | real => simp [F.intrinsic, F.intrinsicCore] at hint
    | bool => simp [F.intrinsic, F.intrinsicCore] at hint

/-! ### Kind inquiries: refuted for non-default kinds, and for int-valued reals -/

def FNP_R0086_obligation : Prop := ∀ o Γ, Call1Sound o Γ .kind .f_kind

/-- FNP-D0007: `kind(x4)` with `real(4) :: x4`. -/
theorem FNP_R0086_refuted : ¬ FNP_R0086_obligation := by
  intro h
  have := h noPrim emptyEnv _ _ (realLit_sound .r4 ⟨1, 0⟩) emptyF emptyP relState_empty conforms_empty
    (.int .i4 4) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r4 ⟨1, 0⟩) = some (.s (.real .r4 (Decimal.toReal ⟨1, 0⟩))) from rfl,
        Option.bind_some]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.floatLit ⟨1, 0⟩) = some (.s (.float (Decimal.toReal ⟨1, 0⟩))) from rfl,
    Option.bind_some,
    show P.helper noPrim.P .f_kind [.s (.float (Decimal.toReal ⟨1, 0⟩))] = some (.s (.int 8)) from rfl] at hw
  cases hw
  cases hrel

def FNP_R0087_obligation : Prop := ∀ o Γ, Call1Sound o Γ .huge .f_huge

/-- HUGE for the default integer kind, and for REAL(8) *when the Python value
is a float*: `_f_huge` dispatches on the Python type, so a REAL holding an
integer-valued `int` answers the INTEGER huge (see `FNP_R0087_refuted_realInt`). -/
theorem FNP_R0087_sound_default (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr)
    (ha : SoundScalar o Γ a a')
    (hk : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ a = some (.s v) →
      (∃ n, v = .int .i4 n) ∨ (∃ r, v = .real .r8 r ∧ ∀ w, P.eval o.P τ a' = some (.s w) → w = .float r)) :
    SoundScalar o Γ (.call .huge (.cons a .nil) none) (Tr.fold1 .f_huge a') := by
  intro σ τ hR hC v hv
  rw [F_eval_call1] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, hint⟩ := hv
  cases va with
  | arr av => exfalso; rw [show F.intrinsic o.F .huge [.arr av] none = none from rfl] at hint; cases hint
  | s x =>
    obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
    rcases hk σ τ hR hC x hva with ⟨n, rfl⟩ | ⟨r, rfl, hfloat⟩
    · rw [RelS.int_eq hrx] at hx'
      rw [show F.intrinsic o.F .huge [.s (.int .i4 n)] none = some (.s (.int .i4 (IKind.huge .i4))) from rfl] at hint
      cases hint
      refine ⟨.int (IKind.huge .i4), ?_, RelS.int⟩
      rw [P_eval_call1, hx', Option.bind_some,
        show P.helper o.P .f_huge [.s (.int n)] = some (.s (.int 2147483647)) from rfl]
      simp [IKind.huge, IKind.bits]
    · have hw := hfloat x' hx'
      subst hw
      rw [show F.intrinsic o.F .huge [.s (.real .r8 r)] none = some (.s (.real .r8 (RKind.hugeR .r8))) from rfl] at hint
      cases hint
      refine ⟨.float (RKind.hugeR .r8), ?_, RelS.realF⟩
      rw [P_eval_call1, hx']
      rfl

/-- FNP-D0008: `huge(k8)` with `integer(8) :: k8`. -/
theorem FNP_R0087_refuted : ¬ FNP_R0087_obligation := by
  intro h
  have := h noPrim emptyEnv _ _ (FNP_R0001_sound noPrim emptyEnv .i8 1) emptyF emptyP relState_empty conforms_empty
    (.int .i8 (IKind.huge .i8)) (by rw [F_eval_call1, F_eval_intLit, Option.bind_some]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.intLit 1) = some (.s (.int 1)) from rfl, Option.bind_some,
    show P.helper noPrim.P .f_huge [.s (.int 1)] = some (.s (.int 2147483647)) from rfl] at hw
  cases hw
  cases hrel

/-- FNP-D0039: a REAL(8) variable holding an integer-valued Python `int`
(e.g. after `x = 0`) makes `_f_huge(x)` answer the INTEGER huge. -/
def declX : F.Decl := ⟨.real .r8, [], 0⟩
def envX : F.Env := fun x => if x = "x" then some declX else none
def stateX : F.State := fun x => if x = "x" then some (.s (.real .r8 0)) else none
def pstateX : P.State := fun x => if x = "x" then some (.s (.int 0)) else none

theorem relStateX : RelState envX stateX pstateX := by
  intro x d hx
  simp only [envX] at hx
  split_ifs at hx with hxa
  · subst hxa
    refine ⟨.s (.real .r8 0), .s (.int 0), by simp [stateX], by simp [pstateX], RelV.s ?_⟩
    exact RelS.real (by simp [P.toR])

theorem conformsX : Conforms noPrim.F envX stateX := by
  intro x d hx
  simp only [envX] at hx
  split_ifs at hx with hxa
  · subst hxa; cases hx
    exact ⟨fun _ => ⟨.real .r8 0, by simp [stateX], rfl⟩, fun h => absurd rfl h⟩

theorem FNP_R0087_refuted_realInt : ¬ FNP_R0087_obligation := by
  intro h
  have hx : SoundScalar noPrim envX (.var "x") (.name "x") :=
    (FNP_R0010_sound noPrim envX "x" declX (by simp [envX])).toScalar
  have := h noPrim envX _ _ hx stateX pstateX relStateX conformsX (.real .r8 (RKind.hugeR .r8)) (by
    rw [F_eval_call1]
    rw [show F.eval noPrim.F stateX (.var "x") = some (.s (.real .r8 0)) by simp [F.eval, stateX], Option.bind_some]
    rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P pstateX (.name "x") = some (.s (.int 0)) by simp [P.eval, pstateX], Option.bind_some,
    show P.helper noPrim.P .f_huge [.s (.int 0)] = some (.s (.int 2147483647)) from rfl] at hw
  cases hw
  have := RelS.real_toR hrel
  simp only [P.toR, Option.some.injEq] at this
  -- 2^1023 * (2 - 2^-52) ≠ 2147483647
  have hbig : (2147483647 : ℝ) < RKind.hugeR .r8 := by
    show (2147483647 : ℝ) < (2 - 2 ^ (-(52 : ℤ))) * 2 ^ (1023 : ℤ)
    have h2 : (2 : ℝ) ^ (-(52 : ℤ)) ≤ 1 := by
      rw [zpow_neg]; exact inv_le_one_of_one_le₀ (one_le_zpow₀ (by norm_num) (by norm_num))
    have h3 : (2147483648 : ℝ) ≤ 2 ^ (1023 : ℤ) := by
      have := zpow_le_zpow_right₀ (a := (2:ℝ)) (by norm_num) (show (31:ℤ) ≤ 1023 by norm_num)
      rwa [show (2:ℝ) ^ (31:ℤ) = 2147483648 by norm_num] at this
    calc (2147483647 : ℝ) < 2147483648 := by norm_num
      _ ≤ 2 ^ (1023 : ℤ) := h3
      _ = 1 * 2 ^ (1023 : ℤ) := (one_mul _).symm
      _ ≤ (2 - 2 ^ (-(52 : ℤ))) * 2 ^ (1023 : ℤ) :=
          mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  rw [← this] at hbig
  norm_num at hbig

def FNP_R0088_obligation : Prop := ∀ o Γ, Call1Sound o Γ .precision .f_precision

/-- FNP-D0008: `precision(x4)` with `real(4) :: x4`. -/
theorem FNP_R0088_refuted : ¬ FNP_R0088_obligation := by
  intro h
  have := h noPrim emptyEnv _ _ (realLit_sound .r4 ⟨1, 0⟩) emptyF emptyP relState_empty conforms_empty
    (.int .i4 6) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r4 ⟨1, 0⟩) = some (.s (.real .r4 (Decimal.toReal ⟨1, 0⟩))) from rfl,
        Option.bind_some]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call1] at hw
  rw [show P.eval noPrim.P emptyP (.floatLit ⟨1, 0⟩) = some (.s (.float (Decimal.toReal ⟨1, 0⟩))) from rfl,
    Option.bind_some,
    show P.helper noPrim.P .f_precision [.s (.float (Decimal.toReal ⟨1, 0⟩))] = some (.s (.int 15)) from rfl] at hw
  cases hw
  cases hrel

/-- PRECISION/EPSILON/TINY for REAL(8) are sound (they ignore the argument's value). -/
theorem FNP_R0088_sound_r8 (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr)
    (ha : SoundScalar o Γ a a')
    (hk : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ a = some (.s v) → ∃ r, v = .real .r8 r) :
    SoundScalar o Γ (.call .precision (.cons a .nil) none) (Tr.fold1 .f_precision a') ∧
    SoundScalar o Γ (.call .epsilon (.cons a .nil) none) (Tr.fold1 .f_epsilon a') ∧
    SoundScalar o Γ (.call .tiny (.cons a .nil) none) (Tr.fold1 .f_tiny a') := by
  refine ⟨?_, ?_, ?_⟩ <;> intro σ τ hR hC v hv <;> rw [F_eval_call1] at hv <;>
    simp only [Option.map_none, Option.bind_eq_some_iff] at hv <;>
    obtain ⟨va, hva, hint⟩ := hv
  all_goals
    cases va with
    | arr av => exfalso; first
        | (rw [show F.intrinsic o.F .precision [.arr av] none = none from rfl] at hint; cases hint)
        | (rw [show F.intrinsic o.F .epsilon [.arr av] none = none from rfl] at hint; cases hint)
        | (rw [show F.intrinsic o.F .tiny [.arr av] none = none from rfl] at hint; cases hint)
    | s x =>
      obtain ⟨x', hx', _⟩ := ha σ τ hR hC x hva
      obtain ⟨r, rfl⟩ := hk σ τ hR hC x hva
      rw [P_eval_call1, hx', Option.bind_some]
      first
      | (rw [show F.intrinsic o.F .precision [.s (.real .r8 r)] none = some (.s (.int .i4 15)) from rfl] at hint
         cases hint; exact ⟨.int 15, rfl, RelS.int⟩)
      | (rw [show F.intrinsic o.F .epsilon [.s (.real .r8 r)] none = some (.s (.real .r8 (RKind.epsR .r8))) from rfl] at hint
         cases hint; exact ⟨.float (RKind.epsR .r8), rfl, RelS.realF⟩)
      | (rw [show F.intrinsic o.F .tiny [.s (.real .r8 r)] none = some (.s (.real .r8 (RKind.tinyR .r8))) from rfl] at hint
         cases hint; exact ⟨.float (RKind.tinyR .r8), rfl, RelS.realF⟩)

/-! ### Bit intrinsics -/

def FNP_R0093_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .ishft (.cons a (.cons b .nil)) none) (Tr.call2 .f_ishft a' b')

/-- FNP-D0009: `ishft(1, 31)` in the default kind. -/
theorem FNP_R0093_refuted : ¬ FNP_R0093_obligation := by
  intro h
  have := h noPrim emptyEnv _ _ _ _ (FNP_R0001_sound noPrim emptyEnv .i4 1) (FNP_R0001_sound noPrim emptyEnv .i4 31)
    emptyF emptyP relState_empty conforms_empty (.int .i4 (ishftK 32 1 31))
    (by rw [F_eval_call2, F_eval_intLit, F_eval_intLit, Option.bind_some, Option.bind_some]; rfl)
  obtain ⟨w, hw, hrel⟩ := this
  rw [P_eval_call2] at hw
  rw [show P.eval noPrim.P emptyP (.intLit 1) = some (.s (.int 1)) from rfl,
    show P.eval noPrim.P emptyP (.intLit 31) = some (.s (.int 31)) from rfl, Option.bind_some, Option.bind_some,
    show P.helper noPrim.P .f_ishft [.s (.int 1), .s (.int 31)] = some (.s (.int (ishftUnbounded 1 31))) from rfl] at hw
  cases hw
  cases hrel

/-! ### DIGITS: no spelling; the emitter falls back to a subscript -/

def FNP_R0089_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr), SoundScalar o Γ a a' →
    SoundScalar o Γ (.call .digits (.cons a .nil) none)
      (.index "digits" (.cons (.scalar (.bin .sub a' (.intLit 1))) .nil))

/-- FNP-D0027: `digits(1.0d0)` is defined (53) while `digits[x - 1]` names nothing. -/
theorem FNP_R0089_refuted : ¬ FNP_R0089_obligation := by
  intro h
  have := h noPrim emptyEnv _ _ (realLit_sound .r8 ⟨1, 0⟩) emptyF emptyP relState_empty conforms_empty
    (.int .i4 53) (by
      rw [F_eval_call1]
      rw [show F.eval noPrim.F emptyF (.realLit .r8 ⟨1, 0⟩) = some (.s (.real .r8 (Decimal.toReal ⟨1, 0⟩))) from rfl,
        Option.bind_some]; rfl)
  obtain ⟨w, hw, _⟩ := this
  have : P.eval noPrim.P emptyP (.index "digits" (.cons (.scalar (.bin .sub (.floatLit ⟨1, 0⟩) (.intLit 1))) .nil)) = none := by
    simp [P.eval, emptyP]
  rw [this] at hw
  cases hw

end Recast.Rules
