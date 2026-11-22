import RecastProofs.Rules.Basic
import RecastProofs.Rules.Literals

/-!
# Rules FNP-R0014 .. FNP-R0026: scalar arithmetic

Each rule lemma is *compositional*: it assumes the soundness of the operand
pairs (the induction hypotheses of the expression theorem) and the typing
premises the emitter relies on, and concludes soundness of the operator
node.  The typing premises are stated dynamically (`IntTyped`, `RealTyped`);
`RecastProofs.Rules.Typing` connects them to the static `tyOf` the modelled
emitter consults.

A Fortran REAL operand is related to *any* Python number with the same value
(`RelS.real`), so every real-valued case is proved through `P.toR`: the
lemmas `P_arith_addsubmul_toR`, `P_arith_truediv_toR`, `P_pow_int_toR` and
`P_pow_real_toR` describe Python arithmetic on such numbers.
-/

namespace Recast.Rules

open Recast Rel Exact

/-- Every value the expression takes (in related states) is an integer. -/
def IntTyped (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) → ∃ k n, v = .int k n

/-- Every value the expression takes (in related states) is a real. -/
def RealTyped (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) → ∃ k r, v = .real k r

/-! ### Evaluation unfolding lemmas -/

theorem F_eval_bin (o : F.Oracle) (σ : F.State) (op : F.BinOp) (a b : F.Expr) :
    F.eval o σ (.bin op a b) =
      (F.eval o σ a).bind fun va => (F.eval o σ b).bind fun vb => F.lift2 (F.arith op) va vb := by
  simp only [F.eval]; rfl

theorem P_eval_bin (o : P.Oracle) (τ : P.State) (op : P.BinOp) (a b : P.Expr) :
    P.eval o τ (.bin op a b) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.lift2 (P.arith op) va vb := by
  simp only [P.eval]; rfl

theorem F_lift2_arr_left (f : SVal → SVal → Option SVal) (a : FArr) (v : F.Val) (c : SVal) :
    F.lift2 f (.arr a) v ≠ some (.s c) := by
  intro h
  cases v <;> simp only [F.lift2] at h <;> (repeat' split at h) <;> simp at h

theorem F_lift2_arr_right (f : SVal → SVal → Option SVal) (b : FArr) (a : SVal) (c : SVal) :
    F.lift2 f (.s a) (.arr b) ≠ some (.s c) := by
  intro h
  simp only [F.lift2] at h; (repeat' split at h) <;> simp at h

theorem F_lift2_scalar_eq (f : SVal → SVal → Option SVal) (a b c : SVal)
    (h : F.lift2 f (.s a) (.s b) = some (.s c)) : f a b = some c := by
  simpa [F.lift2] using h

/-- The scalar value of a lifted binary operation forces scalar operands. -/
theorem F_lift2_eq_scalar {f : SVal → SVal → Option SVal} {va vb : F.Val} {c : SVal}
    (h : F.lift2 f va vb = some (.s c)) : ∃ a b, va = .s a ∧ vb = .s b ∧ f a b = some c := by
  cases va with
  | s a =>
    cases vb with
    | s b => exact ⟨a, b, rfl, rfl, F_lift2_scalar_eq f a b c h⟩
    | arr b => exact absurd h (F_lift2_arr_right f b a c)
  | arr a => exact absurd h (F_lift2_arr_left f a vb c)

theorem F_lift1_eq_scalar {f : SVal → Option SVal} {va : F.Val} {c : SVal}
    (h : F.lift1 f va = some (.s c)) : ∃ a, va = .s a ∧ f a = some c := by
  cases va with
  | s a => exact ⟨a, rfl, by simpa [F.lift1] using h⟩
  | arr a =>
    exfalso
    simp only [F.lift1] at h; (repeat' split at h) <;> simp at h

theorem P_lift2_scalar (f : PVal → PVal → Option PVal) (a b : PVal) :
    P.lift2 f (.s a) (.s b) = (f a b).map .s := by simp [P.lift2]

theorem P_eval_paren (o : P.Oracle) (τ : P.State) (e : P.Expr) :
    P.eval o τ (.paren e) = P.eval o τ e := rfl

theorem P_eval_call2 (o : P.Oracle) (τ : P.State) (h : P.Helper) (a b : P.Expr) :
    P.eval o τ (Tr.call2 h a b) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.helper o h [va, vb] := by
  cases h1 : P.eval o τ a <;> cases h2 : P.eval o τ b <;>
    simp only [Tr.call2, P.eval, P.evalExprs, h1, h2] <;> rfl

theorem P_eval_call1 (o : P.Oracle) (τ : P.State) (h : P.Helper) (a : P.Expr) :
    P.eval o τ (Tr.fold1 h a) = (P.eval o τ a).bind fun va => P.helper o h [va] := by
  cases h1 : P.eval o τ a <;> simp only [Tr.fold1, P.eval, P.evalExprs, h1] <;> rfl

theorem P_lift1_scalar (f : PVal → Option PVal) (a : PVal) :
    P.lift1 f (.s a) = (f a).map .s := by simp [P.lift1]

/-- `_f_int_div` on two ints: the body `int(a / b)` (by `rfl` on the helper table). -/
theorem helper_f_int_div (o : P.Oracle) (a b : ℤ) :
    P.helper o .f_int_div [.s (.int a), .s (.int b)] =
      (P.arith .truediv (.int a) (.int b)).bind fun q => match q with
        | .float r => some (.s (.int (truncR r)))
        | _ => none := rfl

/-! ### Python numbers denoting a real -/

theorem P_toR_int (n : ℤ) : P.toR (.int n) = some (n : ℝ) := rfl
theorem P_toR_float (r : ℝ) : P.toR (.float r) = some r := rfl

/-- A Python value denoting the real `r` is an `int` with that value or the `float r`. -/
theorem P_toR_cases {w : PVal} {r : ℝ} (h : P.toR w = some r) :
    (∃ n : ℤ, w = .int n ∧ r = n) ∨ w = .float r := by
  cases w
  case int n => exact Or.inl ⟨n, rfl, (Option.some.inj h).symm⟩
  case float x => exact Or.inr (by rw [Option.some.inj h])
  all_goals simp [P.toR] at h

/-- The real meaning of `+ - *`. -/
def opR : P.BinOp → ℝ → ℝ → ℝ
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | _, a, b => a * b

theorem P_arith_addsubmul_toR (op : P.BinOp) (hop : op = .add ∨ op = .sub ∨ op = .mul)
    {w₁ w₂ : PVal} {a b : ℝ} (h1 : P.toR w₁ = some a) (h2 : P.toR w₂ = some b) :
    ∃ c, P.arith op w₁ w₂ = some c ∧ P.toR c = some (opR op a b) := by
  rcases P_toR_cases h1 with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases h2 with ⟨m, rfl, rfl⟩ | rfl <;>
  rcases hop with rfl | rfl | rfl
  all_goals first
    | exact ⟨.int _, rfl, by simp [P.toR, opR]⟩
    | exact ⟨.float _, rfl, rfl⟩
    | exact ⟨.float _, by simp [P.arith, P.toR], by simp [P.toR, opR]⟩

theorem P_arith_truediv_toR {w₁ w₂ : PVal} {a b : ℝ} (h1 : P.toR w₁ = some a) (h2 : P.toR w₂ = some b)
    (hb : b ≠ 0) : P.arith .truediv w₁ w₂ = some (.float (a / b)) := by
  rcases P_toR_cases h1 with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases h2 with ⟨m, rfl, rfl⟩ | rfl
  · have hm : m ≠ 0 := by exact_mod_cast hb
    simp [P.arith, hm]
  all_goals simp [P.arith, P.toR, hb]

/-- Python `**` with an `int` exponent on a number denoting `r`, wherever Fortran's
`zpowF` is defined. -/
theorem P_pow_int_toR {w : PVal} {r : ℝ} (h : P.toR w = some r) (n : ℤ) (hdef : ¬ (r = 0 ∧ n ≤ 0)) :
    ∃ c, P.arith .pow w (.int n) = some c ∧ P.toR c = some (r ^ n) := by
  rcases P_toR_cases h with ⟨m, rfl, rfl⟩ | rfl
  · by_cases hn : 0 ≤ n
    · refine ⟨.int (m ^ n.toNat), by simp [P.arith, P.pyPow, hn], ?_⟩
      obtain ⟨k, rfl⟩ : ∃ k : ℕ, n = k := ⟨n.toNat, (Int.toNat_of_nonneg hn).symm⟩
      simp [P.toR, zpow_natCast]
    · have hm : m ≠ 0 := by
        rintro rfl
        exact hdef ⟨by simp, by omega⟩
      exact ⟨.float ((m : ℝ) ^ n), by simp [P.arith, P.pyPow, hn, hm], rfl⟩
  · have hno : ¬ (r = 0 ∧ n < 0) := fun ⟨h0, hneg⟩ => hdef ⟨h0, hneg.le⟩
    exact ⟨.float (r ^ n), by simp [P.arith, P.toR, P.pyPow, hno], rfl⟩

/-- Python `**` on two numbers denoting reals agrees with Fortran's real power
wherever the latter is defined. -/
theorem P_pow_real_toR {w₁ w₂ : PVal} {a b y : ℝ} (h1 : P.toR w₁ = some a) (h2 : P.toR w₂ = some b)
    (hy : F.rpowF a b = some y) : ∃ c, P.arith .pow w₁ w₂ = some c ∧ P.toR c = some y := by
  unfold F.rpowF at hy
  split_ifs at hy with hpos hzero
  · simp only [Option.some.injEq] at hy; subst hy
    rcases P_toR_cases h1 with ⟨m, rfl, rfl⟩ | rfl <;> rcases P_toR_cases h2 with ⟨n, rfl, rfl⟩ | rfl
    · by_cases hn : 0 ≤ n
      · refine ⟨.int (m ^ n.toNat), by simp [P.arith, P.pyPow, hn], ?_⟩
        obtain ⟨k, rfl⟩ : ∃ k : ℕ, n = k := ⟨n.toNat, (Int.toNat_of_nonneg hn).symm⟩
        simp [P.toR, Real.rpow_eq_pow, Real.rpow_natCast]
      · have hm : m ≠ 0 := by rintro rfl; simp at hpos
        refine ⟨.float ((m : ℝ) ^ n), by simp [P.arith, P.pyPow, hn, hm], ?_⟩
        simp [P.toR, Real.rpow_eq_pow, Real.rpow_intCast]
    · exact ⟨.float (Real.rpow m b), by simp [P.arith, P.pyPow, P.pyPow.pyPowR, hpos], rfl⟩
    · have hno : ¬ (a = 0 ∧ n < 0) := fun h => by rw [h.1] at hpos; exact lt_irrefl _ hpos
      refine ⟨.float (a ^ n), by simp [P.arith, P.pyPow, hno], ?_⟩
      simp [P.toR, Real.rpow_eq_pow, Real.rpow_intCast]
    · exact ⟨.float (Real.rpow a b), by simp [P.arith, P.pyPow, P.pyPow.pyPowR, hpos], rfl⟩
  · simp only [Option.some.injEq] at hy; subst hy
    obtain ⟨rfl, hb⟩ := hzero
    rcases P_toR_cases h1 with ⟨m, rfl, hm0⟩ | rfl <;> rcases P_toR_cases h2 with ⟨n, rfl, rfl⟩ | rfl
    · have hm : m = 0 := by exact_mod_cast hm0.symm
      subst hm
      have hn : 0 < n := by exact_mod_cast hb
      refine ⟨.int (0 ^ n.toNat), by simp [P.arith, P.pyPow, hn.le], ?_⟩
      have hk : n.toNat ≠ 0 := by omega
      simp [P.toR, zero_pow hk]
    · have hm : m = 0 := by exact_mod_cast hm0.symm
      subst hm
      exact ⟨.float 0, by simp [P.arith, P.toR, P.pyPow, P.pyPow.pyPowR, hb], rfl⟩
    · have hn : 0 < n := by exact_mod_cast hb
      refine ⟨.float ((0:ℝ) ^ n), by simp [P.arith, P.pyPow, hn.le], ?_⟩
      simp [P.toR, zero_zpow n (ne_of_gt hn)]
    · exact ⟨.float 0, by simp [P.arith, P.toR, P.pyPow, P.pyPow.pyPowR, hb], rfl⟩

/-! ### FNP-R0014: unary minus -/

def FNP_R0014_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr),
    SoundScalar o Γ e e' → SoundScalar o Γ (.neg e) (.paren (.neg e'))

theorem FNP_R0014_sound : FNP_R0014_obligation := by
  intro o Γ e e' he σ τ hR hC v hv
  simp only [F.eval, Option.bind_eq_some_iff] at hv
  obtain ⟨u, hu, hlift⟩ := hv
  obtain ⟨a, rfl, hfa⟩ := F_lift1_eq_scalar hlift
  obtain ⟨w, hw, hrel⟩ := he σ τ hR hC a hu
  cases a with
  | int k n =>
    rw [RelS.int_eq hrel] at hw
    simp only [Option.some.injEq] at hfa; subst hfa
    refine ⟨.int (-n), ?_, RelS.int⟩
    simp [P.eval, hw, P.lift1]
  | real k r =>
    have ht := RelS.real_toR hrel
    simp only [Option.some.injEq] at hfa; subst hfa
    rcases P_toR_cases ht with ⟨n, rfl, rfl⟩ | rfl
    · refine ⟨.int (-n), ?_, RelS.real (by simp [P.toR])⟩
      simp [P.eval, hw, P.lift1]
    · refine ⟨.float (-r), ?_, RelS.realF⟩
      simp [P.eval, hw, P.lift1]
  | bool _ => simp at hfa
  | str _ => simp at hfa

/-! ### FNP-R0015: parentheses -/

def FNP_R0015_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr),
    SoundPair o Γ e e' → SoundPair o Γ (.paren e) (.paren e')

theorem FNP_R0015_sound : FNP_R0015_obligation := by
  intro o Γ e e' he σ τ hR hC v hv
  simp only [F.eval] at hv
  obtain ⟨w, hw, hrel⟩ := he σ τ hR hC v hv
  exact ⟨w, by simp [P.eval, hw], hrel⟩

/-! ### FNP-R0016: `+ - *` on scalars (any int/real mix) -/

def toP : F.BinOp → P.BinOp
  | .add => .add | .sub => .sub | .mul => .mul | .div => .truediv | .pow => .pow

theorem arith_addsubmul_rel {op : F.BinOp} (hop : op = .add ∨ op = .sub ∨ op = .mul)
    {a b : SVal} {a' b' : PVal} (ha : RelS a a') (hb : RelS b b') {c : SVal}
    (hc : F.arith op a b = some c) : ∃ c', P.arith (toP op) a' b' = some c' ∧ RelS c c' := by
  have hopP : toP op = .add ∨ toP op = .sub ∨ toP op = .mul := by
    rcases hop with rfl | rfl | rfl <;> simp [toP]
  cases a with
  | int k₁ x =>
    rw [RelS.int_eq ha]
    cases b with
    | int k₂ y =>
      rw [RelS.int_eq hb]
      rcases hop with rfl | rfl | rfl <;> simp [F.arith] at hc <;> subst hc <;> exact ⟨_, rfl, RelS.int⟩
    | real k₂ y =>
      have ht := RelS.real_toR hb
      rcases hop with rfl | rfl | rfl <;> simp [F.arith] at hc <;> subst hc <;>
        exact (P_arith_addsubmul_toR _ hopP (P_toR_int x) ht).imp fun c' ⟨hc', hr'⟩ => ⟨hc', RelS.real hr'⟩
    | bool _ => simp [F.arith] at hc
    | str _ => simp [F.arith] at hc
  | real k₁ x =>
    have ht₁ := RelS.real_toR ha
    cases b with
    | int k₂ y =>
      rw [RelS.int_eq hb]
      rcases hop with rfl | rfl | rfl <;> simp [F.arith] at hc <;> subst hc <;>
        exact (P_arith_addsubmul_toR _ hopP ht₁ (P_toR_int y)).imp fun c' ⟨hc', hr'⟩ => ⟨hc', RelS.real hr'⟩
    | real k₂ y =>
      have ht₂ := RelS.real_toR hb
      rcases hop with rfl | rfl | rfl <;> simp [F.arith] at hc <;> subst hc <;>
        exact (P_arith_addsubmul_toR _ hopP ht₁ ht₂).imp fun c' ⟨hc', hr'⟩ => ⟨hc', RelS.real hr'⟩
    | bool _ => simp [F.arith] at hc
    | str _ => simp [F.arith] at hc
  | bool _ => cases b <;> simp [F.arith] at hc
  | str _ => cases b <;> simp [F.arith] at hc

/-- Generic composition: a binary node is sound when the scalar operation is
related pointwise. -/
theorem bin_sound_of_rel (o : Oracles) (Γ : F.Env) (op : F.BinOp) (op' : P.BinOp)
    (a b : F.Expr) (a' b' : P.Expr) (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b')
    (hrel : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ x y x' y' c, F.eval o.F σ a = some (.s x) →
      F.eval o.F σ b = some (.s y) → RelS x x' → RelS y y' → F.arith op x y = some c →
      ∃ c', P.arith op' x' y' = some c' ∧ RelS c c') :
    SoundScalar o Γ (.bin op a b) (.paren (.bin op' a' b')) := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb
  obtain ⟨c', hc', hrc⟩ := hrel σ τ hR hC x y x' y' v hva hvb hrx hry hxy
  refine ⟨c', ?_, hrc⟩
  rw [P_eval_paren, P_eval_bin, hx', hy']
  simp [P.lift2, hc']

def FNP_R0016_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (op : F.BinOp) (a b : F.Expr) (a' b' : P.Expr),
    (op = .add ∨ op = .sub ∨ op = .mul) →
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.bin op a b) (.paren (.bin (toP op) a' b'))

theorem FNP_R0016_sound : FNP_R0016_obligation := by
  intro o Γ op a b a' b' hop ha hb
  exact bin_sound_of_rel o Γ op (toP op) a b a' b' ha hb
    (fun _ _ _ _ _ _ _ _ _ _ _ hrx hry hxy => arith_addsubmul_rel hop hrx hry hxy)

/-! ### FNP-R0017: integer `/` -> `_f_int_div(a, b)` -/

def FNP_R0017_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → IntTyped o Γ a → IntTyped o Γ b →
    SoundScalar o Γ (.bin .div a b) (Tr.call2 .f_int_div a' b')

theorem FNP_R0017_sound : FNP_R0017_obligation := by
  intro o Γ a b a' b' ha hb hia hib σ τ hR hC v hv
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
    rw [P_eval_call2, hx', hy', Option.bind_some, Option.bind_some, helper_f_int_div]
    simp [P.arith, hny, truncR_div_intCast nx ny hny]

/-! ### FNP-R0018: real `/` -> `(a / b)` when an operand is real -/

theorem arith_div_real_rel {a b : SVal} {a' b' : PVal} (ha : RelS a a') (hb : RelS b b')
    (hreal : (∃ k r, a = .real k r) ∨ (∃ k r, b = .real k r)) {c : SVal}
    (hc : F.arith .div a b = some c) : ∃ c', P.arith .truediv a' b' = some c' ∧ RelS c c' := by
  cases a with
  | int k₁ x =>
    rw [RelS.int_eq ha]
    cases b with
    | int k₂ y => exfalso; rcases hreal with ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> cases h
    | real k₂ y =>
      have ht := RelS.real_toR hb
      simp only [F.arith] at hc
      split_ifs at hc with hz
      cases hc
      exact ⟨_, P_arith_truediv_toR (P_toR_int x) ht hz, RelS.realF⟩
    | bool _ => simp [F.arith] at hc
    | str _ => simp [F.arith] at hc
  | real k₁ x =>
    have ht₁ := RelS.real_toR ha
    cases b with
    | int k₂ y =>
      rw [RelS.int_eq hb]
      simp only [F.arith] at hc
      split_ifs at hc with hz
      cases hc
      exact ⟨_, P_arith_truediv_toR ht₁ (P_toR_int y) (by exact_mod_cast hz), RelS.realF⟩
    | real k₂ y =>
      have ht₂ := RelS.real_toR hb
      simp only [F.arith] at hc
      split_ifs at hc with hz
      cases hc
      exact ⟨_, P_arith_truediv_toR ht₁ ht₂ hz, RelS.realF⟩
    | bool _ => simp [F.arith] at hc
    | str _ => simp [F.arith] at hc
  | bool _ => cases b <;> simp [F.arith] at hc
  | str _ => cases b <;> simp [F.arith] at hc

def FNP_R0018_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → (RealTyped o Γ a ∨ RealTyped o Γ b) →
    SoundScalar o Γ (.bin .div a b) (.paren (.bin .truediv a' b'))

theorem FNP_R0018_sound : FNP_R0018_obligation := by
  intro o Γ a b a' b' ha hb hreal
  refine bin_sound_of_rel o Γ .div .truediv a b a' b' ha hb ?_
  intro σ τ hR hC x y x' y' c hx hy hrx hry hxy
  refine arith_div_real_rel hrx hry ?_ hxy
  rcases hreal with h | h
  · exact Or.inl (h σ τ hR hC x hx)
  · exact Or.inr (h σ τ hR hC y hy)

/-! ### FNP-R0019: `x ** n`, literal integer `n ≠ 0`, real base, no-expand profile -/

theorem F_eval_intLit (o : F.Oracle) (σ : F.State) (k : IKind) (n : ℤ) :
    F.eval o σ (.intLit k n) = some (.s (.int k n)) := by simp [F.eval]

def FNP_R0019_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (n : ℤ), n ≠ 0 →
    SoundScalar o Γ a a' → RealTyped o Γ a →
    ∀ (k : IKind), SoundScalar o Γ (.bin .pow a (.intLit k n)) (.paren (.bin .pow a' (.intLit n)))

theorem FNP_R0019_sound : FNP_R0019_obligation := by
  intro o Γ a a' n hn ha hreal k
  refine bin_sound_of_rel o Γ .pow .pow a (.intLit k n) a' (.intLit n) ha (FNP_R0001_sound o Γ k n) ?_
  intro σ τ hR hC x y x' y' c hx hy hrx hry hxy
  obtain ⟨kx, r, rfl⟩ := hreal σ τ hR hC x hx
  rw [F_eval_intLit] at hy
  simp only [Option.some.injEq, F.Val.s.injEq] at hy
  subst hy
  have ht := RelS.real_toR hrx
  rw [RelS.int_eq hry]
  simp only [F.arith, F.zpowF] at hxy
  split at hxy
  · simp at hxy
  · rename_i hdef
    simp only [Option.map_some, Option.some.injEq] at hxy
    subst hxy
    obtain ⟨c', hc', hr'⟩ := P_pow_int_toR ht n hdef
    exact ⟨c', hc', RelS.real hr'⟩

/-! ### FNP-R0020: `i ** n`, literal integer `n`, integer base -/

/-- Sound for a positive literal exponent. -/
def FNP_R0020_obligation_pos : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (n : ℤ), 0 < n →
    SoundScalar o Γ a a' → IntTyped o Γ a →
    ∀ (k : IKind), SoundScalar o Γ (.bin .pow a (.intLit k n)) (.paren (.bin .pow a' (.intLit n)))

theorem FNP_R0020_sound_pos : FNP_R0020_obligation_pos := by
  intro o Γ a a' n hn ha hint k
  refine bin_sound_of_rel o Γ .pow .pow a (.intLit k n) a' (.intLit n) ha (FNP_R0001_sound o Γ k n) ?_
  intro σ τ hR hC x y x' y' c hx hy hrx hry hxy
  obtain ⟨kx, m, rfl⟩ := hint σ τ hR hC x hx
  rw [F_eval_intLit] at hy
  simp only [Option.some.injEq, F.Val.s.injEq] at hy
  subst hy
  rw [RelS.int_eq hrx, RelS.int_eq hry]
  simp only [F.arith] at hxy
  rw [if_pos hn.le] at hxy
  simp only [Option.some.injEq] at hxy
  subst hxy
  exact ⟨.int (m ^ n.toNat), by simp [P.arith, P.pyPow, hn.le], RelS.int⟩

/-- The general obligation (every non-zero literal exponent) -- refuted by `2 ** (-1)`. -/
def FNP_R0020_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (n : ℤ), n ≠ 0 →
    SoundScalar o Γ a a' → IntTyped o Γ a →
    ∀ (k : IKind), SoundScalar o Γ (.bin .pow a (.intLit k n)) (.paren (.bin .pow a' (.intLit n)))

def noPrim : Oracles := { prim := fun _ _ => none }

theorem intLit_intTyped (o : Oracles) (Γ : F.Env) (k : IKind) (n : ℤ) : IntTyped o Γ (.intLit k n) := by
  intro σ τ _ _ v hv; simp [F.eval] at hv; exact ⟨_, _, hv.symm⟩

/-- FNP-D0012: `k = 2**(-1)` is `0` in Fortran and `0.5` in the emitted Python. -/
theorem FNP_R0020_refuted : ¬ FNP_R0020_obligation := by
  intro h
  have := h noPrim emptyEnv (.intLit .i4 2) (.intLit 2) (-1) (by norm_num)
    (FNP_R0001_sound _ _ _ _) (intLit_intTyped _ _ .i4 2) .i4 emptyF emptyP relState_empty conforms_empty (.int .i4 0)
    (by simp [F.eval, F.lift2, F.arith, F.ipowNeg, tquot, IKind.join])
  obtain ⟨w, hw, hrel⟩ := this
  simp [P.eval, P.lift2, P.arith, P.pyPow] at hw
  subst hw
  exact RelS.int_ne_float hrel

/-! ### FNP-R0021: gfortran profile, literal exponent expanded to multiplications -/

/-- `e` evaluates to a Python number denoting the real `x`. -/
def EvalsToR (o : P.Oracle) (τ : P.State) (e : P.Expr) (x : ℝ) : Prop :=
  ∃ w, P.eval o τ e = some (.s w) ∧ P.toR w = some x

/-- One step of `expand_power`: multiply the accumulator in when the low bit is set. -/
def nextResult (remaining : ℕ) (result : Option P.Expr) (square : P.Expr) : Option P.Expr :=
  if remaining % 2 = 1 then
    (match result with | none => some square | some r => some (.paren (.bin .mul r square)))
  else result

/-- One step of `expand_power`: square unless no bits remain. -/
def nextSquare (remaining : ℕ) (square : P.Expr) : P.Expr :=
  if remaining / 2 ≠ 0 then P.Expr.paren (.bin .mul square square) else square

/-- Square-and-multiply, LSB first (`expand_power`), as a well-founded
recursion so the value can be reasoned about. -/
def goPow (remaining : ℕ) (result : Option P.Expr) (square : P.Expr) : Option P.Expr :=
  if h : remaining = 0 then result
  else goPow (remaining / 2) (nextResult remaining result square) (nextSquare remaining square)
termination_by remaining
decreasing_by
  simp_wf; omega

theorem goPow_zero (result : Option P.Expr) (square : P.Expr) : goPow 0 result square = result := by
  unfold goPow; rw [dif_pos rfl]

theorem goPow_succ (remaining : ℕ) (h0 : remaining ≠ 0) (result : Option P.Expr) (square : P.Expr) :
    goPow remaining result square =
      goPow (remaining / 2) (nextResult remaining result square) (nextSquare remaining square) := by
  conv_lhs => unfold goPow
  rw [dif_neg h0]

/-- The emitted expression for `base ** n` under `int_pow_expand`
(`expressions.py:1692-1708`): `(1.0 / <product>)` for a negative exponent. -/
def expandPowerWF (base : P.Expr) (n : ℤ) : P.Expr :=
  let body := (goPow n.natAbs none base).getD base
  if n < 0 then .paren (.bin .truediv (.floatLit ⟨10, -1⟩) body) else body

theorem P_eval_mul_toR (o : P.Oracle) (τ : P.State) (a b : P.Expr) {x y : ℝ}
    (ha : EvalsToR o τ a x) (hb : EvalsToR o τ b y) :
    EvalsToR o τ (.paren (.bin .mul a b)) (x * y) := by
  obtain ⟨wa, hwa, hxa⟩ := ha
  obtain ⟨wb, hwb, hxb⟩ := hb
  obtain ⟨c, hc, hr⟩ := P_arith_addsubmul_toR .mul (Or.inr (Or.inr rfl)) hxa hxb
  refine ⟨c, ?_, hr⟩
  rw [P_eval_paren, P_eval_bin, hwa, hwb]
  simp [P.lift2, hc]

/-- Value of the accumulated product: `result` (or 1) times `square^remaining`.
The accumulator is non-empty at the end whenever it is non-empty at the start
or some iteration remains. -/
theorem goPow_value (o : P.Oracle) (τ : P.State) :
    ∀ (remaining : ℕ) (result : Option P.Expr) (square : P.Expr) (rv s : ℝ),
      (match result with
       | none => rv = 1
       | some r => EvalsToR o τ r rv) →
      EvalsToR o τ square s →
      (remaining = 0 → result.isSome) →
      ∃ e, goPow remaining result square = some e ∧ EvalsToR o τ e (rv * s ^ remaining) := by
  intro remaining
  induction remaining using Nat.strong_induction_on with
  | _ remaining ih =>
    intro result square rv s hres hsq hsome
    by_cases h0 : remaining = 0
    · subst h0
      cases result with
      | none => simp at hsome
      | some r =>
        refine ⟨r, goPow_zero _ _, ?_⟩
        simp only [pow_zero, mul_one]; exact hres
    · have hlt : remaining / 2 < remaining := Nat.div_lt_self (Nat.pos_of_ne_zero h0) (by norm_num)
      have hres' : (match nextResult remaining result square with
          | none => (if remaining % 2 = 1 then rv * s else rv) = 1
          | some r => EvalsToR o τ r (if remaining % 2 = 1 then rv * s else rv)) := by
        unfold nextResult
        split_ifs with hodd
        · cases result with
          | none => simp only [hres]; simpa using hsq
          | some r => exact P_eval_mul_toR o τ r square hres hsq
        · exact hres
      have hsq' : EvalsToR o τ (nextSquare remaining square) (if remaining / 2 ≠ 0 then s * s else s) := by
        unfold nextSquare
        split_ifs with hnz
        · exact P_eval_mul_toR o τ square square hsq hsq
        · exact hsq
      have hsome' : remaining / 2 = 0 → (nextResult remaining result square).isSome := by
        intro hz
        unfold nextResult
        have hodd : remaining % 2 = 1 := by omega
        rw [if_pos hodd]
        cases result <;> simp
      obtain ⟨e, he, hval⟩ := ih (remaining / 2) hlt (nextResult remaining result square)
        (nextSquare remaining square) _ _ hres' hsq' hsome'
      refine ⟨e, ?_, ?_⟩
      · rw [goPow_succ remaining h0]; exact he
      · have key : (if remaining % 2 = 1 then rv * s else rv) *
            (if remaining / 2 ≠ 0 then s * s else s) ^ (remaining / 2) = rv * s ^ remaining := by
          obtain ⟨q, hq⟩ : ∃ q, remaining / 2 = q := ⟨_, rfl⟩
          rw [hq]
          by_cases hz : q = 0
          · have hone : remaining = 1 := by omega
            simp [hz, hone]
          · have hz' : q ≠ 0 := hz
            simp only [hz', ne_eq, not_false_eq_true, if_true]
            split_ifs with hodd
            · rw [show remaining = 2 * q + 1 by omega, pow_succ, pow_mul]
              ring
            · rw [show remaining = 2 * q by omega, pow_mul]
              ring
        rw [key] at hval
        exact hval

/-- `expandPowerWF base n` evaluates to `x ^ n` when `base` evaluates to a
number denoting `x` (for `n < 0`, when `x ≠ 0`). -/
theorem expandPowerWF_value (o : P.Oracle) (τ : P.State) (base : P.Expr) (x : ℝ) (n : ℤ)
    (hn : n ≠ 0) (hbase : EvalsToR o τ base x) (hx : n < 0 → x ≠ 0) :
    EvalsToR o τ (expandPowerWF base n) (x ^ n) := by
  have hpos : 0 < n.natAbs := Int.natAbs_pos.mpr hn
  obtain ⟨e, he, hv⟩ := goPow_value o τ n.natAbs none base 1 x rfl hbase
    (fun h => absurd h (Nat.pos_iff_ne_zero.mp hpos))
  simp only [one_mul] at hv
  unfold expandPowerWF
  simp only [he, Option.getD_some]
  split_ifs with hneg
  · have hxne := hx hneg
    have h1 : n = -(n.natAbs : ℤ) := by
      have := Int.ofNat_natAbs_of_nonpos hneg.le
      omega
    have hpow : x ^ n = (x ^ n.natAbs)⁻¹ := by
      have h2 : x ^ n = x ^ (-(n.natAbs : ℤ)) := by rw [← h1]
      rw [h2, zpow_neg, zpow_natCast]
    rw [hpow]
    have hnz : x ^ n.natAbs ≠ 0 := pow_ne_zero _ hxne
    have hone : Decimal.toReal ⟨10, -1⟩ = (1 : ℝ) := by simp [Decimal.toReal]
    obtain ⟨w, hw, hxw⟩ := hv
    refine ⟨.float (1 / x ^ n.natAbs), ?_, by simp [P.toR]⟩
    rw [P_eval_paren, P_eval_bin, hw]
    simp only [P.eval, hone, Option.bind_some, P_lift2_scalar]
    rw [P_arith_truediv_toR (P_toR_float 1) hxw hnz]
    rfl
  · have h1 : n = (n.natAbs : ℤ) := (Int.natAbs_of_nonneg (not_lt.mp hneg)).symm
    have h2 : x ^ n = x ^ (n.natAbs : ℤ) := by rw [← h1]
    rw [h2, zpow_natCast]
    exact hv

def FNP_R0021_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (n : ℤ), n ≠ 0 → n.natAbs ≤ 16 →
    SoundScalar o Γ a a' → RealTyped o Γ a →
    ∀ (k : IKind), SoundScalar o Γ (.bin .pow a (.intLit k n)) (expandPowerWF a' n)

theorem FNP_R0021_sound : FNP_R0021_obligation := by
  intro o Γ a a' n hn _ ha hreal k σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨kx, r, rfl⟩ := hreal σ τ hR hC x hva
  rw [F_eval_intLit] at hvb
  simp only [Option.some.injEq, F.Val.s.injEq] at hvb
  subst hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  have ht := RelS.real_toR hrx
  simp only [F.arith, F.zpowF] at hxy
  split at hxy
  · simp at hxy
  · rename_i hdef
    simp only [Option.map_some, Option.some.injEq] at hxy
    subst hxy
    obtain ⟨w, hw, hr⟩ := expandPowerWF_value o.P τ a' r n hn ⟨x', hx', ht⟩ (fun hneg h0 => hdef ⟨h0, hneg.le⟩)
    exact ⟨w, hw, RelS.real hr⟩

/-! ### FNP-R0023: non-literal exponent, scalar, real base -/

theorem arith_pow_real_rel {a b : SVal} {a' b' : PVal} (ha : RelS a a') (hb : RelS b b')
    (hreal : ∃ k r, a = .real k r) {c : SVal}
    (hc : F.arith .pow a b = some c) : ∃ c', P.arith .pow a' b' = some c' ∧ RelS c c' := by
  obtain ⟨k, r, rfl⟩ := hreal
  have ht₁ := RelS.real_toR ha
  cases b with
  | int k₂ n =>
    rw [RelS.int_eq hb]
    simp only [F.arith, F.zpowF] at hc
    split at hc
    · simp at hc
    · rename_i hdef
      simp only [Option.map_some, Option.some.injEq] at hc
      subst hc
      obtain ⟨c', hc', hr'⟩ := P_pow_int_toR ht₁ n hdef
      exact ⟨c', hc', RelS.real hr'⟩
  | real k₂ s =>
    have ht₂ := RelS.real_toR hb
    simp only [F.arith] at hc
    cases hy : F.rpowF r s with
    | none => rw [hy] at hc; simp at hc
    | some y =>
      rw [hy] at hc
      simp only [Option.map_some, Option.some.injEq] at hc
      subst hc
      obtain ⟨c', hc', hr'⟩ := P_pow_real_toR ht₁ ht₂ hy
      exact ⟨c', hc', RelS.real hr'⟩
  | bool _ => simp [F.arith] at hc
  | str _ => simp [F.arith] at hc

def FNP_R0023_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → RealTyped o Γ a →
    SoundScalar o Γ (.bin .pow a b) (.paren (.bin .pow a' b'))

theorem FNP_R0023_sound : FNP_R0023_obligation := by
  intro o Γ a b a' b' ha hb hreal
  refine bin_sound_of_rel o Γ .pow .pow a b a' b' ha hb ?_
  intro σ τ hR hC x y x' y' c hx _ hrx hry hxy
  exact arith_pow_real_rel hrx hry (hreal σ τ hR hC x hx) hxy

/-! ### FNP-R0024: non-literal exponent, integer base -- refuted -/

def FNP_R0024_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → IntTyped o Γ a → IntTyped o Γ b →
    SoundScalar o Γ (.bin .pow a b) (.paren (.bin .pow a' b'))

/-- FNP-D0012 (variable exponent): `j ** (-1)` with `j = 2`. -/
theorem FNP_R0024_refuted : ¬ FNP_R0024_obligation := by
  intro h
  have := h noPrim emptyEnv (.intLit .i4 2) (.intLit .i4 (-1)) (.intLit 2) (.intLit (-1))
    (FNP_R0001_sound _ _ _ _) (FNP_R0001_sound _ _ _ _) (intLit_intTyped _ _ .i4 2) (intLit_intTyped _ _ .i4 (-1))
    emptyF emptyP relState_empty conforms_empty (.int .i4 0)
    (by simp [F.eval, F.lift2, F.arith, F.ipowNeg, tquot, IKind.join])
  obtain ⟨w, hw, hrel⟩ := this
  simp [P.eval, P.lift2, P.arith, P.pyPow] at hw
  subst hw
  exact RelS.int_ne_float hrel

end Recast.Rules
