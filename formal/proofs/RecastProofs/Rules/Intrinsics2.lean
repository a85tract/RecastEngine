import RecastProofs.Rules.Intrinsics

/-!
# Rules FNP-R0075 (ATAN2), FNP-R0082 (MERGE) and FNP-R0090 (IAND/IOR/IEOR)
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### FNP-R0075: `atan2(y, x)` -> `math.atan2(y', x')` (shared primitive) -/

def FNP_R0075_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .atan2 (.cons a (.cons b .nil)) none) (Tr.call2 .math_atan2 a' b')

theorem FNP_R0075_sound : FNP_R0075_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .atan2 .math_atan2 a b a' b' ha hb ?_
    (fun va vb c h => intrinsic2_arr_ne o.F .atan2 va vb c h (by simp))
  intro x y x' y' c hrx hry hF
  cases x with
  | real k₁ p =>
    have ht₁ := RelS.real_toR hrx
    cases y with
    | real k₂ q =>
      have ht₂ := RelS.real_toR hry
      have e : F.intrinsic o.F .atan2 [.s (.real k₁ p), .s (.real k₂ q)] none =
          ((o.prim "atan2" [p, q]).map (SVal.real (RKind.join k₁ k₂))).map F.Val.s := rfl
      rw [e, Option.map_map] at hF
      simp only [Option.map_eq_some_iff, Function.comp] at hF
      obtain ⟨r, hr, hc⟩ := hF
      cases hc
      refine ⟨.float r, ?_, RelS.realF⟩
      have e2 : P.helper o.P .math_atan2 [.s x', .s y'] = (P.toR x').bind fun p => (P.toR y').bind fun q =>
          (o.prim "atan2" [p, q]).map fun r => P.Val.s (.float r) := rfl
      rw [e2, ht₁, ht₂, Option.bind_some, Option.bind_some, hr]
      rfl
    | int _ _ => simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF
    | bool _ => simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF
    | str _ => simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF
  | int _ _ => cases y <;> simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF
  | bool _ => cases y <;> simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF
  | str _ => cases y <;> simp [F.intrinsic, F.intrinsicCore, F.lift2] at hF

/-! ### FNP-R0082: scalar `merge(t, f, m)` -> `(t') if (m') else (f')` -/

theorem F_eval_call3 (o : F.Oracle) (σ : F.State) (f : F.Intr) (a b c : F.Expr) (k : Option IKind) :
    F.eval o σ (.call f (.cons a (.cons b (.cons c .nil))) k) = (F.eval o σ a).bind fun va =>
      (F.eval o σ b).bind fun vb => (F.eval o σ c).bind fun vc =>
      F.intrinsic o f [va, vb, vc] (k.map fun kk => match kk with | .i4 => (4 : ℤ) | .i8 => 8) := by
  cases h1 : F.eval o σ a <;> cases h2 : F.eval o σ b <;> cases h3 : F.eval o σ c <;>
    simp only [F.eval, F.evalExprs, h1, h2, h3] <;> rfl

theorem P_eval_ifexp (o : P.Oracle) (τ : P.State) (t c e : P.Expr) :
    P.eval o τ (.ifexp t c e) = (P.eval o τ c).bind fun vc => match vc with
      | .s (.bool true) => P.eval o τ t
      | .s (.bool false) => P.eval o τ e
      | _ => none := by
  cases h : P.eval o τ c <;> simp only [P.eval, h] <;> rfl

theorem F_intrinsic_merge (o : F.Oracle) (vt ve vm : F.Val) :
    F.intrinsic o .merge [vt, ve, vm] none = match vt, ve, vm with
      | .s tv, .s fv, .s (.bool b) => some (.s (if b then tv else fv))
      | _, _, _ => none := rfl

def FNP_R0082_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (t e m : F.Expr) (t' e' m' : P.Expr),
    SoundScalar o Γ t t' → SoundScalar o Γ e e' → SoundScalar o Γ m m' →
    SoundScalar o Γ (.call .merge (.cons t (.cons e (.cons m .nil))) none)
      (.paren (.ifexp (.paren t') (.paren m') (.paren e')))

theorem FNP_R0082_sound : FNP_R0082_obligation := by
  intro o Γ t e m t' e' m' ht he hm σ τ hR hC v hv
  rw [F_eval_call3] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨vt, hvt, ve, hve, vm, hvm, hint⟩ := hv
  rw [F_intrinsic_merge] at hint
  cases vt with
  | arr _ => simp at hint
  | s tv =>
    cases ve with
    | arr _ => simp at hint
    | s fv =>
      cases vm with
      | arr _ => simp at hint
      | s mv =>
        cases mv with
        | bool b =>
          simp only [Option.some.injEq, F.Val.s.injEq] at hint
          subst hint
          obtain ⟨w, hw, hrw⟩ := hm σ τ hR hC _ hvm
          rw [RelS.bool_eq hrw] at hw
          rw [P_eval_paren, P_eval_ifexp, P_eval_paren, hw]
          cases b
          · obtain ⟨w', hw', hrw'⟩ := he σ τ hR hC _ hve
            exact ⟨w', by simp [P_eval_paren, hw'], by simpa using hrw'⟩
          · obtain ⟨w', hw', hrw'⟩ := ht σ τ hR hC _ hvt
            exact ⟨w', by simp [P_eval_paren, hw'], by simpa using hrw'⟩
        | int _ _ => simp at hint
        | real _ _ => simp at hint
        | str _ => simp at hint

/-! ### FNP-R0090: `iand/ior/ieor` on non-negative in-range operands -/

theorem wrapSigned_of_lt (bits : ℕ) (v : ℤ) (h0 : 0 ≤ v) (hv : v < 2 ^ (bits - 1)) (hb : 1 ≤ bits) :
    wrapSigned bits v = v := by
  unfold wrapSigned
  have hm : (2 : ℤ) ^ bits = 2 * 2 ^ (bits - 1) := by
    rw [← pow_succ']; congr 1; omega
  have hlt : v < 2 ^ bits := by
    rw [hm]
    have h2 : (0:ℤ) < 2 ^ (bits - 1) := by positivity
    linarith
  have hmod : v % 2 ^ bits = v := Int.emod_eq_of_lt h0 hlt
  simp only [hmod]
  rw [hm, show (2 * 2 ^ (bits - 1)) / 2 = (2 : ℤ) ^ (bits - 1) by
    rw [mul_comm]; exact Int.mul_ediv_cancel _ (by norm_num)]
  rw [if_neg (not_le.mpr hv)]

theorem unsignedK_of_nonneg (bits : ℕ) (v : ℤ) (h0 : 0 ≤ v) (hv : v < 2 ^ (bits - 1)) (hb : 1 ≤ bits) :
    unsignedK bits v = v.toNat := by
  unfold unsignedK
  have hm : (2 : ℤ) ^ bits = 2 * 2 ^ (bits - 1) := by
    rw [← pow_succ']; congr 1; omega
  have hlt : v < 2 ^ bits := by
    rw [hm]
    have h2 : (0:ℤ) < 2 ^ (bits - 1) := by positivity
    linarith
  rw [Int.emod_eq_of_lt h0 hlt]

/-- For non-negative operands the `ℤ` bit operations are the `ℕ` ones on `toNat`. -/
theorem int_land_toNat (x y : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) : Int.land x y = Int.ofNat (x.toNat &&& y.toNat) := by
  obtain ⟨m, rfl⟩ := Int.eq_ofNat_of_zero_le hx
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hy
  rfl
theorem int_lor_toNat (x y : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) : Int.lor x y = Int.ofNat (x.toNat ||| y.toNat) := by
  obtain ⟨m, rfl⟩ := Int.eq_ofNat_of_zero_le hx
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hy
  rfl
theorem int_xor_toNat (x y : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) : Int.xor x y = Int.ofNat (x.toNat ^^^ y.toNat) := by
  obtain ⟨m, rfl⟩ := Int.eq_ofNat_of_zero_le hx
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hy
  rfl

/-- The three bit operations on the kind's width agree with Python's unbounded
ones on non-negative operands below `2^(bits-1)`. -/
theorem bitK_eq_unbounded (bits : ℕ) (x y : ℤ) (hb : 1 ≤ bits)
    (hx0 : 0 ≤ x) (hx : x < 2 ^ (bits - 1)) (hy0 : 0 ≤ y) (hy : y < 2 ^ (bits - 1)) :
    bitAndK bits x y = Int.ofNat (x.toNat &&& y.toNat) ∧
    bitOrK bits x y = Int.ofNat (x.toNat ||| y.toNat) ∧
    bitXorK bits x y = Int.ofNat (x.toNat ^^^ y.toNat) := by
  have hxN : x.toNat < 2 ^ (bits - 1) := by
    have : ((x.toNat : ℤ)) < 2 ^ (bits - 1) := by rw [Int.toNat_of_nonneg hx0]; exact hx
    exact_mod_cast this
  have hyN : y.toNat < 2 ^ (bits - 1) := by
    have : ((y.toNat : ℤ)) < 2 ^ (bits - 1) := by rw [Int.toNat_of_nonneg hy0]; exact hy
    exact_mod_cast this
  rw [bitAndK, bitOrK, bitXorK, unsignedK_of_nonneg bits x hx0 hx hb, unsignedK_of_nonneg bits y hy0 hy hb]
  refine ⟨?_, ?_, ?_⟩
  · rw [wrapSigned_of_lt bits _ (by positivity) ?_ hb]; rfl
    exact_mod_cast Nat.and_lt_two_pow x.toNat hyN
  · rw [wrapSigned_of_lt bits _ (by positivity) ?_ hb]; rfl
    exact_mod_cast Nat.or_lt_two_pow hxN hyN
  · rw [wrapSigned_of_lt bits _ (by positivity) ?_ hb]; rfl
    exact_mod_cast Nat.xor_lt_two_pow hxN hyN

theorem IKind.one_le_bits (k : IKind) : 1 ≤ k.bits := by cases k <;> simp [IKind.bits]

/-- In-range premise: both operands are non-negative and below the joined
kind's `HUGE`. -/
def BitOpsInRange (o : Oracles) (Γ : F.Env) (a b : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k₁ x k₂ y,
    F.eval o.F σ a = some (.s (.int k₁ x)) → F.eval o.F σ b = some (.s (.int k₂ y)) →
    0 ≤ x ∧ x < 2 ^ ((IKind.join k₁ k₂).bits - 1) ∧ 0 ≤ y ∧ y < 2 ^ ((IKind.join k₁ k₂).bits - 1)

def FNP_R0090_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → IntTyped o Γ a → IntTyped o Γ b → BitOpsInRange o Γ a b →
    SoundScalar o Γ (.call .iand (.cons a (.cons b .nil)) none) (Tr.call2 .f_iand a' b') ∧
    SoundScalar o Γ (.call .ior (.cons a (.cons b .nil)) none) (Tr.call2 .f_ior a' b') ∧
    SoundScalar o Γ (.call .ieor (.cons a (.cons b .nil)) none) (Tr.call2 .f_ieor a' b')

theorem bitop_sound (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr) (f : F.Intr) (h : P.Helper)
    (hf : (f = .iand ∧ h = .f_iand) ∨ (f = .ior ∧ h = .f_ior) ∨ (f = .ieor ∧ h = .f_ieor))
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hia : IntTyped o Γ a) (hib : IntTyped o Γ b)
    (hrange : BitOpsInRange o Γ a b) :
    SoundScalar o Γ (.call f (.cons a (.cons b .nil)) none) (Tr.call2 h a' b') := by
  intro σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  have hf' : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨
      f = .atan2 ∨ f = .iand ∨ f = .ior ∨ f = .ieor ∨ f = .ishft := by
    rcases hf with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp
  cases va with
  | arr av => exact absurd hint (intrinsic2_arr_ne o.F f _ _ _ (Or.inl ⟨av, rfl⟩) hf')
  | s x =>
    cases vb with
    | arr bv => exact absurd hint (intrinsic2_arr_ne o.F f _ _ _ (Or.inr ⟨bv, rfl⟩) hf')
    | s y =>
      obtain ⟨k₁, xv, rfl⟩ := hia σ τ hR hC x hva
      obtain ⟨k₂, yv, rfl⟩ := hib σ τ hR hC y hvb
      obtain ⟨hx0, hxlt, hy0, hylt⟩ := hrange σ τ hR hC k₁ xv k₂ yv hva hvb
      obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
      obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
      rw [RelS.int_eq hrx] at hx'
      rw [RelS.int_eq hry] at hy'
      obtain ⟨hand, hor, hxor⟩ :=
        bitK_eq_unbounded (IKind.join k₁ k₂).bits xv yv (IKind.one_le_bits _) hx0 hxlt hy0 hylt
      rw [P_eval_call2, hx', hy', Option.bind_some, Option.bind_some]
      rcases hf with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · rw [show F.intrinsic o.F .iand [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitAndK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.land xv yv), rfl, ?_⟩
        rw [int_land_toNat xv yv hx0 hy0]
        rw [hand]; exact RelS.int
      · rw [show F.intrinsic o.F .ior [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitOrK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.lor xv yv), rfl, ?_⟩
        rw [int_lor_toNat xv yv hx0 hy0]
        rw [hor]; exact RelS.int
      · rw [show F.intrinsic o.F .ieor [.s (.int k₁ xv), .s (.int k₂ yv)] none =
            some (.s (.int (IKind.join k₁ k₂) (bitXorK (IKind.join k₁ k₂).bits xv yv))) from rfl] at hint
        cases hint
        refine ⟨.int (Int.xor xv yv), rfl, ?_⟩
        rw [int_xor_toNat xv yv hx0 hy0]
        rw [hxor]; exact RelS.int

theorem FNP_R0090_sound_nonneg : FNP_R0090_obligation := by
  intro o Γ a b a' b' ha hb hia hib hrange
  exact ⟨bitop_sound o Γ a b a' b' _ _ (Or.inl ⟨rfl, rfl⟩) ha hb hia hib hrange,
    bitop_sound o Γ a b a' b' _ _ (Or.inr (Or.inl ⟨rfl, rfl⟩)) ha hb hia hib hrange,
    bitop_sound o Γ a b a' b' _ _ (Or.inr (Or.inr ⟨rfl, rfl⟩)) ha hb hia hib hrange⟩

end Recast.Rules
