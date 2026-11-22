import RecastProofs.Rules.Arith

/-!
# Rules FNP-R0022 and FNP-R0026: the `gfortran` profile's power spellings
-/

namespace Recast.Rules

open Recast Rel Exact

theorem F_eval_realLit (o : F.Oracle) (σ : F.State) (k : RKind) (d : Decimal) :
    F.eval o σ (.realLit k d) = some (.s (.real k d.toReal)) := by simp [F.eval]

theorem Decimal_two : Decimal.toReal ⟨2, 0⟩ = (2 : ℝ) := by norm_num [Decimal.toReal]

/-! ### FNP-R0022: `x ** 2.0` -> `(x' * x')` -/

def FNP_R0022_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (k₂ : RKind),
    SoundScalar o Γ a a' → RealTyped o Γ a →
    SoundScalar o Γ (.bin .pow a (.realLit k₂ ⟨2, 0⟩)) (.paren (.bin .mul a' a'))

theorem FNP_R0022_sound : FNP_R0022_obligation := by
  intro o Γ a a' k₂ ha hreal σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨kx, r, rfl⟩ := hreal σ τ hR hC x hva
  rw [F_eval_realLit] at hvb
  simp only [Option.some.injEq, F.Val.s.injEq] at hvb
  subst hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  have ht := RelS.real_toR hrx
  simp only [F.arith] at hxy
  rw [Decimal_two] at hxy
  cases hy : F.rpowF r 2 with
  | none => rw [hy] at hxy; simp at hxy
  | some y =>
    rw [hy] at hxy
    simp only [Option.map_some, Option.some.injEq] at hxy
    subst hxy
    have hsq : y = r * r := by
      unfold F.rpowF at hy
      split_ifs at hy with hpos hzero
      · simp only [Option.some.injEq] at hy; subst hy
        rw [show (2:ℝ) = ((2:ℕ) : ℝ) by norm_num, Real.rpow_eq_pow, Real.rpow_natCast]; ring
      · simp only [Option.some.injEq] at hy; subst hy; rw [hzero.1]; ring
    obtain ⟨c, hc, hcr⟩ := P_arith_addsubmul_toR .mul (Or.inr (Or.inr rfl)) ht ht
    refine ⟨c, ?_, RelS.real (by rw [hcr, hsq]; rfl)⟩
    rw [P_eval_paren, P_eval_bin, hx']
    simp [P.lift2, hc]

/-! ### FNP-R0026: real base, integer exponent -> `_f_powi(x', j')` -/

def FNP_R0026_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → RealTyped o Γ a → IntTyped o Γ b →
    SoundScalar o Γ (.bin .pow a b) (Tr.call2 .f_powi a' b')

theorem helper_f_powi_float (o : P.Oracle) (a : ℝ) (n : ℤ) :
    P.helper o .f_powi [.s (.float a), .s (.int n)] =
      (if n = 0 then some (.s (.float 1))
       else if n < 0 then (if a = 0 then none else some (.s (.float (a ^ n))))
       else some (.s (.float (a ^ n)))) := rfl

theorem helper_f_powi_int (o : P.Oracle) (a : ℤ) (n : ℤ) :
    P.helper o .f_powi [.s (.int a), .s (.int n)] =
      (if n = 0 then some (.s (.int 1))
       else if n < 0 then (if a = 0 then none else some (.s (.float ((a : ℝ) ^ n))))
       else some (.s (.int (a ^ n.toNat)))) := rfl

theorem FNP_R0026_sound : FNP_R0026_obligation := by
  intro o Γ a b a' b' ha hb hreal hint σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨kx, r, rfl⟩ := hreal σ τ hR hC x hva
  obtain ⟨ky, n, rfl⟩ := hint σ τ hR hC y hvb
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
  have ht := RelS.real_toR hrx
  rw [RelS.int_eq hry] at hy'
  simp only [F.arith, F.zpowF] at hxy
  split at hxy
  · simp at hxy
  · rename_i hdef
    simp only [Option.map_some, Option.some.injEq] at hxy
    subst hxy
    rw [P_eval_call2, hx', hy', Option.bind_some, Option.bind_some]
    rcases P_toR_cases ht with ⟨m, rfl, rfl⟩ | rfl
    · rw [helper_f_powi_int]
      by_cases h0 : n = 0
      · subst h0
        exact ⟨.int 1, by simp, RelS.real (by simp [P.toR])⟩
      · by_cases hneg : n < 0
        · have hm : m ≠ 0 := by rintro rfl; exact hdef ⟨by simp, hneg.le⟩
          exact ⟨.float ((m:ℝ) ^ n), by simp [h0, hneg, hm], RelS.realF⟩
        · refine ⟨.int (m ^ n.toNat), by simp [h0, hneg], RelS.real ?_⟩
          simp only [P.toR, Option.some.injEq]
          obtain ⟨k, rfl⟩ : ∃ k : ℕ, n = k := ⟨n.toNat, (Int.toNat_of_nonneg (not_lt.mp hneg)).symm⟩
          simp [zpow_natCast]
    · rw [helper_f_powi_float]
      by_cases h0 : n = 0
      · subst h0
        exact ⟨.float 1, by simp, by rw [zpow_zero]; exact RelS.realF⟩
      · by_cases hneg : n < 0
        · have hr : r ≠ 0 := fun h => hdef ⟨h, hneg.le⟩
          exact ⟨.float (r ^ n), by simp [h0, hneg, hr], RelS.realF⟩
        · exact ⟨.float (r ^ n), by simp [h0, hneg], RelS.realF⟩

end Recast.Rules
