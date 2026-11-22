import RecastProofs.Rules.Basic

/-!
# Rules FNP-R0001 .. FNP-R0011: literals and names
-/

namespace Recast.Rules

open Recast Rel

/-! ### FNP-R0001 / FNP-R0002: integer literals -/

/-- Obligation of FNP-R0001: an inline integer literal. -/
def FNP_R0001_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (k : IKind) (v : ℤ), SoundScalar o Γ (.intLit k v) (.intLit v)

theorem FNP_R0001_sound : FNP_R0001_obligation := by
  intro o Γ k v σ τ _ _ w hw
  simp only [F.eval] at hw
  cases hw
  exact ⟨.int v, by simp [P.eval], RelS.int⟩

/-- Obligation of FNP-R0002: a hoisted integer literal, referenced by the
name `I_<n>` that the constants module binds to `np.int32(n)`/`np.int64(n)`.
The rule is sound under the environment premise that the name is bound to
the value (the constants-module rule supplies it). -/
def FNP_R0002_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (k : IKind) (v : ℤ) (hoisted : String) (σ : F.State) (τ : P.State),
    τ hoisted = some (.s (.int v)) →
    ∀ val, F.eval o.F σ (.intLit k v) = some (.s val) →
      ∃ w, P.eval o.P τ (.name hoisted) = some (.s w) ∧ RelS val w

theorem FNP_R0002_sound : FNP_R0002_obligation := by
  intro o Γ k v hoisted σ τ hbind val hval
  simp only [F.eval] at hval
  cases hval
  exact ⟨.int v, by simp [P.eval, hbind], RelS.int⟩

/-! ### FNP-R0003 / FNP-R0004 / FNP-R0005: real literals -/

/-- Obligation of FNP-R0003: a whitelisted real literal written inline. The
inline text `repr(float(text))` spells one of `0.0`, `1.0`, `0.5`; the Lean
side represents the emitted token by its exact decimal. -/
def FNP_R0003_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (k : RKind) (d : Decimal),
    d ∈ Decimal.whitelistReal → SoundScalar o Γ (.realLit k d) (.floatLit d)

theorem FNP_R0003_sound : FNP_R0003_obligation := by
  intro o Γ k d _ σ τ _ _ v hv
  simp only [F.eval] at hv
  cases hv
  exact ⟨.float d.toReal, by simp [P.eval], RelS.realF⟩

/-- Obligation of FNP-R0004: hoisted real literal, name bound to
`np.float64('<text>')`, which under the exact literal policy denotes the
decimal of the text (the same `Decimal`). -/
def FNP_R0004_obligation : Prop :=
  ∀ (o : Oracles) (k : RKind) (d : Decimal) (hoisted : String) (σ : F.State) (τ : P.State),
    τ hoisted = some (.s (.float d.toReal)) →
    ∀ val, F.eval o.F σ (.realLit k d) = some (.s val) →
      ∃ w, P.eval o.P τ (.name hoisted) = some (.s w) ∧ RelS val w

theorem FNP_R0004_sound : FNP_R0004_obligation := by
  intro o k d hoisted σ τ hbind val hval
  simp only [F.eval] at hval
  cases hval
  exact ⟨.float d.toReal, by simp [P.eval, hbind], RelS.realF⟩

/-- FNP-R0005 is FNP-R0004 with the `np.float32` cast, an identity in the model. -/
def FNP_R0005_obligation : Prop := FNP_R0004_obligation

theorem FNP_R0005_sound : FNP_R0005_obligation := FNP_R0004_sound

/-! ### FNP-R0006: logical literals -/

def FNP_R0006_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (b : Bool), SoundScalar o Γ (.boolLit b) (.boolLit b)

theorem FNP_R0006_sound : FNP_R0006_obligation := by
  intro o Γ b σ τ _ _ v hv
  simp only [F.eval] at hv
  cases hv
  exact ⟨.bool b, by simp [P.eval], RelS.bool⟩

/-! ### FNP-R0007: character literals (no embedded quotes) -/

def FNP_R0007_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (s : List Char), '\'' ∉ s → SoundScalar o Γ (.strLit s) (.strLit s)

theorem FNP_R0007_sound : FNP_R0007_obligation := by
  intro o Γ s _ σ τ _ _ v hv
  simp only [F.eval] at hv
  cases hv
  exact ⟨.str s, by simp [P.eval], RelS.str⟩

/-! ### FNP-R0010 / FNP-R0011: names -/

/-- Obligation of FNP-R0010: a declared name evaluates to related values
under the state relation (identity renaming in the model). -/
def FNP_R0010_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (x : String) (d : F.Decl), Γ x = some d →
    SoundPair o Γ (.var x) (.name x)

theorem FNP_R0010_sound : FNP_R0010_obligation := by
  intro o Γ x d hx σ τ hR hC v hv
  obtain ⟨v', w, hσ, hτ, hrel⟩ := hR x d hx
  simp only [F.eval] at hv
  rw [hσ] at hv
  cases hv
  exact ⟨w, by simp [P.eval, hτ], hrel⟩

/-- FNP-R0011: a module parameter referenced through its upper-case constant
name is the same lookup lemma, with the constants module supplying the binding. -/
def FNP_R0011_obligation : Prop :=
  ∀ (o : Oracles) (σ : F.State) (τ : P.State) (x upper : String) (v : SVal) (w : PVal),
    σ x = some (.s v) → τ upper = some (.s w) → RelS v w →
    ∀ val, F.eval o.F σ (.var x) = some (.s val) →
      ∃ w', P.eval o.P τ (.name upper) = some (.s w') ∧ RelS val w'

theorem FNP_R0011_sound : FNP_R0011_obligation := by
  intro o σ τ x upper v w hσ hτ hrel val hval
  simp only [F.eval] at hval
  rw [hσ] at hval
  cases hval
  exact ⟨w, by simp [P.eval, hτ], hrel⟩

end Recast.Rules
