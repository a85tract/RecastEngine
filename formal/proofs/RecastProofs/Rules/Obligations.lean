import RecastProofs.Rules.Kernel
import RecastProofs.Rules.SeqAssoc

/-!
# Obligations stated as named propositions

Rules whose theorem carries its premises as binders get the ∀-closure here
(`FNP_Rxxxx_obligation`, with `_holds` witnessing that the theorem proves it);
rules that are only *stated* (no proof yet) get their obligation here too.
-/

namespace Recast.Rules

open Recast Rel Exact

def FNP_R0012_obligation : Prop := FNP_R0011_obligation
def FNP_R0013_obligation : Prop := FNP_R0011_obligation
def FNP_R0053_obligation : Prop := FNP_R0010_obligation
def FNP_R0172_obligation : Prop := FNP_R0004_obligation

def FNP_R0173_obligation : Prop := ∀ (a b : ℤ), b ≠ 0 → truncR ((a : ℝ) / (b : ℝ)) = tquot a b
theorem FNP_R0173_holds : FNP_R0173_obligation := truncR_div_intCast

def FNP_R0043_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lb' : P.Expr) (lo hi : F.Expr) (lo' hi' : P.Expr),
    Rank1Decl o Γ a lbE → SoundScalar o Γ lbE lb' → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    SectionInRange o Γ a (.given lo) (.given hi) →
    SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil))
      (.index a (.cons (.slice (.some (.bin .sub (.paren lo') (.paren lb')))
        (.some (.bin .add (.bin .sub (.paren hi') (.paren lb')) (.intLit 1))) .none) .nil))
theorem FNP_R0043_holds : FNP_R0043_obligation := FNP_R0043_sound

def FNP_R0044_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (k : IKind) (lo hi : Option F.Expr) (lo' hi' : Option P.Expr),
    Rank1Decl o Γ a (.intLit k 1) →
    (∀ e, lo = some e → ∃ e', lo' = some e' ∧ SoundScalar o Γ e e') → (lo = none → lo' = none) →
    (∀ e, hi = some e → ∃ e', hi' = some e' ∧ SoundScalar o Γ e e') → (hi = none → hi' = none) →
    SectionInRange o Γ a (lo.elim .omitted .given) (hi.elim .omitted .given) →
    SoundPair o Γ (.ref a (.cons (.triplet (lo.elim .omitted .given) (hi.elim .omitted .given) .omitted) .nil))
      (.index a (.cons (.slice (lo'.elim .none fun e' => .some (.bin .sub e' (.intLit 1)))
        (hi'.elim .none .some) .none) .nil))
theorem FNP_R0044_holds : FNP_R0044_obligation := FNP_R0044_sound

def FNP_R0045_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (k ks : IKind) (lo hi : F.Expr) (lo' hi' : P.Expr) (s : ℤ), 0 < s →
    Rank1Decl o Γ a (.intLit k 1) → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    SectionInRange o Γ a (.given lo) (.given hi) →
    SoundPair o Γ (.ref a (.cons (.triplet (.given lo) (.given hi) (.given (.intLit ks s))) .nil))
      (.index a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') (.some (.intLit s))) .nil))
theorem FNP_R0045_holds : FNP_R0045_obligation := FNP_R0045_sound

def FNP_R0047_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (lbE : F.Expr) (lb' : P.Expr) (lo hi : Option F.Expr)
    (lo' hi' : Option P.Expr) (st : F.Expr) (st' : P.Expr),
    Rank1Decl o Γ a lbE → SoundScalar o Γ lbE lb' →
    (∀ e, lo = some e → ∃ e', lo' = some e' ∧ SoundScalar o Γ e e') → (lo = none → lo' = none) →
    (∀ e, hi = some e → ∃ e', hi' = some e' ∧ SoundScalar o Γ e e') → (hi = none → hi' = none) →
    SoundScalar o Γ st st' → IntTyped o Γ st →
    (∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k s, F.eval o.F σ st = some (.s (.int k s)) → 0 < s) →
    SectionInRange o Γ a (lo.elim .omitted .given) (hi.elim .omitted .given) →
    SoundPair o Γ (.ref a (.cons (.triplet (lo.elim .omitted .given) (hi.elim .omitted .given) (.given st)) .nil))
      (.index a (.cons (.scalar (Tr.call4 .f_rstep_any (lo'.elim .none id) (hi'.elim .none id) st' lb')) .nil))
theorem FNP_R0047_holds : FNP_R0047_obligation := FNP_R0047_sound_pos

def FNP_R0112_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl), Γ a = some d → ∀ (lbE : F.Expr), d.lbs = [lbE] →
    StaticBounds Γ → Rank1Decl o Γ a lbE → ∀ (e : F.Expr) (e' : P.Expr), SoundScalar o Γ e e' →
    ScalarCompat o Γ d e →
    (∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ sv, v = .s sv) →
    StmtSound o Γ (.assign a .nil e) (.assignAll a e')
theorem FNP_R0112_holds : FNP_R0112_obligation := FNP_R0112_sound_scalar

def FNP_R0113_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl) (k : IKind), Γ a = some d → d.lbs = [.intLit k 1] →
    StaticBounds Γ → Rank1Decl o Γ a (.intLit k 1) → ∀ (i : F.Expr) (i' : P.Expr) (e : F.Expr) (e' : P.Expr),
    SoundScalar o Γ i i' → IntTyped o Γ i → SoundScalar o Γ e e' → ScalarCompat o Γ d e →
    StmtSound o Γ (.assign a (.cons (.scalar i) .nil) e)
      (.assignIndex a (.cons (.scalar (.bin .sub i' (.intLit 1))) .nil) e')
theorem FNP_R0113_holds : FNP_R0113_obligation := FNP_R0113_sound

def FNP_R0123_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env), StaticBounds Γ → ∀ (v : String) (kv : IKind) (cl : ℕ),
    Γ v = some ⟨.int kv, [], cl⟩ → ∀ (lo hi : F.Expr) (body : F.Stmt) (lo' hi' : P.Expr) (body' : P.Stmt),
    StmtSound o Γ body body' → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    IntTyped o Γ lo → IntTyped o Γ hi → Stable o.P v body' lo' → Stable o.P v body' hi' →
    StmtSound o Γ (.doLoop v lo hi none body)
      (.forRange v lo' (.bin .add hi' (.intLit 1)) none body' (some (completion v lo' hi' (.intLit 1))))
theorem FNP_R0123_holds : FNP_R0123_obligation := FNP_R0123_sound

def FNP_R0126_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env), StaticBounds Γ → ∀ (v : String) (kv ks : IKind) (cl : ℕ),
    Γ v = some ⟨.int kv, [], cl⟩ → ∀ (lo hi : F.Expr) (s : ℤ), s ≠ 0 → ∀ (body : F.Stmt)
    (lo' hi' : P.Expr) (body' : P.Stmt),
    StmtSound o Γ body body' → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    IntTyped o Γ lo → IntTyped o Γ hi → Stable o.P v body' lo' → Stable o.P v body' hi' →
    StmtSound o Γ (.doLoop v lo hi (some (.intLit ks s)) body)
      (.forRange v lo' (stepEdge hi' s) (some (.intLit s)) body' (some (completion v lo' hi' (.intLit s))))
theorem FNP_R0126_holds : FNP_R0126_obligation := FNP_R0126_sound

def FNP_R0125_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env), StaticBounds Γ → ∀ (v : String) (kv ks : IKind) (cl : ℕ),
    Γ v = some ⟨.int kv, [], cl⟩ → ∀ (lo hi : F.Expr) (s : ℤ), s < 0 → ∀ (body : F.Stmt)
    (lo' hi' : P.Expr) (body' : P.Stmt),
    StmtSound o Γ body body' → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    IntTyped o Γ lo → IntTyped o Γ hi → Stable o.P v body' lo' → Stable o.P v body' hi' →
    StmtSound o Γ (.doLoop v lo hi (some (.intLit ks s)) body)
      (.forRange v lo' (.bin .sub hi' (.intLit 1)) (some (.intLit s)) body' (some (completion v lo' hi' (.intLit s))))
theorem FNP_R0125_holds : FNP_R0125_obligation := FNP_R0125_sound

def FNP_R0127_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env), StaticBounds Γ → ∀ (v : String) (kv : IKind) (cl : ℕ),
    Γ v = some ⟨.int kv, [], cl⟩ → ∀ (lo hi st : F.Expr) (body : F.Stmt) (lo' hi' st' : P.Expr) (body' : P.Stmt),
    StmtSound o Γ body body' → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' → SoundScalar o Γ st st' →
    IntTyped o Γ lo → IntTyped o Γ hi → IntTyped o Γ st →
    Stable o.P v body' lo' → Stable o.P v body' hi' → Stable o.P v body' st' →
    StmtSound o Γ (.doLoop v lo hi (some st) body)
      (.forRange v lo' (.bin .add (.paren hi') (.paren (.ifexp (.intLit 1) (.cmp .gt (.paren st') (.intLit 0)) (.intLit (-1)))))
        (some st') body' (some (completion v lo' hi' st')))
theorem FNP_R0127_holds : FNP_R0127_obligation := FNP_R0127_sound

/-! ### Stated obligations (no proof yet) -/

/-- FNP-R0025: array-valued `**` -> `_f_vpow(a', b')`. -/
def FNP_R0025_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ b b' → SoundPair o Γ (.bin .pow a b) (Tr.call2 .f_vpow a' b')

/-- FNP-R0035: array-valued `.AND.`/`.OR.` -> `&`/`|`. -/
def FNP_R0035_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ b b' →
    SoundPair o Γ (.logic .and a b) (.paren (.bitand a' b')) ∧ SoundPair o Γ (.logic .or a b) (.paren (.bitor a' b'))

/-- FNP-R0100: elemental intrinsics over array arguments through the array spelling table. -/
def FNP_R0100_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (f : F.Intr) (h : P.Helper), Tr.arrayHelper f = some h →
    ∀ (a : F.Expr) (a' : P.Expr), SoundPair o Γ a a' → SoundPair o Γ (.call f (.cons a .nil) none) (Tr.fold1 h a')

/-- FNP-R0103 (DOT_PRODUCT part): refuted for INTEGER operands (FNP-D0040). -/
def FNP_R0103_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundPair o Γ a a' → SoundPair o Γ b b' →
    SoundScalar o Γ (.call .dot_product (.cons a (.cons b .nil)) none) (Tr.call2 .f_vdot a' b')

/-- FNP-R0114: section assignment on a unit-origin rank-1 array. -/
def FNP_R0114_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (k : IKind) (lo hi e : F.Expr) (lo' hi' e' : P.Expr),
    Rank1Decl o Γ a (.intLit k 1) → SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' → SoundPair o Γ e e' →
    SectionInRange o Γ a (.given lo) (.given hi) →
    StmtSound o Γ (.assign a (.cons (.triplet (.given lo) (.given hi) .omitted) .nil) e)
      (.assignIndex a (.cons (.slice (.some (.bin .sub lo' (.intLit 1))) (.some hi') .none) .nil) e')

end Recast.Rules
