import RecastProofs.Relations.State
import RecastProofs.Translation.Expr

/-!
# Rule obligations: shared definitions

The universal obligation of an expression rule: for every related pair of
states, if the source expression evaluates to `v`, the target expression
evaluates to a related `w`.  `SoundPair` is the array-inclusive form,
`SoundScalar` the scalar form.  Both are *statements about the semantics*;
nothing here mentions the emitter.
-/

namespace Recast.Rules

open Recast Rel

/-- Both semantics interpret uninterpreted primitives identically (the
congruence contract for `erf`, `gamma`, `atan2`, ...). -/
structure Oracles where
  prim : String → List ℝ → Option ℝ
  /-- The callee environments: the Fortran subprograms a `CALL` can reach and the
  emitted `def`s its translation calls.  Empty unless a rule is about calls. -/
  fprocs : F.ProcEnv := fun _ => none
  pprocs : P.ProcEnv := fun _ => none

def Oracles.F (o : Oracles) : F.Oracle := { prim := o.prim, procs := o.fprocs }
def Oracles.P (o : Oracles) : P.Oracle := { prim := o.prim, procs := o.pprocs }

/-- Expression-level soundness of a source/target pair. -/
def SoundPair (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ w, P.eval o.P τ e' = some w ∧ RelV v w

/-- Scalar-only soundness. -/
def SoundScalar (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) →
    ∃ w, P.eval o.P τ e' = some (.s w) ∧ RelS v w

theorem SoundPair.toScalar {o Γ e e'} (h : SoundPair o Γ e e') : SoundScalar o Γ e e' := by
  intro σ τ hR hC v hv
  obtain ⟨w, hw, hrel⟩ := h σ τ hR hC (.s v) hv
  cases hrel with
  | s hs => exact ⟨_, hw, hs⟩

/-- A refutation: some related states on which the source is defined and the
target is undefined or unrelated. -/
def Refuted (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr) : Prop := ¬ SoundPair o Γ e e'

theorem RelS.int_ne_float {k : IKind} {v : ℤ} {r : ℝ} : ¬ RelS (.int k v) (.float r) := by
  intro h; cases h

theorem RelS.int_eq {k : IKind} {n : ℤ} {w : PVal} (h : RelS (.int k n) w) : w = .int n := by
  cases h; rfl

theorem RelS.bool_eq {b : Bool} {w : PVal} (h : RelS (.bool b) w) : w = .bool b := by
  cases h; rfl

theorem RelS.str_eq {s : List Char} {w : PVal} (h : RelS (.str s) w) : w = .str s := by
  cases h; rfl

/-- The empty environment / states, for closed counterexamples. -/
def emptyEnv : F.Env := fun _ => none
def emptyF : F.State := fun _ => none
def emptyP : P.State := fun _ => none

theorem relState_empty : RelState emptyEnv emptyF emptyP := by
  intro x d h; simp [emptyEnv] at h

theorem conforms_empty {o : F.Oracle} : Conforms o emptyEnv emptyF := by
  intro x d h; simp [emptyEnv] at h

end Recast.Rules
