import RecastProofs.Rules.Obligations

/-!
# FNP-R0140 / FNP-R0141 / FNP-R0142: the procedure-call convention

The emitted `def` takes the IN, INOUT and UNKNOWN dummies as parameters
(`Subprograms._passes`, subprograms.py:668-670) and returns the OUT and INOUT
dummies as a tuple (`Statements.returned_value`, statements.py:2570-2586); the
call site assigns the tuple back to the actuals (`Statements._call`,
statements.py:2180-2439).  The prologue zeroes every OUT dummy and every local
(`Subprograms._prologue`, FNP-R0142), which is what `F.initLocals` and
`P.Proc.locals` carry in the model.

The asymmetry between those two tables is a defect: a dummy whose intent is
neither declared nor inferable keeps the `UNKNOWN` label, is passed, and is
*not* returned, so the callee's write to it never reaches the caller.  Fortran
associates a dummy with no declared intent by reference (F2018 15.5.2.4), so the
write is the caller's to see.  `FNP_R0141_refuted_unknown` is that counter-
example in the model; `probes/intentchar` is it end to end, where the intent
inference declines because the dummy is a CHARACTER (FNP-D0047).
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### The modelled emission -/

/-- `Subprograms._passes`: IN, INOUT and UNKNOWN dummies are parameters of the
emitted `def`; an OUT dummy is not (its storage is the callee's). -/
def passes : F.Intent → Bool
  | .out => false
  | _ => true

/-- `Statements.returned_value`: only OUT and INOUT dummies come back. -/
def returned : F.Intent → Bool
  | .out => true
  | .inout => true
  | _ => false

/-- An actual argument as an expression of the target language. -/
def transActual (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) : F.Actual → Option P.Expr
  | .var x => some (.name x)
  | .expr e => Tr.trans Tr.ifx Γ Λ consts e

/-- The arguments the call passes, in dummy order. -/
def transInputs (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) :
    List F.Dummy → List F.Actual → Option P.Exprs
  | [], [] => some .nil
  | d :: ds, a :: as =>
      if passes d.intent then
        (transActual Γ Λ consts a).bind fun e' =>
          (transInputs Γ Λ consts ds as).map fun r => .cons e' r
      else transInputs Γ Λ consts ds as
  | _, _ => none

/-- The caller variables the returned tuple is assigned to, in dummy order. -/
def transOutputs : List F.Dummy → List F.Actual → Option (List String)
  | [], [] => some []
  | d :: ds, a :: as =>
      if returned d.intent then
        match a with
        | .var x => (transOutputs ds as).map fun r => x :: r
        | .expr _ => none
      else transOutputs ds as
  | _, _ => none

/-- `Statements._call` on the modelled fragment. -/
def transCallStmt (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (pr : F.Proc) (p : String)
    (acts : List F.Actual) : Option P.Stmt :=
  (transInputs Γ Λ consts pr.dummies acts).bind fun ins =>
  (transOutputs pr.dummies acts).map fun outs => .callAssign outs p ins

/-! ### The obligation -/

/-- The emitted `def` matches the subprogram: its parameters are the passed
dummies and its return tuple is the returned dummies, both in dummy order. -/
structure ProcMatch (pr : F.Proc) (ppr : P.Proc) : Prop where
  params : ppr.params = (pr.dummies.filter fun d => passes d.intent).map F.Dummy.name
  returns : ppr.returns = (pr.dummies.filter fun d => returned d.intent).map F.Dummy.name

/-- FNP-R0141: a `CALL` and the call statement the emitter writes for it. -/
def FNP_R0141_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (p : String)
    (pr : F.Proc) (ppr : P.Proc) (acts : List F.Actual) (s' : P.Stmt),
    o.fprocs p = some pr → o.pprocs p = some ppr → ProcMatch pr ppr →
    StmtSound o pr.env pr.body ppr.body →
    transCallStmt Γ Λ consts pr p acts = some s' →
    StmtSound o Γ (.call p acts) s'

/-! ### The counterexample: an UNKNOWN-intent dummy the callee writes -/

def declI4 : F.Decl := ⟨.int .i4, [], 0⟩

/-- `subroutine q(s)` with `integer :: s` (no declared intent) and body `s = 1`. -/
def envQ : F.Env := fun y => if y = "s" then some declI4 else none

def procQF : F.Proc :=
  { env := envQ, dummies := [⟨"s", .unknown⟩], locals := [], result := none,
    body := .assign "s" .nil (.intLit .i4 1) }

/-- Its translation: `def q(s): s = 1; return` -- `s` is a parameter, and the
return tuple is empty because `returned_value` lists only OUT and INOUT. -/
def procQP : P.Proc :=
  { params := ["s"], locals := [], returns := [], body := .assign "s" (.intLit 1) }

theorem procQ_match : ProcMatch procQF procQP := ⟨rfl, rfl⟩

/-- The caller: `integer :: x` holding `0`, and `call q(x)`. -/
def envCx : F.Env := fun y => if y = "x" then some declI4 else none
def stateCx : F.State := fun y => if y = "x" then some (.s (.int .i4 0)) else none
def pstateCx : P.State := fun y => if y = "x" then some (.s (.int 0)) else none

def oQ : Oracles :=
  { prim := fun _ _ => none
    fprocs := fun q => if q = "q" then some procQF else none
    pprocs := fun q => if q = "q" then some procQP else none }

def Λ0 : Tr.BoundEnv := fun _ => []

theorem relStateCx : RelState envCx stateCx pstateCx := by
  intro y d hy
  simp only [envCx] at hy
  split_ifs at hy with hyx
  subst hyx
  exact ⟨.s (.int .i4 0), .s (.int 0), by simp [stateCx], by simp [pstateCx], RelV.s RelS.int⟩

theorem conformsCx : Conforms oQ.F envCx stateCx := by
  intro y d hy
  simp only [envCx] at hy
  split_ifs at hy with hyx
  subst hyx
  cases hy
  exact ⟨fun _ => ⟨.int .i4 0, by simp [stateCx], rfl⟩, fun h => absurd rfl h⟩

theorem staticBoundsQ : StaticBounds envQ := by
  intro y d hy
  simp only [envQ] at hy
  split_ifs at hy with hys
  subst hys; cases hy
  intro e he; simp [declI4] at he

/-- The callee's body is sound: both sides store the integer `1` into `s`. -/
theorem procQ_body_sound : StmtSound oQ envQ procQF.body procQP.body := by
  intro σ τ hR hC σ' out h
  cases h with
  | assign hv hassign =>
    rename_i v
    obtain ⟨old, hσ, _⟩ := conforms_scalar hC "s" declI4 (by simp [envQ]) rfl
    cases v with
    | arr _ => simp [F.assign, envQ, hσ] at hassign
    | s sv =>
      rw [F_assign_scalar oQ.F envQ σ "s" declI4 old sv (by simp [envQ]) hσ] at hassign
      rw [F_eval_intLit] at hv
      simp only [Option.some.injEq, F.Val.s.injEq] at hv
      subst hv
      simp only [F.convertTo, declI4, Option.map_some, Option.some.injEq] at hassign
      subst hassign
      exact ⟨_, P.Exec.assign (by simp [P.eval]), relState_update hR "s" (RelV.s RelS.int),
        conforms_update_scalar hC staticBoundsQ "s" declI4 (by simp [envQ]) rfl _ rfl⟩

/-- The emitted call: `q(x)` with an empty target list, because `s` is not
returned. -/
theorem transQ : transCallStmt envCx Λ0 [] procQF "q" [.var "x"] =
    some (.callAssign [] "q" (.cons (.name "x") .nil)) := rfl

/-- The Fortran execution: `s` enters holding the actual's `0`, the body sets it
to `1`, and copy-back writes `1` into `x` (F2018 15.5.2.4: a dummy with no
declared intent may be defined). -/
theorem execQ : F.Exec oQ.F envCx (.call "q" [.var "x"]) stateCx
    (Function.update stateCx "x" (some (.s (.int .i4 1)))) .normal := by
  refine F.Exec.call (pr := procQF) (binds := [("s", .s (.int .i4 0))]) (locs := [])
    (σc' := Function.update (F.stateOf [("s", F.Val.s (.int .i4 0))]) "s" (some (.s (.int .i4 1))))
    (out := .normal) rfl ?_ rfl ?_ ?_
  · show F.bindDummies oQ.F stateCx envQ [⟨"s", .unknown⟩] [.var "x"] = _
    simp [F.bindDummies, envQ, F.actualVal, stateCx, F.dummyEntry, declI4]
  · refine F.Exec.assign (v := .s (.int .i4 1)) (by simp [F.eval]) ?_
    show F.assign oQ.F envQ (F.stateOf [("s", F.Val.s (.int .i4 0))]) "s" .nil (.s (.int .i4 1)) = _
    simp [F.assign, envQ, F.stateOf, F.convertTo, declI4]
  · show F.copyBack _ [⟨"s", .unknown⟩] [.var "x"] stateCx = _
    simp [F.copyBack, F.stateOf]

/-- FNP-D0047 in the model: the callee's write to an UNKNOWN-intent dummy never
reaches the caller, because the emitted `def` does not return it. -/
theorem FNP_R0141_refuted_unknown : ¬ FNP_R0141_obligation := by
  intro h
  have hs := h oQ envCx Λ0 [] "q" procQF procQP [.var "x"] _ rfl rfl
    procQ_match procQ_body_sound transQ
  obtain ⟨τ', hex, hR', _⟩ := hs stateCx pstateCx relStateCx conformsCx _ _ execQ
  -- the target list is empty, so the caller's state is untouched
  cases hex with
  | callAssign hproc hargs hbind hbody hret hassign =>
    rename_i pr args vs binds τc' outs
    simp only [oQ, Oracles.P, if_true] at hproc
    cases hproc
    simp only [procQP, P.readReturns, Option.some.injEq] at hret
    subst hret
    simp only [P.assignTargets, Option.some.injEq] at hassign
    subst hassign
    obtain ⟨v, w, hv, hw, hrel⟩ := hR' "x" declI4 (by simp [envCx])
    simp only [Function.update_self, Option.some.injEq] at hv
    subst hv
    simp only [pstateCx, if_true, Option.some.injEq] at hw
    subst hw
    cases hrel with
    | s hs' =>
      have h0 := RelS.int_eq hs'
      simp at h0

end Recast.Rules
