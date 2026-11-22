import RecastProofs.Rules.Stmt
import RecastProofs.Rules.Intrinsics2

/-!
# Rules FNP-R0120 (IF), FNP-R0123/R0125/R0126/R0127 (counted DO), FNP-R0128
(completion value), FNP-R0129/R0130 (DO WHILE), FNP-R0131 (EXIT/CYCLE)

Each construct is proved *compositionally*: the body/branch simulations are
hypotheses (the statement induction hypotheses), and the construct's own
simulation is concluded.  `Soundness.lean` assembles them along the big-step
derivation.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Trivial statements: FNP-R0131 and CONTINUE -/

theorem skip_sound (o : Oracles) (Γ : F.Env) : StmtSound o Γ .skip .pass := by
  intro σ τ hR hC σ' out h
  cases h
  exact ⟨τ, P.Exec.pass, hR, hC⟩

def FNP_R0131_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env), StmtSound o Γ .exit .break ∧ StmtSound o Γ .cycle .continue

theorem FNP_R0131_sound : FNP_R0131_obligation := by
  intro o Γ
  constructor <;> intro σ τ hR hC σ' out h <;> cases h
  · exact ⟨τ, P.Exec.brk, hR, hC⟩
  · exact ⟨τ, P.Exec.cont, hR, hC⟩

/-! ### Sequencing -/

theorem seq_sound (o : Oracles) (Γ : F.Env) (s₁ s₂ : F.Stmt) (s₁' s₂' : P.Stmt)
    (h₁ : StmtSound o Γ s₁ s₁') (h₂ : StmtSound o Γ s₂ s₂') : StmtSound o Γ (.seq s₁ s₂) (.seq s₁' s₂') := by
  intro σ τ hR hC σ' out h
  cases h with
  | seq_normal ha hb =>
    obtain ⟨τ₁, hex₁, hR₁, hC₁⟩ := h₁ σ τ hR hC _ _ ha
    obtain ⟨τ₂, hex₂, hR₂, hC₂⟩ := h₂ _ τ₁ hR₁ hC₁ _ _ hb
    exact ⟨τ₂, P.Exec.seq_normal hex₁ hex₂, hR₂, hC₂⟩
  | seq_abrupt ha hne =>
    obtain ⟨τ₁, hex₁, hR₁, hC₁⟩ := h₁ σ τ hR hC _ _ ha
    exact ⟨τ₁, P.Exec.seq_abrupt hex₁ (outMap_ne_normal hne), hR₁, hC₁⟩

/-! ### FNP-R0120: IF -/

def FNP_R0120_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (c : F.Expr) (c' : P.Expr) (t e : F.Stmt) (t' e' : P.Stmt),
    SoundScalar o Γ c c' → StmtSound o Γ t t' → StmtSound o Γ e e' →
    StmtSound o Γ (.ite c t e) (.ite c' t' e')

theorem FNP_R0120_sound : FNP_R0120_obligation := by
  intro o Γ c c' t e t' e' hc ht he σ τ hR hC σ' out h
  cases h with
  | ite_true hcond hex =>
    obtain ⟨w, hw, hrw⟩ := hc σ τ hR hC _ hcond
    rw [RelS.bool_eq hrw] at hw
    obtain ⟨τ', hex', hR', hC'⟩ := ht σ τ hR hC _ _ hex
    exact ⟨τ', P.Exec.ite_true hw hex', hR', hC'⟩
  | ite_false hcond hex =>
    obtain ⟨w, hw, hrw⟩ := hc σ τ hR hC _ hcond
    rw [RelS.bool_eq hrw] at hw
    obtain ⟨τ', hex', hR', hC'⟩ := he σ τ hR hC _ _ hex
    exact ⟨τ', P.Exec.ite_false hw hex', hR', hC'⟩

/-! ### FNP-R0129 / FNP-R0130: DO WHILE -/

def FNP_R0129_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (c : F.Expr) (c' : P.Expr) (body : F.Stmt) (body' : P.Stmt),
    SoundScalar o Γ c c' → StmtSound o Γ body body' →
    StmtSound o Γ (.doWhile c body) (.while c' body')

theorem FNP_R0129_sound : FNP_R0129_obligation := by
  intro o Γ c c' body body' hc hbody σ₀ τ₀ hR₀ hC₀ σ₀' out₀ hexec
  suffices key : ∀ (s : F.Stmt) (σ σ' : F.State) (out : F.Outcome), F.Exec o.F Γ s σ σ' out →
      Γ = Γ → s = .doWhile c body → ∀ τ, RelState Γ σ τ → Conforms o.F Γ σ →
        ∃ τ', P.Exec o.P (.while c' body') τ τ' (outMap out) ∧ RelState Γ σ' τ' ∧ Conforms o.F Γ σ' from
    key _ _ _ _ hexec rfl rfl τ₀ hR₀ hC₀
  intro s σ σ' out h
  refine F.Exec.rec (motive_1 := fun Γ' s σ σ' out _ => Γ' = Γ → s = .doWhile c body → ∀ τ, RelState Γ σ τ →
      Conforms o.F Γ σ → ∃ τ', P.Exec o.P (.while c' body') τ τ' (outMap out) ∧ RelState Γ σ' τ' ∧ Conforms o.F Γ σ')
    (motive_2 := fun _ _ _ _ _ _ _ _ _ => True)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ h
  · intro _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ heq; cases heq
  · intro _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · -- doWhile_false
    intro _ c₀ body₀ σ hfalse hΓ heq τ hR hC
    cases hΓ; cases heq
    obtain ⟨w, hw, hrw⟩ := hc σ τ hR hC _ hfalse
    rw [RelS.bool_eq hrw] at hw
    exact ⟨τ, P.Exec.while_false hw, hR, hC⟩
  · -- doWhile_exit
    intro _ c₀ body₀ σ σ' htrue hb _ hΓ heq τ hR hC
    cases hΓ; cases heq
    obtain ⟨w, hw, hrw⟩ := hc σ τ hR hC _ htrue
    rw [RelS.bool_eq hrw] at hw
    obtain ⟨τ', hex', hR', hC'⟩ := hbody σ τ hR hC _ _ hb
    exact ⟨τ', P.Exec.while_break hw hex', hR', hC'⟩
  · -- doWhile_step
    intro _ c₀ body₀ σ σ₁ σ' out htrue hb hne _ _ ih hΓ heq τ hR hC
    cases hΓ; cases heq
    obtain ⟨w, hw, hrw⟩ := hc σ τ hR hC _ htrue
    rw [RelS.bool_eq hrw] at hw
    obtain ⟨τ₁, hex₁, hR₁, hC₁⟩ := hbody σ τ hR hC _ _ hb
    obtain ⟨τ', hex', hR', hC'⟩ := ih rfl rfl τ₁ hR₁ hC₁
    exact ⟨τ', P.Exec.while_step hw hex₁ (outMap_ne_brk hne) hex', hR', hC'⟩
  · intro _ _ _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ heq; cases heq
  · intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ heq; cases heq
  · intros; trivial
  · intros; trivial
  · intros; trivial

def FNP_R0130_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (body : F.Stmt) (body' : P.Stmt), StmtSound o Γ body body' →
    StmtSound o Γ (.doWhile (.boolLit true) body) (.while (.boolLit true) body')

theorem FNP_R0130_sound : FNP_R0130_obligation := by
  intro o Γ body body' hbody
  exact FNP_R0129_sound o Γ (.boolLit true) (.boolLit true) body body'
    (fun σ τ _ _ v hv => by simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv; exact ⟨_, rfl, RelS.bool⟩)
    hbody

/-! ### Trip counts (FNP-R0128) -/

theorem max_fdiv_tdiv_pos (x s : ℤ) (hs : 0 < s) : max 0 (Int.fdiv x s) = max (Int.tdiv x s) 0 := by
  rw [Int.fdiv_eq_ediv_of_nonneg x hs.le]
  by_cases hx : 0 ≤ x
  · rw [Int.tdiv_eq_ediv_of_nonneg hx]
    have := Int.ediv_nonneg hx hs.le
    rw [max_eq_right this, max_eq_left this]
  · have hneg : x < 0 := by omega
    have h1 : x / s < 0 := Int.ediv_neg_of_neg_of_pos hneg hs
    have h2 : Int.tdiv x s ≤ 0 := by
      rw [show x = -(-x) by ring, Int.neg_tdiv]
      have := Int.tdiv_nonneg (show 0 ≤ -x by omega) hs.le
      omega
    rw [max_eq_left h1.le, max_eq_right h2]

theorem max_fdiv_tdiv (x s : ℤ) (hs : s ≠ 0) : max 0 (Int.fdiv x s) = max (Int.tdiv x s) 0 := by
  rcases lt_or_gt_of_ne hs with hneg | hpos
  · rw [← Int.neg_fdiv_neg x s, ← Int.neg_tdiv_neg x s]
    exact max_fdiv_tdiv_pos (-x) (-s) (by omega)
  · exact max_fdiv_tdiv_pos x s hpos

/-- `max(0, (hi - lo + st) // st)` (Python floor division) is the Fortran trip count. -/
theorem FNP_R0128_tripcount (l h s : ℤ) (hs : s ≠ 0) :
    max 0 (Int.fdiv (h - l + s) s) = (Fort.tripCount l h s : ℤ) := by
  unfold Fort.tripCount Exact.tquot
  rw [max_fdiv_tdiv _ _ hs, Int.toNat_of_nonneg (le_max_right _ _)]

def FNP_R0128_obligation : Prop :=
  ∀ (l h s : ℤ), s ≠ 0 → l + max 0 (Int.fdiv (h - l + s) s) * s = l + (Fort.tripCount l h s : ℤ) * s

theorem FNP_R0128_sound : FNP_R0128_obligation := by
  intro l h s hs; rw [FNP_R0128_tripcount l h s hs]

/-! ### The iteration values -/

/-- The `n` index values `cur, cur + st, …`. -/
def loopVals (cur st : ℤ) (n : ℕ) : List ℤ := List.map (fun (k : ℕ) => cur + (k : ℤ) * st) (List.range n)

theorem loopVals_zero (cur st : ℤ) : loopVals cur st 0 = [] := rfl

theorem loopVals_succ (cur st : ℤ) (n : ℕ) : loopVals cur st (n + 1) = cur :: loopVals (cur + st) st n := by
  unfold loopVals
  rw [List.range_succ_eq_map, List.map_cons, List.map_map]
  simp only [Nat.cast_zero, zero_mul, add_zero, List.cons.injEq, true_and]
  apply List.map_congr_left
  intro k _
  simp only [Function.comp, Nat.succ_eq_add_one, Nat.cast_add, Nat.cast_one]
  ring

/-- Python's `range(lo, hi + 1, st)` for `st > 0` enumerates the Fortran iteration values. -/
theorem pyRange_pos (l h s : ℤ) (hs : 0 < s) : P.pyRange l (h + 1) s = loopVals l s (Fort.tripCount l h s) := by
  unfold P.pyRange loopVals
  have := tripCount_eq_sliceLen l h 0 s hs
  simp only [sub_zero] at this
  rw [this]

/-- Python's `range(lo, hi - 1, st)` for `st < 0` enumerates the Fortran iteration values. -/
theorem pyRange_neg (l h s : ℤ) (hs : s < 0) : P.pyRange l (h - 1) s = loopVals l s (Fort.tripCount l h s) := by
  unfold P.pyRange loopVals
  congr 1
  have h1 : Py.sliceLen l (h - 1) s = Py.sliceLen h (l + 1) (-s) := by
    unfold Py.sliceLen
    rw [if_neg (show ¬ (s > 0) by omega), if_pos (show -s > 0 by omega)]
    by_cases hlt : h - 1 < l
    · rw [if_pos hlt, if_pos (show h < l + 1 by omega)]
      have : l - (h - 1) - 1 = l + 1 - h - 1 := by ring
      rw [this]
    · rw [if_neg hlt, if_neg (show ¬ (h < l + 1) by omega)]
  have h2 : Fort.tripCount l h s = Fort.tripCount h l (-s) := by
    unfold Fort.tripCount Exact.tquot
    rw [← Int.neg_tdiv_neg (h - l + s) s]
    congr 3; ring
  rw [h1, h2]
  have := tripCount_eq_sliceLen h l 0 (-s) (by omega)
  simp only [sub_zero] at this
  rw [this]

/-! ### The loop simulation -/

/-- `e` keeps its value across the loop body and across updates of the loop variable. -/
def Stable (o : P.Oracle) (v : String) (body' : P.Stmt) (e : P.Expr) : Prop :=
  (∀ τ τ' out, P.Exec o body' τ τ' out → P.eval o τ' e = P.eval o τ e) ∧
  (∀ τ i, P.eval o (Function.update τ v (some (.s (.int i)))) e = P.eval o τ e)

/-- The emitted completion assignment `v = (lo) + max(0, (((hi) - (lo) + (st)) // (st))) * (st)`. -/
def completion (v : String) (lo' hi' st' : P.Expr) : P.Stmt :=
  .assign v (.bin .add (.paren lo') (.bin .mul
    (.call .py_max (.cons (.intLit 0) (.cons
      (.paren (.bin .floordiv (.paren (.bin .add (.bin .sub (.paren hi') (.paren lo')) (.paren st'))) (.paren st')))
      .nil)))
    (.paren st')))

theorem P_eval_call2' (o : P.Oracle) (τ : P.State) (h : P.Helper) (a b : P.Expr) :
    P.eval o τ (.call h (.cons a (.cons b .nil))) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.helper o h [va, vb] :=
  P_eval_call2 o τ h a b

theorem helper_py_max_int (o : P.Oracle) (a b : ℤ) :
    P.helper o .py_max [.s (.int a), .s (.int b)] = some (.s (.int (max a b))) := by
  have e : P.helper o .py_max [.s (.int a), .s (.int b)] =
      (P.toR (.int a)).bind fun p => (P.toR (.int b)).map fun q => P.Val.s (if p < q then .int b else .int a) := rfl
  rw [e]
  simp only [P.toR, Option.bind_some, Option.map_some, Option.some.injEq, P.Val.s.injEq]
  split_ifs with hlt
  · rw [max_eq_right (le_of_lt (by exact_mod_cast hlt))]
  · rw [max_eq_left (by exact_mod_cast (not_lt.mp hlt))]

theorem completion_exec (o : P.Oracle) (τ : P.State) (v : String) (lo' hi' st' : P.Expr) (l h s : ℤ) (hs : s ≠ 0)
    (hlo : P.eval o τ lo' = some (.s (.int l))) (hhi : P.eval o τ hi' = some (.s (.int h)))
    (hst : P.eval o τ st' = some (.s (.int s))) :
    P.Exec o (completion v lo' hi' st') τ (Function.update τ v (some (.s (.int (l + (Fort.tripCount l h s : ℤ) * s))))) .normal := by
  have hinner : P.eval o τ (.paren (.bin .floordiv (.paren (.bin .add (.bin .sub (.paren hi') (.paren lo')) (.paren st'))) (.paren st'))) =
      some (.s (.int (Int.fdiv (h - l + s) s))) := by
    simp only [P_eval_paren, P_eval_bin, hhi, hlo, hst, Option.bind_some, P_lift2_scalar]
    simp [P.lift2, P.arith, hs, fquot]
  have hmax : P.eval o τ (.call .py_max (.cons (.intLit 0) (.cons (.paren (.bin .floordiv (.paren (.bin .add (.bin .sub (.paren hi') (.paren lo')) (.paren st'))) (.paren st'))) .nil))) =
      some (.s (.int (Fort.tripCount l h s))) := by
    rw [P_eval_call2', hinner]
    simp only [P.eval, Option.bind_some, helper_py_max_int]
    rw [FNP_R0128_tripcount l h s hs]
  have hval : P.eval o τ (.bin .add (.paren lo') (.bin .mul (.call .py_max (.cons (.intLit 0) (.cons
      (.paren (.bin .floordiv (.paren (.bin .add (.bin .sub (.paren hi') (.paren lo')) (.paren st'))) (.paren st')))
      .nil))) (.paren st'))) = some (.s (.int (l + (Fort.tripCount l h s : ℤ) * s))) := by
    simp only [P_eval_bin, P_eval_paren, hlo, hmax, hst, Option.bind_some, P_lift2_scalar]
    simp [P.lift2, P.arith]
  exact P.Exec.assign hval

/-- The DO-variable store: `v` is a declared INTEGER scalar. -/
theorem F_assign_loopvar (o : F.Oracle) (Γ : F.Env) (σ : F.State) (hC : Conforms o Γ σ) (v : String) (kv : IKind)
    (cl : ℕ) (hv : Γ v = some ⟨.int kv, [], cl⟩) (cur : ℤ) :
    F.assign o Γ σ v .nil (.s (.int .i4 cur)) = some (Function.update σ v (some (.s (.int kv cur)))) := by
  obtain ⟨sv, hσ, _⟩ := conforms_scalar hC v _ hv rfl
  rw [F_assign_scalar o Γ σ v _ sv _ hv hσ]
  rfl

theorem loop_sim (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (body : F.Stmt) (body' : P.Stmt) (hbody : StmtSound o Γ body body')
    (st : ℤ) (orelse : P.Stmt) (cfin : ℤ) (Inv : P.State → Prop)
    (hInv_body : ∀ τ τ' out, Inv τ → P.Exec o.P body' τ τ' out → Inv τ')
    (hInv_upd : ∀ τ i, Inv τ → Inv (Function.update τ v (some (.s (.int i)))))
    (hor : ∀ τ, Inv τ → P.Exec o.P orelse τ (Function.update τ v (some (.s (.int cfin)))) .normal) :
    ∀ (n : ℕ) (cur : ℤ) (σ σ' : F.State), F.Loop o.F Γ v body st n cur σ σ' → cur + n * st = cfin →
      ∀ τ, RelState Γ σ τ → Conforms o.F Γ σ → Inv τ →
        ∃ τ', P.ForLoop o.P v body' (some orelse) (loopVals cur st n) τ τ' ∧ RelState Γ σ' τ' ∧ Conforms o.F Γ σ' := by
  intro n
  induction n with
  | zero =>
    intro cur σ σ' hloop hfin τ hR hC hI
    cases hloop with
    | done hassign =>
      rw [F_assign_loopvar o.F Γ σ hC v kv cl hv cur] at hassign
      cases hassign
      simp only [Nat.cast_zero, zero_mul, add_zero] at hfin
      subst hfin
      refine ⟨_, P.ForLoop.done_else (hor τ hI), relState_update hR v (RelV.s RelS.int), ?_⟩
      exact conforms_update_scalar hC hS v _ hv rfl _ rfl
  | succ n ih =>
    intro cur σ σ' hloop hfin τ hR hC hI
    cases hloop with
    | step hassign hb hne hrest =>
      rw [F_assign_loopvar o.F Γ σ hC v kv cl hv cur] at hassign
      cases hassign
      have hR₁ : RelState Γ (Function.update σ v (some (.s (.int kv cur)))) (Function.update τ v (some (.s (.int cur)))) :=
        relState_update hR v (RelV.s RelS.int)
      have hC₁ : Conforms o.F Γ (Function.update σ v (some (.s (.int kv cur)))) :=
        conforms_update_scalar hC hS v _ hv rfl _ rfl
      obtain ⟨τ₂, hex₂, hR₂, hC₂⟩ := hbody _ _ hR₁ hC₁ _ _ hb
      have hI₂ := hInv_body _ _ _ (hInv_upd τ cur hI) hex₂
      obtain ⟨τ', hfor, hR', hC'⟩ := ih (cur + st) _ _ hrest (by push_cast at hfin ⊢; linarith) τ₂ hR₂ hC₂ hI₂
      refine ⟨τ', ?_, hR', hC'⟩
      rw [loopVals_succ]
      exact P.ForLoop.step hex₂ (outMap_ne_brk hne) hfor
    | exit hassign hb =>
      rw [F_assign_loopvar o.F Γ σ hC v kv cl hv cur] at hassign
      cases hassign
      have hR₁ : RelState Γ (Function.update σ v (some (.s (.int kv cur)))) (Function.update τ v (some (.s (.int cur)))) :=
        relState_update hR v (RelV.s RelS.int)
      have hC₁ : Conforms o.F Γ (Function.update σ v (some (.s (.int kv cur)))) :=
        conforms_update_scalar hC hS v _ hv rfl _ rfl
      obtain ⟨τ₂, hex₂, hR₂, hC₂⟩ := hbody _ _ hR₁ hC₁ _ _ hb
      refine ⟨τ₂, ?_, hR₂, hC₂⟩
      rw [loopVals_succ]
      exact P.ForLoop.brk hex₂

/-- The step value of a DO: `1` when omitted, else the step expression's value. -/
def StepVal (o : F.Oracle) (σ : F.State) : Option F.Expr → ℤ → Prop
  | none, s => s = 1
  | some e, s => ∃ k, F.eval o σ e = some (.s (.int k s))

def PStepVal (o : P.Oracle) (τ : P.State) : Option P.Expr → ℤ → Prop
  | none, s => s = 1
  | some e, s => P.eval o τ e = some (.s (.int s))

/-- The generic counted-DO simulation with the completion `else`: the Python
`range` edges evaluate to the Fortran bounds (`hedge` gives `hi ± 1` by the
step's sign), the step agrees, and the bound expressions are stable. -/
theorem doLoop_sound_gen (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi : F.Expr) (st : Option F.Expr) (body : F.Stmt)
    (lo' hedge' stc' : P.Expr) (st' : Option P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body')
    (hlo : SoundScalar o Γ lo lo') (hilo : IntTyped o Γ lo)
    (hedges : ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ l h s k₁ k₂,
      F.eval o.F σ lo = some (.s (.int k₁ l)) → F.eval o.F σ hi = some (.s (.int k₂ h)) →
      StepVal o.F σ st s →
      P.eval o.P τ hedge' = some (.s (.int (h + if 0 < s then 1 else -1))) ∧
      PStepVal o.P τ st' s ∧
      P.eval o.P τ stc' = some (.s (.int s)))
    (hstab_lo : Stable o.P v body' lo') (hstab_st : Stable o.P v body' stc')
    (hi' : P.Expr) (hhi : SoundScalar o Γ hi hi') (hihi : IntTyped o Γ hi) (hstab_hi : Stable o.P v body' hi') :
    StmtSound o Γ (.doLoop v lo hi st body) (.forRange v lo' hedge' st' body' (some (completion v lo' hi' stc'))) := by
  intro σ τ hR hC σ' out h
  cases h with
  | doLoop hl hh hst hs0 hloop =>
    rename_i k₁ k₂ l hv' s
    obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hl
    rw [RelS.int_eq hrwl] at hwl
    obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hh
    rw [RelS.int_eq hrwh] at hwh
    have hstv : StepVal o.F σ st s := by cases st <;> exact hst
    obtain ⟨hedge, hst', hstc⟩ := hedges σ τ hR hC l hv' s k₁ k₂ hl hh hstv
    -- the invariant: the three expressions keep their values
    let Inv : P.State → Prop := fun τ₁ =>
      P.eval o.P τ₁ lo' = some (.s (.int l)) ∧ P.eval o.P τ₁ hi' = some (.s (.int hv')) ∧
      P.eval o.P τ₁ stc' = some (.s (.int s))
    have hI : Inv τ := ⟨hwl, hwh, hstc⟩
    have hInv_body : ∀ τ₁ τ₂ out, Inv τ₁ → P.Exec o.P body' τ₁ τ₂ out → Inv τ₂ := by
      intro τ₁ τ₂ out ⟨h1, h2, h3⟩ hex
      exact ⟨by rw [hstab_lo.1 _ _ _ hex]; exact h1, by rw [hstab_hi.1 _ _ _ hex]; exact h2,
        by rw [hstab_st.1 _ _ _ hex]; exact h3⟩
    have hInv_upd : ∀ τ₁ i, Inv τ₁ → Inv (Function.update τ₁ v (some (.s (.int i)))) := by
      intro τ₁ i ⟨h1, h2, h3⟩
      exact ⟨by rw [hstab_lo.2]; exact h1, by rw [hstab_hi.2]; exact h2, by rw [hstab_st.2]; exact h3⟩
    have hor : ∀ τ₁, Inv τ₁ → P.Exec o.P (completion v lo' hi' stc') τ₁
        (Function.update τ₁ v (some (.s (.int (l + (Fort.tripCount l hv' s : ℤ) * s))))) .normal := by
      intro τ₁ ⟨h1, h2, h3⟩
      exact completion_exec o.P τ₁ v lo' hi' stc' l hv' s hs0 h1 h2 h3
    obtain ⟨τ', hfor, hR', hC'⟩ := loop_sim o Γ hS v kv cl hv body body' hbody s _ _ Inv hInv_body hInv_upd hor
      (Fort.tripCount l hv' s) l σ σ' hloop rfl τ hR hC hI
    refine ⟨τ', ?_, hR', hC'⟩
    have hrange : P.pyRange l (hv' + if 0 < s then 1 else -1) s = loopVals l s (Fort.tripCount l hv' s) := by
      split_ifs with hpos
      · exact pyRange_pos l hv' s hpos
      · exact pyRange_neg l hv' s (by omega)
    rw [← hrange] at hfor
    exact P.Exec.forRange hwl hedge (by cases st' <;> exact hst') hs0 hfor

/-! ### Instances -/

def stepEdge (hi' : P.Expr) (s : ℤ) : P.Expr :=
  if 0 < s then .bin .add hi' (.intLit 1) else .bin .sub hi' (.intLit 1)

/-- FNP-R0123: unit step, `for v in range(lo', hi' + 1): body' else: v = ...`. -/
theorem FNP_R0123_sound (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi : F.Expr) (body : F.Stmt) (lo' hi' : P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body') (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hilo : IntTyped o Γ lo) (hihi : IntTyped o Γ hi)
    (hstab_lo : Stable o.P v body' lo') (hstab_hi : Stable o.P v body' hi') :
    StmtSound o Γ (.doLoop v lo hi none body)
      (.forRange v lo' (.bin .add hi' (.intLit 1)) none body' (some (completion v lo' hi' (.intLit 1)))) := by
  refine doLoop_sound_gen o Γ hS v kv cl hv lo hi none body lo' (.bin .add hi' (.intLit 1)) (.intLit 1) none body'
    hbody hlo hilo ?_ hstab_lo ⟨fun _ _ _ _ => rfl, fun _ _ => rfl⟩ hi' hhi hihi hstab_hi
  intro σ τ hR hC l h s k₁ k₂ hl hh hst
  simp only [StepVal] at hst
  subst hst
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hh
  rw [RelS.int_eq hrwh] at hwh
  refine ⟨?_, rfl, rfl⟩
  rw [P_eval_bin, hwh]
  simp [P.eval, P.lift2, P.arith]

/-- FNP-R0125 / FNP-R0126: a literal step `s ≠ 0`: `range(lo', hi' ± 1, s)`. -/
theorem FNP_R0126_sound (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv ks : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi : F.Expr) (s : ℤ) (hs : s ≠ 0) (body : F.Stmt)
    (lo' hi' : P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body') (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hilo : IntTyped o Γ lo) (hihi : IntTyped o Γ hi)
    (hstab_lo : Stable o.P v body' lo') (hstab_hi : Stable o.P v body' hi') :
    StmtSound o Γ (.doLoop v lo hi (some (.intLit ks s)) body)
      (.forRange v lo' (stepEdge hi' s) (some (.intLit s)) body' (some (completion v lo' hi' (.intLit s)))) := by
  refine doLoop_sound_gen o Γ hS v kv cl hv lo hi (some (.intLit ks s)) body lo' (stepEdge hi' s) (.intLit s)
    (some (.intLit s)) body' hbody hlo hilo ?_ hstab_lo ⟨fun _ _ _ _ => rfl, fun _ _ => rfl⟩ hi' hhi hihi hstab_hi
  intro σ τ hR hC l h s₀ k₁ k₂ hl hh hst
  simp only [StepVal] at hst
  obtain ⟨k, hk⟩ := hst
  rw [F_eval_intLit] at hk
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hk
  obtain ⟨_, rfl⟩ := hk
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hh
  rw [RelS.int_eq hrwh] at hwh
  refine ⟨?_, rfl, rfl⟩
  unfold stepEdge
  by_cases hpos : 0 < s
  · rw [if_pos hpos, if_pos hpos, P_eval_bin, hwh]; simp [P.eval, P.lift2, P.arith]
  · rw [if_neg hpos, if_neg hpos, P_eval_bin, hwh]; simp [P.eval, P.lift2, P.arith]; ring

theorem FNP_R0125_sound (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv ks : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi : F.Expr) (s : ℤ) (hs : s < 0) (body : F.Stmt)
    (lo' hi' : P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body') (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hilo : IntTyped o Γ lo) (hihi : IntTyped o Γ hi)
    (hstab_lo : Stable o.P v body' lo') (hstab_hi : Stable o.P v body' hi') :
    StmtSound o Γ (.doLoop v lo hi (some (.intLit ks s)) body)
      (.forRange v lo' (.bin .sub hi' (.intLit 1)) (some (.intLit s)) body' (some (completion v lo' hi' (.intLit s)))) := by
  have := FNP_R0126_sound o Γ hS v kv ks cl hv lo hi s (ne_of_lt hs) body lo' hi' body' hbody hlo hhi hilo hihi hstab_lo hstab_hi
  unfold stepEdge at this
  rwa [if_neg (not_lt.mpr hs.le)] at this

/-- FNP-R0127: a variable step: `range(lo', (hi') + (1 if (st') > 0 else -1), st')`. -/
theorem FNP_R0127_sound (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi st : F.Expr) (body : F.Stmt)
    (lo' hi' st' : P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body') (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hst : SoundScalar o Γ st st') (hilo : IntTyped o Γ lo) (hihi : IntTyped o Γ hi) (hist : IntTyped o Γ st)
    (hstab_lo : Stable o.P v body' lo') (hstab_hi : Stable o.P v body' hi') (hstab_st : Stable o.P v body' st') :
    StmtSound o Γ (.doLoop v lo hi (some st) body)
      (.forRange v lo' (.bin .add (.paren hi') (.paren (.ifexp (.intLit 1) (.cmp .gt (.paren st') (.intLit 0)) (.intLit (-1)))))
        (some st') body' (some (completion v lo' hi' st'))) := by
  refine doLoop_sound_gen o Γ hS v kv cl hv lo hi (some st) body lo' _ st' (some st') body'
    hbody hlo hilo ?_ hstab_lo hstab_st hi' hhi hihi hstab_hi
  intro σ τ hR hC l h s k₁ k₂ hl hh hstv
  simp only [StepVal] at hstv
  obtain ⟨k, hk⟩ := hstv
  obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hh
  rw [RelS.int_eq hrwh] at hwh
  obtain ⟨ws, hws, hrws⟩ := hst σ τ hR hC _ hk
  rw [RelS.int_eq hrws] at hws
  refine ⟨?_, hws, hws⟩
  have hcmp : P.eval o.P τ (.cmp .gt (.paren st') (.intLit 0)) = some (.s (.bool (decide (0 < s)))) := by
    rw [P_eval_cmp, P_eval_paren, hws]
    simp [P.eval, P.lift2, P.compare, P.cmpR, P.toR]
  have hif : P.eval o.P τ (.paren (.ifexp (.intLit 1) (.cmp .gt (.paren st') (.intLit 0)) (.intLit (-1)))) =
      some (.s (.int (if 0 < s then 1 else -1))) := by
    rw [P_eval_paren, P_eval_ifexp, hcmp]
    by_cases hpos : 0 < s
    · simp [hpos, P.eval]
    · simp [hpos, P.eval]
  rw [P_eval_bin, P_eval_paren, hwh, hif]
  simp [P.lift2, P.arith]

end Recast.Rules
