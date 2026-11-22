import RecastProofs.Rules.Obligations

/-!
# FNP-R0123 without the completion `else` (dead loop index), and the
literal-exponent power rules on folded literals

The emitter omits the completion assignment when its analysis
(`_loops_whose_index_is_read_after`) finds that the DO variable is not read
after the loop.  Without it the two states differ at the loop variable only
(Fortran: one step past the last value; Python: the last value, or unchanged
when the loop ran zero times).  We formalise this as a simulation *modulo the
loop variable* (`StmtSoundExcept`), prove the no-`else` loop sound in that
sense, and prove the frame lemma that lets a following statement that does not
read the variable be composed (`seq_sound_except`).

The second part restates FNP-R0019 (and the `ifx` spelling behind FNP-R0021/
R0023 with a literal exponent) for exponents written `(n)`, `-(n)`, `(-n)`:
the emitter's `integer_literal` folds them (`Tr.intLiteral`).
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### The state relation modulo one variable -/

/-- Every declared name except `x` is related. -/
def RelStateExcept (Γ : F.Env) (x : String) (σ : F.State) (τ : P.State) : Prop :=
  ∀ y d, Γ y = some d → y ≠ x → ∃ v w, σ y = some v ∧ τ y = some w ∧ RelV v w

theorem RelState.except {Γ : F.Env} {σ : F.State} {τ : P.State} (h : RelState Γ σ τ) (x : String) :
    RelStateExcept Γ x σ τ :=
  fun y d hy _ => h y d hy

/-- Forward simulation up to the value of `x` in the final states. -/
def StmtSoundExcept (o : Oracles) (Γ : F.Env) (x : String) (s : F.Stmt) (s' : P.Stmt) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ σ' out, F.Exec o.F Γ s σ σ' out →
    ∃ τ', P.Exec o.P s' τ τ' (outMap out) ∧ RelStateExcept Γ x σ' τ' ∧ Conforms o.F Γ σ'

theorem StmtSound.except {o : Oracles} {Γ : F.Env} {s : F.Stmt} {s' : P.Stmt} (h : StmtSound o Γ s s')
    (x : String) : StmtSoundExcept o Γ x s s' := by
  intro σ τ hR hC σ' out hex
  obtain ⟨τ', hex', hR', hC'⟩ := h σ τ hR hC σ' out hex
  exact ⟨τ', hex', RelState.except hR' x, hC'⟩

/-- Updating the excepted variable on the Fortran side keeps the relation modulo `x`. -/
theorem relStateExcept_updateF {Γ : F.Env} {σ : F.State} {τ : P.State} {x : String}
    (h : RelStateExcept Γ x σ τ) (v : F.Val) :
    RelStateExcept Γ x (Function.update σ x (some v)) τ := by
  intro y d hy hyx
  obtain ⟨v', w', hv', hw', hr⟩ := h y d hy hyx
  exact ⟨v', w', by simp [Function.update, hyx, hv'], hw', hr⟩

/-- Updating the excepted variable on the Python side keeps the relation modulo `x`. -/
theorem relStateExcept_updateP {Γ : F.Env} {σ : F.State} {τ : P.State} {x : String}
    (h : RelStateExcept Γ x σ τ) (w : P.Val) :
    RelStateExcept Γ x σ (Function.update τ x (some w)) := by
  intro y d hy hyx
  obtain ⟨v', w', hv', hw', hr⟩ := h y d hy hyx
  exact ⟨v', w', hv', by simp [Function.update, hyx, hw'], hr⟩

/-- Restoring the full relation by fixing the Python value of `x`. -/
theorem relState_of_except {Γ : F.Env} {σ : F.State} {τ : P.State} {x : String}
    (h : RelStateExcept Γ x σ τ) {v : F.Val} {w : P.Val} (hσ : σ x = some v) (hvw : RelV v w) :
    RelState Γ σ (Function.update τ x (some w)) := by
  intro y d hy
  by_cases hyx : y = x
  · subst hyx; exact ⟨v, w, hσ, by simp, hvw⟩
  · obtain ⟨v', w', hv', hw', hr⟩ := h y d hy hyx
    exact ⟨v', w', hv', by simp [Function.update, hyx, hw'], hr⟩

/-! ### The loop simulation without `else` -/

/-- `Loop` without a completion clause: the iterations restore the full relation
(both sides set the variable before each body); after the last iteration only
the Fortran side steps the variable once more. -/
theorem loop_sim_noelse (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (body : F.Stmt) (body' : P.Stmt) (hbody : StmtSound o Γ body body')
    (st : ℤ) :
    ∀ (n : ℕ) (cur : ℤ) (σ σ' : F.State), F.Loop o.F Γ v body st n cur σ σ' →
      ∀ τ, RelState Γ σ τ → Conforms o.F Γ σ →
        ∃ τ', P.ForLoop o.P v body' none (loopVals cur st n) τ τ' ∧ RelStateExcept Γ v σ' τ' ∧ Conforms o.F Γ σ' := by
  intro n
  induction n with
  | zero =>
    intro cur σ σ' hloop τ hR hC
    cases hloop with
    | done hassign =>
      rw [F_assign_loopvar o.F Γ σ hC v kv cl hv cur] at hassign
      cases hassign
      refine ⟨τ, P.ForLoop.done_noelse, relStateExcept_updateF (RelState.except hR v) _, ?_⟩
      exact conforms_update_scalar hC hS v _ hv rfl _ rfl
  | succ n ih =>
    intro cur σ σ' hloop τ hR hC
    cases hloop with
    | step hassign hb hne hrest =>
      rw [F_assign_loopvar o.F Γ σ hC v kv cl hv cur] at hassign
      cases hassign
      have hR₁ : RelState Γ (Function.update σ v (some (.s (.int kv cur)))) (Function.update τ v (some (.s (.int cur)))) :=
        relState_update hR v (RelV.s RelS.int)
      have hC₁ : Conforms o.F Γ (Function.update σ v (some (.s (.int kv cur)))) :=
        conforms_update_scalar hC hS v _ hv rfl _ rfl
      obtain ⟨τ₂, hex₂, hR₂, hC₂⟩ := hbody _ _ hR₁ hC₁ _ _ hb
      obtain ⟨τ', hfor, hR', hC'⟩ := ih (cur + st) _ _ hrest τ₂ hR₂ hC₂
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
      refine ⟨τ₂, ?_, RelState.except hR₂ v, hC₂⟩
      rw [loopVals_succ]
      exact P.ForLoop.brk hex₂

/-- FNP-R0123 without the completion `else`: `for v in range(lo', hi' + 1): body'`
is a simulation modulo `v`. -/
theorem FNP_R0123_sound_noelse (o : Oracles) (Γ : F.Env) (hS : StaticBounds Γ) (v : String) (kv : IKind) (cl : ℕ)
    (hv : Γ v = some ⟨.int kv, [], cl⟩) (lo hi : F.Expr) (body : F.Stmt) (lo' hi' : P.Expr) (body' : P.Stmt)
    (hbody : StmtSound o Γ body body') (hlo : SoundScalar o Γ lo lo') (hhi : SoundScalar o Γ hi hi')
    (hilo : IntTyped o Γ lo) (hihi : IntTyped o Γ hi) :
    StmtSoundExcept o Γ v (.doLoop v lo hi none body)
      (.forRange v lo' (.bin .add hi' (.intLit 1)) none body' none) := by
  intro σ τ hR hC σ' out h
  cases h with
  | doLoop hl hh hst hs0 hloop =>
    rename_i k₁ k₂ l hv' s
    obtain ⟨wl, hwl, hrwl⟩ := hlo σ τ hR hC _ hl
    rw [RelS.int_eq hrwl] at hwl
    obtain ⟨wh, hwh, hrwh⟩ := hhi σ τ hR hC _ hh
    rw [RelS.int_eq hrwh] at hwh
    have hs1 : s = 1 := hst
    subst hs1
    obtain ⟨τ', hfor, hR', hC'⟩ := loop_sim_noelse o Γ hS v kv cl hv body body' hbody 1
      (Fort.tripCount l hv' 1) l σ σ' hloop τ hR hC
    refine ⟨τ', ?_, hR', hC'⟩
    have hedge : P.eval o.P τ (.bin .add hi' (.intLit 1)) = some (.s (.int (hv' + 1))) := by
      rw [P_eval_bin, hwh]; simp [P.eval, P.lift2, P.arith]
    rw [← pyRange_pos l hv' 1 (by norm_num)] at hfor
    exact P.Exec.forRange hwl hedge rfl (by norm_num) hfor

/-! ### The Python frame lemma: a statement that does not read `x` -/

mutual
  /-- Names an expression reads. -/
  def preads : P.Expr → List String
    | .intLit _ | .floatLit _ | .boolLit _ | .strLit _ | .none => []
    | .name x => [x]
    | .neg e | .paren e | .not e | .invert e => preads e
    | .bin _ a b | .cmp _ a b | .and a b | .or a b | .bitand a b | .bitor a b => preads a ++ preads b
    | .ifexp t c e => preads t ++ preads c ++ preads e
    | .index a idx => a :: preadsIdxs idx
    | .indexBy a e => a :: preads e
    | .call _ args => preadsExprs args
    | .call1k _ arg kw => preads arg ++ preads kw
    | .maskIndex base m => preads base ++ preads m
  def preadsExprs : P.Exprs → List String
    | .nil => []
    | .cons e rest => preads e ++ preadsExprs rest
  def preadsBound : P.Bound → List String
    | .none => []
    | .some e => preads e
  def preadsIdx : P.Idx → List String
    | .scalar e => preads e
    | .slice s e st => preadsBound s ++ preadsBound e ++ preadsBound st
    | .array e => preads e
  def preadsIdxs : P.Idxs → List String
    | .nil => []
    | .cons i rest => preadsIdx i ++ preadsIdxs rest
end

mutual
  /-- Names a statement reads (an indexed store reads the array it updates). -/
  def preadsS : P.Stmt → List String
    | .assign _ e => preads e
    | .assignIndex x idx e => x :: (preadsIdxs idx ++ preads e)
    | .assignAll x e => x :: preads e
    | .assignMask x m e => x :: (preads m ++ preads e)
    | .callAssign _ _ args => preadsExprs args
    | .seq a b => preadsS a ++ preadsS b
    | .pass | .break | .continue => []
    | .ite c t e => preads c ++ preadsS t ++ preadsS e
    | .forRange _ lo hi st body orelse => preads lo ++ preads hi ++ preadsOptE st ++ preadsS body ++ preadsOptS orelse
    | .while c body => preads c ++ preadsS body
  def preadsOptE : Option P.Expr → List String
    | none => []
    | some e => preads e
  def preadsOptS : Option P.Stmt → List String
    | none => []
    | some s => preadsS s
end

/-- Two states agreeing on a set of names. -/
def AgreeOn (τ₁ τ₂ : P.State) (names : List String) : Prop := ∀ y ∈ names, τ₁ y = τ₂ y

/-- Two states agreeing everywhere except possibly at `x`. -/
def AgreeExcept (x : String) (τ₁ τ₂ : P.State) : Prop := ∀ y, y ≠ x → τ₁ y = τ₂ y

theorem agreeExcept_agreeOn {x : String} {τ₁ τ₂ : P.State} (h : AgreeExcept x τ₁ τ₂) (names : List String)
    (hx : x ∉ names) : AgreeOn τ₁ τ₂ names :=
  fun y hy => h y (fun hyx => hx (hyx ▸ hy))

theorem agreeOn_append_left {τ₁ τ₂ : P.State} {l₁ l₂ : List String} (h : AgreeOn τ₁ τ₂ (l₁ ++ l₂)) :
    AgreeOn τ₁ τ₂ l₁ := fun y hy => h y (List.mem_append_left _ hy)

theorem agreeOn_append_right {τ₁ τ₂ : P.State} {l₁ l₂ : List String} (h : AgreeOn τ₁ τ₂ (l₁ ++ l₂)) :
    AgreeOn τ₁ τ₂ l₂ := fun y hy => h y (List.mem_append_right _ hy)

theorem agreeOn_cons_head {τ₁ τ₂ : P.State} {y : String} {l : List String} (h : AgreeOn τ₁ τ₂ (y :: l)) :
    τ₁ y = τ₂ y := h y (List.mem_cons_self ..)

theorem agreeOn_cons_tail {τ₁ τ₂ : P.State} {y : String} {l : List String} (h : AgreeOn τ₁ τ₂ (y :: l)) :
    AgreeOn τ₁ τ₂ l := fun z hz => h z (List.mem_cons_of_mem _ hz)

mutual
  /-- Evaluation depends only on the names read. -/
  theorem eval_agree (o : P.Oracle) (τ₁ τ₂ : P.State) : ∀ (e : P.Expr), AgreeOn τ₁ τ₂ (preads e) →
      P.eval o τ₁ e = P.eval o τ₂ e
    | .intLit _, _ => rfl
    | .floatLit _, _ => rfl
    | .boolLit _, _ => rfl
    | .strLit _, _ => rfl
    | .none, _ => rfl
    | .name x, h => by simp only [P.eval]; exact agreeOn_cons_head h
    | .neg e, h => by simp only [P.eval]; rw [eval_agree o τ₁ τ₂ e h]
    | .paren e, h => by simp only [P.eval]; exact eval_agree o τ₁ τ₂ e h
    | .not e, h => by simp only [P.eval]; rw [eval_agree o τ₁ τ₂ e h]
    | .invert e, h => by simp only [P.eval]; rw [eval_agree o τ₁ τ₂ e h]
    | .bin op a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .cmp op a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .and a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .or a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .bitand a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .bitor a b, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ a (agreeOn_append_left h), eval_agree o τ₁ τ₂ b (agreeOn_append_right h)]
    | .ifexp t c e, h => by
        simp only [P.eval]
        have ht := agreeOn_append_left (agreeOn_append_left h)
        have hc := agreeOn_append_right (agreeOn_append_left h)
        have he := agreeOn_append_right h
        rw [eval_agree o τ₁ τ₂ t ht, eval_agree o τ₁ τ₂ c hc, eval_agree o τ₁ τ₂ e he]
    | .index a idx, h => by
        simp only [P.eval]
        rw [agreeOn_cons_head h, evalIdxs_agree o τ₁ τ₂ idx (agreeOn_cons_tail h)]
    | .indexBy a e, h => by
        simp only [P.eval]
        rw [agreeOn_cons_head h, eval_agree o τ₁ τ₂ e (agreeOn_cons_tail h)]
    | .call f args, h => by
        simp only [P.eval]
        rw [evalExprs_agree o τ₁ τ₂ args h]
    | .call1k f arg kw, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ arg (agreeOn_append_left h), eval_agree o τ₁ τ₂ kw (agreeOn_append_right h)]
    | .maskIndex base m, h => by
        simp only [P.eval]
        rw [eval_agree o τ₁ τ₂ base (agreeOn_append_left h), eval_agree o τ₁ τ₂ m (agreeOn_append_right h)]
  theorem evalExprs_agree (o : P.Oracle) (τ₁ τ₂ : P.State) : ∀ (es : P.Exprs), AgreeOn τ₁ τ₂ (preadsExprs es) →
      P.evalExprs o τ₁ es = P.evalExprs o τ₂ es
    | .nil, _ => rfl
    | .cons e rest, h => by
        simp only [P.evalExprs]
        rw [eval_agree o τ₁ τ₂ e (agreeOn_append_left h), evalExprs_agree o τ₁ τ₂ rest (agreeOn_append_right h)]
  theorem evalBound_agree (o : P.Oracle) (τ₁ τ₂ : P.State) : ∀ (b : P.Bound), AgreeOn τ₁ τ₂ (preadsBound b) →
      P.evalBound o τ₁ b = P.evalBound o τ₂ b
    | .none, _ => rfl
    | .some e, h => by simp only [P.evalBound]; rw [eval_agree o τ₁ τ₂ e h]
  theorem evalIdx_agree (o : P.Oracle) (τ₁ τ₂ : P.State) : ∀ (i : P.Idx), AgreeOn τ₁ τ₂ (preadsIdx i) →
      P.evalIdx o τ₁ i = P.evalIdx o τ₂ i
    | .scalar e, h => by simp only [P.evalIdx]; rw [eval_agree o τ₁ τ₂ e h]
    | .slice s e st, h => by
        simp only [P.evalIdx]
        have hs := agreeOn_append_left (agreeOn_append_left h)
        have he := agreeOn_append_right (agreeOn_append_left h)
        have hst := agreeOn_append_right h
        rw [evalBound_agree o τ₁ τ₂ s hs, evalBound_agree o τ₁ τ₂ e he, evalBound_agree o τ₁ τ₂ st hst]
    | .array e, h => by simp only [P.evalIdx]; rw [eval_agree o τ₁ τ₂ e h]
  theorem evalIdxs_agree (o : P.Oracle) (τ₁ τ₂ : P.State) : ∀ (is : P.Idxs), AgreeOn τ₁ τ₂ (preadsIdxs is) →
      P.evalIdxs o τ₁ is = P.evalIdxs o τ₂ is
    | .nil, _ => rfl
    | .cons i rest, h => by
        simp only [P.evalIdxs]
        rw [evalIdx_agree o τ₁ τ₂ i (agreeOn_append_left h), evalIdxs_agree o τ₁ τ₂ rest (agreeOn_append_right h)]
end

theorem agreeExcept_update {x : String} {τ₁ τ₂ : P.State} (h : AgreeExcept x τ₁ τ₂) (y : String) (v : Option P.Val) :
    AgreeExcept x (Function.update τ₁ y v) (Function.update τ₂ y v) := by
  intro z hz
  by_cases hzy : z = y
  · subst hzy; simp
  · simp [Function.update, hzy, h z hz]

theorem agreeExcept_refl (x : String) (τ : P.State) : AgreeExcept x τ τ := fun _ _ => rfl

theorem eval_agree_except (o : P.Oracle) {x : String} {τ₁ τ₂ : P.State} (h : AgreeExcept x τ₁ τ₂) (e : P.Expr)
    (hx : x ∉ preads e) : P.eval o τ₁ e = P.eval o τ₂ e :=
  eval_agree o τ₁ τ₂ e (agreeExcept_agreeOn h _ hx)

/-- Assigning the same values from states that agree off `x` keeps them agreeing. -/
theorem assignTargets_agree {x : String} : ∀ (targets : List String) (vs : List P.Val) (τ₁ τ₁' τ₂ : P.State),
    P.assignTargets targets vs τ₁ = some τ₁' → AgreeExcept x τ₁ τ₂ →
    ∃ τ₂', P.assignTargets targets vs τ₂ = some τ₂' ∧ AgreeExcept x τ₁' τ₂'
  | [], [], τ₁, τ₁', τ₂, h, hag => by
      simp only [P.assignTargets, Option.some.injEq] at h; subst h; exact ⟨τ₂, rfl, hag⟩
  | y :: ys, v :: vs, τ₁, τ₁', τ₂, h, hag => by
      simp only [P.assignTargets] at h
      obtain ⟨τ₂', h₂, hag'⟩ := assignTargets_agree ys vs _ τ₁' _ h (agreeExcept_update hag y _)
      exact ⟨τ₂', by simpa only [P.assignTargets] using h₂, hag'⟩
  | [], _ :: _, _, _, _, h, _ => by simp [P.assignTargets] at h
  | _ :: _, [], _, _, _, h, _ => by simp [P.assignTargets] at h

/-- A statement that does not read `x` executes the same way from states that
agree off `x`, and ends in states that agree off `x`. -/
theorem exec_frame (o : P.Oracle) (x : String) (s : P.Stmt) (τ₁ τ₁' : P.State) (out : P.Outcome)
    (h : P.Exec o s τ₁ τ₁' out) : x ∉ preadsS s → ∀ τ₂, AgreeExcept x τ₁ τ₂ →
      ∃ τ₂', P.Exec o s τ₂ τ₂' out ∧ AgreeExcept x τ₁' τ₂' := by
  refine P.Exec.rec (motive_1 := fun s τ₁ τ₁' out _ => x ∉ preadsS s → ∀ τ₂, AgreeExcept x τ₁ τ₂ →
      ∃ τ₂', P.Exec o s τ₂ τ₂' out ∧ AgreeExcept x τ₁' τ₂')
    (motive_2 := fun v body orelse vals τ₁ τ₁' _ => x ∉ preadsS body → x ∉ preadsOptS orelse →
      ∀ τ₂, AgreeExcept x τ₁ τ₂ → ∃ τ₂', P.ForLoop o v body orelse vals τ₂ τ₂' ∧ AgreeExcept x τ₁' τ₂')
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ h
  · -- pass
    intro τ _ τ₂ hag; exact ⟨τ₂, P.Exec.pass, hag⟩
  · -- assign
    intro τ y e v hv hx τ₂ hag
    simp only [preadsS] at hx
    refine ⟨_, P.Exec.assign (by rw [← eval_agree_except o hag e hx]; exact hv), agreeExcept_update hag y _⟩
  · -- assignAll
    intro τ y e v a a' hv hτ hst hx τ₂ hag
    simp only [preadsS, List.mem_cons, not_or] at hx
    have hy : τ₂ y = some (.arr a) := by rw [← hag y (Ne.symm hx.1)]; exact hτ
    refine ⟨_, P.Exec.assignAll (by rw [← eval_agree_except o hag e hx.2]; exact hv) hy hst, agreeExcept_update hag y _⟩
  · -- assignIndex
    intro τ y idx e v a ps a' hv hτ hidx hst hx τ₂ hag
    simp only [preadsS, List.mem_cons, List.mem_append, not_or] at hx
    have hy : τ₂ y = some (.arr a) := by rw [← hag y (Ne.symm hx.1)]; exact hτ
    have hidx' : P.evalIdxs o τ₂ idx = some ps := by
      rw [← evalIdxs_agree o τ τ₂ idx (agreeExcept_agreeOn hag _ hx.2.1)]; exact hidx
    refine ⟨_, P.Exec.assignIndex (by rw [← eval_agree_except o hag e hx.2.2]; exact hv) hy hidx' hst,
      agreeExcept_update hag y _⟩
  · -- assignMask
    intro τ y m e v a mv a' hv hτ hm hst hx τ₂ hag
    simp only [preadsS, List.mem_cons, List.mem_append, not_or] at hx
    have hy : τ₂ y = some (.arr a) := by rw [← hag y (Ne.symm hx.1)]; exact hτ
    refine ⟨_, P.Exec.assignMask (by rw [← eval_agree_except o hag e hx.2.2]; exact hv) hy
      (by rw [← eval_agree_except o hag m hx.2.1]; exact hm) hst, agreeExcept_update hag y _⟩
  · -- seq_normal
    intro s₁ s₂ τ τ₁ τ₂ out _ _ ih₁ ih₂ hx τ₂' hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih₁ hx.1 τ₂' hag
    obtain ⟨τb, hexb, hagb⟩ := ih₂ hx.2 τa haga
    exact ⟨τb, P.Exec.seq_normal hexa hexb, hagb⟩
  · -- seq_abrupt
    intro s₁ s₂ τ τ₁ out _ hne ih₁ hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih₁ hx.1 τ₂ hag
    exact ⟨τa, P.Exec.seq_abrupt hexa hne, haga⟩
  · -- ite_true
    intro c t e τ τ' out hc _ ih hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih hx.1.2 τ₂ hag
    exact ⟨τa, P.Exec.ite_true (by rw [← eval_agree_except o hag c hx.1.1]; exact hc) hexa, haga⟩
  · -- ite_false
    intro c t e τ τ' out hc _ ih hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih hx.2 τ₂ hag
    exact ⟨τa, P.Exec.ite_false (by rw [← eval_agree_except o hag c hx.1.1]; exact hc) hexa, haga⟩
  · -- break
    intro τ _ τ₂ hag; exact ⟨τ₂, P.Exec.brk, hag⟩
  · -- continue
    intro τ _ τ₂ hag; exact ⟨τ₂, P.Exec.cont, hag⟩
  · -- forRange
    intro v lo hi st body orelse τ τ' l h s hlo hhi hst hs0 _ ih hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨⟨⟨⟨hxlo, hxhi⟩, hxst⟩, hxb⟩, hxe⟩ := hx
    obtain ⟨τa, hfor, haga⟩ := ih hxb hxe τ₂ hag
    refine ⟨τa, P.Exec.forRange (by rw [← eval_agree_except o hag lo hxlo]; exact hlo)
      (by rw [← eval_agree_except o hag hi hxhi]; exact hhi) ?_ hs0 hfor, haga⟩
    cases st with
    | none => exact hst
    | some e =>
      simp only [preadsOptE] at hxst
      simp only at hst ⊢
      rw [← eval_agree_except o hag e hxst]; exact hst
  · -- while_false
    intro c body τ hc hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    exact ⟨τ₂, P.Exec.while_false (by rw [← eval_agree_except o hag c hx.1]; exact hc), hag⟩
  · -- while_break
    intro c body τ τ' hc _ ih hx τ₂ hag
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih hx.2 τ₂ hag
    exact ⟨τa, P.Exec.while_break (by rw [← eval_agree_except o hag c hx.1]; exact hc) hexa, haga⟩
  · -- while_step
    intro c body τ τ₁ τ' out hc _ hne _ ih₁ ih₂ hx τ₂ hag
    have hx' := hx
    simp only [preadsS, List.mem_append, not_or] at hx
    obtain ⟨τa, hexa, haga⟩ := ih₁ hx.2 τ₂ hag
    obtain ⟨τb, hexb, hagb⟩ := ih₂ hx' τa haga
    exact ⟨τb, P.Exec.while_step (by rw [← eval_agree_except o hag c hx.1]; exact hc) hexa hne hexb, hagb⟩
  · -- callAssign: the arguments are read, the callee runs in its own scope
    intro _ _ _ _ _ _ _ _ _ _ hproc hargs hbind hbody hret hassign _ hx τ₂ hag
    simp only [preadsS] at hx
    obtain ⟨τ₂', h₂, hag₂⟩ := assignTargets_agree _ _ _ _ τ₂ hassign hag
    refine ⟨τ₂', P.Exec.callAssign hproc ?_ hbind hbody hret h₂, hag₂⟩
    rw [← evalExprs_agree o _ τ₂ _ (agreeExcept_agreeOn hag _ hx)]
    exact hargs
  · -- done_noelse
    intro v body τ _ _ τ₂ hag; exact ⟨τ₂, P.ForLoop.done_noelse, hag⟩
  · -- done_else
    intro v body e τ τ' _ ih _ hxe τ₂ hag
    simp only [preadsOptS] at hxe
    obtain ⟨τa, hexa, haga⟩ := ih hxe τ₂ hag
    exact ⟨τa, P.ForLoop.done_else hexa, haga⟩
  · -- step
    intro v body orelse i rest τ τ₁ τ' out _ hne _ ih_body ih_rest hxb hxe τ₂ hag
    obtain ⟨τa, hexa, haga⟩ := ih_body hxb _ (agreeExcept_update hag v _)
    obtain ⟨τb, hexb, hagb⟩ := ih_rest hxb hxe τa haga
    exact ⟨τb, P.ForLoop.step hexa hne hexb, hagb⟩
  · -- brk
    intro v body orelse i rest τ τ₁ _ ih hxb _ τ₂ hag
    obtain ⟨τa, hexa, haga⟩ := ih hxb _ (agreeExcept_update hag v _)
    exact ⟨τa, P.ForLoop.brk hexa, haga⟩

/-! ### Composition: a dead loop index followed by statements that do not read it -/

/-- `s₁ ; s₂` where `s₁` is sound modulo `x` (an INTEGER scalar) and `s₂` is
sound and its translation does not read `x`: the sequence is sound modulo `x`. -/
theorem seq_sound_except (o : Oracles) (Γ : F.Env) (x : String) (kx : IKind) (cl : ℕ)
    (hx : Γ x = some ⟨.int kx, [], cl⟩) (s₁ s₂ : F.Stmt) (s₁' s₂' : P.Stmt)
    (h₁ : StmtSoundExcept o Γ x s₁ s₁') (h₂ : StmtSound o Γ s₂ s₂') (hread : x ∉ preadsS s₂') :
    StmtSoundExcept o Γ x (.seq s₁ s₂) (.seq s₁' s₂') := by
  intro σ τ hR hC σ' out h
  cases h with
  | seq_normal ha hb =>
    rename_i σ₁
    obtain ⟨τ₁, hex₁, hR₁, hC₁⟩ := h₁ σ τ hR hC _ _ ha
    -- fix the Python value of `x` to restore the full relation
    obtain ⟨sv, hσx, hty⟩ := conforms_scalar hC₁ x _ hx rfl
    obtain ⟨n, hn⟩ : ∃ n : ℤ, sv = .int kx n := by
      cases sv with
      | int k m => simp only [SVal.ty, FTy.int.injEq] at hty; exact ⟨m, by rw [hty]⟩
      | real _ _ => simp [SVal.ty] at hty
      | bool _ => simp [SVal.ty] at hty
      | str _ => simp [SVal.ty] at hty
    subst hn
    have hR₁' : RelState Γ σ₁ (Function.update τ₁ x (some (.s (.int n)))) :=
      relState_of_except hR₁ hσx (RelV.s RelS.int)
    obtain ⟨τ₂', hex₂', hR₂', hC₂⟩ := h₂ _ _ hR₁' hC₁ _ _ hb
    have hag : AgreeExcept x (Function.update τ₁ x (some (.s (.int n)))) τ₁ := by
      intro y hy; simp [Function.update, hy]
    obtain ⟨τ₂, hex₂, hag₂⟩ := exec_frame o.P x s₂' _ _ _ hex₂' hread τ₁ hag
    refine ⟨τ₂, P.Exec.seq_normal hex₁ hex₂, ?_, hC₂⟩
    intro y d hy hyx
    obtain ⟨v, w, hv, hw, hr⟩ := hR₂' y d hy
    exact ⟨v, w, hv, by rw [← hag₂ y hyx]; exact hw, hr⟩
  | seq_abrupt ha hne =>
    obtain ⟨τ₁, hex₁, hR₁, hC₁⟩ := h₁ σ τ hR hC _ _ ha
    exact ⟨τ₁, P.Exec.seq_abrupt hex₁ (outMap_ne_normal hne), hR₁, hC₁⟩

/-! ### FNP-R0019 / R0021 / R0023 with folded literal exponents -/

/-- FNP-R0019 for any exponent expression the emitter reads as the literal `n`
(`Tr.intLiteral`: a literal through parentheses and unary minus). -/
theorem FNP_R0019_sound_lit (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' : P.Expr) (n : ℤ) (hn : n ≠ 0)
    (hb : Tr.intLiteral b = some n) (ha : SoundScalar o Γ a a') (hreal : RealTyped o Γ a) :
    SoundScalar o Γ (.bin .pow a b) (.paren (.bin .pow a' (.intLit n))) :=
  pow_real_lit_sound o Γ a b a' n hn ha hreal (fun σ => intLiteral_eval o.F σ b n hb)

/-- FNP-R0021 (gfortran expansion) for any exponent expression read as the literal `n`. -/
theorem FNP_R0021_sound_lit (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' : P.Expr) (n : ℤ) (hn : n ≠ 0)
    (hn16 : n.natAbs ≤ 16) (hb : Tr.intLiteral b = some n) (ha : SoundScalar o Γ a a') (hreal : RealTyped o Γ a) :
    SoundScalar o Γ (.bin .pow a b) (expandPowerWF a' n) := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨k, hk⟩ := intLiteral_eval o.F σ b n hb
  rw [hk] at hvb
  simp only [Option.some.injEq] at hvb
  subst hvb
  have := FNP_R0021_sound o Γ a a' n hn hn16 ha hreal k σ τ hR hC v (by
    rw [F_eval_bin, hva, F_eval_intLit]; simpa using hlift)
  exact this

/-- FNP-R0020 (positive literal, INTEGER base) for any exponent expression read as the literal `n > 0`. -/
theorem FNP_R0020_sound_pos_lit (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' : P.Expr) (n : ℤ) (hn : 0 < n)
    (hb : Tr.intLiteral b = some n) (ha : SoundScalar o Γ a a') (hint : IntTyped o Γ a) :
    SoundScalar o Γ (.bin .pow a b) (.paren (.bin .pow a' (.intLit n))) := by
  intro σ τ hR hC v hv
  rw [F_eval_bin] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨k, hk⟩ := intLiteral_eval o.F σ b n hb
  rw [hk] at hvb
  simp only [Option.some.injEq] at hvb
  subst hvb
  exact FNP_R0020_sound_pos o Γ a a' n hn ha hint k σ τ hR hC v (by
    rw [F_eval_bin, hva, F_eval_intLit]; simpa using hlift)

end Recast.Rules
