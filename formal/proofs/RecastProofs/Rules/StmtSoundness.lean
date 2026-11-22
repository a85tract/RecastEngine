import RecastProofs.Rules.Soundness

/-!
# Statement composition: the modelled statement emitter is sound on the fragment

`transStmt` mirrors `statements.py`'s decisions for the statement fragment
(scalar/array/element assignment, IF, counted DO with the completion `else`,
DO WHILE, EXIT/CYCLE); `WFS` is the well-formedness the fragment needs;
`transStmt_sound` composes the statement rules along the syntax, discharging
the loop-bound stability premise from a syntactic write set.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Written variables -/

mutual
def pwrites : P.Stmt → List String
  | .assign x _ => [x]
  | .assignIndex x _ _ => [x]
  | .assignAll x _ => [x]
  | .assignMask x _ _ => [x]
  | .callAssign targets _ _ => targets
  | .seq a b => pwrites a ++ pwrites b
  | .pass => []
  | .ite _ t e => pwrites t ++ pwrites e
  | .forRange v _ _ _ body orelse => v :: (pwrites body ++ pwritesOpt orelse)
  | .while _ body => pwrites body
  | .break => []
  | .continue => []
def pwritesOpt : Option P.Stmt → List String
  | none => []
  | some s => pwrites s
end

/-- Names a statement (or its translation, for the mask temporary `_wm`) may write. -/
def fwrites : F.Stmt → List String
  | .assign x _ _ => [x]
  | .seq a b => fwrites a ++ fwrites b
  | .skip => []
  | .ite _ t e => fwrites t ++ fwrites e
  | .doLoop v _ _ _ body => v :: fwrites body
  | .doWhile _ body => fwrites body
  | .exit => []
  | .cycle => []
  | .forall1 v _ _ _ _ target _ _ => [v, target]
  | .whereBlock _ asgs els => "_wm" :: (asgs.map Prod.fst ++ els.map Prod.fst)  -- `_wm` is the translation's mask temporary
  | .caseArm _ _ body rest => fwrites body ++ fwrites rest
  | .associate n x body => n :: x :: fwrites body
  | .allocate x _ => [x]
  | .call _ acts => acts.filterMap fun a => match a with | .var x => some x | .expr _ => none

/-- `assignTargets` touches only its targets. -/
theorem assignTargets_unwritten : ∀ (targets : List String) (vs : List P.Val) (τ τ' : P.State),
    P.assignTargets targets vs τ = some τ' → ∀ x, x ∉ targets → τ' x = τ x
  | [], [], τ, τ', h, x, _ => by simp only [P.assignTargets, Option.some.injEq] at h; subst h; rfl
  | y :: ys, v :: vs, τ, τ', h, x, hx => by
      simp only [P.assignTargets] at h
      simp only [List.mem_cons, not_or] at hx
      rw [assignTargets_unwritten ys vs _ τ' h x hx.2]
      exact Function.update_of_ne hx.1 _ _
  | [], _ :: _, _, _, h, _, _ => by simp [P.assignTargets] at h
  | _ :: _, [], _, _, h, _, _ => by simp [P.assignTargets] at h

/-- Variables a statement does not write keep their values. -/
theorem exec_unwritten (o : P.Oracle) (s : P.Stmt) (τ τ' : P.State) (out : P.Outcome)
    (h : P.Exec o s τ τ' out) : ∀ x, x ∉ pwrites s → τ' x = τ x := by
  refine P.Exec.rec (motive_1 := fun s τ τ' out _ => ∀ x, x ∉ pwrites s → τ' x = τ x)
    (motive_2 := fun v body orelse _ τ τ' _ => ∀ x, x ≠ v → x ∉ pwrites body → x ∉ pwritesOpt orelse → τ' x = τ x)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ h
  · intro τ x _; rfl
  · intro τ y e v _ x hx; simp [pwrites] at hx; exact Function.update_of_ne hx _ _
  · intro τ y e v a a' _ _ _ x hx; simp [pwrites] at hx; exact Function.update_of_ne hx _ _
  · intro τ y idx e v a ps a' _ _ _ _ x hx; simp [pwrites] at hx; exact Function.update_of_ne hx _ _
  · intro τ y m e v a mv a' _ _ _ _ x hx; simp [pwrites] at hx; exact Function.update_of_ne hx _ _
  · intro s₁ s₂ τ τ₁ τ₂ out _ _ ih₁ ih₂ x hx
    simp only [pwrites, List.mem_append, not_or] at hx
    rw [ih₂ x hx.2, ih₁ x hx.1]
  · intro s₁ s₂ τ τ₁ out _ _ ih₁ x hx
    simp only [pwrites, List.mem_append, not_or] at hx
    exact ih₁ x hx.1
  · intro c t e τ τ' out _ _ ih x hx
    simp only [pwrites, List.mem_append, not_or] at hx
    exact ih x hx.1
  · intro c t e τ τ' out _ _ ih x hx
    simp only [pwrites, List.mem_append, not_or] at hx
    exact ih x hx.2
  · intro τ x _; rfl
  · intro τ x _; rfl
  · intro v lo hi st body orelse τ τ' l h s _ _ _ _ _ ih x hx
    simp only [pwrites, List.mem_cons, List.mem_append, not_or] at hx
    exact ih x hx.1 hx.2.1 hx.2.2
  · intro c body τ _ x _; rfl
  · intro c body τ τ' _ _ ih x hx; exact ih x hx
  · intro c body τ τ₁ τ' out _ _ _ _ ih₁ ih₂ x hx
    rw [ih₂ x hx, ih₁ x hx]
  · -- callAssign: only the targets are written
    intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hassign _ x hx
    simp only [pwrites] at hx
    exact assignTargets_unwritten _ _ _ _ hassign x hx
  · intro v body τ x _ _ _; rfl
  · intro v body e τ τ' _ ih x _ _ hxe; exact ih x (by simpa [pwritesOpt] using hxe)
  · intro v body orelse i rest τ τ₁ τ' out _ _ _ ih_body ih_rest x hxv hxb hxe
    rw [ih_rest x hxv hxb hxe, ih_body x hxb]
    exact Function.update_of_ne hxv _ _
  · intro v body orelse i rest τ τ₁ _ ih x hxv hxb _
    rw [ih x hxb]
    exact Function.update_of_ne hxv _ _

theorem stable_lit (o : P.Oracle) (v : String) (body' : P.Stmt) (n : ℤ) : Stable o v body' (.intLit n) :=
  ⟨fun _ _ _ _ => rfl, fun _ _ => rfl⟩

theorem stable_name (o : P.Oracle) (v : String) (body' : P.Stmt) (x : String) (hxv : x ≠ v)
    (hx : x ∉ pwrites body') : Stable o v body' (.name x) :=
  ⟨fun τ τ' out h => by simp only [P.eval]; exact exec_unwritten o body' τ τ' out h x hx,
   fun τ i => by simp only [P.eval]; exact Function.update_of_ne hxv _ _⟩

/-! ### The statement fragment and its translation -/

/-- A DO bound the fragment accepts: a literal, or a variable the body does not write. -/
def LoopBound (v : String) (body : F.Stmt) : F.Expr → Prop
  | .intLit _ _ => True
  | .var x => x ≠ v ∧ x ∉ fwrites body
  | _ => False

/-- Static compatibility of an RHS type with the target's declared type. -/
def StaticCompat (ty : FTy) (t : Option FTy) : Prop :=
  ∃ te, t = some te ∧ ((te.base = ty.base ∧ ty ≠ .str) ∨ (ty.base = .real ∧ te.base = .int))

def WFS (Γ : F.Env) : F.Stmt → Prop
  | .skip => True
  | .exit => True
  | .cycle => True
  | .seq s₁ s₂ => WFS Γ s₁ ∧ WFS Γ s₂
  | .ite c t e => WFE Γ c ∧ WFS Γ t ∧ WFS Γ e
  | .doWhile c body => WFE Γ c ∧ WFS Γ body
  | .doLoop v lo hi none body => (∃ kv cl, Γ v = some ⟨.int kv, [], cl⟩) ∧ WFE Γ lo ∧ WFE Γ hi ∧
      isIntTy (Tr.tyOf Γ lo) ∧ isIntTy (Tr.tyOf Γ hi) ∧ LoopBound v body lo ∧ LoopBound v body hi ∧ WFS Γ body
  | .doLoop _ _ _ (some _) _ => False
  | .assign x .nil e => WFE Γ e ∧ ∃ d, Γ x = some d ∧ StaticCompat d.ty (Tr.tyOf Γ e) ∧
      (d.lbs = [] ∨ ∃ k, d.lbs = [.intLit k 1])
  | .assign a (.cons (.scalar i) .nil) e => WFE Γ e ∧ WFE Γ i ∧ isIntTy (Tr.tyOf Γ i) ∧
      ∃ d k, Γ a = some d ∧ d.lbs = [.intLit k 1] ∧ StaticCompat d.ty (Tr.tyOf Γ e)
  | .assign _ _ _ => False
  | .forall1 .. => False
  | .whereBlock .. => False
  | .caseArm .. => False
  | .associate .. => False
  | .allocate .. => False
  | .call .. => False

def listToExprs : List P.Expr → P.Exprs
  | [] => .nil
  | e :: rest => .cons e (listToExprs rest)

/-- The masked assignments of a WHERE body under the rendered mask `m'`
(`Statements._masked_assignment`): array RHS gathered by the mask, scalar RHS broadcast. -/
def transMasked (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (m' : P.Expr) :
    List (String × F.Expr) → Option P.Stmt
  | [] => some .pass
  | (a, e) :: rest => (Tr.trans Tr.ifx Γ Λ consts e).bind fun e' => (transMasked Γ Λ consts m' rest).bind fun rest' =>
      let rhs := if (Tr.rankOf Γ e).getD 0 > 0 then P.Expr.maskIndex (.paren e') m' else e'
      some (.seq (.assignMask a m' rhs) rest')

/-- One `CASE` item as a Python condition (`Statements._case_construct`): `(s' == v')`,
`((lo' <= s') and (s' <= hi'))`; character selectors compare with `_fstr_eq`; a character
*range* is refused. -/
def transCaseItem (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (sel : F.Expr) (sel' : P.Expr) :
    F.CaseItem → Option P.Expr
  | .value v => (Tr.trans Tr.ifx Γ Λ consts v).map fun v' =>
      if Tr.isStr (Tr.tyOf Γ sel) then Tr.call2 .fstr_eq sel' v' else .paren (.cmp .eq sel' v')
  | .range lo hi =>
      if Tr.isStr (Tr.tyOf Γ sel) then none else
      match lo, hi with
      | some l, some h => (Tr.trans Tr.ifx Γ Λ consts l).bind fun l' => (Tr.trans Tr.ifx Γ Λ consts h).map fun h' =>
          .paren (.and (.paren (.cmp .le l' sel')) (.paren (.cmp .le sel' h')))
      | some l, none => (Tr.trans Tr.ifx Γ Λ consts l).map fun l' => .paren (.paren (.cmp .le l' sel'))
      | none, some h => (Tr.trans Tr.ifx Γ Λ consts h).map fun h' => .paren (.paren (.cmp .le sel' h'))
      | none, none => none

def transCaseCond (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (sel : F.Expr) (sel' : P.Expr) :
    List F.CaseItem → Option P.Expr
  | [] => some (.boolLit false)
  | [it] => transCaseItem Γ Λ consts sel sel' it
  | it :: rest => (transCaseItem Γ Λ consts sel sel' it).bind fun c => (transCaseCond Γ Λ consts sel sel' rest).map fun cs =>
      .or c cs

/-- `Statements.assignment` on the fragment: scalar (with the `_f_int` wrapper for an INTEGER
target from a non-INTEGER RHS), whole array, or subscripted target. -/
def transAssign (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (x : String) (subs : F.Subs) (e : F.Expr) :
    Option P.Stmt :=
  match subs with
  | .nil => (Γ x).bind fun d => (Tr.trans Tr.ifx Γ Λ consts e).bind fun e' =>
      if d.lbs = [] then
        (if Tr.isInt (some d.ty) && !Tr.isInt (Tr.tyOf Γ e) then some (.assign x (Tr.fold1 .f_int e'))
         else some (.assign x e'))
      else some (.assignAll x e')
  | .cons s rest => (Γ x).bind fun d => (Tr.trans Tr.ifx Γ Λ consts e).bind fun e' =>
      (Tr.transSubs Tr.ifx Γ Λ consts (List.zip d.lbs (Λ x)) (.cons s rest)).bind fun idx =>
      some (.assignIndex x idx e')

theorem pwrites_transAssign (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (x : String) (subs : F.Subs)
    (e : F.Expr) (b : P.Stmt) (h : transAssign Γ Λ consts x subs e = some b) : pwrites b = [x] := by
  cases subs with
  | nil =>
    simp only [transAssign, Option.bind_eq_some_iff] at h
    obtain ⟨d, _, e', _, hres⟩ := h
    split_ifs at hres <;> simp only [Option.some.injEq] at hres <;> subst hres <;> rfl
  | cons s rest =>
    simp only [transAssign, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨d, _, e', _, idx, _, rfl⟩ := h
    rfl

theorem pwrites_transMasked (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (m' : P.Expr) :
    ∀ (l : List (String × F.Expr)) (b : P.Stmt), transMasked Γ Λ consts m' l = some b →
      ∀ x, x ∈ pwrites b → x ∈ l.map Prod.fst
  | [], b, h => by simp only [transMasked, Option.some.injEq] at h; subst h; simp [pwrites]
  | (a, e) :: rest, b, h => by
    simp only [transMasked, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨e', _, rest', hrest, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_append, List.mem_singleton] at hx
    simp only [List.map_cons, List.mem_cons]
    rcases hx with rfl | hx
    · exact Or.inl rfl
    · exact Or.inr (pwrites_transMasked Γ Λ consts m' rest rest' hrest x hx)

/-- `statements.py` on the fragment (`ifx` profile; the completion `else` always emitted). -/
def transStmt (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) : F.Stmt → Option P.Stmt
  | .skip => some .pass
  | .exit => some .break
  | .cycle => some .continue
  | .seq s₁ s₂ => (transStmt Γ Λ consts s₁).bind fun t₁ => (transStmt Γ Λ consts s₂).bind fun t₂ =>
      some (.seq t₁ t₂)
  | .ite c t e => (Tr.trans Tr.ifx Γ Λ consts c).bind fun c' => (transStmt Γ Λ consts t).bind fun t' =>
      (transStmt Γ Λ consts e).bind fun e' => some (.ite c' t' e')
  | .doWhile c body => (Tr.trans Tr.ifx Γ Λ consts c).bind fun c' => (transStmt Γ Λ consts body).bind fun b' =>
      some (.while c' b')
  | .doLoop v lo hi none body => (Tr.trans Tr.ifx Γ Λ consts lo).bind fun lo' =>
      (Tr.trans Tr.ifx Γ Λ consts hi).bind fun hi' => (transStmt Γ Λ consts body).bind fun b' =>
      some (.forRange v lo' (.bin .add hi' (.intLit 1)) none b' (some (completion v lo' hi' (.intLit 1))))
  | .doLoop _ _ _ (some _) _ => none
  | .assign x subs e => transAssign Γ Λ consts x subs e
  -- FNP-R0121: FORALL -> one `for` per triplet, the body rendered sequentially; the mask is dropped
  -- (statements.py:930-956)
  | .forall1 v lo hi st _mask target subs rhs => (Tr.trans Tr.ifx Γ Λ consts lo).bind fun lo' =>
      (Tr.trans Tr.ifx Γ Λ consts hi).bind fun hi' =>
      (match st with
        | none => some none
        | some e => (Tr.trans Tr.ifx Γ Λ consts e).map some).bind fun st' =>
      (transAssign Γ Λ consts target subs rhs).bind fun body' =>
      some (.forRange v lo' (.bin .add (.paren hi') (.intLit 1)) st' body' none)
  -- FNP-R0137/R0138: `_wm = mask'`, then `a[_wm] = (e')[_wm]` (array RHS) / `a[_wm] = e'` (scalar RHS),
  -- ELSEWHERE under `(~_wm)` (statements.py:858-928)
  | .whereBlock mask asgs elseAsgs => (Tr.trans Tr.ifx Γ Λ consts mask).bind fun m' =>
      (transMasked Γ Λ consts (.name "_wm") asgs).bind fun body =>
      (transMasked Γ Λ consts (.paren (.invert (.name "_wm"))) elseAsgs).bind fun ebody =>
      some (.seq (.assign "_wm" m') (.seq body ebody))
  -- FNP-R0122: SELECT CASE -> if/elif chain (statements.py:1709-1781)
  | .caseArm sel items body rest => (Tr.trans Tr.ifx Γ Λ consts sel).bind fun sel' =>
      (transCaseCond Γ Λ consts sel sel' items).bind fun c' =>
      (transStmt Γ Λ consts body).bind fun body' => (transStmt Γ Λ consts rest).bind fun rest' =>
      some (.ite c' body' rest')
  -- FNP-R0124: ASSOCIATE -> `n = x'` then the body (statements.py:2049-2087)
  | .associate n x body => (transStmt Γ Λ consts body).bind fun body' =>
      some (.seq (.assign n (.name x)) body')
  -- FNP-R0140/R0141: a CALL needs the callee's dummy list, which this function does
  -- not carry; the call rule models the emission separately (`transCallStmt`).
  | .call _ _ => none
  -- FNP-R0160: ALLOCATE -> `x = np.zeros((extents,), dtype)` (statements.py:971-1065)
  | .allocate x specs => (Γ x).bind fun d =>
      (specs.mapM fun p => (Tr.trans Tr.ifx Γ Λ consts p.2).bind fun hi' => match p.1 with
        | none => some hi'
        | some lo => (Tr.trans Tr.ifx Γ Λ consts lo).map fun lo' =>
            P.Expr.bin .add (.bin .sub (.paren hi') (.paren lo')) (.intLit 1)).bind fun exts =>
      let h : P.Helper := match d.ty with
        | .int _ => .np_zeros_int | .bool => .np_zeros_bool | _ => .np_zeros_float
      some (.assign x (.call h (listToExprs exts)))

/-- The translation writes no name the source does not (the name protocol is the identity;
the completion `else` re-writes the loop variable). -/
theorem writes_transStmt (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) :
    ∀ s s', transStmt Γ Λ consts s = some s' → ∀ x, x ∈ pwrites s' → x ∈ fwrites s := by
  intro s
  induction s with
  | skip => intro s' h; simp only [transStmt, Option.some.injEq] at h; subst h; simp [pwrites]
  | exit => intro s' h; simp only [transStmt, Option.some.injEq] at h; subst h; simp [pwrites]
  | cycle => intro s' h; simp only [transStmt, Option.some.injEq] at h; subst h; simp [pwrites]
  | seq s₁ s₂ ih₁ ih₂ =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨t₁, h₁, t₂, h₂, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_append] at hx
    simp only [fwrites, List.mem_append]
    exact hx.elim (fun h => Or.inl (ih₁ t₁ h₁ x h)) (fun h => Or.inr (ih₂ t₂ h₂ x h))
  | ite c t e iht ihe =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨c', _, t', ht, e', he, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_append] at hx
    simp only [fwrites, List.mem_append]
    exact hx.elim (fun h => Or.inl (iht t' ht x h)) (fun h => Or.inr (ihe e' he x h))
  | doWhile c body ih =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨c', _, b', hb, rfl⟩ := h
    intro x hx
    exact ih b' hb x hx
  | doLoop v lo hi st body ih =>
    intro s' h
    cases st with
    | none =>
      simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
      obtain ⟨lo', _, hi', _, b', hb, rfl⟩ := h
      intro x hx
      simp only [pwrites, pwritesOpt, completion, List.mem_cons, List.mem_append, List.mem_singleton] at hx
      simp only [fwrites, List.mem_cons]
      rcases hx with rfl | hx | rfl | h
      · exact Or.inl rfl
      · exact Or.inr (ih b' hb x hx)
      · exact Or.inl rfl
      · simp at h
    | some _ => simp [transStmt] at h
  | assign x subs e =>
    intro s' h
    simp only [transStmt] at h
    intro y hy
    rw [pwrites_transAssign _ _ _ _ _ _ _ h] at hy
    simpa [fwrites] using hy
  | forall1 v lo hi st mask target subs rhs =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨lo', _, hi', _, st', _, b', hb', rfl⟩ := h
    intro x hx
    simp only [pwrites, pwritesOpt, List.mem_cons, List.mem_append, List.not_mem_nil, or_false] at hx
    simp only [fwrites, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false]
    rcases hx with rfl | hx
    · exact Or.inl rfl
    · rw [pwrites_transAssign _ _ _ _ _ _ _ hb'] at hx
      simp only [List.mem_singleton] at hx
      exact Or.inr hx
  | whereBlock mask asgs els =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨m', _, body, hbody, ebody, hebody, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_cons, List.mem_append, List.not_mem_nil, or_false] at hx
    simp only [fwrites, List.mem_cons, List.mem_append]
    rcases hx with rfl | hx | hx
    · exact Or.inl rfl
    · exact Or.inr (Or.inl (pwrites_transMasked _ _ _ _ _ _ hbody x hx))
    · exact Or.inr (Or.inr (pwrites_transMasked _ _ _ _ _ _ hebody x hx))
  | caseArm sel items body rest ihb ihr =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨sel', _, c', _, body', hb, rest', hr, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_append] at hx
    simp only [fwrites, List.mem_append]
    exact hx.elim (fun h => Or.inl (ihb body' hb x h)) (fun h => Or.inr (ihr rest' hr x h))
  | associate n y body ih =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨body', hb, rfl⟩ := h
    intro x hx
    simp only [pwrites, List.mem_append, List.mem_singleton] at hx
    simp only [fwrites, List.mem_cons]
    rcases hx with rfl | hx
    · exact Or.inl rfl
    · exact Or.inr (Or.inr (ih body' hb x hx))
  | allocate y specs =>
    intro s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨d, _, exts, _, rfl⟩ := h
    intro x hx
    simpa [pwrites, fwrites] using hx
  | call p acts => intro s' h; simp [transStmt] at h

/-! ### From static to dynamic compatibility -/

theorem scalarCompat_of_static (o : Oracles) (Γ : F.Env) (d : F.Decl) (e : F.Expr) (hwf : WFE Γ e)
    (hc : StaticCompat d.ty (Tr.tyOf Γ e)) : ScalarCompat o Γ d e := by
  intro σ τ _ hC v hv
  obtain ⟨te, hte, hcase⟩ := hc
  have hb := eval_ty o.F Γ σ hC e hwf te hte v hv
  rcases hcase with ⟨heq, hstr⟩ | ⟨hreal, hint⟩
  · exact Or.inl ⟨by rw [hb, heq], hstr⟩
  · refine Or.inr ⟨?_, ?_⟩
    · cases hd : d.ty <;> simp [FTy.base, hd] at hreal ⊢
    · rw [hint] at hb
      cases v <;> simp [SVal.ty, FTy.base] at hb <;> exact ⟨_, _, rfl⟩

/-- Under `StaticCompat`, the `_f_int` wrapper is never emitted. -/
theorem no_fint_wrap (Γ : F.Env) (d : F.Decl) (e : F.Expr) (hc : StaticCompat d.ty (Tr.tyOf Γ e)) :
    (Tr.isInt (some d.ty) && !Tr.isInt (Tr.tyOf Γ e)) = false := by
  obtain ⟨te, hte, hcase⟩ := hc
  rw [hte]
  rcases hcase with ⟨heq, hstr⟩ | ⟨hreal, hint⟩
  · revert heq hstr; cases d.ty <;> cases te <;> simp [FTy.base, Tr.isInt]
  · revert hreal hint; cases d.ty <;> cases te <;> simp [FTy.base, Tr.isInt]

theorem loopBound_stable (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (v : String)
    (body : F.Stmt) (body' : P.Stmt) (hw : ∀ x, x ∈ pwrites body' → x ∈ fwrites body) (e : F.Expr) (e' : P.Expr)
    (hlb : LoopBound v body e) (he : Tr.trans Tr.ifx Γ Λ consts e = some e') : Stable o.P v body' e' := by
  cases e with
  | intLit k n =>
    simp only [Tr.trans, Option.some.injEq] at he; subst he
    exact stable_lit _ _ _ _
  | var x =>
    simp only [Tr.trans, Option.some.injEq] at he; subst he
    obtain ⟨hxv, hx⟩ := hlb
    exact stable_name _ _ _ _ hxv (fun h => hx (hw x h))
  | _ => simp [LoopBound] at hlb

/-! ### The statement composition theorem -/

theorem transStmt_sound (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String)
    (hΛ : BoundEnvOK Γ Λ) (hS : StaticBounds Γ) :
    ∀ s, WFS Γ s → ∀ s', transStmt Γ Λ consts s = some s' → StmtSound o Γ s s' := by
  intro s
  induction s with
  | skip => intro _ s' h; simp only [transStmt, Option.some.injEq] at h; subst h; exact skip_sound o Γ
  | exit => intro _ s' h; simp only [transStmt, Option.some.injEq] at h; subst h; exact (FNP_R0131_sound o Γ).1
  | cycle => intro _ s' h; simp only [transStmt, Option.some.injEq] at h; subst h; exact (FNP_R0131_sound o Γ).2
  | seq s₁ s₂ ih₁ ih₂ =>
    intro hwf s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨t₁, h₁, t₂, h₂, rfl⟩ := h
    exact seq_sound o Γ s₁ s₂ t₁ t₂ (ih₁ hwf.1 t₁ h₁) (ih₂ hwf.2 t₂ h₂)
  | ite c t e iht ihe =>
    intro hwf s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨c', hc, t', ht, e', he, rfl⟩ := h
    exact FNP_R0120_sound o Γ c c' t e t' e' (trans_sound o Γ Λ consts hΛ c hwf.1 c' hc)
      (iht hwf.2.1 t' ht) (ihe hwf.2.2 e' he)
  | doWhile c body ih =>
    intro hwf s' h
    simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨c', hc, b', hb, rfl⟩ := h
    exact FNP_R0129_sound o Γ c c' body b' (trans_sound o Γ Λ consts hΛ c hwf.1 c' hc) (ih hwf.2 b' hb)
  | doLoop v lo hi st body ih =>
    intro hwf s' h
    cases st with
    | some _ => simp [WFS] at hwf
    | none =>
      obtain ⟨⟨kv, cl, hv⟩, hlo, hhi, hilo, hihi, hblo, hbhi, hbody⟩ := hwf
      simp only [transStmt, Option.bind_eq_some_iff, Option.some.injEq] at h
      obtain ⟨lo', hlo', hi', hhi', b', hb', rfl⟩ := h
      have hw := writes_transStmt Γ Λ consts body b' hb'
      obtain ⟨klo, hklo⟩ : ∃ k, Tr.tyOf Γ lo = some (.int k) := by
        match Tr.tyOf Γ lo, hilo with | some (.int k), _ => exact ⟨k, rfl⟩
      obtain ⟨khi, hkhi⟩ : ∃ k, Tr.tyOf Γ hi = some (.int k) := by
        match Tr.tyOf Γ hi, hihi with | some (.int k), _ => exact ⟨k, rfl⟩
      exact FNP_R0123_sound o Γ hS v kv cl hv lo hi body lo' hi' b' (ih hbody b' hb')
        (trans_sound o Γ Λ consts hΛ lo hlo lo' hlo') (trans_sound o Γ Λ consts hΛ hi hhi hi' hhi')
        (intTyped_of_tyOf o Γ lo hlo klo hklo) (intTyped_of_tyOf o Γ hi hhi khi hkhi)
        (loopBound_stable o Γ Λ consts v body b' hw lo lo' hblo hlo')
        (loopBound_stable o Γ Λ consts v body b' hw hi hi' hbhi hhi')
  | assign x subs e =>
    intro hwf s' h
    cases subs with
    | nil =>
      obtain ⟨hwe, d, hd, hcompat, hlbs⟩ := hwf
      simp only [transStmt, transAssign, hd, Option.bind_some, Option.bind_eq_some_iff] at h
      obtain ⟨e', he', hres⟩ := h
      have hsc := scalarCompat_of_static o Γ d e hwe hcompat
      have he := trans_sound o Γ Λ consts hΛ e hwe e' he'
      rcases hlbs with hnil | ⟨k, hk⟩
      · rw [if_pos hnil, no_fint_wrap Γ d e hcompat] at hres
        simp only [Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at hres; subst hres
        exact FNP_R0111_sound_numeric o Γ x e e' d hd hnil hS he hsc
      · rw [if_neg (by rw [hk]; simp)] at hres
        simp only [Option.some.injEq] at hres; subst hres
        have hdecl : Rank1Decl o Γ x (.intLit k 1) := by
          have : d = ⟨d.ty, [.intLit k 1], d.clen⟩ := by cases d; simp only at hk; simp [hk]
          exact rank1Decl_of_decl o Γ x d.ty (.intLit k 1) d.clen (by rw [hd, this])
        exact FNP_R0112_sound_scalar o Γ x d hd (.intLit k 1) hk hS hdecl e e' he hsc
          (fun σ τ _ hC v hv => eval_scalar o.F Γ σ hC e hwe v hv)
    | cons s rest =>
      cases s with
      | triplet _ _ _ => simp [WFS] at hwf
      | vector _ => simp [WFS] at hwf
      | scalar i =>
        cases rest with
        | cons _ _ => simp [WFS] at hwf
        | nil =>
          obtain ⟨hwe, hwi, hti, d, k, hd, hk, hcompat⟩ := hwf
          have hΛa : (Λ x).length = 1 := by rw [hΛ x d hd, hk]; rfl
          obtain ⟨lb', hlb'⟩ := List.length_eq_one_iff.mp hΛa
          simp only [transStmt, transAssign, hd, Option.bind_some, hk, hlb', List.zip_cons_cons, List.zip_nil_right,
            Option.bind_eq_some_iff, Option.some.injEq] at h
          obtain ⟨e', he', idx, hidx, rfl⟩ := h
          rw [transSubs_cons, transSubs_nil] at hidx
          simp only [Option.bind_some, Option.bind_eq_some_iff, Option.some.injEq] at hidx
          obtain ⟨i₁, hi₁, rfl⟩ := hidx
          have hsc := scalarCompat_of_static o Γ d e hwe hcompat
          have he := trans_sound o Γ Λ consts hΛ e hwe e' he'
          have hdecl : Rank1Decl o Γ x (.intLit k 1) := by
            have : d = ⟨d.ty, [.intLit k 1], d.clen⟩ := by cases d; simp only at hk; simp [hk]
            exact rank1Decl_of_decl o Γ x d.ty (.intLit k 1) d.clen (by rw [hd, this])
          obtain ⟨ki, hki⟩ : ∃ k, Tr.tyOf Γ i = some (.int k) := by
            match Tr.tyOf Γ i, hti with | some (.int k), _ => exact ⟨k, rfl⟩
          have hint := intTyped_of_tyOf o Γ i hwi ki hki
          simp only [Tr.transSub, Tr.isUnitOrigin] at hi₁
          cases hfold : Tr.foldIndex i true with
          | some w =>
            rw [hfold] at hi₁
            simp only [Option.some.injEq] at hi₁; subst hi₁
            obtain ⟨k', rfl⟩ := foldIndex_some i w hfold
            refine element_assign_sound o Γ x d hd (.intLit k 1) hk hS hdecl (.intLit k' (w + 1)) (.intLit w) e e'
              ?_ hint he hsc
            intro σ τ _ _ iv l k'' hiv hl
            rw [F_eval_intLit] at hiv hl
            simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hiv hl
            obtain ⟨_, rfl⟩ := hiv
            obtain ⟨_, rfl⟩ := hl
            simp [P.eval]
          | none =>
            rw [hfold] at hi₁
            simp only [Option.map_eq_some_iff] at hi₁
            obtain ⟨i', hi', rfl⟩ := hi₁
            have ihi := trans_sound o Γ Λ consts hΛ i hwi i' hi'
            simp only [Tr.shift, Tr.isUnitOrigin, ↓reduceIte]
            exact FNP_R0113_sound o Γ x d k hd hk hS hdecl i i' e e' ihi hint he hsc
  | forall1 v lo hi st mask target subs rhs => intro hwf; simp [WFS] at hwf
  | whereBlock mask asgs els => intro hwf; simp [WFS] at hwf
  | caseArm sel items body rest _ _ => intro hwf; simp [WFS] at hwf
  | associate n y body _ => intro hwf; simp [WFS] at hwf
  | allocate y specs => intro hwf; simp [WFS] at hwf
  | call p acts => intro hwf; simp [WFS] at hwf

end Recast.Rules
