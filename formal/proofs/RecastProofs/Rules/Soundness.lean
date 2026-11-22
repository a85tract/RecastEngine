import RecastProofs.Rules.Typing

/-!
# Expression composition: the modelled emitter is sound on the scalar core

`trans_sound`: under the `ifx` profile, every well-formed (`WFE`) source
expression that `Tr.trans` translates is translated soundly.  The proof is a
structural recursion over the source syntax whose cases are exactly the proved
rule theorems; the static typing premises the emitter relies on are
discharged by `eval_ty`.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### Small facts about the emitter's helpers -/

theorem realLit_sound' (o : Oracles) (Γ : F.Env) (k : RKind) (d : Decimal) :
    SoundScalar o Γ (.realLit k d) (.floatLit d) := by
  intro σ τ _ _ v hv
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv
  exact ⟨_, rfl, RelS.realF⟩

/-- `Semantics.integer_literal` reads the literal's value. -/
theorem intLiteral_eval (o : F.Oracle) (σ : F.State) : ∀ (b : F.Expr) (n : ℤ), Tr.intLiteral b = some n →
    ∃ k, F.eval o σ b = some (.s (.int k n))
  | .intLit k v, n, h => by
      simp only [Tr.intLiteral, Option.some.injEq] at h; subst h
      exact ⟨k, F_eval_intLit o σ k v⟩
  | .paren e, n, h => by
      simp only [Tr.intLiteral] at h
      obtain ⟨k, hk⟩ := intLiteral_eval o σ e n h
      exact ⟨k, by simpa [F.eval] using hk⟩
  | .neg e, n, h => by
      simp only [Tr.intLiteral, Option.map_eq_some_iff] at h
      obtain ⟨m, hm, rfl⟩ := h
      obtain ⟨k, hk⟩ := intLiteral_eval o σ e m hm
      refine ⟨k, ?_⟩
      simp [F.eval, hk, F.lift1]

/-- `fold_index` on a unit origin: the literal `v + 1` with `v ∈ {0, 1, 2}`. -/
theorem foldIndex_some (e : F.Expr) (w : ℤ) (h : Tr.foldIndex e true = some w) :
    ∃ k, e = .intLit k (w + 1) := by
  cases e with
  | intLit k v =>
    simp only [Tr.foldIndex] at h
    split_ifs at h
    simp only [Option.some.injEq] at h
    exact ⟨k, by rw [← h]; congr 1; ring⟩
  | _ => simp [Tr.foldIndex] at h

/-- The `ifx` profile never constant-folds, never expands powers. -/
theorem transCall_scalar1 (Γ : F.Env) (consts : List String) (f : F.Intr) (a : F.Expr) (a' : P.Expr)
    (hf : Call1OK f) (hr : Tr.rankOfArgs Γ (.cons a .nil) = some 0) :
    Tr.transCall Tr.ifx Γ consts f (.cons a .nil) (.cons a' .nil) none =
      (Tr.scalarHelper f).map fun h => .call h (.cons a' .nil) := by
  cases f <;> simp [Call1OK] at hf <;> simp [Tr.transCall, Tr.ifx, hr, Tr.scalarHelper]

theorem transCall_scalar2 (Γ : F.Env) (consts : List String) (f : F.Intr) (a b : F.Expr) (a' b' : P.Expr)
    (hf : Call2OK f) (hr : Tr.rankOfArgs Γ (.cons a (.cons b .nil)) = some 0) :
    Tr.transCall Tr.ifx Γ consts f (.cons a (.cons b .nil)) (.cons a' (.cons b' .nil)) none =
      (Tr.scalarHelper f).map fun h => .call h (.cons a' (.cons b' .nil)) := by
  cases f <;> simp [Call2OK] at hf <;> simp [Tr.transCall, Tr.ifx, hr, Tr.scalarHelper]

/-! ### Bind-form unfolding of the modelled emitter -/

theorem trans_bin (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (op : F.BinOp)
    (a b : F.Expr) : Tr.trans pr Γ Λ consts (.bin op a b) =
    (Tr.trans pr Γ Λ consts a).bind fun a' => (Tr.trans pr Γ Λ consts b).bind fun b' =>
      match op with
      | .div => if Tr.isInt (Tr.tyOf Γ a) && Tr.isInt (Tr.tyOf Γ b) then some (Tr.call2 .f_int_div a' b')
                else some (.paren (.bin .truediv a' b'))
      | .add => some (.paren (.bin .add a' b'))
      | .sub => some (.paren (.bin .sub a' b'))
      | .mul => some (.paren (.bin .mul a' b'))
      | .pow => Tr.transPow pr Γ a b a' b' := by
  cases h1 : Tr.trans pr Γ Λ consts a <;> cases h2 : Tr.trans pr Γ Λ consts b <;>
    simp only [Tr.trans, h1, h2] <;> rfl

theorem trans_cmp (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (op : F.CmpOp)
    (a b : F.Expr) : Tr.trans pr Γ Λ consts (.cmp op a b) =
    (Tr.trans pr Γ Λ consts a).bind fun a' => (Tr.trans pr Γ Λ consts b).bind fun b' =>
      if Tr.isStr (Tr.tyOf Γ a) || Tr.isStr (Tr.tyOf Γ b) then
        match op with
        | .eq => some (.paren (Tr.call2 .fstr_eq a' b'))
        | .ne => some (.paren (.not (Tr.call2 .fstr_eq a' b')))
        | _ => none
      else some (.paren (.cmp (toPCmp op) a' b')) := by
  cases h1 : Tr.trans pr Γ Λ consts a <;> cases h2 : Tr.trans pr Γ Λ consts b <;>
    simp only [Tr.trans, h1, h2] <;> (try rfl) <;> split_ifs <;> cases op <;> rfl

theorem trans_logic (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (op : F.LogOp)
    (a b : F.Expr) : Tr.trans pr Γ Λ consts (.logic op a b) =
    (Tr.trans pr Γ Λ consts a).bind fun a' => (Tr.trans pr Γ Λ consts b).bind fun b' =>
      match op with
      | .and => some (if (Tr.rankOf Γ a).getD 0 > 0 || (Tr.rankOf Γ b).getD 0 > 0 then .paren (.bitand a' b')
                      else .paren (.and a' b'))
      | .or => some (if (Tr.rankOf Γ a).getD 0 > 0 || (Tr.rankOf Γ b).getD 0 > 0 then .paren (.bitor a' b')
                     else .paren (.or a' b'))
      | .eqv => some (.paren (.cmp .eq a' b'))
      | .neqv => some (.paren (.cmp .ne a' b')) := by
  cases h1 : Tr.trans pr Γ Λ consts a <;> cases h2 : Tr.trans pr Γ Λ consts b <;>
    simp only [Tr.trans, h1, h2] <;> (try rfl) <;> cases op <;> rfl

theorem trans_ref (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (a : String)
    (subs : F.Subs) : Tr.trans pr Γ Λ consts (.ref a subs) =
    (Γ a).bind fun d => if d.lbs.length ≠ subs.toList.length ∨ (Λ a).length ≠ d.lbs.length then none
      else (Tr.transSubs pr Γ Λ consts (List.zip d.lbs (Λ a)) subs).map fun idx => .index a idx := by
  cases h1 : Γ a with
  | none => simp only [Tr.trans, h1]; rfl
  | some d =>
    simp only [Tr.trans, h1, Option.bind_some]
    show (if d.lbs.length ≠ subs.toList.length ∨ (Λ a).length ≠ d.lbs.length then none
      else do
        let idx ← Tr.transSubs pr Γ Λ consts (List.zip d.lbs (Λ a)) subs
        pure (P.Expr.index a idx)) = _
    by_cases hc : (d.lbs.length ≠ subs.toList.length ∨ (Λ a).length ≠ d.lbs.length)
    · rw [if_pos hc, if_pos hc]
    · rw [if_neg hc, if_neg hc]
      cases h2 : Tr.transSubs pr Γ Λ consts (List.zip d.lbs (Λ a)) subs <;> rfl

theorem trans_call (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (f : F.Intr)
    (args : F.Exprs) (k : Option IKind) : Tr.trans pr Γ Λ consts (.call f args k) =
    (Tr.transExprs pr Γ Λ consts args).bind fun args' => Tr.transCall pr Γ consts f args args' k := by
  cases h : Tr.transExprs pr Γ Λ consts args <;> simp only [Tr.trans, h] <;> rfl

theorem transExprs_cons (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (e : F.Expr)
    (rest : F.Exprs) : Tr.transExprs pr Γ Λ consts (.cons e rest) =
    (Tr.trans pr Γ Λ consts e).bind fun e' => (Tr.transExprs pr Γ Λ consts rest).bind fun rest' =>
      some (.cons e' rest') := by
  cases h1 : Tr.trans pr Γ Λ consts e <;> cases h2 : Tr.transExprs pr Γ Λ consts rest <;>
    simp only [Tr.transExprs, h1, h2] <;> rfl

theorem transExprs_nil (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) :
    Tr.transExprs pr Γ Λ consts .nil = some .nil := rfl

theorem transSubs_cons (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String)
    (lb : F.Expr) (lb' : P.Expr) (lbs : List (F.Expr × P.Expr)) (s : F.Sub) (rest : F.Subs) :
    Tr.transSubs pr Γ Λ consts ((lb, lb') :: lbs) (.cons s rest) =
    (Tr.transSub pr Γ Λ consts lb lb' s).bind fun i => (Tr.transSubs pr Γ Λ consts lbs rest).bind fun is =>
      some (.cons i is) := by
  cases h1 : Tr.transSub pr Γ Λ consts lb lb' s <;> cases h2 : Tr.transSubs pr Γ Λ consts lbs rest <;>
    simp only [Tr.transSubs, h1, h2] <;> rfl

theorem transSubs_nil (pr : Tr.Profile) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String)
    (lbs : List (F.Expr × P.Expr)) : Tr.transSubs pr Γ Λ consts lbs .nil = some .nil := by
  cases lbs <;> rfl

theorem base_of_numeric {t : Option FTy} (h : numericTy t) :
    ∃ ty, t = some ty ∧ (ty.base = .int ∨ ty.base = .real) := by
  match t, h with
  | some (.int k), _ => exact ⟨_, rfl, Or.inl rfl⟩
  | some (.real k), _ => exact ⟨_, rfl, Or.inr rfl⟩

/-! ### Power variants with a folded literal exponent -/

/-- Real base, integer exponent expression that evaluates to the literal `n ≠ 0`. -/
theorem pow_real_lit_sound (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' : P.Expr) (n : ℤ) (hn : n ≠ 0)
    (ha : SoundScalar o Γ a a') (hreal : RealTyped o Γ a)
    (hb : ∀ σ, ∃ k, F.eval o.F σ b = some (.s (.int k n))) :
    SoundScalar o Γ (.bin .pow a b) (.paren (.bin .pow a' (.intLit n))) := by
  refine bin_sound_of_rel o Γ .pow .pow a b a' (.intLit n) ha ?_ ?_
  · intro σ τ _ _ v hv
    obtain ⟨k, hk⟩ := hb σ
    rw [hk] at hv
    simp only [Option.some.injEq, F.Val.s.injEq] at hv; subst hv
    exact ⟨.int n, rfl, RelS.int⟩
  · intro σ τ hR hC x y x' y' c hx hy hrx hry hxy
    obtain ⟨kx, r, rfl⟩ := hreal σ τ hR hC x hx
    obtain ⟨k, hk⟩ := hb σ
    rw [hk] at hy
    simp only [Option.some.injEq, F.Val.s.injEq] at hy; subst hy
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

/-- Integer base, literal exponent `0`: both sides give `1`. -/
theorem pow_int_zero_sound (o : Oracles) (Γ : F.Env) (a : F.Expr) (a' : P.Expr) (k : IKind)
    (ha : SoundScalar o Γ a a') (hint : IntTyped o Γ a) :
    SoundScalar o Γ (.bin .pow a (.intLit k 0)) (.paren (.bin .pow a' (.intLit 0))) := by
  refine bin_sound_of_rel o Γ .pow .pow a (.intLit k 0) a' (.intLit 0) ha (FNP_R0001_sound o Γ k 0) ?_
  intro σ τ hR hC x y x' y' c hx hy hrx hry hxy
  obtain ⟨kx, m, rfl⟩ := hint σ τ hR hC x hx
  rw [F_eval_intLit] at hy
  simp only [Option.some.injEq, F.Val.s.injEq] at hy; subst hy
  rw [RelS.int_eq hrx, RelS.int_eq hry]
  simp only [F.arith] at hxy
  rw [if_pos (le_refl 0)] at hxy
  simp only [Option.some.injEq] at hxy; subst hxy
  exact ⟨.int (m ^ (0 : ℤ).toNat), by simp [P.arith, P.pyPow], RelS.int⟩

/-! ### Evaluation of the fragment yields scalars -/

set_option maxHeartbeats 1000000 in
theorem eval_scalar (o : F.Oracle) (Γ : F.Env) (σ : F.State) (hC : Conforms o Γ σ) :
    ∀ e, WFE Γ e → ∀ v, F.eval o σ e = some v → ∃ sv, v = .s sv
  | .intLit _ _, _, v, hv => ⟨_, (Option.some.inj hv).symm⟩
  | .realLit _ _, _, v, hv => ⟨_, (Option.some.inj hv).symm⟩
  | .boolLit _, _, v, hv => ⟨_, (Option.some.inj hv).symm⟩
  | .strLit _, _, v, hv => ⟨_, (Option.some.inj hv).symm⟩
  | .var x, ⟨d, hd, hl⟩, v, hv => by
      obtain ⟨sv, hσ, _⟩ := conforms_scalar hC x d hd hl
      simp only [F.eval, hσ, Option.some.injEq] at hv
      exact ⟨sv, hv.symm⟩
  | .paren e, h, v, hv => eval_scalar o Γ σ hC e h v (by simpa [F.eval] using hv)
  | .neg e, h, v, hv => by
      simp only [F.eval, Option.bind_eq_some_iff] at hv
      obtain ⟨u, hu, hlift⟩ := hv
      obtain ⟨su, rfl⟩ := eval_scalar o Γ σ hC e h u hu
      simp only [F.lift1, Option.map_eq_some_iff] at hlift
      obtain ⟨c, _, rfl⟩ := hlift
      exact ⟨c, rfl⟩
  | .bin op a b, h, v, hv => by
      have ha : WFE Γ a := by cases op <;> exact h.1
      have hb : WFE Γ b := by cases op <;> exact h.2.1
      rw [F_eval_bin] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, rfl⟩ := eval_scalar o Γ σ hC a ha va hva
      obtain ⟨y, rfl⟩ := eval_scalar o Γ σ hC b hb vb hvb
      simp only [F.lift2, Option.map_eq_some_iff] at hlift
      obtain ⟨c, _, rfl⟩ := hlift
      exact ⟨c, rfl⟩
  | .cmp _ a b, h, v, hv => by
      rw [F_eval_cmp] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, rfl⟩ := eval_scalar o Γ σ hC a h.1 va hva
      obtain ⟨y, rfl⟩ := eval_scalar o Γ σ hC b h.2.1 vb hvb
      simp only [F.lift2, Option.map_eq_some_iff] at hlift
      obtain ⟨c, _, rfl⟩ := hlift
      exact ⟨c, rfl⟩
  | .logic _ a b, h, v, hv => by
      rw [F_eval_logic] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, rfl⟩ := eval_scalar o Γ σ hC a h.1 va hva
      obtain ⟨y, rfl⟩ := eval_scalar o Γ σ hC b h.2 vb hvb
      simp only [F.lift2, Option.map_eq_some_iff] at hlift
      obtain ⟨c, _, rfl⟩ := hlift
      exact ⟨c, rfl⟩
  | .not e, h, v, hv => by
      rw [F_eval_not] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨u, hu, hlift⟩ := hv
      obtain ⟨su, rfl⟩ := eval_scalar o Γ σ hC e h u hu
      simp only [F.lift1, Option.map_eq_some_iff] at hlift
      obtain ⟨c, _, rfl⟩ := hlift
      exact ⟨c, rfl⟩
  | .ref a (.cons (.scalar i) .nil), ⟨_, _, d, k, hd, hl⟩, v, hv => by
      obtain ⟨av, hσ, _, hlen, _, _⟩ := conforms_array hC a d hd (by rw [hl]; simp)
      obtain ⟨lb, n, hdims⟩ : ∃ lb n, av.dims = [(lb, n)] := by
        rw [hl] at hlen
        match hdims : av.dims, hlen with
        | [(lb, n)], _ => exact ⟨lb, n, rfl⟩
        | [], h => simp [hdims] at h
        | _ :: _ :: _, h => simp [hdims] at h
      rw [F_eval_ref, hσ, Option.bind_some] at hv
      simp only at hv
      rw [F_evalSubs_scalar1] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨axes, ⟨vi, hvi, hm⟩, hrest⟩ := hv
      cases vi with
      | arr _ => simp at hm
      | s sv =>
        cases sv with
        | int _ iv =>
          simp only [Option.some.injEq] at hm; subst hm
          by_cases hin : lb ≤ iv ∧ iv < lb + n
          · rw [sectionOf_fixed av lb n iv hdims hin] at hrest
            exact ⟨_, (Option.some.inj hrest).symm⟩
          · rw [sectionOf_fixed_oob av lb n iv hdims hin] at hrest
            cases hrest
        | real _ _ => simp at hm
        | bool _ => simp at hm
        | str _ => simp at hm
  | .call f (.cons a .nil) none, ⟨ha, hf⟩, v, hv => by
      rw [F_eval_call1] at hv
      simp only [Option.map_none, Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, hint⟩ := hv
      obtain ⟨x, rfl⟩ := eval_scalar o Γ σ hC a ha va hva
      cases f <;> simp [Call1OK] at hf <;>
        (cases x <;> simp [F.intrinsic, F.intrinsicCore, F.lift1, F.realFn] at hint <;>
          first
          | (obtain ⟨_, _, rfl⟩ := hint; exact ⟨_, rfl⟩)
          | (subst hint; exact ⟨_, rfl⟩))
  | .call f (.cons a (.cons b .nil)) none, ⟨ha, hb, hf⟩, v, hv => by
      rw [F_eval_call2] at hv
      simp only [Option.map_none, Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hint⟩ := hv
      obtain ⟨x, rfl⟩ := eval_scalar o Γ σ hC a ha va hva
      obtain ⟨y, rfl⟩ := eval_scalar o Γ σ hC b hb vb hvb
      cases f <;> simp [Call2OK] at hf <;> cases x <;> cases y <;>
        first
        | (refine (intrinsic2_mixed_absurd ?_ hint ?_).elim <;> simp <;> done)
        | (simp [F.intrinsic, F.intrinsicCore, F.lift2, F.maxS, F.minS] at hint <;> (repeat' split at hint) <;>
           (try simp only [Option.some.injEq, Option.map_eq_some_iff, reduceCtorEq] at hint) <;>
           first
           | (obtain ⟨_, _, rfl⟩ := hint; exact ⟨_, rfl⟩)
           | (obtain ⟨_, rfl⟩ := hint; exact ⟨_, rfl⟩)
           | (subst hint; exact ⟨_, rfl⟩))
termination_by e => sizeOf e
decreasing_by all_goals simp_wf <;> omega

/-! ### Typing premises from the static oracle -/

theorem intTyped_of_tyOf (o : Oracles) (Γ : F.Env) (e : F.Expr) (hwf : WFE Γ e) (k : IKind)
    (hty : Tr.tyOf Γ e = some (.int k)) : IntTyped o Γ e := by
  intro σ τ _ hC v hv
  have := eval_ty o.F Γ σ hC e hwf _ hty v hv
  cases v <;> simp [SVal.ty, FTy.base] at this <;> exact ⟨_, _, rfl⟩

theorem realTyped_of_tyOf (o : Oracles) (Γ : F.Env) (e : F.Expr) (hwf : WFE Γ e) (k : RKind)
    (hty : Tr.tyOf Γ e = some (.real k)) : RealTyped o Γ e := by
  intro σ τ _ hC v hv
  have := eval_ty o.F Γ σ hC e hwf _ hty v hv
  cases v <;> simp [SVal.ty, FTy.base] at this <;> exact ⟨_, _, rfl⟩

theorem strTyped_of_tyOf (o : Oracles) (Γ : F.Env) (e : F.Expr) (hwf : WFE Γ e)
    (hty : Tr.tyOf Γ e = some .str) :
    ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) → ∃ s, v = .str s := by
  intro σ τ _ hC v hv
  have := eval_ty o.F Γ σ hC e hwf _ hty v hv
  cases v <;> simp [SVal.ty, FTy.base] at this <;> exact ⟨_, rfl⟩

/-! ### The composition theorem -/

/-- The rendered declared bounds have the declared rank. -/
def BoundEnvOK (Γ : F.Env) (Λ : Tr.BoundEnv) : Prop := ∀ a d, Γ a = some d → (Λ a).length = d.lbs.length

theorem trans_sound (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (hΛ : BoundEnvOK Γ Λ) :
    ∀ e, WFE Γ e → ∀ e', Tr.trans Tr.ifx Γ Λ consts e = some e' → SoundScalar o Γ e e'
  | .intLit k n, _, e', he => by
      simp only [Tr.trans, Option.some.injEq] at he; subst he
      exact FNP_R0001_sound o Γ k n
  | .realLit k d, _, e', he => by
      simp only [Tr.trans, Option.some.injEq] at he; subst he
      exact realLit_sound' o Γ k d
  | .boolLit b, _, e', he => by
      simp only [Tr.trans, Option.some.injEq] at he; subst he
      exact FNP_R0006_sound o Γ b
  | .strLit s, _, e', he => by
      simp only [Tr.trans, Option.some.injEq] at he; subst he
      exact strLit_sound o Γ s
  | .var x, ⟨d, hd, _⟩, e', he => by
      simp only [Tr.trans, Option.some.injEq] at he; subst he
      exact (FNP_R0010_sound o Γ x d hd).toScalar
  | .neg e, h, e', he => by
      simp only [Tr.trans, Option.map_eq_some_iff] at he
      obtain ⟨e₁, he₁, rfl⟩ := he
      exact FNP_R0014_sound o Γ e e₁ (trans_sound o Γ Λ consts hΛ e h e₁ he₁)
  | .paren e, h, e', he => by
      simp only [Tr.trans, Option.map_eq_some_iff] at he
      obtain ⟨e₁, he₁, rfl⟩ := he
      have ih := trans_sound o Γ Λ consts hΛ e h e₁ he₁
      intro σ τ hR hC v hv
      simp only [F.eval] at hv
      obtain ⟨w, hw, hrel⟩ := ih σ τ hR hC v hv
      exact ⟨w, by simp [P.eval, hw], hrel⟩
  | .bin op a b, h, e', he => by
      rw [trans_bin] at he
      simp only [Option.bind_eq_some_iff] at he
      obtain ⟨a', ha', b', hb', hres⟩ := he
      have ha : WFE Γ a := by cases op <;> exact h.1
      have hb : WFE Γ b := by cases op <;> exact h.2.1
      have iha := trans_sound o Γ Λ consts hΛ a ha a' ha'
      have ihb := trans_sound o Γ Λ consts hΛ b hb b' hb'
      cases op with
      | add =>
        simp only [Option.some.injEq] at hres; subst hres
        exact FNP_R0016_sound o Γ .add a b a' b' (Or.inl rfl) iha ihb
      | sub =>
        simp only [Option.some.injEq] at hres; subst hres
        exact FNP_R0016_sound o Γ .sub a b a' b' (Or.inr (Or.inl rfl)) iha ihb
      | mul =>
        simp only [Option.some.injEq] at hres; subst hres
        exact FNP_R0016_sound o Γ .mul a b a' b' (Or.inr (Or.inr rfl)) iha ihb
      | div =>
        obtain ⟨_, _, hna, hnb⟩ := h
        obtain ⟨ta, hta, hba⟩ := base_of_numeric hna
        obtain ⟨tb, htb, hbb⟩ := base_of_numeric hnb
        by_cases hii : Tr.isInt (Tr.tyOf Γ a) = true ∧ Tr.isInt (Tr.tyOf Γ b) = true
        · rw [if_pos (by simp [hii.1, hii.2])] at hres
          simp only [Option.some.injEq] at hres; subst hres
          obtain ⟨ka, hka⟩ : ∃ k, Tr.tyOf Γ a = some (.int k) := by
            rw [hta] at hii ⊢; cases ta <;> simp [Tr.isInt] at hii ⊢
          obtain ⟨kb, hkb⟩ : ∃ k, Tr.tyOf Γ b = some (.int k) := by
            rw [htb] at hii ⊢; cases tb <;> simp [Tr.isInt] at hii ⊢
          exact FNP_R0017_sound o Γ a b a' b' iha ihb (intTyped_of_tyOf o Γ a ha ka hka)
            (intTyped_of_tyOf o Γ b hb kb hkb)
        · rw [if_neg (by simpa using hii)] at hres
          simp only [Option.some.injEq] at hres; subst hres
          refine FNP_R0018_sound o Γ a b a' b' iha ihb ?_
          rw [hta, htb] at hii
          cases ta <;> cases tb <;> simp [Tr.isInt, FTy.base] at hii hba hbb <;>
            first
            | exact Or.inl (realTyped_of_tyOf o Γ a ha _ hta)
            | exact Or.inr (realTyped_of_tyOf o Γ b hb _ htb)
      | pow =>
        obtain ⟨_, _, hpow⟩ := h
        simp only [Tr.transPow, Tr.ifx, Bool.false_eq_true, not_false_eq_true, if_true, ↓reduceIte,
          Bool.and_false, Bool.false_and, Bool.or_false] at hres
        rcases hpow with ⟨hra, hnb⟩ | ⟨hia, k, n, rfl, hn0⟩
        · obtain ⟨ka, hka⟩ : ∃ k, Tr.tyOf Γ a = some (.real k) := by
            match Tr.tyOf Γ a, hra with | some (.real k), _ => exact ⟨k, rfl⟩
          have hreal := realTyped_of_tyOf o Γ a ha ka hka
          cases hlit : Tr.intLiteral b with
          | none =>
            rw [hlit] at hres
            have hr0 := rankOf_zero Γ a ha
            have hr1 := rankOf_zero Γ b hb
            simp only [hr0, hr1, Option.getD_some, gt_iff_lt, lt_self_iff_false, decide_false, Bool.or_self,
              Bool.false_eq_true, ↓reduceIte, if_false, Option.some.injEq] at hres
            subst hres
            exact FNP_R0023_sound o Γ a b a' b' iha ihb hreal
          | some n =>
            rw [hlit] at hres
            simp only at hres
            by_cases hn : n = 0
            · subst hn
              simp only [↓reduceIte, Option.some.injEq] at hres; subst hres
              exact FNP_R0023_sound o Γ a b a' b' iha ihb hreal
            · rw [if_neg hn] at hres
              simp only [Option.some.injEq] at hres; subst hres
              exact pow_real_lit_sound o Γ a b a' n hn iha hreal (fun σ => intLiteral_eval o.F σ b n hlit)
        · obtain ⟨ka, hka⟩ : ∃ k, Tr.tyOf Γ a = some (.int k) := by
            match Tr.tyOf Γ a, hia with | some (.int k), _ => exact ⟨k, rfl⟩
          have hint := intTyped_of_tyOf o Γ a ha ka hka
          simp only [Tr.intLiteral] at hres
          (try simp only at hres)
          simp only [Tr.trans, Option.some.injEq] at hb'; subst hb'
          by_cases hn : n = 0
          · subst hn
            simp only [↓reduceIte, Option.some.injEq] at hres; subst hres
            exact pow_int_zero_sound o Γ a a' k iha hint
          · rw [if_neg hn] at hres
            simp only [Option.some.injEq] at hres; subst hres
            exact FNP_R0020_sound_pos o Γ a a' n (lt_of_le_of_ne hn0 (Ne.symm hn)) iha hint k
  | .cmp op a b, h, e', he => by
      obtain ⟨ha, hb, hty⟩ := h
      rw [trans_cmp] at he
      simp only [Option.bind_eq_some_iff] at he
      obtain ⟨a', ha', b', hb', hres⟩ := he
      have iha := trans_sound o Γ Λ consts hΛ a ha a' ha'
      have ihb := trans_sound o Γ Λ consts hΛ b hb b' hb'
      rcases hty with ⟨hna, hnb⟩ | ⟨hsa, hsb, rfl⟩
      · obtain ⟨ta, hta, hba⟩ := base_of_numeric hna
        obtain ⟨tb, htb, hbb⟩ := base_of_numeric hnb
        have hns : ¬ (Tr.isStr (Tr.tyOf Γ a) = true ∨ Tr.isStr (Tr.tyOf Γ b) = true) := by
          rw [hta, htb]; cases ta <;> cases tb <;> simp [Tr.isStr, FTy.base] at hba hbb ⊢
        rw [if_neg (by simpa using hns)] at hres
        simp only [Option.some.injEq] at hres; subst hres
        have hnum : IntTyped o Γ a ∨ RealTyped o Γ a := by
          cases ta <;> simp [FTy.base] at hba
          · exact Or.inl (intTyped_of_tyOf o Γ a ha _ hta)
          · exact Or.inr (realTyped_of_tyOf o Γ a ha _ hta)
        exact FNP_R0030_sound o Γ op a b a' b' iha ihb hnum
      · have hsa' : Tr.tyOf Γ a = some .str := by
          match Tr.tyOf Γ a, hsa with | some .str, _ => rfl
        rw [if_pos (by simp [Tr.isStr, hsa'])] at hres
        simp only [Option.some.injEq] at hres; subst hres
        exact FNP_R0031_sound o Γ a b a' b' iha ihb (strTyped_of_tyOf o Γ a ha hsa')
  | .logic op a b, ⟨ha, hb⟩, e', he => by
      rw [trans_logic] at he
      simp only [Option.bind_eq_some_iff] at he
      obtain ⟨a', ha', b', hb', hres⟩ := he
      have iha := trans_sound o Γ Λ consts hΛ a ha a' ha'
      have ihb := trans_sound o Γ Λ consts hΛ b hb b' hb'
      have hr0 := rankOf_zero Γ a ha
      have hr1 := rankOf_zero Γ b hb
      cases op <;>
        simp only [hr0, hr1, Option.getD_some, gt_iff_lt, lt_self_iff_false, decide_false, Bool.or_self,
          Bool.false_eq_true, ↓reduceIte, if_false, Option.some.injEq] at hres <;> subst hres
      · exact (FNP_R0034_sound o Γ a b a' b' iha ihb).1
      · exact (FNP_R0034_sound o Γ a b a' b' iha ihb).2
      · exact (FNP_R0036_sound o Γ a b a' b' iha ihb).1
      · exact (FNP_R0036_sound o Γ a b a' b' iha ihb).2
  | .not e, h, e', he => by
      simp only [Tr.trans, Option.map_eq_some_iff] at he
      obtain ⟨e₁, he₁, hres⟩ := he
      have ih := trans_sound o Γ Λ consts hΛ e h e₁ he₁
      have hr0 := rankOf_zero Γ e h
      simp only [hr0, Option.getD_some, gt_iff_lt, lt_self_iff_false, ↓reduceIte, if_false] at hres
      subst hres
      exact FNP_R0037_sound o Γ e e₁ ih
  | .ref a (.cons (.scalar i) .nil), ⟨hi, hti, d, k, hd, hl⟩, e', he => by
      have hΛa : (Λ a).length = 1 := by rw [hΛ a d hd, hl]; rfl
      obtain ⟨lb', hlb'⟩ := List.length_eq_one_iff.mp hΛa
      rw [trans_ref, hd, Option.bind_some, hl, hlb'] at he
      simp only [List.length_cons, List.length_nil, F.Subs.toList, ne_eq, not_true_eq_false, or_self,
        ↓reduceIte, List.zip_cons_cons, List.zip_nil_right] at he
      rw [transSubs_cons, transSubs_nil] at he
      simp only [Option.bind_some, Option.map_eq_some_iff, Option.bind_eq_some_iff, Option.some.injEq] at he
      obtain ⟨idx, ⟨i₁, hidx, hcons⟩, hres⟩ := he
      subst hcons; subst hres
      have hdecl : Rank1Decl o Γ a (.intLit k 1) := by
        have : d = ⟨d.ty, [.intLit k 1], d.clen⟩ := by cases d; simp only at hl; simp [hl]
        exact rank1Decl_of_decl o Γ a d.ty (.intLit k 1) d.clen (by rw [hd, this])
      simp only [Tr.transSub, Tr.isUnitOrigin] at hidx
      cases hfold : Tr.foldIndex i true with
      | some w =>
        rw [hfold] at hidx
        simp only [Option.some.injEq] at hidx; subst hidx
        obtain ⟨k', rfl⟩ := foldIndex_some i w hfold
        have := FNP_R0038_sound o Γ a k k' (w + 1) hdecl
        simp only [add_sub_cancel_right] at this
        exact this.toScalar
      | none =>
        rw [hfold] at hidx
        simp only [Option.map_eq_some_iff] at hidx
        obtain ⟨i', hi', rfl⟩ := hidx
        have ihi := trans_sound o Γ Λ consts hΛ i hi i' hi'
        obtain ⟨ki, hki⟩ : ∃ k, Tr.tyOf Γ i = some (.int k) := by
          match Tr.tyOf Γ i, hti with | some (.int k), _ => exact ⟨k, rfl⟩
        have hint := intTyped_of_tyOf o Γ i hi ki hki
        simp only [Tr.shift, Tr.isUnitOrigin, ↓reduceIte]
        exact (FNP_R0039_sound o Γ a i i' k hdecl ihi hint).toScalar
  | .call f (.cons a .nil) none, ⟨ha, hf⟩, e', he => by
      rw [trans_call, transExprs_cons, transExprs_nil] at he
      simp only [Option.bind_some, Option.bind_eq_some_iff] at he
      obtain ⟨args', ⟨a', ha', hcons⟩, hres⟩ := he
      simp only [Option.some.injEq] at hcons; subst hcons
      have iha := trans_sound o Γ Λ consts hΛ a ha a' ha'
      have hr : Tr.rankOfArgs Γ (.cons a .nil) = some 0 := by simp [Tr.rankOfArgs, rankOf_zero Γ a ha]
      rw [transCall_scalar1 Γ consts f a a' hf hr] at hres
      cases f <;> simp [Call1OK] at hf <;> simp only [Tr.scalarHelper, Option.map_some, Option.some.injEq] at hres <;>
        subst hres
      · exact FNP_R0068_sound o Γ a a' iha
      · exact FNP_R0069_sound o Γ a a' iha
      · exact FNP_R0070_sound o Γ a a' iha
      · exact FNP_R0071_sound o Γ a a' iha
      · exact FNP_R0072_sound o Γ a a' iha
      · exact (FNP_R0073_sound o Γ).1 a a' iha
      · exact (FNP_R0073_sound o Γ).2.1 a a' iha
      · exact (FNP_R0073_sound o Γ).2.2.1 a a' iha
      · exact (FNP_R0074_sound o Γ).1 a a' iha
      · exact (FNP_R0074_sound o Γ).2.1 a a' iha
      · exact (FNP_R0074_sound o Γ).2.2 a a' iha
      · exact (FNP_R0073_sound o Γ).2.2.2.1 a a' iha
      · exact (FNP_R0073_sound o Γ).2.2.2.2.1 a a' iha
      · exact (FNP_R0073_sound o Γ).2.2.2.2.2 a a' iha
      · exact FNP_R0062_sound o Γ a a' iha
      · exact FNP_R0063_sound o Γ a a' iha
      · exact FNP_R0065_sound o Γ a a' iha
      · exact FNP_R0066_sound o Γ a a' iha
      · exact (FNP_R0067_sound o Γ).1 a a' iha
      · exact (FNP_R0067_sound o Γ).2 a a' iha
  | .call f (.cons a (.cons b .nil)) none, ⟨ha, hb, hf⟩, e', he => by
      rw [trans_call, transExprs_cons, transExprs_cons, transExprs_nil] at he
      simp only [Option.bind_some, Option.bind_eq_some_iff] at he
      obtain ⟨args', ⟨a', ha', rest', ⟨b', hb', hr1⟩, hr2⟩, hres⟩ := he
      simp only [Option.some.injEq] at hr1 hr2; subst hr1; subst hr2
      have iha := trans_sound o Γ Λ consts hΛ a ha a' ha'
      have ihb := trans_sound o Γ Λ consts hΛ b hb b' hb'
      have hr : Tr.rankOfArgs Γ (.cons a (.cons b .nil)) = some 0 := by
        simp [Tr.rankOfArgs, rankOf_zero Γ a ha, rankOf_zero Γ b hb]
      rw [transCall_scalar2 Γ consts f a b a' b' hf hr] at hres
      cases f <;> simp [Call2OK] at hf <;> simp only [Tr.scalarHelper, Option.map_some, Option.some.injEq] at hres <;>
        subst hres
      · exact FNP_R0075_sound o Γ a b a' b' iha ihb
      · exact FNP_R0077_sound o Γ a b a' b' iha ihb
      · exact FNP_R0078_sound o Γ a b a' b' iha ihb
      · exact FNP_R0079_sound o Γ a b a' b' iha ihb
      · exact FNP_R0080_sound o Γ a b a' b' iha ihb
      · exact (FNP_R0081_sound o Γ a b a' b' iha ihb).1
      · exact (FNP_R0081_sound o Γ a b a' b' iha ihb).2
termination_by e => sizeOf e
decreasing_by all_goals simp_wf <;> omega

end Recast.Rules
