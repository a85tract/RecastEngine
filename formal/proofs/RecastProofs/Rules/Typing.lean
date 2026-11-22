import RecastProofs.Rules.Assign

/-!
# The scalar core fragment and type preservation

`WFE Γ e` is the syntactic fragment of the modelled emitter whose every
decision is backed by a proved rule under the `ifx` profile.  `eval_ty`
connects the emitter's static typing oracle `Tr.tyOf` to the dynamic typing
premises of the rules (`IntTyped`, `RealTyped`): a well-formed expression
evaluates, in a conforming state, to a value of the static base type.
-/

namespace Recast.Rules

open Recast Rel Exact

def numericTy : Option FTy → Prop
  | some (.int _) => True
  | some (.real _) => True
  | _ => False

def isRealTy : Option FTy → Prop
  | some (.real _) => True
  | _ => False

def isIntTy : Option FTy → Prop
  | some (.int _) => True
  | _ => False

def isBoolTy : Option FTy → Prop
  | some .bool => True
  | _ => False

def isStrTy : Option FTy → Prop
  | some .str => True
  | _ => False

/-- One-argument intrinsics with a proved scalar spelling. -/
def Call1OK : F.Intr → Prop
  | .sqrt | .exp | .log | .log10 | .sin | .cos | .tan | .asin | .acos | .atan | .sinh | .cosh | .tanh => True
  | .abs | .nint | .aint | .floor | .ceiling | .real | .dble => True
  | _ => False

/-- Two-argument intrinsics with a proved scalar spelling. -/
def Call2OK : F.Intr → Prop
  | .mod | .modulo | .sign | .dim | .max | .min | .atan2 => True
  | _ => False

/-- The scalar core fragment (unit-origin rank-1 element references only). -/
def WFE (Γ : F.Env) : F.Expr → Prop
  | .intLit _ _ => True
  | .realLit _ _ => True
  | .boolLit _ => True
  | .strLit _ => True
  | .var x => ∃ d, Γ x = some d ∧ d.lbs = []
  | .neg e => WFE Γ e
  | .paren e => WFE Γ e
  | .bin .pow a b => WFE Γ a ∧ WFE Γ b ∧
      ((isRealTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)) ∨
       (isIntTy (Tr.tyOf Γ a) ∧ ∃ k n, b = .intLit k n ∧ 0 ≤ n))
  | .bin _ a b => WFE Γ a ∧ WFE Γ b ∧ numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)
  | .cmp op a b => WFE Γ a ∧ WFE Γ b ∧
      ((numericTy (Tr.tyOf Γ a) ∧ numericTy (Tr.tyOf Γ b)) ∨
       (isStrTy (Tr.tyOf Γ a) ∧ isStrTy (Tr.tyOf Γ b) ∧ op = .eq))
  | .logic _ a b => WFE Γ a ∧ WFE Γ b
  | .not e => WFE Γ e
  | .ref a (.cons (.scalar i) .nil) => WFE Γ i ∧ isIntTy (Tr.tyOf Γ i) ∧ ∃ d k, Γ a = some d ∧ d.lbs = [.intLit k 1]
  | .ref _ _ => False
  | .call f (.cons a .nil) none => WFE Γ a ∧ Call1OK f
  | .call f (.cons a (.cons b .nil)) none => WFE Γ a ∧ WFE Γ b ∧ Call2OK f
  | .call _ _ _ => False
  | .arrayCtor _ => False

/-! ### Static rank -/

theorem rankOf_zero (Γ : F.Env) : ∀ e, WFE Γ e → Tr.rankOf Γ e = some 0
  | .intLit _ _, _ => rfl
  | .realLit _ _, _ => rfl
  | .boolLit _, _ => rfl
  | .strLit _, _ => rfl
  | .var x, ⟨d, hd, hl⟩ => by simp [Tr.rankOf, hd, hl]
  | .neg e, h => by simp only [Tr.rankOf]; exact rankOf_zero Γ e h
  | .paren e, h => by simp only [Tr.rankOf]; exact rankOf_zero Γ e h
  | .not e, h => by simp only [Tr.rankOf]; exact rankOf_zero Γ e h
  | .bin op a b, h => by
      have ha : WFE Γ a := by cases op <;> exact h.1
      have hb : WFE Γ b := by cases op <;> exact h.2.1
      simp [Tr.rankOf, rankOf_zero Γ a ha, rankOf_zero Γ b hb]
  | .cmp _ a b, h => by simp [Tr.rankOf, rankOf_zero Γ a h.1, rankOf_zero Γ b h.2.1]
  | .logic _ a b, h => by simp [Tr.rankOf, rankOf_zero Γ a h.1, rankOf_zero Γ b h.2]
  | .ref a (.cons (.scalar i) .nil), _ => by simp [Tr.rankOf, F.Subs.rank]
  | .call f (.cons a .nil) none, ⟨ha, hf⟩ => by
      have := rankOf_zero Γ a ha
      cases f <;> simp [Call1OK] at hf <;> simp [Tr.rankOf, Tr.rankOfArgs, this]
  | .call f (.cons a (.cons b .nil)) none, ⟨ha, hb, hf⟩ => by
      have h1 := rankOf_zero Γ a ha
      have h2 := rankOf_zero Γ b hb
      cases f <;> simp [Call2OK] at hf <;> simp [Tr.rankOf, Tr.rankOfArgs, h1, h2]

/-! ### Type preservation (kinds erased) -/

/-- The static type of a binary numeric operation (`Tr.tyOf`'s join). -/
def joinTy : FTy → FTy → Option FTy
  | .int k₁, .int k₂ => some (.int (IKind.join k₁ k₂))
  | .real k₁, .real k₂ => some (.real (RKind.join k₁ k₂))
  | .int _, .real k => some (.real k)
  | .real k, .int _ => some (.real k)
  | _, _ => none

theorem tyOf_bin (Γ : F.Env) (op : F.BinOp) (a b : F.Expr) :
    Tr.tyOf Γ (.bin op a b) = (Tr.tyOf Γ a).bind fun ta => (Tr.tyOf Γ b).bind fun tb => joinTy ta tb := by
  cases h1 : Tr.tyOf Γ a with
  | none => simp [Tr.tyOf, h1]
  | some ta =>
    cases h2 : Tr.tyOf Γ b with
    | none => simp [Tr.tyOf, h1, h2]
    | some tb => cases ta <;> cases tb <;> simp [Tr.tyOf, h1, h2, joinTy]

/-- A well-formed expression evaluates to the static base type. -/
theorem eval_ty (o : F.Oracle) (Γ : F.Env) (σ : F.State) (hC : Conforms o Γ σ) :
    ∀ e, WFE Γ e → ∀ ty, Tr.tyOf Γ e = some ty → ∀ v, F.eval o σ e = some (.s v) → v.ty.base = ty.base
  | .intLit _ _, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv; rfl
  | .realLit _ _, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv; rfl
  | .boolLit _, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv; rfl
  | .strLit _, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv; rfl
  | .var x, ⟨d, hd, hl⟩, ty, hty, v, hv => by
      simp only [Tr.tyOf, hd, Option.map_some, Option.some.injEq] at hty; subst hty
      obtain ⟨sv, hσ, hsty⟩ := conforms_scalar hC x d hd hl
      simp only [F.eval, hσ, Option.some.injEq, F.Val.s.injEq] at hv; subst hv
      rw [hsty]
  | .paren e, h, ty, hty, v, hv => eval_ty o Γ σ hC e h ty hty v (by simpa [F.eval] using hv)
  | .neg e, h, ty, hty, v, hv => by
      simp only [Tr.tyOf] at hty
      simp only [F.eval, Option.bind_eq_some_iff] at hv
      obtain ⟨u, hu, hlift⟩ := hv
      obtain ⟨a, rfl, hfa⟩ := F_lift1_eq_scalar hlift
      have := eval_ty o Γ σ hC e h ty hty a hu
      cases a with
      | int k n => simp only [Option.some.injEq] at hfa; subst hfa; simpa [SVal.ty] using this
      | real k r => simp only [Option.some.injEq] at hfa; subst hfa; simpa [SVal.ty] using this
      | bool _ => simp at hfa
      | str _ => simp at hfa
  | .bin op a b, h, ty, hty, v, hv => by
      have ha : WFE Γ a := by cases op <;> exact h.1
      have hb : WFE Γ b := by cases op <;> exact h.2.1
      rw [F_eval_bin] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
      rw [tyOf_bin] at hty
      simp only [Option.bind_eq_some_iff] at hty
      obtain ⟨ta, hta, tb, htb, hres⟩ := hty
      have hx := eval_ty o Γ σ hC a ha ta hta x hva
      have hy := eval_ty o Γ σ hC b hb tb htb y hvb
      cases ta <;> cases tb <;> simp only [joinTy, Option.some.injEq, reduceCtorEq] at hres <;> subst hres <;>
        cases x <;> cases y <;> simp [SVal.ty, FTy.base] at hx hy <;>
        cases op <;> simp only [F.arith] at hxy <;> (repeat' split at hxy) <;>
        simp only [Option.some.injEq, Option.map_eq_some_iff, reduceCtorEq] at hxy <;>
        first
        | (obtain ⟨_, _, rfl⟩ := hxy; rfl)
        | (subst hxy; rfl)
  | .cmp op a b, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      rw [F_eval_cmp] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
      cases op <;> cases x <;> cases y <;> simp [F.compare] at hxy <;> (try subst hxy) <;> rfl
  | .logic op a b, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      rw [F_eval_logic] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
      obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
      cases op <;> cases x <;> cases y <;> simp [F.logic] at hxy <;> (try subst hxy) <;> rfl
  | .not e, _, ty, hty, v, hv => by
      simp only [Tr.tyOf, Option.some.injEq] at hty; subst hty
      rw [F_eval_not] at hv
      simp only [Option.bind_eq_some_iff] at hv
      obtain ⟨u, hu, hlift⟩ := hv
      obtain ⟨a, rfl, hfa⟩ := F_lift1_eq_scalar hlift
      cases a <;> simp at hfa <;> subst hfa <;> rfl
  | .ref a (.cons (.scalar i) .nil), ⟨_, _, d, k, hd, hl⟩, ty, hty, v, hv => by
      simp only [Tr.tyOf, hd, Option.map_some, Option.some.injEq] at hty; subst hty
      obtain ⟨av, hσ, hety, hlen, _, hwt⟩ := conforms_array hC a d hd (by rw [hl]; simp)
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
            cases hrest
            rw [hwt [iv] (inBounds_rank1 av lb n hdims iv hin), hety]
          · rw [sectionOf_fixed_oob av lb n iv hdims hin] at hrest
            cases hrest
        | real _ _ => simp at hm
        | bool _ => simp at hm
        | str _ => simp at hm
  | .call f (.cons a .nil) none, ⟨ha, hf⟩, ty, hty, v, hv => by
      rw [F_eval_call1] at hv
      simp only [Option.map_none, Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, hint⟩ := hv
      cases va with
      | arr av =>
        exfalso
        cases f <;> simp [Call1OK] at hf <;>
          first
          | exact intrinsic1_arr_ne _ _ av v (by simp) hint
          | exact lift1_arr_ne _ _ _ hint
      | s x =>
        cases f <;> simp [Call1OK] at hf <;> simp only [Tr.tyOf, Tr.tyOfCall] at hty <;>
        first
        | (have hx := eval_ty o Γ σ hC a ha ty hty x hva
           cases x <;> simp [F.intrinsic, F.intrinsicCore, F.lift1, F.realFn] at hint <;>
             first
             | (obtain ⟨_, _, rfl⟩ := hint; exact hx)
             | (subst hint; exact hx))
        | (simp only [Option.some.injEq, reduceCtorEq, if_false] at hty; subst hty
           cases x <;> simp [F.intrinsic, F.intrinsicCore, F.lift1] at hint <;> subst hint <;> rfl)
  | .call f (.cons a (.cons b .nil)) none, ⟨ha, hb, hf⟩, ty, hty, v, hv => by
      rw [F_eval_call2] at hv
      simp only [Option.map_none, Option.bind_eq_some_iff] at hv
      obtain ⟨va, hva, vb, hvb, hint⟩ := hv
      have hf' : f = .mod ∨ f = .modulo ∨ f = .sign ∨ f = .dim ∨ f = .max ∨ f = .min ∨
          f = .atan2 ∨ f = .iand ∨ f = .ior ∨ f = .ieor ∨ f = .ishft := by
        cases f <;> simp [Call2OK] at hf <;> simp
      cases va with
      | arr av => exact absurd hint (intrinsic2_arr_ne _ _ _ _ _ (Or.inl ⟨av, rfl⟩) hf')
      | s x =>
        cases vb with
        | arr bv => exact absurd hint (intrinsic2_arr_ne _ _ _ _ _ (Or.inr ⟨bv, rfl⟩) hf')
        | s y =>
          have hty' : Tr.tyOf Γ a = some ty := by
            cases f <;> simp [Call2OK] at hf <;> simpa [Tr.tyOf, Tr.tyOfCall] using hty
          have hx := eval_ty o Γ σ hC a ha ty hty' x hva
          cases f <;> simp [Call2OK] at hf <;> cases x <;> cases y <;>
            first
            | (refine (intrinsic2_mixed_absurd ?_ hint ?_).elim <;> simp <;> done)
            | (simp [F.intrinsic, F.intrinsicCore, F.lift2, F.maxS, F.minS] at hint <;> (repeat' split at hint) <;>
               (try simp only [Option.some.injEq, Option.map_eq_some_iff, reduceCtorEq] at hint) <;>
               first
               | (obtain ⟨_, _, rfl⟩ := hint; exact hx)
               | (obtain ⟨_, rfl⟩ := hint; exact hx)
               | (subst hint; exact hx))
termination_by e => sizeOf e
decreasing_by all_goals simp_wf <;> omega

end Recast.Rules
