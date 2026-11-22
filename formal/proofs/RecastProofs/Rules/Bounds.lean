import RecastProofs.Rules.Stmts2

/-!
# Array bounds fixed when the array comes into being (FNP-R0049, FNP-R0160)

An array's bounds are fixed when it comes into being: on entry for an
explicit-shape dummy or an automatic local (F2018 10.1.11), at `ALLOCATE` for an
allocatable (F2018 9.7.1.2).  Redefining a variable the bound expression
mentions afterwards does not change them.  The emitter records the bound
*text* and re-renders it at every use (`a[(i) - (<lb text>)]`), so the target
evaluates it in the current state.

`Rank1Decl` (Index.lean), and `Conforms` behind it, state the lower-bound
premise in the current state; that is what the source does *not* guarantee.
`FNP_R0049_obligation` states the subscript rule with the premise the source
does guarantee -- the array's dims are the ones `ALLOCATE` fixed, evaluated in
the allocation state -- and `FNP_R0049_refuted` refutes it with the probe's
`redef` (FNP-D0049): `allocate(a(-n:n))` with `n = 2`, then `n = n + 1`,
then `a(0)`.

`FNP_R0160_refuted_empty`: `allocate(a(lo:hi))` with `hi < lo` is a zero-sized
array in Fortran (`F.execAllocate` clamps the extent with `toNat`), while the
emitted `np.zeros(((hi) - (lo) + 1,), ...)` is undefined on a negative extent
(FNP-D0050).  `FNP_R0160_sound` (Stmts2.lean) remains the proved sub-case:
`allocate(x(hi))` with a unit lower bound and a non-negative extent.
-/

namespace Recast.Rules

open Recast Rel Exact

/-- `Γ` with `a` removed: the declarations whose bounds are still tied to the
current values of their bound expressions. -/
def envWithout (Γ : F.Env) (a : String) : F.Env := fun x => if x = a then none else Γ x

theorem relState_envWithout {Γ : F.Env} {σ : F.State} {τ : P.State} (h : RelState Γ σ τ) (a : String) :
    RelState (envWithout Γ a) σ τ := by
  intro x d hx
  simp only [envWithout] at hx
  split_ifs at hx
  exact h x d hx

/-! ### FNP-R0049: subscripts on an allocated array with a variable lower bound -/

/-- FNP-R0049 with the premise the source guarantees: `a` was allocated by
`allocate (a(lbE:hiE))` in some state `σ₀`, so its lower bound is `lbE`
evaluated in `σ₀`; in the use state `σ` the array still has those dims (its
elements may have been assigned since).  Every other declared name conforms to
its declaration in `σ`; the subscript and the rendered bound are sound
renderings in `σ`.  The use state is *any* related state with these
properties -- it is not required to be reachable from `σ₁`, but the
counterexample below is (the probe's `redef` reaches it). -/
def FNP_R0049_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (lbE hiE : F.Expr) (lb' : P.Expr) (i : F.Expr) (i' : P.Expr),
    SoundScalar o (envWithout Γ a) i i' → SoundScalar o (envWithout Γ a) lbE lb' →
    IntTyped o (envWithout Γ a) i →
    ∀ σ₀ σ₁, F.Exec o.F Γ (.allocate a [(some lbE, hiE)]) σ₀ σ₁ .normal →
    ∀ σ τ av, RelState Γ σ τ → Conforms o.F (envWithout Γ a) σ → σ a = some (.arr av) →
      (∃ av₁, σ₁ a = some (.arr av₁) ∧ av.dims = av₁.dims) →
      ∀ v, F.eval o.F σ (.ref a (.cons (.scalar i) .nil)) = some v →
        ∃ w, P.eval o.P τ (.index a (.cons (.scalar (.bin .sub (.paren i') (.paren lb'))) .nil)) = some w ∧
          RelV v w

/-- `real(8), allocatable :: a(:)` whose ALLOCATE is recorded with lower bound `-n`, and `integer :: n`. -/
def declAN : F.Decl := ⟨.real .r8, [.neg (.var "n")], 0⟩
def declN : F.Decl := ⟨.int .i4, [], 0⟩
def envAN : F.Env := fun x => if x = "a" then some declAN else if x = "n" then some declN else none

theorem envAN_n : envAN "n" = some declN := by simp [envAN]
theorem envWithout_AN_n : envWithout envAN "a" "n" = some declN := by simp [envWithout, envAN]

/-- The rendered lower bound `- n` is a sound rendering of `-n` (FNP-R0014, FNP-R0010). -/
theorem soundScalar_negN (o : Oracles) (Γ : F.Env) (h : Γ "n" = some declN) :
    SoundScalar o Γ (.neg (.var "n")) (.neg (.name "n")) :=
  FNP_R0014_sound o Γ (.var "n") (.name "n") (FNP_R0010_sound o Γ "n" declN h).toScalar

/-- The state at the ALLOCATE: `n = 2`, `a` not yet allocated. -/
def stateAlloc : F.State := fun x => if x = "n" then some (.s (.int .i4 2)) else none

/-- What `allocate (a(-n:n))` makes of it: `a` has dims `[(-2, 5)]`. -/
def arrAlloc : FArr := { elemTy := .real .r8, dims := [(-2, 5)], get := fun _ => .real .r8 0 }

theorem exec_allocAN :
    F.Exec noPrim.F envAN (.allocate "a" [(some (.neg (.var "n")), .var "n")]) stateAlloc
      (Function.update stateAlloc "a" (some (.arr arrAlloc))) .normal := by
  refine F.Exec.allocate ?_
  simp [F.execAllocate, envAN, declAN, stateAlloc, F.evalInt, F.eval, F.lift1, F.zeroOf, arrAlloc]

/-- After the loop `a(i) = 10*i`: the array the use state holds. -/
def arrUse : FArr := { elemTy := .real .r8, dims := [(-2, 5)], get := fun idx => .real .r8 (10 * (idx.headD 0 : ℝ)) }
def parrUse : PArr := { shape := [5], get := fun idx => .float (10 * (((idx.headD 0 : ℕ) : ℝ) - 2)) }

theorem relUse : RelA arrUse parrUse := by
  refine ⟨rfl, ?_⟩
  intro idx hidx
  obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ -2 ≤ j := by
    simp only [FArr.inBounds, arrUse] at hidx
    match idx, hidx with
    | [j], hidx =>
      simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
      exact ⟨j, rfl, hidx.1.1⟩
    | [], hidx => simp [FArr.inBoundsB] at hidx
    | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
  simp only [rebase, arrUse, parrUse, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.headD_cons]
  refine RelS.real ?_
  have hn : (((j - -2).toNat : ℕ) : ℤ) = j + 2 := by omega
  have hr : (((j - -2).toNat : ℕ) : ℝ) = (j : ℝ) + 2 := by
    rw [← Int.cast_natCast, hn]; push_cast; ring
  simp only [P.toR, hr, Option.some.injEq]
  ring

/-- The use state after `n = n + 1`: `n = 3`, `a` unchanged. -/
def stateUse : F.State := fun x =>
  if x = "a" then some (.arr arrUse) else if x = "n" then some (.s (.int .i4 3)) else none
def pstateUse : P.State := fun x =>
  if x = "a" then some (.arr parrUse) else if x = "n" then some (.s (.int 3)) else none

theorem relStateUse : RelState envAN stateUse pstateUse := by
  intro x d hx
  simp only [envAN] at hx
  split_ifs at hx with h1 h2 <;> cases hx
  · subst h1; exact ⟨.arr arrUse, .arr parrUse, by simp [stateUse], by simp [pstateUse], RelV.arr relUse⟩
  · subst h2; exact ⟨.s (.int .i4 3), .s (.int 3), by simp [stateUse], by simp [pstateUse], RelV.s RelS.int⟩

theorem conformsUse : Conforms noPrim.F (envWithout envAN "a") stateUse := by
  intro x d hx
  simp only [envWithout, envAN] at hx
  split_ifs at hx with h1 h2
  cases hx
  subst h2
  exact ⟨fun _ => ⟨.int .i4 3, by simp [stateUse], rfl⟩, fun h => absurd rfl h⟩

/-- FNP-D0049: Fortran's `a(0)` is the element at offset 2 from the allocated
lower bound -2 (value 0.0); the emitted `a[(0) - (- n)]` with the new `n = 3`
reads offset 3 (value 10.0). -/
theorem FNP_R0049_refuted : ¬ FNP_R0049_obligation := by
  intro h
  have hs := h noPrim envAN "a" (.neg (.var "n")) (.var "n") (.neg (.name "n")) (.intLit .i4 0) (.intLit 0)
    (FNP_R0001_sound _ _ _ _) (soundScalar_negN _ _ envWithout_AN_n) (intLit_intTyped _ _ .i4 0)
    stateAlloc _ exec_allocAN stateUse pstateUse arrUse relStateUse conformsUse (by simp [stateUse])
    ⟨arrAlloc, by simp, rfl⟩
  -- the source: `a(0)` is in bounds of `-2:2`
  have hF : F.eval noPrim.F stateUse (.ref "a" (.cons (.scalar (.intLit .i4 0)) .nil))
      = some (.s (arrUse.get [0])) := by
    rw [F_eval_ref]
    simp only [stateUse, ite_true, Option.bind_some]
    rw [F_evalSubs_scalar1, F_eval_intLit, Option.bind_some]
    exact sectionOf_fixed arrUse (-2) 5 0 rfl (by norm_num)
  obtain ⟨w, hw, hrel⟩ := hs _ hF
  -- the target: `(0) - (- n)` is `3` in the use state
  have hP : P.eval noPrim.P pstateUse
      (.index "a" (.cons (.scalar (.bin .sub (.paren (.intLit 0)) (.paren (.neg (.name "n"))))) .nil))
      = some (.s (parrUse.get [3])) := by
    rw [P_eval_index]
    simp only [pstateUse, ite_true, Option.bind_some]
    rw [P_evalIdxs_scalar1]
    have hi : P.eval noPrim.P pstateUse (.bin .sub (.paren (.intLit 0)) (.paren (.neg (.name "n"))))
        = some (.s (.int 3)) := by
      rw [P_eval_bin, P_eval_paren, P_eval_paren]
      simp [P.eval, P.lift1, P.lift2, P.arith, pstateUse]
    rw [hi, Option.bind_some]
    exact indexArr_at parrUse 5 3 rfl (by norm_num)
  rw [hP] at hw
  cases hw
  cases hrel with
  | s hS =>
    have := RelS.real_toR hS
    simp only [parrUse, P.toR, List.headD_cons, Option.some.injEq] at this
    norm_num at this

/-! ### FNP-R0160: ALLOCATE with an explicit lower bound -/

/-- The emitted extent `(hi') - (lo') + 1` (`statements.py:1005-1007`). -/
def allocExtent (lo' hi' : P.Expr) : P.Expr := .bin .add (.bin .sub (.paren hi') (.paren lo')) (.intLit 1)

/-- FNP-R0160 as the engine accepts it: `allocate (x(lo:hi))` with a lower
bound other than the literal `1` becomes `x = np.zeros(((hi') - (lo') + 1,), dtype)`,
for any integer bounds. -/
def FNP_R0160_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (x : String) (d : F.Decl), Γ x = some d → NumericTy d.ty →
    ∀ (lo hi : F.Expr) (lo' hi' : P.Expr), SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
      IntTyped o Γ lo → IntTyped o Γ hi →
      StmtSound o Γ (.allocate x [(some lo, hi)])
        (.assign x (.call (zerosHelper d.ty) (.cons (allocExtent lo' hi') .nil)))

/-- `np.zeros((n,), dtype=np.float64)` on a negative extent: NumPy raises. -/
theorem helper_zeros_float_neg (o : P.Oracle) (n : ℤ) (hn : n < 0) :
    P.helper o .np_zeros_float [.s (.int n)] = none := by
  have e : P.helper o .np_zeros_float [.s (.int n)]
      = ((([.s (.int n)] : List P.Val).mapM fun (v : P.Val) => match v with
          | .s (.int m) => if 0 ≤ m then some m.toNat else none | _ => none).map
          fun ns => P.Val.arr { shape := ns, get := fun _ => .float 0 }) := rfl
  rw [e]
  simp [show ¬ (0 ≤ n) by omega]

/-- Before the ALLOCATE: `n = -1`, and `a` holds some array whose lower bound
is `-n = 1` (the previous binding is irrelevant: ALLOCATE replaces it). -/
def stateEmpty : F.State := fun x =>
  if x = "a" then some (.arr twoF) else if x = "n" then some (.s (.int .i4 (-1))) else none
def pstateEmpty : P.State := fun x =>
  if x = "a" then some (.arr twoP) else if x = "n" then some (.s (.int (-1))) else none

theorem relStateEmpty : RelState envAN stateEmpty pstateEmpty := by
  intro x d hx
  simp only [envAN] at hx
  split_ifs at hx with h1 h2 <;> cases hx
  · subst h1; exact ⟨.arr twoF, .arr twoP, by simp [stateEmpty], by simp [pstateEmpty], RelV.arr twoRel⟩
  · subst h2; exact ⟨.s (.int .i4 (-1)), .s (.int (-1)), by simp [stateEmpty], by simp [pstateEmpty], RelV.s RelS.int⟩

theorem conformsEmpty : Conforms noPrim.F envAN stateEmpty := by
  intro x d hx
  simp only [envAN] at hx
  split_ifs at hx with h1 h2 <;> cases hx
  · subst h1
    refine ⟨fun h => by simp [declAN] at h, fun _ => ⟨twoF, by simp [stateEmpty], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
    intro k hk
    simp only [declAN] at hk
    match k, hk with
    | 0, _ => simp [declAN, stateEmpty, F.eval, F.lift1, twoF, FArr.lbs]
  · subst h2; exact ⟨fun _ => ⟨.int .i4 (-1), by simp [stateEmpty], rfl⟩, fun h => absurd rfl h⟩

/-- After `allocate (a(-n:n))` with `n = -1`: the zero-sized `a(1:0)`. -/
def stateEmptyAlloc : F.State :=
  Function.update stateEmpty "a" (some (.arr { elemTy := .real .r8, dims := [(1, 0)], get := fun _ => .real .r8 0 }))

theorem intTyped_varN(o : Oracles) (Γ : F.Env) (h : Γ "n" = some declN) : IntTyped o Γ (.var "n") :=
  intTyped_of_tyOf o Γ _ ⟨declN, h, rfl⟩ .i4 (by simp [Tr.tyOf, h, declN])

theorem intTyped_negN (o : Oracles) (Γ : F.Env) (h : Γ "n" = some declN) : IntTyped o Γ (.neg (.var "n")) :=
  intTyped_of_tyOf o Γ _ ⟨declN, h, rfl⟩ .i4 (by simp [Tr.tyOf, h, declN])

/-- FNP-D0050: `allocate (a(-n:n))` with `n = -1` is a zero-sized array in
Fortran; the emitted `np.zeros(((n) - ((-n)) + 1,), dtype=np.float64)` has the
extent `-1`, on which `np.zeros` is undefined, so the target has no execution. -/
theorem FNP_R0160_refuted_empty : ¬ FNP_R0160_obligation := by
  intro h
  have hs := h noPrim envAN "a" declAN (by simp [envAN]) (Or.inr ⟨.r8, rfl⟩)
    (.neg (.var "n")) (.var "n") (.paren (.neg (.name "n"))) (.name "n")
    (FNP_R0014_sound _ _ _ _ (FNP_R0010_sound _ _ "n" declN envAN_n).toScalar)
    (FNP_R0010_sound _ _ "n" declN envAN_n).toScalar
    (intTyped_negN _ _ envAN_n) (intTyped_varN _ _ envAN_n)
  -- the source: a zero-sized array with dims [(1, 0)]
  have hF : F.Exec noPrim.F envAN (.allocate "a" [(some (.neg (.var "n")), .var "n")]) stateEmpty
      stateEmptyAlloc .normal := by
    refine F.Exec.allocate ?_
    simp [F.execAllocate, envAN, declAN, stateEmpty, stateEmptyAlloc, F.evalInt, F.eval, F.lift1, F.zeroOf]
  obtain ⟨τ', hex, -, -⟩ := hs stateEmpty pstateEmpty relStateEmpty conformsEmpty stateEmptyAlloc .normal hF
  -- the target: `np.zeros((-1,), ...)` is undefined
  have hP : P.eval noPrim.P pstateEmpty
      (.call (zerosHelper declAN.ty) (.cons (allocExtent (.paren (.neg (.name "n"))) (.name "n")) .nil)) = none := by
    have hz : zerosHelper declAN.ty = .np_zeros_float := rfl
    have hc : P.Expr.call .np_zeros_float (.cons (allocExtent (.paren (.neg (.name "n"))) (.name "n")) .nil)
        = Tr.fold1 .np_zeros_float (allocExtent (.paren (.neg (.name "n"))) (.name "n")) := rfl
    rw [hz, hc, P_eval_call1]
    have he : P.eval noPrim.P pstateEmpty (allocExtent (.paren (.neg (.name "n"))) (.name "n")) = some (.s (.int (-1))) := by
      simp only [allocExtent]
      rw [P_eval_bin, P_eval_bin, P_eval_paren, P_eval_paren, P_eval_paren]
      simp [P.eval, P.lift1, P.lift2, P.arith, pstateEmpty]
    rw [he, Option.bind_some]
    exact helper_zeros_float_neg noPrim.P (-1) (by norm_num)
  cases hex with
  | assign hev => rw [hP] at hev; cases hev

end Recast.Rules
