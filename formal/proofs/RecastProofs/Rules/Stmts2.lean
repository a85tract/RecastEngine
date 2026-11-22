import RecastProofs.Rules.Obligations
import RecastProofs.Rules.Reductions
import RecastProofs.Rules.Control2
import RecastProofs.Rules.StmtSoundness

/-!
# Rules for the Semantics v1.1 statements and the vector subscript

* FNP-R0041: vector subscript `a(v)` on a unit-origin rank-1 array ↦ `a[((v') - 1)]`.
* FNP-R0122: `SELECT CASE` as the if/elif chain (numeric selectors).
* FNP-R0160: `ALLOCATE` with literal lower bounds ↦ `np.zeros(...)`.
* FNP-R0124: `ASSOCIATE` with a scalar selector, refuted (FNP-D0015).
* FNP-R0137: `WHERE` with whole-array rank-1 targets.
* FNP-R0121: `FORALL`, refuted (FNP-D0002 evaluation order, FNP-D0003 dropped mask),
  and sound when the right-hand side does not read the target.
-/

namespace Recast.Rules

open Recast Rel Exact

/-! ### List helpers -/

/-- A `mapM` over `Option` whose function is everywhere `some` is the `map`. -/
theorem mapM_eq_map_of_some {α β : Type} (f : α → Option β) (g : α → β) :
    ∀ (l : List α), (∀ x ∈ l, f x = some (g x)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.mapM_cons, h x (List.mem_cons_self ..), mapM_eq_map_of_some f g xs (fun y hy => h y (List.mem_cons_of_mem _ hy))]
    rfl

theorem getD_map_range {β : Type} (g : ℕ → β) (n k : ℕ) (hk : k < n) (d : β) :
    ((List.range n).map g).getD k d = g k := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk]
  rfl

theorem mem_map_range {β : Type} (g : ℕ → β) (n : ℕ) (y : β) :
    y ∈ (List.range n).map g ↔ ∃ k, k < n ∧ g k = y := by
  simp only [List.mem_map, List.mem_range]

/-- `F.evalSub` on a vector subscript iterates over `List.range n` *coerced to `ℤ`*
(the index arithmetic `lb + k` forces `k : ℤ`, so the elaborated list is the
List-monad `do let a ← List.range n; pure ↑a`).  This is that `mapM`. -/
theorem mapM_coe {β : Type} (f : ℤ → Option β) (g : ℕ → β) :
    ∀ (l : List ℕ), (∀ k ∈ l, f (k : ℤ) = some (g k)) →
      (do let a ← l; pure (a : ℤ)).mapM f = some (l.map g)
  | [], _ => rfl
  | x :: xs, h => by
    have hcons : (do let a ← (x :: xs); pure (a : ℤ)) = (x : ℤ) :: (do let a ← xs; pure (a : ℤ)) := rfl
    rw [hcons, List.mapM_cons, h x (List.mem_cons_self ..),
      mapM_coe f g xs (fun y hy => h y (List.mem_cons_of_mem _ hy))]
    rfl

/-! ### FNP-R0041: vector subscripts (unit origin, rank 1) -/

/-- The INTEGER values of a rank-1 array, as `F.evalSub` reads them. -/
def intVals (av : FArr) (lb : ℤ) (n : ℕ) : Option (List ℤ) :=
  (do let a ← List.range n; pure (a : ℤ)).mapM fun k => match av.get [lb + k] with
    | .int _ i => some i
    | _ => none

/-- The value read at position `k` (`0` on a non-INTEGER element). -/
def intAt (av : FArr) (lb : ℤ) (k : ℕ) : ℤ :=
  match av.get [lb + k] with
  | .int _ i => i
  | _ => 0

theorem intVals_of_int (av : FArr) (lb : ℤ) (n : ℕ)
    (h : ∀ k, k < n → ∃ kk i, av.get [lb + k] = .int kk i) :
    intVals av lb n = some ((List.range n).map (intAt av lb)) := by
  refine mapM_coe _ (intAt av lb) (List.range n) ?_
  intro k hk
  rw [List.mem_range] at hk
  obtain ⟨kk, i, hi⟩ := h k hk
  simp [hi, intAt]

/-- Bind form of `F.evalSub` on a vector subscript (the `do` block unfolds to a
raw `match`, so this is the shape rewriting needs). -/
theorem F_evalSub_vector (o : F.Oracle) (σ : F.State) (e : F.Expr) :
    F.evalSub o σ (.vector e) = (F.eval o σ e).bind fun v => match v with
      | .arr av => (match av.dims with
          | [(lb, n)] => (intVals av lb n).bind fun cs => some (F.Axis.vector cs)
          | _ => none)
      | .s _ => none := by
  cases h : F.eval o σ e <;> simp only [F.evalSub, h] <;> rfl

theorem F_evalSubs_vector1 (o : F.Oracle) (σ : F.State) (e : F.Expr) (av : FArr) (lb : ℤ) (n : ℕ) (cs : List ℤ)
    (hv : F.eval o σ e = some (.arr av)) (hd : av.dims = [(lb, n)]) (hcs : intVals av lb n = some cs) :
    F.evalSubs o σ (.cons (.vector e) .nil) = some [.vector cs] := by
  have h1 : F.evalSub o σ (.vector e) = some (.vector cs) := by
    rw [F_evalSub_vector, hv, Option.bind_some]
    simp only
    rw [hd]
    show (intVals av lb n).bind (fun cs => some (F.Axis.vector cs)) = some (.vector cs)
    rw [hcs, Option.bind_some]
  simp only [F.evalSubs, h1, Option.bind_some, Option.pure_def]
  rfl

/-- The Fortran side: a vector subscript whose values are in bounds. -/
theorem sectionOf_vector (a : FArr) (lb : ℤ) (n : ℕ) (cs : List ℤ) (hd : a.dims = [(lb, n)])
    (hin : ∀ i ∈ cs, lb ≤ i ∧ i < lb + n) :
    ∃ r : FArr, F.sectionOf a [.vector cs] = some (.arr r) ∧ r.dims = [(1, cs.length)] ∧
      ∀ j : ℤ, r.get [j] = a.get [cs.getD (j - 1).toNat 0] := by
  have hin' : (∀ x ∈ cs, lb ≤ x ∧ x < lb + (n : ℤ)) = True := eq_true hin
  cases hsec : F.sectionOf a [.vector cs] with
  | none => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
  | some v =>
    cases v with
    | s _ => exfalso; simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
    | arr r =>
      simp [F.sectionOf, F.resolveAxis, hd, hin'] at hsec
      subst hsec
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

theorem sectionOf_vector_oob (a : FArr) (lb : ℤ) (n : ℕ) (cs : List ℤ) (hd : a.dims = [(lb, n)])
    (hout : ¬ ∀ i ∈ cs, lb ≤ i ∧ i < lb + n) : F.sectionOf a [.vector cs] = none := by
  simp [F.sectionOf, F.resolveAxis, hd, hout]

/-- The Python side: `evalIdx` of an integer index array. -/
def pyIntVals (pv : PArr) (n : ℕ) : Option (List ℤ) :=
  (List.range n).mapM fun k => match pv.get [k] with
    | .int i => some i
    | _ => none

/-- Bind form of `P.evalIdx` on an index array. -/
theorem P_evalIdx_array (o : P.Oracle) (τ : P.State) (e : P.Expr) :
    P.evalIdx o τ (.array e) = (P.eval o τ e).bind fun v => match v with
      | .arr av => (match av.shape with
          | [n] => (pyIntVals av n).bind fun is => some (P.PIdx.gather is)
          | _ => none)
      | _ => none := by
  cases h : P.eval o τ e <;> simp only [P.evalIdx, h] <;> rfl

theorem P_evalIdxs_array1 (o : P.Oracle) (τ : P.State) (e : P.Expr) (pv : PArr) (n : ℕ) (is : List ℤ)
    (hv : P.eval o τ e = some (.arr pv)) (hs : pv.shape = [n]) (his : pyIntVals pv n = some is) :
    P.evalIdxs o τ (.cons (.array e) .nil) = some [.gather is] := by
  have h1 : P.evalIdx o τ (.array e) = some (.gather is) := by
    rw [P_evalIdx_array, hv, Option.bind_some]
    simp only
    rw [hs]
    show (pyIntVals pv n).bind (fun is => some (P.PIdx.gather is)) = some (.gather is)
    rw [his, Option.bind_some]
  simp only [P.evalIdxs, h1, Option.bind_some, Option.pure_def]
  rfl

theorem wrapIndex_of_range (n : ℕ) (i : ℤ) (hi : 0 ≤ i ∧ i < n) : Py.wrapIndex n i = some i.toNat := by
  have h1 : ¬ (i < 0) := by omega
  simp [Py.wrapIndex, h1, hi.1, hi.2]

theorem indexArr_gather (p : PArr) (n : ℕ) (is : List ℤ) (hs : p.shape = [n])
    (hin : ∀ i ∈ is, 0 ≤ i ∧ i < n) :
    ∃ r : PArr, P.indexArr p [.gather is] = some (.arr r) ∧ r.shape = [is.length] ∧
      ∀ j : ℕ, r.get [j] = p.get [(is.map Int.toNat).getD j 0] := by
  have hL : is.mapM (Py.wrapIndex n) = some (is.map Int.toNat) :=
    mapM_eq_map_of_some _ _ is (fun i hi => wrapIndex_of_range n i (hin i hi))
  cases hidx : P.indexArr p [.gather is] with
  | none => exfalso; simp [P.indexArr, hs, hL] at hidx
  | some v =>
    cases v with
    | s _ => exfalso; simp [P.indexArr, hs, hL] at hidx
    | slice _ _ _ => exfalso; simp [P.indexArr, hs, hL] at hidx
    | arr r =>
      simp [P.indexArr, hs, hL] at hidx
      subst hidx
      refine ⟨_, rfl, ?_, ?_⟩
      · simp
      · intro j; simp [List.getD_eq_getElem?_getD]

/-- The subscript array is a homogeneous INTEGER rank-1 array. -/
def IntVector (o : Oracles) (Γ : F.Env) (v : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ vv, F.eval o.F σ v = some (.arr vv) →
    ∃ lb n kv, vv.dims = [(lb, n)] ∧ vv.elemTy = .int kv ∧ FArr.WellTyped vv

def FNP_R0041_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a : String) (k : IKind) (v : F.Expr) (v' : P.Expr),
    Rank1Decl o Γ a (.intLit k 1) → SoundPair o Γ v v' → IntVector o Γ v →
    SoundPair o Γ (.ref a (.cons (.vector v) .nil))
      (.index a (.cons (.array (.bin .sub (.paren v') (.intLit 1))) .nil))

/-- `v' - 1` on a related INTEGER index array: the Python array of the values minus one. -/
theorem sub_one_arr (o : P.Oracle) (τ : P.State) (v' : P.Expr) (pv : PArr) (n : ℕ) (g : ℕ → ℤ)
    (hv : P.eval o τ v' = some (.arr pv)) (hs : pv.shape = [n]) (hg : ∀ k, k < n → pv.get [k] = .int (g k)) :
    ∃ pv' : PArr, P.eval o τ (.bin .sub (.paren v') (.intLit 1)) = some (.arr pv') ∧ pv'.shape = [n] ∧
      ∀ k, k < n → pv'.get [k] = .int (g k - 1) := by
  rw [P_eval_bin, P_eval_paren, hv, P_eval_intLit, Option.bind_some, Option.bind_some]
  by_cases hn : n = 0
  · subst hn
    have hsz : pv.size = 0 := PArr_size_rank1 pv 0 hs
    refine ⟨pv, ?_, hs, fun k hk => absurd hk (Nat.not_lt_zero k)⟩
    simp [P.lift2, hsz]
  · have hsz : pv.size ≠ 0 := by rw [PArr_size_rank1 pv n hs]; exact hn
    have h0 : pv.get (pv.shape.map fun _ => 0) = .int (g 0) := by
      rw [hs]; simpa using hg 0 (Nat.pos_of_ne_zero hn)
    refine ⟨{ shape := pv.shape, get := fun i => (P.arith .sub (pv.get i) (.int 1)).getD (.int (g 0 - 1)) }, ?_, hs, ?_⟩
    · simp only [P.lift2, hsz, ↓reduceIte, h0]
      simp [P.arith]
    · intro k hk
      simp [hg k hk, P.arith]

theorem FNP_R0041_sound : FNP_R0041_obligation := by
  intro o Γ a k v v' hdecl hv hint σ τ hR hC r hr
  obtain ⟨av, p, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
  rw [F_eval_intLit] at hl
  simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
  obtain ⟨_, rfl⟩ := hl
  rw [F_eval_ref, hσ, Option.bind_some] at hr
  simp only at hr
  -- the subscript array
  cases hvv : F.eval o.F σ v with
  | none => rw [show F.evalSubs o.F σ (.cons (.vector v) .nil) = none by simp [F.evalSubs, F.evalSub, hvv]] at hr; cases hr
  | some vval =>
    cases vval with
    | s _ => rw [show F.evalSubs o.F σ (.cons (.vector v) .nil) = none by simp [F.evalSubs, F.evalSub, hvv]] at hr; cases hr
    | arr vv =>
      obtain ⟨lbv, nv, kv, hdv, hty, hwt⟩ := hint σ τ hR hC vv hvv
      have hints : ∀ kk, kk < nv → ∃ k2 i, vv.get [lbv + kk] = .int k2 i := by
        intro kk hkk
        have hb : vv.inBounds [lbv + kk] := inBounds_rank1 vv lbv nv hdv _ ⟨by omega, by omega⟩
        have := hwt _ hb
        rw [hty] at this
        cases hg : vv.get [lbv + kk] with
        | int k2 i => exact ⟨k2, i, rfl⟩
        | real _ _ => rw [hg] at this; simp [SVal.ty] at this
        | bool _ => rw [hg] at this; simp [SVal.ty] at this
        | str _ => rw [hg] at this; simp [SVal.ty] at this
      have hcs := intVals_of_int vv lbv nv hints
      rw [F_evalSubs_vector1 o.F σ v vv lbv nv _ hvv hdv hcs, Option.bind_some] at hr
      set cs := (List.range nv).map (intAt vv lbv) with hcs_def
      by_cases hin : ∀ i ∈ cs, 1 ≤ i ∧ i < 1 + n
      · obtain ⟨rF, hrF, hrd, hrg⟩ := sectionOf_vector av 1 n cs hd hin
        rw [hrF] at hr
        cases hr
        -- the Python index array
        obtain ⟨w, hw, hrw⟩ := hv σ τ hR hC _ hvv
        cases hrw with
        | arr hA =>
          rename_i pv
          have hps := RelA_dims1 hA lbv nv hdv
          have hpg : ∀ kk, kk < nv → pv.get [kk] = .int (intAt vv lbv kk) := by
            intro kk hkk
            have hrs := RelA_get1 hA lbv nv hdv (lbv + kk) ⟨by omega, by omega⟩
            obtain ⟨k2, i, hi⟩ := hints kk hkk
            rw [hi] at hrs
            have : intAt vv lbv kk = i := by simp [intAt, hi]
            rw [this]
            have e : ((lbv + (kk : ℤ) - lbv).toNat) = kk := by omega
            rw [e] at hrs
            exact RelS.int_eq hrs
          obtain ⟨pv', hpv', hps', hpg'⟩ := sub_one_arr o.P τ v' pv nv (intAt vv lbv) hw hps hpg
          have his : pyIntVals pv' nv = some ((List.range nv).map fun kk => intAt vv lbv kk - 1) := by
            unfold pyIntVals
            apply mapM_eq_map_of_some
            intro kk hkk
            rw [List.mem_range] at hkk
            simp [hpg' kk hkk]
          set is := (List.range nv).map fun kk => intAt vv lbv kk - 1 with his_def
          have hin' : ∀ i ∈ is, 0 ≤ i ∧ i < n := by
            intro i hi
            rw [his_def, mem_map_range] at hi
            obtain ⟨kk, hkk, rfl⟩ := hi
            have := hin (intAt vv lbv kk) (by rw [hcs_def, mem_map_range]; exact ⟨kk, hkk, rfl⟩)
            omega
          have hpshape : p.shape = [n] := RelA_dims1 hrel 1 n hd
          obtain ⟨r', hr', hrs', hrg'⟩ := indexArr_gather p n is hpshape hin'
          refine ⟨.arr r', ?_, RelV.arr ?_⟩
          · rw [P_eval_index, hτ, Option.bind_some]
            simp only
            rw [P_evalIdxs_array1 o.P τ _ pv' nv is hpv' hps' his, Option.bind_some, hr']
          · refine ⟨?_, ?_⟩
            · rw [hrs']; simp [FArr.extents, hrd, his_def, hcs_def]
            · intro idx hidx
              obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ 1 ≤ j ∧ j < 1 + (cs.length : ℤ) := by
                simp only [FArr.inBounds, FArr.inBoundsB, hrd] at hidx
                match idx, hidx with
                | [j], hidx =>
                  simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
                  exact ⟨j, rfl, hidx.1⟩
                | [], hidx => simp [FArr.inBoundsB] at hidx
                | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
              have hlen : cs.length = nv := by simp [hcs_def]
              have hjn : (j - 1).toNat < nv := by omega
              rw [hrg j]
              simp only [rebase, FArr.lbs, hrd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
                List.zipWith_nil_right]
              rw [hrg' (j - 1).toNat]
              have e1 : cs.getD (j - 1).toNat 0 = intAt vv lbv (j - 1).toNat := by
                rw [hcs_def]; exact getD_map_range _ _ _ hjn 0
              have e2 : (is.map Int.toNat).getD (j - 1).toNat 0 = (intAt vv lbv (j - 1).toNat - 1).toNat := by
                rw [his_def, List.map_map]
                exact getD_map_range (fun kk => (intAt vv lbv kk - 1).toNat) nv _ hjn 0
              rw [e1, e2]
              have hc := hin (intAt vv lbv (j - 1).toNat) (by rw [hcs_def, mem_map_range]; exact ⟨_, hjn, rfl⟩)
              have := RelA_get1 hrel 1 n hd _ hc
              convert this using 3
      · rw [sectionOf_vector_oob av 1 n cs hd hin] at hr
        cases hr

/-! ### FNP-R0122: SELECT CASE (numeric selector) -/

def NumTyped (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop := IntTyped o Γ e ∨ RealTyped o Γ e

theorem soundScalar_paren' {o : Oracles} {Γ : F.Env} {e : F.Expr} {e' : P.Expr} (h : SoundScalar o Γ e e') :
    SoundScalar o Γ e (.paren e') := by
  intro σ τ hR hC v hv
  obtain ⟨w, hw, hr⟩ := h σ τ hR hC v hv
  exact ⟨w, by rw [P_eval_paren]; exact hw, hr⟩

theorem soundScalar_unparen' {o : Oracles} {Γ : F.Env} {e : F.Expr} {e' : P.Expr} (h : SoundScalar o Γ e (.paren e')) :
    SoundScalar o Γ e e' := by
  intro σ τ hR hC v hv
  obtain ⟨w, hw, hr⟩ := h σ τ hR hC v hv
  exact ⟨w, by rw [P_eval_paren] at hw; exact hw, hr⟩

/-- Numeric comparison with the numeric premise on the *right* operand. -/
theorem cmp_sound_right (o : Oracles) (Γ : F.Env) (op : F.CmpOp) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') (hnum : NumTyped o Γ b) :
    SoundScalar o Γ (.cmp op a b) (.paren (.cmp (toPCmp op) a' b')) := by
  intro σ τ hR hC v hv
  rw [F_eval_cmp] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb
  have hny : (∃ k n, y = .int k n) ∨ (∃ k r, y = .real k r) := by
    rcases hnum with h | h
    · exact Or.inl (h σ τ hR hC y hvb)
    · exact Or.inr (h σ τ hR hC y hvb)
  have hn : (∃ k n, x = .int k n) ∨ (∃ k r, x = .real k r) := by
    rcases hny with ⟨k, n, rfl⟩ | ⟨k, r, rfl⟩ <;> cases x <;> simp [F.compare] at hxy <;> simp
  obtain ⟨c', hc', hrc⟩ := compare_numeric_rel hrx hry hn hxy
  refine ⟨c', ?_, hrc⟩
  rw [P_eval_paren, P_eval_cmp, hx', hy']
  simp only [Option.bind_some, P_lift2_scalar, hc', Option.map_some]

/-- Scalar `.OR.` without the outer parentheses. -/
theorem or_sound (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr)
    (ha : SoundScalar o Γ a a') (hb : SoundScalar o Γ b b') :
    SoundScalar o Γ (.logic .or a b) (.or a' b') :=
  soundScalar_unparen' (FNP_R0034_sound o Γ a b a' b' ha hb).2

/-- The expressions a `CASE` item list mentions. -/
def itemExprs : F.CaseItem → List F.Expr
  | .value v => [v]
  | .range lo hi => lo.toList ++ hi.toList

def itemsExprs : List F.CaseItem → List F.Expr
  | [] => []
  | it :: rest => itemExprs it ++ itemsExprs rest

theorem caseItem_sound (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (hΛ : BoundEnvOK Γ Λ)
    (sel : F.Expr) (sel' : P.Expr) (hsel : SoundScalar o Γ sel sel') (hnum : NumTyped o Γ sel)
    (hnostr : Tr.isStr (Tr.tyOf Γ sel) = false) (it : F.CaseItem) (c' : P.Expr)
    (hwf : ∀ e ∈ itemExprs it, WFE Γ e) (h : transCaseItem Γ Λ consts sel sel' it = some c') :
    SoundScalar o Γ (F.caseItemCond sel it) c' := by
  cases it with
  | value v =>
    simp only [transCaseItem, hnostr, Bool.false_eq_true, ↓reduceIte, Option.map_eq_some_iff] at h
    obtain ⟨v', hv', rfl⟩ := h
    have hsv := trans_sound o Γ Λ consts hΛ v (hwf v (by simp [itemExprs])) v' hv'
    exact FNP_R0030_sound o Γ .eq sel v sel' v' hsel hsv hnum
  | range lo hi =>
    simp only [transCaseItem, hnostr, Bool.false_eq_true, ↓reduceIte] at h
    cases lo with
    | none =>
      cases hi with
      | none => cases h
      | some hiE =>
        simp only [Option.map_eq_some_iff] at h
        obtain ⟨h', hh', rfl⟩ := h
        have hsh := trans_sound o Γ Λ consts hΛ hiE (hwf hiE (by simp [itemExprs])) h' hh'
        simp only [F.caseItemCond]
        exact soundScalar_paren' (FNP_R0030_sound o Γ .le sel hiE sel' h' hsel hsh hnum)
    | some loE =>
      cases hi with
      | none =>
        simp only [Option.map_eq_some_iff] at h
        obtain ⟨l', hl', rfl⟩ := h
        have hsl := trans_sound o Γ Λ consts hΛ loE (hwf loE (by simp [itemExprs])) l' hl'
        simp only [F.caseItemCond]
        exact soundScalar_paren' (cmp_sound_right o Γ .le loE sel l' sel' hsl hsel hnum)
      | some hiE =>
        simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
        obtain ⟨l', hl', h', hh', rfl⟩ := h
        have hsl := trans_sound o Γ Λ consts hΛ loE (hwf loE (by simp [itemExprs])) l' hl'
        have hsh := trans_sound o Γ Λ consts hΛ hiE (hwf hiE (by simp [itemExprs])) h' hh'
        simp only [F.caseItemCond]
        exact (FNP_R0034_sound o Γ _ _ _ _ (cmp_sound_right o Γ .le loE sel l' sel' hsl hsel hnum)
          (FNP_R0030_sound o Γ .le sel hiE sel' h' hsel hsh hnum)).1

theorem caseCond_sound (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (hΛ : BoundEnvOK Γ Λ)
    (sel : F.Expr) (sel' : P.Expr) (hsel : SoundScalar o Γ sel sel') (hnum : NumTyped o Γ sel)
    (hnostr : Tr.isStr (Tr.tyOf Γ sel) = false) :
    ∀ (items : List F.CaseItem) (c' : P.Expr), (∀ e ∈ itemsExprs items, WFE Γ e) →
      transCaseCond Γ Λ consts sel sel' items = some c' → SoundScalar o Γ (F.caseCond sel items) c'
  | [], c', _, h => by
    simp only [transCaseCond, Option.some.injEq] at h; subst h
    intro σ τ _ _ v hv
    simp only [F.caseCond, F.eval, Option.some.injEq, F.Val.s.injEq] at hv; subst hv
    exact ⟨_, rfl, RelS.bool⟩
  | [it], c', hwf, h => by
    simp only [transCaseCond] at h
    simp only [F.caseCond]
    exact caseItem_sound o Γ Λ consts hΛ sel sel' hsel hnum hnostr it c'
      (fun e he => hwf e (by simpa [itemsExprs] using he)) h
  | it :: r :: rest, c', hwf, h => by
    simp only [transCaseCond, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨c, hc, cs, hcs, rfl⟩ := h
    simp only [F.caseCond]
    have h1 := caseItem_sound o Γ Λ consts hΛ sel sel' hsel hnum hnostr it c
      (fun e he => hwf e (by simp [itemsExprs, he])) hc
    have h2 := caseCond_sound o Γ Λ consts hΛ sel sel' hsel hnum hnostr (r :: rest) cs
      (fun e he => hwf e (List.mem_append_right _ he)) hcs
    exact or_sound o Γ _ _ _ _ h1 h2

/-- FNP-R0122: one `CASE` arm ↦ `if c': body' else rest'`. -/
theorem FNP_R0122_sound (o : Oracles) (Γ : F.Env) (Λ : Tr.BoundEnv) (consts : List String) (hΛ : BoundEnvOK Γ Λ)
    (sel : F.Expr) (sel' : P.Expr) (items : List F.CaseItem) (body rest : F.Stmt) (c' : P.Expr) (body' rest' : P.Stmt)
    (hsel : SoundScalar o Γ sel sel') (hnum : NumTyped o Γ sel) (hnostr : Tr.isStr (Tr.tyOf Γ sel) = false)
    (hwf : ∀ e ∈ itemsExprs items, WFE Γ e) (hc : transCaseCond Γ Λ consts sel sel' items = some c')
    (hb : StmtSound o Γ body body') (hr : StmtSound o Γ rest rest') :
    StmtSound o Γ (.caseArm sel items body rest) (.ite c' body' rest') := by
  intro σ τ hR hC σ' out h
  cases h with
  | caseArm hite =>
    exact FNP_R0120_sound o Γ _ c' body rest body' rest'
      (caseCond_sound o Γ Λ consts hΛ sel sel' hsel hnum hnostr items c' hwf hc) hb hr σ τ hR hC σ' out hite


/-! ### FNP-R0124: ASSOCIATE with a scalar selector (refuted, FNP-D0015)

`associate (t => x) … end associate` associates the name `t` with the *variable*
`x`: a write through `t` is a write to `x` (F2018 11.1.3.3).  The emitter binds
once, `t = x'`, and the body then writes the Python local `t`; the selector keeps
its value.  Probe `proofs/probes/associate`. -/

/-- The rule as the emitter spells it: `associate (n => x) body` ↦ `n = x'; body'`. -/
def FNP_R0124_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (n x : String) (body : F.Stmt) (body' : P.Stmt),
    StmtSound o Γ body body' →
    StmtSound o Γ (.associate n x body) (.seq (.assign n (.name x)) body')

def declR8 : F.Decl := ⟨.real .r8, [], 0⟩
def envAs : F.Env := fun s => if s = "x" then some declR8 else if s = "t" then some declR8 else none
def stateAs : F.State := fun s =>
  if s = "x" then some (.s (.real .r8 1)) else if s = "t" then some (.s (.real .r8 0)) else none
def pstateAs : P.State := fun s =>
  if s = "x" then some (.s (.float 1)) else if s = "t" then some (.s (.float 0)) else none

theorem relStateAs : RelState envAs stateAs pstateAs := by
  intro y d hy
  simp only [envAs] at hy
  split_ifs at hy with h1 h2 <;> cases hy
  · subst h1
    exact ⟨.s (.real .r8 1), .s (.float 1), by simp [stateAs], by simp [pstateAs], RelV.s RelS.realF⟩
  · subst h2
    exact ⟨.s (.real .r8 0), .s (.float 0), by simp [stateAs], by simp [pstateAs], RelV.s RelS.realF⟩

theorem conformsAs : Conforms noPrim.F envAs stateAs := by
  intro y d hy
  simp only [envAs] at hy
  split_ifs at hy with h1 h2 <;> cases hy
  · subst h1
    exact ⟨fun _ => ⟨.real .r8 1, by simp [stateAs], rfl⟩, fun h => absurd rfl h⟩
  · subst h2
    exact ⟨fun _ => ⟨.real .r8 0, by simp [stateAs], rfl⟩, fun h => absurd rfl h⟩

theorem staticBoundsAs : StaticBounds envAs := by
  intro y d hy
  simp only [envAs] at hy
  split_ifs at hy <;> cases hy <;> intro e he <;> simp [declR8] at he

/-- FNP-D0015: `associate (t => x); t = 2.0; end associate` with `real(8) :: x = 1.0`.
Fortran writes the selector (`x` becomes `2.0`); the translation writes the local
`t` and leaves `x` at `1.0`. -/
theorem FNP_R0124_refuted : ¬ FNP_R0124_obligation := by
  intro h
  -- the body `t = 2.0` and its translation, sound on their own
  have hbody : StmtSound noPrim envAs (.assign "t" .nil (.realLit .r8 ⟨2, 0⟩))
      (.assign "t" (.floatLit ⟨2, 0⟩)) := by
    refine FNP_R0111_sound_numeric noPrim envAs "t" _ _ declR8 (by simp [envAs]) rfl staticBoundsAs
      (realLit_sound' noPrim envAs .r8 ⟨2, 0⟩) ?_
    intro σ τ _ _ v hv
    simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv
    subst hv
    exact Or.inl ⟨rfl, by simp [declR8]⟩
  have hs := h noPrim envAs "t" "x" _ _ hbody
  -- the source: the body renamed, so `x` is written
  have hren : F.renameS "t" "x" (.assign "t" .nil (.realLit .r8 ⟨2, 0⟩))
      = .assign "x" .nil (.realLit .r8 ⟨2, 0⟩) := by simp [F.renameS, F.renameSubs, F.renameE]
  have hF : F.Exec noPrim.F envAs (.associate "t" "x" (.assign "t" .nil (.realLit .r8 ⟨2, 0⟩))) stateAs
      (Function.update stateAs "x" (some (.s (.real .r8 2)))) .normal := by
    refine F.Exec.associate ?_
    rw [hren]
    refine F.Exec.assign (v := .s (.real .r8 (Decimal.toReal ⟨2, 0⟩))) rfl ?_
    rw [F_assign_scalar noPrim.F envAs stateAs "x" declR8 (.real .r8 1) _ (by simp [envAs]) (by simp [stateAs])]
    simp [F.convertTo, declR8, Decimal.toReal]
  obtain ⟨τ', hex, hR', _⟩ := hs stateAs pstateAs relStateAs conformsAs _ _ hF
  -- the target: `t = x'` then `t = 2.0`; `x` is untouched
  have hτ : τ' "x" = pstateAs "x" := by
    cases hex with
    | seq_normal h1 h2 =>
      cases h1 with
      | assign _ =>
        cases h2 with
        | assign _ => simp [Function.update_of_ne]
    | seq_abrupt h1 hne => cases h1 with | assign _ => exact absurd rfl hne
  obtain ⟨v, w, hv, hw, hrel⟩ := hR' "x" declR8 (by simp [envAs])
  rw [hτ] at hw
  simp only [pstateAs, if_true, Option.some.injEq] at hw
  simp only [Function.update_self, Option.some.injEq] at hv
  subst hv; subst hw
  cases hrel with
  | s hs2 =>
    have := RelS.real_toR hs2
    simp only [P.toR, Option.some.injEq] at this
    norm_num at this

/-! ### FNP-R0121: FORALL (refuted)

`FORALL` is not a loop: every right-hand side and every subscript is evaluated
*before* any element is assigned (F2018 10.2.4), and an optional mask selects the
active index values.  The emitter renders one `for` per triplet and drops the
mask (`statements.py:930-956`), so both properties are lost: FNP-D0002
(evaluation order, probe `proofs/probes/forall`) and FNP-D0003 (dropped mask).

The obligation is stated *modulo the index variable* (`StmtSoundExcept`), because
`FORALL`'s index is local to the construct while Python's `for` leaves it at its
last value; refuting the weaker statement refutes the rule a fortiori. -/

def FNP_R0121_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (v : String) (lo hi : F.Expr) (mask : Option F.Expr)
    (target : String) (subs : F.Subs) (rhs : F.Expr) (lo' hi' : P.Expr) (body' : P.Stmt),
    SoundScalar o Γ lo lo' → SoundScalar o Γ hi hi' →
    StmtSound o Γ (.assign target subs rhs) body' →
    StmtSoundExcept o Γ v (.forall1 v lo hi none mask target subs rhs)
      (.forRange v lo' (.bin .add (.paren hi') (.intLit 1)) none body' none)

/-! A unit-origin INTEGER array `a(3) = [1, 2, 3]` and an INTEGER index `i`. -/

def declIA : F.Decl := ⟨.int .i4, [.intLit .i4 1], 0⟩
def declIS : F.Decl := ⟨.int .i4, [], 0⟩
def envFa : F.Env := fun s => if s = "a" then some declIA else if s = "i" then some declIS else none
def arr3 : FArr := { elemTy := .int .i4, dims := [(1, 3)], get := fun idx => .int .i4 (idx.headD 1) }
def parr3 : PArr := { shape := [3], get := fun idx => .int ((idx.headD 0 : ℕ) + 1) }
def stateFa : F.State := fun s =>
  if s = "a" then some (.arr arr3) else if s = "i" then some (.s (.int .i4 0)) else none
def pstateFa : P.State := fun s =>
  if s = "a" then some (.arr parr3) else if s = "i" then some (.s (.int 0)) else none

theorem relA3 : RelA arr3 parr3 := by
  refine ⟨rfl, ?_⟩
  intro idx hidx
  obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ 1 ≤ j ∧ j < 1 + (3 : ℤ) := by
    simp only [FArr.inBounds, FArr.inBoundsB, arr3] at hidx
    match idx, hidx with
    | [j], hidx =>
      simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
      exact ⟨j, rfl, hidx.1⟩
    | [], hidx => simp [FArr.inBoundsB] at hidx
    | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
  simp only [rebase, arr3, parr3, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.headD_cons]
  have he : (((j - 1).toNat : ℕ) : ℤ) + 1 = j := by omega
  rw [he]
  exact RelS.int

theorem relStateFa : RelState envFa stateFa pstateFa := by
  intro y d hy
  simp only [envFa] at hy
  split_ifs at hy with h1 h2 <;> cases hy
  · subst h1; exact ⟨.arr arr3, .arr parr3, by simp [stateFa], by simp [pstateFa], RelV.arr relA3⟩
  · subst h2; exact ⟨.s (.int .i4 0), .s (.int 0), by simp [stateFa], by simp [pstateFa], RelV.s RelS.int⟩

theorem conformsFa : Conforms noPrim.F envFa stateFa := by
  intro y d hy
  simp only [envFa] at hy
  split_ifs at hy with h1 h2 <;> cases hy
  · subst h1
    refine ⟨fun h => by simp [declIA] at h, fun _ => ⟨arr3, by simp [stateFa], rfl, rfl, ?_, fun _ _ => rfl⟩⟩
    intro k hk
    simp only [declIA] at hk
    match k, hk with
    | 0, _ => simp [declIA, F_eval_intLit, arr3, FArr.lbs]
  · subst h2; exact ⟨fun _ => ⟨.int .i4 0, by simp [stateFa], rfl⟩, fun h => absurd rfl h⟩

theorem staticBoundsFa : StaticBounds envFa := by
  intro y d hy
  simp only [envFa] at hy
  split_ifs at hy <;> cases hy <;> intro e he
  · simp only [declIA, List.mem_singleton] at he; exact ⟨.i4, 1, he⟩
  · simp [declIS] at he

theorem rank1Fa : Rank1Decl noPrim envFa "a" (.intLit .i4 1) :=
  rank1Decl_of_decl noPrim envFa "a" (.int .i4) (.intLit .i4 1) 0 (by simp [envFa, declIA])

theorem intTyped_var_i : IntTyped noPrim envFa (.var "i") := by
  intro σ τ _ hC v hv
  obtain ⟨sv, hσ, hty⟩ := conforms_scalar hC "i" declIS (by simp [envFa]) rfl
  simp only [F.eval, hσ, Option.some.injEq, F.Val.s.injEq] at hv
  subst hv
  cases sv with
  | int k n => exact ⟨k, n, rfl⟩
  | real _ _ => simp [SVal.ty, declIS] at hty
  | bool _ => simp [SVal.ty, declIS] at hty
  | str _ => simp [SVal.ty, declIS] at hty

theorem soundScalar_var_i : SoundScalar noPrim envFa (.var "i") (.name "i") :=
  (FNP_R0010_sound noPrim envFa "i" declIS (by simp [envFa])).toScalar

/-- `a(i) = 9` and its translation `a[i - 1] = 9`, sound on their own. -/
theorem bodyFa_sound : StmtSound noPrim envFa (.assign "a" (.cons (.scalar (.var "i")) .nil) (.intLit .i4 9))
    (.assignIndex "a" (.cons (.scalar (.bin .sub (.name "i") (.intLit 1))) .nil) (.intLit 9)) := by
  refine FNP_R0113_sound noPrim envFa "a" declIA .i4 (by simp [envFa]) rfl staticBoundsFa rank1Fa
    (.var "i") (.name "i") (.intLit .i4 9) (.intLit 9) soundScalar_var_i intTyped_var_i
    (FNP_R0001_sound _ _ _ _) ?_
  intro σ τ _ _ v hv
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv
  subst hv
  exact Or.inl ⟨rfl, by simp [declIA]⟩

/-- FNP-D0003: `forall (i = 1:1, .false.) a(i) = 9` assigns nothing in Fortran
(the mask selects no index value); the translation drops the mask and assigns. -/
theorem FNP_R0121_refuted_mask : ¬ FNP_R0121_obligation := by
  intro h
  have hs := h noPrim envFa "i" (.intLit .i4 1) (.intLit .i4 1) (some (.boolLit false)) "a"
    (.cons (.scalar (.var "i")) .nil) (.intLit .i4 9) (.intLit 1) (.intLit 1) _
    (FNP_R0001_sound _ _ _ _) (FNP_R0001_sound _ _ _ _) bodyFa_sound
  have hF : F.Exec noPrim.F envFa
      (.forall1 "i" (.intLit .i4 1) (.intLit .i4 1) none (some (.boolLit false)) "a"
        (.cons (.scalar (.var "i")) .nil) (.intLit .i4 9)) stateFa
      (Function.update stateFa "i" (stateFa "i")) .normal := by
    refine F.Exec.forall1 ?_
    simp [F.execForall, F.evalInt, F.eval, Fort.sectionCoords, Fort.tripCount, Exact.tquot, envFa]
  obtain ⟨τ', hex, hR', _⟩ := hs stateFa pstateFa relStateFa conformsFa _ _ hF
  cases hex with
  | forRange hl hh hst hs0 hfor =>
    rename_i l hval s
    simp only [P.eval, Option.some.injEq, P.Val.s.injEq, PVal.int.injEq] at hl
    subst hl
    have hh2 : (2 : ℤ) = hval := by
      rw [P_eval_bin, P_eval_paren] at hh
      simpa [P.eval, P.lift2, P.arith] using hh
    subst hh2
    simp only at hst
    subst hst
    have hrange : P.pyRange 1 2 1 = [1] := by
      simp [P.pyRange, Py.sliceLen]
    rw [hrange] at hfor
    cases hfor with
    | brk hbody => cases hbody
    | step hbody _ hrest =>
      cases hrest with
      | done_noelse =>
        cases hbody with
        | assignIndex hv hτ hidx hstore =>
          -- the array is `parr3`, the index `1 - 1 = 0`, the value `9`
          rw [P_evalIdxs_scalar1, P_eval_bin] at hidx
          simp [P.eval, P.lift2, P.arith] at hidx
          subst hidx
          rw [Function.update_of_ne (by decide)] at hτ
          simp only [pstateFa, if_true, Option.some.injEq, P.Val.arr.injEq] at hτ
          subst hτ
          simp only [P.eval, Option.some.injEq] at hv
          subst hv
          obtain ⟨q, hq, _, hq0, _⟩ := P_storeArr_at parr3 3 rfl 0 ⟨le_refl _, by norm_num⟩ (.int 9)
          rw [hq, Option.some.injEq] at hstore
          subst hstore
          -- the Fortran array is untouched, so the relation fails at `a(1)`
          obtain ⟨fv, pv, hfv, hpv, hrel⟩ := hR' "a" declIA (by simp [envFa]) (by decide)
          rw [Function.update_of_ne (by decide)] at hfv
          simp only [stateFa, if_true, Option.some.injEq] at hfv
          simp only [Function.update_self, Option.some.injEq] at hpv
          subst hfv; subst hpv
          cases hrel with
          | arr hA =>
            have hb : arr3.inBounds [1] := by simp [FArr.inBounds, FArr.inBoundsB, arr3]
            have hel := hA.elems [1] hb
            simp only [rebase, arr3, FArr.lbs, List.map_cons, List.map_nil, List.zipWith_cons_cons,
              List.zipWith_nil_right, List.headD_cons, sub_self, Int.toNat_zero] at hel
            simp only [Int.toNat_zero] at hq0
            rw [hq0] at hel
            have := RelS.int_eq hel
            simp only [PVal.int.injEq] at this
            omega

/-! ### FNP-R0160: ALLOCATE with a unit lower bound

`allocate (x(hi))` ↦ `x = np.zeros((hi,), dtype)` (`statements.py:971-1065`).
The Fortran model fills the new array with the zero of the declared type (the
values are undefined in Fortran, so a program that reads them carries no
obligation); NumPy's `np.zeros` does the same.

`np.zeros` raises on a negative extent, while Fortran's `(hi - lo + 1)` merely
clamps to a zero-size array, so the rule carries a non-negative-extent premise
that the emitter does not check. -/

/-- The `np.zeros` dtype the emitter picks from the declared type. -/
def zerosHelper : FTy → P.Helper
  | .int _ => .np_zeros_int
  | .bool => .np_zeros_bool
  | _ => .np_zeros_float

/-- The upper bound is non-negative: `np.zeros` is defined (NOT enforced). -/
def NonNegExtent (o : Oracles) (Γ : F.Env) (hi : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ k v, F.eval o.F σ hi = some (.s (.int k v)) → 0 ≤ v

/-- A numeric declared type: `F.zeroOf` is defined (LOGICAL and CHARACTER
allocation have no zero in the model). -/
def NumericTy (ty : FTy) : Prop := (∃ k, ty = .int k) ∨ (∃ k, ty = .real k)

/-- Storing a *fresh* array of the declared shape into a declared array name
preserves conformance (the previous binding is irrelevant: `ALLOCATE` replaces it). -/
theorem conforms_update_fresh {o : F.Oracle} {Γ : F.Env} {σ : F.State} (hC : Conforms o Γ σ)
    (hS : StaticBounds Γ) (x : String) (d : F.Decl) (hx : Γ x = some d)
    (hlbs : d.lbs = [.intLit .i4 1]) (n : ℕ) (z : SVal) (hz : z.ty = d.ty) :
    Conforms o Γ (Function.update σ x (some (.arr
      { elemTy := d.ty, dims := [(1, n)], get := fun _ => z }))) := by
  intro y d' hy
  by_cases hyx : y = x
  · subst hyx
    rw [hx] at hy; cases hy
    refine ⟨fun hnil => by rw [hlbs] at hnil; simp at hnil,
      fun _ => ⟨{ elemTy := d.ty, dims := [(1, n)], get := fun _ => z }, by simp, rfl, ?_, ?_, ?_⟩⟩
    · rw [hlbs]; rfl
    · intro i hi
      have hlen : d.lbs.length = 1 := by rw [hlbs]; rfl
      have hi0 : i = 0 := by omega
      subst hi0
      simp [hlbs, F_eval_intLit, FArr.lbs]
    · intro idx _; exact hz
  · obtain ⟨h1, h2⟩ := hC y d' hy
    refine ⟨fun hl => ?_, fun hl => ?_⟩
    · obtain ⟨sv, hs, ht⟩ := h1 hl
      exact ⟨sv, by simp [Function.update, hyx, hs], ht⟩
    · obtain ⟨a, ha, hty, hlen, hlbs', hwt⟩ := h2 hl
      refine ⟨a, by simp [Function.update, hyx, ha], hty, hlen, fun i hi => ?_, hwt⟩
      have := hlbs' i hi
      obtain ⟨kk, vv, he⟩ := hS y d' hy (d'.lbs[i]) (List.getElem_mem hi)
      rw [he, F_eval_intLit] at this ⊢
      exact this

/-- `np.zeros((n,), dtype=np.int32)` on a non-negative extent. -/
theorem helper_zeros_int (o : P.Oracle) (n : ℤ) (hn : 0 ≤ n) :
    P.helper o .np_zeros_int [.s (.int n)] = some (.arr { shape := [n.toNat], get := fun _ => .int 0 }) := by
  have e : P.helper o .np_zeros_int [.s (.int n)]
      = ((([.s (.int n)] : List P.Val).mapM fun (v : P.Val) => match v with
          | .s (.int m) => if 0 ≤ m then some m.toNat else none | _ => none).map
          fun ns => P.Val.arr { shape := ns, get := fun _ => .int 0 }) := rfl
  rw [e]
  simp [hn]

/-- `np.zeros((n,), dtype=np.float64)` on a non-negative extent. -/
theorem helper_zeros_float (o : P.Oracle) (n : ℤ) (hn : 0 ≤ n) :
    P.helper o .np_zeros_float [.s (.int n)] = some (.arr { shape := [n.toNat], get := fun _ => .float 0 }) := by
  have e : P.helper o .np_zeros_float [.s (.int n)]
      = ((([.s (.int n)] : List P.Val).mapM fun (v : P.Val) => match v with
          | .s (.int m) => if 0 ≤ m then some m.toNat else none | _ => none).map
          fun ns => P.Val.arr { shape := ns, get := fun _ => .float 0 }) := rfl
  rw [e]
  simp [hn]

theorem FNP_R0160_sound (o : Oracles) (Γ : F.Env) (x : String) (d : F.Decl) (hx : Γ x = some d)
    (hlbs : d.lbs = [.intLit .i4 1]) (hS : StaticBounds Γ) (hnum : NumericTy d.ty)
    (hi : F.Expr) (hi' : P.Expr) (hhi : SoundScalar o Γ hi hi') (hint : IntTyped o Γ hi)
    (hnn : NonNegExtent o Γ hi) :
    StmtSound o Γ (.allocate x [(none, hi)]) (.assign x (.call (zerosHelper d.ty) (.cons hi' .nil))) := by
  intro σ τ hR hC σ' out h
  cases h with
  | allocate hall =>
    cases hev : F.evalInt o.F σ hi with
    | none =>
      rw [show F.execAllocate o.F Γ σ x [(none, hi)] = none by simp [F.execAllocate, hx, hev]] at hall
      cases hall
    | some hv =>
      obtain ⟨kk, hkk⟩ : ∃ k, F.eval o.F σ hi = some (.s (.int k hv)) := by
        unfold F.evalInt at hev
        cases hfe : F.eval o.F σ hi with
        | none => rw [hfe] at hev; cases hev
        | some w =>
          cases w with
          | arr _ => rw [hfe] at hev; cases hev
          | s sv =>
            cases sv with
            | int k m =>
              rw [hfe] at hev
              simp only [Option.bind_some, Option.some.injEq] at hev
              subst hev
              exact ⟨k, rfl⟩
            | real _ _ => rw [hfe] at hev; cases hev
            | bool _ => rw [hfe] at hev; cases hev
            | str _ => rw [hfe] at hev; cases hev
      have hnn' := hnn σ τ hR hC kk hv hkk
      obtain ⟨w, hw, hrw⟩ := hhi σ τ hR hC _ hkk
      rw [RelS.int_eq hrw] at hw
      rcases hnum with ⟨k, hk⟩ | ⟨k, hk⟩
      · -- INTEGER: `np.zeros((hi,), dtype=np.int32)`
        have hallv : F.execAllocate o.F Γ σ x [(none, hi)] = some (Function.update σ x (some (.arr
            { elemTy := .int k, dims := [(1, hv.toNat)], get := fun _ => .int k 0 }))) := by
          simp [F.execAllocate, hx, hev, hlbs, hk, F.zeroOf]
        rw [hallv, Option.some.injEq] at hall
        subst hall
        have hpy : P.eval o.P τ (.call (zerosHelper d.ty) (.cons hi' .nil))
            = some (.arr { shape := [hv.toNat], get := fun _ => .int 0 }) := by
          show P.eval o.P τ (Tr.fold1 (zerosHelper d.ty) hi') = _
          rw [P_eval_call1, hw, Option.bind_some, hk]
          exact helper_zeros_int o.P hv hnn'
        refine ⟨_, P.Exec.assign hpy, ?_, ?_⟩
        · refine relState_update hR x (RelV.arr ⟨?_, ?_⟩)
          · simp only [FArr.extents, List.map_cons, List.map_nil]
          · intro idx _; exact RelS.int
        · have := conforms_update_fresh hC hS x d hx hlbs hv.toNat (.int k 0) (by rw [hk]; rfl)
          rwa [hk] at this
      · -- REAL: `np.zeros((hi,), dtype=np.float64)`
        have hallv : F.execAllocate o.F Γ σ x [(none, hi)] = some (Function.update σ x (some (.arr
            { elemTy := .real k, dims := [(1, hv.toNat)], get := fun _ => .real k 0 }))) := by
          simp [F.execAllocate, hx, hev, hlbs, hk, F.zeroOf]
        rw [hallv, Option.some.injEq] at hall
        subst hall
        have hpy : P.eval o.P τ (.call (zerosHelper d.ty) (.cons hi' .nil))
            = some (.arr { shape := [hv.toNat], get := fun _ => .float 0 }) := by
          show P.eval o.P τ (Tr.fold1 (zerosHelper d.ty) hi') = _
          rw [P_eval_call1, hw, Option.bind_some, hk]
          exact helper_zeros_float o.P hv hnn'
        refine ⟨_, P.Exec.assign hpy, ?_, ?_⟩
        · refine relState_update hR x (RelV.arr ⟨?_, ?_⟩)
          · simp only [FArr.extents, List.map_cons, List.map_nil]
          · intro idx _; exact RelS.realF
        · have := conforms_update_fresh hC hS x d hx hlbs hv.toNat (.real k 0) (by rw [hk]; rfl)
          rwa [hk] at this

/-! ### FNP-R0137: WHERE with whole-array rank-1 targets

`where (m) a = e` ↦ `_wm = m'; a[_wm] = e'` (`statements.py:858-887`).

Two model facts are in play.  (1) On a rank-1 array NumPy's boolean-mask
enumeration (`P.maskCoords`, row-major) and Fortran's array element order
coincide — they do *not* on rank ≥ 2, which is defect FNP-D0041.  (2) The
Fortran model `F.maskedAssign` evaluates the whole right-hand side before
masking, while Fortran evaluates it only where the mask is true; that is the
definedness gap FNP-D0038, and it is why the theorem below carries a
definedness premise on every element rather than only the masked ones. -/

theorem cCoords_rank1 (n k : ℕ) : P.cCoords [n] k = [k % n] := by
  simp [P.cCoords, CM.coords]

theorem rowMajor_rank1 (n : ℕ) : P.rowMajor [n] = (List.range n).map (fun k => [k % n]) := by
  simp [P.rowMajor, cCoords_rank1]

/-- On a rank-1 array the row-major mask enumeration is exactly the set of
positions where the mask is true. -/
theorem maskCoords_rank1_mem (pm : PArr) (n k : ℕ) (hs : pm.shape = [n]) (hk : k < n) :
    [k] ∈ P.maskCoords pm ↔ pm.get [k] = .bool true := by
  unfold P.maskCoords
  rw [hs, rowMajor_rank1]
  rw [List.mem_filter]
  constructor
  · rintro ⟨_, h2⟩
    revert h2
    cases hg : pm.get [k] with
    | bool b => cases b <;> simp
    | int _ => simp
    | float _ => simp
    | str _ => simp
    | none => simp
  · intro hg
    refine ⟨?_, ?_⟩
    · rw [List.mem_map]
      exact ⟨k, List.mem_range.mpr hk, by rw [Nat.mod_eq_of_lt hk]⟩
    · rw [hg]

theorem execWhere_single (o : F.Oracle) (Γ : F.Env) (σ : F.State) (mask : F.Expr) (a : String) (e : F.Expr)
    (mv : FArr) (hm : F.eval o σ mask = some (.arr mv)) :
    F.execWhere o Γ σ mask [(a, e)] [] = F.maskedAssign o Γ σ mv true a e := by
  simp [F.execWhere, hm]

/-- The Fortran mask test inside `F.maskedAssign` (with `sense = true`). -/
noncomputable def maskPick (m av : FArr) (idx : List ℤ) : Bool :=
  match m.get (List.zipWith (fun (lm : ℤ) (i : ℤ) => lm + i) m.lbs
      (F.maskedAssign.rebaseZ av.lbs idx)) with
  | .bool b => b
  | _ => false

theorem maskedAssign_scalar (o : F.Oracle) (Γ : F.Env) (σ : F.State) (m : FArr) (a : String) (e : F.Expr)
    (d : F.Decl) (av : FArr) (sv cv : SVal)
    (hd : Γ a = some d) (hσ : σ a = some (.arr av)) (hext : av.extents = m.extents)
    (he : F.eval o σ e = some (.s sv)) (hcv : F.convertTo d sv = some cv) :
    F.maskedAssign o Γ σ m true a e = some (Function.update σ a (some (.arr
      { av with get := fun idx => if maskPick m av idx then cv else av.get idx }))) := by
  simp [F.maskedAssign, maskPick, hd, hσ, hext, he, hcv]
  congr 1

theorem storeMask_scalar (pa pm : PArr) (w : PVal) (hs : pa.shape = pm.shape) :
    P.storeMask pa pm (.s w)
      = some { pa with get := fun i => if (P.maskCoords pm).contains i then w else pa.get i } := by
  simp [P.storeMask, hs]

/-- Updating an *undeclared* name keeps the state relation (the mask temporary
`_wm` is not a Fortran variable). -/
theorem relState_update_undeclared {Γ : F.Env} {σ : F.State} {τ : P.State} (h : RelState Γ σ τ)
    (y : String) (hy : Γ y = none) (w : P.Val) : RelState Γ σ (Function.update τ y (some w)) := by
  intro z d hz
  have hzy : z ≠ y := by intro hh; rw [hh, hy] at hz; cases hz
  obtain ⟨v, w', hv, hw', hr⟩ := h z d hz
  exact ⟨v, w', hv, by simp [Function.update, hzy, hw'], hr⟩

/-- The mask conforms to the target: same shape, LOGICAL elements. -/
def MaskConforms (o : Oracles) (Γ : F.Env) (a : String) (mask : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ mv av, F.eval o.F σ mask = some (.arr mv) →
    σ a = some (.arr av) → mv.dims = av.dims ∧ ∀ idx, mv.inBounds idx → ∃ b, mv.get idx = .bool b

/-- The right-hand side is scalar-valued. -/
def ScalarRhs (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some v → ∃ sv, v = .s sv

/-- FNP-R0137 with one whole-array assignment and a scalar right-hand side, on a
unit-origin rank-1 target. -/
theorem FNP_R0137_sound_scalar_rhs (o : Oracles) (Γ : F.Env) (a : String) (d : F.Decl)
    (hx : Γ a = some d) (hlbs : d.lbs = [.intLit .i4 1]) (hS : StaticBounds Γ)
    (hwm : Γ "_wm" = none) (hane : a ≠ "_wm") (hdecl : Rank1Decl o Γ a (.intLit .i4 1))
    (mask : F.Expr) (m' : P.Expr) (e : F.Expr) (e' : P.Expr)
    (hmask : SoundPair o Γ mask m') (hmc : MaskConforms o Γ a mask)
    (he : SoundScalar o Γ e e') (hcompat : ScalarCompat o Γ d e) (hsc : ScalarRhs o Γ e) :
    StmtSound o Γ (.whereBlock mask [(a, e)] [])
      (.seq (.assign "_wm" m') (.seq (.seq (.assignMask a (.name "_wm") e') .pass) .pass)) := by
  intro σ τ hR hC σ' out h
  cases h with
  | whereBlock hw =>
    obtain ⟨av, pa, l, n, hσ, hτ, hrel, hd, hl⟩ := hdecl σ τ hR hC
    rw [F_eval_intLit] at hl
    simp only [Option.some.injEq, F.Val.s.injEq, SVal.int.injEq] at hl
    obtain ⟨_, rfl⟩ := hl
    -- the mask value
    cases hm : F.eval o.F σ mask with
    | none => rw [show F.execWhere o.F Γ σ mask [(a, e)] [] = none by simp [F.execWhere, hm]] at hw; cases hw
    | some mval =>
      cases mval with
      | s _ => rw [show F.execWhere o.F Γ σ mask [(a, e)] [] = none by simp [F.execWhere, hm]] at hw; cases hw
      | arr mv =>
        obtain ⟨hmdims, hmbool⟩ := hmc σ τ hR hC mv av hm hσ
        -- the right-hand side value
        cases hev : F.eval o.F σ e with
        | none =>
          rw [execWhere_single o.F Γ σ mask a e mv hm] at hw
          rw [show F.maskedAssign o.F Γ σ mv true a e = none by
            simp [F.maskedAssign, hx, hσ, hmdims, FArr.extents, hev]] at hw
          cases hw
        | some ev =>
          obtain ⟨sv, rfl⟩ := hsc σ τ hR hC _ hev
          obtain ⟨w, hw', hrw⟩ := he σ τ hR hC sv hev
          obtain ⟨cv, hcv, hrelc, hcvty⟩ := convertTo_compat_rel d sv w hrw (hcompat σ τ hR hC sv hev)
          rw [execWhere_single o.F Γ σ mask a e mv hm,
            maskedAssign_scalar o.F Γ σ mv a e d av sv cv hx hσ (by simp [FArr.extents, hmdims]) hev hcv,
            Option.some.injEq] at hw
          subst hw
          -- the Python side
          obtain ⟨pmval, hpm, hrpm⟩ := hmask σ τ hR hC _ hm
          cases hrpm with
          | arr hAm =>
            rename_i pm
            have hpmshape : pm.shape = [n] := by rw [hAm.shape]; simp [FArr.extents, hmdims, hd]
            have hpashape : pa.shape = [n] := RelA_dims1 hrel 1 n hd
            -- the state after `_wm = mask'`
            set τ₁ := Function.update τ "_wm" (some (P.Val.arr pm)) with hτ₁
            have hR₁ : RelState Γ σ τ₁ := relState_update_undeclared hR "_wm" hwm _
            obtain ⟨w₂, hw₂, hrw₂⟩ := he σ τ₁ hR₁ hC sv hev
            obtain ⟨cv₂, hcv₂, hrelc₂, _⟩ := convertTo_compat_rel d sv w₂ hrw₂ (hcompat σ τ hR hC sv hev)
            rw [hcv, Option.some.injEq] at hcv₂
            subst hcv₂
            have hτ₁a : τ₁ a = some (.arr pa) := by rw [hτ₁, Function.update_of_ne hane]; exact hτ
            have hτ₁m : P.eval o.P τ₁ (.name "_wm") = some (.arr pm) := by
              simp only [P.eval, hτ₁, Function.update_self]
            obtain ⟨pa', hstore⟩ : ∃ pa', P.storeMask pa pm (.s w₂) = some pa' :=
              ⟨_, storeMask_scalar pa pm w₂ (by rw [hpashape, hpmshape])⟩
            refine ⟨Function.update τ₁ a (some (.arr pa')), ?_, ?_, ?_⟩
            · exact P.Exec.seq_normal (P.Exec.assign hpm)
                (P.Exec.seq_normal (P.Exec.seq_normal (P.Exec.assignMask hw₂ hτ₁a hτ₁m hstore) P.Exec.pass)
                  P.Exec.pass)
            · -- the relation
              rw [storeMask_scalar pa pm w₂ (by rw [hpashape, hpmshape]), Option.some.injEq] at hstore
              subst hstore
              refine relState_update hR₁ a (RelV.arr ⟨?_, ?_⟩)
              · rw [hpashape]; simp [FArr.extents, hd]
              · intro idx hidx
                obtain ⟨j, rfl, hj⟩ : ∃ j, idx = [j] ∧ 1 ≤ j ∧ j < 1 + (n : ℤ) := by
                  simp only [FArr.inBounds, FArr.inBoundsB, hd] at hidx
                  match idx, hidx with
                  | [j], hidx =>
                    simp only [FArr.inBoundsB, Bool.and_eq_true, decide_eq_true_eq] at hidx
                    exact ⟨j, rfl, hidx.1⟩
                  | [], hidx => simp [FArr.inBoundsB] at hidx
                  | _ :: _ :: _, hidx => simp [FArr.inBoundsB] at hidx
                simp only [rebase, FArr.lbs, hd, List.map_cons, List.map_nil, List.zipWith_cons_cons,
                  List.zipWith_nil_right]
                -- the two mask tests agree
                have hmlbs : mv.lbs = [1] := by rw [FArr.lbs, hmdims, hd]; rfl
                have hpick : maskPick mv av [j] = true ↔ [(j - 1).toNat] ∈ P.maskCoords pm := by
                  have hidx' : List.zipWith (fun (lm : ℤ) (i : ℤ) => lm + i) mv.lbs
                      (F.maskedAssign.rebaseZ av.lbs [j]) = [j] := by
                    rw [hmlbs]
                    simp [F.maskedAssign.rebaseZ, FArr.lbs, hd]
                  have hb : mv.inBounds [j] := by
                    simp [FArr.inBounds, FArr.inBoundsB, hmdims, hd]; omega
                  obtain ⟨b, hbv⟩ := hmbool [j] hb
                  have hrel_m := hAm.elems [j] hb
                  rw [hbv] at hrel_m
                  simp only [rebase, FArr.lbs, hmdims, hd, List.map_cons, List.map_nil,
                    List.zipWith_cons_cons, List.zipWith_nil_right] at hrel_m
                  rw [maskCoords_rank1_mem pm n (j - 1).toNat hpmshape (by omega),
                    RelS.bool_eq hrel_m]
                  simp only [maskPick, hidx', hbv]
                  cases b <;> simp
                by_cases hsel : maskPick mv av [j] = true
                · rw [if_pos hsel, if_pos (List.contains_iff_mem.mpr (hpick.mp hsel))]
                  exact hrelc₂
                · rw [if_neg hsel, if_neg (fun hc => hsel (hpick.mpr (List.contains_iff_mem.mp hc)))]
                  exact RelA_get1 hrel 1 n hd j hj
            · refine conforms_update_array hC hS a d hx av _ hσ rfl rfl ?_
              refine wellTyped_rank1_of _ av 1 n hd rfl rfl (fun j hj => ?_)
              by_cases hsel : maskPick mv av [j] = true
              · simp only [hsel, if_true]
                obtain ⟨av₀, hσ₀, hty_av, _, _, _⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
                rw [hσ] at hσ₀; cases hσ₀
                rw [hcvty, hty_av]
              · simp only [hsel, Bool.false_eq_true, if_false]
                obtain ⟨av₀, hσ₀, _, _, _, hwt⟩ := conforms_array hC a d hx (by rw [hlbs]; simp)
                rw [hσ] at hσ₀; cases hσ₀
                exact hwt [j] (inBounds_rank1 av 1 n hd j hj)

end Recast.Rules
