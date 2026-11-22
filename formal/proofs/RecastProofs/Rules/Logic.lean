import RecastProofs.Rules.Arith

/-!
# Rules FNP-R0030 .. FNP-R0037: comparisons and logical operators (scalar cases)
-/

namespace Recast.Rules

open Recast Rel

theorem F_eval_cmp (o : F.Oracle) (σ : F.State) (op : F.CmpOp) (a b : F.Expr) :
    F.eval o σ (.cmp op a b) =
      (F.eval o σ a).bind fun va => (F.eval o σ b).bind fun vb => F.lift2 (F.compare op) va vb := by
  simp only [F.eval]; rfl

theorem P_eval_cmp (o : P.Oracle) (τ : P.State) (op : P.CmpOp) (a b : P.Expr) :
    P.eval o τ (.cmp op a b) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => P.lift2 (P.compare op) va vb := by
  simp only [P.eval]; rfl

theorem F_eval_logic (o : F.Oracle) (σ : F.State) (op : F.LogOp) (a b : F.Expr) :
    F.eval o σ (.logic op a b) =
      (F.eval o σ a).bind fun va => (F.eval o σ b).bind fun vb => F.lift2 (F.logic op) va vb := by
  simp only [F.eval]; rfl

theorem F_eval_not (o : F.Oracle) (σ : F.State) (e : F.Expr) :
    F.eval o σ (.not e) = (F.eval o σ e).bind (F.lift1 fun v => match v with
        | .bool b => some (.bool (!b))
        | _ => none) := rfl

def toPCmp : F.CmpOp → P.CmpOp
  | .eq => .eq | .ne => .ne | .lt => .lt | .le => .le | .gt => .gt | .ge => .ge

/-! ### FNP-R0030: numeric comparison -/

/-- Python comparison of two numbers denoting reals. -/
theorem P_compare_toR {w₁ w₂ : PVal} {a b : ℝ} (h1 : P.toR w₁ = some a) (h2 : P.toR w₂ = some b)
    (op : P.CmpOp) : P.compare op w₁ w₂ = some (.bool (P.cmpR op a b)) := by
  rcases P_toR_cases h1 with ⟨n, rfl, rfl⟩ | rfl <;> rcases P_toR_cases h2 with ⟨m, rfl, rfl⟩ | rfl <;>
    simp [P.compare, P.toR]

theorem cmpR_toP (op : F.CmpOp) (a b : ℝ) : F.cmpR op a b = P.cmpR (toPCmp op) a b := by
  cases op <;> rfl

theorem compare_numeric_rel {op : F.CmpOp} {a b : SVal} {a' b' : PVal}
    (ha : RelS a a') (hb : RelS b b') (hnum : (∃ k n, a = .int k n) ∨ (∃ k r, a = .real k r))
    {c : SVal} (hc : F.compare op a b = some c) :
    ∃ c', P.compare (toPCmp op) a' b' = some c' ∧ RelS c c' := by
  have key : ∀ (x y : ℝ), P.toR a' = some x → P.toR b' = some y → c = .bool (F.cmpR op x y) →
      ∃ c', P.compare (toPCmp op) a' b' = some c' ∧ RelS c c' := by
    intro x y hx hy hcc
    subst hcc
    exact ⟨_, P_compare_toR hx hy _, by rw [cmpR_toP]; exact RelS.bool⟩
  cases a with
  | int k₁ x =>
    obtain rfl := RelS.int_eq ha
    cases b with
    | int k₂ y =>
      obtain rfl := RelS.int_eq hb
      simp only [F.compare, Option.some.injEq] at hc
      exact key _ _ (P_toR_int x) (P_toR_int y) hc.symm
    | real k₂ y =>
      simp only [F.compare, Option.some.injEq] at hc
      exact key _ _ (P_toR_int x) (RelS.real_toR hb) hc.symm
    | bool _ => simp [F.compare] at hc
    | str _ => simp [F.compare] at hc
  | real k₁ x =>
    cases b with
    | int k₂ y =>
      obtain rfl := RelS.int_eq hb
      simp only [F.compare, Option.some.injEq] at hc
      exact key _ _ (RelS.real_toR ha) (P_toR_int y) hc.symm
    | real k₂ y =>
      simp only [F.compare, Option.some.injEq] at hc
      exact key _ _ (RelS.real_toR ha) (RelS.real_toR hb) hc.symm
    | bool _ => simp [F.compare] at hc
    | str _ => simp [F.compare] at hc
  | bool _ => exfalso; rcases hnum with ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> cases h
  | str _ => exfalso; rcases hnum with ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> cases h

def FNP_R0030_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (op : F.CmpOp) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → (IntTyped o Γ a ∨ RealTyped o Γ a) →
    SoundScalar o Γ (.cmp op a b) (.paren (.cmp (toPCmp op) a' b'))

theorem FNP_R0030_sound : FNP_R0030_obligation := by
  intro o Γ op a b a' b' ha hb hnum σ τ hR hC v hv
  rw [F_eval_cmp] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb
  have hn : (∃ k n, x = .int k n) ∨ (∃ k r, x = .real k r) := by
    rcases hnum with h | h
    · exact Or.inl (h σ τ hR hC x hva)
    · exact Or.inr (h σ τ hR hC x hva)
  obtain ⟨c', hc', hrc⟩ := compare_numeric_rel hrx hry hn hxy
  refine ⟨c', ?_, hrc⟩
  rw [P_eval_paren, P_eval_cmp, hx', hy']
  simp [P.lift2, hc']

/-! ### FNP-R0031: character equality with blank padding -/

theorem takeWhile_all (p : Char → Bool) : ∀ l : List Char, ∀ x ∈ l.takeWhile p, p x = true
  | [], _, h => by simp at h
  | a :: l, x, h => by
    by_cases ha : p a = true
    · simp only [List.takeWhile_cons, ha, if_true, List.mem_cons] at h
      rcases h with rfl | h
      · exact ha
      · exact takeWhile_all p l x h
    · simp [List.takeWhile_cons, ha] at h

/-- Every list is its trailing-blank strip followed by blanks. -/
theorem rstrip_decomp (s : List Char) :
    s = P.rstripBlanks s ++ List.replicate (s.length - (P.rstripBlanks s).length) ' ' := by
  unfold P.rstripBlanks
  have hsplit := List.takeWhile_append_dropWhile (p := fun c => decide (c = ' ')) (l := s.reverse)
  have hrep : (s.reverse.takeWhile fun c => decide (c = ' ')) =
      List.replicate (s.reverse.takeWhile fun c => decide (c = ' ')).length ' ' := by
    rw [List.eq_replicate_iff]
    refine ⟨rfl, fun b hb => ?_⟩
    have := takeWhile_all (fun c => decide (c = ' ')) s.reverse b hb
    simpa using this
  have hlen : s.length = (s.reverse.dropWhile fun c => decide (c = ' ')).length +
      (s.reverse.takeWhile fun c => decide (c = ' ')).length := by
    have := congrArg List.length hsplit
    simp only [List.length_append, List.length_reverse] at this
    omega
  conv_lhs => rw [← List.reverse_reverse s, ← hsplit]
  rw [List.reverse_append, List.length_reverse]
  congr 1
  rw [hrep, List.reverse_replicate]
  congr 1
  omega

/-- Appending blanks does not change the trailing-blank strip. -/
theorem rstrip_append_blanks (s : List Char) (k : ℕ) :
    P.rstripBlanks (s ++ List.replicate k ' ') = P.rstripBlanks s := by
  unfold P.rstripBlanks
  rw [List.reverse_append, List.reverse_replicate, List.dropWhile_append, List.dropWhile_replicate]
  simp

theorem padTo_eq (n : ℕ) (s : List Char) : F.padTo n s = s ++ List.replicate (n - s.length) ' ' := rfl

/-- Blank-padding the shorter operand is the same as stripping trailing blanks from both. -/
theorem strEq_iff_rstrip (a b : List Char) :
    F.strEq a b = true ↔ P.rstripBlanks a = P.rstripBlanks b := by
  unfold F.strEq
  simp only [decide_eq_true_eq]
  constructor
  · intro h
    have := congrArg P.rstripBlanks h
    rwa [padTo_eq, padTo_eq, rstrip_append_blanks, rstrip_append_blanks] at this
  · intro h
    obtain ⟨r, hr⟩ : ∃ r, P.rstripBlanks a = r := ⟨_, rfl⟩
    have ha' := rstrip_decomp a
    have hb' := rstrip_decomp b
    rw [hr] at ha'
    rw [← h, hr] at hb'
    have hla : r.length ≤ a.length := by
      have := congrArg List.length ha'; simp only [List.length_append, List.length_replicate] at this; omega
    have hlb : r.length ≤ b.length := by
      have := congrArg List.length hb'; simp only [List.length_append, List.length_replicate] at this; omega
    rw [padTo_eq, padTo_eq]
    generalize hn : max a.length b.length = n
    have hna : a.length ≤ n := by omega
    have hnb : b.length ≤ n := by omega
    generalize hla' : a.length = la at *
    generalize hlb' : b.length = lb at *
    rw [ha', hb']
    rw [List.append_assoc, List.append_assoc, ← List.replicate_add, ← List.replicate_add]
    congr 2
    omega

theorem strEq_rel (a b : List Char) :
    P.rstripBlanks a = P.rstripBlanks b ↔ F.strEq a b = true := (strEq_iff_rstrip a b).symm

def FNP_R0031_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    (∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ a = some (.s v) → ∃ s, v = .str s) →
    SoundScalar o Γ (.cmp .eq a b) (.paren (Tr.call2 .fstr_eq a' b'))

theorem FNP_R0031_sound : FNP_R0031_obligation := by
  intro o Γ a b a' b' ha hb hstr σ τ hR hC v hv
  rw [F_eval_cmp] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hlift⟩ := hv
  obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift
  obtain ⟨s, rfl⟩ := hstr σ τ hR hC x hva
  obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
  obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb
  cases hrx
  cases hry with
  | str =>
    rename_i t
    simp only [F.compare, Option.some.injEq] at hxy
    subst hxy
    refine ⟨.bool (F.strEq s t), ?_, RelS.bool⟩
    rw [P_eval_paren, P_eval_call2, hx', hy']
    simp only [Option.bind_some]
    have hdec : decide (P.rstripBlanks s = P.rstripBlanks t) = F.strEq s t := by
      rw [Bool.eq_iff_iff, decide_eq_true_eq]; exact strEq_rel s t
    have e : P.helper o.P .fstr_eq [.s (.str s), .s (.str t)] =
        some (.s (.bool (decide (P.rstripBlanks s = P.rstripBlanks t)))) := rfl
    rw [e, hdec]
  | int => simp [F.compare] at hxy
  | real => simp [F.compare] at hxy
  | bool => simp [F.compare] at hxy

/-! ### FNP-R0034: scalar `.AND.` / `.OR.` -/

def logicOp : F.LogOp → Bool → Bool → Bool
  | .and, p, q => p && q
  | .or, p, q => p || q
  | .eqv, p, q => p == q
  | .neqv, p, q => p != q

theorem logic_rel {op : F.LogOp} {a b : SVal} {a' b' : PVal} (ha : RelS a a') (hb : RelS b b')
    {c : SVal} (hc : F.logic op a b = some c) :
    ∃ p q : Bool, a = .bool p ∧ b = .bool q ∧ a' = .bool p ∧ b' = .bool q ∧ c = .bool (logicOp op p q) := by
  cases ha <;> cases hb <;> cases op <;> simp [F.logic] at hc <;> subst hc <;>
    exact ⟨_, _, rfl, rfl, rfl, rfl, rfl⟩

def FNP_R0034_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.logic .and a b) (.paren (.and a' b')) ∧
    SoundScalar o Γ (.logic .or a b) (.paren (.or a' b'))

theorem FNP_R0034_sound : FNP_R0034_obligation := by
  intro o Γ a b a' b' ha hb
  constructor <;> intro σ τ hR hC v hv <;> rw [F_eval_logic] at hv <;>
    simp only [Option.bind_eq_some_iff] at hv <;>
    obtain ⟨va, hva, vb, hvb, hlift⟩ := hv <;>
    obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift <;>
    obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva <;>
    obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb <;>
    obtain ⟨p, q, rfl, rfl, rfl, rfl, rfl⟩ := logic_rel hrx hry hxy
  · refine ⟨.bool (logicOp .and p q), ?_, RelS.bool⟩
    rw [P_eval_paren]
    cases p <;> simp [P.eval, hx', hy', logicOp]
  · refine ⟨.bool (logicOp .or p q), ?_, RelS.bool⟩
    rw [P_eval_paren]
    cases p <;> simp [P.eval, hx', hy', logicOp]

/-! ### FNP-R0036: `.EQV.` / `.NEQV.` -/

def FNP_R0036_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.logic .eqv a b) (.paren (.cmp .eq a' b')) ∧
    SoundScalar o Γ (.logic .neqv a b) (.paren (.cmp .ne a' b'))

theorem FNP_R0036_sound : FNP_R0036_obligation := by
  intro o Γ a b a' b' ha hb
  constructor <;> intro σ τ hR hC v hv <;> rw [F_eval_logic] at hv <;>
    simp only [Option.bind_eq_some_iff] at hv <;>
    obtain ⟨va, hva, vb, hvb, hlift⟩ := hv <;>
    obtain ⟨x, y, rfl, rfl, hxy⟩ := F_lift2_eq_scalar hlift <;>
    obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC x hva <;>
    obtain ⟨y', hy', hry⟩ := hb σ τ hR hC y hvb <;>
    obtain ⟨p, q, rfl, rfl, rfl, rfl, rfl⟩ := logic_rel hrx hry hxy
  · refine ⟨.bool (logicOp .eqv p q), ?_, RelS.bool⟩
    rw [P_eval_paren, P_eval_cmp, hx', hy']
    simp [P.lift2, P.compare, logicOp]
  · refine ⟨.bool (logicOp .neqv p q), ?_, RelS.bool⟩
    rw [P_eval_paren, P_eval_cmp, hx', hy']
    simp [P.lift2, P.compare, logicOp]

/-! ### FNP-R0037: scalar `.NOT.` -/

def FNP_R0037_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (e : F.Expr) (e' : P.Expr),
    SoundScalar o Γ e e' → SoundScalar o Γ (.not e) (.paren (.not e'))

theorem FNP_R0037_sound : FNP_R0037_obligation := by
  intro o Γ e e' he σ τ hR hC v hv
  rw [F_eval_not] at hv
  simp only [Option.bind_eq_some_iff] at hv
  obtain ⟨u, hu, hlift⟩ := hv
  obtain ⟨a, rfl, hfa⟩ := F_lift1_eq_scalar hlift
  obtain ⟨w, hw, hrel⟩ := he σ τ hR hC a hu
  cases hrel with
  | bool =>
    simp only [Option.some.injEq] at hfa; subst hfa
    refine ⟨.bool (!_), ?_, RelS.bool⟩
    rw [P_eval_paren]
    simp [P.eval, hw]
  | int => simp at hfa
  | real => simp at hfa
  | str => simp at hfa

end Recast.Rules
