import RecastProofs.Rules.Obligations

/-!
# Character rules: FNP-R0028 (`//`), FNP-R0085 (TRIM, ADJUSTL, INDEX, SCAN,
CHAR, ICHAR), substrings, and FNP-R0008 (BOZ literals)

The character carrier is `List Char` on both sides and `RelS.str` is equality,
so every character rule is a statement that the emitted Python spelling computes
the same list.  The one place where the two languages genuinely differ is the
*edge convention* of a substring: Fortran's `s(i:j)` is empty whenever `j < i`
(F2018 9.4.1), CPython's `s[i-1:j]` wraps a negative `j` to `len + j`
(`Py.adjust`), the same divergence as FNP-D0037 for array sections.

Stored CHARACTER values may already be wrong by FNP-D0021 (the declared length
is not padded on assignment); the rules below are about the operations on the
values the two states hold, and are sound on *related* operands.
-/

namespace Recast.Rules

open Recast Rel Exact

/-- Every value the expression takes (in related states) is a CHARACTER string. -/
def StrTyped (o : Oracles) (Γ : F.Env) (e : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ v, F.eval o.F σ e = some (.s v) → ∃ s, v = .str s

theorem strLit_strTyped (o : Oracles) (Γ : F.Env) (s : List Char) : StrTyped o Γ (.strLit s) := by
  intro σ τ _ _ v hv
  simp only [F.eval, Option.some.injEq, F.Val.s.injEq] at hv
  exact ⟨s, hv.symm⟩

/-- Three-argument Python call unfolding (`str_slice`, `_f_mvbits`, ...). -/
theorem P_eval_call3 (o : P.Oracle) (τ : P.State) (h : P.Helper) (a b c : P.Expr) :
    P.eval o τ (.call h (.cons a (.cons b (.cons c .nil)))) =
      (P.eval o τ a).bind fun va => (P.eval o τ b).bind fun vb => (P.eval o τ c).bind fun vc =>
        P.helper o h [va, vb, vc] := by
  cases h1 : P.eval o τ a <;> cases h2 : P.eval o τ b <;> cases h3 : P.eval o τ c <;>
    simp only [P.eval, P.evalExprs, h1, h2, h3] <;> rfl

/-! ### The two index searches are the same function -/

theorem strIndex_go_eq (t : List Char) : ∀ (s : List Char) (p : ℕ), F.strIndex.go t s p = P.pyIndex.go t s p
  | [], p => by simp only [F.strIndex.go, P.pyIndex.go]
  | c :: rest, p => by
      simp only [F.strIndex.go, P.pyIndex.go]
      split_ifs <;> first | rfl | exact strIndex_go_eq t rest (p + 1)

/-- `_f_index` (Python `str.find` + 1) is Fortran `INDEX`. -/
theorem strIndex_eq_pyIndex (s t : List Char) : F.strIndex s t = P.pyIndex s t := strIndex_go_eq t s 1

theorem strScan_go_eq (set : List Char) : ∀ (s : List Char) (p : ℕ), F.strScan.go set s p = P.pyScan.go set s p
  | [], p => by simp only [F.strScan.go, P.pyScan.go]
  | c :: rest, p => by
      simp only [F.strScan.go, P.pyScan.go]
      split_ifs <;> first | rfl | exact strScan_go_eq set rest (p + 1)

/-- `_f_scan` is Fortran `SCAN`. -/
theorem strScan_eq_pyScan (s set : List Char) : F.strScan s set = P.pyScan s set := strScan_go_eq set s 1

/-! ### FNP-R0028: `a // b` ↦ `(a' + b')` -/

def FNP_R0028_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr),
    SoundScalar o Γ a a' → SoundScalar o Γ b b' → StrTyped o Γ a → StrTyped o Γ b →
    SoundScalar o Γ (.call .concat (.cons a (.cons b .nil)) none) (.paren (.bin .add a' b'))

/-- `CONCAT` never turns an array operand into a scalar (the model returns `none`). -/
theorem concat_arr_ne (o : F.Oracle) (va vb : F.Val) (c : SVal)
    (h : (∃ av, va = .arr av) ∨ (∃ bv, vb = .arr bv)) : F.intrinsic o .concat [va, vb] none ≠ some (.s c) := by
  intro hc
  rcases h with ⟨av, rfl⟩ | ⟨bv, rfl⟩
  · have e : F.intrinsic o .concat [.arr av, vb] none = none := by
      cases vb with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc
  · have e : F.intrinsic o .concat [va, .arr bv] none = none := by
      cases va with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc

/-- Concatenation of the two related strings: Python `str + str` is list append
(`P.arith .add` on two `.str`), exactly Fortran's `//` (F2018 10.1.5.3). The
operands are the *stored* values; padding defects of the stores (FNP-D0021) are
the assignment rule's, not this rule's. -/
theorem FNP_R0028_sound : FNP_R0028_obligation := by
  intro o Γ a b a' b' ha hb hsa hsb σ τ hR hC v hv
  rw [F_eval_call2] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨va, hva, vb, hvb, hint⟩ := hv
  cases va with
  | arr av => exact absurd hint (concat_arr_ne o.F _ _ _ (Or.inl ⟨av, rfl⟩))
  | s x =>
    cases vb with
    | arr bv => exact absurd hint (concat_arr_ne o.F _ _ _ (Or.inr ⟨bv, rfl⟩))
    | s y =>
      obtain ⟨s, rfl⟩ := hsa σ τ hR hC x hva
      obtain ⟨t, rfl⟩ := hsb σ τ hR hC y hvb
      have e : F.intrinsic o.F .concat [.s (.str s), .s (.str t)] none = some (.s (.str (s ++ t))) := rfl
      rw [e] at hint
      cases hint
      obtain ⟨x', hx', hrx⟩ := ha σ τ hR hC _ hva
      obtain ⟨y', hy', hry⟩ := hb σ τ hR hC _ hvb
      rw [RelS.str_eq hrx] at hx'
      rw [RelS.str_eq hry] at hy'
      refine ⟨.str (s ++ t), ?_, RelS.str⟩
      rw [P_eval_paren, P_eval_bin, hx', hy']
      simp only [Option.bind_some, P_lift2_scalar]
      rfl

/-! ### FNP-R0085: one-argument character intrinsics -/

theorem trim_arr_ne (o : F.Oracle) (av : FArr) (c : SVal) : F.intrinsic o .trim [.arr av] none ≠ some (.s c) := by
  intro h; have e : F.intrinsic o .trim [.arr av] none = none := rfl; rw [e] at h; cases h
theorem adjustl_arr_ne (o : F.Oracle) (av : FArr) (c : SVal) : F.intrinsic o .adjustl [.arr av] none ≠ some (.s c) := by
  intro h; have e : F.intrinsic o .adjustl [.arr av] none = none := rfl; rw [e] at h; cases h
theorem char_arr_ne (o : F.Oracle) (av : FArr) (c : SVal) : F.intrinsic o .char [.arr av] none ≠ some (.s c) := by
  intro h; have e : F.intrinsic o .char [.arr av] none = none := rfl; rw [e] at h; cases h
theorem ichar_arr_ne (o : F.Oracle) (av : FArr) (c : SVal) : F.intrinsic o .ichar [.arr av] none ≠ some (.s c) := by
  intro h; have e : F.intrinsic o .ichar [.arr av] none = none := rfl; rw [e] at h; cases h

def FNP_R0085_trim_obligation : Prop := ∀ o Γ, Call1Sound o Γ .trim .f_trim
def FNP_R0085_adjustl_obligation : Prop := ∀ o Γ, Call1Sound o Γ .adjustl .f_adjustl
def FNP_R0085_char_obligation : Prop := ∀ o Γ, Call1Sound o Γ .char .py_chr
def FNP_R0085_ichar_obligation : Prop := ∀ o Γ, Call1Sound o Γ .ichar .py_ord

/-- `TRIM` ↦ `_f_trim` (`s.rstrip(" ")`): `F.strTrim` and `P.rstripBlanks` are the same list function. -/
theorem FNP_R0085_trim_sound : FNP_R0085_trim_obligation := fun o Γ =>
  call1_sound_of_rel o Γ .trim .f_trim (fun av c => trim_arr_ne o.F av c) (by
    intro x x' c hx hint
    cases x with
    | str s =>
      rw [RelS.str_eq hx]
      have e : F.intrinsic o.F .trim [.s (.str s)] none = some (.s (.str (F.strTrim s))) := rfl
      rw [e] at hint; cases hint
      exact ⟨.str (P.rstripBlanks s), rfl, RelS.str⟩
    | int k n => have e : F.intrinsic o.F .trim [.s (.int k n)] none = none := rfl; rw [e] at hint; cases hint
    | real k r => have e : F.intrinsic o.F .trim [.s (.real k r)] none = none := rfl; rw [e] at hint; cases hint
    | bool b => have e : F.intrinsic o.F .trim [.s (.bool b)] none = none := rfl; rw [e] at hint; cases hint)

/-- `ADJUSTL` ↦ `_f_adjustl` (`s.lstrip(" ").ljust(len(s))`). -/
theorem FNP_R0085_adjustl_sound : FNP_R0085_adjustl_obligation := fun o Γ =>
  call1_sound_of_rel o Γ .adjustl .f_adjustl (fun av c => adjustl_arr_ne o.F av c) (by
    intro x x' c hx hint
    cases x with
    | str s =>
      rw [RelS.str_eq hx]
      have e : F.intrinsic o.F .adjustl [.s (.str s)] none = some (.s (.str (F.strAdjustl s))) := rfl
      rw [e] at hint; cases hint
      exact ⟨.str (F.strAdjustl s), rfl, RelS.str⟩
    | int k n => have e : F.intrinsic o.F .adjustl [.s (.int k n)] none = none := rfl; rw [e] at hint; cases hint
    | real k r => have e : F.intrinsic o.F .adjustl [.s (.real k r)] none = none := rfl; rw [e] at hint; cases hint
    | bool b => have e : F.intrinsic o.F .adjustl [.s (.bool b)] none = none := rfl; rw [e] at hint; cases hint)

/-- `CHAR(i)` ↦ `chr(i)`: Fortran defines it on `0..255` (the default character
kind), `chr` on `0..0x10FFFF`; wherever the source is defined the two agree. -/
theorem FNP_R0085_char_sound : FNP_R0085_char_obligation := fun o Γ =>
  call1_sound_of_rel o Γ .char .py_chr (fun av c => char_arr_ne o.F av c) (by
    intro x x' c hx hint
    cases x with
    | int k i =>
      rw [RelS.int_eq hx]
      have e : F.intrinsic o.F .char [.s (.int k i)] none =
          (if 0 ≤ i ∧ i < 256 then some (.s (.str [Char.ofNat i.toNat])) else none) := rfl
      rw [e] at hint
      split_ifs at hint with hr
      cases hint
      have e' : P.helper o.P .py_chr [.s (.int i)] =
          (if 0 ≤ i ∧ i < 1114112 then some (.s (.str [Char.ofNat i.toNat])) else none) := rfl
      refine ⟨.str [Char.ofNat i.toNat], ?_, RelS.str⟩
      rw [e', if_pos ⟨hr.1, by omega⟩]
    | str s => have e : F.intrinsic o.F .char [.s (.str s)] none = none := rfl; rw [e] at hint; cases hint
    | real k r => have e : F.intrinsic o.F .char [.s (.real k r)] none = none := rfl; rw [e] at hint; cases hint
    | bool b => have e : F.intrinsic o.F .char [.s (.bool b)] none = none := rfl; rw [e] at hint; cases hint)

/-- `ICHAR(c)` ↦ `ord(c)` on a one-character string. -/
theorem FNP_R0085_ichar_sound : FNP_R0085_ichar_obligation := fun o Γ =>
  call1_sound_of_rel o Γ .ichar .py_ord (fun av c => ichar_arr_ne o.F av c) (by
    intro x x' c hx hint
    cases x with
    | str s =>
      rw [RelS.str_eq hx]
      cases s with
      | nil => have e : F.intrinsic o.F .ichar [.s (.str [])] none = none := rfl; rw [e] at hint; cases hint
      | cons ch rest =>
        cases rest with
        | nil =>
          have e : F.intrinsic o.F .ichar [.s (.str [ch])] none = some (.s (.int .i4 ch.toNat)) := rfl
          rw [e] at hint; cases hint
          exact ⟨.int ch.toNat, rfl, RelS.int⟩
        | cons d more =>
          have e : F.intrinsic o.F .ichar [.s (.str (ch :: d :: more))] none = none := rfl
          rw [e] at hint; cases hint
    | int k n => have e : F.intrinsic o.F .ichar [.s (.int k n)] none = none := rfl; rw [e] at hint; cases hint
    | real k r => have e : F.intrinsic o.F .ichar [.s (.real k r)] none = none := rfl; rw [e] at hint; cases hint
    | bool b => have e : F.intrinsic o.F .ichar [.s (.bool b)] none = none := rfl; rw [e] at hint; cases hint)

/-! ### FNP-R0085: two-argument character intrinsics -/

/-- `INDEX`/`SCAN` are undefined unless both scalar operands are strings. -/
theorem strfn2_none (o : F.Oracle) (f : F.Intr) (hf : f = .index_ ∨ f = .scan) (x y : SVal)
    (h : (∀ s, x ≠ .str s) ∨ (∀ t, y ≠ .str t)) : F.intrinsic o f [.s x, .s y] none = none := by
  rcases hf with rfl | rfl <;> rcases h with hx | hy
  · cases x with
    | str s => exact absurd rfl (hx s)
    | int _ _ => cases y <;> rfl
    | real _ _ => cases y <;> rfl
    | bool _ => cases y <;> rfl
  · cases y with
    | str t => exact absurd rfl (hy t)
    | int _ _ => cases x <;> rfl
    | real _ _ => cases x <;> rfl
    | bool _ => cases x <;> rfl
  · cases x with
    | str s => exact absurd rfl (hx s)
    | int _ _ => cases y <;> rfl
    | real _ _ => cases y <;> rfl
    | bool _ => cases y <;> rfl
  · cases y with
    | str t => exact absurd rfl (hy t)
    | int _ _ => cases x <;> rfl
    | real _ _ => cases x <;> rfl
    | bool _ => cases x <;> rfl

theorem strfn2_arr_ne (o : F.Oracle) (f : F.Intr) (hf : f = .index_ ∨ f = .scan) (va vb : F.Val) (c : SVal)
    (h : (∃ av, va = .arr av) ∨ (∃ bv, vb = .arr bv)) : F.intrinsic o f [va, vb] none ≠ some (.s c) := by
  intro hc
  rcases hf with rfl | rfl <;> rcases h with ⟨av, rfl⟩ | ⟨bv, rfl⟩
  · have e : F.intrinsic o .index_ [.arr av, vb] none = none := by
      cases vb with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc
  · have e : F.intrinsic o .index_ [va, .arr bv] none = none := by
      cases va with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc
  · have e : F.intrinsic o .scan [.arr av, vb] none = none := by
      cases vb with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc
  · have e : F.intrinsic o .scan [va, .arr bv] none = none := by
      cases va with
      | arr _ => rfl
      | s sv => cases sv <;> rfl
    rw [e] at hc; cases hc

def FNP_R0085_index_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr), SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .index_ (.cons a (.cons b .nil)) none) (Tr.call2 .f_index a' b')

def FNP_R0085_scan_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (a b : F.Expr) (a' b' : P.Expr), SoundScalar o Γ a a' → SoundScalar o Γ b b' →
    SoundScalar o Γ (.call .scan (.cons a (.cons b .nil)) none) (Tr.call2 .f_scan a' b')

/-- `INDEX(s, t)` ↦ `_f_index(s', t')` (`str.find` shifted to 1-based, `0` if absent). -/
theorem FNP_R0085_index_sound : FNP_R0085_index_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .index_ .f_index a b a' b' ha hb ?_
    (fun va vb c h => strfn2_arr_ne o.F .index_ (Or.inl rfl) va vb c h)
  intro x y x' y' c hx hy hint
  cases x with
  | str s =>
    cases y with
    | str t =>
      rw [RelS.str_eq hx, RelS.str_eq hy]
      have e : F.intrinsic o.F .index_ [.s (.str s), .s (.str t)] none = some (.s (.int .i4 (F.strIndex s t))) := rfl
      rw [e] at hint; cases hint
      refine ⟨.int (P.pyIndex s t), rfl, ?_⟩
      rw [strIndex_eq_pyIndex]; exact RelS.int
    | int _ _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
    | real _ _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
    | bool _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
  | int _ _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint
  | real _ _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint
  | bool _ => rw [strfn2_none o.F .index_ (Or.inl rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint

/-- `SCAN(s, set)` ↦ `_f_scan(s', set')`. -/
theorem FNP_R0085_scan_sound : FNP_R0085_scan_obligation := by
  intro o Γ a b a' b' ha hb
  refine call2_sound_of_rel o Γ .scan .f_scan a b a' b' ha hb ?_
    (fun va vb c h => strfn2_arr_ne o.F .scan (Or.inr rfl) va vb c h)
  intro x y x' y' c hx hy hint
  cases x with
  | str s =>
    cases y with
    | str t =>
      rw [RelS.str_eq hx, RelS.str_eq hy]
      have e : F.intrinsic o.F .scan [.s (.str s), .s (.str t)] none = some (.s (.int .i4 (F.strScan s t))) := rfl
      rw [e] at hint; cases hint
      refine ⟨.int (P.pyScan s t), rfl, ?_⟩
      rw [strScan_eq_pyScan]; exact RelS.int
    | int _ _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
    | real _ _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
    | bool _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inr fun _ h => by cases h)] at hint; cases hint
  | int _ _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint
  | real _ _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint
  | bool _ => rw [strfn2_none o.F .scan (Or.inr rfl) _ _ (Or.inl fun _ h => by cases h)] at hint; cases hint

/-! ### Substrings `s(i:j)` ↦ `s'[i' - 1:j']` -/

/-- The rule as the emitter accepts it: any integer edges. -/
def substr_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (s i j : F.Expr) (s' i' j' : P.Expr),
    SoundScalar o Γ s s' → SoundScalar o Γ i i' → SoundScalar o Γ j j' →
    StrTyped o Γ s → IntTyped o Γ i → IntTyped o Γ j →
    SoundScalar o Γ (.call .substr (.cons s (.cons i (.cons j .nil))) none)
      (.call .str_slice (.cons s' (.cons (.bin .sub i' (.intLit 1)) (.cons j' .nil))))

/-- Edge premise: `1 ≤ i` and `0 ≤ j ≤ len` (a non-empty substring's edges, plus the
empty ones whose rendered stop stays in `[0, len]`). -/
def SubstrInRange (o : Oracles) (Γ : F.Env) (s i j : F.Expr) : Prop :=
  ∀ σ τ, RelState Γ σ τ → Conforms o.F Γ σ → ∀ str ki iv kj jv,
    F.eval o.F σ s = some (.s (.str str)) → F.eval o.F σ i = some (.s (.int ki iv)) →
    F.eval o.F σ j = some (.s (.int kj jv)) → 1 ≤ iv ∧ 0 ≤ jv ∧ jv ≤ str.length

/-- CPython `s[a:e]` is empty when `0 ≤ e ≤ n`, `0 ≤ a` and `e ≤ a` (no wrap, no gap). -/
theorem sliceIndices_empty_of_le (n : ℕ) (a e : ℤ) (ha : 0 ≤ a) (he0 : 0 ≤ e) (hen : e ≤ n) (hea : e ≤ a) :
    Py.sliceIndices n (some a) (some e) 1 = [] := by
  have hadj : Py.adjust n (some a) (some e) 1 = (if a ≥ (n : ℤ) then (n : ℤ) else a, e) := by
    simp only [Py.adjust, Prod.mk.injEq]
    constructor <;> (split_ifs <;> omega)
  have hlen : Py.sliceLen (if a ≥ (n : ℤ) then (n : ℤ) else a) e 1 = 0 := by
    unfold Py.sliceLen
    split_ifs <;> omega
  unfold Py.sliceIndices
  rw [hadj]
  simp only [hlen, List.range_zero, List.map_nil]

/-- `(s.drop a).take m` is the map of `getD` over `range m` when `a + m ≤ len`. -/
theorem drop_take_eq_map_getD (s : List Char) (a m : ℕ) (h : a + m ≤ s.length) :
    (s.drop a).take m = (List.range m).map fun k => s.getD (a + k) ' ' := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_drop, List.length_map, List.length_range]; omega
  · intro k h₁ h₂
    simp only [List.length_take, List.length_drop] at h₁
    have hk : a + k < s.length := by omega
    rw [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk]
    rfl

/-- Substrings are sound under the edge premise. -/
theorem substr_sound (o : Oracles) (Γ : F.Env) (s i j : F.Expr) (s' i' j' : P.Expr)
    (hs : SoundScalar o Γ s s') (hi : SoundScalar o Γ i i') (hj : SoundScalar o Γ j j')
    (hst : StrTyped o Γ s) (hit : IntTyped o Γ i) (hjt : IntTyped o Γ j) (hrange : SubstrInRange o Γ s i j) :
    SoundScalar o Γ (.call .substr (.cons s (.cons i (.cons j .nil))) none)
      (.call .str_slice (.cons s' (.cons (.bin .sub i' (.intLit 1)) (.cons j' .nil)))) := by
  intro σ τ hR hC v hv
  rw [F_eval_call3] at hv
  simp only [Option.map_none, Option.bind_eq_some_iff] at hv
  obtain ⟨vs, hvs, vi, hvi, vj, hvj, hint⟩ := hv
  -- the arguments are a string and two integers (else the intrinsic is undefined)
  cases vs with
  | arr av =>
    exfalso
    have e : F.intrinsic o.F .substr [.arr av, vi, vj] none = none := by
      cases vi <;> cases vj <;> rfl
    rw [e] at hint; cases hint
  | s xs =>
    obtain ⟨str, rfl⟩ := hst σ τ hR hC xs hvs
    cases vi with
    | arr av =>
      exfalso
      have e : F.intrinsic o.F .substr [.s (.str str), .arr av, vj] none = none := by
        cases vj <;> rfl
      rw [e] at hint; cases hint
    | s xi =>
      obtain ⟨ki, iv, rfl⟩ := hit σ τ hR hC xi hvi
      cases vj with
      | arr bv =>
        exfalso
        have e : F.intrinsic o.F .substr [.s (.str str), .s (.int ki iv), .arr bv] none = none := rfl
        rw [e] at hint; cases hint
      | s xj =>
        obtain ⟨kj, jv, rfl⟩ := hjt σ τ hR hC xj hvj
        obtain ⟨h1, hj0, hjn⟩ := hrange σ τ hR hC str ki iv kj jv hvs hvi hvj
        have e : F.intrinsic o.F .substr [.s (.str str), .s (.int ki iv), .s (.int kj jv)] none =
            (if jv < iv then some (.s (.str [])) else
              if 1 ≤ iv ∧ jv ≤ str.length then some (.s (.str ((str.drop (iv - 1).toNat).take (jv - iv + 1).toNat)))
              else none) := rfl
        rw [e] at hint
        -- the Python side
        obtain ⟨ws, hws, hrs⟩ := hs σ τ hR hC _ hvs
        obtain ⟨wi, hwi, hri⟩ := hi σ τ hR hC _ hvi
        obtain ⟨wj, hwj, hrj⟩ := hj σ τ hR hC _ hvj
        rw [RelS.str_eq hrs] at hws
        rw [RelS.int_eq hri] at hwi
        rw [RelS.int_eq hrj] at hwj
        have hsub : P.eval o.P τ (.bin .sub i' (.intLit 1)) = some (.s (.int (iv - 1))) := by
          rw [P_eval_bin, hwi]; simp [P.eval, P.lift2, P.arith]
        have hpy : P.eval o.P τ (.call .str_slice (.cons s' (.cons (.bin .sub i' (.intLit 1)) (.cons j' .nil)))) =
            some (.s (.str (P.strSlice str (some (iv - 1)) (some jv)))) := by
          rw [P_eval_call3, hws, hsub, hwj]
          rfl
        rw [hpy]
        split_ifs at hint with hlt hdom
        · -- empty substring: `j < i`
          cases hint
          refine ⟨.str [], ?_, RelS.str⟩
          have : P.strSlice str (some (iv - 1)) (some jv) = [] := by
            unfold P.strSlice
            rw [sliceIndices_empty_of_le str.length (iv - 1) jv (by omega) hj0 hjn (by omega)]
            rfl
          rw [this]
        · -- non-empty: the two lists agree element by element
          cases hint
          have hle : iv ≤ jv := not_lt.mp hlt
          refine ⟨.str ((str.drop (iv - 1).toNat).take (jv - iv + 1).toNat), ?_, RelS.str⟩
          unfold P.strSlice
          rw [sliceIndices_unit str.length (iv - 1) jv (by omega) (by omega) hj0 hjn]
          rw [if_pos (by omega), List.map_map]
          rw [drop_take_eq_map_getD str (iv - 1).toNat (jv - iv + 1).toNat (by omega)]
          have hm : (jv - (iv - 1)).toNat = (jv - iv + 1).toNat := by omega
          rw [hm]
          congr 3
          apply List.map_congr_left
          intro k _
          simp only [Function.comp]
          congr 1
          try omega

/-- `s(1:-1)` on `'abc'`: Fortran `''` (F2018 9.4.1: `j < i` is the empty
string), CPython `s[0:-1] = 'ab'` (negative stop wraps to `len - 1`). The
D0037 class for substrings; needs an engine probe of how `s(i:j)` is rendered
before it becomes a defect record. -/
theorem substr_refuted : ¬ substr_obligation := by
  intro h
  have hs := h noPrim emptyEnv (.strLit ['a', 'b', 'c']) (.intLit .i4 1) (.intLit .i4 (-1))
    (.strLit ['a', 'b', 'c']) (.intLit 1) (.intLit (-1))
    (strLit_sound _ _ _) (FNP_R0001_sound _ _ _ _) (FNP_R0001_sound _ _ _ _)
    (strLit_strTyped _ _ _) (intLit_intTyped _ _ .i4 1) (intLit_intTyped _ _ .i4 (-1))
  have hF : F.eval noPrim.F emptyF (.call .substr (.cons (.strLit ['a', 'b', 'c'])
      (.cons (.intLit .i4 1) (.cons (.intLit .i4 (-1)) .nil))) none) = some (.s (.str [])) := by
    rw [F_eval_call3]
    simp only [F.eval, Option.bind_some, Option.map_none]
    have e : F.intrinsic noPrim.F .substr [.s (.str ['a', 'b', 'c']), .s (.int .i4 1), .s (.int .i4 (-1))] none =
        (if (-1 : ℤ) < 1 then some (.s (.str [])) else
          if 1 ≤ (1 : ℤ) ∧ (-1 : ℤ) ≤ (['a', 'b', 'c'] : List Char).length then
            some (.s (.str (((['a', 'b', 'c'] : List Char).drop ((1 : ℤ) - 1).toNat).take ((-1 : ℤ) - 1 + 1).toNat)))
          else none) := rfl
    rw [e, if_pos (by norm_num)]
  obtain ⟨w, hw, hrel⟩ := hs emptyF emptyP relState_empty conforms_empty _ hF
  have hP : P.eval noPrim.P emptyP (.call .str_slice (.cons (.strLit ['a', 'b', 'c'])
      (.cons (.bin .sub (.intLit 1) (.intLit 1)) (.cons (.intLit (-1)) .nil)))) = some (.s (.str ['a', 'b'])) := by
    rw [P_eval_call3]
    have hsub : P.eval noPrim.P emptyP (.bin .sub (.intLit 1) (.intLit 1)) = some (.s (.int 0)) := by
      rw [P_eval_bin]; simp [P.eval, P.lift2, P.arith]
    rw [hsub]
    simp only [P.eval, Option.bind_some]
    have e : P.helper noPrim.P .str_slice [.s (.str ['a', 'b', 'c']), .s (.int 0), .s (.int (-1))] =
        some (.s (.str (P.strSlice ['a', 'b', 'c'] (some 0) (some (-1))))) := rfl
    rw [e]
    have : P.strSlice ['a', 'b', 'c'] (some 0) (some (-1)) = ['a', 'b'] := by decide
    rw [this]
  rw [hP] at hw
  cases hw
  have := RelS.str_eq hrel
  cases this

/-! ### FNP-R0008: BOZ literals -/

/-- A BOZ constant in an INTEGER context denotes the integer with that bit
pattern; the emitter spells it `0x…`/`0o…`/`0b…`, a Python `int` of the same
value. The obligation is FNP-R0001's on the value; a BOZ above `HUGE` of the
default kind is processor dependent (F2018 7.7) and outside the model, hence the
premise `v < 2^31`. -/
def FNP_R0008_obligation : Prop :=
  ∀ (o : Oracles) (Γ : F.Env) (v : ℤ), 0 ≤ v → v < 2 ^ 31 → SoundScalar o Γ (.intLit .i4 v) (.intLit v)

theorem FNP_R0008_sound : FNP_R0008_obligation := fun o Γ v _ _ => FNP_R0001_sound o Γ .i4 v

/-! ### Support for character SELECT CASE (FNP-R0122) / `==` (FNP-R0031)

`strEq_iff_rstrip : F.strEq a b = true ↔ P.rstripBlanks a = P.rstripBlanks b`
already exists in `Rules/Logic.lean` under exactly this name; nothing to add. -/

end Recast.Rules
