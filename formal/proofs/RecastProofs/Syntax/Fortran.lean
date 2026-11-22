import RecastProofs.Semantics.Values
import RecastProofs.Semantics.Literal

/-!
# Fortran source syntax (the fragment covered by the inventoried rules)

An abstract syntax for the Fortran expressions and statements that the
engine's built-in rules translate.  It is *source* syntax: kinds on literals,
1-based inclusive sections with optional edges, intrinsic calls by name, and a
declaration environment carrying declared lower bounds as *expressions*
(a declared bound may name another dummy argument, exactly as the engine's
`dims[axis]["lb"]` text does).

Nothing in this file refers to Python.
-/

namespace Recast.F

inductive CmpOp where
  | eq | ne | lt | le | gt | ge
  deriving DecidableEq, Repr

inductive BinOp where
  | add | sub | mul | div | pow
  deriving DecidableEq, Repr

inductive LogOp where
  | and | or | eqv | neqv
  deriving DecidableEq, Repr

/-- Intrinsic functions appearing in the inventoried rules.  `prim` is an
uninterpreted pure real function shared by both semantics (e.g. `erf`,
`gamma`), treated by congruence under an explicit contract. -/
inductive Intr where
  | abs | sqrt | exp | log | log10 | sin | cos | tan | asin | acos | atan | atan2
  | sinh | cosh | tanh
  | int | nint | aint | anint | floor | ceiling | real | dble
  | mod | modulo | sign | dim | max | min | merge
  | sum | product | maxval | minval | dot_product | size | lbound | ubound
  | kind | huge | tiny | epsilon | precision | digits
  | iand | ior | ieor | ishft
  | len | len_trim
  /-- Character operations modelled as intrinsic nodes: `a // b`, `s(i:j)`,
  TRIM, ADJUSTL, INDEX, SCAN, CHAR, ICHAR (FNP-R0028, R0085). -/
  | concat | substr | trim | adjustl | index_ | scan | char | ichar
  /-- Array transforms with optional arguments encoded by flags; the argument
  list holds the present arguments in standard order (FNP-R0102..R0107).
  `reshape hasPad hasOrder`: `[source, shape (, pad) (, order)]`;
  `pack hasVector`: `[array, mask (, vector)]`;
  `cshift hasDim`: `[array, shift (, dim)]`;
  `eoshift hasBoundary hasDim`: `[array, shift (, boundary) (, dim)]`;
  `maxloc`/`minloc hasMask`: `[array (, mask)]`. -/
  | reshape (hasPad hasOrder : Bool) | pack (hasVector : Bool) | cshift (hasDim : Bool)
  | eoshift (hasBoundary hasDim : Bool) | maxloc (hasMask : Bool) | minloc (hasMask : Bool)
  | transpose | matmul | shape
  /-- MVBITS as a function returning the new TO (FNP-R0094). -/
  | mvbits
  | prim (name : String)
  deriving DecidableEq, Repr

mutual
  /-- Expressions. -/
  inductive Expr where
    | intLit (k : IKind) (v : ℤ)
    | realLit (k : RKind) (d : Decimal)
    | boolLit (b : Bool)
    | strLit (s : List Char)
    | var (x : String)
    | neg (e : Expr)
    | paren (e : Expr)
    | bin (op : BinOp) (a b : Expr)
    | cmp (op : CmpOp) (a b : Expr)
    | logic (op : LogOp) (a b : Expr)
    | not (e : Expr)
    /-- `A(s₁, …, sₙ)`: an element (all scalar subscripts) or a section. -/
    | ref (a : String) (subs : Subs)
    /-- Intrinsic call; `k` is an explicit `KIND=` argument where the intrinsic
    takes one (`INT(x, KIND=8)`). -/
    | call (f : Intr) (args : Exprs) (k : Option IKind)
    /-- Array constructor `[e₁, …, eₙ]` / `(/ … /)` of scalars (rank 1, unit origin). -/
    | arrayCtor (es : Exprs)
  inductive Exprs where
    | nil
    | cons (e : Expr) (rest : Exprs)
  /-- One subscript position. -/
  inductive Sub where
    | scalar (e : Expr)
    | triplet (lo hi st : Bound)
    /-- Vector subscript `A(v)` with an INTEGER rank-1 array `v` (F2018 9.5.3.3.3). -/
    | vector (e : Expr)
  inductive Subs where
    | nil
    | cons (s : Sub) (rest : Subs)
  /-- A section edge, possibly omitted. -/
  inductive Bound where
    | omitted
    | given (e : Expr)
end

namespace Exprs
def toList : Exprs → List Expr
  | .nil => []
  | .cons e r => e :: toList r
def length (es : Exprs) : ℕ := es.toList.length
end Exprs

namespace Subs
def toList : Subs → List Sub
  | .nil => []
  | .cons s r => s :: toList r
/-- Number of triplet positions: the rank of the section. -/
def rank : Subs → ℕ
  | .nil => 0
  | .cons (.triplet _ _ _) r => rank r + 1
  | .cons (.vector _) r => rank r + 1
  | .cons (.scalar _) r => rank r
end Subs

/-- An actual argument of a `CALL`: a variable (which an OUT/INOUT dummy writes
back into) or a general expression (which only an IN dummy may take). -/
inductive Actual where
  | var (x : String)
  | expr (e : Expr)

/-- Dummy-argument intents (F2018 8.5.10). `unknown` is the engine's label for a
dummy whose intent it could not infer; it is treated as INOUT. -/
inductive Intent where
  | inn | out | inout | unknown
  deriving DecidableEq, Repr

/-- One dummy argument: its name inside the procedure and its intent. -/
structure Dummy where
  name : String
  intent : Intent

/-- A `CASE` selector item: a single value or a closed/open range `lo:hi`. -/
inductive CaseItem where
  | value (e : Expr)
  | range (lo hi : Option Expr)

/-- A declaration: scalar type, one lower-bound expression per axis
(length = rank; scalars have `lbs = []`), and the declared character length
for `CHARACTER(LEN=clen)` entities (`0` otherwise). -/
structure Decl where
  ty : FTy
  lbs : List Expr
  clen : ℕ := 0

/-- The declaration environment of one subprogram (names are lower-cased). -/
def Env := String → Option Decl

/-- Statements. -/
inductive Stmt where
  /-- `x = e`, `A = e`, `A(i,j) = e`, `A(lo:hi) = e`: target is a name with subscripts. -/
  | assign (target : String) (subs : Subs) (e : Expr)
  | seq (s₁ s₂ : Stmt)
  | skip
  | ite (c : Expr) (t e : Stmt)
  /-- `DO v = lo, hi [, st]` with an optional step. -/
  | doLoop (v : String) (lo hi : Expr) (st : Option Expr) (body : Stmt)
  | doWhile (c : Expr) (body : Stmt)
  | exit
  | cycle
  /-- `FORALL (v = lo:hi[:st] [, mask]) target(subs) = rhs`: all right-hand sides
  are evaluated before any element is assigned (F2018 10.2.4). -/
  | forall1 (v : String) (lo hi : Expr) (st : Option Expr) (mask : Option Expr)
      (target : String) (subs : Subs) (rhs : Expr)
  /-- `WHERE (mask) a₁ = e₁; …  [ELSEWHERE b₁ = f₁; …] END WHERE` with whole-array
  targets; the mask is evaluated once (F2018 10.2.3). -/
  | whereBlock (mask : Expr) (asgs : List (String × Expr)) (elseAsgs : List (String × Expr))
  /-- One arm of `SELECT CASE (sel)`: `CASE (items) body`, then the remaining arms
  (`rest` is the next arm, the default body, or `skip`). -/
  | caseArm (sel : Expr) (items : List CaseItem) (body : Stmt) (rest : Stmt)
  /-- `ASSOCIATE (n => x) body END ASSOCIATE` with a *variable* selector: `n` names `x`. -/
  | associate (n : String) (x : String) (body : Stmt)
  /-- `ALLOCATE (x(lo₁:hi₁, …))`: `x` becomes an array of those bounds (values undefined;
  the model zero-fills). An omitted lower bound is 1. -/
  | allocate (x : String) (specs : List (Option Expr × Expr))
  /-- `CALL p (a₁, …, aₙ)` (F2018 15.5.2: argument association, positional only). -/
  | call (p : String) (actuals : List Actual)

/-- A subprogram: its own declaration environment (dummies, locals and, for a
function, the result), its dummy list in argument order, the result name for a
function, and its body. -/
structure Proc where
  env : Env
  dummies : List Dummy
  /-- Declared local scalars, zero-initialised on entry (the emitted prologue,
  FNP-R0142).  Fortran leaves them undefined; see the idealisation note. -/
  locals : List String
  result : Option String
  body : Stmt

/-- The procedures visible at a call site. -/
def ProcEnv := String → Option Proc

end Recast.F
