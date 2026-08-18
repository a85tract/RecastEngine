# Splitting the translator

`CESM-language-translator/pipeline/translate.py` is 2,883 lines that are parser,
rule library, and emitter at once. That is why nothing else can reuse any part
of it, and why `rwset.py` had to import from it to ask a question about Fortran.

This is the inventory the split works from, and the boundary it draws. It is
here rather than in a commit message because the split lands over several
commits and each one has to be able to say which side of the line it is on.

## What is actually in there

Measured, not estimated -- top-level spans and per-method spans of
`class Translator`.

| Lines | What | Belongs to |
|---:|---|---|
| 210 | name tables, regexes, compiler profiles | NumPy backend |
| 326 | runtime shim library, inside a string constant | NumPy backend |
| 241 | type and shape reasoning (11 methods) | Fortran frontend |
| 66 | generic-interface dispatch (1 method) | Fortran frontend |
| 192 | index and slice rewriting, 1-based to 0-based (6 methods) | rules |
| 409 | expression emission (12 methods) | NumPy backend |
| 664 | statement emission (18 methods) | NumPy backend |
| 322 | subprogram assembly (6 methods) | NumPy backend |
| 69 | construction and companion loading (2 methods) | Transform |
| 303 | `main()` -- the CLI driver | `Transform.apply` |

Roughly 307 lines answer questions about Fortran and would be the same for any
target language. Roughly 1,931 lines are how *this* target spells things. The
remaining 192 are the rewrite rules proper -- decisions that are neither, and
that a second backend would want to reuse rather than re-derive.

## Where the line falls

**`recast.fortran`** answers what the source *means*: the rank of an
expression, whether it is integer-valued, whether it is a constant expression,
which specific procedure a generic call dispatches to. Nothing here can produce
a line of target-language source, and every answer would be identical for a
Julia or C++ backend.

**`recast.transform.rules`** holds decisions that are target-independent but
not facts about the source: a Fortran array starts at 1 and a Python one at 0,
so every index shifts; sequence association means a rank-2 actual can bind to a
rank-1 formal; a statement function is inlined at its call sites. These are
choices about *how to move* between languages, and they are expressed as data
so a backend can render them rather than re-decide them.

**`recast.transform.numpy`** is everything that knows the target exists.

The test of whether the line is in the right place is `rwset.py`. It needed one
question answered -- "is this name an intrinsic" -- and had to import a
2,883-line emitter to ask it. After the split it asks `recast.fortran` and the
emitter is not involved.

## Order, and why

1. **Runtime shim library.** It is 326 lines of Python living inside a string
   constant, so nothing lints it, nothing type-checks it, and nothing tests it
   -- and it is pasted verbatim into every generated file. It is also where the
   risk is concentrated: these 46 functions are the definitions of `sign`,
   `mod`, `nint` and `transfer`, and each one is a place where Python's answer
   differs from Fortran's. `_f_mod` truncates where Python's `%` floors;
   `_f_nint` rounds half away from zero where Python's `round` rounds to even.
   A wrong one corrupts every number that flows through it and no structural
   check notices. Cleanest cut in the file and the highest value per line.

2. **Name tables and compiler profiles.** Data, no coupling.

3. **Fortran semantics into the frontend.** Removes the duplication that
   already exists: `translate.py` and `rwset.py` each carry a generic-dispatch
   implementation, and they disagree -- one refuses on ambiguity, the other
   scores the candidates and picks the best. Across the thirty translated CAM
   modules there are only three generic call sites and all three match cleanly,
   so the disagreement is latent rather than active. It is a trap, not a fire,
   and it gets closed here rather than after it costs someone a day.

4. **Index and slice rewriting into rules.**

5. **The emitters**, which is most of the remaining volume and the part that
   should move last, because by then it has somewhere to call into.

## The gate this runs against

`static.rwset` already exists and does not depend on any of this. Every slice
that changes emitted output can be checked against the 38 files the original
pipeline produced, and against the read/write cross-check, before it lands.
`tools/golden_diff.py` is the same idea for analysis output.
