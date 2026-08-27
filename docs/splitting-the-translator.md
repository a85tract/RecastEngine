# Splitting the translator

The translator's `pipeline/translate.py` was 2,883 lines that are parser,
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

2. **Name tables and compiler profiles.** Data, no coupling -- but not all of
   it belongs here. Of the 210 lines, 131 are stubs for CAM's `cam_history`,
   MCT, ESMF, PIO and GPTL: which framework calls carry no physics and can
   become `pass`. That is CESM knowledge, not knowledge of NumPy, and it goes
   to the domain extension rather than into the engine. The compiler profiles are
   neither frontend nor backend -- they describe the *source* compiler whose
   output the translation has to match -- so they sit on the
   target-independent side where a second backend can read them.

3. **Fortran semantics into the frontend.** Done: `recast.fortran.semantics`.
   The duplicated generic dispatch is closed on the strict implementation --
   the one that refuses when a call matches none of its specifics or more than
   one -- and `rwset` reads a refusal as an unresolved external rather than
   scoring the candidates itself. All three generic call sites in the thirty
   translated CAM modules still resolve, and the read/write sets are unchanged.

   `dim_lb` and `dim_expr` look like siblings of these and stayed behind: their
   answer is Python text, so they belong to whoever has a target language.

4. **Index and slice rewriting into rules.** Done: `recast.transform.rules`.
   Less of it moved than the line count suggested. `dataref_expr`, `_bound_py`
   and most of the sequence-association code build Python text and stayed with
   the emitter; what came across is the part that decides -- which positions
   are indices, ranges or gathers, what each shifts by, whether a literal may
   be folded, and where a construct has no mechanical rewrite at all.

   Rules produce plans that refer to source nodes without rendering them, so a
   second zero-based backend renders once rather than re-deriving. Checked
   against the original over 3,883 subscripts in five CAM modules: identical
   shape and identical acceptance on every one, and 39 literal folds agreeing.
   No subscript in the corpus takes a refusing path, so the refusals have
   tests and nothing else.

5. **The emitters**, moving last because by then they have somewhere to call
   into. Expression emission is done: `recast.transform.numpy.expressions`,
   the layer where the four earlier slices meet, checked byte-identical
   against the pipeline over the translated corpus. Statement emission is
   done: `recast.transform.numpy.statements`, the floor above it --
   assignment's copy-into-storage semantics, WHERE masks, do bounds shifting
   by the sign of the step, the two goto shapes that structure cleanly, and
   the call statement's intent rewriting. Subprogram assembly is done:
   `recast.transform.numpy.subprograms`, the function around the statements
   -- the signature that turns `intent(out)` arguments into return values,
   the determinizing prologue for everything Fortran leaves undefined, the
   block markers everything downstream keys on, patches, deferral, and the
   block report that travels with the Candidate. Checked whole-subprogram
   against the pipeline over the six schemes with full operator tables and
   the twenty-one batch-swept modules: 276 subprograms, 2,793 blocks, 18,520
   emitted lines, byte-identical with refusal prose normalized and refusal
   placement compared strictly. Module-level rendering is done:
   `recast.transform.numpy.modules`, everything in a generated file that is
   not inside a `def` -- derived-type factories, module state with its
   save-initializers, the embedded signature table, and the runtime pasted
   in whole so the file stands alone. The module *body* is held
   byte-identical to a patch-free run of the pipeline's `main()`; the header
   is deliberately the engine's own, because the runtime here is real, typed,
   tested code rather than the pipeline's string constant, and its emitted
   text follows the code. `tools/emit_diff.py` keeps all of it standing, the
   emission analog of `golden_diff.py`; it discovers the swept modules from
   the translator's `extracted_auto/` at run time, so a new sweep widens the
   check by itself.

   And the wiring is done. `recast.transform.numpy.constants` renders the
   generated constants module (byte-identical to `extract_constants.py`'s
   over the corpus, held by the same tool) and the use-constants module,
   from the same `Expr` trees the oracle's Fortran stand-in will render --
   agreement by construction. `recast.transform.numpy.translate` is
   `main()` reduced to what it always was underneath: a `Transform.apply`
   that takes a Unit and its Facts, consults the operator's tables, and
   returns a Candidate carrying the whole product -- module, constants,
   use-constants, the block report, the deferred list that is the agent
   queue, and the name-protocol table. It registers as `translate.numpy`,
   which is the name the `translate` recipe's transform stage asks for.

   The split is complete. Every line of `translate.py`,
   `extract_constants.py`'s emission half, and `resolve_use.py`'s Python
   half now has a home in the engine, and the differential says the homes
   emit what the originals emitted.

## Where this repository disagrees with the pipeline: it does not

The pipeline's answers have been run against bit-exact gates on real CESM
cases. Nothing here has. So the rule for every slice is that a difference from
it is a bug in the migration until proven otherwise, and where the two can
both be defended, the pipeline wins.

That rule has already caught one mistake in this repository's own reporting.
The read/write analysis was said to have fixed the pipeline's handling of
`deallocate`; it had not. The pipeline treats it as a write and always has --
what the migration was being compared against was a golden `interface.json`
older than the pipeline that produced it. Comparing against stored output
rather than against the code is how that happens, and the differential tools
run against the code now.

Widening `emit_diff` to the swept modules caught a second, in the expression
layer: it consulted the function-stub table for every unknown reference,
where the pipeline answers from that table only for references parsed as
structure constructors. So a `hist_fld_active(name_out)` the pipeline defers
to a human came out as `if False:` -- emitted, dead, and silent about it.
Narrowed to the pipeline's placement, and pinned by a test.

Three real inconsistencies survive, reproduced deliberately, each one a place
where the pipeline disagrees with itself:

* An out-argument counts a derived-type component name as a read; an
  assignment to the same thing does not.
* Local parameters are visible to the type and array queries and not to the
  shape query, so a lookup table is an array to one and a scalar to the other.
* A companion module's generic interfaces get a rank and the module's own
  refuse one.

Each has a tidier answer, and none of the sites where the difference shows has
a translation to check the tidier answer against -- either the module was never
translated, or the block went to the agent queue and was written by hand. They
stay as they are, and the tests say so rather than pretending the behaviour is
intended.

The statement slice adds a fourth kind of finding: an intended refusal that
never fires. The pipeline means to refuse a `case` value range, but its check
looks at the selector's children, which hold a range *list* rather than a bare
range -- so `case (1:2)` slips through and its endpoints emit as equality
tests, right for that range by luck and wrong for any wider one. Reproduced,
because no translated module contains a case range for the tidier answer to be
checked against, and pinned by a test that says the behaviour is inherited.

The duplications the split removes are a different matter, because collapsing
two copies of one rule onto the copy that is already tested changes nothing:
the literal whitelist, the renaming rule that existed three times, the
generic dispatch that existed twice.

## The gate this runs against

`static.rwset` already exists and does not depend on any of this. Every slice
that changes emitted output can be checked against the 38 files the original
pipeline produced, and against the read/write cross-check, before it lands.
`tools/golden_diff.py` is the same idea for analysis output.
