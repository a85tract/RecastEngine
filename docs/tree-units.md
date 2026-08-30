# Units of a model tree

`translate` was written for a file: one module, flat dummies, constants in
the same file. A unit of a *model tree* -- a CESM component, a land or ocean
model -- differs in four ways that are not about any one model, and the
engine answers each of them mechanically. What it does not have is a table
of its own: which modules hold constants, which are framework stubs, what
the framework answers, how the tree spells the driver's range. Those arrive
from a domain extension, which is where knowledge of a tree belongs.

| the unit… | the engine… | where |
|---|---|---|
| use-imports constants from sibling modules | resolves each one from the tree into the candidate's own `<module>_use_constants.py`, so the candidate imports nothing that is not in its own files and the constant is the same parsed expression on both sides of the gate | `recast.transform.numpy.tree` |
| imports framework modules the frontend stubs | writes a **stand-in** file per stub module the emitted header imports: the module's initialized entities resolved from the tree, a `_Record` per derived-type variable, and whatever the extension's `framework` table says a standalone run answers | `recast.transform.numpy.standins` |
| calls into siblings | **bundles** the companions' translations into the candidate, recursively, so a call reaches the sibling's translation rather than a stand-in | `recast.transform.numpy.tree` |
| takes a derived-type object with a few hundred pointer components | **flattens** the interface: one plan, two adapters (below) | `recast.fortran.flatten`, `recast.oracle.flat`, `recast.transform.numpy.flat` |
| aborts on generated inputs (an energy-balance check against state it did not compute) | **records** the inputs and outputs of a real run for `dump-replay` | `recast.oracle.record` |
| has run-control variables set by a namelist | rewrites the resolved constant under `constant_overrides`, keeping the source line as a comment and the override on the candidate | `recast.transform.numpy.tree` |

## Flattening a derived-type interface

f2py's wrapper cannot spell `type(model_type), intent(inout) :: inst`, and
the differential gate cannot sample an object. The frontend, with
`flatten=True`, stores a **plan** per public subroutine that takes such a
dummy on `Facts.extra["flat_plans"]`:

* which components of which object the body touches and whether it writes
  them -- from the engine's own read/write analysis, with `associate`
  aliases resolved back to `object%component`, and transitively through
  every call the object is passed to (and every procedure passed as an
  actual and called back with it);
* each component's type, rank and allocation bounds, from the type's
  definition and the `allocate (this%…)` statements in its module --
  parameters folded, the driver's range (`begp:endp`) spelled `1:np_`, a
  bound over a module *variable* kept symbolic and bound to that variable's
  flat name;
* the plain module variables the body reads that the run sets -- a table
  read from a file, a namelist value -- passed like a component under
  `<module>__<name>`.

The plan is a flat signature in the engine's interface vocabulary, and the
three consumers read the one analysis rather than each redoing it:

* `recast.oracle.flat` -- `f2py-golden-flat` -- writes the Fortran
  `<name>_flat`: allocate the object to its original bounds, copy the flat
  arrays in, set the module state, call the original, copy the written
  components back. It also compiles the unit's whole `use` closure into one
  static library and hands f2py only the adapter module, because handing
  f2py the siblings makes it build a module-variable object for every one
  with public variables, and one it cannot build is a NULL that segfaults
  the interpreter on import. Subprograms the wrapper still cannot spell (a
  procedure dummy) are listed on the handle as **ungated**, and
  `differential.bitexact` carries the list into its verdict.
* `recast.transform.numpy.flat` writes the Python `<name>_flat` into the
  emitted module, beside its `_SIGNATURES` entry: build the same object out
  of the same arrays, set the same module state, call the translation,
  return the written arrays.
* `recast.oracle.record` writes a Fortran **recorder** module with one
  probe per adapted subprogram and a **probed copy** of the tree in which
  every call is bracketed by the probes. Building and running the probed
  copy is the case's business; the dumps it writes are what `dump-replay`
  reads, and `differential.bitexact` types the recorded inputs by the
  signature that declares them.

What is *not* claimed: the adapter allocates every touched component over
the one extent the driver chooses and the model's fixed layer counts, with
values the gate generated. That is a kernel-in-isolation test, and the
recording is the model's own operating point for the calls that were
recorded -- neither is a column run.

## The extension's half

`FlatConventions` (frontend) and `TreeConventions` (transform) are the whole
of what an extension supplies: kind assumptions, constants and stub
modules, the stub tables, the `framework` stand-in table, the compiler
profile, the frontend that analysed the unit, and the spelling of the
driver's range (`patch_count`, `bounds_pattern`, `counter_prefix`). Two
frontend options are assumptions about the tree rather than facts of it,
and both are recorded in `Facts.provenance`: `derived_intent_out_as_inout`
(a tree that reads through an `intent(out)` object relies on its pointer
components surviving, which every compiler in practice allows) and
`buffer_out_arrays="all"` (every `intent(out)` array is the caller's
storage). `static.rwset` takes `waive_stub_blocks` to waive, by name in the
verdict, a block whose every emitted statement is a framework-stub marker.
