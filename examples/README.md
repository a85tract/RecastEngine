# Examples

Each directory is a source tree a shipped recipe runs over end to end, with
the operator config beside it. They are the public form of the check the
roadmap names: the recipe has to work *here*, on sources anyone can read,
not only on the private corpus it was migrated against.

    recast run translate examples/toy_physics --config examples/toy_physics/recast.json \
        --summary examples/toy_physics/verification.json

`toy_physics` needs a Fortran compiler and the f2py build backend
(`pip install 'recast-engine[fortran,translate,verify]'` and a `gfortran` on
PATH). The run translates the module, cross-checks its dataflow, compiles
the untouched Fortran as the reference, compares every output bit for bit,
and writes the evidence manifests under `.recast/evidence/`.

Two records come out of that, and they are for different readers.

`.recast/evidence/` holds the **manifests**: one immutable, content-addressed
CC-Test document per verdict per run, append-only, carrying the full metrics
and the environment. That is the audit trail, and it accumulates a file per
attempt -- including the attempts that failed, which is the point of an audit
trail and the reason it is not committed.

`verification.json` is the **current state**: one entry per unit and verifier,
regenerated rather than appended. It carries the confidence, the artifact
digest, the oracle's cache key, and the countable metrics, and deliberately
omits wall-clock time and paths -- so two runs over the same revisions produce
the same bytes. That is what makes it worth committing: it diffs like a
lockfile, and a change in it is a change in what has been verified.
[`toy_physics/verification.json`](toy_physics/verification.json) is the one
this example produces, checked in so that a reader can see the claim without
owning a Fortran compiler.
