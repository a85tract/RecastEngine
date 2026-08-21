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
digest, the oracle's *name*, and the countable metrics, and deliberately omits
wall-clock time and paths -- so two runs over the same revisions produce the
same bytes. That is what makes it worth committing: it diffs like a lockfile,
and a change in it is a change in what has been verified.

The oracle's **cache key** is deliberately not among those, and the reason is
what makes the file diffable at all. The key folds the compiler's version, so
recording it would make the summary a fact about the machine: this example's
committed bytes come from gfortran 16 and CI's come from whatever the runner's
distribution ships, and the byte comparison would fail there while nothing was
wrong. The key belongs in the evidence manifest, which is a record of one run
and is not compared to anything. What survives into the summary is only what
two correct runs agree on.
[`toy_physics/verification.json`](toy_physics/verification.json) is the one
this example produces, checked in so that a reader can see the claim without
owning a Fortran compiler.

## The same module, ported

`toy_physics` also runs the `port` recipe, over the same sources and the same
sampling config:

    recast run port examples/toy_physics --config examples/toy_physics/port.json \
        --summary examples/toy_physics/port-verification.json

This one needs `jax` (`pip install 'recast-engine[fortran,translate,jax]'`) and
**no Fortran compiler** — which is the anchoring decision showing through rather
than a convenience. The reference is `numpy-anchor`: the NumPy translation of
the same unit, re-derived from the same Facts, so nothing in this run builds
Fortran. That makes the port's claim a chain — NumPy bit-exact against the
Fortran above, JAX ULP-bounded against the NumPy here — and the honest part is
that this run cannot check the first link. The run above is what checks it.

The verdict is `ulp_bounded` rather than `bit_exact`, and that is the ceiling
rather than a shortfall: XLA's transcendentals are not libm's. On this module —
which has none — 76 of 85 points land bit-exact and the remaining nine within
1 ULP, against a gate of 32.

[`toy_physics/port-verification.json`](toy_physics/port-verification.json) is
checked in for the same reason as its bit-exact sibling, with one difference in
how much it is allowed to prove. A ULP count is not device-independent — XLA's
CPU backend does not promise the same last bit on x86 as on arm64, and the
summary records `candidate_device` and `reference_device` so a reader can see
which machine produced it. So CI runs this example and gates on the verdict, but
does not diff the file the way it diffs `verification.json`. Treat a changed ULP
count here as a question, not an alarm.
