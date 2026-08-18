# Examples

Each directory is a source tree a shipped recipe runs over end to end, with
the operator config beside it. They are the public form of the check the
roadmap names: the recipe has to work *here*, on sources anyone can read,
not only on the private corpus it was migrated against.

    recast run translate examples/toy_physics --config examples/toy_physics/recast.json

`toy_physics` needs a Fortran compiler and the f2py build backend
(`pip install 'recast-engine[fortran,translate,verify]'` and a `gfortran` on
PATH). The run translates the module, cross-checks its dataflow, compiles
the untouched Fortran as the reference, compares every output bit for bit,
and writes the evidence manifests under `.recast/evidence/`.
