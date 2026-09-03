# probe_kernel

Two "programs" that print what an instrumented C kernel prints -- a `PASS`
line and the probe lines of the `gate.h` protocol (`GATE:SUM`, `GATE:STAT`)
-- as shell scripts, so the `executable-golden` oracle, the
`differential.probes` gate and the `performance.benchmark` verifier can be
exercised on a machine with no C compiler and no GPU. `golden/` is the
reference, `cand/` the candidate that agrees with it. The conformance suite
and `tests/test_c_kernels.py` use them; nothing here is a benchmark.
