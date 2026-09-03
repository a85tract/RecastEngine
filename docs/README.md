# docs/

| | |
|---|---|
| [`getting-started.md`](getting-started.md) | start here if you have a Fortran file and do not live in a terminal: install, run the example, translate your own module, read a run that is not all green |
| [`reading-the-evidence.md`](reading-the-evidence.md) | the two records a run leaves, field by field in plain words: the summary to commit, the manifests to open, and what a passing run does not claim |
| [`case-clm-ml.md`](case-clm-ml.md) | a real model under the engine: a 76-module canopy model taken to NumPy bit-exact for a month and to JAX within the paper's band, what it took, what the gates found, where the claim stops |
| [`cli.md`](cli.md) | every command, the six recipes, config keys, `run` flags, exit codes |
| [`corpus-numfor-example.md`](corpus-numfor-example.md) | two units of one corpus case walked through the recipe, one passing and one not, and how far the passing one reaches |
| [`corpus-lapack-example.md`](corpus-lapack-example.md) | the one corpus unit whose Python does not import -- a library over LAPACK -- why that is the right outcome, and what an externals shim would change |
| [`architecture.md`](architecture.md) | the spine, the ten interfaces, where the boundaries fall |
| [`tree-units.md`](tree-units.md) | a unit of a model tree: use-constants, stand-ins, bundled companions, derived-type interfaces flattened, a real run recorded -- and the extension's half of it |
| [`translation-engines.md`](translation-engines.md) | immutable engine manifests and the JSON catalog used by outer pipeline builders |
| [`phased-execution.md`](phased-execution.md) | canonical CandidateBundle boundary and transform-independent verification reports |
| [`observing-a-run.md`](observing-a-run.md) | the read-only run event boundary for UIs, audit stores, and outer orchestration |
| [`writing-a-plugin.md`](writing-a-plugin.md) | how to extend the engine, and the conformance suite a plugin must satisfy |
| [`roadmap.md`](roadmap.md) | phases P0–P6, where each recipe stands, and how far each claim's evidence reaches |
| [`disclosure-ledger.md`](disclosure-ledger.md) | what stays private, case by case, with the reason and the mechanism holding it |
| [`security-review.md`](security-review.md) | the boundary review P6 requires, against the engine's own surfaces |
| [`splitting-the-translator.md`](splitting-the-translator.md) | the inventory the P2 migration works from, and the boundary it draws |
