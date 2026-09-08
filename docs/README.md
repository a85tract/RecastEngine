# docs/

| | |
|---|---|
| [`getting-started.md`](getting-started.md) | start here if you have a Fortran file and do not live in a terminal: install, run the example, translate your own module, read a run that is not all green |
| [`reading-the-evidence.md`](reading-the-evidence.md) | the two records a run leaves, field by field in plain words: the summary to commit, the manifests to open, and what a passing run does not claim |
| [`cli.md`](cli.md) | every command, the recipes, config keys, `run` flags, exit codes |
| [`corpus-numfor-example.md`](corpus-numfor-example.md) | two units of one corpus case walked through the recipe, one passing and one not, and how far the passing one reaches |
| [`corpus-lapack-example.md`](corpus-lapack-example.md) | the one corpus unit whose Python does not import -- a library over LAPACK -- why that is the right outcome, and what an externals shim would change |
| [`architecture.md`](architecture.md) | the spine, the ten interfaces, where the boundaries fall |
| [`translation-engines.md`](translation-engines.md) | immutable engine manifests and the JSON catalog used by outer pipeline builders |
| [`phased-execution.md`](phased-execution.md) | canonical CandidateBundle boundary and transform-independent verification reports |
| [`observing-a-run.md`](observing-a-run.md) | the read-only run event boundary for UIs, audit stores, and outer orchestration |
| [`writing-a-plugin.md`](writing-a-plugin.md) | how to extend the engine, and the conformance suite a plugin must satisfy |
