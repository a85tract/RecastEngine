# RecastEngine

**The modernization engine of [SciRecast](https://github.com/a85tract/SciRecast).**

LLM agents do the labor of modernizing legacy scientific software; RecastEngine
is the part that is reusable across every such effort, and the part that refuses
to let anything through that has not been checked against the original.

## Install

Python 3.11 or newer. Activate the environment once and every command after
that is a bare `recast`.

```bash
uv venv --python 3.11                              # fetches a 3.11 if you have none
source .venv/bin/activate                          # Windows: .venv\Scripts\activate
uv pip install -e ".[fortran,translate,verify]"
```

Without `uv`: `python3 -m venv .venv` and `pip install -e` with the same
extras. The core itself has zero dependencies and stays importable without a
compiler, a GPU, or a model provider — `fortran`, `translate`, `verify`,
`numba`, `jax`, `agents` and `all` are the extras.

Extras install optional runtime dependencies; the catalog separately describes
the artifact boundary each engine implements. The shipped catalog contains
`recast.fortran-python.numpy`, `recast.python-numpy.numba`, and
`recast.python-numpy.jax`. The latter two use an independent stdlib-AST Python
frontend, the untouched source module as their oracle, and backend-specific
numerical gates. Install `.[numba,verify]` or `.[jax,verify]` to execute them. See
[`docs/translation-engines.md`](docs/translation-engines.md).

```console
$ recast doctor
recast 0.0.1.dev0  python 3.11.16
```

The remaining `doctor` output is a live plugin count and registry inventory;
installing an out-of-tree plugin changes it without changing RecastEngine.

## Your first run

The shipped example, all the way through. Needs a `gfortran` on PATH — the
reference really is compiled:

```console
$ recast run translate examples/toy_physics --config examples/toy_physics/recast.json
fortran:toy_physics
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy
  [ok ] verifier   static.rwset                sampled: 4 blocks match
  [ok ] oracle     f2py-golden                 f2py:toy_physics:3f3e0f78...
  [ok ] verifier   differential.bitexact       bit_exact: 85 points across 2 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

One command, eight stages: the module translated, its dataflow cross-checked
against the source's, the untouched Fortran compiled as the reference, every
output compared bit for bit. Nothing is written into the source tree — it all
lands in `output/`, under the project's own name:

| | |
|---|---|
| `output/toy_physics/translate/fortran_toy_physics/candidate/` | the generated Python, every block carrying the source lines it came from |
| `output/toy_physics/evidence/fortran_toy_physics/*.json` | one manifest per verdict — artifact digest, oracle key, metrics |

`toy_physics` is the project's name; `fortran_toy_physics` is the one unit it
declares — the `fortran:toy_physics` printed at the top of the run, with the
colon made path-safe. A tree that declares several units gets one such
directory each.

For the same thing on code this project did not write, [`corpus/`](corpus/)
pins twelve open-source Fortran libraries and runs the engine over them with no
domain extension installed.
[`docs/corpus-numfor-example.md`](docs/corpus-numfor-example.md) walks one of
those units through the same eight stages, and says how far its passing run
reaches.

## The commands

| | |
|---|---|
| `recast doctor` | version, interpreter, how much is registered |
| `recast recipes` | the six shipped recipes |
| `recast plugins` | what is registered, by kind |
| `recast engines [--json]` | immutable translation-engine manifests for an outer pipeline builder |
| `recast plan <recipe>` | the stages a run would walk, and which slots nothing fills |
| `recast run <recipe> <tree>` | walk them over a source tree |
| `recast version` | the version alone |

The six recipes, the config keys, the flags on `run` and what its exit codes
mean: [`docs/cli.md`](docs/cli.md).

## Documentation

| | |
|---|---|
| [`docs/`](docs/) | the CLI, the architecture, writing a plugin, the roadmap |
| [`examples/`](examples/) | source trees the shipped recipes run over end to end |
| [`corpus/`](corpus/) | twelve third-party Fortran libraries the engine alone is held to, and the record of how far it gets |
| [`conformance/`](conformance/) | what a plugin must satisfy, as a suite that runs |

## Contact

| | |
|---|---|
| Bugs, features | issues in this repository |
| Vulnerabilities | [`SECURITY.md`](SECURITY.md) — private advisory, never a public issue |
| Collaboration, licensing | **Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu> |

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
