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

First time? [`docs/getting-started.md`](docs/getting-started.md) walks from
an empty machine to a verified translation of your own module, with what
every command prints.

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
| Collaboration, licensing | **Yueqi Chen** — <yueqi.chen@colorado.edu> |

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
