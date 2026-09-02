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

```console
$ recast doctor
recast 0.0.1.dev0  python 3.11.16
23 plugin(s) registered across 10 kinds
```

First time? [`docs/getting-started.md`](docs/getting-started.md) walks from
an empty machine to a verified translation of your own module, with what
every command prints.

## The commands

| | |
|---|---|
| `recast doctor` | version, interpreter, how much is registered |
| `recast recipes` | the four workloads |
| `recast plugins` | what is registered, by kind |
| `recast plan <recipe>` | the stages a run would walk, and which slots nothing fills |
| `recast run <recipe> <tree>` | walk them over a source tree |
| `recast version` | the version alone |

The four recipes, the config keys, the flags on `run` and what its exit codes
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
