# RecastEngine

**The modernization engine of [SciRecast](https://github.com/a85tract/SciRecast).**

LLM agents do the labor of modernizing legacy scientific software; RecastEngine
is the part that is reusable across every such effort, and the part that refuses
to let anything through that has not been checked against the original.

> **Status: pre-alpha.** The plugin contract and the CLI's introspection surface
> exist. Transforms, oracles, and verifiers land in P2–P4 — see
> [`docs/roadmap.md`](docs/roadmap.md).

## Four workloads, one spine

RecastEngine is not a translator. Translation is one recipe of four, and all
four are the same five steps with different plugins in the slots:

```
discover  ->  analyze  ->  transform  ->  verify  ->  record
  Unit        Facts        Candidate     Verdict    Evidence
```

| Recipe | What it does | Example product |
|---|---|---|
| `translate` | Fortran → NumPy / Numba / CUDA, by deterministic rules | CESM-language-translator |
| `refactor` | carve a Python control plane into a Fortran monolith, numerics untouched | [freeCAM](https://github.com/a85tract/freeCAM) |
| `port` | retarget a kernel to an accelerator | CESM-jax-kernels |
| `audit` | secret scan, SBOM+CVE+VEX, LLM source audit, sanitizer builds | CC-Test (cyber half) |

## Quick start

```bash
uv sync --extra dev
uv run recast doctor      # check the installation
uv run recast recipes     # what workloads exist
uv run recast plugins     # what is registered
```

`recast plan` dry-runs a recipe and reports what is missing, before anything
costs compute:

```console
$ uv run recast plan translate --config '{"target": "numba"}'
 1. [ok ] executor     local
 2. [ok ] frontend     fortran
 3. [MISS] transform    translate.numba
 4. [MISS] verifier     static.rwset                 gate
 5. [MISS] oracle       f2py-golden
 6. [MISS] verifier     differential.bitexact        gate
 7. [opt] verifier     symbolic.notary              optional
 8. [ok ] store        fs-evidence
```

Those `MISS` lines are the pre-alpha status, not a configuration error. The core
installs with zero dependencies and stays importable without a compiler, a GPU,
or a model provider; everything heavier is an extra (`fortran`, `verify`,
`numba`, `jax`, `agents`).

To extend it, implement one of the ten interfaces in
[`src/recast/plugins/`](src/recast/plugins/) and register an entry point —
see [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md).

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | the spine, the ten interfaces, where the boundaries fall |
| [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md) | how to extend the engine |
| [`docs/roadmap.md`](docs/roadmap.md) | phases P0–P6 |
| [`conformance/`](conformance/) | what a plugin must satisfy |

## Contact

| | |
|---|---|
| Bugs, features | issues in this repository |
| Vulnerabilities | [`SECURITY.md`](SECURITY.md) — private advisory, never a public issue |
| Collaboration, licensing | **Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu> |

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).
Improving the engine improves it for everyone using it — the plugin contract is
the same one every extension uses, so a plugin you write is not second-class.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
