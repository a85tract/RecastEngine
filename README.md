# RecastEngine

**The modernization engine of [SciRecast](https://github.com/a85tract/SciRecast).**

LLM agents do the labor of modernizing legacy scientific software; RecastEngine
is the part that is reusable across every such effort, and the part that refuses
to let anything through that has not been checked against the original.

> **Status: pre-alpha.** The plugin contract, the CLI, and the whole
> `translate` recipe exist and run end to end — Fortran in, verified NumPy
> and evidence manifests out, gated bit-exact against the compiled original.
> The other three recipes have their contracts and await their plugins
> (P3–P4) — see [`docs/roadmap.md`](docs/roadmap.md).

## Four workloads, one spine

RecastEngine is not a translator. Translation is one recipe of four, and all
four are the same five steps with different plugins in the slots:

```
discover  ->  analyze  ->  transform  ->  verify  ->  record
  Unit        Facts        Candidate     Verdict    Evidence
```

| Recipe | What it does | Example product |
|---|---|---|
| `translate` | Fortran → NumPy / Numba / CUDA | [PyCAM5](https://github.com/a85tract/PyCAM5) |
| `refactor` | carve a Python control plane into a Fortran monolith, numerics untouched | [freeCAM](https://github.com/a85tract/freeCAM) |
| `port` | retarget a kernel to an accelerator | [JaxCAM6](https://github.com/a85tract/CESM-jax-kernels) |
| `audit` | secret scan, SBOM+CVE+VEX, LLM source audit, sanitizer builds | CC-Test (cyber half, restricted access) |

## Quick start

```bash
uv sync --extra fortran --extra translate --extra verify --extra dev
uv run recast doctor      # check the installation
uv run recast recipes     # what workloads exist
```

Then run the whole `translate` recipe over the shipped example (needs a
`gfortran` on PATH — the reference really is compiled):

```console
$ uv run recast run translate examples/toy_physics       --config examples/toy_physics/recast.json
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

One command, eight stages: the module is translated, its dataflow
cross-checked against the source's, the untouched Fortran compiled as the
reference, every output compared bit for bit, and three evidence manifests
written under `examples/toy_physics/.recast/evidence/` — because a candidate
without evidence is a draft, however good it looks. The generated Python
itself lands beside them under `.recast/translate/.../candidate/`.

`recast plan` dry-runs a recipe and reports what is missing before anything
costs compute — here, that the Numba backend is not migrated yet:

```console
$ uv run recast plan translate --config '{"target": "numba"}'
 1. [ok ] executor     local
 2. [ok ] frontend     fortran
 3. [MISS] transform    translate.numba
 4. [ok ] verifier     static.rwset                 gate
 5. [ok ] oracle       f2py-golden
 6. [ok ] verifier     differential.bitexact        gate
 7. [ok ] verifier     symbolic.notary              optional
 8. [ok ] store        fs-evidence
```

The core installs with zero dependencies and stays importable without a
compiler, a GPU, or a model provider; everything heavier is an extra
(`fortran`, `translate`, `verify`, `numba`, `jax`, `agents`).

To extend it, implement one of the ten interfaces in
[`src/recast/plugins/`](src/recast/plugins/) and register an entry point —
see [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md).

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | the spine, the ten interfaces, where the boundaries fall |
| [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md) | how to extend the engine |
| [`docs/roadmap.md`](docs/roadmap.md) | phases P0–P6 |
| [`examples/`](examples/) | source trees the shipped recipes run over end to end |
| [`conformance/`](conformance/) | what a plugin must satisfy, as a suite that runs |

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
