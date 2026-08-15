# RecastEngine

**The modernization engine of [SciRecast](https://github.com/a85tract/SciRecast) — framework and basic functionality, with no domain knowledge in it.**

Legacy scientific software gets modernized by LLM agents doing the labor under
human oversight. RecastEngine is the part that is reusable across every such
effort: it discovers units of work, analyzes them, transforms them, and — the
part that matters — refuses to let anything through that has not been checked
against the original.

> **Status: pre-alpha.** The plugin contract and the CLI's introspection
> surface exist. The transforms, oracles, and verifiers land in P2–P4 (see
> [`docs/roadmap.md`](docs/roadmap.md)).

## Four workloads, one spine

RecastEngine is not a translator. Translation is one recipe out of four, and
they are all the same five steps with different plugins in the slots:

```
discover  ->  analyze  ->  transform  ->  verify  ->  record
  Unit        Facts        Candidate     Verdict    Evidence
```

| Recipe | What it does | Oracle | Abstracted from |
|---|---|---|---|
| `translate` | Fortran → NumPy / Numba / CUDA, by deterministic rules | compiled f2py truth module | CESM-language-translator |
| `refactor` | carve a Python control plane into a Fortran monolith, numerics untouched | pinned full-model run | [freeCAM](https://github.com/a85tract/freeCAM) |
| `port` | retarget a kernel to an accelerator | captured production dumps | CESM-jax-kernels |
| `audit` | secret scan, SBOM+CVE+VEX, LLM source audit, sanitizer builds | — produces findings | CC-Test (cyber half) |

`recast plan <recipe>` prints the stages and tells you which plugins are missing,
before anything costs compute.

## The two rules the design enforces in code

**A transform never judges its own output.** `Transform` produces a `Candidate`;
only a `Verifier` produces a `Verdict`; only a gated `Verdict` yields `Evidence`.
Confidence is a stated level — `SAMPLED`, `TOLERANCED`, `ULP_BOUNDED`,
`BIT_EXACT`, `SYMBOLIC` — not a boolean, and a verifier that cannot run its
comparison returns `FAILED`, never a weaker pass.

**An embargoed finding cannot reach a public store.** The `audit` recipe
produces unpatched vulnerabilities. `Finding` defaults to `EMBARGOED`, stores
declare a `max_access` ceiling, and `FindingStore.guard()` raises
`AccessViolation` on any write above it. This is a code check, not a convention,
because the failure mode is not recoverable by deleting the log afterwards. See
[`SECURITY.md`](SECURITY.md).

## Where the pieces live

```
RecastEngine            this repository — framework, plugin contract, local execution
  └── recast-cesm       CESM domain plugin, in CESM-modernization-overview
  └── RecastRuntime     commercial extension: batch schedulers, multi-agent
                        orchestration, Sec-Track integration, multi-tenant ops
```

Everything in `src/recast/` is written against the ABCs in
[`src/recast/plugins/`](src/recast/plugins/). Nothing in the core imports
Fortran, CESM, JAX, or a scheduler — [`tests/test_contract.py`](tests/test_contract.py)
asserts that mechanically, so the claim stays true rather than aspirational.

Correctness evidence is emitted in
[CC-Test](https://github.com/a85tract/CESM-CC-Test)'s `evidence-manifest.v1`
schema. The engine does not define a competing format; CC-Test owns it and this
is a producer.

## Install

```bash
uv sync --extra dev
uv run recast doctor
uv run recast plan translate --config '{"target": "numba"}'
```

The core installs with zero dependencies and stays importable without a
compiler, a GPU, or a model provider. Everything heavier is an extra:
`verify`, `numba`, `jax`, `agents`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).
Security issues: [`SECURITY.md`](SECURITY.md) — never a public issue.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
