# Architecture

## Why the abstraction is not "translator"

RecastEngine started as a Fortran→Python translator. It is now driving three
structurally different modernization efforts, and a fourth workload that is not
modernization at all:

| Project | What it actually does |
|---|---|
| CESM-language-translator | rule-driven Fortran → NumPy/Numba/CUDA, gated bit-exact against f2py |
| freeCAM | generates C-interop adapters + 16 ordered source patches that carve a Python control plane into iCESM1.3.1, gated on a 512-rank 50-step bit-for-bit run |
| CESM-jax-kernels | rewrites CLUBB/MG/Kessler/RTE-RRTMGP kernels for JAX, gated against captured Fortran dumps |
| CC-Test (cyber half) | secret scan, SBOM+CVE+VEX, LLM source audit, ASan — findings to Sec-Track |

The first three share a shape that survives the differences, and freeCAM's own
`tools/` directory is the clearest evidence for it: `capture_*`/`extract_*` are
analysis, `*_codegen.py`/`apply_*_patches` are transformation, eleven
`validate_*` tools plus a `validation/` evidence tree are the gate. Same spine,
different plugins.

```
discover  ->  analyze  ->  transform  ->  verify  ->  record
  Unit        Facts        Candidate     Verdict    Evidence
                                              \
  scan  ->  adjudicate  ->  Finding  -------->  Sec-Track (embargoed)
```

The fourth workload branches at `verify`: cyber testing produces `Finding`s
about defects that exist regardless of any modernization, and those flow to a
different store under different access rules.

## The types

Defined in [`src/recast/model.py`](../src/recast/model.py). Small on purpose —
every field one of the four workloads does not need is a field that will drift.

| Type | Meaning |
|---|---|
| `Unit` | an addressable piece of software: module, subprogram, scheme, component |
| `Facts` | what analysis learned: interface, constants, callgraph, effects, provenance |
| `Candidate` | a proposed change — new files plus ordered patches. Not yet trusted |
| `OracleRef` | a materialized reference implementation, cached by behavioural key |
| `Verdict` | a comparison result with a `Confidence` level and the numbers behind it |
| `Evidence` | an immutable record, rendered as a CC-Test `evidence-manifest.v1` |
| `Finding` | a security defect, with CWE, exploitability, disclosure state, access class |

### Confidence is a level, not a boolean

```
FAILED  <  SAMPLED  <  TOLERANCED  <  ULP_BOUNDED  <  BIT_EXACT  <  SYMBOLIC
```

A `Verifier` declares the strongest level it can ever award. This lets a recipe
compose a cheap static check, a sampled differential test, and an expensive
full-model gate without any of them being mistaken for the others — and it makes
"verified" a claim with a number attached rather than a green checkmark.

Failing closed is mandatory. A verifier whose build failed or whose scheduler
rejected the job returns `FAILED`. Returning `SAMPLED` because nothing
disagreed-with-nothing is how a validation system starts lying.

### Access is enforced, not documented

`Finding.access` defaults to `EMBARGOED`; stores declare `max_access`;
`FindingStore.guard()` raises `AccessViolation` above the ceiling. `Access` and
`Disclosure` are separate axes because they move independently — a finding can
be `CONFIRMED` for months while still embargoed, and only `PUBLISHED` plus
`PUBLIC` makes it publishable.

## The plugin contract

Ten kinds, in [`src/recast/plugins/`](../src/recast/plugins/):

| Kind | Responsibility | Must not |
|---|---|---|
| `Frontend` | source → Units and Facts | write source |
| `Transform` | Facts → Candidate | judge its own output |
| `Oracle` | materialize the reference behaviour | see the Candidate |
| `Verifier` | Candidate vs Oracle → Verdict | pass when it could not run |
| `Scanner` | find defects → Findings | fix them; classify below `EMBARGOED` |
| `Adjudicator` | verify and reclassify a Finding | never refute anything |
| `Executor` | run Jobs | silently downgrade requested resources |
| `EvidenceStore` | append-only correctness record | permit overwrite |
| `FindingStore` | restricted vulnerability record | accept above its ceiling |
| `AgentProvider` | the LLM boundary | misreport which model answered |

`Recipe` composes the rest. It contains no logic — it declares which plugin
fills each slot, which is why adding a workload does not change the core.

Registration is `importlib.metadata` entry points, one group per kind
(`recast.transforms`, `recast.oracles`, …). Discovery is failure-isolated: a
broken third-party plugin becomes unavailable, it does not break `recast --help`.

## Where the boundaries fall

**Domain.** Nothing in `recast.*` imports numpy, sympy, numba, jax, anthropic,
or netCDF4. `tests/test_contract.py::test_core_imports_no_domain_packages`
enforces it by walking the package. Domain knowledge lives in `recast-cesm`;
language knowledge lives in `recast-fortran`.

**Open source vs commercial.** The engine ships enough to translate, port, and
verify one kernel on one machine: the full frontend/rule/verification stack, the
NumPy/Numba/JAX/CUDA backends, and the `local` executor. RecastRuntime adds
scale and operations — batch schedulers, cross-cluster routing, relay/resume of
multi-day runs, multi-agent orchestration with budget control, Sec-Track
integration, multi-tenant ops. It plugs in through these same ABCs, so it is an
extension and not a fork. The `conformance/` suite is what a Runtime must pass.

The boundary is drawn so the open-source engine is independently useful, not
crippled: a researcher can modernize and bit-exactly verify a scheme without any
commercial component. What is sold is scale, compliance, and continuous
operation.

## Open questions

- **Sampler as an ABC.** Input generation currently lives inside verifiers
  (Hypothesis strategies, dump selection). If dump-driven and property-driven
  sampling need to compose, it becomes an eleventh kind.
- **Rule packs as data.** `translate.py`'s rules are Python today. If they become
  declarative, an agent can synthesize and a notary can verify a rule without a
  code review in the loop — but the rule language becomes a public interface.
- **Candidate application.** Ordered `Patch` application against a moving
  upstream is the freeCAM pain point; whether the engine owns three-way merge or
  delegates to git is unresolved.
