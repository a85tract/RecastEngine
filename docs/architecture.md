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

### Execution is passed in, not reached for

`Oracle.materialize` and `Verifier.verify` take an `Executor` as an argument.
Nothing that leaves the process may use `subprocess` directly.

This is what keeps the two halves separable. A differential verifier's logic —
build both sides, feed them the same inputs, diff the results — is identical on
a laptop and on 512 ranks of a batch system; only where the work lands differs.
Inline the submission and that verifier grows a queue name, an allocation
account, and a site path, which is to say it can no longer be public. The seam
is why `tools/check_hygiene.py` can be a hard gate on this repository while the
real scheduler plugins live outside it.

Taking it as a parameter rather than pulling it from the registry also puts
"this plugin executes things" in the signature, and lets a test substitute a
recording or refusing executor without the plugin's cooperation.

Which executor gets passed is declared by an `executor` stage, first in the
list — it is not a step the engine walks but the ambient choice everything else
inherits, so a recipe that materializes an oracle or awards a verdict has to
name one. It names it *through config*, never literally: `pbs-<site>` is site
knowledge and the four shipped recipes have to stay publishable. `refactor`
goes further and rejects the default outright, because its gate is a pinned
multi-rank run that `local` cannot finish — better a failed `recast plan` in a
second than a failed build an hour in.

### A gate stops the Unit; it does not drive a retry

`Stage.gate` means the Unit stops. A `Verdict` never flows back into a
`Transform`, and no stage re-runs because a later one failed.

Retrying would be a no-op where it is safe and unsafe where it would do
something. A `deterministic` Transform reproduces its bytes, so a second attempt
yields the same `Candidate.digest()` — nothing to gain. An agentic Transform
does vary, and that is exactly the case where a loop is dangerous: handing the
gate's own numbers back to the thing being gated turns the Oracle into a fitness
function, and "iterate until the diff is zero on these 512 points" is how a
Candidate gets overfitted to the sample that was supposed to judge it. The wall
between Transform and Verifier is load-bearing precisely because the Transform
cannot see through it.

Two further costs, if the temptation ever returns. `refactor` gates on a
512-rank pinned run, so a loop around that gate is priced in node-hours. And
`Evidence` records no attempt count, so a bit-exact claim reached on attempt 47
would be indistinguishable from one reached on attempt 1 — a multiple-comparisons
problem the manifest has no way to disclose.

Iteration belongs in three places instead, all of them outside the gate:
`Candidate.deferred` handed from a rule Transform to an agentic one, a
Transform's own loop against cheap checks before it emits, and improving the
rules out of band, where one fix amortizes across the corpus.

## Two placements of the LLM

A `Transform` may be rule-driven or it may consult an LLM, and the engine treats
these as two placements of the same slot rather than two kinds. The `Transform`
carries a `deterministic` flag; an agentic Transform sets it `False` and reaches
the model through the `AgentProvider` boundary, so which model answers is a
swappable plugin, not a hardcoded dependency.

The two placements answer different questions, and a recipe picks per stage:

| | rule Transform (`deterministic = True`) | agentic Transform (`deterministic = False`) |
|---|---|---|
| the LLM is | out of the per-unit path — it improves the rules out of band | in `apply`, translating the units the rules refuse |
| per-unit output | reproducible bit for bit | varies across runs |
| pays off when | a construct recurs across a large corpus, so one rule amortizes | units are few or novel and no rule exists yet to amortize |

They compose through `deferred`. A rule Transform handles the bulk and lists what
it could not do; an agentic Transform consumes that list and attempts it; both
emit a `Candidate` into the **same** gate, which judges them without knowing
which produced what.

**The agentic placement makes the wall higher, not lower.** A Transform never
judges its own output; with an LLM in the loop this is doubly load-bearing,
because a model that both writes and grades will pass its own hallucination. Two
disciplines follow, and `conformance/` enforces both:

- **Reproducibility is by provenance, not by digest.** An LLM does not reproduce
  its bytes even at temperature zero across provider versions. A
  `deterministic = False` Transform records the model, prompt digest, sampling
  parameters, and the sites it filled in `Candidate.notes`, so its Evidence
  replays to a *valid* artifact rather than to the same bytes. The plan stays
  reproducible — the stage list does not change — only the artifact does.
- **An agentic Transform is only safe under a hard gate.** It emits plausible
  output for exactly the cases the rules refuse, and a plausible wrong answer is
  invisible to every counting metric — only execution against the Oracle catches
  it. A recipe with a `deterministic = False` Transform must gate on a Verifier
  that awards `BIT_EXACT` or an explicit tolerance.

The agentic Transform's remit is the novel *local* construct — the language-level
gap a rule has not yet closed. It does not extend to runtime-mechanism design
(what `pbuf` means in the target), which cannot be inferred from a local
translation failure and stays a human `L5`-style decision. That boundary is the
same one the "Rule packs as data" open question circles below.

## Engine, extension, product

RecastEngine produces nothing on its own, and that is deliberate. A product is
what comes out when the engine is combined with the knowledge a particular
effort needs and the source it is modernizing:

```
  RecastEngine        the spine: model, contract, registry, recipes
+ recast-fortran      language knowledge -- the reference frontend, in-tree
+ recast-cesm         domain knowledge -- CESM rules, catalogs, golden sets
+ a PBS executor      site knowledge -- private, never public
+ refactor + config   which recipe, which reference commit, how many ranks
+ CAM's source        the thing being modernized
--------------------------------------------------------------------------
= freeCAM             a product
```

The last two lines are what make this a production rather than a composition.
The engine and its extensions are machinery; the legacy source is input;
freeCAM, PyCAM5, and CESM-jax-kernels are outputs. Point the same machinery at
different source and it produces a different product — which is the only reason
the abstraction is worth what it costs.

**Extensions have no visibility requirement.** Entry points do not know whether
the package declaring them is public. A private `recast-cesm`, a private site
executor, and a public engine install and compose identically. That is what lets
an effort keep its filesystem paths, its allocation account, and its embargoed
findings out of the public engine without giving up any capability.

**In-tree and out-of-tree extensions are both extensions.** The difference is
where the code ships, not what it is permitted to do:

| Extension | Ships | Why there |
|---|---|---|
| `recast-fortran` | in-tree, under `recast.` | the reference frontend; the engine has to be useful with nothing else installed |
| `recast-cesm` | its own repository | P4's check is that the engine passes with it *uninstalled*, which only means something if it is a separable distribution |
| site and scale plugins | wherever their owner keeps them | schedulers, cross-cluster routing, restricted finding stores |

`recast-fortran` shipping in-tree is a packaging decision, not permission to
bypass the contract. It registers through the same entry points as everything
else, so swapping in a different frontend takes no engine change.

## Where the boundaries fall

**Domain.** Nothing in `recast.*` imports numpy, sympy, numba, jax, anthropic,
or netCDF4. `tests/test_contract.py::test_core_imports_no_domain_packages`
enforces it by walking the package. Domain knowledge lives out of tree in
`recast-cesm`; language knowledge lives in `recast-fortran`, which ships in-tree
as the reference frontend.

**In-tree vs plugin.** The engine ships what one person needs on one machine:
the full frontend/rule/verification stack, the NumPy/Numba/JAX/CUDA backends,
and the `local` executor. Scale and operations arrive as plugins — batch
schedulers, cross-cluster routing, relay/resume of multi-day runs, multi-agent
orchestration with budget control, restricted finding stores. They register
through these same ABCs, so they are extensions and not forks, and
`conformance/` is what they have to pass.

The boundary is drawn so the engine is independently useful, not a demo: a
researcher can modernize and bit-exactly verify a scheme with nothing but this
repository.

## Open questions

- **Whether a Scanner takes an Executor too.** `Oracle` and `Verifier` now do;
  `Scanner.scan` does not, deliberately and for now. A `needs_build` scanner
  runs sanitizer builds and fuzz harnesses — `audit` declares a `dynamic.asan`
  stage — so it has the same claim on the seam, and leaving it out is the one
  place a plugin can still reach for `subprocess` without contradicting the
  contract. Deferred rather than settled, because the first real scanner should
  say what it actually needs.
- **Sampler as an ABC.** Input generation currently lives inside verifiers
  (Hypothesis strategies, dump selection). If dump-driven and property-driven
  sampling need to compose, it becomes an eleventh kind.
- **Rule packs as data.** `translate.py`'s rules are Python today. If they become
  declarative, an agent can synthesize and a notary can verify a rule without a
  code review in the loop — but the rule language becomes a public interface.
- **Candidate application.** Ordered `Patch` application against a moving
  upstream is the freeCAM pain point; whether the engine owns three-way merge or
  delegates to git is unresolved.
