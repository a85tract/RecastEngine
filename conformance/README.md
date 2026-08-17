# Conformance suite

What a plugin must satisfy to be usable, and what an out-of-tree extension must
pass to be an extension rather than a fork.

**Status: specified, not implemented.** The checks below are the contract; the
harness lands in P2 alongside the first real plugins.

## Per-kind checks

| Kind | Must hold |
|---|---|
| `Frontend` | `discover` is deterministic and side-effect free; re-running on unchanged source yields identical `Unit` sets; `preprocess` records its flags in `Facts.provenance` |
| `Transform` | `applicable` never raises; unhandled sites land in `deferred`, not exceptions; a `deterministic` Transform yields an identical `Candidate.digest()` for identical inputs, while a `deterministic = False` one instead records model, prompt digest, and sampling parameters in `Candidate.notes` so its Evidence replays to a valid artifact |
| `Oracle` | `key` changes when compiler, flags, source, or rank count change; two materializations under one key are behaviourally identical; `release` is idempotent |
| `Verifier` | a broken candidate produces `FAILED`; an unavailable oracle produces `FAILED`, never a weaker pass; an executor that refuses the requested scale produces `FAILED`, not a retry at a smaller one; `metrics` is populated on both outcomes |
| `Scanner` | findings default to `EMBARGOED`/`PLAUSIBLE`; a scan of a clean tree yields nothing; a scan of the seeded fixture yields the seeded defect |
| `Adjudicator` | can return `REFUTED` — the suite feeds it a known false positive and requires it to be killed |
| `Executor` | refuses resources it cannot honestly satisfy; `wait` is idempotent; `cancel` on an unknown handle is a no-op, not a crash |
| `EvidenceStore` | append-only: re-`put` of altered content under an existing key is rejected; output validates against CC-Test `evidence-manifest.v1` |
| `FindingStore` | `guard` rejects above `max_access`; storage is not group- or world-readable |
| `AgentProvider` | `AgentResult.model` reports the model that actually answered, including after a fallback |

## Cross-cutting

- **No domain imports in the core.** Enforced by
  `tests/test_contract.py::test_core_imports_no_domain_packages`.
- **Every recipe has a gate.** A recipe with no gating stage cannot produce
  trustworthy evidence.
- **No stage is both `gate` and `optional`.** An optional gate is not a gate.
- **An agentic Transform needs a hard gate.** A recipe containing a
  `deterministic = False` Transform must have a gating Verifier that awards
  `BIT_EXACT` or an explicit tolerance. An LLM-backed Transform emits plausible
  output for the cases the rules refuse, so only execution against the Oracle
  separates a correct translation from a plausible-but-wrong one.
- **Plans are reproducible.** Same config → same stage list, or evidence cannot
  be replayed.
- **Execution goes through the Executor.** No `Oracle` or `Verifier` may call
  `subprocess`, `os.system`, or a scheduler client directly. The suite passes a
  refusing executor and requires the plugin to surface that as `FAILED` rather
  than route around it.
- **A recipe declares the executor it needs, from config.** Any recipe with an
  `oracle` or `verifier` stage declares exactly one `executor` stage, first in
  the list, and takes its name from config — a hardcoded scheduler name is a
  site leak in a public recipe.
- **A failed gate does not drive a retry.** No `Verdict` reaches a `Transform`,
  and no stage re-runs because a later one failed. The suite runs a recipe whose
  gate always fails and requires exactly one `Transform.apply` call per Unit.

## Running it against your plugins

```bash
uv run pytest conformance/ --plugin-set <name>
```

Release your plugin only on a green run against the engine minor it targets.
Engine minors do not break a plugin that satisfied the previous minor; that is
the SemVer promise everything out-of-tree rests on.
