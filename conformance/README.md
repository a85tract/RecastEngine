# Conformance suite

What a plugin must satisfy to be usable, and what an out-of-tree extension must
pass to be an extension rather than a fork.

**Status: specified, not implemented.** The checks below are the contract; the
harness lands in P2 alongside the first real plugins.

## Per-kind checks

| Kind | Must hold |
|---|---|
| `Frontend` | `discover` is deterministic and side-effect free; re-running on unchanged source yields identical `Unit` sets; `preprocess` records its flags in `Facts.provenance` |
| `Transform` | `applicable` never raises; unhandled sites land in `deferred`, not exceptions; identical inputs yield an identical `Candidate.digest()` |
| `Oracle` | `key` changes when compiler, flags, source, or rank count change; two materializations under one key are behaviourally identical; `release` is idempotent |
| `Verifier` | a broken candidate produces `FAILED`; an unavailable oracle produces `FAILED`, never a weaker pass; `metrics` is populated on both outcomes |
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
- **Plans are reproducible.** Same config → same stage list, or evidence cannot
  be replayed.

## Running it against your plugins

```bash
uv run pytest conformance/ --plugin-set <name>
```

Release your plugin only on a green run against the engine minor it targets.
Engine minors do not break a plugin that satisfied the previous minor; that is
the SemVer promise everything out-of-tree rests on.
