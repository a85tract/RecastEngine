# Conformance suite

What a plugin must satisfy to be usable, and what an out-of-tree extension must
pass to be an extension rather than a fork.

**Status: five kinds — `Executor`, `Oracle`, `Verifier`, `EvidenceStore`,
`FindingStore` — and the cross-cutting rules run; five kinds are still
specification.** The tables below are the whole contract, and each row says
whether it is executable yet. Where a check exists, a plugin that fails it fails
the suite. Where one does not, the row is a claim nothing verifies -- which is
what this entire file was before the harness landed, and the reason the
distinction is now marked rather than left to the reader.

An undeclared kind is reported as **unexercised, not passed**. pytest skips a
check whose parameter set is empty, and that skip is the honest answer: the
suite was not given anything to check.

## Declaring your plugins

Most kinds cannot be checked from the plugin alone. Asking whether a Verifier
fails closed means handing it a candidate, an oracle and a workspace, and only
you know what a valid one looks like. So you declare a `PluginSet` -- a list of
cases, each carrying the least material its checks need:

```python
# yourpkg/conformance.py
from pathlib import Path

from recast.conformance import ExecutorCase, FindingStoreCase, PluginSet

PLUGIN_SET = PluginSet(
    name="your-extension",
    executors=(ExecutorCase(name="pbs", unsatisfiable={"nodes": 4096}),),
    finding_stores=(FindingStoreCase(name="sec-track", build=lambda root: YourStore(root)),),
)
```

```toml
[project.entry-points."recast.conformance"]
your-extension = "yourpkg.conformance:PLUGIN_SET"
```

`recast/conformance/builtin.py` is the engine's own set, and the worked example
worth reading before writing yours. Everything a case can declare is documented
on the dataclasses in `recast/conformance/__init__.py`.

## Running it

```bash
uv run pytest conformance/ --plugin-set your-extension
```

Three forms of the name are accepted, in this order: `recast` (the engine's own
set, always available), an entry-point name from the `recast.conformance` group,
or a dotted path such as `yourpkg.conformance:PLUGIN_SET` for a set you are
still developing and have not installed.

Release your plugin only on a green run against the engine minor it targets.
Engine minors do not break a plugin that satisfied the previous minor; that is
the SemVer promise everything out-of-tree rests on.

## Per-kind checks

| Kind | Must hold | Checked |
|---|---|---|
| `Frontend` | `discover` is deterministic and side-effect free; re-running on unchanged source yields identical `Unit` sets; `preprocess` records its flags in `Facts.provenance` | not yet |
| `Transform` | `applicable` never raises; unhandled sites land in `deferred`, not exceptions; a `deterministic` Transform yields an identical `Candidate.digest()` for identical inputs, while a `deterministic = False` one instead records model, prompt digest, and sampling parameters in `Candidate.notes` so its Evidence replays to a valid artifact | not yet |
| `Oracle` | `key` is stable for unchanged inputs and changes for every perturbation the case declares — flags, rank count, wrapped surface — and when the source moves; the ref is filed under the key `key` reports; a refusing executor produces a `RecastError`, not an exception the runner does not catch; `release` is idempotent | `test_oracle.py` |
| `Verifier` | a good candidate earns the verdict its case declares; a broken candidate produces `FAILED`; an unavailable oracle produces `FAILED`, never a weaker pass; an executor that refuses the requested scale produces `FAILED`, not a retry at a smaller one; `Verdict.candidate` is the digest that was judged; `metrics` on any real comparison, `detail` on any failure | `test_verifier.py` |
| `Scanner` | findings default to `EMBARGOED`/`PLAUSIBLE`; a scan of a clean tree yields nothing; a scan of the seeded fixture yields the seeded defect | not yet |
| `Adjudicator` | can return `REFUTED` — the suite feeds it a known false positive and requires it to be killed | not yet |
| `Executor` | refuses resources it cannot honestly satisfy; `wait` is idempotent; `cancel` on an unknown handle is a no-op, not a crash | `test_executor.py` |
| `EvidenceStore` | append-only: one URI never comes to denote two different documents, whether by addressing altered content separately or by refusing to write it; output validates against CC-Test `evidence-manifest.v1` | `test_evidence_store.py` |
| `FindingStore` | `guard` rejects above `max_access`, and `put` calls it; storage is not group- or world-readable | `test_finding_store.py` |
| `AgentProvider` | `AgentResult.model` reports the model that actually answered, including after a fallback | not yet |

`Frontend` and `Transform` are the ones to do next: both have an in-tree
implementation to hold, which `Scanner`, `Adjudicator` and `AgentProvider` do
not. Writing checks for a kind with no plugin anywhere produces checks nothing
runs, and this file already carries enough of those; those three are better
written beside the first implementation of each.

**The `Oracle` row grew a clause, and the suite found the hole it describes.**
`errors.py` has declared `OracleUnavailable` -- "the reference could not be
materialized" -- since P1, and nothing had ever raised it. `run.py` catches
`RecastError` around an oracle stage and marks that unit failed; `f2py-golden`
let the executor's refusal out as a bare `RuntimeError`, which escapes that
handler and ends the whole run. One build that could not be scheduled took
every other unit's verdict with it. The fix is in `oracle/f2py.py` and the
regression is pinned in `tests/test_f2py_oracle.py`; what is worth keeping is
how it was found, which is that the rule was written before its subject was
examined.

**The `Verifier` row grew two clauses while being implemented.** The first is
the good candidate. Every other clause in that row is satisfied by a Verifier
that returns `FAILED` unconditionally, so the suite requires a case to show a
verdict being *earned* before it will believe a verdict being refused; a gate
that never passes anything is as useless as one that never fails anything, it
just fails in a direction nobody complains about.

The second is what "`metrics` is populated on both outcomes" turned out to mean.
Taken literally it fails the engine's own `differential.bitexact`, which returns
`{}` on the paths where nothing was compared at all -- no numpy, no compiled
module on the oracle's handle, a candidate that does not import. That is not the
defect the rule is about. There are no numbers when no comparison ran, and
inventing `{"compared": 0}` to satisfy a checker is ceremony. So the rule splits:
`metrics` must carry the numbers whenever a comparison actually happened, and
every failing verdict -- including the ones where nothing ran -- must carry a
`detail` saying what was seen. A `FAILED` with neither is the real hazard, and
that is now what is caught.

**One row moved while being implemented.** The `EvidenceStore` rule used to read
"re-`put` of altered content under an existing key is rejected", which a
content-addressed store cannot satisfy as written: altered content never lands
under an existing key, so there is nothing to reject. The property that was
meant is the one now stated -- a URI never comes to denote two documents -- and
both designs satisfy it, one by construction and one by refusing.

## Cross-cutting

| Rule | Checked |
|---|---|
| **No domain imports in the core.** | `tests/test_contract.py::test_core_imports_no_domain_packages` |
| **Every recipe has a gate.** A recipe with no gating stage cannot produce trustworthy evidence. | `test_recipes.py` |
| **No stage is both `gate` and `optional`.** An optional gate is not a gate. | `test_recipes.py` |
| **An agentic Transform needs a hard gate.** A recipe containing a `deterministic = False` Transform must have a gating Verifier that awards `BIT_EXACT` or an explicit tolerance. An LLM-backed Transform emits plausible output for the cases the rules refuse, so only execution against the Oracle separates a correct translation from a plausible-but-wrong one. | `test_recipes.py` |
| **Plans are reproducible.** Same config → same stage list, or evidence cannot be replayed. | `test_recipes.py` |
| **A recipe declares the executor it needs, from config.** Any recipe with an `oracle` or `verifier` stage declares exactly one `executor` stage, first in the list, and takes its name from config — a hardcoded scheduler name is a site leak in a public recipe. | `test_recipes.py` |
| **A failed gate does not drive a retry.** No `Verdict` reaches a `Transform`, and no stage re-runs because a later one failed. The suite runs a recipe whose gate always fails and requires exactly one `Transform.apply` call per Unit. | `test_runner.py` |
| **A failed gate is still recorded.** A gate that failed and was recorded is audit trail; one that failed and vanished is a rumor. | `test_runner.py` |
| **Execution goes through the Executor.** No `Oracle` or `Verifier` may route around it. | `test_oracle.py`; `test_verifier.py` has no in-tree subject — see below |

## What the suite deliberately does not check

**"No `subprocess` in an Oracle or a Verifier" is not a grep.** The engine's own
`f2py-golden` oracle calls `subprocess` in `_compiler_version`, and it is right
to: asking a compiler its version is a metadata query that has to happen *before*
anyone decides whether to build, because the answer goes into the cache key. A
static rule would have flagged the reference implementation, and the fix would
have been to weaken the rule with a waiver rather than to state it correctly.
The rule is behavioural instead: hand the plugin `RefusingExecutor` from
`recast.conformance.doubles` and require a `FAILED` verdict rather than a result.

That check runs against `f2py-golden`, which compiles through the executor and
is the only in-tree plugin that submits a job at all. **No in-tree Verifier
does**, which is a finding rather than an oversight: `static.rwset` parses the
emitted Python, `symbolic.notary` samples expressions, and
`differential.bitexact` imports the candidate and calls it in this process
alongside whatever the oracle handed over. All three take an `Executor` and none
uses it. So the Verifier half of this rule waits for P5's batch-backed gate. A
case declares `submits_jobs` to opt in, and the suite skips the check by name
rather than passing it, so the day a Verifier does execute the check is already
waiting for it.

**"Two materializations under one key are behaviourally identical" is not
checked, because for the one oracle here it cannot be.** `f2py-golden` imports
its compiled module by name, and the second import in a process returns the
first one out of `sys.modules` -- so the two observations would be the same
object and the check would pass without having compared anything. A case
declares no observation hook today; the honest version needs either a
subprocess or an oracle whose handle is not a module, and both belong with the
plugin that first needs them.

**One perturbation the `f2py-golden` case does not declare: the compiler.** Its
key folds `gfortran --version`, which is the right thing to fold, and moving it
means having a second working toolchain on the machine. Pointing `fc` at one
that is not installed does not produce a different key, it produces a
`ConfigError` -- also correct, and not a test of this rule. So the case declares
the perturbations it can express and this one waits for a machine with two
compilers.

**`submit` must not block on completion.** Real, and not checkable from outside:
an executor that finishes the job during `submit` is indistinguishable from one
that queued it and got lucky, and the timing test that would tell them apart is
a flake generator. Left to review.

**Reading evidence back.** `EvidenceStore.get` and `.query` are part of the ABC
but the suite only requires `put`, because a case declares `read_manifest` to
show the suite what was written. A store may therefore pass while `get` raises
`NotImplementedError` — as the in-tree filesystem store's does. That is a real
gap in the reference implementation, and naming it here is preferable to a check
that would pass it anyway.
