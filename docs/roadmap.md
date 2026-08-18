# Roadmap

Phases, not dates. Each one has a check that says whether it is done.

## P0 — decisions (open)

| Decision | Options | Status |
|---|---|---|
| Contributor agreement | DCO (lightweight) vs CLA (allows relicensing later) | open |
| PyPI name | `recast-engine` — `recast` is taken by an unrelated 2021 package | **decided**, not yet uploaded |
| What never goes public | Sec-Track content, PBS D41 research, `cpg_audit/` entries | open |

## P1 — scaffold (this commit)

Plugin contract, model types, registry, `local` executor, filesystem stores, the
four recipes as declarations, CLI introspection, CI, hygiene gate, licence.

**Done when:** `recast doctor`, `recast recipes`, and `recast plan` run with zero
optional dependencies installed, and `test_contract.py` passes.

## P2 — migrate the translator

Move CESM-language-translator's `pipeline/` (22 modules, ~10k lines) in with
history via `git filter-repo` path rewrite, refactoring as it lands:

- `translate.py` (2,883 lines) splits into `rules/` + `backend/numpy`. It is
  currently parser, rule library, and emitter at once, which is why nothing else
  can reuse any part of it.
- `extract_interface` / `extract_constants` / `chunk` / `resolve_use` become
  `recast-fortran`'s `Frontend`. **Landed** as `recast.fortran`, behind the
  `fortran` extra: the analysis is unchanged, but the rendering it used to do
  went with the Transform that has a target language, and the kind table it
  reached into as a module global is now an argument.
- `notary` / `highprec_verify` / `rwset` become `Verifier`s with honest
  `provides` levels.
- `gen_wrapper` + the f2py build become the `f2py-golden` `Oracle`.

**Done when:** the `translate` recipe runs end to end on `examples/` and
reproduces the existing bit-exact result for one scheme, with 408 files' worth
of `/glade` paths gone (`tools/check_hygiene.py` is the check).

## P3 — triage the 664 agent scripts

CESM-Agent-Produced-Scripts, into four buckets:

| Bucket | Destination | Rough count |
|---|---|---|
| reusable, domain-independent | rewritten into engine modules, with tests | 40–60 |
| HPC execution | interface here, implementation in an executor plugin | ~170 |
| kernel implementations (`15_kernel_impl/`) | Product Layer repos — they are ports, not tooling | 83 |
| CESM-specific and one-shot | stay in the archived repository | ~350 |

**Done when:** every promoted script has a test and a home; the rest is tagged
read-only. Nothing is copied in to make the repository look fuller.

## P4 — empty out the domain

Build `recast-cesm` as its own repository and move every CESM-specific rule,
catalog, and golden set into it. Bring freeCAM and CESM-jax-kernels onto the
`refactor` and `port` recipes.

A separate repository rather than a directory in CESM-modernization-overview,
because this phase's check is that the engine passes with `recast-cesm`
*uninstalled* — which is only a real check if it is a separately installable
distribution.

**Done when:** the engine passes its tests with `recast-cesm` uninstalled, and
freeCAM's validation gate runs through `Verifier` rather than its own
`validate_*` scripts. This phase is the only real proof that the engine is
domain-independent.

## P5 — prove the contract out of tree

Build the first extension that lives outside this repository and depends only on
the published contract: PBS/Slurm executors, relay/resume of multi-day runs, and
a restricted `FindingStore` for Sec-Track. It passes `conformance/`.

**Done when:** the engine works without it, and it needed no engine patches. Any
patch it did need is a hole in the contract, and the hole is the finding.

## P6 — public

Scrub → security review → archive the two source repositories read-only →
repoint SciRecast's `.gitmodules` (`RecastEngine` currently resolves to
CESM-language-translator) → flip visibility.

**Irreversible.** Both source repositories are private and carry NCAR paths, a
username, an allocation account, PBS vulnerability research, and CPG audit
entries. Filtering at migration time (P2/P3) rather than before the flip is what
makes this safe, because git history keeps whatever was ever committed.
