# Security review

The review P6 names in its arrow chain, done 2026-08-21 against `main` at the
commit that adds this file. `docs/roadmap.md` ("What the security review is")
sets the standard: not a generic audit — this engine compiles, imports and runs
other people's code on purpose — but a review of **boundaries**, where every
surface ends in one of two states: a boundary stated in code with a test that
holds it, or a written decision that it is deliberately unbounded and the
operator is the trust boundary. A surface with no conclusion fails the review.

Two trust classes run through everything below, and naming them is most of the
work:

- **The operator.** Whoever writes the config and runs `recast`. They already
  choose the compiler, the source tree and the executor; a config key that
  reaches an argv gives them nothing they did not have. Operator-supplied
  values are **not** a boundary, and the review says so per surface rather than
  pretending otherwise.
- **The code under examination.** The Fortran being translated, the repository
  being audited, the dependency metadata in it. This is other people's input by
  design, and anything that lets it choose what the engine *does* — rather than
  what the engine *computes* — is a defect.

The distinction is the whole review. Three of the four defects below are the
second class reaching a place only the first should.

## Fixed

| Surface | What was wrong | Fix, and the test that holds it |
|---|---|---|
| **Finding uid → filename** (`store/filesystem.py`) | `root / f"{uid}.json"`. A uid is built from what the scanner saw — a dependency name and version from grype, a rule id from a `.gitleaks.toml` the audited repository supplies itself — so the audited code chose where its own finding was written, including outside the `0700` directory every other check exists to keep it in. | `_record_name`: unsafe characters to `_`, length bounded, a digest of the original uid appended so replaced characters cannot collide. The uid is unchanged inside the record. `test_a_uid_with_path_separators_stays_inside_the_store` |
| **Character initializer → emitted Python** (`transform/numpy/modules.py`) | A Fortran `"..."` constant was emitted verbatim with its quotes swapped. `"it's"` was a syntax error; `"x'; import os; …; x='y"` was a statement sequence in a module the verifier imports. | `_character_literal`: `repr` of the value, which is what `expressions.py` already did for the same constant in an expression. `test_a_character_initializer_is_a_python_literal_not_python_source` |
| **Declared extent → `eval`** (`verify/bitexact.py`) | `eval(text, {"__builtins__": {}}, {})` on a dimension expression from the source under verification. Empty builtins do not make `eval` safe. | `_arithmetic`: an `ast` walk over numbers and the arithmetic operators, `ValueError` on anything else, which the caller already turned into the default extent. `test_an_extent_expression_cannot_reach_anything_but_arithmetic` |
| **`config["blocks_on"]` → `Severity()`** (`run.py`) | A typo was a `ValueError` two scans into the run. Not a boundary — it is the operator's value — but the repository's stated preference is a second, not three stages in. | `_require_valid_bars`, before the walk. `test_a_blocks_on_typo_is_refused_before_any_work` |

Every one of the three boundary defects was in a path written **this week**, or
exercised for the first time this week: the finding store had never received a
real finding until the scanners ran, the extent evaluator had only ever seen
the pipeline's own test corpus. That is the argument for doing this review
again whenever a new kind of input starts flowing, rather than once.

## Deliberately unbounded: the operator is the boundary

Each of these reaches a process or the interpreter, and each is fed by the
operator's config. Recorded so the next reader does not re-find them as
defects.

| Surface | What reaches where | Decision |
|---|---|---|
| `config["fc"]` | The Fortran compiler's name, as argv[0] of `--version` (`_compiler_version`, the one `subprocess` outside the executor, documented as a metadata query that has to run before the cache key exists) and as `FC`/`F90` for f2py. | Operator's. They chose the compiler. |
| `config["fflags"]` | `--f90flags=` to f2py, which shlex-splits it; a `-fplugin=` here loads code at compile time. | Operator's. The same person chose `fc`. |
| `config["extra_sources"]`, `config["wrapper_parameters"]` | Paths compiled into the oracle; integer constants spelled into the generated Fortran wrapper. | Operator's. The wrapper is built from the interface the frontend read plus these; a non-integer parameter fails the build, it does not escape it. |
| `config["gitleaks"]`, `["syft"]`, `["grype"]` | argv[0] of a `Job`, resolved through PATH, after `shutil.which`. | Operator's. `recast plan` reports the resolved name before the run. |
| `config["range"]` | `--log-opts` to gitleaks. | Operator's, and the hook's, where it is computed from the refs git supplies. |
| `config["patches"]`, `config["deferred_handler"]` | Python source emitted verbatim into the candidate (`agentic.py`); a callable consulted during the run. Documented as operator-audited code, by design. `deferred_handler` cannot arrive from JSON or TOML at all. | Operator's. This *is* the mechanism for a human to fill a site the rules refused. |
| `--config` file, `plan --config` string | `json.loads` / `tomllib.loads`. | Safe parsers; no code execution on load. |
| `RECAST_BIN`, `RECAST_FINDINGS_HOME` | The executable the pre-push hook runs; the base directory for embargoed findings. | Environment variables in the operator's own shell. Anyone who can set them can already run anything as that user. |

## Deliberately unbounded: the product

| Surface | Decision |
|---|---|
| `verify/bitexact.py` and `oracle/numpy_anchor.py` import a module by path. | The candidate is what the engine emitted from the operator's source; the anchor is a previous run's evidence the operator pointed at. Importing them is the verification. The boundary is the one above: what the emitter is allowed to write, which is now held by the character-literal fix and by `repr` everywhere else a source string reaches Python. |
| `oracle/f2py.py` imports the compiled extension. | It is the reference. Compiling and loading the operator's Fortran is the oracle's definition. |
| `executors/local.py` | Runs the argv it is given, with the env it is given (or the parent's when none). Refuses multi-rank jobs rather than downgrading them. It is the seam every process crossing goes through, and it is where a sandboxing executor would go. |
| The `AgentProvider` contract | A provider's text can become a deferred site's body, be emitted, and be imported by the verifier — it runs in-process with the operator's privileges, and nothing in the contract sandboxes it. **Stated on the contract now** (`plugins/agent.py`), with the pointer to where less trust would be implemented: the Executor the Verifier is handed. No implementation exists in this repository, and the security-related ones will not (see "Security LLM stays private" in the roadmap). |

## Inherited from the source, and kept

Two places where the **code under audit** influences the audit, both exactly
as `hpc-devsecops` has them, and kept under the rule that the source is right
where the port differs. Recorded because they look like defects on first
reading and are decisions.

- **`.vex/openvex.json` in the audited repository** is passed to grype. The
  repository can suppress its own CVE matches. That is what VEX is for — the
  project's triage of its own dependencies — and the script does the same.
- **`.gitleaks.toml` in the audited repository** is honoured by gitleaks
  (fourth in its precedence, after `--config` and two env vars). The
  repository can allowlist its own secrets. The script passes that file
  explicitly; the engine does not, and gitleaks finds it anyway.

Both mean a hostile repository can make itself look clean to `audit`. An audit
gate was never a defence against a repository that is lying to it on purpose —
it is a defence against mistakes, and a project that ships a VEX waiving its
own Critical is making a statement under its own name. Worth knowing; not a
boundary this engine claims.

## Not a surface

- **Tool output → `Finding` fields** (`sarif.py`, `composition.py`). Titles,
  paths and rule ids from gitleaks and grype are data: stored as JSON, never
  executed, and never printed by the CLI, which prints counts. The one place
  they became a path is the first row of "Fixed".
- **The verification summary.** Excludes findings entirely (a count is a
  statement about embargoed material) and excludes the machine (so its diffs
  mean something). Nothing in it came from outside the engine.

## What this review did not do

It read. It did not fuzz the Fortran frontend, did not run the translator over
hostile input at scale, and did not review `tools/check_hygiene.py` or the CI
scripts, which are about what leaves this repository rather than what enters
the engine. The findings-store and literal defects were found by asking "where
does the audited code's text go" and following it; the same question asked of
`fortran/` — every string a frontend copies out of the source into `Facts` — is
the next pass, and is why "again, whenever a new kind of input starts flowing"
is the closing sentence of the first section rather than a hope.
