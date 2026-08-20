# Disclosure ledger

What never goes public, one case at a time, with the reason and the mechanism
that holds it.

There is no upfront list, because a list written before the material is in front
of you is a guess. The cases arrive as the work does — P3 moves 662 scripts out
of a private collection, and P6 flips two private repositories to public — so
each is ruled on when it turns up and recorded here.

The cost of that choice is that the record has to be real. "Case by case" with
no ledger is indistinguishable from never having decided, and at P6 nobody could
audit what was ruled and what was merely forgotten. So: **P6 does not flip while
this file has an open case**, and every `EXCLUDE` in P3's triage points at a row
here.

## Three mechanisms, not interchangeable

| Mechanism | Holds | Checked by |
|---|---|---|
| Hygiene pattern | a *string* that must appear nowhere in the tree, ever | `tools/check_hygiene.py`, every push |
| Stays behind | a *file or directory* that is never migrated out of the private repo | the migration manifest — a file not moved cannot leak |
| Restricted store | a *runtime record* the engine produces about someone else's software | `FindingStore.guard()`, `SECURITY.md` |

Picking the wrong one is how something slips. A secret string in a file that
"stays behind" still leaks the moment someone copies that file forward; a
directory kept out of the migration is not protected by a regex that was never
written for it.

Note the third row governs findings the engine *produces at runtime* — that
machinery is specified in `SECURITY.md` and enforced in `recast.plugins.store`,
and it is a different problem from this file's. This ledger is about material
that already exists in the two source repositories.

## Settled

| # | What | Mechanism | Why | Settled |
|---|---|---|---|---|
| 1 | NCAR filesystem paths under `/glade` (408 files in CESM-language-translator) | hygiene `site-path`, `home-path` | An NCAR path is useless to an outside reader and identifies the site and the user. | P1 |
| 2 | The allocation account (`UCUB####`) | hygiene `allocation` | A billable identifier. Publishing it invites use of someone else's compute. | P1 |
| 3 | The scheduler hostname (`@desched#`) | hygiene `scheduler-host` | Names an internal host and the batch system in front of it. | P1 |
| 4 | Credentials of any kind — AWS keys, private keys, API tokens | hygiene `aws-key`, `private-key`, `anthropic-key`, `github-token`; `gitleaks` over full history | A leaked key is not recoverable by a later commit. | P1 |
| 5 | Unpatched findings from the `audit` recipe | restricted store | Coordinated disclosure. `EMBARGOED` by default, and nothing reaches a public store or CI log before `PUBLIC` **and** `PUBLISHED`. | P1, `SECURITY.md` |
| 6 | `08_cpg_tools/` — the Code Property Graph toolchain, 42 files | stays behind | A static-analysis toolchain aimed at finding defects in software this project does not own. Publishing the tooling publishes the aim. | P3 triage |
| 7 | `14_pbs_security/` — PBS D41 security research, 8 files | stays behind | Vulnerability research against a scheduler running in production at a real site. Same rule as row 5, applied to work that predates the engine. | P3 triage |
| 8 | The name of the private CESM extension repository | hygiene `private-repo` | Naming a private repository publishes that it exists, and every mention is a link that 404s for the reader. The engine refers to it by what it is — "the domain extension" — which is also the more accurate word, since nothing in the contract is specific to that one package. | 2026-08-19 |

Rows 6 and 7 are P3's 50-file `EXCLUDE` bucket in full, as that phase's triage
table records it. They stay in the archived private repository, which is the
whole reason that repository is archived read-only rather than deleted.

That the counts have to *reconcile* — 42 and 8 against the triage's own total —
is not bookkeeping. P3's triage briefly said 49, because it inferred that
`08_cpg_tools/gen_stubs.py` was already in the translator's `pipeline/` and
therefore P2's to migrate here. The two files share a filename and nothing else:
one generates Fortran stubs so a static-analysis IR will build, the other
generates Python signatures from an interface dump. A name-match had quietly
moved a static-analysis tool out of a bucket this ledger says stays behind.

So the mechanism in row 6 and row 7 is stronger than "these files are not
migrated": **the two buckets are sealed.** Nothing leaves them by inference —
not by filename, not by content hash, not by a rule that looked right for
another bucket. Only by a decision written here first. The triage enforces it,
and the count reconciling against this file is what catches it if the triage
stops.

## Open

| # | What | Noticed | Still undecided |
|---|---|---|---|
| A | That same name in git history — six commits' content and six commit messages, reaching back to the first commit | 2026-08-19, while settling row 8 | Row 8's mechanism holds the *working tree* going forward. History is a different object: `git filter-repo` is the only thing that removes a string from it, and a rewrite changes every commit hash after the earliest edit — including `6333399`, which the SciRecast site cites by name. Either the rewrite happens before the flip and the citation is updated, or the name stays in history and row 8 is a partial measure. **P6 cannot flip until this is answered.** |

Row A is the general shape of this problem, so it is worth stating once: a
hygiene pattern proves a string is absent from `HEAD`, and says nothing about
whether it was ever committed. `gitleaks` covers full history in the hygiene
workflow, but only for credential-shaped secrets — a repository name, a
directory name, or a sentence is invisible to it. Anything that has to be gone
from *history* needs a rewrite, and a rewrite is its own decision with its own
blast radius.

## Adding a case

When P3 or P6 turns up material that should not go public:

1. Add a row to **Open** the moment it is noticed — before deciding. A case
   that is only in someone's head is the failure mode this file exists for.
2. Name the mechanism from the table above, not just the intent. "Do not
   publish X" is not a mechanism.
3. When the mechanism is actually in place — the pattern is in
   `check_hygiene.py`, the path is off the migration manifest, the record class
   is guarded — move the row to **Settled** with the phase that settled it.
4. If a case is examined and found harmless, it still gets a settled row saying
   so. The record of what was *considered and cleared* is what makes the ledger
   auditable rather than merely long.

A new hygiene pattern needs a test that the scanner catches it. A pattern that
matches nothing, because it was written from memory of the string rather than
from the string, is worse than no pattern: it reads as coverage.
