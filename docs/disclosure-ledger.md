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
| 5 | Unpatched findings from the `audit` recipe | restricted store: `FindingStore.guard()` on the record, and a findings root outside any checkout | Coordinated disclosure. `EMBARGOED` by default, and nothing reaches a public store or CI log before `PUBLIC` **and** `PUBLISHED`. **Amended 2026-08-21, because the mechanism named only half of itself.** The access class was never the likely accident, and `FilesystemFindingStore`'s own docstring said so: the accident is "an embargoed finding written into a repository checkout that later gets pushed". The only check in front of that looked at directory *permissions* — which a `0700` directory inside a repository passes, and which is exactly what the runner's default produced, at `<project>/.recast/findings`, in a repository whose `.gitignore` did not mention `.recast/`. Nothing had been written there, because until this week the runner never walked a scanner stage; the row was correct only by accident. Now: the default resolves outside the project (`~/.recast/findings/<project>-<key>`, or `RECAST_FINDINGS_HOME`), the store refuses any root inside a git working tree no matter who chose it, and `conformance/test_finding_store.py` checks the half its own docstring had been asserting in prose. `.recast/` was added to `.gitignore` the same day, and **this row does not rest on it**: an ignore rule is one line anyone can delete and it holds nothing back from `git add -f`. It is there so a run does not litter `git status`, which is a different problem. | P1, `SECURITY.md`, amended 2026-08-21 |
| 6 | `08_cpg_tools/` — the Code Property Graph toolchain, 42 files | stays behind | A static-analysis toolchain aimed at finding defects in software this project does not own. Publishing the tooling publishes the aim. | P3 triage |
| 7 | `14_pbs_security/` — PBS D41 security research, 8 files | stays behind | Vulnerability research against a scheduler running in production at a real site. Same rule as row 5, applied to work that predates the engine. | P3 triage |
| 8 | The name of the private CESM extension repository | hygiene `private-repo` | **Re-decided 2026-08-20, and half of the original reason is void.** It was settled on two grounds: that naming a private repository publishes that it exists, and that every mention is a link which 404s for the reader. The first no longer holds — the SciRecast site is public, its Pages build is live, and its `engine.md` names the extension in prose, twice. The project publishes the fact itself, so the engine withholding it protects nothing. What survives is the second ground, which is a documentation-quality concern rather than a disclosure control: until the extension is public, every mention here is a dead link, and "the domain extension" is the more accurate phrase anyway since nothing in the contract is specific to one package. The pattern stays on that ground and **retires when the extension goes public**. | 2026-08-19, re-decided 2026-08-20 |
| 9 | Whether that same name in git history has to be rewritten out | none — nothing left to hold | **Examined and cleared, 2026-08-20.** Six commits' content and six commit messages carry the name, reaching back to the first commit, and row 8's pattern holds only the working tree. The case was open because removing a string from history needs `git filter-repo`, and a rewrite from the first commit changes every hash in the repository — including `6333399`, which the SciRecast site cites twice. It is cleared rather than executed, because the thing a rewrite would hide is already published by the project's own public site (see row 8). Rewriting every hash to conceal a fact the reader can look up is cost without a benefit. If the extension's name ever does become sensitive, the first action is the public site and not this history. | 2026-08-20 |

Rows 6 and 7 are P3's 50-file `EXCLUDE` bucket in full, as that phase's triage
table records it. They stay in the archived private repository, which is the
whole reason that repository is archived read-only rather than deleted.

That repository is a student's, not this project's, so archiving it is agreed
with its author rather than decided here. The mechanism does not rest on the
agreement holding. "Stays behind" is enforced by nobody migrating those two
directories, and every route from private material into this repository runs
through a maintainer — so the control is one this project exercises, and the
archiving is what preserves the *evidence* rather than what supplies the
protection.

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

None. The one case that was open — the extension's name in git history — is
settled as row 9, cleared rather than executed.

Row 9 is the general shape of a problem worth stating once: a
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
