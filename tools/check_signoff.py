#!/usr/bin/env python3
"""Refuse a commit whose author has not signed the CLA, or that does not certify itself.

RecastEngine takes contributions under a Contributor License Agreement
(``CLA.md`` at the repository root). A contributor signs it once, by adding a
row to ``CLA-SIGNATORIES.md``; every commit after that carries a
``Signed-off-by`` trailer, which is the statement that this commit is
submitted under the agreement and the certification of origin the agreement
folds in (CLA section 4). An agreement nothing checks is a sentence in
CONTRIBUTING.md, so this is the check, and it has two rules:

1. Every non-merge commit carries a sign-off whose e-mail matches the commit's
   author. A sign-off in someone else's name certifies nothing, and merge
   commits are written by the forge rather than by a contributor, so they are
   exempt.
2. Every non-merge commit's author e-mail appears in ``CLA-SIGNATORIES.md``
   as it stands at the head of the range. A first-time contributor adds their
   row in the same pull request, so the file the check reads already has it.

The agreement began when it was adopted; commits before that are not
rewritten, because rewriting published history to add a certification nobody
gave at the time would be a worse record than none. So this checks a *range*
-- what a pull request adds -- and never the whole history.

Usage:
    python tools/check_signoff.py [<range>]      # default: origin/main..HEAD
"""

from __future__ import annotations

import re
import subprocess
import sys

DEFAULT_RANGE = "origin/main..HEAD"
SIGNATORIES = "CLA-SIGNATORIES.md"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SIGNOFF = re.compile(
    r"^Signed-off-by:\s*(?P<name>[^<]+?)\s*<(?P<email>[^>]+)>\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# git renders %x1f as byte 0x1f in its *output*; the format string itself stays
# printable, because a NUL cannot be passed through argv.
SEP = "\x1f"


def git(*args: str) -> str:
    """Run git, or exit with its complaint on one line rather than a traceback."""
    try:
        done = subprocess.run(  # noqa: S603 -- arguments are fixed by this file
            ["git", *args],  # noqa: S607 -- git from PATH, as every other git hook does
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        sys.exit("check_signoff: git is not on PATH")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        sys.exit(f"check_signoff: git {' '.join(args)}: {detail[0] if detail else 'failed'}")
    return done.stdout


def commits(rev_range: str) -> list[tuple[str, str, str, str]]:
    """(sha, subject, author e-mail, message) for each non-merge commit in range."""
    shas = git("rev-list", "--no-merges", rev_range).split()
    out = []
    for sha in shas:
        raw = git("show", "-s", "--format=%s%x1f%ae%x1f%B", sha)
        subject, email, message = raw.split(SEP, 2)
        out.append((sha, subject, email.strip().lower(), message))
    return out


def signatories(head: str) -> set[str]:
    """Every e-mail in the signatories file as it stands at ``head``."""
    try:
        text = git("show", f"{head}:{SIGNATORIES}")
    except SystemExit:
        return set()
    rows = [line for line in text.splitlines() if line.startswith("|")][2:]  # past the header
    return {m.group(0).lower() for row in rows for m in EMAIL.finditer(row)}


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        return usage()
    rev_range = argv[0] if argv else DEFAULT_RANGE
    head = rev_range.split("..")[-1] or "HEAD"
    signed = signatories(head)

    unsigned: list[tuple[str, str, str]] = []
    strangers: list[tuple[str, str, str]] = []
    for sha, subject, author, message in commits(rev_range):
        signers = {m.group("email").strip().lower() for m in SIGNOFF.finditer(message)}
        if author not in signers:
            if signers:
                why = f"signed off by {', '.join(sorted(signers))}, not the author"
            else:
                why = "no sign-off"
            unsigned.append((sha, subject, f"{why} <{author}>"))
        if author not in signed:
            strangers.append((sha, subject, f"author <{author}> is not in {SIGNATORIES}"))

    if not unsigned and not strangers:
        return 0

    if unsigned:
        print(f"{len(unsigned)} commit(s) in {rev_range} are not signed off:\n", file=sys.stderr)
        for sha, subject, why in unsigned:
            print(f"  {sha[:9]}  {subject}", file=sys.stderr)
            print(f"             {why}", file=sys.stderr)
        print(
            "\nThe CLA (see ./CLA.md) is asserted by a trailer on each commit. Add it with\n"
            "'git commit -s' from now on; for commits already made, the fix is\n"
            "\n"
            f"    git rebase --signoff {rev_range.split('..')[0]}\n"
            "\n"
            "and a force-push of your branch. Sign-off must carry the author's own\n"
            "name and e-mail -- it is a certification, not a formality.\n",
            file=sys.stderr,
        )
    if strangers:
        print(
            f"{len(strangers)} commit(s) in {rev_range} have an author who has not signed the CLA:",
            "",
            sep="\n",
            file=sys.stderr,
        )
        for sha, subject, why in strangers:
            print(f"  {sha[:9]}  {subject}", file=sys.stderr)
            print(f"             {why}", file=sys.stderr)
        print(
            f"\nRead ./CLA.md, then add a row for yourself to ./{SIGNATORIES} in this pull\n"
            "request, listing every e-mail you author commits with. That row, in a\n"
            "signed-off commit, is your signature.",
            file=sys.stderr,
        )
    return 1


def usage() -> int:
    print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
