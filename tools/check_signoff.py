#!/usr/bin/env python3
"""Refuse a commit that does not certify its own provenance.

RecastEngine takes contributions under the Developer Certificate of Origin
(``DCO`` at the repository root), not a CLA. The DCO's whole mechanism is one
trailer -- ``Signed-off-by: Name <email>`` -- so a DCO that nothing checks is
not a DCO at all, it is a sentence in CONTRIBUTING.md. This is the check.

The rule is the one every DCO project uses: every non-merge commit carries a
sign-off whose e-mail matches the commit's author. A sign-off in someone else's
name certifies nothing, and merge commits are written by the forge rather than
by a contributor, so they are exempt.

Sign-off began when the DCO was adopted; commits before that are not rewritten,
because rewriting published history to add a certification nobody gave at the
time would be a worse record than none. So this checks a *range* -- what a pull
request adds -- and never the whole history.

Usage:
    python tools/check_signoff.py [<range>]      # default: origin/main..HEAD
"""

from __future__ import annotations

import re
import subprocess
import sys

DEFAULT_RANGE = "origin/main..HEAD"

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


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        return usage()
    rev_range = argv[0] if argv else DEFAULT_RANGE

    bad: list[tuple[str, str, str]] = []
    for sha, subject, author, message in commits(rev_range):
        signers = {m.group("email").strip().lower() for m in SIGNOFF.finditer(message)}
        if author in signers:
            continue
        if signers:
            why = f"signed off by {', '.join(sorted(signers))}, not the author"
        else:
            why = "no sign-off"
        bad.append((sha, subject, f"{why} <{author}>"))

    if not bad:
        return 0

    print(f"{len(bad)} commit(s) in {rev_range} are not signed off:\n", file=sys.stderr)
    for sha, subject, why in bad:
        print(f"  {sha[:9]}  {subject}", file=sys.stderr)
        print(f"             {why}", file=sys.stderr)
    print(
        "\nThe DCO (see ./DCO) is certified by a trailer on each commit. Add it with\n"
        "'git commit -s' from now on; for commits already made, the fix is\n"
        "\n"
        f"    git rebase --signoff {rev_range.split('..')[0]}\n"
        "\n"
        "and a force-push of your branch. Sign-off must carry the author's own\n"
        "name and e-mail -- it is a certification, not a formality.",
        file=sys.stderr,
    )
    return 1


def usage() -> int:
    print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
