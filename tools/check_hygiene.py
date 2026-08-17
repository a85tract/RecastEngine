#!/usr/bin/env python3
"""Refuse to commit anything that ties this repository to one site.

RecastEngine's two source repositories carry a lot of NCAR: 408 files in
CESM-language-translator alone hardcode ``/glade`` paths, a username, an
allocation account, and a scheduler hostname. Those are fine in a private case
repository and fatal in a public engine -- and unlike a bad commit, a leaked
path cannot be taken back once the repository is public and indexed.

So this runs in CI on every push, and the migration in P2/P3 has to satisfy it
file by file rather than as a cleanup pass at the end.

Usage:
    python tools/check_hygiene.py [paths...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# (label, pattern, why it must not appear)
#
# Patterns that match their own source text carry the escape marker, so this
# file stays clean when scanned as a copy -- checking out history into a temp
# directory defeats the SELF check below, and that is exactly when the scan
# matters most.
RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("site-path", re.compile(r"/glade/\S*"), "NCAR filesystem path"),  # hygiene: allow
    ("allocation", re.compile(r"\bUCUB\d{4}\b"), "NCAR allocation account"),
    ("scheduler-host", re.compile(r"@desched\d"), "Derecho scheduler hostname"),
    ("home-path", re.compile(r"/glade/u/home/\w+"), "user home"),  # hygiene: allow
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    ("private-key", re.compile(r"-----BEGIN (RSA|OPENSSH|EC|PGP) PRIVATE KEY"), "private key"),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"), "Anthropic API key"),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}

# This file necessarily contains the patterns it forbids.
SELF = Path(__file__).resolve()


def iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file():
            out.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and not (SKIP_DIRS & set(child.parts)):
                out.append(child)
    return out


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path.cwd()]
    violations: list[str] = []

    for file in iter_files(roots):
        if file.resolve() == SELF:
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; secret scanning proper covers these
        for lineno, line in enumerate(text.splitlines(), 1):
            if "hygiene: allow" in line:
                continue
            for label, pattern, why in RULES:
                match = pattern.search(line)
                if match:
                    violations.append(f"{file}:{lineno}: [{label}] {why}: {match.group(0)[:60]}")

    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        print(f"\n{len(violations)} hygiene violation(s).", file=sys.stderr)
        print(
            "Site-specific values belong in a config file or a case repository, "
            "never in engine source.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
