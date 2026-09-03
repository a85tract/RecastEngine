"""The source distribution ships the package and nothing about the project.

``docs/`` holds the disclosure ledger and the roadmap, ``corpus/`` other
projects' sources under their own licences; an sdist that shipped either
would publish what the repository keeps out of a release. hatchling ships
everything not ignored unless told otherwise, so the telling is checked.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("docs", "corpus", ".github", "AGENTS.md")


def test_the_sdist_names_what_it_ships_and_what_it_must_not() -> None:
    build = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]
    sdist = build["targets"]["sdist"]
    include = sdist["include"]
    assert "src" in include and "pyproject.toml" in include and "LICENSE" in include
    for name in FORBIDDEN:
        assert name not in include
        assert name in sdist["exclude"], f"{name} is not excluded from the sdist"
