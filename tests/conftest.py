"""Test-wide fixtures.

Kept to the one thing every test in this suite needs and none of them should
have to say.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disposable_output(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send every run's ``output/`` somewhere the test run can throw away.

    ``run_recipe``'s default is ``<cwd>/output/<project>``, and the cwd of a
    test is this repository. Without this the suite writes one directory per
    ``run_recipe`` call into the checkout -- which is not a failure any
    assertion catches, only a mess that shows up in ``git status`` later.
    """
    monkeypatch.setenv("RECAST_OUTPUT_HOME", str(tmp_path_factory.mktemp("output")))
