"""What the security review concluded on, held by a test.

`docs/security-review.md` is the record; these are the clauses in it that
said "fixed" rather than "deliberately unbounded". Each names the surface it
holds so the review can be re-run by reading this file's names.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from recast.errors import ConfigError
from recast.model import Access, Finding
from recast.store.filesystem import FilesystemFindingStore, _record_name
from recast.verify.bitexact import _arithmetic, _resolve_extent

# --- a finding's uid is data, not a path -----------------------------------------


def _finding(uid: str) -> Finding:
    return Finding(uid=uid, unit="u", scanner="s", title="t", access=Access.EMBARGOED)


def test_a_uid_with_path_separators_stays_inside_the_store(tmp_path: Path) -> None:
    """The uid is built from what the scanner saw, and what it saw came from
    the repository under audit -- a dependency name out of grype, a rule id
    out of a .gitleaks.toml the target supplies itself."""
    store = FilesystemFindingStore(root=tmp_path / "vault" / "findings")
    uri = store.put(_finding("composition:repository:x:CVE-1:../../../../tmp/evil@1"))
    written = Path(url2pathname(urlparse(uri).path))
    assert written.parent == (tmp_path / "vault" / "findings").resolve()
    assert not (tmp_path / "tmp").exists()
    assert json.loads(written.read_text())["uid"].endswith("../../../../tmp/evil@1")


def test_two_uids_that_differ_only_in_replaced_characters_do_not_collide() -> None:
    assert _record_name("a/b") != _record_name("a\\b")
    assert _record_name("a/b") != _record_name("a_b")


def test_a_record_name_is_bounded_and_safe() -> None:
    name = _record_name("x" * 500 + "/../" + "y" * 500)
    assert len(name) < 140
    assert "/" not in name and ".." not in name
    assert name.endswith(".json")


# --- a declared extent is arithmetic, not Python ----------------------------------


def test_an_extent_expression_is_evaluated_as_arithmetic() -> None:
    assert _arithmetic("2*(3+1)") == 8
    assert _arithmetic("-4 + 10 // 3") == -1
    assert _resolve_extent("n*2+1", {"n": 4}) == 9


def test_an_extent_expression_cannot_reach_anything_but_arithmetic() -> None:
    """``eval`` with empty builtins is not safe, and this text comes from a
    declared dimension in the source under verification."""
    for text in (
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('os').system('true')",
        "abs(1)",
        "[1][0]",
    ):
        with pytest.raises(ValueError):
            _arithmetic(text)
    assert _resolve_extent("().__class__", {"default_dim": 3}) == 3


# --- an operator's bar is checked before the run ----------------------------------


def test_a_blocks_on_typo_is_refused_before_any_work(tmp_path: Path) -> None:
    from recast.plugins.recipe import Stage
    from recast.run import _require_valid_bars

    class R:
        name = "r"

    stages = [Stage("scanner", "secret")]
    with pytest.raises(ConfigError, match="blocks_on must be one of"):
        _require_valid_bars(R(), stages, {"stages": {"secret": {"blocks_on": "hgih"}}})
    _require_valid_bars(R(), stages, {"stages": {"secret": {"blocks_on": "high"}}})
    _require_valid_bars(R(), stages, {})


# --- a character initializer cannot escape its literal --------------------------


def test_a_character_initializer_is_a_python_literal_not_python_source() -> None:
    from recast.transform.numpy.modules import _character_literal

    assert _character_literal("'abc'") == "'abc'"
    assert _character_literal('"it\'s"') == repr("it's")
    assert _character_literal("'it''s'") == repr("it's")
    hostile = "\"x'; import os; os.system('id'); x='y\""
    emitted = _character_literal(hostile)
    # One expression, whose value is the attacker's text, and nothing executes.
    import ast

    tree = ast.parse(emitted, mode="eval")
    assert isinstance(tree.body, ast.Constant)
    assert tree.body.value == "x'; import os; os.system('id'); x='y"
