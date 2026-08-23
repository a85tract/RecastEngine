"""How a sibling module's names are spelled from this one.

``companion_tables`` derives the alias map from the companions' records; the
case of a parameter follows the companion's constants file, which is the
pipeline's rule and the one a generated module has to agree with to import.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.transform.numpy.translate import companion_tables

RECORD = {
    "subprograms": [{"name": "helper"}],
    "generics": {},
    "types": {},
    "module_parameters": [{"name": "br"}, {"name": "i8"}],
    "module_state": [{"name": "props"}],
}


def _constants(*defined: str) -> dict:
    """A constants record whose classified parameters are ``defined``."""
    return {
        "module_parameters": [
            {
                "name": n,
                "kind": "real",
                "payload": "1.0",
                "init_expr": "1.0",
                "line": 1,
                "base_type": "REAL",
            }
            if n in defined
            else {
                "name": n,
                "kind": "kind",
                "payload": "kind parameter",
                "init_expr": "x",
                "line": 1,
                "base_type": "INTEGER",
            }
            for n in ("br", "i8")
        ]
    }


def test_a_parameter_is_spelled_as_the_companion_constants_file_spells_it() -> None:
    entry = {
        "alias": "_mgu",
        "module_py": "mgu_numpy",
        "record": RECORD,
        "constants": _constants("br"),
    }
    _, _, globals_, _ = companion_tables([entry])
    assert globals_["br"] == "_mgu.BR"  # defined there, upper-case
    assert globals_["i8"] == "_mgu.i8"  # only a SKIPPED line there, so lower-case
    assert globals_["props"] == "_mgu.props"  # state is never renamed


def test_without_the_companion_constants_every_parameter_is_lower_case() -> None:
    entry = {"alias": "_mgu", "module_py": "mgu_numpy", "record": RECORD}
    _, _, globals_, _ = companion_tables([entry])
    assert globals_["br"] == "_mgu.br"


def test_a_rename_overrides_a_global_registered_first() -> None:
    first = {"alias": "_a", "module_py": "a_numpy", "record": RECORD}
    second = {
        "alias": "_b",
        "module_py": "b_numpy",
        "record": RECORD,
        "renames": {"br": "br"},
        "constants": _constants("br"),
    }
    _, _, globals_, _ = companion_tables([first, second])
    assert globals_["br"] == "_b.BR"
