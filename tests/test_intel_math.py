"""The libimf binding: what it refuses, and what it spells.

The library itself is not on a test machine, which is the case the tests can
cover: a translation under ``ifx`` imports, and says what is missing and
what the operator can do when a number is asked for. Whether libimf's
``exp`` matches ifx's is measured on a machine that has both, by the gates.
"""

from __future__ import annotations

import pytest

from recast.transform.numpy import intel_math, vocabulary


def _no_libimf() -> bool:
    try:
        intel_math.library()
    except RuntimeError:
        return True
    return False


@pytest.mark.skipif(not _no_libimf(), reason="a libimf is present; the refusal cannot be seen")
def test_a_missing_libimf_is_refused_at_the_first_call_and_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RECAST_LIBIMF", raising=False)
    monkeypatch.delenv("CTP_LIBIMF", raising=False)
    with pytest.raises(RuntimeError) as refused:
        intel_math.exp(1.0)
    assert "RECAST_LIBIMF" in str(refused.value)
    assert "gfortran" in str(refused.value)


def test_every_spelling_in_the_intel_tables_is_something_this_module_defines() -> None:
    for table in (vocabulary.INTEL_SCALAR, vocabulary.INTEL_ARRAY):
        for target in table.values():
            module, _, name = target.partition(".")
            assert module == "intel_math", target
            assert callable(getattr(intel_math, name)), target


def test_the_intel_tables_override_intrinsics_the_plain_tables_already_spell() -> None:
    """An override of a name the backend never emits is dead text. ``fabs``
    is carried from the pipeline's table as it stands; it is not a Fortran
    intrinsic and nothing reaches it."""
    assert set(vocabulary.INTEL_SCALAR) - set(vocabulary.ELEMENTAL_SCALAR) == {"fabs"}
    assert set(vocabulary.INTEL_ARRAY) <= set(vocabulary.ELEMENTAL_ARRAY)
