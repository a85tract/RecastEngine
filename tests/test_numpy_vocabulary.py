"""Tests for the backend's name tables.

Mostly invariants rather than examples. A table of 79 entries is not worth
asserting entry by entry, but the relationships between the tables are worth
holding: they are what went wrong when one file held all of them and nothing
checked that the halves agreed.
"""

from __future__ import annotations

import keyword

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.fortran import intrinsics
from recast.transform import profiles
from recast.transform.numpy import runtime, vocabulary


def test_every_name_this_backend_spells_is_one_the_frontend_knows() -> None:
    """The two halves of the same fact. That ``sqrt`` is an intrinsic lives in
    the frontend; that it becomes ``math.sqrt`` lives here. An entry here with
    no counterpart there is a name the read/write analysis would report as a
    variable read -- which is exactly how ``transpose``, ``minloc`` and
    ``cshift`` came to be counted as variables at seven sites in CAM.
    """
    spelled = (
        set(vocabulary.ELEMENTAL_SCALAR)
        | set(vocabulary.ELEMENTAL_ARRAY)
        | set(vocabulary.REDUCTIONS)
        | vocabulary.ARRAY_TRANSFORM
    )
    assert spelled <= intrinsics.ALL, sorted(spelled - intrinsics.ALL)


def test_every_array_variant_has_a_scalar_one() -> None:
    """The array table is a set of overrides, not a second vocabulary. An
    intrinsic that appears only there has no spelling for a scalar argument."""
    assert set(vocabulary.ELEMENTAL_ARRAY) <= set(vocabulary.ELEMENTAL_SCALAR)


def test_every_runtime_shim_named_in_a_table_exists() -> None:
    """A table entry pointing at a ``_f_`` name that the runtime does not
    define emits a file that imports nothing and fails at first call."""
    named = {
        target
        for table in (
            vocabulary.ELEMENTAL_SCALAR,
            vocabulary.ELEMENTAL_ARRAY,
            vocabulary.REDUCTIONS,
        )
        for target in table.values()
        if target.startswith("_f")
    }
    assert named <= set(vars(runtime)), sorted(named - set(vars(runtime)))


def test_targets_are_either_a_shim_a_module_call_or_a_builtin() -> None:
    """Nothing else can appear: the emitted file imports ``math`` and ``np``,
    carries the runtime, and has the builtins."""
    for table in (
        vocabulary.ELEMENTAL_SCALAR,
        vocabulary.ELEMENTAL_ARRAY,
        vocabulary.REDUCTIONS,
    ):
        for source, target in table.items():
            ok = target.startswith(("_f", "math.", "np.")) or target in dir(__builtins__)
            assert ok or target in ("abs", "int", "len", "max", "min", "ord", "chr", "complex"), (
                f"{source} -> {target}"
            )


def test_the_literal_whitelist_is_the_frontend_s_and_not_a_copy() -> None:
    """It was duplicated: the same three integers and three reals written out
    in the frontend and again at the top of the emitter. Whichever one someone
    edited, the other would have gone on disagreeing silently."""
    from recast.fortran import constants

    assert vocabulary.WHITELIST_INT is constants.WHITELIST_INT
    assert vocabulary.WHITELIST_REAL is constants.WHITELIST_REAL


# --- the renaming rule -------------------------------------------------------


def test_a_python_keyword_gets_a_trailing_underscore() -> None:
    """Fortran has variables called ``in``, ``is`` and ``lambda``."""
    assert vocabulary.pysafe("lambda") == "lambda_"
    assert vocabulary.pysafe("in") == "in_"


def test_a_module_alias_gets_one_too() -> None:
    """A Fortran dummy named ``np`` would shadow NumPy in the translation."""
    assert vocabulary.pysafe("np") == "np_"
    assert vocabulary.pysafe("math") == "math_"


def test_an_ordinary_name_is_left_alone() -> None:
    assert vocabulary.pysafe("temperature") == "temperature"
    assert vocabulary.pysafe("sum") == "sum", "a Python builtin is not a collision"


def test_the_verifier_can_undo_exactly_what_pysafe_does() -> None:
    """These two have to be inverses or the cross-check reports a mismatch on
    every renamed variable. They used to be three separate spellings of the
    rule, in three files, agreeing by coincidence."""
    from recast.verify.rwset import Protocol, _Visitor

    protocol = Protocol(reserved=vocabulary.RESERVED)
    visitor = _Visitor(protocol)
    for name in ["lambda", "in", "is", "np", "math", "os", "copy", "mp", "temperature"]:
        assert visitor.back(vocabulary.pysafe(name), store=False) == name


def test_reserved_names_are_the_ones_the_emitted_file_imports() -> None:
    imported = {line.split()[-1] for line in runtime.REQUIRED_IMPORTS if line.startswith("import")}
    assert imported <= vocabulary.RESERVED | {"np"}
    assert not vocabulary.RESERVED & set(keyword.kwlist), "keywords are handled separately"


# --- compiler profiles -------------------------------------------------------


def test_the_default_profile_exists_and_is_named() -> None:
    assert profiles.DEFAULT in profiles.PROFILES
    assert profiles.PROFILES[profiles.DEFAULT].name == profiles.DEFAULT


def test_the_generic_profile_assumes_neither_behaviour() -> None:
    """Every profile flag turns on a rewrite that matches one compiler and
    mismatches the other, so the safe default is to preserve the source form."""
    generic = profiles.PROFILES["generic"]
    assert not generic.int_pow_expand
    assert not generic.cfold_mpfr


def test_the_two_real_compilers_disagree() -> None:
    """The reason the table exists. gfortran expands integer powers and folds
    constant intrinsics with MPFR; ifx does neither, and a translation matching
    one is failing against the other."""
    gfortran, ifx = profiles.PROFILES["gfortran"], profiles.PROFILES["ifx"]
    assert (gfortran.int_pow_expand, gfortran.cfold_mpfr) != (ifx.int_pow_expand, ifx.cfold_mpfr)
