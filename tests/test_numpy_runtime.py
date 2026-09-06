"""Tests for the runtime a translated module carries.

These functions exist because Python's answer differs from Fortran's, so every
test here is a case where the naive spelling is wrong. They had no tests at
all: the code lived inside a string constant, where nothing could import it.

A wrong answer in any of them is close, plausible, and invisible to every
structural check -- only a bit-exact comparison against the original catches
it, and only on inputs that happen to reach the case. That is what makes these
worth pinning by hand rather than trusting to the differential gate.
"""

from __future__ import annotations

import ast
import math

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[verify]")

import numpy as np

from recast.transform.numpy import runtime

# --- the traps ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.5, 1), (1.5, 2), (2.5, 3), (-0.5, -1), (-2.5, -3), (0.4, 0), (-0.4, 0)],
)
def test_nint_rounds_half_away_from_zero(x: float, expected: int) -> None:
    """Python's ``round`` rounds half to even: ``round(0.5) == 0`` and
    ``round(2.5) == 2``. Fortran's NINT does not, and CAM rounds a lot."""
    assert runtime._f_nint(x) == expected


@pytest.mark.parametrize(("a", "p", "expected"), [(7, 3, 1), (-7, 3, -1), (7, -3, 1), (-7, -3, -1)])
def test_mod_truncates_and_takes_the_sign_of_the_dividend(a: int, p: int, expected: int) -> None:
    """Fortran MOD truncates; Python ``%`` floors. They agree only when the
    operands share a sign, which is why this survives casual testing."""
    assert runtime._f_mod(a, p) == expected


@pytest.mark.parametrize(("a", "p", "expected"), [(7, 3, 1), (-7, 3, 2), (7, -3, -2), (-7, -3, -1)])
def test_modulo_floors_and_takes_the_sign_of_the_divisor(a: int, p: int, expected: int) -> None:
    """MODULO is the one that matches Python. Fortran has both, and a
    translation that picks the wrong one is wrong only for negatives."""
    assert runtime._f_modulo(a, p) == expected


def test_mod_and_modulo_disagree_exactly_where_the_signs_differ() -> None:
    """Stated as a property, because the point is *when* they diverge: only on
    a non-zero remainder with operands of opposite sign. Anywhere else a
    translation can use either and still be right, which is why the mistake
    survives review."""
    divergences = 0
    for a in range(-8, 9):
        for p in (-3, -2, 2, 3):
            truncated, floored = runtime._f_mod(a, p), runtime._f_modulo(a, p)
            opposite = (a < 0) != (p < 0)
            assert (truncated != floored) == (opposite and truncated != 0)
            divergences += truncated != floored
    assert divergences, "the property is vacuous if they never diverge"


@pytest.mark.parametrize(("a", "b", "expected"), [(7, 2, 3), (-7, 2, -3), (7, -2, -3)])
def test_integer_division_truncates_toward_zero(a: int, b: int, expected: int) -> None:
    """Python's ``//`` floors: ``-7 // 2 == -4``, Fortran gives -3."""
    assert runtime._f_int_div(a, b) == expected


def test_sign_with_an_integer_second_argument_treats_zero_as_positive() -> None:
    """The classic port trap. ``copysign`` reads the sign bit, so it gives
    ``-|a|`` for an integer 0 that happens to be spelled -0.0; Fortran compares
    the value, and integer zero is positive."""
    assert runtime._f_sign(3, 0) == 3
    assert runtime._f_sign(3, -1) == -3


def test_sign_with_a_real_second_argument_honours_negative_zero() -> None:
    """And here Fortran *does* read the sign bit -- gfortran distinguishes
    ``-0.0``. The two rules are opposite, which is why one shim cannot serve
    both without looking at the type."""
    assert runtime._f_sign(3.0, -0.0) == -3.0
    assert runtime._f_sign(3.0, 0.0) == 3.0


def test_character_comparison_ignores_trailing_blanks() -> None:
    """Fortran pads the shorter operand; Python does not."""
    assert runtime._fstr_eq("abc  ", "abc")
    assert not runtime._fstr_eq("abc", "abd")


def test_len_trim_counts_to_the_last_non_blank() -> None:
    assert runtime._f_len_trim("ab   ") == 2
    assert runtime._f_len_trim("   ") == 0


def test_index_is_one_based_and_zero_when_absent() -> None:
    """Fortran INDEX returns a 1-based position, and 0 -- not -1 -- for no
    match. Passing Python's -1 through arithmetic gives a plausible wrong
    subscript rather than an error."""
    assert runtime._f_index("hello", "ll") == 3
    assert runtime._f_index("hello", "z") == 0


# --- numeric agreement -------------------------------------------------------


def test_strict_libm_is_the_default() -> None:
    """``np.exp`` and glibc differ by an ULP, and this backend serves the
    bit-exact gates. Throughput is what the njit and CUDA backends are for."""
    assert runtime._LIBM_STRICT is True


def test_dot_product_accumulates_in_order() -> None:
    """``np.dot`` is pairwise or BLAS and rounds differently. Fortran's
    DOT_PRODUCT accumulates left to right, and a bit-exact gate sees it."""
    a = np.array([1e16, 1.0, -1e16])
    b = np.ones(3)
    # Left to right, the 1.0 is lost to rounding against 1e16 and the answer is
    # 0.0. Any smarter summation keeps it and answers 1.0 -- a better number,
    # and the wrong one for a gate that has to match the Fortran bit for bit.
    assert runtime._f_vdot(a, b) == 0.0


def test_dot_product_stops_at_the_shorter_operand() -> None:
    """Not a property worth having, and kept anyway: this is the emitted
    runtime, and it went through bit-exact gates in this form. A length
    mismatch is invalid Fortran that never reaches here, so tightening it
    would change gated code to guard against something that cannot happen."""
    assert runtime._f_vdot(np.ones(3), np.ones(4)) == 3.0


def test_huge_and_tiny_follow_the_argument_type() -> None:
    assert runtime._f_huge(np.float64(1.0)) == np.finfo(np.float64).max
    assert runtime._f_tiny(np.float64(1.0)) == np.finfo(np.float64).tiny


def test_eoshift_fills_with_zero_rather_than_wrapping() -> None:
    """``np.roll`` wraps. Fortran's EOSHIFT does not, and a wrapped edge is a
    plausible value in the wrong cell."""
    shifted = runtime._f_eoshift(np.array([1, 2, 3, 4]), 1)
    assert list(shifted) == [2, 3, 4, 0]


# --- emitting it -------------------------------------------------------------


def test_emit_returns_parseable_source_with_its_imports() -> None:
    text = "\n".join(runtime.REQUIRED_IMPORTS) + "\n" + runtime.emit()
    ast.parse(text)


def test_emit_covers_every_runtime_definition_and_nothing_else() -> None:
    """It is read out of the live module, so what ships is what was tested.
    ``emit`` itself must not be in there -- a generated file has no use for the
    function that generated it."""
    emitted = {
        node.name
        for node in ast.parse(runtime.emit()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    defined = {
        name
        for name in vars(runtime)
        if (
            name.startswith(("_f", "_F", "_new", "_copy", "_fstr"))
            and callable(vars(runtime)[name])
        )
    }
    assert defined <= emitted
    assert "emit" not in emitted


def test_the_emitted_text_actually_runs() -> None:
    """The point of emitting source rather than importing: the generated file
    is the product and has to stand alone."""
    namespace: dict[str, object] = {}
    exec("\n".join(runtime.REQUIRED_IMPORTS) + "\n" + runtime.emit(), namespace)
    assert namespace["_f_nint"](2.5) == 3  # type: ignore[operator]
    assert namespace["_f_mod"](-7, 3) == -1  # type: ignore[operator]


# --- more places the two languages disagree ----------------------------------


def test_ceiling_and_floor_return_a_default_integer() -> None:
    """Fortran's CEILING and FLOOR return INTEGER, not REAL. Leaving them
    floating works until the result is used as a subscript."""
    assert runtime._f_vceil(np.float64(2.1)).dtype == np.int32
    assert runtime._f_vfloor(np.float64(2.9)).dtype == np.int32
    assert runtime._f_vceil(np.float64(-2.1)) == -2
    assert runtime._f_vfloor(np.float64(-2.1)) == -3


def test_sqrt_of_a_negative_is_a_nan_rather_than_an_exception() -> None:
    """Fortran carries on with the NaN. ``math.sqrt`` raises ValueError, which
    turns a number the compiled reference keeps computing with into a crash --
    and one the differential gate reports as "the candidate raised" rather
    than as the NaN both sides hold."""
    assert np.isnan(runtime._f_sqrt(-1.0))
    assert np.isnan(runtime._f_sqrt(float("nan")))


def test_sqrt_of_a_non_negative_is_the_hardware_root_to_the_bit() -> None:
    """Nothing is traded for the case above: the ordinary argument still goes
    through ``math.sqrt``, which is the correctly-rounded hardware square root
    the compiled reference calls."""
    for x in (0.0, 1.0, 2.0, 1e-300, 1e300, 0.1):
        assert runtime._f_sqrt(x) == math.sqrt(x)
    assert math.copysign(1.0, runtime._f_sqrt(-0.0)) == -1.0


def test_min_and_max_absorb_a_nan_on_the_left_and_propagate_one_on_the_right() -> None:
    """Not a tidy rule, and not a choice: it is what gfortran's SSE ``minsd``
    fold does, measured. ``min(NaN, 0)`` is 0 and ``min(0, NaN)`` is NaN.

    Python's builtin ``min`` returns its first argument on a NaN, which is the
    opposite of this on one side and the same on the other -- so a translation
    using it agrees on half the cases and silently disagrees on the rest.
    """
    assert runtime._f_min(np.nan, 1.0) == 1.0
    assert np.isnan(runtime._f_min(1.0, np.nan))
    assert runtime._f_max(np.nan, 1.0) == 1.0
    assert np.isnan(runtime._f_max(1.0, np.nan))
    assert runtime._f_min(2.0, 1.0) == 1.0


def test_the_vectorised_min_and_max_keep_the_same_asymmetry() -> None:
    """Elementwise, and each element behaves like the scalar fold -- otherwise
    a loop and its vectorised form would disagree on NaN alone."""
    a = np.array([np.nan, 2.0, 3.0])
    b = np.array([1.0, np.nan, 1.0])
    vector = list(runtime._f_vmin(a, b))
    scalar = [runtime._f_min(x, y) for x, y in zip(a, b, strict=True)]
    assert vector[0] == scalar[0] == 1.0
    assert np.isnan(vector[1]) and np.isnan(scalar[1])
    assert vector[2] == scalar[2] == 1.0


def test_strict_libm_matches_the_c_library_elementwise() -> None:
    """The whole reason ``_LIBM_STRICT`` exists: ``np.exp`` on an array goes
    through SIMD paths that differ from glibc by an ULP, and an ULP is the
    difference between a bit-exact gate passing and failing."""
    import math

    x = np.array([0.1, 1.0, 7.5])
    assert list(runtime._f_vexp(x)) == [math.exp(v) for v in x]
    assert list(runtime._f_vlog(x)) == [math.log(v) for v in x]
    assert list(runtime._f_vlog10(x)) == [math.log10(v) for v in x]
    assert list(runtime._f_vpow(x, 2.0)) == [math.pow(v, 2.0) for v in x]


def test_a_constant_argument_intrinsic_is_folded_at_compile_precision() -> None:
    """gfortran evaluates these at compile time with MPFR, correctly rounded,
    and that value matches neither runtime libm."""
    pytest.importorskip("mpmath")
    assert runtime._f_cfold("gamma", 1.8) == pytest.approx(0.93138377098024, abs=1e-14)


def test_trim_and_adjustl_only_touch_blanks() -> None:
    assert runtime._f_trim("ab  ") == "ab"
    assert runtime._f_adjustl("  ab") == "ab  "
    assert len(runtime._f_adjustl("  ab")) == 4, "ADJUSTL preserves length"


def test_scan_is_one_based_and_zero_when_absent() -> None:
    assert runtime._f_scan("hello", "le") == 2
    assert runtime._f_scan("hello", "xyz") == 0


def test_bit_intrinsics_operate_on_integers() -> None:
    assert runtime._f_iand(12, 10) == 8
    assert runtime._f_ior(12, 10) == 14
    assert runtime._f_ieor(12, 10) == 6
    assert runtime._f_ishft(1, 3) == 8
    assert runtime._f_ishft(8, -3) == 1, "a negative shift is a right shift"


def test_huge_distinguishes_real_from_integer() -> None:
    assert runtime._f_huge(np.float64(1.0)) == np.finfo(np.float64).max
    assert runtime._f_huge(np.int32(1)) == np.iinfo(np.int32).max


def test_epsilon_and_tiny_are_float64_only() -> None:
    """A known limitation, pinned so it is a documented answer rather than a
    surprise. Fortran's EPSILON and TINY depend on the argument's kind; these
    return the double-precision value whatever they are handed.

    Harmless where it is used -- CESM physics is ``r8`` throughout -- and wrong
    the day a single-precision kernel asks. Left as it was rather than changed
    under cover of moving the file: the emitted runtime should change in a
    commit that is about changing it.
    """
    assert runtime._f_epsilon(np.float64(1.0)) == np.finfo(np.float64).eps
    assert runtime._f_epsilon(np.float32(1.0)) == np.finfo(np.float64).eps
    assert runtime._f_tiny(np.float32(1.0)) == np.finfo(np.float64).tiny


def test_lbound_is_one_for_a_translated_array() -> None:
    """The arrays are NumPy's, so every lower bound is 1 after the shift the
    rules apply. Reporting 0 here would double-count the shift."""
    assert list(runtime._f_lbound(np.zeros((3, 4)))) == [1, 1]
    assert runtime._f_lbound(np.zeros((3, 4)), 1) == 1


def test_dim_clamps_at_zero() -> None:
    """Fortran DIM(x, y) is ``max(x - y, 0)``, not ``x - y``."""
    assert runtime._f_dim(5, 3) == 2
    assert runtime._f_dim(3, 5) == 0


def test_transfer_reinterprets_rather_than_converts() -> None:
    """TRANSFER is a bit-pattern reinterpretation. Converting instead gives a
    number that is right in the wrong units."""
    bits = runtime._f_transfer(np.float64(1.0), np.int64(0))
    assert bits == 4607182418800017408  # IEEE-754 1.0


def test_a_goto_region_raises_the_scaffold_exception() -> None:
    assert issubclass(runtime._FGoto, Exception)


def test_a_derived_type_local_is_an_attribute_container() -> None:
    obj = runtime._new_derived()
    obj.q = np.zeros(3)
    clone = runtime._copy_derived(obj)
    clone.q[0] = 1.0
    assert obj.q[0] == 0.0, "copying a derived type copies its components"


def test_list_directed_write_starts_with_a_blank_and_pads_to_width() -> None:
    """The output of this is compared against gfortran's byte for byte, so its
    leading blank and column widths are the answer, not formatting taste."""
    record = runtime._f_list_write(np.int32(42))
    assert record.startswith(" ")
    assert record == " " + "42".rjust(12) + " "


# --- copy-out ----------------------------------------------------------------


def test_copy_out_writes_the_overlap_and_leaves_the_rest() -> None:
    """A pcols-wide buffer receiving an ncol-wide result keeps its tail, as
    a by-reference OUT did; an unsupplied optional is left alone."""
    buffer = np.full(4, -1.0)
    runtime._f_copy_out(buffer, np.array([1.0, 2.0]))
    assert list(buffer) == [1.0, 2.0, -1.0, -1.0]
    two_d = np.zeros((3, 3))
    runtime._f_copy_out(two_d, np.ones((2, 2)))
    assert two_d.sum() == 4.0 and two_d[2, 2] == 0.0
    same = np.zeros(2)
    runtime._f_copy_out(same, np.array([5.0, 6.0]))
    assert list(same) == [5.0, 6.0]
    runtime._f_copy_out(same, 7.0)
    assert list(same) == [7.0, 7.0]
    runtime._f_copy_out(None, np.ones(2))  # nothing to write into, no error


def test_sum_accumulates_in_fortran_element_order() -> None:
    """gfortran's inlined SUM is a loop in element order; np.sum pairs its
    terms and rounds differently -- CLUBB's vertical_integral drifted 12 ULP.
    The sequential helper matches the loop exactly, whole or along an axis."""
    import numpy as np

    from recast.transform.numpy import runtime

    rng = np.random.default_rng(7)
    a = np.asfortranarray(rng.uniform(-1e6, 1e6, size=(37, 23)))
    loop = np.float64(0)
    for x in np.ravel(a, order="F"):
        loop = loop + x
    assert runtime._f_vsum(a) == loop
    along = np.zeros(23)
    for i in range(37):
        along = along + a[i, :]
    assert np.array_equal(runtime._f_vsum(a, axis=0), along)
    assert runtime._f_vsum(np.array([1, 2, 3], dtype=np.int32)) == 6
