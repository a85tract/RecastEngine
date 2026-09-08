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
from typing import Any

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


def test_min_and_max_absorb_a_nan_operand_wherever_it_falls() -> None:
    """Measured against the f2py-built reference, not a standalone toy:
    gfortran's MIN/MAX at the golden ``-O1 -fno-fast-math`` flags absorb a
    NaN operand in *either* position -- ``min(NaN, x)`` and ``min(x, NaN)``
    are both ``x`` -- and yield NaN only when every operand is NaN. This is
    ``fmin``/``fmax`` order. A ``quadpack`` body reaching ``min(1.0_wp, x)``
    with ``x`` gone NaN keeps the 1.0, so a "propagate on the right" model
    would mismatch the reference (``dqk15i`` did). Python's builtin ``min``
    returns its first argument on a NaN, a different trap again.
    """
    assert runtime._f_min(np.nan, 1.0) == 1.0
    assert runtime._f_min(1.0, np.nan) == 1.0
    assert runtime._f_max(np.nan, 1.0) == 1.0
    assert runtime._f_max(1.0, np.nan) == 1.0
    assert np.isnan(runtime._f_min(np.nan, np.nan))
    assert np.isnan(runtime._f_max(np.nan, np.nan))
    assert runtime._f_min(2.0, 1.0) == 1.0
    assert runtime._f_max(2.0, 1.0) == 2.0


def test_the_vectorised_min_and_max_keep_the_same_nan_absorption() -> None:
    """Elementwise, and each element behaves like the scalar fold -- otherwise
    a loop and its vectorised form would disagree on NaN alone."""
    a = np.array([np.nan, 2.0, 3.0])
    b = np.array([1.0, np.nan, 1.0])
    vmin = list(runtime._f_vmin(a, b))
    vmax = list(runtime._f_vmax(a, b))
    scalar_min = [runtime._f_min(x, y) for x, y in zip(a, b, strict=True)]
    scalar_max = [runtime._f_max(x, y) for x, y in zip(a, b, strict=True)]
    assert vmin == scalar_min == [1.0, 2.0, 1.0]
    assert vmax == scalar_max == [1.0, 2.0, 3.0]


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


def test_list_directed_write_uses_gfortrans_own_column_widths() -> None:
    """The output of this is compared against gfortran's byte for byte, so the
    column widths are the answer, not formatting taste. Every string below was
    read off gfortran's own ``write(u,*)``.

    The leading blank a list-directed record starts with is the first field's
    padding rather than a separate prefix. Prepending one as well put every
    record a column out -- invisible until a subprogram whose only product is
    the file it writes was compared against the compiler.
    """
    assert runtime._f_list_write(np.int32(5)) == "           5"
    assert runtime._f_list_write(np.int32(-12345)) == "      -12345"
    assert runtime._f_list_write("abc") == " abc"
    assert runtime._f_list_write("abc", np.int32(7)) == " abc           7"
    assert runtime._f_list_write(np.bool_(True), np.bool_(False)) == " T F"
    # A real(8) is a G25.17E3 field and the blank that follows it: seventeen
    # significant figures, and zero counts as one digit before the point.
    assert runtime._f_list_write(0.0) == "   0.0000000000000000     "
    assert runtime._f_list_write(0.5) == "  0.50000000000000000     "
    assert runtime._f_list_write(10.5) == "   10.500000000000000     "
    assert runtime._f_list_write(-1.0e-12) == "  -9.9999999999999998E-013"
    assert runtime._f_list_write(1.0, np.int32(2)) == "   1.0000000000000000                2"


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


def test_sum_folds_left_in_element_order() -> None:
    """Fortran SUM is a left fold; np.sum is pairwise past eight elements and
    rounds differently. ELM's hydraulic-stress kernel sums ten soil layers,
    and the pairwise sum put 925 of 267,264 recorded points up to 75 ULP off;
    the left fold put every one of them on the recording."""
    a = np.array([1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1e16])
    left = 0.0
    for x in a:
        left += x
    assert runtime._f_vsum(a) == left
    assert np.sum(a) != left  # the case that told the two apart
    two_d = np.arange(6.0).reshape(2, 3)
    assert runtime._f_vsum(two_d) == 15.0
    assert np.array_equal(runtime._f_vsum(two_d, axis=0), np.array([3.0, 5.0, 7.0]))


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


# --- external files ----------------------------------------------------------


def test_a_list_directed_read_takes_the_values_the_item_list_asks_for(
    tmp_path: Any,
) -> None:
    """One record per READ, however many records the values are spread over,
    and the rest of the last record discarded -- which is what makes counting
    the rows of a file by reading one value per record work."""
    path = tmp_path / "grid.txt"
    path.write_text("1 2 3\n4 5 6\n")
    _, unit = runtime._f_open(None, str(path), status="old")
    ios, first = runtime._f_read(unit, None, [("float64", None, None)])
    assert (ios, first) == (0, 1.0), "the rest of the record is discarded"
    ios, row = runtime._f_read(unit, None, [("float64", 3, None)])
    assert ios == 0 and list(row) == [4.0, 5.0, 6.0]
    ios, _ = runtime._f_read(unit, None, [("float64", None, None)], strict=False)
    assert ios == -1, "end of file is IOSTAT_END, not an exception, when asked for"
    runtime._f_close(unit)


def test_a_non_advancing_read_stops_at_the_end_of_its_record(tmp_path: Any) -> None:
    """``read(u, '(a)', advance='no')`` walks one record a character at a
    time and reports IOSTAT_EOR at its end. Counting the columns of a text
    file is written this way, and a shim that ran on into the next record
    would count every column in the file."""
    path = tmp_path / "row.txt"
    path.write_text("ab\ncd\n")
    _, unit = runtime._f_open(None, str(path), status="old")
    read = []
    while True:
        ios, char = runtime._f_read(unit, "(a)", [("str", None, 1)], advance="no", strict=False)
        if ios != 0:
            break
        read.append(char)
    assert read == ["a", "b"] and ios == -2
    ios, char = runtime._f_read(unit, "(a)", [("str", None, 1)], advance="no", strict=False)
    assert (ios, char) == (0, "c"), "EOR left the file positioned at the next record"
    runtime._f_rewind(unit)
    ios, char = runtime._f_read(unit, "(a)", [("str", None, 1)], advance="no", strict=False)
    assert (ios, char) == (0, "a")
    runtime._f_close(unit)


def test_inquire_answers_for_the_units_that_are_connected(tmp_path: Any) -> None:
    """``inquire(unit=n, opened=inuse)`` is how a program picks a free unit,
    so NEWUNIT= hands out negative numbers the way gfortran does and leaves
    every number such a scan walks free."""
    path = tmp_path / "f.txt"
    path.write_text("x\n")
    _, unit = runtime._f_open(None, str(path), status="old")
    assert int(unit) < 0
    assert runtime._f_inquire(unit, None, "opened") is True
    assert runtime._f_inquire(10, None, "opened") is False
    assert runtime._f_inquire(6, None, "opened") is True, "stdout is preconnected"
    assert runtime._f_inquire(None, str(path), "exist") is True
    assert runtime._f_inquire(None, str(tmp_path / "no.txt"), "exist") is False
    runtime._f_close(unit)
    assert runtime._f_inquire(unit, None, "opened") is False


def test_an_open_that_cannot_connect_raises_unless_iostat_was_asked_for(
    tmp_path: Any,
) -> None:
    """A statement without IOSTAT= aborts the program in Fortran; one with it
    carries on with the status in a variable."""
    missing = str(tmp_path / "absent.txt")
    with pytest.raises(OSError, match="does not exist"):
        runtime._f_open(None, missing, status="old")
    ios, _ = runtime._f_open(None, missing, status="old", strict=False)
    assert int(ios) == 2


def test_a_zero_width_field_is_as_wide_as_its_value(tmp_path: Any) -> None:
    """``(i0)`` and ``(f0.6)`` are how the corpus converts a number to a
    string. A zero width is not an overflow: it asks for the shortest field
    the value fits in, and treating it as one wrote asterisks -- or, worse,
    nothing at all."""
    assert runtime._f_fmt_write("(i0)", np.int32(42)) == "42"
    assert runtime._f_fmt_write("(f0.6)", 1.5) == "1.500000"


def test_a_stream_connection_is_unformatted_and_pos_counts_bytes(tmp_path: Any) -> None:
    """``access='stream'`` with no FORM= is an UNFORMATTED connection --
    gfortran reports it as one -- and it is how a program reads a file byte
    by byte. A PPM is the case: the header is read as text through one
    connection, INQUIRE(POS=) says where it ended, and the pixels come back
    through a second one positioned there with POS=. Reading those bytes as
    text records takes a pixel of value 10 for the end of a record.
    """
    path = tmp_path / "img.ppm"
    path.write_bytes(b"P6\n2 1\n255\n" + bytes([7, 8, 9, 250, 251, 252]) + b"\n")

    _, unit = runtime._f_open(None, str(path), access="stream", form="formatted", status="old")
    ios, signature = runtime._f_read(unit, "(a2)", [("str", None, 2)])
    assert (int(ios), signature) == (0, "P6")
    _, w, h = runtime._f_read(unit, None, [("int32", None, None), ("int32", None, None)])
    _, ncol = runtime._f_read(unit, None, [("int32", None, None)])
    assert (int(w), int(h), int(ncol)) == (2, 1, 255)
    offset = runtime._f_inquire(unit, None, "pos")
    assert int(offset) == 12, "the byte after the header, counted from one"
    runtime._f_close(unit)

    _, unit = runtime._f_open(None, str(path), access="stream", status="old")
    assert runtime._f_inquire(unit, None, "form") == "UNFORMATTED"
    ios, ccode = runtime._f_read(unit, None, [("str", None, 1)], pos=int(offset) - 1)
    assert (int(ios), ccode) == (0, "\n"), "POS= is where the read starts, not where it ends"
    pixels = [ord(runtime._f_read(unit, None, [("str", None, 1)])[1]) for _ in range(6)]
    assert pixels == [7, 8, 9, 250, 251, 252]
    ios, _ = runtime._f_read(unit, None, [("str", None, 1)], strict=False)
    assert int(ios) == 0, "the record terminator the file ends with"
    ios, _ = runtime._f_read(unit, None, [("str", None, 1)], strict=False)
    assert int(ios) == -1, "a short read is end of file"
    runtime._f_close(unit)


def test_an_unformatted_sequential_read_is_still_refused(tmp_path: Any) -> None:
    """A stream has no records to guess at; an unformatted *sequential* file
    is wrapped in length markers only its compiler can spell, and reading it
    here would put wrong numbers in the right variables."""
    path = tmp_path / "raw.dat"
    path.write_bytes(b"\x04\x00\x00\x00")
    _, unit = runtime._f_open(None, str(path), form="unformatted", status="old")
    with pytest.raises(OSError, match="unformatted unit"):
        runtime._f_read(unit, None, [("int32", None, None)])
    runtime._f_close(unit)


def test_a_format_shorter_than_its_item_list_reverts_and_ends_the_record() -> None:
    """``write(u, '(3a1)') achar(pixel)`` on more than three components writes
    more than one record: Fortran reverts to the start of the format and, in
    doing so, ends the record. Stopping at the first pass instead dropped
    every value after the third."""
    assert runtime._f_fmt_records("(3a1)", list("abcdefgh")) == ["abc", "def", "gh"]
    assert runtime._f_fmt_records("(i0,' ',i0)", [8, 8]) == ["8 8"]
    assert runtime._f_fmt_records("(a2)", ["P6"]) == ["P6"]
    # ``/`` ends a record the same way, and a data descriptor with no value
    # left ends the transfer where it stands.
    assert runtime._f_fmt_records("(i0,/,i0)", [1, 2]) == ["1", "2"]
    assert runtime._f_fmt_records("(i0,i0)", [1]) == ["1"]


def test_an_external_write_puts_its_records_in_the_file(tmp_path: Any) -> None:
    """The bytes below are what gfortran's own ``saveppm`` writes for a
    three-component pixel: a header, then non-advancing pixel writes that
    continue one record, and the record terminator CLOSE puts on the
    incomplete record the last of them left open."""
    path = tmp_path / "out.ppm"
    _, unit = runtime._f_open(None, str(path), status="replace")
    runtime._f_write(unit, "(a2)", ["P6"])
    runtime._f_write(unit, "(i0,' ',i0)", [2, 1])
    runtime._f_write(unit, "(i0)", [255])
    for pixel in ([1, 2, 3], [4, 5, 6]):
        runtime._f_write(unit, "(3a1)", [runtime._f_vachar(np.array(pixel))], advance="no")
    runtime._f_close(unit)
    assert path.read_bytes() == b"P6\n2 1\n255\n" + bytes([1, 2, 3, 4, 5, 6]) + b"\n"


def test_a_write_to_a_unit_nothing_connected_is_a_log(capsys: Any, tmp_path: Any) -> None:
    """A unit no OPEN in the translation connected is the log destination it
    always was: the records go where PRINT's go, and no file is invented for
    a connection this translation does not hold."""
    runtime._f_write(6, None, ["hello"])
    assert capsys.readouterr().out == " hello\n"
    assert not list(tmp_path.iterdir())


def test_achar_over_an_array_is_one_character_per_element() -> None:
    """A fixed-width NumPy string array would pad every element; the item list
    a WRITE formats one value at a time must not be padded."""
    rendered = runtime._f_vachar(np.array([[65, 10], [13, 250]], dtype=np.int32))
    assert rendered.tolist() == [["A", "\n"], ["\r", "\xfa"]]


def test_log_outside_its_domain_is_an_ieee_value_not_an_exception() -> None:
    """``math.log`` raises where the compiled reference carries on with
    ``-Infinity`` and ``NaN``. Raising turns a number both sides agree on into
    "the candidate raised", which the differential gate reports as no
    comparison at all."""
    assert runtime._f_log(math.e) == 1.0
    assert runtime._f_log(0.0) == float("-inf")
    assert math.isnan(runtime._f_log(-1.0))
    assert runtime._f_log10(100.0) == 2.0
    assert math.isnan(runtime._f_log10(-1.0))


def test_int_of_a_value_no_integer_holds_is_the_conversion_the_compiler_emits() -> None:
    """Python's ``int`` raises on a NaN and grows without bound past the range
    of an INTEGER; gfortran emits the hardware conversion, which answers every
    value it cannot represent with the most negative integer of the kind."""
    assert runtime._f_int(2.9) == 2
    assert runtime._f_int(-2.9) == -2
    assert runtime._f_int(float("nan")) == -(2**31)
    assert runtime._f_int(1e30) == -(2**31)
    assert runtime._f_int(1e30, 8) == -(2**63)
    assert runtime._f_int(7) == 7


@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 4, 5, 7, 12, -1, -3])
def test_a_runtime_integer_power_is_the_expansion_a_literal_one_gets(exponent: int) -> None:
    """``(xe(i)-x0)**(j-1)`` has no literal for ``expand_power`` to expand, so
    what was left was Python's ``**`` -- a libm ``pow`` call gfortran never
    makes for an integer exponent. libgcc squares and multiplies, LSB first,
    which is exactly what ``expand_power`` writes out when it can, so the two
    routes have to reach the same bits or a subprogram is bit-exact or not
    depending on whether its exponent happened to be a literal."""
    from recast.transform.numpy.expressions import expand_power

    for x in (0.31672977626795387, -3.25, 1.0000000001, 7.5e-8):
        spelled = eval(expand_power("x", exponent), {"x": x}) if exponent else 1.0
        assert runtime._f_powi(x, exponent) == spelled, f"{x}**{exponent}"


def test_a_runtime_integer_power_is_not_the_pow_call_python_would_make() -> None:
    """The point of the helper, stated as the difference it exists for: one
    ULP, on an ordinary value, in a direction nothing structural can see."""
    x = 0.31672977626795387
    assert runtime._f_powi(x, 3) != x**3
    assert runtime._f_powi(x, 3) == x * (x * x)


def test_seq_tail_is_the_column_major_storage_from_the_element_on() -> None:
    """``a(i, 1)`` for ``dx(*)``: Fortran hands the callee the memory from
    that element to the end of the array in column-major order. A view of a
    Fortran-contiguous actual, so the callee's writes land in the caller's
    array; ``x(2, *)`` folds it onto the leading extent with the last axis
    taking the whole columns left."""
    a = np.asfortranarray(np.arange(1.0, 13.0).reshape(3, 4, order="F"))
    tail = runtime._f_seq_tail(a, 1)  # a(2, 1) onward: 2, 3, 4, ..., 12
    assert tail.tolist() == list(range(2, 13))
    assert np.shares_memory(tail, a)
    tail[0] = -1.0
    assert a[1, 0] == -1.0
    folded = runtime._f_seq_tail(a, 1, 2)  # 11 elements: five whole columns of 2
    assert folded.shape == (2, 5)
    assert folded[:, 0].tolist() == [-1.0, 3.0]
    assert np.shares_memory(folded, a)
    assert runtime._f_seq_tail(a, 0).tolist() == [-1.0 if v == 2.0 else v for v in range(1, 13)]


def test_seq_tail_with_the_matrix_s_own_leading_extent_is_the_matrix_from_that_row() -> None:
    """``h12(..., a(i, 1), mda, ...)`` walks row ``i`` with the matrix's own
    leading extent: ``u(1, j)`` is ``a(i, j)`` for every column, the last
    one included though the storage from ``a(i, 1)`` holds only part of it.
    That is the slice ``a[i-1:, :]``, a view in either memory order, where
    folding onto whole columns lost the last column altogether (SLSQP's
    ``hfti``)."""
    for order in ("F", "C"):
        a = np.array(np.arange(1.0, 13.0).reshape(3, 4, order="F"), order=order)
        u = runtime._f_seq_tail(a, 1, 3)  # a(2, 1) with iue = mda = 3
        assert u.shape == (2, 4)
        assert u[0].tolist() == a[1].tolist()
        assert np.shares_memory(u, a)
        u[0, 3] = -4.0
        assert a[1, 3] == -4.0
        c = runtime._f_seq_tail(a, 1, 3)
        c[0, 0] = 0.0
        runtime._f_seq_tail_out(a, 1, c)  # a view already: nothing to redo
        assert a[1, 0] == 0.0 and a[1, 3] == -4.0


def test_seq_tail_out_reaches_a_c_ordered_matrix() -> None:
    """Where the actual is not Fortran-contiguous and no slice spells the
    tail, it was a copy, and the callee's writes reach the caller only
    through the write-back: the values land at the same column-major
    positions, and nothing is written for an empty result."""
    c_ordered = np.arange(1.0, 13.0).reshape(3, 4)
    tail = runtime._f_seq_tail(c_ordered, 4)  # (2, 2) onward, a copy
    assert not np.shares_memory(tail, c_ordered)
    tail[:] = -tail
    runtime._f_seq_tail_out(c_ordered, 4, tail)
    expected = np.arange(1.0, 13.0).reshape(3, 4)
    flat = expected.ravel(order="F")
    flat[4:] = -flat[4:]
    assert np.array_equal(c_ordered, flat.reshape(3, 4, order="F"))
    runtime._f_seq_tail_out(c_ordered, 4, np.zeros(0))
    assert np.array_equal(c_ordered, flat.reshape(3, 4, order="F"))
