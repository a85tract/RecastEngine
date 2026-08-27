"""Defects found in the reference pipeline, checked for here.

The translator this backend was migrated from catalogued six translation
defects against real BSD Fortran libraries and fixed them upstream. Five were
present here too -- the same *finding*, reached by different code, which is
what the relay in ``NOTICE`` records. The sixth, an assumed-size ``x(*)``
read as rank 2, this repository had already got right.

Every test below carries its own positive control: the assertion is written
against the symptom, so reverting the fix it guards makes it fail with that
symptom rather than with a message about a refactor. Four of the five are
import-time or emit-time failures, loud once they happen. The fifth --
``CYCLE`` naming an outer loop -- is the one worth the file: it emitted a
program that runs, terminates, and is wrong.

None of them fires on the CAM corpus, so ``emit_diff``, ``numba_diff`` and
``cuda_diff`` are unmoved by the whole set, and a gate a change cannot move
is not evidence about that change. The gate that *can* is ``corpus/``, whose
twelve libraries include the six the defects were found against: the relay
takes it from 37 of 59 units importing to 51, and from 58 parsing to 59,
with every other column unchanged. These tests are the fast, local statement
of the same thing -- a corpus run is minutes and needs submodules, and a unit
that stops importing does not say which of five reasons it stopped for.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
np = pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.fortran import interface  # noqa: E402
from recast.transform.numpy import runtime  # noqa: E402
from recast.transform.numpy.constants import np_int_literal  # noqa: E402
from recast.transform.numpy.names import bind_use_statements  # noqa: E402
from recast.transform.numpy.subprograms import Subprograms  # noqa: E402

from .test_numpy_statements import build  # noqa: E402


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.f90"
    path.write_text(text)
    return path


def _runtime_namespace() -> dict[str, Any]:
    """The runtime as a generated module gets it: exec'd text, not an import."""
    namespace: dict[str, Any] = {"np": np, "math": math, "os": os, "Any": Any}
    exec(runtime.emit(), namespace)
    return namespace


# --- #11: a standalone DIMENSION statement ------------------------------------

DIMENSION_STMT = """\
module specfun_like
implicit none
integer, parameter :: mm = 4, n = 3
contains
subroutine legendre(pm, x)
double precision pm, x
dimension pm(0:mm,0:n)
integer i
do i = 0, n
   pm(i,i) = x * i
end do
end subroutine
end module
"""


def test_a_dimension_statement_gives_its_name_a_shape(tmp_path: Path) -> None:
    """``dimension pm(0:mm,0:n)`` is the only place ``pm``'s rank is written.

    Nothing read the standalone form, so ``pm`` stayed a scalar.
    """
    record = interface.extract(_write(tmp_path, "dims", DIMENSION_STMT))
    argument = next(a for a in record["subprograms"][0]["args"] if a["name"] == "pm")
    assert argument["dims"] == [{"lb": "0", "ub": "mm"}, {"lb": "0", "ub": "n"}]


def test_a_dimensioned_name_is_indexed_not_defined(tmp_path: Path) -> None:
    """The symptom: a scalar with dummy subscripts is a statement function.

    ``pm(i,i) = x*i`` emitted ``def pm(i, i):`` -- a duplicate-argument
    ``SyntaxError``, so the module did not import and nothing in it ran.
    """
    statements, nodes = build(_write(tmp_path, "dims", DIMENSION_STMT), "legendre")
    body = "\n".join(line for node in nodes for line in statements.render(node, 1))
    assert "def pm(" not in body
    assert "pm[" in body


def test_a_dimension_statement_can_be_the_whole_declaration(tmp_path: Path) -> None:
    """A name no type declaration mentions still gets a shape, not nothing."""
    source = """\
module implicit_dims
implicit none
contains
subroutine s()
dimension work(10)
work(1) = 0.0
end subroutine
end module
"""
    record = interface.extract(_write(tmp_path, "implicit_dims", source))
    locals_ = {local["name"]: local for local in record["subprograms"][0]["locals"]}
    assert locals_["work"]["dims"] == [{"lb": "1", "ub": "10"}]


def test_a_module_level_dimension_statement_is_read_too(tmp_path: Path) -> None:
    """The form appears at both levels, and one of the two is not enough.

    A name the DIMENSION statement is the *only* mention of has no declared
    type, so its dtype stays unsettled rather than being guessed at -- the
    rules downstream refuse it, which is the right answer and not the same
    as calling it a scalar.
    """
    source = """\
module moddim
implicit none
real :: tbl
dimension tbl(100)
integer :: known(3)
dimension extra(5)
end module
"""
    record = interface.extract(_write(tmp_path, "moddim", source))
    state = {entry["name"]: entry for entry in record["module_state"]}
    assert state["tbl"]["dims"] == [{"lb": "1", "ub": "100"}]
    # An entity that declared its own shape keeps it.
    assert state["known"]["dims"] == [{"lb": "1", "ub": "3"}]
    assert state["extra"]["dims"] == [{"lb": "1", "ub": "5"}]
    assert state["extra"]["dtype"].startswith("UNKNOWN")


# --- #9: an integer constant wider than int32 ---------------------------------


def test_an_integer_constant_is_as_wide_as_its_value() -> None:
    """``np.int32`` of an out-of-range value raises while the module imports.

    The width comes from the value, so everything that already fit is
    untouched -- that is what makes this safe to apply to a blessed corpus.
    """
    assert np_int_literal(0) == "np.int32(0)"
    assert np_int_literal(2**31 - 1) == "np.int32(2147483647)"
    assert np_int_literal(-(2**31)) == "np.int32(-2147483648)"
    assert np_int_literal(2**31) == "np.int64(2147483648)"
    assert np_int_literal(0x9908B0DF) == "np.int64(2567483615)"
    assert np_int_literal(6364136223846793005) == "np.int64(6364136223846793005)"


def test_the_wide_spelling_is_the_one_that_survives_evaluation() -> None:
    """The positive control, run rather than compared."""
    with pytest.raises(OverflowError):
        eval("np.int32(6364136223846793005)", {"np": np})
    assert eval(np_int_literal(6364136223846793005), {"np": np}) == 6364136223846793005


# --- #12: USE of an intrinsic module ------------------------------------------


def test_an_intrinsic_module_binds_to_the_runtime_and_imports_nothing() -> None:
    """``import iso_fortran_env_numpy`` names a file that can never exist.

    Both halves are the defect: the ``USE, INTRINSIC ::`` spelling was not
    matched at all, and a module that is not a companion was assumed to have
    one.
    """
    record = {
        "use_statements": [
            "USE, INTRINSIC :: iso_fortran_env, ONLY: output_unit, real64",
            "use ieee_arithmetic",
            "use shr_kind_mod, only: r8",
        ],
        "module_parameters": [],
        "module_state": [],
    }
    bindings, stubs, intrinsic = bind_use_statements(record, set(), set(), {})
    assert bindings["output_unit"] == "_iso_fortran_env.output_unit"
    assert bindings["real64"] == "_iso_fortran_env.real64"
    # An ordinary module still gets its stub import; the intrinsic ones do not.
    assert stubs == {"shr_kind_mod": "_shr_kind_mod"}
    # And they are reported, because nothing downstream can infer an alias
    # that no import line announces.
    assert intrinsic == {"_iso_fortran_env", "_ieee_arithmetic"}


def test_the_intrinsic_namespaces_are_in_every_generated_module() -> None:
    """They are part of the inlined runtime, which is why nothing is imported."""
    namespace = _runtime_namespace()
    assert namespace["_iso_fortran_env"].real64 == 8
    assert namespace["_iso_fortran_env"].output_unit == 6
    assert namespace["_iso_fortran_env"].iostat_end == -1
    assert namespace["_ieee_arithmetic"].ieee_is_nan(np.nan)
    assert namespace["_omp_lib"].omp_get_thread_num() == 0
    assert namespace["_omp_lib"].omp_in_parallel() is False


def test_c_null_char_is_a_character_and_not_its_escape() -> None:
    """``C_NULL_CHAR`` is ``ACHAR(0)``.

    Spelled ``"\\\\0"`` it is a backslash and a zero, so a string terminated
    with it is not terminated -- and every length that counts it is one too
    long. Found by reading the original rather than by any test; the
    reference pipeline still has it.
    """
    binding = _runtime_namespace()["_iso_c_binding"]
    assert binding.c_null_char == chr(0)
    assert binding.c_new_line == chr(10)
    assert len(binding.c_null_char) == 1


# --- #8: a local PARAMETER the token pass cannot classify ---------------------

IMPLIED_DO_PARAMETER = """\
module quadpack_like
implicit none
integer, parameter :: nq = 4
contains
subroutine q(x)
real, intent(out) :: x(nq)
real, parameter :: w(nq) = (/ (cos(real(k)), k = 1, nq) /)
integer :: i
do i = 1, nq
   x(i) = w(i)
end do
end subroutine
end module
"""


def test_an_initializer_the_token_pass_cannot_read_is_parsed(tmp_path: Path) -> None:
    """It used to be upper-cased and emitted, which is not Python.

    The control is the ``compile``: the old spelling was
    ``np.array([(COS(REAL(K)), K = 1, NQ)])``, a ``SyntaxError`` that takes
    the whole module down at import rather than the one line it could not
    translate.
    """
    source = _write(tmp_path, "quadpack_like", IMPLIED_DO_PARAMETER)
    statements, _ = build(source, "q")
    record = interface.extract(source)
    subprogram = next(s for s in record["subprograms"] if s["name"] == "q")
    parameter = subprogram["local_parameters"][0]
    own = frozenset({parameter["name"]})

    value = Subprograms._parameter_value(
        Subprograms.__new__(Subprograms), parameter["init_expr"].strip(), own, statements
    )
    compile(value, "<initializer>", "eval")
    assert eval(value, {"np": np, "math": math, "NQ": 4}).shape == (4,)


def test_an_implied_do_is_the_constructor_not_an_element_of_it(tmp_path: Path) -> None:
    """Shape ``(n,)``, not ``(1, n)``.

    numpy broadcasts the wrong one rather than refusing it, so the error
    surfaces as an answer.
    """
    source = """\
module ac
implicit none
contains
subroutine s(a, b)
real, intent(out) :: a(3), b(4)
a = (/ (2.0*k, k = 1, 3) /)
b = (/ 0.0, (2.0*k, k = 1, 3) /)
end subroutine
end module
"""
    statements, nodes = build(_write(tmp_path, "ac", source), "s")
    lone, mixed = (statements.render(node, 1)[0] for node in nodes[:2])
    assert "np.array([[" not in lone
    assert lone.endswith("np.array([(F32_2P0 * k) for k in range(1, I_3 + 1)])")
    # Beside other elements it is spliced, which is the same shape claim.
    assert "*[(F32_2P0 * k) for k in range(1, I_3 + 1)]" in mixed


# --- #10: EXIT and CYCLE naming an enclosing construct ------------------------

NAMED_CONSTRUCTS = """\
module dadadj_like
implicit none
contains
subroutine s(ncol, a)
integer, intent(in) :: ncol
real, intent(inout) :: a(ncol)
integer :: i, jiter
col: do i = 1, ncol
   do jiter = 1, 10
      if (a(i) > 0.0) cycle col
      if (a(i) < -1.0) exit col
      a(i) = a(i) + 1.0
   end do
end do col
end subroutine
end module
"""


def test_a_named_cycle_from_an_inner_loop_leaves_the_named_one(tmp_path: Path) -> None:
    """``cycle col`` continued the inner ``jiter`` loop.

    A bare ``continue`` is what both statements used to emit. It is not a
    shape difference: the inner loop keeps running and the outer one never
    advances, so the program takes a path the Fortran had abandoned.
    """
    statements, nodes = build(_write(tmp_path, "dadadj_like", NAMED_CONSTRUCTS), "s")
    body = "\n".join(line for node in nodes for line in statements.render(node, 1))
    assert "raise _FLoopCycle('col')" in body
    assert "raise _FLoopExit('col')" in body
    assert "except _FLoopCycle as _lc:" in body
    assert "except _FLoopExit as _le:" in body


def test_the_named_forms_mean_what_the_fortran_meant() -> None:
    """The emitted shape, executed, against the loop it stands for."""
    namespace = _runtime_namespace()
    loop_cycle, loop_exit = namespace["_FLoopCycle"], namespace["_FLoopExit"]

    def emitted(a: Any, ncol: int) -> list[tuple[int, int]]:
        trace: list[tuple[int, int]] = []
        try:  # exit col
            for i in range(1, ncol + 1):
                try:
                    for jiter in range(1, 10 + 1):
                        trace.append((i, jiter))
                        if a[i - 1] > 0.0:
                            raise loop_cycle("col")
                        if a[i - 1] < -1.0:
                            raise loop_exit("col")
                        a[i - 1] = a[i - 1] + 1.0
                except loop_cycle as _lc:
                    if _lc.args[0] != "col":
                        raise
        except loop_exit as _le:
            if _le.args[0] != "col":
                raise
        return trace

    def meant(a: Any, ncol: int) -> list[tuple[int, int]]:
        """``cycle col`` / ``exit col`` written out directly."""
        trace: list[tuple[int, int]] = []
        for i in range(1, ncol + 1):
            for jiter in range(1, 10 + 1):
                trace.append((i, jiter))
                if a[i - 1] > 0.0:
                    break  # cycle col -> the next i
                if a[i - 1] < -1.0:
                    return trace  # exit col -> out of the outer loop
                a[i - 1] = a[i - 1] + 1.0
        return trace

    def bare_keywords(a: Any, ncol: int) -> list[tuple[int, int]]:
        """What this backend emitted before: both bound to the inner loop."""
        trace: list[tuple[int, int]] = []
        for i in range(1, ncol + 1):
            for jiter in range(1, 10 + 1):
                trace.append((i, jiter))
                if a[i - 1] > 0.0:
                    continue
                if a[i - 1] < -1.0:
                    break
                a[i - 1] = a[i - 1] + 1.0
        return trace

    for vector in ([5.0, 0.0], [0.0, -9.0, 0.0]):
        ncol = len(vector)
        got, want, old = (np.array(vector) for _ in range(3))
        assert emitted(got, ncol) == meant(want, ncol)
        assert np.array_equal(got, want)
        # The positive control: the old spelling disagrees, and by a lot.
        assert bare_keywords(old, ncol) != meant(np.array(vector), ncol)


def test_a_named_loop_nothing_crosses_is_spelled_as_before(tmp_path: Path) -> None:
    """No catcher unless something actually crosses a loop boundary.

    This is what keeps the corpus byte-identical: a named loop whose EXITs
    sit in its own body emits exactly the ``break``/``continue`` it always
    did, so ``emit_diff`` cannot see this change at all.
    """
    source = """\
module plain
implicit none
contains
subroutine t(n, a)
integer, intent(in) :: n
real, intent(inout) :: a(n)
integer :: i
outer: do i = 1, n
   if (a(i) > 0.0) cycle outer
   if (a(i) < 0.0) exit outer
end do outer
end subroutine
end module
"""
    statements, nodes = build(_write(tmp_path, "plain", source), "t")
    body = "\n".join(line for node in nodes for line in statements.render(node, 1))
    assert "_FLoop" not in body
    assert "continue" in body
    assert "break" in body


def test_an_exit_naming_a_block_does_not_break_a_loop(tmp_path: Path) -> None:
    """A BLOCK is inlined, so there is no construct for ``break`` to leave.

    Emitted inside a loop, the old spelling left the *loop*.
    """
    source = """\
module blocky
implicit none
contains
subroutine s(n, a)
integer, intent(in) :: n
real, intent(inout) :: a(n)
integer :: i
do i = 1, n
   blk: block
      integer :: k
      k = 1
      if (k == 1) exit blk
      a(i) = 2.0
   end block blk
   a(i) = a(i) + 1.0
end do
end subroutine
end module
"""
    statements, nodes = build(_write(tmp_path, "blocky", source), "s")
    body = "\n".join(line for node in nodes for line in statements.render(node, 1))
    assert "raise _FBlockExit('blk')" in body
    assert "except _FBlockExit as _be:" in body
    assert "break" not in body
