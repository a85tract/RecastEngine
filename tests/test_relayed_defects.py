"""Defects found in the reference pipeline, checked for here.

The translator this backend was migrated from catalogued six translation
defects against real BSD Fortran libraries and fixed them upstream. Five were
present here too -- the same *finding*, reached by different code, which is
what the relay in ``NOTICE`` records. The sixth, an assumed-size ``x(*)``
read as rank 2, this repository had already got right.

A seventh, from the same upstream and the same weeks, is here as well: generic
dispatch had only the rank and integer-ness axes, and could not separate two
overloads that differ by declared type. That one is not like the others -- it
was found on CAM, so it is the one defect in this file the emission
differentials *can* see, and they do.

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
with every other column unchanged. The dispatch axis moves both: `emit_diff`
5 to 2, `numba_diff` 1 to 0, and the corpus's deferred-block total 377 to 366.
These tests are the fast, local statement
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


# --- #4: the declared-dtype axis of generic dispatch --------------------------

BY_DERIVED_TYPE = """\
module coords
implicit none
integer, parameter :: r8 = selected_real_kind(12)
type :: cartesian2d_t
  real(r8) :: x, y
end type
type :: cartesian3d_t
  real(r8) :: x, y, z
end type
interface distance
  module procedure distance_cart2d
  module procedure distance_cart3d
end interface
contains
function distance_cart2d(a, b) result(d)
  type(cartesian2d_t), intent(in) :: a, b
  real(r8) :: d
  d = a%x - b%x
end function
function distance_cart3d(a, b) result(d)
  type(cartesian3d_t), intent(in) :: a, b
  real(r8) :: d
  d = a%x - b%x
end function
subroutine drive(p, q, out)
  type(cartesian2d_t), intent(in) :: p, q
  real(r8), intent(out) :: out
  out = distance(p, q)
end subroutine
end module
"""


def _semantics_for(tmp_path: Path, name: str, text: str, subprogram: str) -> Any:
    from recast.fortran import semantics

    return semantics.for_subprogram(interface.extract(_write(tmp_path, name, text)), subprogram)


def test_a_generic_resolves_on_the_declared_type_alone(tmp_path: Path) -> None:
    """Two overloads of the same rank and neither integer.

    Rank and integer-ness are the two axes this had, and they cannot tell
    ``distance_cart2d`` from ``distance_cart3d`` -- both take two scalars of
    a derived type. Fortran resolves it on the type, which is the axis being
    relayed; without it every call to a type-overloaded generic is ambiguous
    and the block is deferred.
    """
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    sem = _semantics_for(tmp_path, "coords", BY_DERIVED_TYPE, "drive")
    actuals = list(Actual_Arg_Spec_List("p, q").children)
    # The positive control is the shape of the question: both candidates are
    # present, same rank, same integer-ness.
    assert sorted(sem.generics["distance"]) == ["distance_cart2d", "distance_cart3d"]
    assert sem.dispatch("distance", actuals) == "distance_cart2d"


BY_REAL_KIND = """\
module kinds_overload
implicit none
integer, parameter :: r4 = selected_real_kind(6)
integer, parameter :: r8 = selected_real_kind(12)
interface widen
  module procedure widen_r4
  module procedure widen_r8
end interface
contains
subroutine widen_r4(x)
  real(r4), intent(inout) :: x
  x = x + 1.0
end subroutine
subroutine widen_r8(x)
  real(r8), intent(inout) :: x
  x = x + 1.0
end subroutine
subroutine drive(a, b)
  real(r4), intent(inout) :: a
  real(r8), intent(inout) :: b
  call widen(a)
  call widen(b)
end subroutine
end module
"""


def test_a_generic_resolves_on_real_kind(tmp_path: Path) -> None:
    """``real(r4)`` and ``real(r8)`` are different types to Fortran's TKR rule."""
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    sem = _semantics_for(tmp_path, "kinds_overload", BY_REAL_KIND, "drive")
    assert sem.dispatch("widen", list(Actual_Arg_Spec_List("a").children)) == "widen_r4"
    assert sem.dispatch("widen", list(Actual_Arg_Spec_List("b").children)) == "widen_r8"


def test_an_unresolved_kind_rejects_nothing(tmp_path: Path) -> None:
    """``UNKNOWN_REAL_KIND(wp)`` is a real whose kind this stage could not settle.

    Letting it reject a candidate is how a new axis flips a resolution that
    was already unique, so it constrains nothing and the call stays exactly
    as ambiguous as it was.
    """
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    from recast.fortran import semantics

    source = BY_REAL_KIND.replace(
        "  real(r4), intent(inout) :: a\n", "  real(wp), intent(inout) :: a\n"
    )
    sem = _semantics_for(tmp_path, "unknown_kind", source, "drive")
    assert sem.declaration("a")["dtype"].startswith("UNKNOWN_REAL_KIND")
    with pytest.raises(semantics.AmbiguousDispatch, match="ambiguous"):
        sem.dispatch("widen", list(Actual_Arg_Spec_List("a").children))


def test_a_computed_actual_constrains_nothing(tmp_path: Path) -> None:
    """The axis reads declarations; it does not infer a type for an expression.

    An inferred dtype that is wrong does not refuse a call -- it picks a
    different overload, and nothing downstream re-checks which one.
    """
    from fparser.two.Fortran2003 import Expr, Name, Part_Ref

    sem = _semantics_for(tmp_path, "coords2", BY_DERIVED_TYPE, "drive")
    # ``dtype_of`` upper-cases the base type, so the two sides of a derived-type
    # comparison are spelled the same however the source wrote them.
    assert sem.declared_dtype(Name("p")) == "UNKNOWN(TYPE(CARTESIAN2D_T))"
    assert sem.declared_dtype(Expr("p % x - q % x")) is None
    # A reference that is not a declared array is a call, and a result dtype
    # is not a declared one.
    assert sem.declared_dtype(Part_Ref("distance_cart2d(p, q)")) is None


def test_the_axis_only_narrows(tmp_path: Path) -> None:
    """A call that already resolved on rank still resolves the same way.

    The monotonicity claim, which is what makes this safe to add to a
    blessed corpus: the axis can take candidates away from an ambiguous set
    and can never add one or change a single match into a different one.
    """
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    sem = _semantics_for(tmp_path, "coords3", BY_DERIVED_TYPE, "drive")
    # Both overloads take two arguments of a derived type; asking with one
    # argument matches neither, and the dtype axis does not invent a match.
    with pytest.raises(Exception, match="no match"):
        sem.dispatch("distance", list(Actual_Arg_Spec_List("p").children))


# --- #37, #38: what a call reads ----------------------------------------------

KIND_AND_BUFFER = """\
module rw_mod
implicit none
integer, parameter :: wp = selected_real_kind(12)
contains
subroutine callee(a, x)
  real(wp), intent(in) :: a
  real(wp), intent(out) :: x(:)
  x(1) = a
end subroutine
subroutine drive(pd, a, x, z, n)
  real(wp), intent(in) :: pd, a
  real(wp), intent(inout) :: x(10)
  complex(wp), intent(out) :: z
  integer, intent(out) :: n
  z = cmplx(pd, 0.0_wp, wp)
  n = int(a, kind=8)
  call callee(a, x)
end subroutine
end module
"""


def _rwsets(tmp_path: Path, name: str, text: str, subprogram: str) -> dict[str, Any]:
    from recast.fortran import rwset
    from recast.fortran._parse import f03, parse, walk

    src = _write(tmp_path, name, text)
    record = interface.extract(src)
    node = next(
        s
        for s in walk(parse(src), f03.Subroutine_Subprogram)
        if str(walk(s, f03.Subroutine_Stmt)[0].children[1]).lower() == subprogram
    )
    return {b["id"]: b for b in rwset.block_rwsets(node, rwset.scope_for(record, subprogram))}


def test_a_kind_name_is_not_a_read_wherever_a_conversion_puts_it(tmp_path: Path) -> None:
    """The exclusion stopped at the two-argument conversions, so ``cmplx(x,
    y, wp)`` reported ``wp`` as a read and disagreed with the emission, which
    drops the KIND wherever it sits (#37). The rule is the emitter's: a
    ``kind=`` keyword anywhere, the second positional of a two-argument
    conversion -- except ``cmplx``, whose second is the imaginary part --
    and ``cmplx``'s third positional."""
    blocks = _rwsets(tmp_path, "rw", KIND_AND_BUFFER, "drive")
    assert blocks["B001"]["reads"] == ["pd"], "cmplx's third positional is the kind"
    assert blocks["B002"]["reads"] == ["a"], "a kind= keyword is not a read"


def test_a_two_argument_cmplx_reads_both(tmp_path: Path) -> None:
    """The one conversion whose second positional is a value, not a kind."""
    text = KIND_AND_BUFFER.replace(
        "  complex(wp), intent(out) :: z\n",
        "  complex(wp), intent(out) :: z, z2, z3\n  real(wp), intent(out) :: r\n",
    ).replace(
        "  call callee(a, x)\n",
        "  z2 = cmplx(a, pd)\n  z3 = cmplx(a, pd, kind=wp)\n  r = real(a, wp)\n"
        "  call callee(a, x)\n",
    )
    blocks = _rwsets(tmp_path, "rw2", text, "drive")
    assert blocks["B003"]["reads"] == ["a", "pd"], "two positionals are both values"
    assert blocks["B004"]["reads"] == ["a", "pd"], "kind= is a keyword wherever it sits"
    assert blocks["B005"]["reads"] == ["a"], "the second positional of real() is the kind"


def test_a_buffer_out_actual_is_read_as_well_as_written(tmp_path: Path) -> None:
    """``x(:)`` is an intent(out) the callee cannot size, so it is the
    caller's buffer (#36): the emitter passes the actual in *and* unpacks the
    return into it, and the source side has to say the same or every such
    call fails the read/write check (#38)."""
    blocks = _rwsets(tmp_path, "rw", KIND_AND_BUFFER, "drive")
    assert blocks["B003"]["reads"] == ["a", "x"]
    assert blocks["B003"]["writes"] == ["x"]


# --- #20: the target side of a copy-out --------------------------------------


def test_a_copy_out_destination_is_a_write_not_a_read() -> None:
    """``_f_copy_out(dst, src)`` writes into ``dst``. The AST has ``dst`` in
    Load context, so the visitor recorded a read of it and disagreed with
    the Fortran side, which marks the intent(OUT) actual a write (#20). The
    subscripts inside the destination stay reads."""
    import ast

    from recast.verify.rwset import Protocol, span_rwset

    emitted = (
        "_f_copy_out(phis[:, :, ie - 1], fill(a, b))\n"
        "_out = fill(a, b)\n"
        "_f_copy_out(dst, _out[0])\n"
        "wa[0] = _out[1]\n"
    )
    protocol = Protocol(
        procedures=frozenset({"fill"}), scaffolding=frozenset({"_f_copy_out", "_out"})
    )
    reads, writes = span_rwset(ast.parse(emitted), 1, 4, protocol)
    assert writes == {"phis", "dst", "wa"}
    assert reads == {"ie", "a", "b"}
