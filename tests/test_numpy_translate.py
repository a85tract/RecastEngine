"""Tests for the Fortran-to-NumPy Transform.

The emitters below it are held to the pipeline by ``tools/emit_diff.py``;
what is tested here is the contract binding -- that a Unit and its Facts,
straight from the real Frontend, come back as a complete Candidate: files
that compile, a deferred list that is the agent queue, notes that carry the
name protocol, a digest that is reproducible because the Transform claims
``deterministic``, and a refusal to translate a source that changed since
analysis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.errors import ConfigError
from recast.fortran.frontend import FortranFrontend
from recast.fortran.use import resolve
from recast.model import Facts, Unit
from recast.transform.numpy.translate import NumpyTranslation

SOURCE = """\
module wave_mod
  use precision_mod, only: r8 => wp_r8
  use physconst, only: cpair
  implicit none
  real(r8), parameter :: steepness = 3.7_r8
  real(r8) :: state(4)

contains

  subroutine advance(a, n, s)
    real(r8), intent(inout) :: a(10)
    integer, intent(in) :: n
    real(r8), intent(out) :: s
    integer :: i
    s = 0.0_r8
    do i = 1, n
      a(i) = a(i) * steepness / cpair
      s = s + a(i)
    end do
    call missing_sub(s)
  end subroutine advance
end module wave_mod
"""

PHYSCONST = """\
module physconst
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8), parameter :: shr_const_boltz = 1.38065e-23_r8
  real(r8), parameter :: cpair = 1.00464e3_r8 * shr_const_boltz
end module physconst
"""


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tree = tmp_path_factory.mktemp("translate")
    (tree / "wave_mod.f90").write_text(SOURCE)
    (tree / "physconst.f90").write_text(PHYSCONST)
    return tree


@pytest.fixture(scope="module")
def analyzed(root: Path) -> tuple[Unit, Facts]:
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:wave_mod")
    return unit, frontend.analyze(unit, root)


def config_for(root: Path) -> dict[str, object]:
    # ``r8 => wp_r8`` names a kind no file of this tree defines; the run
    # supplies it through ``kind_assumptions``, and so does this.
    resolved = resolve(["cpair"], [root / "physconst.f90"], {"r8": "float64"})
    return {
        "root": root,
        "use_constants": {"module_name": "physconst", "resolved": resolved},
    }


def test_the_candidate_is_the_whole_product(root: Path, analyzed: tuple[Unit, Facts]) -> None:
    unit, facts = analyzed
    transform = NumpyTranslation()
    assert transform.applicable(unit, facts)
    candidate = transform.apply(unit, facts, config_for(root))
    assert sorted(str(p) for p in candidate.files) == [
        "wave_mod_constants.py",
        "wave_mod_numpy.py",
        "wave_mod_use_constants.py",
    ]
    for path, content in candidate.files.items():
        compile(content.decode(), str(path), "exec")


def test_use_imported_constants_agree_by_construction(
    root: Path, analyzed: tuple[Unit, Facts]
) -> None:
    """``cpair`` resolves through the sibling constants module the same
    ``resolve`` call produced, and the generated module spells it CPAIR --
    the same name the Fortran stand-in will carry, which is the parity."""
    unit, facts = analyzed
    candidate = NumpyTranslation().apply(unit, facts, config_for(root))
    module = candidate.files[Path("wave_mod_numpy.py")].decode()
    assert "CPAIR" in module
    use_constants = candidate.files[Path("wave_mod_use_constants.py")].decode()
    assert "SHR_CONST_BOLTZ = np.float64('1.38065e-23')" in use_constants
    assert "CPAIR = (np.float64('1.00464e3') * SHR_CONST_BOLTZ)" in use_constants
    assert "from wave_mod_use_constants import *" in module


def test_the_deferred_list_is_the_agent_queue(root: Path, analyzed: tuple[Unit, Facts]) -> None:
    unit, facts = analyzed
    candidate = NumpyTranslation().apply(unit, facts, config_for(root))
    assert len(candidate.deferred) == 1
    assert candidate.deferred[0].startswith("advance/B003:")
    assert "missing_sub" in candidate.deferred[0]


def test_notes_carry_the_block_report_and_the_name_protocol(
    root: Path, analyzed: tuple[Unit, Facts]
) -> None:
    unit, facts = analyzed
    candidate = NumpyTranslation().apply(unit, facts, config_for(root))
    statuses = {entry["block"]: entry["status"] for entry in candidate.notes["blocks"]}
    assert statuses["B003"] == "agent_queue"
    assert candidate.notes["coverage"] == {"subprograms": ["advance"]}
    # The rename protocol: STEEPNESS in the output is `steepness` in the
    # source, and the read/write cross-check needs that map to undo it.
    assert candidate.notes["renames"]["advance"]["STEEPNESS"] == "steepness"


def test_the_digest_is_reproducible(root: Path, analyzed: tuple[Unit, Facts]) -> None:
    """``deterministic = True`` is a claim conformance can hold the Transform
    to; two applies over the same inputs must agree byte for byte."""
    unit, facts = analyzed
    transform = NumpyTranslation()
    first = transform.apply(unit, facts, config_for(root))
    second = transform.apply(unit, facts, config_for(root))
    assert first.digest() == second.digest()


def test_a_source_that_changed_since_analysis_is_refused(
    root: Path, analyzed: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """Facts carry the digest of what was analyzed; translating a different
    revision would produce a Candidate whose provenance quietly lies."""
    unit, facts = analyzed
    moved = tmp_path / "wave_mod.f90"
    moved.write_text(SOURCE + "! drifted\n")
    (tmp_path / "physconst.f90").write_text(PHYSCONST)
    with pytest.raises(ConfigError, match="changed since analysis"):
        NumpyTranslation().apply(unit, facts, config_for(tmp_path))


def test_subprogram_units_are_not_applicable(root: Path, analyzed: tuple[Unit, Facts]) -> None:
    """The translation renders whole module files; a subprogram Unit is the
    granularity for other transforms, not this one."""
    _, facts = analyzed
    frontend = FortranFrontend()
    sub_unit = next(u for u in frontend.discover(root) if u.kind == "subprogram")
    assert not NumpyTranslation().applicable(sub_unit, facts)


def test_the_digest_does_not_depend_on_where_the_source_lives(
    root: Path, analyzed: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """``deterministic = True`` has to mean across machines, not just across
    runs in one directory. The generated files once carried the absolute path
    they were translated from -- in the module docstring and beside every
    hoisted constant -- so two people translating the same source got two
    digests, and CI regenerating a committed summary reported a change that
    was only a change of address."""
    import shutil

    unit, facts = analyzed
    here = NumpyTranslation().apply(unit, facts, config_for(root))

    elsewhere_root = tmp_path / "somewhere" / "deeper"
    elsewhere_root.mkdir(parents=True)
    for name in ("wave_mod.f90", "physconst.f90"):
        shutil.copy(root / name, elsewhere_root / name)
    moved_unit = next(
        u for u in FortranFrontend().discover(elsewhere_root) if u.uid == "fortran:wave_mod"
    )
    moved_facts = FortranFrontend().analyze(moved_unit, elsewhere_root)
    there = NumpyTranslation().apply(moved_unit, moved_facts, config_for(elsewhere_root))

    assert here.digest() == there.digest()
    for path, content in here.files.items():
        assert str(root) not in content.decode(), f"{path} carries the source's directory"


def test_the_emitters_own_temporaries_are_scaffolding() -> None:
    """A name the emitter invents is machinery, and the verifier has to know.

    ``_out`` holds a multi-output call's tuple for exactly one statement and
    is unpacked on the next lines; the dataflow is to the names it is
    unpacked *into*. ``_g``, ``_be``, ``_lc`` and ``_le`` are the
    ``except ... as`` bindings of the goto, block-exit and named-loop
    catchers. The exception *classes* are already scaffolding because the
    runtime defines them and this list is read out of the runtime -- the
    names they are caught under are defined nowhere a reader could find, so
    the read/write check counted them as data the Fortran never mentions.
    """
    from recast.transform.numpy.translate import _scaffolding_names

    names = _scaffolding_names()
    assert {"_out", "_g", "_be", "_lc", "_le"} <= names
    # Read out of the runtime rather than restated, which is what keeps the
    # catchers' classes in step with the names above.
    assert {"_FGoto", "_FBlockExit", "_FLoopExit", "_FLoopCycle"} <= names
    # And it is still a list of machinery, not a licence to skip data: a
    # Fortran variable that happens to look like one is not in it.
    assert "out" not in names
    assert "result" not in names


def test_the_protocol_names_every_alias_the_header_imports(
    root: Path, analyzed: tuple[Unit, Facts]
) -> None:
    """``_physconst.cpair`` is a read of ``cpair``; the read/write check maps
    it back only for an alias the protocol lists, and a stub or globals-only
    companion alias was imported but not listed -- so CLM-ml's
    ``_pftconmod.pftcon`` came out as a read of ``_pftconmod``."""
    unit, facts = analyzed
    candidate = NumpyTranslation().apply(unit, facts, config_for(root))
    module = candidate.files[Path("wave_mod_numpy.py")].decode()
    imported = {
        line.rsplit(" as ", 1)[-1].strip()
        for line in module.splitlines()
        if line.startswith("import ") and "_numpy as " in line
    }
    assert imported <= set(candidate.notes["rwset"]["aliases"]), imported


def test_an_integer_parameter_divides_the_way_fortran_does(tmp_path: Path) -> None:
    """``integer, parameter :: nrk = runge_kutta_type / 10`` is 4 in Fortran
    and was rendered ``41 / 10`` -- 4.1 -- in the use-constants file."""
    from recast.transform.numpy.constants import use_constants_module

    (tmp_path / "ctl.f90").write_text(
        "module ctl\n  implicit none\n  integer, parameter :: runge_kutta_type = 41\n"
        "  integer, parameter :: nrk = (runge_kutta_type/10)\n"
        "  real(8), parameter :: half = 1.0d0/2\nend module ctl\n"
    )
    resolved = resolve(["nrk", "half"], [tmp_path / "ctl.f90"])
    text = use_constants_module(resolved, "ctl")
    namespace: dict[str, object] = {}
    exec(text, namespace)
    assert namespace["NRK"] == 4 and isinstance(namespace["NRK"], int)
    assert namespace["HALF"] == 0.5


LOGICAL_ARRAYS = """\
module valid_mod
  implicit none
  private
  public :: check
contains
  subroutine check( n, x, l_bad )
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    logical, intent(out) :: l_bad
    logical, dimension(n) :: l_ok
    logical :: l_scalar
    l_ok = x > 0.0
    l_scalar = .true.
    l_bad = any( .not. l_ok .and. x < 1.0 ) .or. .not. l_scalar
  end subroutine check
end module valid_mod
"""


def test_logical_operators_on_arrays_are_elementwise(tmp_path: Path) -> None:
    """``any( .not. l_ok )`` over ``logical, dimension(n) :: l_ok`` (CLUBB's
    new_pdf) is elementwise in Fortran; Python's ``not`` on an array raises.
    Outside a WHERE, an array operand still gets ``~`` / ``&`` / ``|``, and a
    scalar one keeps ``not`` / ``and`` / ``or``."""
    (tmp_path / "valid_mod.f90").write_text(LOGICAL_ARRAYS)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:valid_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    module = candidate.files[Path("valid_mod_numpy.py")].decode()
    assert "(~(l_ok))" in module
    assert " & " in module
    assert "not l_scalar" in module
    assert "not l_ok" not in module


COMPLEX_ROOTS = """\
module roots_mod
  implicit none
  integer, parameter :: rk = selected_real_kind(15)
  private
  public :: quadratic_solve, real_parts, no_kind
contains
  function quadratic_solve( nz, a_coef, b_coef, c_coef ) result( roots )
    integer, intent(in) :: nz
    real( kind = rk ), dimension(nz), intent(in) :: a_coef, b_coef, c_coef
    complex( kind = rk ), dimension(nz,2) :: roots
    real( kind = rk ), dimension(nz) :: determinant
    complex( kind = rk ), dimension(nz) :: sqrt_det
    complex( kind = rk ), parameter :: i_unit = ( 0.0_rk, 1.0_rk )
    determinant = b_coef**2 - 4.0_rk * a_coef * c_coef
    sqrt_det = sqrt( cmplx( determinant, kind = rk ) ) * i_unit * i_unit * (-1.0_rk)
    roots(:,1) = ( -cmplx( b_coef, kind = rk ) + sqrt_det ) / cmplx( 2.0_rk * a_coef, kind = rk )
    roots(:,2) = ( -cmplx( b_coef, kind = rk ) - sqrt_det ) / cmplx( 2.0_rk * a_coef, kind = rk )
  end function quadratic_solve
  subroutine real_parts( nz, a_coef, b_coef, c_coef, re1, im2 )
    integer, intent(in) :: nz
    real( kind = rk ), dimension(nz), intent(in) :: a_coef, b_coef, c_coef
    real( kind = rk ), dimension(nz), intent(out) :: re1, im2
    complex( kind = rk ), dimension(nz,2) :: roots
    roots = quadratic_solve( nz, a_coef, b_coef, c_coef )
    re1 = real( roots(:,1), kind = rk )
    im2 = aimag( roots(:,2) )
  end subroutine real_parts
  subroutine no_kind( x, z )
    real( kind = rk ), intent(in) :: x
    complex( kind = rk ), intent(out) :: z
    z = cmplx( x )
  end subroutine no_kind
end module roots_mod
"""


def test_a_complex_is_spelled_at_its_kind_not_as_float64(tmp_path: Path) -> None:
    """CLUBB's calc_roots: a ``complex(kind = core_rknd)`` result came out
    ``np.zeros(..., dtype=np.float64)`` -- the allocate table's silent
    default for any dtype it did not know -- and ``cmplx(x, kind = k)`` came
    out Python's ``complex(x)``, which takes no array (#20). The dtype is
    complex128, the conversion is ``np.complex128``, the real part of a
    complex is ``np.real``, and a ``cmplx`` without a kind is refused."""
    (tmp_path / "roots_mod.f90").write_text(COMPLEX_ROOTS)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:roots_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    module = candidate.files[Path("roots_mod_numpy.py")].decode()
    assert "roots = np.zeros((nz, 2,), dtype=np.complex128)" in module
    assert "sqrt_det = np.zeros((nz,), dtype=np.complex128)" in module
    assert "np.sqrt(np.complex128(determinant))" in module
    assert "i_unit = np.complex128(complex(0.0, 1.0))" in module
    assert "-np.complex128(b_coef)" in module
    assert "complex(determinant)" not in module and "complex(b_coef)" not in module
    assert "re1[...] = np.real(roots[:, 0])" in module
    assert "im2[...] = np.imag(roots[:, 1])" in module
    assert "    z = 0j" in module
    assert len(candidate.deferred) == 1 and "no_kind" in candidate.deferred[0]
    assert "cmplx without a kind" in candidate.deferred[0]


COMPLEX_CONSTANTS = """\
module cp_mod
  implicit none
  integer, parameter :: rk = selected_real_kind(15)
  complex( kind = rk ), parameter :: i_unit = ( 0.0_rk, 1.0_rk )
  complex, parameter :: i_single = ( 0.0, 1.0 )
  complex( kind = rk ), parameter :: two_c = 2.0_rk
  complex( kind = rk ), parameter :: rich = cmplx( 1.0_rk, 2.0_rk, kind = rk )
  private
  public :: rot, i_unit, i_single, two_c, rich
contains
  subroutine rot( n, z, w )
    integer, intent(in) :: n
    complex( kind = rk ), intent(in) :: z(n)
    complex( kind = rk ), intent(out) :: w(n)
    w = z * i_unit
  end subroutine rot
end module cp_mod
"""


def test_a_module_complex_parameter_is_a_complex_not_a_tuple(tmp_path: Path) -> None:
    """``complex(kind = rk), parameter :: i = (0.0_rk, 1.0_rk)`` came out
    ``I = ( np.float64('0.0') , np.float64('1.0') )`` -- a tuple, which the
    first multiplication raised on. The two-part literal is the complex at
    the declared kind, a real initializer is widened as Fortran widens it,
    and a richer initializer is refused with the reason (#20)."""
    (tmp_path / "cp_mod.f90").write_text(COMPLEX_CONSTANTS)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:cp_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    constants = candidate.files[Path("cp_mod_constants.py")].decode()
    assert "I_UNIT = np.complex128(complex(np.float64('0.0'), np.float64('1.0')))" in constants
    assert "I_SINGLE = np.complex64(complex(np.float64(np.float32('0.0'))" in constants
    assert "TWO_C = np.complex128(np.float64('2.0'))" in constants
    assert "# SKIPPED RICH = " in constants and "two-part literal" in constants
    namespace: dict[str, object] = {}
    exec(constants, namespace)
    assert namespace["I_UNIT"] == 1j and namespace["I_UNIT"].dtype == "complex128"
    assert namespace["I_SINGLE"].dtype == "complex64"


SEARCH_LOOP = """\
module search_mod
  implicit none
  private
  public :: first_above
contains
  subroutine first_above( n, z, zmax, k_found, k_scan )
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: z
    real, intent(in) :: zmax
    integer, intent(out) :: k_found, k_scan
    integer :: k, kk
    do k = 1, n
      if ( z(k) > zmax ) exit
    end do
    k_found = k
    do kk = 1, n, 2
      k_scan = kk
    end do
    ! The bounds of a loop over another variable read kk: still a read.
    do k = 1, kk
      k_scan = k_scan + 0
    end do
    k_scan = kk
  end subroutine first_above
end module search_mod
"""


def test_a_loop_index_read_after_the_loop_has_the_completion_value(tmp_path: Path) -> None:
    """CLUBB's lscale_width_vert_avg searches with ``do k = ...; if (...)
    exit; end do`` and integrates up to ``k`` afterwards. On completion
    Fortran leaves the index one step past the end -- ``n + 1`` for a unit
    step, the first odd value past ``n`` for a step of two -- and after an
    EXIT it keeps the exit value. Python's ``for`` leaves the last value."""
    import importlib
    import sys

    (tmp_path / "search_mod.f90").write_text(SEARCH_LOOP)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:search_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    text = (out / "search_mod_numpy.py").read_text()
    # The first two loops' indices are read after them (one in a later
    # loop's bounds); the last loop's index k is not.
    assert text.count("max(0, ") == 2
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("search_mod_numpy")
        import numpy as np

        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        k_found, k_scan = module.first_above(5, z, np.float32(3.5))
        assert (k_found, k_scan) == (4, 7)  # exit at z(4); 1,3,5 then one step past
        k_found, k_scan = module.first_above(5, z, np.float32(9.0))
        assert (k_found, k_scan) == (6, 7)  # completed: one past n
    finally:
        sys.path.remove(str(out))
        sys.modules.pop("search_mod_numpy", None)


STUBBED_PATH = """
module lapack_wrap
  implicit none
contains
  subroutine band_solvex( n, a, x )
    integer, intent(in) :: n
    real, intent(inout) :: a(n)
    real, intent(out) :: x(n)
    x = a
  end subroutine band_solvex
end module lapack_wrap
"""

CHOOSES_A_SOLVER = """
module solver_mod
  use lapack_wrap, only: band_solvex
  implicit none
contains
  subroutine solve( method, n, m, a, x )
    integer, intent(in) :: method, n, m
    real, intent(inout) :: a(n)
    real, intent(out) :: x(n)
    real :: work(n, max(2, m))
    work = 0.0
    if ( method == 1 ) then
      call band_solvex( n, a, x )
    else
      x = a + work(:, 1)
    end if
  end subroutine solve
end module solver_mod
"""


def test_a_call_into_a_stubbed_module_raises_on_its_own_line(tmp_path: Path) -> None:
    """CLUBB's matrix_solver_wrapper chooses LAPACK or its own LU solver by
    a run-time flag; ``lapack_wrap`` is stubbed. With no rule for the call
    the whole IF was deferred -- condition and LU branch included -- and the
    candidate raised on the path the run takes. The raise belongs to the
    statement; the branch around it stays. And ``work(n, max(2, m))``
    (windm's ``rhs``) is a bound Python can spell."""
    import importlib
    import sys

    (tmp_path / "lapack_wrap.f90").write_text(STUBBED_PATH)
    (tmp_path / "solver_mod.f90").write_text(CHOOSES_A_SOLVER)
    frontend = FortranFrontend(stub_modules=["lapack_wrap"])
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solver_mod")
    facts = frontend.analyze(unit, tmp_path)
    assert facts.interface["stub_procedures"] == ["band_solvex"]
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert not candidate.deferred
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    text = (out / "solver_mod_numpy.py").read_text()
    assert "max(2, m)" in text
    raised = "raise NotImplementedError('band_solvex: procedure of a stubbed module, not ported')"
    assert raised in text
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("solver_mod_numpy")
        import numpy as np

        a = np.array([1.0, 2.0], dtype=np.float32)
        _a, x = module.solve(2, 2, 1, a)  # INOUT a and OUT x come back
        assert x.tolist() == [1.0, 2.0]
        with pytest.raises(NotImplementedError):
            module.solve(1, 2, 1, a)
    finally:
        sys.path.remove(str(out))
        sys.modules.pop("solver_mod_numpy", None)


HANDS_ON_AN_OPTIONAL = """
module relay_mod
  implicit none
contains
  subroutine inner( x, y, rc )
    real, intent(in) :: x
    real, intent(out) :: y
    real, intent(out), optional :: rc
    y = 2.0 * x
    if ( present(rc) ) rc = 1.0 / x
  end subroutine inner
  subroutine outer( x, y, rc, scale )
    real, intent(in) :: x
    real, intent(out) :: y
    real, intent(out), optional :: rc
    real, intent(in), optional :: scale
    call inner( x, y, rc = rc )
    if ( present(scale) ) y = y * scale
  end subroutine outer
end module relay_mod
"""


def test_an_optional_handed_on_carries_its_own_presence(tmp_path: Path) -> None:
    """``call inner( x, y, rc = rc )`` where ``rc`` is the caller's own
    optional OUT: present in the callee exactly when present in the caller.
    Rendered ``want_rc=True`` it was always present, and CLUBB's
    xm_wpxp_solve took the LAPACK diagnostic path on every call."""
    import importlib
    import sys

    (tmp_path / "relay_mod.f90").write_text(HANDS_ON_AN_OPTIONAL)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:relay_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    text = (out / "relay_mod_numpy.py").read_text()
    assert "want_rc=want_rc" in text
    assert "want_rc=True" not in text
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("relay_mod_numpy")
        import numpy as np

        y, rc = module.outer(np.float32(4.0))
        assert y == 8.0 and rc != 0.25  # not asked for, not computed
        y, rc = module.outer(np.float32(4.0), want_rc=True)
        assert y == 8.0 and rc == 0.25
    finally:
        sys.path.remove(str(out))
        sys.modules.pop("relay_mod_numpy", None)


SCALED_CONSTRUCTOR = """
module fit_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(15)
contains
  subroutine polynomial( n, x, y )
    integer, intent(in) :: n
    real(r8), intent(in) :: x(n)
    real(r8), intent(out) :: y(n)
    real(r8), dimension(3), parameter :: &
      a = 100._r8 * (/ 6.09868993_r8, 0.499320233_r8, 0.184672631E-01_r8 /)
    real(r8), dimension(3), parameter :: b = (/ 1._r8, 2._r8, 3._r8 /) / 2._r8
    integer :: i
    do i = 1, n
      y(i) = a(1) + a(2) * x(i) + a(3) * x(i)**2 + b(3)
    end do
  end subroutine polynomial
end module fit_mod
"""


def test_a_constant_expression_over_an_array_constructor_is_a_value(tmp_path: Path) -> None:
    """CLUBB's saturation and pdf_closure (#26): ``100._core_rknd * (/ ... /)``
    as a local parameter. The token pass rendered a bare constructor and
    handed anything around one to the parser, whose literals were never
    hoisted; the whole subprogram was a NotImplementedError."""
    import importlib
    import sys

    (tmp_path / "fit_mod.f90").write_text(SCALED_CONSTRUCTOR)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:fit_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert not candidate.deferred
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    text = (out / "fit_mod_numpy.py").read_text()
    assert "a = 100. * np.array([6.09868993, 0.499320233, 0.184672631E-01])" in text
    assert "b = np.array([1., 2., 3.]) / 2." in text
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("fit_mod_numpy")
        import numpy as np

        y = module.polynomial(2, np.array([0.0, 1.0]))
        a = 100.0 * np.array([6.09868993, 0.499320233, 0.184672631e-01])
        # To rounding: the point is the constructor, not the summation order.
        assert np.allclose(y, [a[0] + 1.5, a[0] + a[1] + a[2] + 1.5], rtol=0, atol=1e-9)
    finally:
        sys.path.remove(str(out))
        sys.modules.pop("fit_mod_numpy", None)


CALLBACK = """\
module callback_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  abstract interface
    subroutine func(n, x, fvec, iflag)
      import :: r8
      implicit none
      integer, intent(in) :: n
      real(r8), intent(in) :: x(n)
      real(r8), intent(out) :: fvec(n)
      integer, intent(inout) :: iflag
    end subroutine func
  end interface
contains
  subroutine sweep(fcn, n, x, work, iflag)
    procedure(func) :: fcn
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(n)
    real(r8), intent(inout) :: work(n)
    integer, intent(inout) :: iflag
    integer :: j
    do j = 1, n
      x(j) = x(j) + 1.0_r8
      call fcn(n, x, work, iflag)
    end do
  end subroutine sweep

  subroutine drive(fcn, n, x, work, iflag)
    procedure(func) :: fcn
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(n)
    real(r8), intent(inout) :: work(n)
    integer, intent(inout) :: iflag
    call sweep(fcn, n, x, work, iflag)
  end subroutine drive
end module callback_mod
"""


@pytest.fixture(scope="module")
def callback_candidate(tmp_path_factory: pytest.TempPathFactory):
    tree = tmp_path_factory.mktemp("callback")
    (tree / "callback_mod.f90").write_text(CALLBACK)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:callback_mod")
    facts = frontend.analyze(unit, tree)
    return unit, NumpyTranslation().apply(unit, facts, {"root": tree})


def test_a_callback_taking_module_translates_whole(callback_candidate) -> None:
    """A dummy procedure argument is the shape every MINPACK-style driver has.
    Refusing the call left the caller's whole block in the agent queue -- and
    with it every subprogram that reaches its result through one."""
    _unit, candidate = callback_candidate
    assert candidate.deferred == []
    module = candidate.files[Path("callback_mod_numpy.py")].decode()
    assert "_out = fcn(n, x, iflag)" in module
    assert "_f_copy_out(work, _out[0])" in module


def test_the_callback_translation_passes_the_dataflow_gate(callback_candidate) -> None:
    """The two halves have to agree about the same call: the source's read of
    ``fcn`` at callee position, and the write the copy-out shim makes."""
    from recast.executors.local import LocalExecutor
    from recast.model import Confidence
    from recast.verify.rwset import ReadWriteSetVerifier

    unit, candidate = callback_candidate
    verdict = ReadWriteSetVerifier().check(unit, candidate, Path("."), LocalExecutor(), {})
    assert verdict.confidence is Confidence.SAMPLED, verdict.detail
    assert verdict.metrics["blocks_matched"] == verdict.metrics["blocks_checked"] > 0


ABSENT_STUB = """\
module uses_absent
  use absent_mod, only: c0, do_thing
  use flags_mod, only: mask
  implicit none
contains
  subroutine step(n, x, l, r)
    integer, intent(in) :: n
    real(8), intent(inout) :: x(n)
    logical, intent(in) :: l(n)
    logical, intent(out) :: r
    x = x * c0
    call do_thing(x)
    r = any(.not. l)
    if (.not. mask) then
      x = -x
    end if
  end subroutine step
end module uses_absent
"""


def test_a_stub_module_the_tree_lacks_is_recorded_as_assumed(tmp_path: Path) -> None:
    """``use absent_mod, only: c0, do_thing`` from a stubbed module no file
    defines: nothing says which name is a procedure. Every one was taken
    for a procedure, silently, and a read of the constant ``c0`` dropped
    from both sides of the read/write check (ledger #32 row 21). The
    assumption is on the record now, and ``stub_procedure_names`` says
    what the tree cannot. The call that no stub answers is rendered as a
    raise, as before, and the block report names it (row 7); ``.not.
    mask`` over a name of unknown rank is spelled scalar and named (row 5)."""
    (tmp_path / "uses_absent.f90").write_text(ABSENT_STUB)
    frontend = FortranFrontend(stub_modules=["absent_mod", "flags_mod"])
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:uses_absent")
    facts = frontend.analyze(unit, tmp_path)
    assert facts.interface["stub_procedures"] == ["c0", "do_thing", "mask"]
    assert facts.interface["stub_procedures_assumed"] == {
        "absent_mod": ["c0", "do_thing"],
        "flags_mod": ["mask"],
    }
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    blocks = {b["block"]: b for b in candidate.notes["blocks"]}
    called = next(b for b in blocks.values() if b.get("not_ported"))
    assert called["not_ported"] == ["do_thing"] and called["status"] == "mechanical"
    guarded = next(b for b in blocks.values() if b.get("assumed_scalar"))
    assert guarded["assumed_scalar"] == ["mask"]
    text = candidate.files[Path("uses_absent_numpy.py")].decode()
    assert "np.any((~(l)))" in text, "a declared array keeps the elementwise spelling"
    assert "not _flags_mod.mask" in text, "the unknown rank is spelled scalar, and said"

    told = FortranFrontend(
        stub_modules=["absent_mod", "flags_mod"],
        stub_procedure_names={"absent_mod": ["do_thing"], "flags_mod": []},
    )
    facts = told.analyze(unit, tmp_path)
    assert facts.interface["stub_procedures"] == ["do_thing"]
    assert "stub_procedures_assumed" not in facts.interface
    assert facts.provenance["stub_procedure_names"] == {"absent_mod": ["do_thing"], "flags_mod": []}


INTEGER_QUOTIENTS = """\
module quot_mod
  implicit none
  integer, parameter :: fbs = 64
  integer, parameter :: hbs = fbs / 2
  integer, parameter :: mixed = fbs / 3 * 2
contains
  subroutine halves(n, out)
    integer, intent(in) :: n
    integer, intent(out) :: out(3)
    integer, parameter :: h(3) = (/ 1, 2, 3 /) / 2
    integer, parameter :: g = 3 / 2
    out = h + g
  end subroutine halves
end module quot_mod
"""


def test_an_integer_parameter_quotient_truncates(tmp_path: Path) -> None:
    """``integer, parameter :: h(3) = (/1, 2, 3/) / 2`` and ``g = 3 / 2``:
    the local-parameter token pass has no integer division and rendered
    float64 1.5s where Fortran truncates (ledger #32 row 18); an integer
    initializer with a quotient takes the parse path and ``_f_int_div``.
    At module level the constants renderer spells the one shape its flat
    tokens can, ``int(A / B)``, and refuses a quotient inside a larger
    expression rather than folding it wrong."""
    import importlib
    import sys

    (tmp_path / "quot_mod.f90").write_text(INTEGER_QUOTIENTS)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:quot_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    body = candidate.files[Path("quot_mod_numpy.py")].decode()
    assert "_f_int_div(" in body, body
    constants = candidate.files[Path("quot_mod_constants.py")].decode()
    assert "HBS = int(FBS / 2)" in constants, constants
    assert "# SKIPPED MIXED" in constants and "integer division inside" in constants, constants
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        import numpy as np

        module = importlib.import_module("quot_mod_numpy")
        assert importlib.import_module("quot_mod_constants").HBS == 32
        got = module.halves(3)  # the OUT array is returned, by the emitted convention
        assert np.asarray(got).tolist() == [1, 2, 2]  # (0, 1, 1) + 1, as Fortran has it
    finally:
        sys.path.remove(str(out))
        for suffix in ("_numpy", "_constants", "_use_constants"):
            sys.modules.pop(f"quot_mod{suffix}", None)


LOOP_INDEX_READ = """\
module idx_mod
  implicit none
contains
  subroutine find(n, a, flag, k, y)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    logical, intent(in) :: flag
    integer, intent(out) :: k
    real(8), intent(out) :: y
    integer :: j
    do j = 1, n
      if (a(j) > 0.5d0) exit
    end do
    if (flag) then
      j = 0
    else
      y = real(j, 8)
    end if
    do k = 1, n
      if (a(k) < 0.0d0) exit
    end do
  end subroutine find
end module idx_mod
"""


def test_a_loop_index_read_in_a_sibling_arm_or_by_the_caller_completes(tmp_path: Path) -> None:
    """Fortran leaves a DO index one step past the end when the loop runs
    out; Python leaves the last value. The completion was emitted only when
    a line-order scan saw a read before a redefinition: ``j = 0`` in one IF
    arm hid the read of ``j`` in the other, and a dummy index (``k``) the
    caller reads got nothing (ledger #32 row 10). Both complete now."""
    import importlib
    import sys

    (tmp_path / "idx_mod.f90").write_text(LOOP_INDEX_READ)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:idx_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        import numpy as np

        module = importlib.import_module("idx_mod_numpy")
        k, y = module.find(3, np.array([0.1, 0.2, 0.3]), False)
        assert (int(k), float(y)) == (4, 4.0), "both loops ran out: index is n + 1"
    finally:
        sys.path.remove(str(out))
        for suffix in ("_numpy", "_constants", "_use_constants"):
            sys.modules.pop(f"idx_mod{suffix}", None)


OPTIONAL_INOUT = """\
module optinout_mod
  implicit none
contains
  subroutine bump(n, x, y, count)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n)
    integer, intent(inout), optional :: count
    y = y + x
    if (present(count)) count = count + 1
  end subroutine bump
  subroutine twice(n, x, y, z, hits)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n, 2)
    real(8), intent(out) :: z(n)
    integer, intent(inout) :: hits
    call bump(n, x, y(:, 1), count = hits)
    call bump(n, x, y(:, 2))
    z = y(:, 1) + y(:, 2)
  end subroutine twice
end module optinout_mod
"""


def test_an_absent_optional_inout_leaves_its_slot_in_the_return(tmp_path: Path) -> None:
    """The callee returns every OUT and INOUT dummy, optional ones included.
    A caller that leaves an optional INOUT out (CLUBB's passive-scalar call
    of xm_wpxp_clipping_and_stats, ``wpxp_cl_num`` absent) unpacked one
    value too few -- and only the direct unpacking form, taken when an
    actual is an array section, was short. The slot is skipped with ``_``,
    as an absent optional OUT's already was."""
    import importlib
    import sys

    (tmp_path / "optinout_mod.f90").write_text(OPTIONAL_INOUT)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:optinout_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert not candidate.deferred, candidate.deferred
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        import numpy as np

        module = importlib.import_module("optinout_mod_numpy")
        y = np.zeros((3, 2), order="F")
        y_after, z, hits = module.twice(3, np.array([1.0, 2.0, 3.0]), y, 5)
        assert np.asarray(y_after).tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        assert np.asarray(z).tolist() == [2.0, 4.0, 6.0]
        assert int(hits) == 6
    finally:
        sys.path.remove(str(out))
        for suffix in ("_numpy", "_constants", "_use_constants"):
            sys.modules.pop(f"optinout_mod{suffix}", None)
