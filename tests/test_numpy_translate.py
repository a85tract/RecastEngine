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
    resolved = resolve(["cpair"], [root / "physconst.f90"])
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
