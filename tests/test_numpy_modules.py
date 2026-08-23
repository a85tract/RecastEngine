"""Tests for the module-file renderer.

The differential holds the module body byte-identical to the pipeline over
the corpus; these pin the state-initializer forms the corpus never uses
(HUGE, EPSILON, constant division, the honest TODO), the factory shapes, the
header the differential deliberately does not compare, and the promise that
``py_lines`` in the report point at the marker lines of the finished file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import constants, interface
from recast.transform.numpy.modules import Modules
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.profiles import PROFILES

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module top_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  integer, parameter :: plev = 4
  real(r8) :: grid(plev, 2)
  real(r8), allocatable :: pool(:)
  integer :: count = 3
  logical :: ready = .false.
  real(r8) :: weight = 2.5_r8
  character(len=8) :: tag = 'abc'
  real(r8) :: biggest = huge(1.0_r8)
  real(r8) :: eps_v = epsilon(qmin)
  real(r8) :: ratio = 2.0_r8 / 7.0_r8
  real(r8) :: table(2) = (/ 1.5_r8, 2.5_r8 /)
  integer :: unset
  real(r8) :: mystery = sin(1.0_r8)

  type cell_t
    real(r8) :: buf(3)
    integer :: id
  end type cell_t

contains

  subroutine touch(x)
    real(r8), intent(inout) :: x
    x = x + 1.0_r8
  end subroutine touch
end module top_mod
"""


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mod") / "top_mod.f90"
    path.write_text(SOURCE)
    return path


@pytest.fixture(scope="module")
def renderer(source: Path) -> Modules:
    return Modules(
        subprograms=Subprograms(
            record=interface.extract(source, kind_assumptions=KINDS),
            constants=constants.extract(source),
            profile=PROFILES["ifx"],
        ),
        companion_imports=("import sibling_numpy as _sib",),
    )


# --- module state ------------------------------------------------------------


def test_array_state_becomes_a_zero_buffer(source: Path, renderer: Modules) -> None:
    """Extents spelled with module parameters or digits render; the buffer is
    zeros because the module's init routine fills it (Fortran SAVE)."""
    body, _ = renderer.body(renderer._subprogram_nodes(source))
    assert "grid = np.zeros((PLEV, 2,), dtype=np.float64)  # module array state" in body
    assert "pool = None  # allocatable/assumed module array, set by init" in body


def test_save_initializers_render_by_form(source: Path, renderer: Modules) -> None:
    body, _ = renderer.body(renderer._subprogram_nodes(source))
    tail = "  # module state (%s), Fortran save-init"
    assert "count = 3" + tail % "int32" in body
    assert "ready = False" + tail % "bool" in body
    assert "weight = np.float64('2.5')" + tail % "float64" in body
    assert "tag = 'abc'" + tail % "str" in body
    assert "biggest = np.finfo(np.float64).max  # HUGE(real(r8))" + tail % "float64" in body
    assert "eps_v = np.finfo(np.float64).eps  # EPSILON" + tail % "float64" in body
    assert "ratio = np.float64(0.2857142857142857)" + tail % "float64" in body
    assert "table = np.array([np.float64('1.5'), np.float64('2.5')])" + tail % "float64" in body


def test_an_unrecognized_initializer_is_an_honest_todo(source: Path, renderer: Modules) -> None:
    """``sin(1.0_r8)`` is a constant expression Fortran folds at compile
    time; guessing its value here would bake in a rounding nobody checked.
    ``None`` with the source text attached is a site for a human."""
    body, _ = renderer.body(renderer._subprogram_nodes(source))
    assert any(line.startswith("mystery = None  # TODO: init") for line in body)
    assert "unset = None  # module state (int32), set by init" in body


# --- factories ---------------------------------------------------------------


def test_a_derived_type_gets_a_factory(source: Path, renderer: Modules) -> None:
    body, _ = renderer.body(renderer._subprogram_nodes(source))
    at = body.index("def _make_cell_t():")
    assert body[at + 1 : at + 6] == [
        '    """factory for type(cell_t) (components per Derived_Type_Def)."""',
        "    o = _new_derived()",
        "    o.buf = np.zeros((3,))",
        "    o.id = 0",
        "    return o",
    ]


# --- the header --------------------------------------------------------------


def test_the_header_carries_imports_runtime_and_signatures(renderer: Modules) -> None:
    header = renderer.header()
    assert "import numpy as np" in header
    assert "from constants import *" in header
    assert "import sibling_numpy as _sib" in header
    assert "_RUNTIME = {'abort_msg': None}" in header
    assert "_LIBM_STRICT" in header  # the runtime rode along whole
    assert "def _fstr_eq" in header  # ...including the shim the pipeline strips
    sigs_line = next(line for line in header.splitlines() if line.startswith("_SIGNATURES = "))
    table = ast.literal_eval(sigs_line.removeprefix("_SIGNATURES = "))
    assert table["touch"]["args"][0] == {
        "name": "x",
        "dtype": "float64",
        "intent": "INOUT",
        "optional": False,
    }


def test_the_rendered_file_is_valid_python(source: Path, renderer: Modules) -> None:
    text, _ = renderer.render(source)
    compile(text, "top_mod_numpy.py", "exec")


def test_py_lines_point_at_the_markers_of_the_finished_file(
    source: Path, renderer: Modules
) -> None:
    """The numbers are scanned back out of the final text, not accumulated
    during emission, so they cannot drift from the file they describe."""
    text, report = renderer.render(source)
    lines = text.splitlines()
    for entry in report:
        first, last = entry["py_lines"]
        assert lines[first - 1].startswith(f"    # {entry['block']} <- ")
        assert last >= first
