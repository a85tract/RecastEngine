"""The two-file toy tree the f2py oracle tests build: a kinds module and a
module that takes its precision from it. Shared by the plain-oracle tests and
the flat-oracle tests, which live in different files because the flat oracle
is a different plugin."""

from __future__ import annotations

from pathlib import Path

from recast.fortran.frontend import FortranFrontend
from recast.model import Unit

KINDS_SOURCE = """\
module toy_kinds
  use, intrinsic :: iso_fortran_env
  implicit none
  integer, parameter :: wp = real64
end module toy_kinds
"""

SPLIT_SOURCE = """\
module toy_split
  use toy_kinds, only: wp
  implicit none
contains
  subroutine scale_all(n, a, x)
    integer, intent(in) :: n
    real(wp), intent(in) :: a
    real(wp), intent(inout) :: x(*)
    integer :: i
    do i = 1, n
      x(i) = a * x(i)
    end do
  end subroutine scale_all
end module toy_split
"""


def _split_tree(tmp_path: Path) -> tuple[Unit, object]:
    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE)
    (tmp_path / "toy_split.f90").write_text(SPLIT_SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_split")
    return unit, frontend.analyze(unit, tmp_path)
