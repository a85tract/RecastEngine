! The constants module CLUBB's solvers read their band layout from: the
! number of diagonals and the position of each within a row of the
! left-hand side, used as extents and subscripts by every routine here;
! and the PDF selector the variance step switches its scalar loops on.
module constants_clubb
  implicit none
  private
  integer, parameter, public :: ndiags3 = 3
  integer, parameter, public :: km1 = 1
  integer, parameter, public :: k0 = 2
  integer, parameter, public :: kp1 = 3
  integer, parameter, public :: ipdf_adg1 = 1
end module constants_clubb
