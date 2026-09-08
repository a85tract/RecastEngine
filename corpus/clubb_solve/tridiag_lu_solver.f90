! A tridiagonal LU solve written the way CLUBB's tridiag_lu_solver writes
! it: the band stored with a declared lower bound, lhs(-1:1, ndim), the
! superdiagonal at -1 and the subdiagonal at +1, an INOUT left-hand side
! and an OUT solution. The gate once drew the band at its upper bound
! alone -- one row for three -- and every draw subscripted past it; and
! the profile beside this tree, which makes the system diagonally
! dominant, reads the band the way Python does (``lhs[-1]``, the last row)
! and was refused for the subscript the translated body may not form.
module tridiag_lu_solver
  use clubb_precision, only: core_rknd
  implicit none
  private
  public :: tridiag_lu_solve_single_rhs_lhs

contains

  subroutine tridiag_lu_solve_single_rhs_lhs( ndim, lhs, rhs, soln )
    integer, intent(in) :: ndim
    real( kind = core_rknd ), intent(in), dimension(ndim) :: rhs
    real( kind = core_rknd ), intent(inout), dimension(-1:1,ndim) :: lhs
    real( kind = core_rknd ), intent(out), dimension(ndim) :: soln
    real( kind = core_rknd ), dimension(ndim) :: upper, lower_diag_invrs
    integer :: k

    lower_diag_invrs(1) = 1.0_core_rknd / lhs(0,1)
    upper(1) = lower_diag_invrs(1) * lhs(-1,1)
    do k = 2, ndim-1
      lower_diag_invrs(k) = 1.0_core_rknd / ( lhs(0,k) - lhs(1,k) * upper(k-1) )
      upper(k) = lower_diag_invrs(k) * lhs(-1,k)
    end do
    lower_diag_invrs(ndim) = 1.0_core_rknd / ( lhs(0,ndim) - lhs(1,ndim) * upper(ndim-1) )

    soln(1) = lower_diag_invrs(1) * rhs(1)
    do k = 2, ndim
      soln(k) = lower_diag_invrs(k) * ( rhs(k) - lhs(1,k) * soln(k-1) )
    end do
    do k = ndim-1, 1, -1
      soln(k) = soln(k) - upper(k) * soln(k+1)
    end do

    ! The factorisation is left in the band, as CLUBB's solver leaves it.
    do k = 1, ndim
      lhs(0,k) = 1.0_core_rknd / lower_diag_invrs(k)
      lhs(-1,k) = upper(k)
    end do
  end subroutine tridiag_lu_solve_single_rhs_lhs

end module tridiag_lu_solver
