! A clipping routine the variance step calls, written the way CLUBB's
! clip_explicit and pos_definite_adj are: the physics first, then a
! debug-level check on error_code's private level deciding whether to
! flag what was clipped. Called from the gated unit, so it is a companion
! whose kernel the step inlines -- or, when the lowering refuses it, a
! call back into the NumPy module that the gate passes all the same, and
! that the summary names in `host_calls` or `refused`.
module clip_module
  use clubb_precision, only: core_rknd
  use error_code, only: clubb_at_least_debug_level_api
  implicit none
  private
  public :: clip_variance

contains

  subroutine clip_variance( nzm, ngrdcol, x, n_clipped )
    integer, intent(in) :: nzm, ngrdcol
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(inout) :: x
    integer, intent(out) :: n_clipped
    integer :: i, k
    ! At debug level 2 and above CLUBB counts what it clips and reports
    ! it; below, it clips silently. The level's answer is part of the
    ! traced condition, not a guard of its own, so the kernel has to carry
    ! the stand-in's value rather than ask the host at trace time.
    n_clipped = 0
    do k = 1, nzm
      do i = 1, ngrdcol
        if ( x(i,k) < 0.0_core_rknd .and. clubb_at_least_debug_level_api( 2 ) ) then
          n_clipped = n_clipped + 1
        end if
        if ( x(i,k) < 0.0_core_rknd ) then
          x(i,k) = 0.0_core_rknd
        end if
      end do
    end do
  end subroutine clip_variance

end module clip_module
