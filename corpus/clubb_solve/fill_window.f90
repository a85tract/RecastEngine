! The shape of CLUBB's fill_holes_widening_windows: a window of a column
! walked in the grid's direction -- k_start:k_end:grid_dir_indx, the step a
! dummy that is 1 or -1 at run time -- asked whether any level is below
! the threshold, summed with its weights, and written back clipped. The
! translation once spelled every such section as an ascending Python
! slice, one element short when the step was negative and empty at the
! first level; every recorded CLUBB case ran the ascending grid, so no
! gate saw it (#75).
module fill_window
  use clubb_precision, only: core_rknd
  implicit none
  private
  public :: fill_window_column, first_hole

contains

  ! The first level below the threshold, walking the column in the grid's
  ! direction; 0 when there is none. Written the way fill_holes_smart_window
  ! is: the INTEGER level is zeroed with the tree's REAL ``zero``, which
  ! Fortran converts on assignment (#78).
  subroutine first_hole( nzm, k_start, k_end, grid_dir_indx, threshold, field, k_first, value )
    use constants_clubb, only: zero
    integer, intent(in) :: nzm, k_start, k_end, grid_dir_indx
    real( kind = core_rknd ), intent(in) :: threshold
    real( kind = core_rknd ), dimension(nzm), intent(in) :: field
    integer, intent(out) :: k_first
    real( kind = core_rknd ), intent(out) :: value
    integer :: k

    k_first = zero
    do k = k_start, k_end, grid_dir_indx
      if ( field(k) < threshold .and. k_first == 0 ) then
        k_first = k
      end if
    end do
    value = field( max( k_first, 1 ) )

  end subroutine first_hole

  subroutine fill_window_column( nzm, k_start, k_end, grid_dir_indx, threshold, &
                                 rho_ds_dz, field, field_avg )
    integer, intent(in) :: nzm, k_start, k_end, grid_dir_indx
    real( kind = core_rknd ), intent(in) :: threshold
    real( kind = core_rknd ), dimension(nzm), intent(in) :: rho_ds_dz
    real( kind = core_rknd ), dimension(nzm), intent(inout) :: field
    real( kind = core_rknd ), intent(out) :: field_avg
    real( kind = core_rknd ) :: invrs_denom, clipped_avg, mass_fraction

    field_avg = 0.0_core_rknd
    if ( any( field(k_start:k_end:grid_dir_indx) < threshold ) ) then
      invrs_denom = 1.0_core_rknd / sum( rho_ds_dz(k_start:k_end:grid_dir_indx) )
      field_avg = sum( rho_ds_dz(k_start:k_end:grid_dir_indx) &
                       * field(k_start:k_end:grid_dir_indx) ) * invrs_denom
      clipped_avg = sum( rho_ds_dz(k_start:k_end:grid_dir_indx) &
                         * max( threshold, field(k_start:k_end:grid_dir_indx) ) ) * invrs_denom
      if ( clipped_avg > threshold ) then
        mass_fraction = ( field_avg - threshold ) / ( clipped_avg - threshold )
        field(k_start:k_end:grid_dir_indx) &
          = threshold + mass_fraction * ( max( threshold, field(k_start:k_end:grid_dir_indx) ) &
                                          - threshold )
      end if
    end if

  end subroutine fill_window_column

end module fill_window
