module physics_mod
  use types_mod, only: canopy_type, nlev
  use state_mod, only: scale
  use grad_mod, only: conductance
  implicit none
  private
  public :: Warm, Fill, Clip
contains
  ! The object-taking function is called inside an expression -- twice in
  ! one sum, as pos_definite_adj multiplies gradzm_2d and
  ! clip_hydromet_conc_mvr cubes mvr_hm_max -- and the expression sits
  ! under an IF whose test is traced, as mvr_hm_max sits under
  ! ``.not. l_frozen_hm(idx)``.
  subroutine Warm(num, filter, dt, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    real(8), intent(in) :: dt
    type(canopy_type), intent(inout) :: inst
    integer :: f, p, ic
    associate (tleaf => inst%tleaf, gs => inst%gs, ncan => inst%ncan)
    do f = 1, num
       p = filter(f)
       do ic = 1, ncan(p)
          if ( gs(p) > 0.0d0 ) then
             tleaf(p,ic) = tleaf(p,ic) + dt * ( conductance(inst, p) + conductance(inst, p) ) * scale
          else
             tleaf(p,ic) = tleaf(p,ic) - dt * conductance(inst, p) * scale
          end if
       end do
    end do
    end associate
  end subroutine Warm

  ! The shape of CLUBB's fill_holes_smart_window: a hole summed over the
  ! layers, a window widened inside a DO WHILE until the mass above the
  ! threshold covers it -- the newly covered layers summed into a scalar
  ! bound before the loop by a DO loop inside its body -- then spread.
  subroutine Fill(num, filter, threshold, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    real(8), intent(in) :: threshold
    type(canopy_type), intent(inout) :: inst
    integer :: f, p, ic, k_end, k_end_new, k_in, n_pass
    real(8) :: hole, stealable
    logical :: l_again
    associate (tleaf => inst%tleaf, ncan => inst%ncan)
    ! Passes over the patches until none is below the threshold, at most
    ! two: a DO WHILE around a DO around an IF around a DO WHILE, the
    ! nesting of fill_holes_parallel.
    l_again = .true.
    n_pass = 0
    do while ( l_again .and. n_pass < 2 )
       l_again = .false.
       n_pass = n_pass + 1
       do f = 1, num
          p = filter(f)
          hole = 0.0d0
          do ic = 1, ncan(p)
             hole = hole + min( tleaf(p,ic) - threshold, 0.0d0 )
          end do
          if ( hole < 0.0d0 ) then
             stealable = max( tleaf(p,1) - threshold, 0.0d0 )
             k_end = 1
             do while ( stealable < abs( hole ) .and. k_end < ncan(p) )
                k_end_new = min( ncan(p), 2 * k_end )
                do k_in = k_end + 1, k_end_new
                   stealable = stealable + max( tleaf(p,k_in) - threshold, 0.0d0 )
                end do
                k_end = k_end_new
             end do
             if ( stealable > 0.0d0 ) then
                do ic = 1, k_end
                   tleaf(p,ic) = tleaf(p,ic) + hole * max( tleaf(p,ic) - threshold, 0.0d0 ) / stealable
                end do
                l_again = .true.
             end if
          end if
       end do
    end do
    end associate
  end subroutine Fill

  ! The shape of CLUBB's fill_holes_widening_windows: a window of the
  ! layers walked in either direction -- k_start:k_end:dir, the edges
  ! found in the data and the step 1 or -1 -- asked with ANY, summed,
  ! written back clipped.
  subroutine Clip(num, filter, threshold, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    real(8), intent(in) :: threshold
    type(canopy_type), intent(inout) :: inst
    integer :: f, p, ic, k_start, k_end, dir
    real(8) :: avg, clipped_avg, frac
    associate (tleaf => inst%tleaf, ncan => inst%ncan)
    do f = 1, num
       p = filter(f)
       dir = 1 - 2 * mod( ncan(p), 2 )
       if ( dir > 0 ) then
          k_start = 1
          k_end = ncan(p)
       else
          k_start = ncan(p)
          k_end = 1
       end if
       if ( any( tleaf(p, k_start:k_end:dir) < threshold ) ) then
          ! the average by a DO loop stepped in the window's direction
          ! (a step found in the data, as fill_holes_widening_windows steps
          ! by window_size * grid_dir_indx)
          avg = 0.0d0
          do ic = k_start, k_end, dir
             avg = avg + tleaf(p, ic)
          end do
          avg = avg / dble( ncan(p) )
          clipped_avg = sum( max( threshold, tleaf(p, k_start:k_end:dir) ) ) / dble( ncan(p) )
          if ( clipped_avg > threshold ) then
             frac = ( avg - threshold ) / ( clipped_avg - threshold )
             tleaf(p, k_start:k_end:dir) = threshold &
                  + frac * ( max( threshold, tleaf(p, k_start:k_end:dir) ) - threshold )
          end if
       end if
    end do
    end associate
  end subroutine Clip
end module physics_mod
