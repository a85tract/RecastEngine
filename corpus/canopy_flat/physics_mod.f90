module physics_mod
  use types_mod, only: canopy_type, nlev
  use state_mod, only: scale
  use grad_mod, only: conductance
  implicit none
  private
  public :: Warm, Fill
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
    integer :: f, p, ic, k_end, k_end_new, k_in
    real(8) :: hole, stealable
    associate (tleaf => inst%tleaf, ncan => inst%ncan)
    do f = 1, num
       p = filter(f)
       hole = 0.0d0
       do ic = 1, ncan(p)
          hole = hole + min( tleaf(p,ic) - threshold, 0.0d0 )
       end do
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
       end if
    end do
    end associate
  end subroutine Fill
end module physics_mod
