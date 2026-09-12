module physics_mod
  use types_mod, only: canopy_type, nlev
  use state_mod, only: scale
  use grad_mod, only: conductance
  implicit none
  private
  public :: Warm
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
end module physics_mod
