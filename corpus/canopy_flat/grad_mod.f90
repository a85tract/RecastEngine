module grad_mod
  use types_mod, only: canopy_type
  implicit none
  private
  public :: conductance
contains
  ! A function that takes the object, the way CLUBB's gradzm_2d takes gr
  ! and mvr_hm_max takes hm_metadata.
  real(8) function conductance( inst, p )
    type(canopy_type), intent(in) :: inst
    integer, intent(in) :: p
    conductance = 2.0d0 * inst%gs(p)
  end function conductance
end module grad_mod
