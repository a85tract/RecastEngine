module driver_mod
  use types_mod, only: canopy_type
  use physics_mod, only: Warm
  implicit none
contains
  subroutine step(inst, n, filt)
    type(canopy_type), intent(inout) :: inst
    integer, intent(in) :: n, filt(:)
    call Warm(n, filt, 0.5d0, inst)
  end subroutine step
end module driver_mod
