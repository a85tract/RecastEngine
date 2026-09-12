module types_mod
  implicit none
  integer, parameter :: nlev = 3
  type :: canopy_type
     real(8), pointer :: tleaf(:,:)
     real(8), pointer :: gs(:)
     integer, pointer :: ncan(:)
  contains
     procedure :: Init
  end type canopy_type
contains
  subroutine Init(this, begp, endp)
    class(canopy_type) :: this
    integer, intent(in) :: begp, endp
    allocate(this%tleaf(begp:endp, 1:nlev))
    allocate(this%gs(begp:endp))
    allocate(this%ncan(begp:endp))
  end subroutine Init
end module types_mod
