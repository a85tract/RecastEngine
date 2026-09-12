! CLUBB's error_code, in the shape the project stands in for: the debug
! level is a private module variable with an initializer, one public
! routine sets it from an argument, one public function compares against
! it, and every physics routine asks that function before it does
! anything optional. No `use` reaches the level, and a run may have set it
! to anything -- which is why the flattener leaves it to the module only
! because this module is named in the conventions' stub_modules, and says
! so: both sides run on this declaration. The JAX lowering once refused
! any companion kernel that read it ("which the plan does not carry") and
! dropped the companion to a host call the gate could not see.
module error_code
  implicit none
  private
  public :: set_clubb_debug_level_api, clubb_at_least_debug_level_api
  integer, save :: clubb_debug_level = 0

contains

  subroutine set_clubb_debug_level_api( level )
    integer, intent(in) :: level
    clubb_debug_level = level
  end subroutine set_clubb_debug_level_api

  logical function clubb_at_least_debug_level_api( level )
    integer, intent(in) :: level
    clubb_at_least_debug_level_api = ( level <= clubb_debug_level )
  end function clubb_at_least_debug_level_api

end module error_code
