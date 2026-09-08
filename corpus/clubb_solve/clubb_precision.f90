! The precision module CLUBB carries: one working kind, named the way
! every CLUBB routine imports it.
module clubb_precision
  implicit none
  private
  integer, parameter, public :: core_rknd = selected_real_kind( 12 )
end module clubb_precision
