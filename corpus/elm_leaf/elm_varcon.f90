! elm_varcon's shapes, each one a refusal or a wrong number somewhere once:
! a default-real literal stored in a double (rounded to single first), a
! negative one, an integer parameter set from a real (truncated), and the
! character names the framework passes around -- which the constant fold
! refused outright for a while, taking every ELM unit with it.
module elm_varcon
  use shr_kind_mod, only: r8 => shr_kind_r8
  use shr_const_mod, only: SHR_CONST_TKFRZ, SHR_CONST_CDAY, SHR_CONST_PI, SHR_CONST_RDAIR
  implicit none
  save
  real(r8), parameter :: n_melt = 0.7
  real(r8), parameter :: mu = 0.13889
  real(r8), parameter :: tcrit = 2.5_r8
  real(r8), public, parameter :: degpsec = 15._r8/3600.0_r8
  real(r8), public, parameter :: secspday = SHR_CONST_CDAY
  integer,  public, parameter :: isecspday = secspday
  real(r8), public, parameter :: spval = 1.e36_r8
  integer,  public, parameter :: ispval = -9999
  real(r8), parameter :: preind_atm_del13c = -6.0
  real(r8), parameter :: c3_del13c = -28._r8
  real(r8), parameter :: c3_r1 = 0.0112372_r8 * (1._r8 + c3_del13c/1000._r8)
  real(r8), parameter :: c3_r2 = c3_r1/(1._r8 + c3_r1)
  real(r8), parameter :: tfrz = SHR_CONST_TKFRZ
  real(r8), parameter :: rpi  = SHR_CONST_PI
  real(r8), parameter :: rair = SHR_CONST_RDAIR
  character(len=16), parameter :: grlnd = 'lndgrid'
  character(len=16), parameter :: namep = 'pft'
end module elm_varcon
