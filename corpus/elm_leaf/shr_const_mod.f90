! shr_const_mod, as ELM carries it: R8 is a private integer parameter renamed from the
! kinds module, the literals spell their kind upper and lower, and the
! derived constants are chains of quotients over earlier ones.
MODULE shr_const_mod
   use shr_kind_mod
   integer(SHR_KIND_IN),parameter,private :: R8 = SHR_KIND_R8
   real(R8),parameter :: SHR_CONST_PI      = 3.14159265358979323846_R8
   real(R8),parameter :: SHR_CONST_CDAY    = 86400.0_R8
   real(R8),parameter :: SHR_CONST_SDAY    = 86164.0_R8
   real(R8),parameter :: SHR_CONST_OMEGA   = 2.0_R8*SHR_CONST_PI/SHR_CONST_SDAY
   real(R8),parameter :: SHR_CONST_BOLTZ   = 1.38065e-23_R8
   real(R8),parameter :: SHR_CONST_AVOGAD  = 6.02214e26_R8
   real(R8),parameter :: SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
   real(R8),parameter :: SHR_CONST_MWDAIR  = 28.966_R8
   real(R8),parameter :: SHR_CONST_MWWV    = 18.016_R8
   real(R8),parameter :: SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
   real(R8),parameter :: SHR_CONST_RWV     = SHR_CONST_RGAS/SHR_CONST_MWWV
   real(R8),parameter :: SHR_CONST_ZVIR    = (SHR_CONST_RWV/SHR_CONST_RDAIR)-1.0_R8
   real(R8),parameter :: SHR_CONST_TKFRZ   = 273.15_R8
   real(R8),parameter :: SHR_CONST_STEBOL  = 5.670374419e-8_R8
   real(R8),parameter :: SHR_CONST_SPVAL   = 1.0e30_R8
   real(R8),parameter :: SHR_CONST_SPVAL_AERODEP= 1.e29_r8
END MODULE shr_const_mod
