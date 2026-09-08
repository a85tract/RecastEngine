! The kinds module ELM carries, in its spelling: the kind is a
! selected_real_kind, named in upper case, renamed downstream. The last
! line is the keyword form (``p=12``) the family's other precision modules
! write it in, which the engine met only through an extension's kind
! assumptions until it was put here.
MODULE shr_kind_mod
  integer,parameter :: SHR_KIND_R8 = selected_real_kind(12) ! 8 byte real
  integer,parameter :: SHR_KIND_R4 = selected_real_kind( 6) ! 4 byte real
  integer,parameter :: SHR_KIND_IN = kind(1)                ! native integer
  integer,parameter :: SHR_KIND_CS = 80                     ! short char
  integer,parameter :: SHR_KIND_DP = selected_real_kind( p=12 )
END MODULE shr_kind_mod
