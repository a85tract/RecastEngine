! Written the way ELM's SoilMoistStressMod and the SimpleMathMod under it
! are. The method switch is a private module variable with no initializer
! that one public argument-less init routine fixes to a module parameter
! (``init_root_moist_stress``); the plan carries it by calling the setter,
! and a run that never called it would fail the gate with that line to
! point at. The generic has two specifics of the same arity that differ by
! the type of one argument alone, and the kernel calls it with computed
! actuals -- ``nlev + 1``, ``size(work)``, ``max(nlev, 2)``, ``-1``, a real
! expression -- the way ``array_normalization(bounds%begp, ...)`` is called.
! The frontend once read every computed actual as a wildcard and refused
! each of these calls as ambiguous, while its own suite stayed green. The
! specific divides by ``real(count, r8)``: a cast of a dummy the caller
! computes, which the port once spelled with a NumPy constructor and
! refused as a tracer where the kernel was inlined (#62).
module soil_stress
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  private
  public :: init_root_stress, stress_layers

  integer, parameter :: moist_stress_clm_default = 1
  integer, parameter :: moist_stress_ncar = 2
  integer :: root_stress_method

  interface normalize
    module procedure normalize_by_count, normalize_by_scale
  end interface normalize

contains

  subroutine init_root_stress()
    root_stress_method = moist_stress_clm_default
  end subroutine init_root_stress

  subroutine normalize_by_count( count, n, arr )
    integer, intent(in) :: count
    integer, intent(in) :: n
    real(r8), intent(inout) :: arr(n)
    integer :: j
    do j = 1, n
      arr(j) = arr(j) / real(count, r8)
    end do
  end subroutine normalize_by_count

  subroutine normalize_by_scale( scale, n, arr )
    real(r8), intent(in) :: scale
    integer, intent(in) :: n
    real(r8), intent(inout) :: arr(n)
    integer :: j
    do j = 1, n
      arr(j) = arr(j) / scale
    end do
  end subroutine normalize_by_scale

  subroutine stress_layers( np, nlev, rootfr, h2osoi, btran )
    integer, intent(in) :: np
    integer, intent(in) :: nlev
    real(r8), intent(in) :: rootfr(nlev)
    real(r8), intent(in) :: h2osoi(np, nlev)
    real(r8), intent(out) :: btran(np)
    integer :: p, j
    real(r8) :: work(nlev)
    do p = 1, np
      do j = 1, nlev
        work(j) = rootfr(j) * h2osoi(p, j)
      end do
      select case (root_stress_method)
      case (moist_stress_clm_default)
        call normalize( nlev + 1, nlev, work )
        call normalize( size(work), nlev, work )
        call normalize( max(nlev, 2), nlev, work )
      case (moist_stress_ncar)
        call normalize( -1, nlev, work )
        call normalize( 2.0_r8 * h2osoi(p, 1), nlev, work )
      end select
      btran(p) = 0.0_r8
      do j = 1, nlev
        btran(p) = btran(p) + work(j)
      end do
    end do
  end subroutine stress_layers

end module soil_stress
