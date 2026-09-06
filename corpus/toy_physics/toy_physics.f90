! The smallest thing worth translating: a column integration with the three
! argument intents, a module parameter, and a function -- enough to walk the
! whole translate recipe and end bit-exact against the compiled original.
module toy_physics
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: gravity = 9.80616_r8

contains

  subroutine settle(n, rho, dz, w, p)
    integer, intent(in) :: n
    real(r8), intent(in) :: rho(n)
    real(r8), intent(in) :: dz(n)
    real(r8), intent(inout) :: w(n)
    real(r8), intent(out) :: p(n)
    integer :: i
    p(1) = rho(1) * gravity * dz(1)
    do i = 2, n
      p(i) = p(i-1) + rho(i) * gravity * dz(i)
      w(i) = w(i) - dz(i) / (1.0_r8 + rho(i))
    end do
  end subroutine settle

  function column_mass(n, rho, dz) result(m)
    integer, intent(in) :: n
    real(r8), intent(in) :: rho(n)
    real(r8), intent(in) :: dz(n)
    real(r8) :: m
    integer :: i
    m = 0.0_r8
    do i = 1, n
      m = m + rho(i) * dz(i)
    end do
  end function column_mass
end module toy_physics
