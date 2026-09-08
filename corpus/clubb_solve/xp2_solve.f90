! A variance step written the way CLUBB's advance_xp2_xpyp_module writes
! it: a public driver, a private routine that assembles a tridiagonal
! system into (ngrdcol, nzm) locals, and a solver whose dummies are one
! rank higher -- rhs(ngrdcol, nzm, nrhs), the solution likewise -- so the
! call hands a rank-2 array to a rank-3 dummy by sequence association,
! INOUT on the way in and OUT on the way back; and, for each passive
! scalar, one slab of a rank-3 local as the solution, in a loop whose
! bound the body hands to a call and whose index is read once the loop is
! done, under a condition on the state.
!
! Each of those failed the port to JAX at some engine commit that passed
! its own suite. The NumPy translation spells the reshaped actual as the
! first ngrdcol*nzm*nrhs cells of the array flattened in column-major
! order; the port could not read that spelling back, dropped the private
! routine's kernel without a note, and left its caller calling a host
! function that does not exist. The anchor assigned the solver's rank-3
! result into a rank-2 slab. The anchor holds the scalar loop's bounds in
! temporaries set beside the loop, and the kernel read them before the
! branch. An OUT array whose leading extent is a module constant was sized
! by its own shape, a name the kernel never receives.
!
! The solver reads its solution back to flip it, as CLUBB's does, so the
! caller's storage is what it writes; and the tree is read under CLUBB's
! convention that every intent(out) array is the caller's buffer. (CLUBB
! hands its tunables around in a derived type, which makes the kernels
! the flat forms; the engine's own port recipe judges what the NumPy
! anchor wraps, so that shape is held in tests/test_jax_transform.py.)
module xp2_solve
  use clubb_precision, only: core_rknd
  use constants_clubb, only: ndiags3, km1, k0, kp1
  implicit none
  private
  public :: advance_xp2

contains

  subroutine advance_xp2( nzm, ngrdcol, sclr_dim, dt, c2rt, l_flip, invrs_tau, rtm, &
                          rtp2, sclrp2, n_solved )
    integer, intent(in) :: nzm, ngrdcol, sclr_dim
    integer, intent(out) :: n_solved
    real( kind = core_rknd ), intent(in) :: dt
    real( kind = core_rknd ), intent(in) :: c2rt
    logical, intent(in) :: l_flip
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: invrs_tau
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: rtm
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(inout) :: rtp2
    real( kind = core_rknd ), dimension(ngrdcol,nzm,sclr_dim), intent(inout) :: sclrp2
    call solve_xp2_with_multiple_lhs( nzm, ngrdcol, sclr_dim, dt, c2rt, l_flip, invrs_tau, &
                                      rtm, rtp2, sclrp2, n_solved )
  end subroutine advance_xp2

  subroutine solve_xp2_with_multiple_lhs( nzm, ngrdcol, sclr_dim, dt, c2rt, l_flip, &
                                          invrs_tau, rtm, rtp2, sclrp2, n_solved )
    integer, intent(in) :: nzm, ngrdcol, sclr_dim
    integer, intent(out) :: n_solved
    real( kind = core_rknd ), intent(in) :: dt
    real( kind = core_rknd ), intent(in) :: c2rt
    logical, intent(in) :: l_flip
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: invrs_tau
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: rtm
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(inout) :: rtp2
    real( kind = core_rknd ), dimension(ngrdcol,nzm,sclr_dim), intent(inout) :: sclrp2
    real( kind = core_rknd ), dimension(ndiags3,ngrdcol,nzm) :: lhs
    real( kind = core_rknd ), dimension(ngrdcol,nzm) :: rhs
    real( kind = core_rknd ), dimension(ngrdcol,nzm) :: rtp2_solution
    real( kind = core_rknd ), dimension(ngrdcol,nzm,sclr_dim) :: sclrp2_solution
    integer :: i, k, sclr

    call xp2_lhs( nzm, ngrdcol, dt, invrs_tau, lhs )
    call xp2_rhs( nzm, ngrdcol, dt, c2rt, rtm, rtp2, rhs )

    ! The rank-2 rhs and solution to the solver's rank-3 dummies, nrhs = 1.
    call xp2_solve_system( nzm, ngrdcol, 1, l_flip, rhs, lhs, rtp2_solution )

    do k = 1, nzm
      do i = 1, ngrdcol
        rtp2(i,k) = max( rtp2_solution(i,k), 0.0_core_rknd )
      end do
    end do

    ! Each passive scalar's variance through the same system, when the
    ! moisture field is not dry: its slab of the rank-3 solution is the
    ! solver's OUT actual; the loop's bound goes to a call in its body,
    ! and its index is read once the loop is done.
    n_solved = 0
    if ( any( rtm > 0.0_core_rknd ) ) then
      do sclr = 1, sclr_dim
        call xp2_lhs( nzm, ngrdcol, dt, invrs_tau, lhs )
        call xp2_sclr_rhs( nzm, ngrdcol, sclr_dim, sclr, dt, c2rt, rtm, sclrp2, rhs )
        call xp2_solve_system( nzm, ngrdcol, 1, l_flip, rhs, lhs, sclrp2_solution(:,:,sclr) )
        do k = 1, nzm
          do i = 1, ngrdcol
            sclrp2(i,k,sclr) = max( sclrp2_solution(i,k,sclr), 0.0_core_rknd )
          end do
        end do
      end do
      n_solved = sclr - 1
    end if
  end subroutine solve_xp2_with_multiple_lhs

  subroutine xp2_sclr_rhs( nzm, ngrdcol, sclr_dim, sclr, dt, c2rt, rtm, sclrp2, rhs )
    integer, intent(in) :: nzm, ngrdcol, sclr_dim, sclr
    real( kind = core_rknd ), intent(in) :: dt
    real( kind = core_rknd ), intent(in) :: c2rt
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: rtm
    real( kind = core_rknd ), dimension(ngrdcol,nzm,sclr_dim), intent(in) :: sclrp2
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(out) :: rhs
    integer :: i, k
    do k = 1, nzm
      do i = 1, ngrdcol
        rhs(i,k) = sclrp2(i,k,sclr) + dt * c2rt * rtm(i,k) * rtm(i,k)
      end do
    end do
  end subroutine xp2_sclr_rhs

  subroutine xp2_lhs( nzm, ngrdcol, dt, invrs_tau, lhs )
    integer, intent(in) :: nzm, ngrdcol
    real( kind = core_rknd ), intent(in) :: dt
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: invrs_tau
    real( kind = core_rknd ), dimension(ndiags3,ngrdcol,nzm), intent(out) :: lhs
    integer :: i, k
    do k = 1, nzm
      do i = 1, ngrdcol
        lhs(km1,i,k) = -0.25_core_rknd * dt * invrs_tau(i,k)
        lhs(k0,i,k) = 1.0_core_rknd + dt * invrs_tau(i,k)
        lhs(kp1,i,k) = -0.25_core_rknd * dt * invrs_tau(i,k)
      end do
    end do
    do i = 1, ngrdcol
      lhs(km1,i,1) = 0.0_core_rknd
      lhs(kp1,i,nzm) = 0.0_core_rknd
    end do
  end subroutine xp2_lhs

  subroutine xp2_rhs( nzm, ngrdcol, dt, c2rt, rtm, rtp2, rhs )
    integer, intent(in) :: nzm, ngrdcol
    real( kind = core_rknd ), intent(in) :: dt
    real( kind = core_rknd ), intent(in) :: c2rt
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: rtm
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(in) :: rtp2
    real( kind = core_rknd ), dimension(ngrdcol,nzm), intent(out) :: rhs
    integer :: i, k
    do k = 1, nzm
      do i = 1, ngrdcol
        rhs(i,k) = rtp2(i,k) + dt * c2rt * rtm(i,k) * rtm(i,k)
      end do
    end do
  end subroutine xp2_rhs

  subroutine xp2_solve_system( nzm, ngrdcol, nrhs, l_flip, rhs, lhs, xapxbp )
    ! The Thomas algorithm, one tridiagonal system per column per right
    ! hand side; the work is done in place on rhs and lhs, the way the
    ! solver wrappers CLUBB calls do it, and the solution is read back to
    ! flip it when the grid runs the other way, as CLUBB's is.
    integer, intent(in) :: nzm, ngrdcol, nrhs
    logical, intent(in) :: l_flip
    real( kind = core_rknd ), dimension(ngrdcol,nzm,nrhs), intent(inout) :: rhs
    real( kind = core_rknd ), dimension(ndiags3,ngrdcol,nzm), intent(inout) :: lhs
    real( kind = core_rknd ), dimension(ngrdcol,nzm,nrhs), intent(out) :: xapxbp
    real( kind = core_rknd ) :: w
    integer :: i, j, k
    do j = 1, nrhs
      do k = 2, nzm
        do i = 1, ngrdcol
          w = lhs(km1,i,k) / lhs(k0,i,k-1)
          lhs(k0,i,k) = lhs(k0,i,k) - w * lhs(kp1,i,k-1)
          rhs(i,k,j) = rhs(i,k,j) - w * rhs(i,k-1,j)
        end do
      end do
      do i = 1, ngrdcol
        xapxbp(i,nzm,j) = rhs(i,nzm,j) / lhs(k0,i,nzm)
      end do
      do k = nzm-1, 1, -1
        do i = 1, ngrdcol
          xapxbp(i,k,j) = ( rhs(i,k,j) - lhs(kp1,i,k) * xapxbp(i,k+1,j) ) / lhs(k0,i,k)
        end do
      end do
    end do
    if ( l_flip ) then
      xapxbp(:,:,:) = xapxbp(:,nzm:1:-1,:)
    end if
  end subroutine xp2_solve_system

end module xp2_solve
