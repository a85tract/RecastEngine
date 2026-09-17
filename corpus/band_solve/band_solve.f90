! A per-column pentadiagonal solve the way ELM's BandDiagonalMod writes it:
! LAPACK band storage assembled from a (col, band, level) array over the
! column's own window of levels, handed to a bare ``call dgbsv`` that no
! module declares, and the solution written back over the same window.
module band_solve
  implicit none
  private
  public :: band_diagonal
contains
  subroutine band_diagonal(lbj, ubj, ncol, jtop, jbot, numf, filter, nband, b, r, u)
    integer , intent(in)    :: lbj, ubj, ncol
    integer , intent(in)    :: jtop(ncol)
    integer , intent(in)    :: jbot(ncol)
    integer , intent(in)    :: numf
    integer , intent(in)    :: nband
    integer , intent(in)    :: filter(:)
    real(8), intent(in)     :: b(ncol, 1:nband, lbj:ubj)
    real(8), intent(in)     :: r(ncol, lbj:ubj)
    real(8), intent(inout)  :: u(ncol, lbj:ubj)
    integer  :: j, ci, fc, info, m, n
    integer  :: kl, ku
    integer, allocatable :: ipiv(:)
    real(8), allocatable :: ab(:,:), temp(:,:)
    real(8), allocatable :: result(:)

    do fc = 1, numf
       ci = filter(fc)
       kl = (nband-1)/2
       ku = kl
       m = 2*kl+ku+1
       n = jbot(ci)-jtop(ci)+1
       allocate(ab(m,n))
       ab = 0.0
       ab(kl+ku-1,3:n)   = b(ci,1,jtop(ci):jbot(ci)-2)
       ab(kl+ku+0,2:n)   = b(ci,2,jtop(ci):jbot(ci)-1)
       ab(kl+ku+1,1:n)   = b(ci,3,jtop(ci):jbot(ci))
       ab(kl+ku+2,1:n-1) = b(ci,4,jtop(ci)+1:jbot(ci))
       ab(kl+ku+3,1:n-2) = b(ci,5,jtop(ci)+2:jbot(ci))
       allocate(temp(m,n))
       temp = ab
       allocate(ipiv(n))
       allocate(result(n))
       result(:) = r(ci,jtop(ci):jbot(ci))
       call dgbsv( n, kl, ku, 1, ab, m, ipiv, result, n, info )
       u(ci,jtop(ci):jbot(ci)) = result(:)
       if (info /= 0) then
          write(*,*) 'index: ', ci
          write(*,*) 'dgbsv info: ', ci, info
          do j = 1, n
             write(*,'(i2,5f18.7)') j, temp(3:7,j)
          end do
          stop
       end if
       deallocate(temp)
       deallocate(ab)
       deallocate(ipiv)
       deallocate(result)
    end do
  end subroutine band_diagonal
end module band_solve
