! A routine written the way ELM's biogeophysics writes them: kinds and
! constants use-imported from the family's modules, a REAL element stored
! into an INTEGER scalar (the number of radiation layers, carried as a real
! in the state object and truncated at the point of use), character
! parameters read for their length and compared, and the special value
! returned where the layer count is not physical. The port to JAX once emitted the truncation
! under a name its runtime did not define; this is the tree that says so.
module leaf_layers
  use shr_kind_mod, only: r8 => shr_kind_r8
  use shr_kind_mod, only: dp => shr_kind_dp
  implicit none
  private
  public :: layer_temperature

contains

  subroutine layer_temperature( np, t_veg, nrad, t_layer, d13c )
    use elm_varcon, only: tfrz, spval, n_melt, mu, tcrit, degpsec, preind_atm_del13c, &
                           c3_r2, isecspday, ispval, namep, grlnd
    integer, intent(in) :: np
    real(r8), intent(in) :: t_veg(np)
    real(r8), intent(in) :: nrad(np)
    real(r8), intent(out) :: t_layer(np)
    real(r8), intent(out) :: d13c(np)
    integer :: p, nl, i
    real(dp) :: acc
    do p = 1, np
      nl = int(nrad(p))
      if ( nl < 1 .or. nl == ispval ) then
        t_layer(p) = spval
        d13c(p) = spval
        cycle
      end if
      acc = 0.0_dp
      do i = 1, nl
        acc = acc + ( t_veg(p) - tfrz ) * n_melt / real(i, r8) &
              + log( t_veg(p) / tfrz ) * ( tcrit / tfrz ) ** i
      end do
      t_layer(p) = acc + degpsec * real(isecspday, r8) / real(len_trim(namep), r8)
      if ( namep == grlnd ) then
        d13c(p) = preind_atm_del13c
      else
        d13c(p) = preind_atm_del13c + mu * ( t_veg(p) - tfrz ) * c3_r2
      end if
    end do
  end subroutine layer_temperature

end module leaf_layers
