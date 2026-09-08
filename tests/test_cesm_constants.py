"""The constant shapes the CESM-family extensions' trees carry, resolved here.

The constant folds learned to refuse a kind they cannot place (3ba13ea) and
the refusals landed on the extensions: every ELM unit failed at the transform
on ``character(len=16), parameter :: namep = 'pft'`` for the 94 commits until
someone re-gated. The engine's own suite and corpus carried none of the
shapes ``shr_const_mod``, ``elm_varcon``, ``elm_varpar`` and
``constants_clubb`` are written in, so the engine was green while the
extensions were red.

This file is that list. Every declaration below is one an extension's tree
has, in the spelling it has it, and the expected value is what gfortran
stores (``print '(z16.16)'`` of each constant, pinned). A fold change that
refuses one of them, or folds it to a different number, fails here rather
than in a recording gate weeks later.

Two shapes the resolver does not take yet are listed as strict expected
failures at the bottom, so the day they are taught, this file says so.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from recast.fortran.expr import UnsupportedExpression
from recast.fortran.use import resolve
from recast.transform.numpy.constants import use_constants_module

pytestmark = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="the expected values are gfortran's"
)

SHR_KIND_MOD = """\
MODULE shr_kind_mod
  integer,parameter :: SHR_KIND_R8 = selected_real_kind(12) ! 8 byte real
  integer,parameter :: SHR_KIND_R4 = selected_real_kind( 6) ! 4 byte real
  integer,parameter :: SHR_KIND_IN = kind(1)                ! native integer
  integer,parameter :: SHR_KIND_CS = 80                     ! short char
END MODULE shr_kind_mod
"""

SHR_CONST_MOD = """\
MODULE shr_const_mod
   use shr_kind_mod
   integer(SHR_KIND_IN),parameter,private :: R8 = SHR_KIND_R8  ! rename for local readability only
   real(R8),parameter :: SHR_CONST_PI      = 3.14159265358979323846_R8  ! pi
   real(R8),parameter :: SHR_CONST_CDAY    = 86400.0_R8      ! sec in calendar day ~ sec
   real(R8),parameter :: SHR_CONST_SDAY    = 86164.0_R8      ! sec in siderial day ~ sec
   real(R8),parameter :: SHR_CONST_OMEGA   = 2.0_R8*SHR_CONST_PI/SHR_CONST_SDAY
   real(R8),parameter :: SHR_CONST_G       = 9.80616_R8      ! acceleration of gravity ~ m/s^2
   real(R8),parameter :: SHR_CONST_BOLTZ   = 1.38065e-23_R8  ! Boltzmann's constant ~ J/K/molecule
   real(R8),parameter :: SHR_CONST_AVOGAD  = 6.02214e26_R8   ! Avogadro's number ~ molecules/kmole
   real(R8),parameter :: SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
   real(R8),parameter :: SHR_CONST_MWDAIR  = 28.966_R8       ! molecular weight dry air ~ kg/kmole
   real(R8),parameter :: SHR_CONST_MWWV    = 18.016_R8       ! molecular weight water vapor
   real(R8),parameter :: SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
   real(R8),parameter :: SHR_CONST_RWV     = SHR_CONST_RGAS/SHR_CONST_MWWV
   real(R8),parameter :: SHR_CONST_ZVIR    = (SHR_CONST_RWV/SHR_CONST_RDAIR)-1.0_R8
   real(R8),parameter :: SHR_CONST_CPDAIR  = 1.00464e3_R8    ! specific heat of dry air   ~ J/kg/K
   real(R8),parameter :: SHR_CONST_CPWV    = 1.810e3_R8      ! specific heat of water vap ~ J/kg/K
   real(R8),parameter :: SHR_CONST_CPVIR   = (SHR_CONST_CPWV/SHR_CONST_CPDAIR)-1.0_R8
   real(R8),parameter :: SHR_CONST_TKFRZ   = 273.15_R8
   real(R8),parameter :: SHR_CONST_TKFRZSW = SHR_CONST_TKFRZ - 1.8_R8
   real(R8),parameter :: SHR_CONST_STEBOL  = 5.670374419e-8_R8
   real(R8),parameter :: SHR_CONST_SPVAL        = 1.0e30_R8                 ! special missing value
   real(R8),parameter :: SHR_CONST_SPVAL_AERODEP= 1.e29_r8
   real(R8),parameter :: SHR_CONST_SPVAL_TOLMIN = 0.99_R8 * SHR_CONST_SPVAL ! min spval tolerance
   real(R8),parameter :: SHR_CONST_VSMOW_18O   = 2005.2e-6_R8   ! 18O/16O in VMSOW
END MODULE shr_const_mod
"""

ELM_VARCON = """\
module elm_varcon
  use shr_kind_mod, only: r8 => shr_kind_r8
  use shr_const_mod, only: SHR_CONST_G, SHR_CONST_STEBOL, SHR_CONST_TKFRZ, SHR_CONST_CDAY, &
                           SHR_CONST_PI, SHR_CONST_RDAIR, SHR_CONST_RWV
  implicit none
  save
  real(r8), parameter :: n_melt=0.7                         !fsca shape parameter
  real(r8), parameter :: e_ice=6.0                          !soil ice impedance factor
  real(r8), parameter :: mu = 0.13889                       !connectivity exponent
  real(r8), parameter :: tcrit  = 2.5_r8
  real(r8), parameter :: mm_epsilon = 0.622_r8              ! ratio of molecular weights
  real(r8), public, parameter :: degpsec = 15._r8/3600.0_r8 ! Degree's earth rotates per second
  real(r8), public, parameter ::  secspday= SHR_CONST_CDAY  ! Seconds per day
  integer,  public, parameter :: isecspday= secspday        ! Integer seconds per day
  real(r8), public, parameter ::  spval = 1.e36_r8          ! special value for real data
  integer , public, parameter :: ispval = -9999             ! special value for int data
  real(r8), parameter :: pa_to_kpa = 0.001_r8               ! Pa to kPa
  real(r8), parameter :: aquifer_water_baseline = 5000._r8  ! baseline value for aquifer water
  real(r8), parameter :: preind_atm_del13c = -6.0   ! preindustrial value for atmospheric del13C
  real(r8), parameter :: c3_del13c = -28._r8
  real(r8), parameter :: c3_r1 = 0.0112372_r8 * (1._r8 + c3_del13c/1000._r8)
  real(r8), parameter :: c3_r2 = c3_r1/(1._r8 + c3_r1)
  real(r8) :: grav   = SHR_CONST_G      !gravity constant [m/s2]
  real(r8) :: sb     = SHR_CONST_STEBOL !stefan-boltzmann constant  [W/m2/K4]
  real(r8) :: tfrz   = SHR_CONST_TKFRZ  !freezing temperature [K]
  real(r8) :: rpi    = SHR_CONST_PI
  real(r8) :: rair   = SHR_CONST_RDAIR  !gas constant for dry air [J/kg/K]
  real(r8) :: rwat   = SHR_CONST_RWV    !gas constant for water vapor [J/(kg K)]
  character(len=16), parameter :: grlnd  = 'lndgrid'      ! name of lndgrid
  character(len=16), parameter :: namep  = 'pft'          ! name of patches
  character(len=256), public, parameter :: elmfates_carbon_only = 'carbon_only'
end module elm_varcon
"""

ELM_VARPAR = """\
module elm_varpar
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  save
  integer, parameter :: nlevcan     =   1     ! number of leaf layers in canopy layer
  integer, parameter :: nvegwcs     =   4     ! number of vegetation water conductance segments
  integer, parameter :: numrad      =   2     ! number of solar radiation bands: vis, nir
  integer, parameter :: nlev_equalspace   = 15
  integer, parameter :: toplev_equalspace =  6
  integer, parameter :: sz_nbr      = 200     ! Number of size bins
  real(r8), parameter :: scalez  = 0.025_r8   ! Soil layer thickness discretization (m)
  real(r8), parameter :: zecoeff = 0.50_r8    ! soil layer depth [m]
end module elm_varpar
"""

CLUBB_PRECISION = """\
module clubb_precision
  implicit none
  private
  public :: time_precision, dp, core_rknd
  integer, parameter :: &
    time_precision = selected_real_kind( p=12 ), &
    dp = selected_real_kind( p=12 ), &
    core_rknd = 8 ! Value from the preprocessor directive
end module clubb_precision
"""

CONSTANTS_CLUBB = """\
module constants_clubb
  use clubb_precision, only: core_rknd, dp
  implicit none
  private
  real( kind = dp ), parameter, public :: &
    sqrt_2pi_dp = 2.5066282746310005024_dp, &
    sqrt_2_dp   = 1.4142135623730950488_dp
  real( kind = core_rknd ), parameter, public :: &
    sqrt_2pi = 2.5066282746310005024_core_rknd, &
    sqrt_2   = 1.4142135623730950488_core_rknd
  real( kind = core_rknd ), parameter, public :: &
    three_halves    = 3.0_core_rknd/2.0_core_rknd, &
    four_thirds     = 4.0_core_rknd/3.0_core_rknd, &
    two_thirds      = 2.0_core_rknd/3.0_core_rknd, &
    one_third       = 1.0_core_rknd/3.0_core_rknd, &
    one             = 1.0_core_rknd, &
    zero            = 0.0_core_rknd
  real( kind = core_rknd ), parameter, public :: &
    Lv = 2.5e6_core_rknd,    & ! Latent heat of vaporization   [J/kg]
    Ls = 2.834e6_core_rknd,  & ! Latent heat of sublimation    [J/kg]
    stefan_boltzmann = 5.6704e-8_core_rknd
  real( kind = core_rknd ), parameter, public :: &
    Rd = 287.04_core_rknd,   & ! Dry air gas constant          [J/kg/K]
    Rv = 461.5_core_rknd       ! Water vapor gas constant      [J/kg/K]
  real( kind = core_rknd ), parameter, public :: &
    ep  = Rd / Rv,    &
    ep1 = (1.0_core_rknd-ep)/ep,&
    ep2 = 1.0_core_rknd/ep
  real( kind = core_rknd ), parameter, public :: &
    eps = 1.0e-10_core_rknd, &
    w_tol_sqd = 4.0e-04_core_rknd, &
    Cp = 1004.67_core_rknd, &
    kappa = Rd / Cp
  real( kind = core_rknd ), parameter, public :: &
    pi = 3.141592654_core_rknd, &
    pi_dp = 3.14159265358979323846_dp
  real( kind = core_rknd ), parameter, public :: &
    grav = 9.81_core_rknd, &
    T_freeze_K = 273.15_core_rknd, &
    fstderr = 0.0_core_rknd
end module constants_clubb
"""

TREE = {
    "shr_kind_mod.f90": SHR_KIND_MOD,
    "shr_const_mod.f90": SHR_CONST_MOD,
    "elm_varcon.f90": ELM_VARCON,
    "elm_varpar.f90": ELM_VARPAR,
    "clubb_precision.f90": CLUBB_PRECISION,
    "constants_clubb.f90": CONSTANTS_CLUBB,
}

REALS = [
    # shr_const_mod: R8 renamed from SHR_KIND_R8 through an integer parameter,
    # literals with the kind spelled upper and lower, chains of quotients
    "shr_const_pi",
    "shr_const_omega",
    "shr_const_rgas",
    "shr_const_rdair",
    "shr_const_zvir",
    "shr_const_cpvir",
    "shr_const_tkfrzsw",
    "shr_const_stebol",
    "shr_const_spval_aerodep",
    "shr_const_spval_tolmin",
    "shr_const_vsmow_18o",
    # elm_varcon: default-real literals stored in a double (rounded to single
    # first), a real from an imported constant, quotient chains
    "n_melt",
    "e_ice",
    "mu",
    "tcrit",
    "mm_epsilon",
    "degpsec",
    "secspday",
    "spval",
    "pa_to_kpa",
    "aquifer_water_baseline",
    "preind_atm_del13c",
    "c3_r1",
    "c3_r2",
    "sb",
    "rair",
    "rwat",
    # elm_varpar
    "scalez",
    "zecoeff",
    # constants_clubb: kind = dp / core_rknd, long literals, quotients of names
    "sqrt_2pi_dp",
    "sqrt_2pi",
    "three_halves",
    "one_third",
    "lv",
    "stefan_boltzmann",
    "ep",
    "ep1",
    "ep2",
    "kappa",
    "pi",
    "pi_dp",
    "t_freeze_k",
]
INTS = ["isecspday", "ispval", "nlevcan", "nvegwcs", "sz_nbr", "nlev_equalspace"]
STRINGS = {"grlnd": "lndgrid", "namep": "pft", "elmfates_carbon_only": "carbon_only"}


def _plant(root: Path) -> list[Path]:
    for name, text in TREE.items():
        (root / name).write_text(text)
    return [root / name for name in TREE]


def _gfortran_bits(root: Path) -> dict[str, int]:
    """What gfortran stores: the bit pattern of every real and the value of
    every integer, printed by a program that uses the modules as the
    extensions' own routines do."""
    # ``grav`` and ``pi`` are declared on both sides of the family, as the
    # trees do, and no routine ever uses both trees; the probe renames
    # CLUBB's, and ``grav`` (a variable on ELM's side) is not asked for.
    uses = (
        "  use shr_const_mod\n  use elm_varcon\n  use elm_varpar\n"
        "  use constants_clubb, clubb_grav => grav, clubb_pi => pi\n"
    )
    renamed = {"pi": "clubb_pi"}
    prints = "".join(f"  print '(a,1x,z16.16)', '{n}', {renamed.get(n, n)}\n" for n in REALS)
    prints += "".join(f"  print '(a,1x,i0)', '{n}', {n}\n" for n in INTS)
    (root / "probe.f90").write_text(f"program probe\n{uses}  implicit none\n{prints}end program\n")
    order = [
        "shr_kind_mod",
        "shr_const_mod",
        "elm_varcon",
        "elm_varpar",
        "clubb_precision",
        "constants_clubb",
        "probe",
    ]
    subprocess.run(
        ["gfortran", "-O0", "-o", "probe", *(f"{m}.f90" for m in order)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    out = subprocess.run(["./probe"], cwd=root, check=True, capture_output=True, text=True)
    bits: dict[str, int] = {}
    for line in out.stdout.splitlines():
        name, value = line.split()
        bits[name.lower()] = int(value, 16) if name.lower() in REALS else int(value)
    return bits


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def test_every_shape_the_extensions_carry_folds_to_what_gfortran_stores(tmp_path: Path) -> None:
    sources = _plant(tmp_path)
    resolved = resolve([*REALS, *INTS, *STRINGS], sources)
    text = use_constants_module(resolved, "cesm")
    namespace: dict[str, object] = {}
    exec(text, namespace)
    expected = _gfortran_bits(tmp_path)
    wrong = {
        name: (namespace[name.upper()], expected[name])
        for name in REALS
        if _bits(namespace[name.upper()]) != expected[name]  # type: ignore[arg-type]
    }
    assert not wrong, wrong
    for name in INTS:
        assert namespace[name.upper()] == expected[name], name
        assert isinstance(namespace[name.upper()], int), name
    for name, value in STRINGS.items():
        assert namespace[name.upper()] == value, name


def test_the_declared_kinds_are_placed(tmp_path: Path) -> None:
    """Every record names the width it folds at; ``None`` is the value that
    turned into a refusal downstream, by name."""
    sources = _plant(tmp_path)
    resolved = {r["name"]: r for r in resolve([*REALS, *INTS, *STRINGS], sources)}
    for name in REALS:
        assert resolved[name]["kind_dtype"] == "float64", name
    for name in INTS:
        assert resolved[name]["kind_dtype"] == "int", name
    for name in STRINGS:
        assert resolved[name]["kind_dtype"] == "str", name


@pytest.mark.xfail(strict=True, raises=UnsupportedExpression, reason="logical literal initializer")
def test_a_logical_parameter_is_not_resolved_yet(tmp_path: Path) -> None:
    """``logical, parameter :: l_diag = .false.`` is common in CLUBB's flag
    modules; the resolver refuses the literal today. When it does not, this
    turns green and the mark comes off."""
    (tmp_path / "flags.f90").write_text(
        "module flags\n  implicit none\n  logical, parameter :: l_diag = .false.\nend module\n"
    )
    resolve(["l_diag"], [tmp_path / "flags.f90"])


@pytest.mark.xfail(strict=True, raises=UnsupportedExpression, reason="concatenation")
def test_a_concatenated_character_parameter_is_not_resolved_yet(tmp_path: Path) -> None:
    (tmp_path / "names.f90").write_text(
        "module names\n  implicit none\n"
        "  character(len=*), parameter :: tag = 'carbon' // '_only'\nend module\n"
    )
    resolve(["tag"], [tmp_path / "names.f90"])
