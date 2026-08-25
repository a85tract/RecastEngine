"""Tests for the Numba backend.

``tools/numba_diff.py`` holds every emitted byte of this backend to the
pipeline's ``numbaize.py`` across 27 modules and 176 kernels, so a diff is what
proves the *rules*. What a diff over CAM cannot do is say why a rule is there,
or cover a shape the corpus happens not to contain -- and it needs a translator
checkout on disk, which CI does not have. These run on synthetic Fortran and
cover the part of the design a byte comparison cannot explain: which
subprograms are eligible and why not, what the state closure contains, and the
order it is expanded in, which both sides fill by position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import constants, interface
from recast.fortran._parse import f03, parse, walk
from recast.transform.numba.backend import (
    Kernels,
    derived_components,
    eligible,
    ineligible_reason,
)
from recast.transform.numba.emitter import Emission, NumbaSubprograms
from recast.transform.profiles import PROFILES

KINDS = {"wp_r8": "float64"}

SOURCE = """\
module nb_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8) :: omeps, tmelt

  type props_t
    real(r8) :: rho
    real(r8) :: eff_dim
  end type props_t

  type(props_t) :: liq_props

contains

  elemental function scaled(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = x * omeps
  end function scaled

  function twice_scaled(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = scaled(x) * tmelt
  end function twice_scaled

  subroutine sets_state(v)
    real(r8), intent(in) :: v
    omeps = v
  end subroutine sets_state

  subroutine named(tag, out)
    character(len=*), intent(in) :: tag
    real(r8), intent(out) :: out
    out = 1.0_r8
  end subroutine named

  elemental function plain(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = x * 2.0_r8
  end function plain

  function from_props(p, x) result(y)
    type(props_t), intent(in) :: p
    real(r8), intent(in) :: x
    real(r8) :: y
    y = x * p%rho
  end function from_props

  function uses_props(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = x * liq_props%rho
  end function uses_props

  subroutine complains(x, errmsg)
    real(r8), intent(in) :: x
    character(len=128), intent(out) :: errmsg
    if (x < 0.0_r8) then
      errmsg = 'negative'
    end if
  end subroutine complains
end module nb_mod
"""


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("nb") / "nb_mod.f90"
    path.write_text(SOURCE)
    return path


@pytest.fixture(scope="module")
def record(source: Path) -> dict[str, Any]:
    return interface.extract(source, kind_assumptions=KINDS)


def subprogram(record: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in record["subprograms"] if s["name"] == name)


def node_of(source: Path, name: str) -> Any:
    return next(
        sub
        for sub in walk(parse(source), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )


def build(source: Path, record: dict[str, Any]) -> tuple[NumbaSubprograms, Emission]:
    kernels = Kernels(record=record)
    emission = Emission(kernels=kernels)
    return (
        NumbaSubprograms(
            record=record,
            constants=constants.extract(source),
            profile=PROFILES["ifx"],
            emission=emission,
        ),
        emission,
    )


# --- eligibility -------------------------------------------------------------


def test_a_subprogram_that_writes_module_state_is_not_a_kernel(record: dict[str, Any]) -> None:
    """It would have to write through a parameter, which is the one thing the
    closure convention cannot express."""
    assert not eligible(subprogram(record, "sets_state"))
    assert ineligible_reason(subprogram(record, "sets_state")) == "[elig] module-state write"


def test_a_character_in_argument_is_not_a_kernel(record: dict[str, Any]) -> None:
    assert not eligible(subprogram(record, "named"))
    assert "character" in ineligible_reason(subprogram(record, "named"))


def test_a_character_out_argument_is_still_a_kernel(record: dict[str, Any]) -> None:
    """Writes to one become an integer code the host wrapper decodes, so it is
    the one CHARACTER shape that does not disqualify a subprogram."""
    assert eligible(subprogram(record, "complains"))


def test_an_external_shim_call_is_not_a_kernel(record: dict[str, Any]) -> None:
    """There is no compiled body behind the shim's name."""
    externals = {"scaled": {"kind": "function"}}
    assert not eligible(subprogram(record, "twice_scaled"), externals)
    assert "external shim" in ineligible_reason(subprogram(record, "twice_scaled"), externals)


# --- the closure -------------------------------------------------------------


def test_the_closure_is_transitive_through_callees(record: dict[str, Any]) -> None:
    """``twice_scaled`` reads only ``tmelt`` itself, but calls ``scaled``,
    which reads ``omeps``. A kernel compiled without ``omeps`` as a parameter
    would freeze whatever it held at first call."""
    kernels = Kernels(record=record)
    assert kernels.state_closure("scaled") == {"omeps"}
    assert kernels.state_closure("twice_scaled") == {"omeps", "tmelt"}


def test_a_closure_that_would_recurse_terminates(record: dict[str, Any]) -> None:
    kernels = Kernels(record=record)
    kernels._closures.clear()
    assert isinstance(kernels.state_closure("twice_scaled"), set)


def test_derived_state_expands_one_parameter_per_component(record: dict[str, Any]) -> None:
    """No namespace object crosses into nopython mode, so the object arrives
    as its components -- in declaration order, which is what the wrapper
    fills by position."""
    kernels = Kernels(record=record)
    assert kernels.own_derived_state() == {"liq_props": ["rho", "eff_dim"]}
    assert kernels.expand({"liq_props", "tmelt"}) == [
        "own__liq_props__rho",
        "own__liq_props__eff_dim",
        "tmelt",
    ]


def test_the_expansion_is_sorted_so_both_sides_agree(record: dict[str, Any]) -> None:
    """These are positional parameters. A set's iteration order would put the
    kernel's signature and the wrapper's call out of step between runs."""
    kernels = Kernels(record=record)
    assert kernels.expand({"tmelt", "omeps"}) == ["omeps", "tmelt"]


# --- the kernel --------------------------------------------------------------


def test_the_state_closure_is_appended_to_the_signature(
    source: Path, record: dict[str, Any]
) -> None:
    assembler, _ = build(source, record)
    lines, _ = assembler.render(node_of(source, "twice_scaled"), "twice_scaled")
    assert lines[1] == "def _twice_scaled_k(x, omeps, tmelt):"


def test_an_elemental_scalar_function_becomes_a_ufunc(
    source: Path, record: dict[str, Any]
) -> None:
    """ELEMENTAL over scalars *is* a ufunc: numba builds one that takes scalar
    or array actuals, which is what ELEMENTAL means."""
    assembler, _ = build(source, record)
    lines, _ = assembler.render(node_of(source, "plain"), "plain")
    assert lines[0] == '@vectorize(["f8(f8)"], nopython=True)'


def test_an_elemental_function_with_state_stays_a_plain_kernel(
    source: Path, record: dict[str, Any]
) -> None:
    """``@vectorize`` has nowhere to put the closure parameters, so an
    ELEMENTAL function that reads module state cannot become a ufunc."""
    assembler, _ = build(source, record)
    lines, _ = assembler.render(node_of(source, "scaled"), "scaled")
    assert lines[0].startswith("@njit(")
    assert "fastmath=False" in lines[0]


def test_a_function_result_is_a_scalar_zero_not_an_allocation(
    source: Path, record: dict[str, Any]
) -> None:
    """The NumPy backend allocates an array-valued result at its declared
    shape; this one writes ``0.0`` whatever the rank. That is the pipeline's
    rule, relayed -- see ``NumbaSubprograms._result_initializer``."""
    assembler, _ = build(source, record)
    lines, _ = assembler.render(node_of(source, "twice_scaled"), "twice_scaled")
    assert "    y = 0.0" in lines
    assert not any("np.zeros" in line for line in lines)


def test_a_character_out_write_becomes_an_indexed_error_code(
    source: Path, record: dict[str, Any]
) -> None:
    assembler, emission = build(source, record)
    lines, _ = assembler.render(node_of(source, "complains"), "complains")
    assert any("_errflag_errmsg = 1" in line for line in lines)
    assert emission.messages == ["negative"]
    assert lines[-3] == "    return _errflag_errmsg"


# --- the wrapper -------------------------------------------------------------


def test_the_wrapper_keeps_the_fortran_signature_and_fills_the_closure(
    source: Path, record: dict[str, Any]
) -> None:
    """The public name takes what the Fortran declared; the state is read off
    the validated NumPy module and passed positionally."""
    assembler, _ = build(source, record)
    assembler.render(node_of(source, "twice_scaled"), "twice_scaled")
    wrapper = assembler.wrapper(subprogram(record, "twice_scaled"))
    assert wrapper[0] == "def twice_scaled(x):"
    assert wrapper[2] == "    return _twice_scaled_k(x, _host.omeps, _host.tmelt)"


def test_the_wrapper_unpacks_a_derived_argument(
    source: Path, record: dict[str, Any]
) -> None:
    """The kernel takes one parameter per component; the wrapper keeps taking
    the object and unpacks it on the way in."""
    assert derived_components(record, subprogram(record, "from_props")) == {
        "p": ["rho", "eff_dim"]
    }
    assembler, _ = build(source, record)
    lines, _ = assembler.render(node_of(source, "from_props"), "from_props")
    assert lines[1] == "def _from_props_k(p__rho, p__eff_dim, x):"
    wrapper = assembler.wrapper(subprogram(record, "from_props"))
    assert wrapper[0] == "def from_props(p, x):"
    assert wrapper[2] == "    return _from_props_k(p.rho, p.eff_dim, x)"


# --- the runtime -------------------------------------------------------------


def test_the_runtime_anchors_are_a_subset_of_the_numpy_backends() -> None:
    """Two backends held to one set of anchors rather than drifting into
    separate notions of what ``sign`` and ``mod`` mean."""
    import ast

    def shims(path: Path) -> set[str]:
        tree = ast.parse(path.read_text())
        return {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith(("_f_", "_fstr_"))
        }

    root = Path(__file__).resolve().parent.parent / "src" / "recast" / "transform"
    numba_shims = shims(root / "numba" / "runtime.py")
    numpy_shims = shims(root / "numpy" / "runtime.py")
    assert numba_shims, "the numba runtime defines no shims"
    assert numba_shims <= numpy_shims, sorted(numba_shims - numpy_shims)


def test_emitting_numba_does_not_require_numba() -> None:
    """The emitter reads the runtime's text off disk rather than importing it,
    for the same reason the JAX backend does."""
    from recast.transform.numba import translate as numba_translate

    text = numba_translate._runtime_text()
    assert text.startswith("import math")
    assert "@njit(cache=True, fastmath=False" in text


# --- the Transform -----------------------------------------------------------


def test_apply_emits_a_module_with_kernels_wrappers_and_delegation(
    source: Path, record: dict[str, Any]
) -> None:
    """The whole product, end to end.

    ``tools/numba_diff.py`` compares kernels and wrappers but builds its own
    subprogram table, so it never runs ``apply``. That is where the node lookup
    lives, and a wrong one delegates every subprogram while reporting success.
    """
    from recast.model import Facts, Unit
    from recast.transform.numba.translate import NumbaTranslation

    facts = Facts(
        unit="nb_mod",
        interface=record,
        constants=constants.extract(source),
        provenance={"source": source.name},
    )
    unit = Unit(uid="nb_mod", kind="module", sources=(Path(source.name),))
    candidate = NumbaTranslation().apply(unit, facts, {"root": str(source.parent)})

    text = candidate.files[Path("nb_mod_njit.py")].decode()
    # a kernel, its wrapper, and the anchor it re-exports the rest from
    assert "def _twice_scaled_k(x, omeps, tmelt):" in text
    assert "    return _twice_scaled_k(x, _host.omeps, _host.tmelt)" in text
    assert "import nb_mod_numpy as _host" in text
    # everything ineligible is delegated rather than guessed at
    assert "sets_state = _host.sets_state" in text
    assert "named = _host.named" in text
    assert candidate.notes["anchor"] == "nb_mod_numpy.py"
    assert "sets_state" in candidate.notes["host_delegated"]
    assert "twice_scaled" in candidate.notes["kernels"]
    # the runtime travels with the file, so it stands alone
    assert "def _f_mod(a: Any, p: Any) -> Any:" in text


def test_apply_is_reproducible(source: Path, record: dict[str, Any]) -> None:
    """A deterministic Transform that emits a different artifact on the second
    run breaks the reproducibility ``Candidate.digest()`` is checked on."""
    from recast.model import Facts, Unit
    from recast.transform.numba.translate import NumbaTranslation

    facts = Facts(
        unit="nb_mod",
        interface=record,
        constants=constants.extract(source),
        provenance={"source": source.name},
    )
    unit = Unit(uid="nb_mod", kind="module", sources=(Path(source.name),))
    config = {"root": str(source.parent)}
    first = NumbaTranslation().apply(unit, facts, config)
    second = NumbaTranslation().apply(unit, facts, config)
    assert first.files == second.files
