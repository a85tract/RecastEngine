"""Tests for the ``f2py-golden`` oracle and the ``differential.bitexact`` gate.

Two layers. The mechanism tests need no compiler: wrapper text, cache keys,
and every fail-closed path of the verifier. The end-to-end test compiles a
real toy module with gfortran and walks the whole translate spine --
frontend, transform, rwset gate, oracle, bit-exact gate -- and then breaks
the candidate on purpose, because a gate that has never been seen to fail
proves nothing by passing.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

import recast.oracle.f2py as f2py_module
from recast.errors import ConfigError, OracleUnavailable
from recast.executors.local import LocalExecutor
from recast.fortran.frontend import FortranFrontend
from recast.model import Candidate, Confidence, OracleRef, Unit
from recast.oracle.f2py import F2pyGoldenOracle, wrappers_for
from recast.plugins.executor import JobResult
from recast.transform.numpy.translate import NumpyTranslation
from recast.verify.bitexact import BitexactVerifier
from recast.verify.rwset import ReadWriteSetVerifier

GFORTRAN = shutil.which("gfortran")
MESON = importlib.util.find_spec("mesonbuild") is not None
"""f2py's build backend, carried by the verify extra. CI's test matrix has a
compiler (the runner image ships one) but not the backend, and the spine job
has both -- so the guard must check both, or the matrix runs half a build."""

RECORD = {
    "module": "demo_mod",
    "generics": {"scale": ["scale_r"]},
    "subprograms": [
        {
            "name": "settle",
            "kind": "subroutine",
            "args": [
                {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                {
                    "name": "rho",
                    "dtype": "float64",
                    "intent": "IN",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {
                    "name": "p",
                    "dtype": "float64",
                    "intent": "OUT",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {"name": "extra", "dtype": "float64", "intent": "OUT", "optional": True},
            ],
        },
        {
            "name": "scale_r",
            "kind": "function",
            "args": [{"name": "x", "dtype": "float64", "intent": "IN", "optional": False}],
            "result": "y",
            "result_dtype": "float64",
        },
    ],
}


# --- wrapper text ------------------------------------------------------------


def test_wrappers_drop_optionals_and_route_generics() -> None:
    text, names = wrappers_for(RECORD, ["settle", "scale_r"])
    assert names == ["w_settle", "w_scale_r"]
    assert "extra" not in text  # optional: the wrapper compares the required surface
    # A specific of a generic is private; the call goes through the generic name.
    assert "use demo_mod, only: scale" in text
    assert "res = scale(x)" in text
    assert "real(8), intent(out) :: p(n)" in text  # dims spelled so f2py can size them


def test_out_arguments_are_defined_before_the_call() -> None:
    """An intent(out) dummy is undefined on entry, and a subprogram that
    returns early -- a guard rejecting its own arguments -- never assigns it.
    What f2py hands back is then whatever the buffer it allocated held, which
    is not a fact about the Fortran and not something a translation can be
    held to. The wrapper defines it instead, so the reference's output buffers
    start where the emitted translation's do."""
    text, _ = wrappers_for(RECORD, ["settle"])
    body = text[text.index("subroutine w_settle") : text.index("end subroutine w_settle")]
    assert "  p = 0" in body
    assert body.index("  p = 0") < body.index("  call settle(")
    # An input is not touched: it is the harness's value, not the wrapper's.
    assert "  rho = 0" not in body


def test_a_caller_buffer_out_array_is_not_defined_by_the_wrapper() -> None:
    """An intent(out) array the callee cannot size -- ``dy(*)`` -- is the
    caller's storage on both sides: the gate generates it and hands the same
    values to the reference and the candidate. Zeroing it in the wrapper would
    fail every cell the callee never writes, and ``dy = 0`` is not even legal
    for an assumed-size dummy (SLSQP's ``dcopy`` stopped the oracle build)."""
    record = {
        "module": "blas_mod",
        "subprograms": [
            {
                "name": "dcopy",
                "kind": "subroutine",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {
                        "name": "dx",
                        "dtype": "float64",
                        "intent": "IN",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None, "assumed_size": True}],
                    },
                    {
                        "name": "dy",
                        "dtype": "float64",
                        "intent": "OUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None, "assumed_size": True}],
                        "buffer": True,
                    },
                ],
            }
        ],
    }
    text, _ = wrappers_for(record, ["dcopy"])
    body = text[text.index("subroutine w_dcopy") : text.index("end subroutine w_dcopy")]
    assert "real(8), intent(out) :: dy(*)" in body
    assert "  dy = 0" not in body


def test_a_dtype_the_wrapper_cannot_spell_refuses() -> None:
    broken = {
        "module": "m",
        "generics": {},
        "subprograms": [
            {
                "name": "s",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "grid",
                        "dtype": "UNKNOWN(TYPE(GRID_T))",
                        "intent": "IN",
                        "optional": False,
                    }
                ],
            }
        ],
    }
    with pytest.raises(ConfigError, match="cannot spell"):
        wrappers_for(broken, ["s"])


CALLBACK_RECORD = {
    "module": "solve_mod",
    "generics": {},
    "interfaces": {
        "residual": {
            "kind": "subroutine",
            "args": [
                {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                {
                    "name": "x",
                    "dtype": "float64",
                    "intent": "IN",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {
                    "name": "fvec",
                    "dtype": "float64",
                    "intent": "OUT",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {"name": "iflag", "dtype": "int32", "intent": "INOUT", "optional": False},
            ],
            "result": None,
            "result_dtype": None,
        }
    },
    "subprograms": [
        {
            "name": "drive",
            "kind": "subroutine",
            "args": [
                {
                    "name": "fcn",
                    "dtype": "PROCEDURE",
                    "intent": "IN",
                    "optional": False,
                    "procedure": True,
                    "interface": "residual",
                },
                {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                {
                    "name": "x",
                    "dtype": "float64",
                    "intent": "INOUT",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
            ],
        }
    ],
}


def test_a_procedure_argument_becomes_an_f2py_call_back() -> None:
    """f2py works a call-back's signature out from a call to it in the body
    being wrapped, and the wrapper has none -- it hands the procedure straight
    on. So the call and its argument declarations are written for f2py alone,
    as ``!f2py`` comments no compiler ever sees."""
    text, names = wrappers_for(CALLBACK_RECORD, ["drive"])
    assert names == ["w_drive"]
    assert "  external fcn" in text
    assert "!f2py  integer, intent(in), required :: cb_fcn_n" in text
    assert "!f2py  real(8), dimension(cb_fcn_n), intent(in) :: cb_fcn_x" in text
    assert "!f2py  real(8), dimension(cb_fcn_n), intent(out) :: cb_fcn_fvec" in text
    assert "!f2py  integer, intent(in,out) :: cb_fcn_iflag" in text
    assert "!f2py  call fcn(cb_fcn_n, cb_fcn_x, cb_fcn_fvec, cb_fcn_iflag)" in text
    # ``n`` sizes another argument, so f2py would make it optional and move it
    # to the end; the translation calls the same object in declaration order.
    assert text.count("required") == 1


def test_a_procedure_argument_with_no_interface_refuses() -> None:
    """``procedure() :: fcn`` says a name is callable and nothing about the
    call. There is nothing to declare, and guessing is not an option."""
    record = {
        **CALLBACK_RECORD,
        "interfaces": {},
    }
    with pytest.raises(ConfigError, match="carries no interface"):
        wrappers_for(record, ["drive"])


def test_a_long_argument_list_is_folded_for_free_form() -> None:
    """gfortran makes a line past column 132 an error, and a subprogram with
    two dozen arguments writes one."""
    wide = {
        "module": "m",
        "generics": {},
        "subprograms": [
            {
                "name": "wide",
                "kind": "subroutine",
                "args": [
                    {
                        "name": f"argument_number_{index:02d}",
                        "dtype": "float64",
                        "intent": "IN",
                        "optional": False,
                    }
                    for index in range(24)
                ],
            }
        ],
    }
    text, _ = wrappers_for(wide, ["wide"])
    assert all(len(line) <= 132 for line in text.splitlines())
    assert "&" in text


def test_the_harness_builds_one_call_back_for_both_sides() -> None:
    """The same Python object reaches the reference and the candidate, so a
    difference between them is a difference in the code under test."""
    import numpy as np

    from recast.verify.bitexact import callback_for

    interface = CALLBACK_RECORD["interfaces"]["residual"]
    callback = callback_for(np, "fcn", interface)
    # Arity is part of the calling convention: f2py reads it off the object.
    assert callback.__code__.co_argcount == 3  # n, x, iflag -- fvec is returned
    fvec, iflag = callback(np.int32(3), np.array([0.5, -0.25, 2.0]), np.int32(1))
    assert fvec.shape == (3,)
    assert iflag == 1  # a control flag is handed back, not invented
    again, _ = callback(np.int32(3), np.array([0.5, -0.25, 2.0]), np.int32(1))
    assert np.array_equal(fvec, again)  # deterministic


def test_a_call_back_this_harness_cannot_supply_says_so() -> None:
    import numpy as np

    from recast.verify.bitexact import callback_for

    opaque = {
        "kind": "subroutine",
        "args": [{"name": "grid", "dtype": "UNKNOWN(TYPE(GRID_T))", "intent": "IN"}],
        "result": None,
        "result_dtype": None,
    }
    with pytest.raises(ValueError, match="cannot supply"):
        callback_for(np, "fcn", opaque)


# --- the verifier fails closed -----------------------------------------------


def _bare_candidate() -> Candidate:
    return Candidate(unit="fortran:demo_mod", transform="translate.numpy")


def _no_oracle() -> OracleRef:
    return OracleRef(unit="fortran:demo_mod", oracle="f2py-golden", key="x", handle=None)


def test_an_oracle_without_a_module_fails_closed(tmp_path: Path) -> None:
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:demo_mod", kind="module"),
        _bare_candidate(),
        _no_oracle(),
        tmp_path,
        LocalExecutor(),
        {},
    )
    assert verdict.confidence is Confidence.FAILED
    assert "no compiled module" in verdict.detail


def test_a_candidate_without_files_fails_closed(tmp_path: Path) -> None:
    ref = OracleRef(
        unit="fortran:demo_mod", oracle="f2py-golden", key="x", handle={"module": object()}
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:demo_mod", kind="module"),
        _bare_candidate(),
        ref,
        tmp_path,
        LocalExecutor(),
        {},
    )
    assert verdict.confidence is Confidence.FAILED
    assert "does not import" in verdict.detail


def test_f2py_scalar_inout_uses_a_writable_rank_zero_buffer(tmp_path: Path) -> None:
    """f2py silently loses a scalar update when handed a NumPy scalar.

    Its generated signature calls the dummy an ``in/output rank-0 array``.
    The verifier must honor that ABI while leaving the candidate's sampled
    scalar alone.  This one call also pins the pre-existing array-INOUT and
    pure-OUT paths: all three outputs have to be paired by declaration name.
    """
    emitted = b"""\
import numpy as np

_SIGNATURES = {
    "step": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "float64", "intent": "INOUT", "optional": False},
            {"name": "a", "dtype": "float64", "intent": "INOUT", "optional": False,
             "dims": [{"lb": "1", "ub": "3"}]},
            {"name": "y", "dtype": "float64", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def step(x, a):
    return x + 1.0, a + 2.0, x * 3.0
"""
    candidate = Candidate(
        unit="fortran:scalar_inout",
        transform="translate.numpy",
        files={Path("scalar_inout_numpy.py"): emitted},
    )
    seen: list[tuple[tuple[int, ...], bool, tuple[int, ...]]] = []

    class Truth:
        @staticmethod
        def w_step(x, a):
            seen.append((x.shape, bool(x.flags.writeable), a.shape))
            original = float(x)
            x[...] = original + 1.0
            a[...] = a + 2.0
            return original * 3.0

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"step": "w_step"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 3, "ranges": {"x": (1.0, 2.0), "a": (2.0, 3.0)}},
    )

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 15
    assert seen == [((), True, (3,))] * 3


@pytest.mark.parametrize("convention", ["emitted", "recorded"])
def test_non_f2py_conventions_keep_scalar_inout_as_a_scalar(convention: str) -> None:
    """A rank-0 buffer is an f2py ABI detail, not a universal convention."""
    import numpy as np

    value = np.float64(2.0)
    argument = {
        "name": "x",
        "dtype": "float64",
        "intent": "INOUT",
        "optional": False,
    }
    assert BitexactVerifier._truth_input(np, argument, value, convention) is value


def test_f2py_logical_inout_fails_closed_before_execution(tmp_path: Path) -> None:
    """No Python buffer spelling is a portable Fortran LOGICAL INOUT ABI."""
    emitted = b"""\
import numpy as np

_SIGNATURES = {
    "flip": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "bool", "intent": "INOUT", "optional": False},
            {"name": "a", "dtype": "bool", "intent": "INOUT", "optional": False,
             "dims": [{"lb": "1", "ub": "3"}]},
            {"name": "y", "dtype": "bool", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def flip(x, a):
    raise AssertionError("candidate subroutine must not execute")
"""
    candidate = Candidate(
        unit="fortran:logical_inout",
        transform="translate.numpy",
        files={Path("logical_inout_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_flip(x, a):
            raise AssertionError("oracle subroutine must not execute")

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"flip": "w_flip"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 3},
    )

    assert verdict.confidence is Confidence.FAILED
    assert "no portable Python buffer ABI" in verdict.detail
    assert "x, a" in verdict.detail


def test_f2py_logical_pure_out_is_normalized(tmp_path: Path) -> None:
    """A pure OUT's nonzero LOGICAL representation compares as Python True."""
    import numpy as np

    emitted = b"""\
import numpy as np

_SIGNATURES = {
    "invert": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "bool", "intent": "IN", "optional": False},
            {"name": "y", "dtype": "bool", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def invert(x):
    return np.logical_not(x)
"""
    candidate = Candidate(
        unit="fortran:logical_out",
        transform="translate.numpy",
        files={Path("logical_out_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_invert(x):
            return np.int32(0 if bool(x) else -7)

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"invert": "w_invert"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 3},
    )

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 3


def test_f2py_logical_function_result_is_normalized(tmp_path: Path) -> None:
    """A function's nonzero LOGICAL result compares equal to Python True."""
    import numpy as np

    emitted = b"""\
_SIGNATURES = {
    "identity": {
        "kind": "function",
        "args": [
            {"name": "x", "dtype": "bool", "intent": "IN", "optional": False},
        ],
        "result": "yes",
        "result_dtype": "bool",
    }
}

def identity(x):
    return x
"""
    candidate = Candidate(
        unit="fortran:logical_function",
        transform="translate.numpy",
        files={Path("logical_function_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_identity(x):
            assert isinstance(x, np.bool_)
            return np.int32(-2 if bool(x) else 0)

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"identity": "w_identity"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 3},
    )

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 3


def test_function_dummy_side_effects_fail_closed_before_execution(tmp_path: Path) -> None:
    """A result-only comparison must not silently ignore an INOUT dummy."""
    emitted = b"""\
_SIGNATURES = {
    "bump": {
        "kind": "function",
        "args": [
            {"name": "x", "dtype": "float64", "intent": "INOUT", "optional": False},
        ],
        "result": "y",
        "result_dtype": "float64",
    }
}

def bump(x):
    raise AssertionError("candidate function must not execute")
"""
    candidate = Candidate(
        unit="fortran:function_side_effect",
        transform="translate.numpy",
        files={Path("function_side_effect_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_bump(x):
            raise AssertionError("oracle function must not execute")

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"bump": "w_bump"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 1},
    )

    assert verdict.confidence is Confidence.FAILED
    assert "cannot pair both its result and side effects" in verdict.detail


def test_unknown_intent_fails_closed_before_execution(tmp_path: Path) -> None:
    """UNKNOWN is wrapped as INOUT, so omitting its side effect is unsound."""
    emitted = b"""\
_SIGNATURES = {
    "touch": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "float64", "intent": "UNKNOWN", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def touch(x):
    raise AssertionError("candidate subroutine must not execute")
"""
    candidate = Candidate(
        unit="fortran:unknown_intent",
        transform="translate.numpy",
        files={Path("unknown_intent_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_touch(x):
            raise AssertionError("oracle subroutine must not execute")

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"touch": "w_touch"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 1},
    )

    assert verdict.confidence is Confidence.FAILED
    assert "UNKNOWN intent" in verdict.detail


def _integer_output_verdict(
    tmp_path: Path,
    expression: str,
    truth_value,
    *,
    dtype: str = "int64",
    **config,
):
    emitted = f"""\
_SIGNATURES = {{
    "measure": {{
        "kind": "subroutine",
        "args": [
            {{"name": "y", "dtype": {dtype!r}, "intent": "OUT", "optional": False}},
        ],
        "result": None,
        "result_dtype": None,
    }}
}}

def measure():
    return {expression}
""".encode()
    candidate = Candidate(
        unit="fortran:integer_output",
        transform="translate.numpy",
        files={Path("integer_output_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_measure():
            return truth_value

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"measure": "w_measure"}},
    )
    return BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {"trials": 1, **config},
    )


def test_large_integer_outputs_compare_without_float64_aliasing(tmp_path: Path) -> None:
    value = 2**53 + 1
    verdict = _integer_output_verdict(tmp_path, str(value), value)

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["integer_points"] == 1
    assert verdict.metrics["integer_mismatch"] == 0


def test_a_float64_collision_is_an_integer_mismatch_even_with_rtol(tmp_path: Path) -> None:
    verdict = _integer_output_verdict(
        tmp_path,
        str(2**53 + 1),
        2**53,
        rtol=1e100,
    )

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["integer_mismatch"] == 1
    assert "cannot be tolerance-excused" in verdict.detail


@pytest.mark.parametrize(
    ("dtype", "expression", "truth_value", "detail"),
    [
        ("int64", "1.0", 1, "non-integer dtype float64"),
        ("int64", str(2**63), 0, "outside"),
        ("int32", str(2**31), 0, "outside"),
    ],
)
def test_declared_integer_outputs_reject_masquerades_and_overflow(
    tmp_path: Path,
    dtype: str,
    expression: str,
    truth_value: int,
    detail: str,
) -> None:
    verdict = _integer_output_verdict(
        tmp_path,
        expression,
        truth_value,
        dtype=dtype,
    )

    assert verdict.confidence is Confidence.FAILED
    assert detail in verdict.detail


def test_integer_output_comparison_preserves_shape(tmp_path: Path) -> None:
    verdict = _integer_output_verdict(tmp_path, "[1, 2]", [[1, 2]])

    assert verdict.confidence is Confidence.FAILED
    assert "shape (2,) vs (1, 2)" in verdict.detail


@pytest.mark.parametrize(
    ("expression", "truth_value", "dtype", "trials"),
    [
        ("1", 1, "int64", 0),
        ("[]", [], "float64", 1),
    ],
)
def test_zero_numerical_points_cannot_be_a_bit_exact_pass(
    tmp_path: Path,
    expression: str,
    truth_value,
    dtype: str,
    trials: int,
) -> None:
    verdict = _integer_output_verdict(
        tmp_path,
        expression,
        truth_value,
        dtype=dtype,
        trials=trials,
    )

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["points"] == 0
    assert "zero numerical points" in verdict.detail


@pytest.mark.parametrize(
    "sub",
    [
        {
            "kind": "subroutine",
            "args": [
                {
                    "name": "z",
                    "dtype": "complex128",
                    "intent": "IN",
                    "optional": False,
                }
            ],
        },
        {
            "kind": "function",
            "args": [
                {
                    "name": "x",
                    "dtype": "float64",
                    "intent": "IN",
                    "optional": False,
                }
            ],
            "result": "grid",
            "result_dtype": "UNKNOWN(TYPE(GRID_T))",
        },
    ],
)
def test_unsupported_declared_dtype_fails_before_execution(sub) -> None:
    import numpy as np

    calls = 0

    def must_not_run(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("neither side may execute")

    outcome = BitexactVerifier()._compare_subprogram(
        np,
        "unsupported",
        sub,
        must_not_run,
        must_not_run,
        1,
        {},
        {},
    )

    assert "unsupported declared dtype" in outcome["error"]
    assert calls == 0


def test_recorded_sample_must_carry_every_required_output(tmp_path: Path) -> None:
    emitted = b"""\
_SIGNATURES = {
    "two_outputs": {
        "kind": "subroutine",
        "args": [
            {"name": "y", "dtype": "int32", "intent": "OUT", "optional": False},
            {"name": "z", "dtype": "int32", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def two_outputs():
    raise AssertionError("partial evidence must be rejected before candidate execution")
"""
    candidate = Candidate(
        unit="fortran:partial_recording",
        transform="translate.numpy",
        files={Path("partial_recording_numpy.py"): emitted},
    )
    ref = OracleRef(
        unit=candidate.unit,
        oracle="dump-replay",
        key="k",
        handle={
            "module": None,
            "input_source": "recorded",
            "return_convention": "recorded",
            "samples": [
                {
                    "subprogram": "two_outputs",
                    "source": "partial.txt",
                    "inputs": {},
                    "outputs": {"y": 1},
                }
            ],
        },
    )
    verdict = BitexactVerifier().verify(
        Unit(uid=candidate.unit, kind="module"),
        candidate,
        ref,
        tmp_path / "work",
        LocalExecutor(),
        {},
    )

    assert verdict.confidence is Confidence.FAILED
    assert "required output(s) z" in verdict.detail
    assert "partial output evidence is not a pass" in verdict.detail


# --- the whole spine, against a real compiler --------------------------------

SOURCE = """\
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
"""

SCALAR_INOUT_SOURCE = """\
module scalar_inout
  implicit none
contains
  subroutine step(x)
    real(8), intent(inout) :: x
    x = x + 1.0d0
  end subroutine step
end module scalar_inout
"""

LOGICAL_SOURCE = """\
module logical_values
  implicit none
contains
  subroutine invert_to(x, y)
    logical, intent(in) :: x
    logical, intent(out) :: y
    y = .not. x
  end subroutine invert_to

  logical function identity(x)
    logical, intent(in) :: x
    identity = x
  end function identity
end module logical_values
"""

LOGICAL_INOUT_SOURCE = """\
module logical_inout
  implicit none
contains
  subroutine flip_scalar(x)
    logical, intent(inout) :: x
    x = .not. x
  end subroutine flip_scalar

  subroutine flip_array(x)
    logical, intent(inout) :: x(2)
    x = .not. x
  end subroutine flip_array
end module logical_inout
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_scalar_inout_is_bit_exact_against_real_f2py(tmp_path: Path) -> None:
    """A real f2py scalar INOUT update must be observable by the gate."""
    (tmp_path / "scalar_inout.f90").write_text(SCALAR_INOUT_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {
        "root": tmp_path,
        "fc": GFORTRAN,
        "trials": 5,
        "ranges": {"x": (-5.0, 5.0)},
    }
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 5
    assert verdict.metrics["bit_exact"] == 5


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_logical_values_are_bit_exact_against_real_f2py(tmp_path: Path) -> None:
    """Real f2py exercises pure OUT and function-result LOGICALs."""
    (tmp_path / "logical_values.f90").write_text(LOGICAL_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 5}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 10
    assert verdict.metrics["bit_exact"] == 10


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_real_f2py_logical_inout_fails_closed(tmp_path: Path) -> None:
    """The real ABI hazard is reported, never mistaken for a mismatch."""
    (tmp_path / "logical_inout.f90").write_text(LOGICAL_INOUT_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 2}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)

    assert verdict.confidence is Confidence.FAILED
    assert "no portable Python buffer ABI" in verdict.detail


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_translate_spine_ends_bit_exact(tmp_path: Path) -> None:
    (tmp_path / "toy_physics.f90").write_text(SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    gate = ReadWriteSetVerifier().check(unit, candidate, workspace, executor, {})
    assert gate.passed, gate.detail

    config = {
        "root": tmp_path,
        "fc": GFORTRAN,
        "trials": 5,
        "dims": {"n": 8},
        "ranges": {"rho": (0.1, 2.0), "dz": (10.0, 100.0), "w": (-5.0, 5.0)},
    }
    oracle = F2pyGoldenOracle()
    key = oracle.key(unit, facts, config)
    assert key == oracle.key(unit, facts, config)  # stable
    assert key != oracle.key(unit, facts, {**config, "fflags": "-O2"})  # flags move it

    ref = oracle.materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] > 0
    assert verdict.metrics["bit_exact"] == verdict.metrics["points"]

    # A gate that has never failed proves nothing by passing: corrupt one
    # constant in the candidate and the same comparison must say FAILED.
    module_path = next(p for p in candidate.files if str(p).endswith("_numpy.py"))
    broken = Candidate(
        unit=candidate.unit,
        transform=candidate.transform,
        files={
            **candidate.files,
            module_path: candidate.files[module_path].replace(b"GRAVITY", b"(GRAVITY * 1.0000001)"),
        },
        deferred=list(candidate.deferred),
        notes=dict(candidate.notes),
    )
    broken_workspace = tmp_path / "broken"
    broken_workspace.mkdir()
    failed = BitexactVerifier().verify(unit, broken, ref, broken_workspace, executor, config)
    assert failed.confidence is Confidence.FAILED
    assert failed.metrics["bit_exact"] < failed.metrics["points"]

    # ...unless the operator explicitly asked for a tolerance that excuses it.
    excused = BitexactVerifier().verify(
        unit, broken, ref, broken_workspace, executor, {**config, "rtol": 1e-3}
    )
    assert excused.confidence is Confidence.TOLERANCED


CALLBACK_SOURCE = """\
module toy_solver
    use iso_fortran_env, only: wp => real64
    implicit none
    real(wp), dimension(2), parameter :: limits = [epsilon(1.0_wp), tiny(1.0_wp)]
    real(wp), parameter :: eps = limits(1)

    abstract interface
        subroutine residual(n, x, fvec, iflag)
            import :: wp
            implicit none
            integer, intent(in) :: n
            real(wp), intent(in) :: x(n)
            real(wp), intent(out) :: fvec(n)
            integer, intent(inout) :: iflag
        end subroutine residual
    end interface

contains

    subroutine sweep(fcn, n, x, Work, Ldw, Iflag)
        implicit none
        procedure(residual) :: fcn
        integer, intent(in) :: n
        integer, intent(in) :: Ldw
        real(wp), intent(inout) :: x(n)
        real(wp), intent(inout) :: Work(Ldw, n)
        integer, intent(inout) :: Iflag
        integer :: j
        real(wp) :: h
        do j = 1, n
            call fcn(n, x, Work(1, j), Iflag)
            h = eps + norm(n, Work(1, j))
            x(j) = x(j) + h
        end do
    end subroutine sweep

    function norm(n, v) result(r)
        implicit none
        integer, intent(in) :: n
        real(wp) :: v(n)
        real(wp) :: r
        integer :: i
        r = 0.0_wp
        do i = 1, n
            r = r + v(i)*v(i)
        end do
        r = sqrt(r)
    end function norm

end module toy_solver
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_a_unit_that_takes_a_procedure_ends_bit_exact(tmp_path: Path) -> None:
    """The whole chain for a subprogram whose argument is something to call.

    Five things have to hold at once, and each one alone used to stop the
    unit: the working precision comes from ``iso_fortran_env``; a parameter
    is an array of type inquiries and the next one subscripts it; ``call
    fcn(...)`` is bound against an abstract interface; ``Work(1, j)`` is
    sequence-associated on both an OUT actual and a function argument; and
    the reference wrapper declares the procedure as an f2py call-back so both
    sides call the *same* Python object.
    """
    (tmp_path / "toy_solver.f90").write_text(CALLBACK_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    gate = ReadWriteSetVerifier().check(unit, candidate, workspace, executor, {})
    assert gate.passed, gate.detail

    config = {
        "root": tmp_path,
        "fc": GFORTRAN,
        "trials": 3,
        "dims": {"n": 4, "ldw": 4},
        "ranges": {"x": (-1.0, 1.0), "work": (-1.0, 1.0), "iflag": (1, 1)},
    }
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["uncovered"] == []
    assert verdict.metrics["points"] > 0


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_example_runs_through_the_cli(tmp_path: Path) -> None:
    """The roadmap's P2 claim, literally: `recast run translate examples/...`
    walks every stage and leaves evidence manifests behind."""
    import json
    import shutil as _shutil

    from recast.cli import main
    from recast.run import output_root

    example = Path(__file__).resolve().parent.parent / "examples" / "toy_physics"
    staged = tmp_path / "toy_physics"
    _shutil.copytree(example, staged, ignore=_shutil.ignore_patterns(".recast", "output"))

    code = main(["run", "translate", str(staged), "--config", str(staged / "recast.json")])
    assert code == 0
    # Not under ``staged``: the run's output is ``output/toy_physics/``, which
    # is what makes the generated Python findable and keeps it out of the tree
    # it was generated from.
    manifests = list((output_root(staged, {}) / "evidence").rglob("*.json"))
    assert len(manifests) == 3  # rwset, bitexact, notary
    results = {json.loads(m.read_text())["result"]["verdict"] for m in manifests}
    assert results == {"sampled", "bit_exact", "symbolic"}


def test_the_oracle_defaults_to_public_subprograms() -> None:
    """The wrappers `use` the module, and a private symbol is not importable
    -- one private specific in the list fails the whole build."""
    from recast.model import Facts

    facts = Facts(
        unit="fortran:m",
        interface={
            "module": "m",
            "subprograms": [
                {"name": "api", "public": True},
                {"name": "detail", "public": False},
            ],
        },
    )
    assert F2pyGoldenOracle._subprograms(facts, {}) == ["api"]
    # Explicit config still wins, and then fails loudly if it names a private.
    assert F2pyGoldenOracle._subprograms(facts, {"subprograms": ["detail"]}) == ["detail"]


def test_wrappers_serve_a_file_of_bare_subprograms() -> None:
    """A file with no module borrows its stem for a name, so a `use` line
    would not compile -- the callee is an external. Dimension names the file
    use-imports arrive as local PARAMETERs, which is what lets f2py fold the
    declared shapes."""
    record = {
        "module": "dadadj",
        "is_module": False,
        "generics": {},
        "subprograms": [
            {
                "name": "dadadj_native",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "t",
                        "dtype": "float64",
                        "intent": "INOUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": "pcols"}, {"lb": "1", "ub": "pver"}],
                    }
                ],
            }
        ],
    }
    text, _ = wrappers_for(record, ["dadadj_native"], parameters={"pcols": 8, "pver": 30})
    assert "use dadadj" not in text
    assert "external dadadj_native" in text
    assert "integer, parameter :: pcols = 8" in text
    assert "real(8), intent(inout) :: t(pcols, pver)" in text


def test_the_gate_lets_a_candidate_shape_its_own_inputs(tmp_path: Path) -> None:
    """Per-name ranges cannot express structure -- a monotone pressure
    column, a consistent thickness field. A candidate may carry
    ``_PREPARE_INPUTS`` the way it carries ``_SIGNATURES``; both sides then
    receive the same shaped arrays, so it chooses the sampled region without
    touching the verdict."""
    import numpy as np

    module = tmp_path / "candidate"
    module.mkdir()
    (module / "shaped_numpy.py").write_text(
        """
import numpy as np

_SIGNATURES = {
    "step": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "float64", "intent": "IN", "optional": False,
             "dims": [{"lb": "1", "ub": "n"}]},
            {"name": "y", "dtype": "float64", "intent": "OUT", "optional": False,
             "dims": [{"lb": "1", "ub": "n"}]},
        ],
        "result": None, "result_dtype": None,
    }
}
SEEN = []


def _PREPARE_INPUTS(name, inputs, rng):
    inputs["x"][:] = 2.0        # every trial sees the same shaped input


def step(x):
    SEEN.append(float(x[0]))
    return np.asarray(x) * 3.0
"""
    )

    class Truth:
        @staticmethod
        def w_step(x):
            return np.asarray(x) * 3.0

    candidate = Candidate(
        unit="fortran:shaped",
        transform="t",
        files={Path("shaped_numpy.py"): (module / "shaped_numpy.py").read_bytes()},
    )
    ref = OracleRef(
        unit="fortran:shaped",
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"step": "w_step"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:shaped", kind="module"),
        candidate,
        ref,
        tmp_path / "ws",
        LocalExecutor(),
        {"trials": 3, "dims": {"n": 4}, "ranges": {"x": (100.0, 200.0)}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT
    staged = tmp_path / "ws" / "candidate"
    sys.path.insert(0, str(staged))
    try:
        import shaped_numpy

        # The hook ran: every trial saw 2.0, not a value from the range.
        assert shaped_numpy.SEEN and all(v == 2.0 for v in shaped_numpy.SEEN)
    finally:
        sys.path.remove(str(staged))
        sys.modules.pop("shaped_numpy", None)


def test_the_oracle_side_is_called_with_lowercased_names(tmp_path: Path) -> None:
    """Fortran is case-insensitive and f2py lowercases every dummy name, so
    a candidate reporting `sl_prePBL` must still reach the same oracle
    argument. The source's spelling is not a fact about the interface."""
    staged = tmp_path / "cand"
    staged.mkdir()
    (staged / "mixed_numpy.py").write_text(
        """
import numpy as np

_SIGNATURES = {
    "step": {
        "kind": "subroutine",
        "args": [
            {"name": "inVal", "dtype": "float64", "intent": "IN", "optional": False},
            {"name": "outVal", "dtype": "float64", "intent": "OUT", "optional": False},
        ],
        "result": None, "result_dtype": None,
    }
}


def step(inVal):
    return inVal * 2.0
"""
    )

    class Truth:
        @staticmethod
        def w_step(**kwargs):
            # f2py's own convention: lowercase only.
            return kwargs["inval"] * 2.0

    candidate = Candidate(
        unit="fortran:mixed",
        transform="t",
        files={Path("mixed_numpy.py"): (staged / "mixed_numpy.py").read_bytes()},
    )
    ref = OracleRef(
        unit="fortran:mixed",
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"step": "w_step"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:mixed", kind="module"),
        candidate,
        ref,
        tmp_path / "ws",
        LocalExecutor(),
        {"trials": 2, "ranges": {"inval": (1.0, 2.0)}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail


@pytest.mark.skipif(GFORTRAN is None, reason="the cache key asks the compiler its version")
def test_a_refused_build_fails_this_stage_and_not_the_run(tmp_path: Path) -> None:
    """An executor that will not run the build is an unavailable oracle.

    ``run_recipe`` catches ``RecastError`` and marks the unit's oracle stage
    failed; anything else escapes it. A refusal that arrives as a bare
    ``RuntimeError`` therefore costs every *other* unit its verdict too, which
    is a much larger blast radius than the one build that could not run.
    """
    from recast.conformance.doubles import RefusingExecutor
    from recast.errors import OracleUnavailable

    (tmp_path / "toy_physics.f90").write_text(SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_physics")
    facts = frontend.analyze(unit, tmp_path)
    with pytest.raises(OracleUnavailable, match="did not run the reference compile"):
        F2pyGoldenOracle().materialize(
            unit,
            facts,
            tmp_path / "work",
            RefusingExecutor(),
            {"root": str(tmp_path)},
        )


KINDS_SOURCE = """\
module toy_kinds
  use, intrinsic :: iso_fortran_env
  implicit none
  integer, parameter :: wp = real64
end module toy_kinds
"""

SPLIT_SOURCE = """\
module toy_split
  use toy_kinds, only: wp
  implicit none
contains
  subroutine scale_all(n, a, x)
    integer, intent(in) :: n
    real(wp), intent(in) :: a
    real(wp), intent(inout) :: x(*)
    integer :: i
    do i = 1, n
      x(i) = a * x(i)
    end do
  end subroutine scale_all
end module toy_split
"""


class _CaptureBuild:
    """Records every job and lets the reference compiles through.

    The build is two phases -- the compiler over the reference sources, then
    f2py over the wrapper alone -- and the tokens of both have to be looked
    at, so the compiles are reported as having succeeded and only the f2py
    job stops the run.
    """

    name = "capture"

    def __init__(self) -> None:
        self.job = None
        self.jobs: list[object] = []

    def run(self, job):
        self.jobs.append(job)
        if "numpy.f2py" not in job.argv:
            for at, token in enumerate(job.argv):
                if token == "-o":
                    (job.cwd / job.argv[at + 1]).write_bytes(b"")
            return JobResult(0, "", "")
        self.job = job
        raise OracleUnavailable("captured before execution")


def _split_tree(tmp_path: Path) -> tuple[Unit, object]:
    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE)
    (tmp_path / "toy_split.f90").write_text(SPLIT_SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_split")
    return unit, frontend.analyze(unit, tmp_path)


def test_the_reference_names_the_siblings_the_unit_uses(tmp_path: Path) -> None:
    """A module that takes its precision from a kinds module one file over
    does not compile alone -- gfortran wants a ``.mod`` nobody built. The
    frontend already resolved the sibling, so the build asks the facts rather
    than making the operator list it by hand."""
    from recast.oracle.f2py import companion_sources

    _unit, facts = _split_tree(tmp_path)
    assert companion_sources(facts, tmp_path) == [(tmp_path / "toy_kinds.f90").resolve()]


def test_f2py_only_receives_canonical_source_and_include_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NumPy joins/splits sources internally, so hostile original path text
    must be absent from every argv token and from both compiler flag strings."""
    project = tmp_path / "source tree" / "-fplugin=must-not-be-a-flag"
    project.mkdir(parents=True)
    unit, facts = _split_tree(project)
    capture = _CaptureBuild()
    monkeypatch.setattr("recast.oracle.f2py._compiler_version", lambda _compiler: "test-fc 1")

    with pytest.raises(OracleUnavailable, match="captured before execution"):
        F2pyGoldenOracle().materialize(
            unit,
            facts,
            tmp_path / "work",
            capture,
            {"root": project, "fflags": "-O2 -fcheck=bounds"},
        )

    assert capture.job is not None
    assert all(
        str(project) not in token and "must-not-be-a-flag" not in token
        for job in capture.jobs
        for token in job.argv
    )

    compiles = [job for job in capture.jobs if job is not capture.job]
    assert [token for job in compiles for token in job.argv if token.startswith("sources/")] == [
        "sources/source_0000.f90",
        "sources/source_0001.f90",
    ]
    assert all("-Iincludes/d0000" in job.argv for job in compiles)

    argv = list(capture.job.argv)
    assert argv[argv.index("--build-dir") + 1] == "f2py-build"
    assert "--f90flags=-O2 -fcheck=bounds" in argv
    assert "--f77flags=-O2 -fcheck=bounds" in argv

    compile_index = argv.index("-c")
    module_index = argv.index("-m", compile_index)
    build_inputs = argv[compile_index + 3 : module_index]
    include_args = [token for token in build_inputs if token.startswith("-I")]
    # Only the generated wrapper is parsed by f2py; the reference arrives as
    # objects the compiler already made.
    assert [token for token in build_inputs if token.endswith(".f90")] == ["sources/wrappers.f90"]
    assert [token for token in build_inputs if token.endswith(".o")] == [
        "object_0000.o",
        "object_0001.o",
    ]
    assert include_args == ["-Iincludes/d0000", "-Iincludes/mods"]
    assert all(" " not in token for token in build_inputs)
    assert (capture.job.cwd / "sources/wrappers.f90").is_file()
    assert (capture.job.cwd / "includes/d0000").is_dir()
    assert (capture.job.cwd / "f2py-build/includes/d0000").is_dir()
    assert (capture.job.cwd / "includes/mods").is_dir()
    assert (capture.job.cwd / "f2py-build/includes/mods").is_dir()


class _FailingBuild:
    """An executor whose f2py run fails the way crackfortran does on bad Fortran.

    The reference compiles before it are reported as having succeeded: the
    build is two phases and this double is about the second one.
    """

    name = "failing"

    def __init__(self, stdout: str, stderr: str) -> None:
        self.stdout = stdout
        self.stderr = stderr

    def run(self, job):
        if "numpy.f2py" not in job.argv:
            for at, token in enumerate(job.argv):
                if token == "-o":
                    (job.cwd / job.argv[at + 1]).write_bytes(b"")
            return JobResult(0, "", "")
        return JobResult(1, self.stdout, self.stderr)


def test_a_failed_build_quotes_the_end_of_its_own_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log sits in a workspace that is gone once the run returns, so a
    path to it explains nothing; the error carries what the tools said."""
    root = tmp_path / "root"
    root.mkdir()
    unit, facts = _split_tree(root)
    monkeypatch.setattr("recast.oracle.f2py._compiler_version", lambda _compiler: "test-fc 1")
    chatter = "\n".join(f"Reading fortran codes... line {index}" for index in range(400))
    stderr = (
        "Traceback (most recent call last):\n"
        "  File crackfortran.py, line 3, in crackline\n"
        "crackfortran: analyzeline: No name/args pattern found for line: subroutine (x\n"
    )

    with pytest.raises(ConfigError) as failure:
        F2pyGoldenOracle().materialize(
            unit, facts, tmp_path / "work", _FailingBuild(chatter, stderr), {"root": root}
        )

    message = str(failure.value)
    assert message.startswith("f2py build for fortran:toy_split failed (exit 1); log at ")
    assert message.endswith(stderr.strip())
    assert "earlier characters of the build output omitted]" in message
    assert "Reading fortran codes... line 0\n" not in message
    assert len(message) < 3500
    log = next((tmp_path / "work").rglob("f2py.log"))
    assert log.read_text() == chatter + "\n" + stderr


def test_log_tail_keeps_short_output_whole_and_cuts_long_output_on_a_line() -> None:
    assert f2py_module._log_tail("  short\n") == "short"
    lines = [f"line {index:04d}" for index in range(100)]
    tail = f2py_module._log_tail("\n".join(lines), limit=200)
    omitted, kept = tail.split("\n", 1)
    assert omitted.startswith("… [") and omitted.endswith(
        " earlier characters of the build output omitted]"
    )
    assert kept.splitlines()[0] in lines
    assert kept.splitlines()[-1] == "line 0099"
    assert len(kept) <= 200
    assert "\n".join(lines).endswith(kept)


@pytest.mark.parametrize("bad_source", ["missing", "directory", "escape"])
def test_main_source_must_be_a_regular_file_inside_root(tmp_path: Path, bad_source: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    unit, facts = _split_tree(root)
    outside = tmp_path / "outside.f90"
    outside.write_text(SPLIT_SOURCE)
    if bad_source == "missing":
        facts.provenance["source"] = "missing.f90"
        match = "does not exist"
    elif bad_source == "directory":
        (root / "directory.f90").mkdir()
        facts.provenance["source"] = "directory.f90"
        match = "not a regular file"
    else:
        (root / "escape.f90").symlink_to(outside)
        facts.provenance["source"] = "escape.f90"
        match = "outside the configured project root"

    with pytest.raises(ConfigError, match=match):
        F2pyGoldenOracle().key(unit, facts, {"root": root})


@pytest.mark.parametrize("bad_source", ["missing", "directory", "escape"])
def test_companions_must_be_regular_files_inside_root(tmp_path: Path, bad_source: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    unit, facts = _split_tree(root)
    companion = facts.provenance["companions"][0]
    outside = tmp_path / "outside.f90"
    outside.write_text(KINDS_SOURCE)
    if bad_source == "missing":
        companion["source"] = "missing.f90"
        match = "does not exist"
    elif bad_source == "directory":
        (root / "directory.f90").mkdir()
        companion["source"] = "directory.f90"
        match = "not a regular file"
    else:
        (root / "escape.f90").symlink_to(outside)
        companion["source"] = "escape.f90"
        match = "outside the configured project root"

    with pytest.raises(ConfigError, match=match):
        F2pyGoldenOracle().key(unit, facts, {"root": root})


@pytest.mark.parametrize("bad_source", ["missing", "directory", "escape"])
def test_the_flat_oracle_holds_its_plan_to_the_same_root(tmp_path: Path, bad_source: str) -> None:
    """The flat oracle plans its library from the unit's source under the
    project root, the way the engine's oracle reads it: the same boundary,
    or a source outside root would be compiled by one oracle and refused by
    the other."""
    from recast.oracle.flat import F2pyFlatOracle

    root = tmp_path / "root"
    root.mkdir()
    unit, facts = _split_tree(root)
    outside = tmp_path / "outside.f90"
    outside.write_text(SPLIT_SOURCE)
    if bad_source == "missing":
        facts.provenance["source"] = "missing.f90"
        match = "does not exist"
    elif bad_source == "directory":
        (root / "directory.f90").mkdir()
        facts.provenance["source"] = "directory.f90"
        match = "not a regular file"
    else:
        (root / "escape.f90").symlink_to(outside)
        facts.provenance["source"] = "escape.f90"
        match = "outside the configured project root"

    with pytest.raises(ConfigError, match=match):
        F2pyFlatOracle().key(unit, facts, {"root": root})


def test_the_flat_oracle_refuses_an_extra_source_it_cannot_read(tmp_path: Path) -> None:
    """A configured extra source that is not there used to drop out of the
    library key without a word and fail the build later, under a message
    about the compiler."""
    from recast.oracle.flat import F2pyFlatOracle

    root = tmp_path / "root"
    root.mkdir()
    unit, facts = _split_tree(root)
    with pytest.raises(ConfigError, match=r"extra source 0 .* does not exist"):
        F2pyFlatOracle().key(unit, facts, {"root": root, "extra_sources": ["nowhere.f90"]})


def test_the_flat_oracle_keeps_an_include_dir_with_a_space_as_one_flag(tmp_path: Path) -> None:
    """The plan carries the compiler flags as one string and the library
    build splits it back into argv. A configured include directory with a
    space in its name used to be appended bare and come apart at the split,
    leaving ``-I/path`` and a stray ``name`` for gfortran to read as a file."""
    import shlex

    from recast.oracle.flat import F2pyFlatOracle

    root = tmp_path / "root"
    root.mkdir()
    include = tmp_path / "head ers"
    include.mkdir()
    unit, facts = _split_tree(root)
    config = {"root": root, "include_dirs": [str(include)]}

    plan = F2pyFlatOracle()._plan(unit, facts, config)
    flags = shlex.split(plan["fflags"])
    assert f"-I{include}" in flags
    assert flags.count(f"-I{include}") == 1
    assert not any(token == "ers" for token in flags)


def test_a_changed_sibling_moves_the_cache_key(tmp_path: Path) -> None:
    """The reference is only a reference if everything that can change what it
    computes is in its key. A kinds module edited from real64 to real32 is a
    different reference, not the same one."""
    unit, facts = _split_tree(tmp_path)
    oracle = F2pyGoldenOracle()
    config = {"root": tmp_path, "fc": GFORTRAN or "gfortran"}
    before = oracle.key(unit, facts, config)
    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE.replace("real64", "real32"))
    assert oracle.key(unit, facts, config) != before


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_reference_builds_across_two_files(tmp_path: Path) -> None:
    """The same case, actually compiled. Every library in the public corpus
    that keeps its working precision in its own module failed here, on a
    ``Cannot open module file`` that named a file sitting beside the source.
    The original directory is intentionally unsafe as an argv/flags spelling:
    staging must make both whitespace and a flag-looking component harmless."""
    project = tmp_path / "project with spaces" / "-fplugin=not-a-real-plugin"
    project.mkdir(parents=True)
    unit, facts = _split_tree(project)
    workspace = tmp_path / "work"
    workspace.mkdir()
    ref = F2pyGoldenOracle().materialize(
        unit, facts, workspace, LocalExecutor(), {"root": project, "fc": GFORTRAN}
    )
    assert ref.handle["wrappers"]["scale_all"] == "w_scale_all"


def test_a_lower_bound_is_spelled_in_the_wrapper() -> None:
    """``lhs(-2:2, ngrdcol, ndim)`` (CLUBB's pentadiagonal solvers) has five
    rows; a wrapper declaring ``lhs(2, ...)`` would hand the callee two."""
    from recast.oracle.f2py import _extent

    assert _extent({"lb": "-2", "ub": "2"}) == "-2:2"
    assert _extent({"lb": "1", "ub": "n"}) == "n"
    assert _extent({"lb": None, "ub": "n"}) == "n"
    assert _extent({"lb": "0", "ub": "nlev"}) == "0:nlev"


BLOCK_SOURCE = """\
module block_mod
  use iso_fortran_env, only: wp => real64
  implicit none
contains
  subroutine clamp(n, x, y)
    integer, intent(in) :: n
    real(wp), intent(in) :: x(n)
    real(wp), intent(out) :: y(n)
    integer :: i
    main: block
      do i = 1, n
        y(i) = x(i)
        if (y(i) < 0.0_wp) then
          y(i) = -y(i)
          cycle
        end if
        if (y(i) > 1.0e3_wp) exit main
      end do
    end block main
  end subroutine clamp
end module block_mod
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_a_reference_f2py_cannot_parse_is_still_compiled(tmp_path: Path) -> None:
    """A BLOCK construct is Fortran 2008 and f2py's own parser does not know
    it: handed the file, crackfortran counts the ``end block`` as closing a
    group it never opened and takes the whole build down. Only the generated
    wrapper is f2py's to read; the reference is the compiler's, which is the
    one thing in this build that understands Fortran."""
    (tmp_path / "block_mod.f90").write_text(BLOCK_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 3, "ranges": {"x": (-5.0, 5.0)}}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["bit_exact"] == verdict.metrics["points"] > 0
