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
from tests._f2py_trees import KINDS_SOURCE, SPLIT_SOURCE, _split_tree

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


def test_an_extent_that_is_an_intrinsic_call_is_not_a_hidden_dummy() -> None:
    """An extent naming neither an argument nor a parameter becomes a hidden
    integer dummy the caller supplies. ``integer :: b(size(a))`` names one of
    each: ``a`` is the argument, and ``size`` is a call. Hiding it declared
    ``integer, intent(in) :: size`` beside ``res(size(a))``, which gfortran
    rejects twice -- PROCEDURE conflicting with INTENT, and a call to
    something not PURE -- and no reference for the corpus's sorting module
    could be built."""
    record = {
        "module": "sort_mod",
        "generics": {"argsort": ["iargsort"]},
        "subprograms": [
            {
                "name": "iargsort",
                "kind": "function",
                "args": [
                    {
                        "name": "a",
                        "dtype": "int32",
                        "intent": "IN",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}],
                    }
                ],
                "result": "b",
                "result_dtype": "int32",
                "result_dims": [{"lb": "1", "ub": "size(a)"}],
            }
        ],
    }
    text, _ = wrappers_for(record, ["iargsort"])
    assert "subroutine w_iargsort(a, res)" in text
    assert "intent(in) :: size" not in text.lower()
    assert "integer, intent(out) :: res(size(a))" in text


def test_an_extent_naming_an_argument_in_another_case_is_not_hidden() -> None:
    """Fortran does not distinguish ``N`` from ``n``. The extent keeps the
    source's spelling and the argument names arrive lowercased from the
    frontend, so ``real(dp) :: mesh(N+1)`` over ``integer, intent(in) :: N``
    hid an ``N`` beside the wrapper's own ``n`` -- a duplicate formal argument
    gfortran refuses, which took the mesh module's three exponential-mesh
    functions out of the reference build."""
    record = {
        "module": "mesh",
        "subprograms": [
            {
                "name": "meshexp",
                "kind": "function",
                "args": [
                    {"name": "rmin", "dtype": "float64", "intent": "IN", "optional": False},
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                ],
                "result": "mesh",
                "result_dtype": "float64",
                "result_dims": [{"lb": "1", "ub": "N + 1"}],
            }
        ],
    }
    text, _ = wrappers_for(record, ["meshexp"])
    assert "subroutine w_meshexp(rmin, n, res)" in text
    assert text.lower().count("intent(in) :: n") == 1
    assert "real(8), intent(out) :: res(N + 1)" in text


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


def test_a_caller_buffer_out_array_is_the_caller_s_on_both_sides() -> None:
    """An intent(out) array the callee cannot size -- ``dy(*)`` -- is the
    caller's storage on both sides: the gate generates it and hands the same
    values to the reference and the candidate. Zeroing it in the wrapper would
    fail every cell the callee never writes, and ``dy = 0`` is not even legal
    for an assumed-size dummy (SLSQP's ``dcopy`` stopped the oracle build).
    And f2py cannot allocate an ``intent(out)`` it cannot size (PCHIP's
    evaluators: ``failed to create intent(cache|hide) array``): the buffer
    is spelled ``inout``, goes in, and is written in place."""
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
    assert "real(8), intent(inout) :: dy(*)" in body
    assert "  dy = 0" not in body


def test_a_character_out_dummy_is_defined_with_a_string() -> None:
    """``ss = 0`` is a type error the compiler rejects outright -- "Cannot
    convert INTEGER(4) to CHARACTER(128)" -- and it cost every module with a
    character output its whole reference, not just that one wrapper."""
    record = {
        "module": "text_mod",
        "subprograms": [
            {
                "name": "getword",
                "kind": "subroutine",
                "args": [
                    {"name": "s", "dtype": "str", "intent": "IN", "optional": False},
                    {"name": "ss", "dtype": "str", "intent": "OUT", "optional": False},
                    {"name": "ok", "dtype": "bool", "intent": "OUT", "optional": False},
                ],
            }
        ],
    }
    text, _ = wrappers_for(record, ["getword"])
    assert "  ss = ''" in text
    assert "  ok = .false." in text


def test_an_allocatable_dummy_is_passed_an_allocatable_actual() -> None:
    """``call loadtxt(filename, d)`` with ``d`` a plain assumed-shape dummy is
    "Actual argument for 'd' must be ALLOCATABLE at (1)": the call does not
    compile, so the unit gets no reference at all. The wrapper keeps its
    caller-side buffer -- f2py has no allocatable to offer -- and calls
    through a local one."""
    record = {
        "module": "io_mod",
        "subprograms": [
            {
                "name": "loadtxt",
                "kind": "subroutine",
                "args": [
                    {"name": "filename", "dtype": "str", "intent": "IN", "optional": False},
                    {
                        "name": "d",
                        "dtype": "float64",
                        "intent": "OUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}, {"lb": "1", "ub": None}],
                        "allocatable": True,
                        "buffer": True,
                    },
                ],
            }
        ],
    }
    text, _ = wrappers_for(record, ["loadtxt"])
    body = text[text.index("subroutine w_loadtxt") : text.index("end subroutine w_loadtxt")]
    assert "real(8), intent(in out) :: d(:, :)" in body
    assert "real(8), allocatable :: d_alloc(:, :)" in body
    assert "  call loadtxt(filename, d_alloc)" in body
    # What the callee allocated, as far as the caller's buffer reaches.
    assert "d_n = min(shape(d), shape(d_alloc))" in body
    assert "d(:d_n(1), :d_n(2)) = d_alloc(:d_n(1), :d_n(2))" in body


def test_a_reference_the_differential_cannot_exercise_is_named_ungated() -> None:
    """The wrapper compiles; calling it is what means nothing. A character
    dummy is fixed at ``len=128`` and has no draw, and an array the callee
    allocates is not the buffer f2py hands back. Named, with the reason, so
    the verdict can say why a public subprogram was not compared -- silence
    is what the gate refuses."""
    record = {
        "module": "mix_mod",
        "subprograms": [
            {
                "name": "loadtxt",
                "kind": "subroutine",
                "args": [
                    {"name": "filename", "dtype": "str", "intent": "IN", "optional": False},
                    {
                        "name": "d",
                        "dtype": "float64",
                        "intent": "OUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}],
                        "allocatable": True,
                    },
                ],
            },
            {
                "name": "arange",
                "kind": "subroutine",
                "args": [
                    {"name": "a", "dtype": "float64", "intent": "IN", "optional": False},
                    {
                        "name": "u",
                        "dtype": "float64",
                        "intent": "OUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}],
                        "allocatable": True,
                    },
                ],
            },
            {
                "name": "label",
                "kind": "function",
                "args": [{"name": "i", "dtype": "int32", "intent": "IN", "optional": False}],
                "result": "s",
                "result_dtype": "str",
            },
            {
                "name": "newunit",
                "kind": "function",
                "args": [{"name": "unit", "dtype": "int32", "intent": "OUT", "optional": True}],
                "result": "n",
                "result_dtype": "int32",
            },
            {
                # PCHIP's ``dpchfe``/``dpchfd``/``dpchcm`` shape: a mandatory
                # scalar LOGICAL SKIP the caller can toggle across repeated
                # calls. f2py marshals a scalar INOUT as a writable rank-0
                # array, so this compares fine.
                "name": "dpchfe",
                "kind": "subroutine",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {"name": "skip", "dtype": "bool", "intent": "INOUT", "optional": False},
                ],
            },
            {
                "name": "monotonic",
                "kind": "subroutine",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {"name": "skip", "dtype": "bool", "intent": "INOUT", "optional": True},
                ],
            },
            {
                # PCHIP's DPCHIA/DPCHID shape: a FUNCTION that also declares
                # a mandatory SKIP/IERR dummy. The verifier's
                # ``_paired_outputs`` only ever pairs a function's single
                # result, so this compares nothing today -- named ungated
                # here rather than reaching that refusal and failing the
                # whole unit's differential gate.
                "name": "dpchia",
                "kind": "function",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {"name": "skip", "dtype": "bool", "intent": "INOUT", "optional": False},
                    {"name": "ierr", "dtype": "int32", "intent": "OUT", "optional": False},
                ],
                "result": "value",
                "result_dtype": "float64",
            },
            {
                # Unlike a scalar, an array LOGICAL INOUT needs f2py's
                # in-place buffer, whose element size this harness's 1-byte
                # bool draw does not match.
                "name": "flags_inout",
                "kind": "subroutine",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {
                        "name": "flags",
                        "dtype": "bool",
                        "intent": "INOUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}],
                    },
                ],
            },
        ],
    }
    reasons = {s["name"]: f2py_module.unexercisable(s) for s in record["subprograms"]}
    assert reasons["loadtxt"].startswith("filename: character dummy")
    assert reasons["arange"] == "u: allocatable array the callee sizes"
    assert reasons["label"] == "character result, fixed at len=128 by the wrapper"
    # An optional dummy is dropped from both calls, so it disqualifies nothing.
    assert reasons["newunit"] is None
    # A scalar LOGICAL INOUT, mandatory or not, compares fine.
    assert reasons["dpchfe"] is None
    assert reasons["monotonic"] is None
    # A function's mandatory OUT/INOUT dummies have no side-effect leg to
    # pair with the result, unlike a subroutine's.
    assert reasons["dpchia"].startswith("declares OUT/INOUT dummy argument(s) skip, ierr")
    assert reasons["flags_inout"].startswith("flags: LOGICAL INOUT array dummy")


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


def _derived_record(**overrides: object) -> dict:
    """A module whose public subroutine carries its state in a derived type
    of scalar components -- SLSQP's ``slsqp`` and its ``slsqpb_data``."""
    record = {
        "module": "m",
        "generics": {},
        "types": {
            "state_t": {
                "a": {"dtype": "float64", "dims": None, "allocatable": False, "pointer": False},
                "i": {"dtype": "int32", "dims": None, "allocatable": False, "pointer": False},
            }
        },
        "public_types": ["state_t"],
        "subprograms": [
            {
                "name": "step",
                "kind": "subroutine",
                "public": True,
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {
                        "name": "x",
                        "dtype": "float64",
                        "intent": "INOUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": "n"}],
                    },
                    {
                        "name": "state",
                        "dtype": "UNKNOWN(TYPE(STATE_T))",
                        "intent": "INOUT",
                        "optional": False,
                    },
                ],
            }
        ],
    }
    record.update(overrides)
    return record


def test_a_derived_type_of_scalars_is_spelled_component_by_component() -> None:
    """f2py cannot marshal a derived type, and a module whose only public
    subprogram takes one had no reference at all. A type of scalar
    components is spelled as one flat dummy per component, copied into a
    local of the type before the call and back out after it, and the plan
    for the candidate side names the same flat dummies in the same order."""
    record = _derived_record()
    text, names = wrappers_for(record, ["step"])
    assert names == ["w_step"]
    assert "subroutine w_step(n, x, state_a, state_i)" in text
    assert "  use m, only: step, state_t" in text
    assert "  type(state_t) :: state" in text
    assert "  real(8), intent(in out) :: state_a" in text
    assert "  integer, intent(in out) :: state_i" in text
    assert text.index("  state%a = state_a") < text.index("  call step(n, x, state)")
    assert text.index("  call step(n, x, state)") < text.index("  state_i = state%i")
    assert f2py_module.unspellable(record, ["step"]) == {}
    plan = f2py_module.flattened_dummies(record, ["step"])
    assert plan == {
        "step": {
            "state": {
                "type": "state_t",
                "components": [
                    {"name": "state_a", "component": "a", "dtype": "float64"},
                    {"name": "state_i", "component": "i", "dtype": "int32"},
                ],
            }
        }
    }


def test_a_derived_type_the_wrapper_cannot_flatten_says_why() -> None:
    """Refused, with the reason, rather than compiled into a wrapper that
    does not build: a type the module does not export cannot be USEd, an
    array component is not a scalar the flat dummy can carry, and a type
    the record never defined has no components to spell."""
    private = _derived_record(public_types=[])
    with pytest.raises(ConfigError, match=r"cannot spell.*not public"):
        wrappers_for(private, ["step"])
    arrays = _derived_record(
        types={
            "state_t": {
                "v": {
                    "dtype": "float64",
                    "dims": [{"lb": "1", "ub": "3"}],
                    "allocatable": False,
                    "pointer": False,
                }
            }
        }
    )
    with pytest.raises(ConfigError, match=r"cannot spell.*not a scalar"):
        wrappers_for(arrays, ["step"])
    unknown = _derived_record(types={})
    with pytest.raises(ConfigError, match=r"cannot spell.*not defined"):
        wrappers_for(unknown, ["step"])
    assert f2py_module.flattened_dummies(unknown, ["step"]) == {}
    assert set(f2py_module.unspellable(unknown, ["step"])) == {"step"}


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


FUNCTION_CALLBACK_RECORD = {
    "module": "optimize",
    "generics": {},
    "interfaces": {
        "func": {
            "kind": "function",
            "args": [{"name": "x", "dtype": "float64", "intent": "IN", "optional": False}],
            "result": "func",
            "result_dtype": "float64",
        }
    },
    "subprograms": [
        {
            "name": "bisect",
            "kind": "function",
            "args": [
                {
                    "name": "f",
                    "dtype": "PROCEDURE",
                    "intent": "IN",
                    "optional": False,
                    "procedure": True,
                    "interface": "func",
                },
                {"name": "a", "dtype": "float64", "intent": "IN", "optional": False},
                {"name": "b", "dtype": "float64", "intent": "IN", "optional": False},
                {"name": "tol", "dtype": "float64", "intent": "IN", "optional": False},
            ],
            "result": "c",
            "result_dtype": "float64",
        }
    ],
}


def test_a_function_procedure_argument_becomes_an_f2py_call_back() -> None:
    """crackfortran tells a function call-back from a subroutine one by the
    statement that uses it: a CALL is a subroutine, and an assignment whose
    right-hand side calls the dummy is a function whose result type is the
    assigned variable's. The dummy carries that type in real Fortran too,
    because the wrapper is ``implicit none``."""
    text, names = wrappers_for(FUNCTION_CALLBACK_RECORD, ["bisect"])
    assert names == ["w_bisect"]
    assert "  real(8), external :: f" in text
    assert "!f2py  real(8), intent(in) :: cb_f_x" in text
    assert "!f2py  real(8) :: cb_f_res" in text
    assert "!f2py  cb_f_res = f(cb_f_x)" in text
    # The declaration has to reach crackfortran before the line that assigns
    # to it, or the call-back's result has no type.
    assert text.index("real(8) :: cb_f_res") < text.index("cb_f_res = f(")


def test_a_function_call_back_that_writes_an_argument_refuses() -> None:
    """Its result and its written argument both come back, and which one f2py
    hands over first is not a convention this wrapper shares with the
    translation."""
    record = {
        **FUNCTION_CALLBACK_RECORD,
        "interfaces": {
            "func": {
                "kind": "function",
                "args": [
                    {"name": "x", "dtype": "float64", "intent": "IN", "optional": False},
                    {"name": "ierr", "dtype": "int32", "intent": "OUT", "optional": False},
                ],
                "result": "func",
                "result_dtype": "float64",
            }
        },
    }
    with pytest.raises(ConfigError, match=r"only\s+read theirs"):
        wrappers_for(record, ["bisect"])


def test_the_harness_builds_a_function_call_back() -> None:
    """A function call-back answers through its return on both sides, so the
    stand-in returns one value rather than a tuple of written arguments."""
    import numpy as np

    from recast.verify.bitexact import callback_for

    callback = callback_for(np, "f", FUNCTION_CALLBACK_RECORD["interfaces"]["func"])
    assert callback.__code__.co_argcount == 1
    value = callback(np.float64(2.0))
    assert isinstance(value, np.floating)
    assert value == callback(np.float64(2.0))  # deterministic


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
    """No Python buffer spelling is a portable Fortran LOGICAL INOUT ABI. A
    scalar goes through the wrapper as an integer, 0 or 1 (PCHIP's ``skip``);
    an array of them has no such path and is refused by name, before either
    side runs."""
    emitted = b"""\
import numpy as np

_SIGNATURES = {
    "flip": {
        "kind": "subroutine",
        "args": [
            {"name": "a", "dtype": "bool", "intent": "INOUT", "optional": False,
             "dims": [{"lb": "1", "ub": "3"}]},
            {"name": "y", "dtype": "bool", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def flip(a):
    raise AssertionError("candidate subroutine must not execute")
"""
    candidate = Candidate(
        unit="fortran:logical_inout_array",
        transform="translate.numpy",
        files={Path("logical_inout_array_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_flip(a):
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
    assert "array dummy argument(s) a " in verdict.detail


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


def test_f2py_badname_argument_is_spelled_with_its_bn_suffix(tmp_path: Path) -> None:
    """A dummy that collides with a C keyword -- PCHIP's ``dpchic`` declares
    one named ``switch`` -- is not the keyword the compiled reference answers
    to. f2py's own ``crackfortran`` frontend renames every name in its
    ``badnames`` table to ``<name>_bn`` before it reaches the extension, so
    the reference call must spell it that way too, not lowercased verbatim."""
    emitted = b"""\
_SIGNATURES = {
    "scale": {
        "kind": "function",
        "args": [
            {"name": "switch", "dtype": "float64", "intent": "IN", "optional": False},
        ],
        "result": "y",
        "result_dtype": "float64",
    }
}

def scale(switch):
    return switch * 2.0
"""
    candidate = Candidate(
        unit="fortran:badname_argument",
        transform="translate.numpy",
        files={Path("badname_argument_numpy.py"): emitted},
    )

    class Truth:
        @staticmethod
        def w_scale(switch_bn):
            return switch_bn * 2.0

    ref = OracleRef(
        unit=candidate.unit,
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"scale": "w_scale"}},
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
                    # A kind the frontend could not resolve: complex itself
                    # is a dtype the gate draws and compares now (#20).
                    "dtype": "UNKNOWN_COMPLEX_KIND(qp)",
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


def test_a_recorded_subroutine_with_no_output_returns_nothing(tmp_path: Path) -> None:
    """CLUBB's finalize_tau_sponge_damp_api deallocates a component and
    returns: no OUT argument, so its adapter returns ``None``. The gate
    counted that as one value against zero out-intent arguments."""
    emitted = b"""\
_SIGNATURES = {
    "release": {
        "kind": "subroutine",
        "args": [
            {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def release(n):
    return None
"""
    candidate = Candidate(
        unit="fortran:no_output",
        transform="translate.numpy",
        files={Path("no_output_numpy.py"): emitted},
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
                    "subprogram": "release",
                    "source": "release.txt",
                    "inputs": {"n": 3},
                    "outputs": {},
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
    # Nothing to compare is still not a pass -- but for the right reason.
    assert "returned 1 value(s)" not in verdict.detail
    assert "zero numerical points" in verdict.detail


def test_a_declared_ungated_subprogram_is_not_compared_on_a_recording(tmp_path: Path) -> None:
    """CLUBB's sponge initializer leaves the levels below the layer undefined
    on both sides; the operator's declaration says so, with the reason, and
    the replay reported it -- then compared the heap against np.empty anyway."""
    emitted = b"""\
_SIGNATURES = {
    "fill": {
        "kind": "subroutine",
        "args": [
            {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
            {"name": "y", "dtype": "int32", "intent": "OUT", "optional": False},
        ],
        "result": None,
        "result_dtype": None,
    }
}

def fill(n):
    return 2
"""
    candidate = Candidate(
        unit="fortran:undefined_tail",
        transform="translate.numpy",
        files={Path("undefined_tail_numpy.py"): emitted},
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
                    "subprogram": "fill",
                    "source": "fill.txt",
                    "inputs": {"n": 3},
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
        {"ungated": {"fill": "the tail is undefined on both sides"}},
    )
    assert "differ" not in verdict.detail
    assert "fill (the tail is undefined on both sides)" in verdict.detail


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


PACKED_SOURCE = """\
module packed_workspace
  implicit none
contains
  subroutine solve(n, lr, r, qtb, x)
    integer, intent(in) :: n
    integer, intent(in) :: lr
    real(8), intent(in) :: r(lr)
    real(8), intent(in) :: qtb(n)
    real(8), intent(out) :: x(n)
    integer :: i, j, jj, jp1, k, l
    real(8) :: sm
    jj = (n*(n + 1))/2 + 1
    do k = 1, n
       j = n - k + 1
       jp1 = j + 1
       jj = jj - k
       l = jj + 1
       sm = 0.0d0
       if (n >= jp1) then
          do i = jp1, n
             sm = sm + r(l)*x(i)
             l = l + 1
          end do
       end if
       x(j) = (qtb(j) - sm)/r(jj)
    end do
  end subroutine solve
end module packed_workspace
"""
"""MINPACK's ``dogleg`` back-substitution: ``r(lr)`` holds the upper triangle
of an order-``n`` matrix, so the subprogram takes no draw where ``lr`` is
``n`` -- which is what every extent nobody pinned defaults to."""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_a_packed_workspace_is_compared_against_real_f2py(tmp_path: Path) -> None:
    """The extent is grown to a shape the body takes and the whole spine runs
    on it: without that, the first subscript the translation forms is past the
    end of ``r`` at every trial, and the subprogram is one no draw compared."""
    (tmp_path / "packed_workspace.f90").write_text(PACKED_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 4}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    solve = verdict.metrics["subprograms"]["solve"]
    # The order stays at the default the run was configured with; the
    # workspace is what grew, and the outcome says so.
    assert solve["extents"] == {"lr": 64}
    assert (solve["points"], solve["redrawn"], solve["reshaped"]) == (32, 0, 0)


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_real_f2py_logical_inout_array_fails_closed_scalar_compares(tmp_path: Path) -> None:
    """The real array ABI hazard is reported, never mistaken for a mismatch,
    but a scalar LOGICAL INOUT in the same module -- PCHIP's ``dpchfe``
    shape -- still gets compared against the real compiled reference."""
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

    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert "flip_scalar" in verdict.metrics["subprograms"]
    assert "LOGICAL INOUT array dummy" in verdict.metrics["ungated"]["flip_array"]


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
    # A call-back is this process's Python object: the reference stays here
    # (an error stop in it would end the run), and the verdict says so.
    assert ref.handle["isolation"].startswith("in-process (call-back arguments: sweep")
    assert verdict.metrics["reference_isolation"] == ref.handle["isolation"]
    assert verdict.metrics["uncovered"] == []
    assert verdict.metrics["points"] > 0


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_example_runs_through_the_cli(tmp_path: Path) -> None:
    """The roadmap's P2 claim, literally: `recast run translate corpus/...`
    walks every stage and leaves evidence manifests behind."""
    import json
    import shutil as _shutil

    from recast.cli import main
    from recast.run import output_root

    example = Path(__file__).resolve().parent.parent / "corpus" / "toy_physics"
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


def test_a_specific_of_a_public_generic_is_reachable_though_its_name_is_not() -> None:
    """A module may publish nothing but generics: the corpus's sorting module
    is ``private`` with ``public sort, sortpairs, argsort`` over twelve
    specifics, every one of them private. Selecting on the specific's own
    accessibility left nothing to wrap and no reference to build, while
    ``wrappers_for`` stood ready to call each one through its generic -- which
    is the name the wrapper ``use``s, and which is public.
    """
    from recast.model import Facts

    facts = Facts(
        unit="fortran:sorting",
        interface={
            "module": "sorting",
            "public": ["sort", "sortpairs"],
            "generics": {"sort": ["sortnums"], "sortpairs": ["sortnumnumpairs"], "hidden": ["aux"]},
            "subprograms": [
                {
                    "name": "sortnums",
                    "kind": "subroutine",
                    "public": False,
                    "args": [
                        {
                            "name": "nums",
                            "dtype": "float64",
                            "intent": "INOUT",
                            "optional": False,
                            "dims": [{"lb": "1", "ub": None}],
                        }
                    ],
                },
                {
                    "name": "sortnumnumpairs",
                    "kind": "subroutine",
                    "public": False,
                    "args": [
                        {
                            "name": "nums1",
                            "dtype": "float64",
                            "intent": "INOUT",
                            "optional": False,
                            "dims": [{"lb": "1", "ub": None}],
                        }
                    ],
                },
                # Behind a generic nothing published: still unreachable.
                {"name": "aux", "kind": "subroutine", "public": False, "args": []},
            ],
        },
    )
    assert F2pyGoldenOracle._subprograms(facts, {}) == ["sortnums", "sortnumnumpairs"]


def test_a_reached_specific_the_wrapper_cannot_spell_is_dropped_not_fatal() -> None:
    """Reached, not exported. ``sortpairs`` also covers a COMPLEX overload,
    which ``FORTRAN_TYPES`` has no spelling for; raising on it would cost the
    other ten specifics their reference for the sake of one the module never
    named. A *public* name of the same dtype is still an error, because the
    module says it is part of its surface.
    """
    from recast.model import Facts

    def module(public_specific: bool) -> dict[str, object]:
        return {
            "module": "sorting",
            "public": ["sortpairs"],
            "generics": {"sortpairs": ["real_pairs", "complex_pairs"]},
            "subprograms": [
                {
                    "name": "real_pairs",
                    "kind": "subroutine",
                    "public": False,
                    "args": [
                        {
                            "name": "nums",
                            "dtype": "float64",
                            "intent": "INOUT",
                            "optional": False,
                            "dims": [{"lb": "1", "ub": None}],
                        }
                    ],
                },
                {
                    "name": "complex_pairs",
                    "kind": "subroutine",
                    "public": public_specific,
                    "args": [
                        {
                            "name": "nums",
                            "dtype": "UNKNOWN(COMPLEX)",
                            "intent": "INOUT",
                            "optional": False,
                            "dims": [{"lb": "1", "ub": None}],
                        }
                    ],
                },
            ],
        }

    reached = Facts(unit="fortran:sorting", interface=module(public_specific=False))
    assert F2pyGoldenOracle._subprograms(reached, {}) == ["real_pairs"]

    exported = Facts(unit="fortran:sorting", interface=module(public_specific=True))
    assert F2pyGoldenOracle._subprograms(exported, {}) == ["real_pairs", "complex_pairs"]
    with pytest.raises(ConfigError, match="cannot spell"):
        wrappers_for(exported.interface, ["complex_pairs"])


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


def test_the_project_profile_shapes_the_generated_inputs(tmp_path: Path) -> None:
    """Per-name ranges cannot express structure -- a monotone pressure
    column, a consistent thickness field. The project carries a
    ``recast_inputs.py`` at its root, and its ``prepare`` shapes every
    generated draw before both sides receive it, so it chooses the sampled
    region without touching the verdict -- and the candidate, which is the
    thing under judgement, has no say in it."""
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


def step(x):
    SEEN.append(float(x[0]))
    return np.asarray(x) * 3.0
"""
    )
    root = tmp_path / "project"
    root.mkdir()
    (root / "recast_inputs.py").write_text(
        "import numpy as np\n"
        "\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    assert unit == 'fortran:shaped' and subprogram == 'step'\n"
        "    inputs['x'] = np.full_like(inputs['x'], 2.0)  # every trial sees the same input\n"
        "    return inputs\n"
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
        {"root": str(root), "trials": 3, "dims": {"n": 4}, "ranges": {"x": (100.0, 200.0)}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT
    assert verdict.metrics["input_profile"] == "recast_inputs.py"
    assert verdict.metrics["shaped"] == ["step"]
    assert verdict.metrics["subprograms"]["step"]["shaped"] == 3
    staged = tmp_path / "ws" / "candidate"
    sys.path.insert(0, str(staged))
    try:
        import shaped_numpy

        # The profile ran: every trial saw 2.0, not a value from the range.
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


def test_the_reference_names_the_siblings_the_unit_uses(tmp_path: Path) -> None:
    """A module that takes its precision from a kinds module one file over
    does not compile alone -- gfortran wants a ``.mod`` nobody built. The
    frontend already resolved the sibling, so the build asks the facts rather
    than making the operator list it by hand."""
    from recast.oracle.f2py import companion_sources

    _unit, facts = _split_tree(tmp_path)
    assert companion_sources(facts, tmp_path) == [(tmp_path / "toy_kinds.f90").resolve()]


USER_SOURCE = """\
module toy_user
  use toy_split, only: scale_all
  implicit none
contains
  subroutine drive(n, x)
    integer, intent(in) :: n
    real, intent(inout) :: x(*)
    call scale_all(n, 2.0, x)
  end subroutine drive
end module toy_user
"""


def test_the_reference_also_gets_what_the_siblings_themselves_use(tmp_path: Path) -> None:
    """``toy_user`` cannot see ``toy_kinds``: ``toy_split`` answers for the
    only-list asked of it, so nothing of ``toy_kinds`` is in scope here. The
    build still needs the file -- ``toy_split.f90`` is compiled from source,
    and gfortran stops at "cannot open module file toy_kinds.mod" -- so the
    closure is staged, dependencies first."""
    from recast.oracle.f2py import companion_sources

    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE)
    (tmp_path / "toy_split.f90").write_text(SPLIT_SOURCE)
    (tmp_path / "toy_user.f90").write_text(USER_SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_user")
    facts = frontend.analyze(unit, tmp_path)
    assert companion_sources(facts, tmp_path) == [
        (tmp_path / "toy_kinds.f90").resolve(),
        (tmp_path / "toy_split.f90").resolve(),
    ]


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


def test_a_scalar_logical_inout_goes_through_the_wrapper_as_an_integer() -> None:
    """PCHIP's ``dpchfe(..., skip, ...)`` takes a LOGICAL it may set. The
    wrapper carries it as an integer -- 0 or 1 in, the logical to the
    callee, the integer out -- so the Python side needs no guess at the
    compiler's true; a function with such a dummy converts the same way."""
    record = {
        "module": "pc",
        "subprograms": [
            {
                "name": "fe",
                "kind": "subroutine",
                "args": [
                    {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                    {"name": "skip", "dtype": "bool", "intent": "INOUT", "optional": False},
                ],
            },
            {
                "name": "ia",
                "kind": "function",
                "result_dtype": "float64",
                "args": [
                    {"name": "skip", "dtype": "bool", "intent": "INOUT", "optional": False},
                ],
            },
        ],
    }
    text, _ = wrappers_for(record, ["fe", "ia"])
    body = text[text.index("subroutine w_fe") : text.index("end subroutine w_fe")]
    assert "integer, intent(inout) :: skip" in body and "logical :: skip_l" in body
    assert "skip_l = (skip /= 0)" in body
    assert "call fe(n, skip_l)" in body
    assert "skip = merge(1, 0, skip_l)" in body
    fn = text[text.index("function w_ia") : text.index("end function w_ia")]
    assert "res = ia(skip_l)" in fn and "skip = merge(1, 0, skip_l)" in fn


COMPLEX_VALUED = """\
module cplx_mod
  implicit none
  complex(8), parameter :: i_unit = ( 0.0d0, 1.0d0 )
contains
  function quadratic_solve( n, a, b, c ) result( roots )
    integer, intent(in) :: n
    real(8), dimension(n), intent(in) :: a, b, c
    complex(8), dimension(n,2) :: roots
    complex(8), dimension(n) :: sqrt_det
    sqrt_det = sqrt( cmplx( b**2 - 4.0d0 * a * c, kind = 8 ) )
    roots(:,1) = ( -cmplx( b, kind = 8 ) + sqrt_det ) / cmplx( 2.0d0 * a, kind = 8 )
    roots(:,2) = ( -cmplx( b, kind = 8 ) - sqrt_det ) / cmplx( 2.0d0 * a, kind = 8 )
  end function quadratic_solve
  subroutine conj_scale( n, z, s, w, re )
    integer, intent(in) :: n
    complex(8), dimension(n), intent(in) :: z
    real(8), intent(in) :: s
    complex(8), dimension(n), intent(out) :: w
    real(8), dimension(n), intent(out) :: re
    w = conjg( z ) * s * i_unit
    re = real( w, kind = 8 ) + aimag( z )
  end subroutine conj_scale
end module cplx_mod
"""


@pytest.mark.skipif(GFORTRAN is None or not MESON, reason="needs gfortran and meson")
def test_a_complex_valued_subprogram_is_compared_on_both_parts(tmp_path: Path) -> None:
    """A complex result or argument was ``unsupported declared dtype(s)`` and
    left uncompared, while the emitter built it as float64 (#20). Complex
    inputs are drawn on both parts, the wrapper spells ``complex(8)``, and
    the comparison is bit-exact on both parts of every element."""
    (tmp_path / "cplx_mod.f90").write_text(COMPLEX_VALUED)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert not candidate.deferred, candidate.deferred
    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 3, "dims": {"n": 4}}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    per = verdict.metrics["subprograms"]
    # 3 trials x n=4 x 2 roots x 2 parts; w's two parts and re's one, per element.
    assert per["quadratic_solve"]["points"] == 3 * 4 * 2 * 2
    assert per["conj_scale"]["points"] == 3 * 4 * 3
    # Conjugation and a real scale are componentwise on both sides: exact.
    assert per["conj_scale"]["bit_exact"] == per["conj_scale"]["points"]
    # Complex division is an algorithm, not an IEEE operation: gfortran's
    # rounds differently from NumPy's by one ULP on some elements, and the
    # gate now says so where it used to say nothing.
    assert per["quadratic_solve"]["max_ulp"] <= 1, per["quadratic_solve"]
    assert per["quadratic_solve"]["bit_exact"] >= per["quadratic_solve"]["points"] // 2
    assert verdict.confidence in (Confidence.BIT_EXACT, Confidence.FAILED), verdict.detail
    if verdict.confidence is Confidence.FAILED:
        assert "points differ (max 1 ULP" in (verdict.detail or ""), verdict.detail


READS_BELOW_THE_ARRAY = """\
module below_mod
  implicit none
contains
  subroutine spacing_end( n, x, h )
    integer, intent(in) :: n
    real(8), intent(in) :: x(*)
    real(8), intent(out) :: h
    ! PCHIP's dpchkt: assumes n >= 2 and does not check it.
    h = x(n) - x(n-1)
  end subroutine spacing_end
end module below_mod
"""


@pytest.mark.skipif(GFORTRAN is None or not MESON, reason="needs gfortran and meson")
def test_a_reference_reading_outside_its_array_declines_the_draw_by_name(tmp_path: Path) -> None:
    """A subscript outside the array is not a value the source computes: the
    reference reads whatever memory sits beside the buffer in its process,
    and the translation's negative index wraps to the other end of the array
    (#42). Neither is a fact about the Fortran. The reference is built with
    the bounds check on, so the draw ends its process with the array and the
    index named, and the gate declines the draw under its own name rather
    than comparing two undefined values -- and, every draw here being one,
    fails the routine by name with the reason a profile would have to
    answer."""
    (tmp_path / "below_mod.f90").write_text(READS_BELOW_THE_ARRAY)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 2, "dims": {"n": 1}}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.passed is False, verdict.detail
    detail = verdict.detail or ""
    assert "spacing_end: no draw this harness could compare" in detail, detail
    assert "reference subscript out of bounds" in detail, detail
    assert "below lower bound" in detail and "'x'" in detail, detail
    assert verdict.metrics["reference_isolation"] == "process"


STOPS_ON_NEGATIVE = """\
module stopper_mod
  implicit none
contains
  subroutine scale_pos( n, x, y )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    if ( any( x < 0.0d0 ) ) error stop 'scale_pos: negative input'
    y = 2.0d0 * x
  end subroutine scale_pos
end module stopper_mod
"""


@pytest.mark.skipif(GFORTRAN is None or not MESON, reason="needs gfortran and meson")
def test_an_error_stop_in_the_reference_is_a_report_not_a_dead_run(tmp_path: Path) -> None:
    """``error stop`` in the compiled reference is ``exit()`` in whatever
    process imported it: no report, no summary, every other unit's verdict
    gone with it (#21). With the reference in a process of its own, a draw
    it stops on is an answer. On generated draws the translated ERROR STOP
    declines the draw first; a profile that asserts the reference takes a
    negative input is told, by name and with the reference's own message,
    that it does not -- and this process is here to read it."""
    from recast.errors import InputProfileError

    (tmp_path / "stopper_mod.f90").write_text(STOPS_ON_NEGATIVE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    config = {
        "root": tmp_path,
        "fc": GFORTRAN,
        "trials": 3,
        "dims": {"n": 4},
        "ranges": {"x": (0.0, 1.0)},
    }
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert ref.handle["isolation"] == "process"
    assert verdict.metrics["reference_isolation"] == "process"

    (tmp_path / "recast_inputs.py").write_text(
        "import numpy as np\n\n\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    inputs['x'] = np.asfortranarray(-np.abs(inputs['x']) - 0.5)\n"
        "    return inputs\n"
    )
    with pytest.raises(InputProfileError) as caught:
        BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    message = str(caught.value)
    assert "ended the process" in message and "negative input" in message, message
    # The reference is there for the next call: a fresh worker, the same verdict.
    (tmp_path / "recast_inputs.py").unlink()
    again = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert again.confidence is Confidence.BIT_EXACT, again.detail
    assert ref.handle["module"].restarts == 1


def test_a_path_the_body_creates_is_exercisable_and_one_it_reads_is_not() -> None:
    """A character dummy has no draw in general. A path an OPEN in the body
    *creates* does: any name works, because the subprogram makes the file
    rather than finding one, and the gate compares what each side left there.
    A path opened ``STATUS='OLD'`` would need a draw that is a file already
    holding something, so it stays ungated -- and says so in those words,
    because "fixed at len=128" is not why."""
    record = {
        "module": "io_mod",
        "generics": {},
        "subprograms": [
            {
                "name": "saveppm",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "filename",
                        "dtype": "str",
                        "intent": "IN",
                        "optional": False,
                        "path": "created",
                    },
                    {
                        "name": "img",
                        "dtype": "int32",
                        "intent": "IN",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": None}, {"lb": "1", "ub": None}],
                    },
                ],
            },
            {
                "name": "loadppm",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "filename",
                        "dtype": "str",
                        "intent": "IN",
                        "optional": False,
                        "path": "existing",
                    },
                    {"name": "n", "dtype": "int32", "intent": "OUT", "optional": False},
                ],
            },
        ],
    }
    reasons = {s["name"]: f2py_module.unexercisable(s) for s in record["subprograms"]}
    assert reasons["saveppm"] is None
    assert reasons["loadppm"] == (
        "filename: names a file the body opens STATUS='OLD', which no generated draw can put there"
    )
    text, _ = wrappers_for(record, ["saveppm"])
    # TRIM, because the wrapper's dummy is padded to 128 and the callee's is
    # ``len=*``: without it the callee sees a length no caller ever passes.
    assert "  call saveppm(trim(filename), img)" in text


FILE_SOURCE = """\
module file_mod
  implicit none
contains
  subroutine save_grid(filename, n, d)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: n
    real(8), intent(in) :: d(n)
    integer :: u, i
    open(newunit=u, file=filename, status="replace")
    write(u, '(i0)') n
    do i = 1, n
      write(u, '(3a1)', advance='no') achar(modulo(int(d(i)), 60) + 40)
    end do
    write(u,*) d
    close(u)
  end subroutine save_grid

  subroutine load_grid(filename, n)
    character(len=*), intent(in) :: filename
    integer, intent(out) :: n
    integer :: u
    open(newunit=u, file=filename, status="old")
    read(u, '(i6)') n
    close(u)
  end subroutine load_grid
end module file_mod
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_a_subprogram_whose_only_product_is_a_file_is_gated_on_that_file(
    tmp_path: Path,
) -> None:
    """``save_grid`` declares two inputs and nothing else: every output the
    harness could pair is absent, and its whole result is the file it writes.
    So the file is the output -- each side writes its own, and the bytes are
    compared. ``load_grid`` reads a file that has to already exist, which no
    draw can produce, and stays ungated with that as the reason.

    The stub this replaces made the point sharply: the emitted ``save_grid``
    opened a file, wrote nothing to it and returned, and passed every
    structural check on the way.
    """
    (tmp_path / "file_mod.f90").write_text(FILE_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 3, "dims": {"n": 6}}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    assert ref.handle["ungated"] == {
        "load_grid": "filename: names a file the body opens STATUS='OLD', "
        "which no generated draw can put there"
    }

    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["bit_exact"] == verdict.metrics["points"] > 0
    # Every point is a byte of the file: the subprogram has no other output.
    assert verdict.metrics["integer_points"] == verdict.metrics["points"]
    assert set(verdict.metrics["subprograms"]) == {"save_grid"}
    assert "load_grid" in verdict.metrics["ungated"]


LIBRARY_RECORDS = [
    {
        "module": "solver",
        "subprograms": [
            {"name": "go", "calls": ["helper"], "external_calls": ["solve_it"]},
            {"name": "helper", "calls": [], "external_calls": []},
            {"name": "outer", "calls": ["go"], "external_calls": []},
        ],
        "interfaces": {},
    },
    {
        "module": "libwrap",
        "subprograms": [],
        "interfaces": {
            "solve_it": {"name": "solve_it", "kind": "subroutine"},
            "helper": {"name": "helper", "kind": "subroutine"},
            "shape_t": {"name": "shape_t"},
        },
    },
]


def test_a_procedure_declared_by_an_interface_and_defined_nowhere_is_named() -> None:
    """The reference build links nothing but the sources staged for it, so a
    name only an INTERFACE block declares is a symbol nothing defines: the
    extension links with it undefined and *importing* it fails, which costs
    the module's every other subprogram its reference too. ``helper`` is
    declared the same way and defined in the tree, so it is not one."""
    assert f2py_module.undefined_externals(LIBRARY_RECORDS, []) == ["solve_it"]


def test_a_definition_the_operator_added_from_outside_the_tree_is_not_stubbed(
    tmp_path: Path,
) -> None:
    """``extra_sources`` is where a build gets what the tree does not hold.
    Stubbing a name that source already defines is a duplicate symbol where
    there was a working reference -- and its own interface block, which is a
    declaration, must not be read as the definition."""
    extra = tmp_path / "lib.f90"
    extra.write_text(
        "interface\n  subroutine solve_it(n)\n  end subroutine\nend interface\n"
        "subroutine solve_it(n)\n  integer :: n\nend subroutine solve_it\n"
    )
    assert f2py_module.undefined_externals(LIBRARY_RECORDS, [extra]) == []


def test_every_caller_of_a_missing_library_is_found_through_the_call_graph() -> None:
    """``outer`` calls ``go``, which calls ``solve_it``. Neither can be run
    against a reference whose ``solve_it`` is a refusal, and only the closure
    says so about the first one."""
    reached = f2py_module.reaching(LIBRARY_RECORDS, {"solve_it"})
    assert reached == {"go": "solve_it", "outer": "solve_it"}


def test_a_stub_for_a_missing_library_refuses_rather_than_returns() -> None:
    """The reference exists to say what the original program computes, and for
    a call into a library this build does not have it cannot say. The symbol
    resolves, so the extension loads and the subprograms that never reach the
    library are compared as usual; anything that does reach it stops, naming
    the routine, rather than returning a number nobody computed."""
    text = f2py_module.unresolved_stubs(["solve_it"])
    assert "subroutine solve_it()" in text
    assert "error stop" in text and "solve_it has no definition" in text


LIBRARY_INTERFACE_SOURCE = """\
module tiny_lapack
  implicit none
  interface
    subroutine dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
      integer :: info, lda, ldb, n, nrhs
      integer :: ipiv(*)
      double precision :: a(lda,*), b(ldb,*)
    end subroutine
  end interface
end module tiny_lapack
"""

LIBRARY_CALLER_SOURCE = """\
module tiny_solver
  use tiny_lapack, only: dgesv
  implicit none
contains
  subroutine solve3(a, rhs, x)
    double precision, intent(in) :: a(3,3)
    double precision, intent(in) :: rhs(3)
    double precision, intent(out) :: x(3)
    double precision :: work(3,3), b(3,1)
    integer :: ipiv(3), info
    work = a
    b(:,1) = rhs
    call dgesv(3, 1, work, 3, ipiv, b, 3, info)
    x = b(:,1)
  end subroutine solve3
end module tiny_solver
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_a_call_into_a_declared_only_library_is_compared_not_disclaimed(tmp_path: Path) -> None:
    """The shape the corpus's ``splines`` has, end to end.

    ``tiny_lapack`` is interface blocks and nothing else -- the bodies are in
    a library neither build links -- so ``dgesv`` used to get a body that
    error-stops on the reference side and nothing at all on the candidate's,
    and every subprogram reaching it came out ungated: the unit passed with
    its one real subprogram never compared. ``recast.references`` defines it
    on both sides instead, from one implementation, so ``solve3`` is compared
    like anything else -- and the verdict says what stood in for the library,
    because the numbers it was compared at are not the ones LAPACK would have
    produced.
    """
    (tmp_path / "tiny_lapack.f90").write_text(LIBRARY_INTERFACE_SOURCE)
    (tmp_path / "tiny_solver.f90").write_text(LIBRARY_CALLER_SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:tiny_solver")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path, "profile": "gfortran"})
    assert candidate.deferred == []

    config = {"root": tmp_path, "fc": GFORTRAN, "trials": 5}
    ref = F2pyGoldenOracle().materialize(unit, facts, workspace, executor, config)
    assert ref.handle["ungated"] == {}, "nothing reaches a procedure with no definition now"
    assert "dgesv" in ref.handle["substituted"]

    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] == 15
    assert verdict.metrics["bit_exact"] == 15
    assert "ungated" not in verdict.metrics
    assert set(verdict.metrics["substituted"]) == {"dgesv"}
    assert "stood in for by recast's own reference implementation" in verdict.detail
