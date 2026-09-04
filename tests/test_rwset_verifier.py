"""Tests for ``static.rwset``, the translate recipe's first gate.

The verifier knows no Fortran: the source's read and write sets arrive already
reduced to names, and everything else it needs -- which output lines a block
became, what was renamed -- the Transform has to say. These tests exercise it
the way a Transform would, with a hand-built Candidate, so nothing here depends
on a Transform existing yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, Unit
from recast.plugins.verifier import StaticVerifier
from recast.verify.rwset import Protocol, ReadWriteSetVerifier, factory, span_rwset

EMITTED = '''\
def fill(out, n, flag, want_gam=False):
    """A stand-in for what a Transform emits."""
    acc = 0.0
    for i in range(1, n + 1):
        acc = acc + POOL[i]
        if flag:
            out[i] = acc
    gam = 0.0
    if want_gam:
        gam = acc * CPAIR
    return out, gam
'''

# Line numbers in EMITTED, 1-based, as a Transform would record them.
SPANS = {"B001": [3, 3], "B002": [4, 7], "B003": [9, 11]}


def _candidate(blocks: list[dict[str, Any]], **protocol: Any) -> Candidate:
    return Candidate(
        unit="fortran:demo/fill",
        transform="translate.numpy",
        files={Path("demo_numpy.py"): EMITTED.encode()},
        notes={
            "rwset": {
                "file": "demo_numpy.py",
                "blocks": blocks,
                "names": {"POOL": "pool", "CPAIR": "cpair"},
                "procedures": ["fill"],
                # A NumPy backend emits `range`; declaring it is how a backend
                # says "this name is mine", rather than the verifier guessing.
                "scaffolding": ["range"],
                **protocol,
            }
        },
    )


def _block(bid: str, reads: list[str], writes: list[str]) -> dict[str, Any]:
    return {
        "subprogram": "fill",
        "block": bid,
        "reads": reads,
        "writes": writes,
        "lines": SPANS[bid],
    }


AGREEING = [
    _block("B001", [], ["acc"]),
    _block("B002", ["acc", "flag", "i", "n", "pool"], ["acc", "i", "out"]),
    _block("B003", ["acc", "cpair", "gam"], ["gam"]),
]


@pytest.fixture
def verify(tmp_path):
    verifier = factory()

    def run(candidate: Candidate, **config: Any):
        return verifier.check(
            Unit(uid=candidate.unit, kind="subprogram"),
            candidate,
            tmp_path,
            LocalExecutor(),
            config,
        )

    return run


def test_it_is_a_static_verifier(verify) -> None:
    assert isinstance(factory(), StaticVerifier)
    assert factory().name == "static.rwset"
    assert factory().provides is Confidence.SAMPLED


def test_matching_sets_pass(verify) -> None:
    verdict = verify(_candidate(AGREEING))
    assert verdict.confidence is Confidence.SAMPLED, verdict.detail
    assert verdict.metrics["blocks_checked"] == 3
    assert verdict.metrics["blocks_matched"] == 3


def test_an_extra_read_on_the_target_side_fails(verify) -> None:
    """As much a failure as a missing one: the translation depends on something
    the source does not, which is how a supposedly pure kernel picks up a
    dependency on state nobody intended it to see."""
    blocks = [
        _block("B001", [], ["acc"]),
        _block("B002", ["acc", "flag", "i", "n"], ["acc", "i", "out"]),
    ]
    verdict = verify(_candidate(blocks))
    assert verdict.confidence is Confidence.FAILED
    failure = verdict.metrics["failures"][0]
    assert failure["block"] == "fill/B002"
    assert failure["reads_target_only"] == ["pool"]


def test_a_missing_write_fails_and_names_the_block(verify) -> None:
    blocks = [_block("B001", [], ["acc", "ghost"])]
    verdict = verify(_candidate(blocks))
    assert verdict.confidence is Confidence.FAILED
    assert "fill/B001" in verdict.detail
    assert verdict.metrics["failures"][0]["writes_source_only"] == ["ghost"]


def test_metrics_carry_the_numbers_not_just_the_conclusion(verify) -> None:
    verdict = verify(_candidate(AGREEING))
    assert set(verdict.metrics) >= {"blocks_checked", "blocks_matched", "blocks_deferred"}


# --- what it will not wave through -------------------------------------------


def test_a_candidate_with_no_protocol_fails_closed(verify) -> None:
    """A Transform that records nothing is not one that passed; it is one this
    gate could not read."""
    bare = Candidate(
        unit="fortran:demo/fill", transform="mystery", files={Path("x.py"): b"a = 1\n"}
    )
    verdict = verify(bare)
    assert verdict.confidence is Confidence.FAILED
    assert "mystery" in verdict.detail


def test_emitted_code_that_does_not_parse_fails(verify) -> None:
    candidate = _candidate(AGREEING)
    candidate.files[Path("demo_numpy.py")] = b"def broken(:\n"
    assert verify(candidate).confidence is Confidence.FAILED


def test_a_deferred_block_is_not_compared(verify) -> None:
    """Nothing was translated mechanically, so there is nothing to compare --
    the block is already on the agent queue."""
    candidate = _candidate([_block("B001", ["nonsense"], [])])
    candidate.deferred = ["fill/B001"]
    verdict = verify(candidate)
    assert verdict.confidence is Confidence.SAMPLED
    assert verdict.metrics["blocks_checked"] == 0


def test_a_waived_block_is_named_in_the_verdict(verify) -> None:
    """A gate that passed because something was skipped has to say what."""
    verdict = verify(
        _candidate([_block("B001", ["nonsense"], [])]),
        waivers={"fill/B001": "external I/O, no comparable dataflow"},
    )
    assert verdict.confidence is Confidence.SAMPLED
    assert "external I/O" in verdict.detail
    assert verdict.metrics["blocks_waived"] == 1


# --- name resolution ---------------------------------------------------------


def test_an_optional_output_sentinel_is_a_read_of_its_variable(verify) -> None:
    """``want_gam`` on the target side is ``present(gam)`` on the source side.

    The pipeline this came from skipped any ``if`` whose test mentioned its
    goto-region label, matched as the substring ``_g`` -- which also matches
    ``want_gam``. Every optional-output block for a variable whose name made
    that substring appear was dropped from verification silently.
    """
    reads, writes = span_rwset(
        __import__("ast").parse(EMITTED),
        *SPANS["B003"],
        Protocol(names={"CPAIR": "cpair"}, procedures=frozenset({"fill"})),
    )
    assert "gam" in reads and "gam" in writes


def test_a_builtin_name_is_still_a_variable(verify) -> None:
    """``sum``, ``min`` and ``len`` are ordinary Fortran variable names. Only
    what a backend declares as scaffolding is skipped."""
    import ast as pyast

    code = "def f():\n    sum = 1.0\n    out = sum\n"
    protocol = Protocol()
    reads, writes = span_rwset(pyast.parse(code), 2, 3, protocol)
    assert reads == {"sum"} and writes == {"sum", "out"}

    declared = Protocol(scaffolding=frozenset({"sum"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 3, declared)
    assert reads == set() and writes == {"out"}


def test_a_store_to_a_procedure_name_is_the_result_convention(verify) -> None:
    """``function f(...)`` returns by assigning to ``f``. Skipping that as a
    call would lose the write the whole routine exists to make."""
    import ast as pyast

    code = "def outer():\n    f = 1.0\n    x = f(2)\n"
    protocol = Protocol(procedures=frozenset({"f"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 3, protocol)
    assert writes == {"f", "x"}
    assert "f" not in reads, "the call on line 3 is not a read of f"


def test_a_hoisted_literal_is_not_a_read(verify) -> None:
    import ast as pyast

    code = "def f():\n    x = F_273P15 + I_5\n"
    reads, _ = span_rwset(pyast.parse(code), 2, 2, Protocol())
    assert reads == set()


def test_a_component_name_is_not_a_read(verify) -> None:
    """``b.q[n]`` is a write of ``b``; the source side says the same."""
    import ast as pyast

    code = "def f():\n    b.q[n] = 0.0\n"
    reads, writes = span_rwset(pyast.parse(code), 2, 2, Protocol())
    assert writes == {"b"} and reads == {"n"}


def test_a_keyword_renamed_variable_maps_back_without_being_told(verify) -> None:
    """``lambda`` is a Python keyword, so ``lambda_`` reads back as ``lambda``
    whatever the backend is. No declaration needed for a fact about the
    language itself."""
    import ast as pyast

    reads, _ = span_rwset(pyast.parse("def f():\n    x = lambda_\n"), 2, 2, Protocol())
    assert reads == {"lambda"}


def test_a_module_alias_collision_has_to_be_declared(verify) -> None:
    """``np`` is only reserved because *this* backend imports NumPy under that
    name. A verifier that assumed it would misread a Fortran variable called
    ``np_`` under any backend that does not."""
    import ast as pyast

    from recast.transform.numpy.vocabulary import RESERVED

    code = "def f():\n    x = np_\n"
    undeclared, _ = span_rwset(pyast.parse(code), 2, 2, Protocol())
    assert undeclared == {"np_"}, "not renamed by anything this verifier knows"

    declared, _ = span_rwset(pyast.parse(code), 2, 2, Protocol(reserved=RESERVED))
    assert declared == {"np"}


def test_the_verifier_reports_against_the_candidate_it_judged(verify) -> None:
    candidate = _candidate(AGREEING)
    verdict = verify(candidate)
    assert verdict.candidate == candidate.digest()
    assert verdict.unit == candidate.unit
    assert verdict.verifier == "static.rwset"


def test_the_verifier_registers_under_its_recipe_name() -> None:
    from recast.registry import REGISTRY

    assert isinstance(REGISTRY.get("verifier", "static.rwset")(), ReadWriteSetVerifier)


# --- the result-variable convention and the alias rule -----------------------


def test_a_load_of_the_blocks_own_name_is_its_result_variable() -> None:
    """Inside ``no_limiter``, the name ``no_limiter`` is the result variable
    -- ``no_limiter = transfer(limiter_off, no_limiter)`` reads real data. A
    load of any *other* procedure name stays a call."""
    import ast as pyast

    code = "def no_limiter():\n    no_limiter = transfer(limiter_off, no_limiter)\n"
    protocol = Protocol(procedures=frozenset({"no_limiter", "transfer"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 2, protocol, own="no_limiter")
    assert "no_limiter" in reads and "no_limiter" in writes
    assert "transfer" not in reads


def test_recursion_is_a_call_not_a_read_of_the_result() -> None:
    import ast as pyast

    code = "def fact(n):\n    fact = n * fact(n - 1)\n"
    protocol = Protocol(procedures=frozenset({"fact"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 2, protocol, own="fact")
    assert "fact" not in reads  # callee position: control flow
    assert "fact" in writes  # the result assignment


def test_a_procedure_passed_as_an_argument_stays_skipped() -> None:
    """``_f_ecall(e_scale, a, s)`` passes the procedure itself; only the
    block's own name is ever data."""
    import ast as pyast

    code = "def caller(a, s):\n    a = _f_ecall(e_scale, a, s)\n"
    protocol = Protocol(procedures=frozenset({"e_scale"}), scaffolding=frozenset({"_f_ecall"}))
    reads, _ = span_rwset(pyast.parse(code), 2, 2, protocol, own="caller")
    assert "e_scale" not in reads


def test_alias_attributes_are_the_siblings_globals() -> None:
    """``_wv.omeps = 1.0 - _wv.epsilo`` is a write of ``omeps`` and a read of
    ``epsilo`` -- the source spells both as bare use-imported names, and an
    alias treated as scaffolding would blind the gate to cross-module state."""
    import ast as pyast

    code = "def init():\n    _wv.omeps = 1.0 - _wv.epsilo\n"
    protocol = Protocol(aliases=frozenset({"_wv"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 2, protocol)
    assert reads == {"epsilo"}
    assert writes == {"omeps"}


def test_an_alias_attribute_naming_a_procedure_is_a_call() -> None:
    import ast as pyast

    code = "def use_it(t, es):\n    es = _wv.wv_sat_svp_water(t)\n"
    protocol = Protocol(aliases=frozenset({"_wv"}), procedures=frozenset({"wv_sat_svp_water"}))
    reads, writes = span_rwset(pyast.parse(code), 2, 2, protocol)
    assert reads == {"t"}
    assert writes == {"es"}


def test_a_raise_nested_in_a_branch_is_scaffolding(verify) -> None:
    """A statement stub for an abort is ``raise RuntimeError(...)``. Standing
    alone in a block it was already skipped; nested in a contained ``if`` it
    reached the generic visit and ``RuntimeError`` became a read the source
    never made, failing every block with a guarded ``endrun`` in it."""
    emitted = EMITTED.replace(
        "        if flag:\n            out[i] = acc\n",
        "        if flag:\n            raise RuntimeError('endrun')\n",
    )
    candidate = _candidate(
        [
            _block("B001", [], ["acc"]),
            _block("B002", ["acc", "flag", "i", "n", "pool"], ["acc", "i"]),
            _block("B003", ["acc", "cpair", "gam"], ["gam"]),
        ]
    )
    candidate.files = {Path("demo_numpy.py"): emitted.encode()}
    verdict = verify(candidate)
    assert verdict.confidence is Confidence.SAMPLED, verdict.detail


def test_a_copy_out_writes_its_target(verify) -> None:
    """``_f_copy_out(u, tridiag(...))`` is how the runtime fills a caller's
    out-argument buffer; the target is written, not read (CLM-ml's
    LongwaveRadiation reported ``utri`` unwritten)."""
    emitted = EMITTED.replace(
        "        if flag:\n            out[i] = acc\n",
        "        if flag:\n            _f_copy_out(out, POOL)\n",
    )
    candidate = _candidate(
        [
            _block("B001", [], ["acc"]),
            _block("B002", ["acc", "flag", "i", "n", "pool"], ["acc", "i", "out"]),
            _block("B003", ["acc", "cpair", "gam"], ["gam"]),
        ],
        scaffolding=["range", "_f_copy_out"],
    )
    candidate.files = {Path("demo_numpy.py"): emitted.encode()}
    verdict = verify(candidate)
    assert verdict.confidence is Confidence.SAMPLED, verdict.detail


def test_the_where_constructs_masks_are_scaffolding() -> None:
    """``_wm``, ``_wn`` and ``_we<depth>_<n>`` are the emitter's masks for a
    where / masked elsewhere / elsewhere; a real name never looks like one."""
    from recast.verify.rwset import DISCARD

    for name in ("_wm", "_wm2", "_wn", "_wn2", "_we0_1", "_we1_3", "_", "_g"):
        assert DISCARD.fullmatch(name), name
    for name in ("_wet", "_we", "_wn_x", "wn", "_f_copy_out", "x_we0_1"):
        assert not DISCARD.fullmatch(name), name
