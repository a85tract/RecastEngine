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


def test_a_keyword_renamed_variable_maps_back(verify) -> None:
    """A Fortran variable called ``lambda`` is emitted as ``lambda_``."""
    import ast as pyast

    code = "def f():\n    x = lambda_ + np_\n"
    reads, _ = span_rwset(pyast.parse(code), 2, 2, Protocol())
    assert reads == {"lambda", "np"}


def test_the_verifier_reports_against_the_candidate_it_judged(verify) -> None:
    candidate = _candidate(AGREEING)
    verdict = verify(candidate)
    assert verdict.candidate == candidate.digest()
    assert verdict.unit == candidate.unit
    assert verdict.verifier == "static.rwset"


def test_the_verifier_registers_under_its_recipe_name() -> None:
    from recast.registry import REGISTRY

    assert isinstance(REGISTRY.get("verifier", "static.rwset")(), ReadWriteSetVerifier)
