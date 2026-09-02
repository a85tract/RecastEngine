"""The probe protocol, held to gate_harness.py's behaviour."""

from __future__ import annotations

from recast.verify import probe_protocol as p

REF = """\
Average execution time per iteration: 0.001 (s)
PASS
GATE:SUM name=jacobi_f_checksum dtype=u32 algo=fnv1a64 value=00ABCDEF n=4194304
GATE:STAT name=jacobi_f_stats dtype=f32 n=4194304 min=-1 max=1 mean=0.001 L1=4000 L2=1200
"""


def _cand(mean: str = "0.001", value: str = "00abcdef") -> str:
    return REF.replace("mean=0.001", f"mean={mean}").replace("00ABCDEF", value)


def test_parse_reads_sums_and_stats() -> None:
    out = p.parse(REF)
    assert out.sums == {"jacobi_f_checksum:u32": "00abcdef"}
    assert out.stats["jacobi_f_stats:f32"]["n"] == 4194304
    assert out.stats["jacobi_f_stats:f32"]["L2"] == 1200.0
    assert out.has_probes and not out.problems


def test_identical_passes_and_reports_counts() -> None:
    r = p.compare(p.parse(REF), [p.parse(_cand()), p.parse(_cand())])
    assert r.passed and r.reason == "pass"
    assert (r.checksums_compared, r.stats_compared, r.runs) == (1, 1, 2)
    assert r.max_rel_err == 0.0


def test_stat_within_tolerance_passes_but_checksum_must_match() -> None:
    assert p.compare(p.parse(REF), [p.parse(_cand(mean="0.005"))]).passed
    bad = p.compare(p.parse(REF), [p.parse(_cand(value="00000001"))])
    assert not bad.passed and bad.reason == "checksum"


def test_stat_outside_tolerance_fails() -> None:
    r = p.compare(p.parse(REF), [p.parse(_cand(mean="0.5"))])
    assert not r.passed and r.reason == "stats"
    assert any("mean" in f for f in r.failures)


def test_nondeterministic_candidate_fails_before_comparison() -> None:
    r = p.compare(p.parse(REF), [p.parse(_cand()), p.parse(_cand(value="00000002"))])
    assert r.reason == "determinism"


def test_missing_probes_are_named() -> None:
    assert p.compare(p.parse("PASS\n"), [p.parse("PASS\n")]).reason == "no_gate_output"
    assert p.compare(p.parse(REF), [p.parse("PASS\n")]).reason == "no_cand_gate_output"
    assert p.compare(p.parse("PASS\n"), [p.parse(REF)]).reason == "no_ref_gate_output"


def test_nonfinite_stat_is_malformed() -> None:
    assert p.compare(p.parse(REF), [p.parse(_cand(mean="nan"))]).reason == "malformed_probe_output"


def test_stdout_fallback_semantics() -> None:
    assert p.stdout_agree("x\nPASS\n", "x\nPASS\n") == (True, "identical")
    assert p.stdout_agree("t=1\nPASS\n", "t=2\nPASS\n") == (True, "both_self_report_pass")
    assert p.stdout_agree("PASS\n", "FAIL\n") == (False, "mismatch")
    same = p.stdout_agree("t=1\nNon-Matching: 3\n", "t=2\nNon-Matching: 3\n")
    assert same == (True, "same_signature")


def test_suite_specific_readings() -> None:
    rodinia = (
        "GPU Runtime: 0.01s\n"
        "Non-Matching CPU-GPU Outputs Beyond Error Threshold of 0.05 Percent: 0\n"
    )
    assert p.is_success_output(rodinia)
    npb = " Verification    =               SUCCESSFUL\n Zeta is  8.59\n Error is  1.7E-13\n"
    assert p.is_success_output(npb)
    assert not p.is_success_output(npb.replace("SUCCESSFUL", "UNSUCCESSFUL"))
