"""The C kernel frontend, the executable oracle, and the two verifiers over
shell scripts standing in for programs: no compiler, everything else real."""

from __future__ import annotations

from pathlib import Path

from recast.c.build import Spec, Toolchain, is_build_artifact, stage_directory, stage_siblings
from recast.c.frontend import CKernelFrontend, makefile_vars
from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, Facts, Unit
from recast.oracle.executable import ExecutableGoldenOracle
from recast.verify.gpu_time import GpuTimeVerifier, parse_nsys_csv, parse_nv_acc_time
from recast.verify.probes import ProbeVerifier

PROBES = (
    "GATE:SUM name=out dtype=u32 algo=fnv1a64 value=00ff n=4\n"
    "GATE:STAT name=out dtype=f32 n=4 min=0 max=3 mean=1.5 L1=6 L2=3.74\n"
)


def _script(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _plant(root: Path, *, mean: str = "1.5") -> None:
    (root / "golden").mkdir(parents=True)
    _script(root / "golden" / "main", f'echo "PASS"\nprintf "{PROBES}"\n')
    (root / "golden" / "input.txt").write_text("1 2 3 4\n")
    (root / "cand").mkdir()
    _script(
        root / "cand" / "main",
        f'echo "PASS"\nprintf "{PROBES.replace("mean=1.5", "mean=" + mean)}"\n',
    )


def _unit(root: Path) -> Unit:
    return Unit(
        uid="c:cand",
        kind="kernel",
        attrs={
            "build": {
                "dir": "cand",
                "steps": [["chmod", "+x", "main"]],
                "program": "main",
                "sources": ["main"],
            },
            "golden": {"dir": "golden", "steps": [], "program": "main", "sources": ["main"]},
        },
    )


def _candidate(root: Path, *, mean: str = "1.5") -> Candidate:
    body = f'#!/bin/sh\necho "PASS"\nprintf "{PROBES.replace("mean=1.5", "mean=" + mean)}"\n'
    return Candidate(unit="c:cand", transform="test", files={Path("cand/main"): body.encode()})


def test_frontend_discovers_directories_with_a_program(tmp_path: Path) -> None:
    kernel = tmp_path / "k1"
    kernel.mkdir()
    (kernel / "Makefile").write_text(
        "program = main\nsource = main.cpp util.cpp\nRUN_ARGS ?= 8 input.txt\n"
        "CFLAGS := -O2\nCFLAGS += -g\n"
    )
    (kernel / "main.cpp").write_text(
        "static int f(int x){return x;}\n"
        "int main(){ for(int i=0;i<2;i++){ for(int j=0;j<2;j++) f(i); } return 0; }\n"
    )
    (tmp_path / "not-a-kernel").mkdir()
    (tmp_path / "not-a-kernel" / "notes.txt").write_text("")
    fe = CKernelFrontend()
    units = list(fe.discover(tmp_path))
    assert [u.uid for u in units] == ["c:k1"]
    assert units[0].attrs["build"]["args"] == ["8", "input.txt"]
    assert units[0].sources == (Path("k1/main.cpp"),)
    facts = fe.analyze(units[0], tmp_path)
    assert facts.interface["subprograms"] == ["f", "main"]
    assert facts.effects["loops"] == 2 and facts.effects["loop_depth"] == 2
    assert makefile_vars(kernel / "Makefile")["CFLAGS"] == "-O2 -g"


def test_artifacts_are_binary_or_by_suffix_and_staging_drops_them(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _script(src / "run", "./main\n")
    (src / "setparams").write_bytes(b"\x7fELF\x00\x00")
    (src / "main.o").write_bytes(b"\x00")
    (src / "main.cpp").write_text('#include "../common/x.h"\nint main(){}\n')
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "x.h").write_text("")
    assert not is_build_artifact(src / "run")
    assert is_build_artifact(src / "setparams") and is_build_artifact(src / "main.o")
    dst = tmp_path / "staged" / "src"
    stage_directory(src, dst)
    assert sorted(p.name for p in dst.iterdir()) == ["main.cpp", "run"]
    assert stage_siblings(src, dst) and (tmp_path / "staged" / "common" / "x.h").exists()


def test_toolchain_renders_and_identifies() -> None:
    tc = Toolchain({"cc": "nvc++", "sm": "cc80", "env": {"OMP_TARGET_OFFLOAD": "MANDATORY"}})
    assert tc.render("{cc} -gpu={sm} -o {program}", program="a.out") == "nvc++ -gpu=cc80 -o a.out"
    assert tc.env()["OMP_TARGET_OFFLOAD"] == "MANDATORY"
    assert "cc80" in tc.identity()
    assert (
        Spec.from_attrs({"dir": "x", "steps": [["make"]], "program": "p{class}"}).program
        == "p{class}"
    )


def test_oracle_builds_runs_and_keys_on_toolchain(tmp_path: Path) -> None:
    _plant(tmp_path)
    unit, facts = _unit(tmp_path), Facts(unit="c:cand")
    oracle = ExecutableGoldenOracle()
    config = {"root": str(tmp_path), "toolchain": {"cc": "cc"}}
    ref = oracle.materialize(unit, facts, tmp_path / "ws", LocalExecutor(), config)
    assert "GATE:SUM" in ref.handle["stdout"] and (Path(ref.handle["dir"]) / "input.txt").exists()
    other = oracle.key(unit, facts, {**config, "toolchain": {"cc": "cc -O0"}})
    assert other != ref.key


def test_probe_verifier_passes_agreeing_and_fails_drifted(tmp_path: Path) -> None:
    _plant(tmp_path)
    unit, facts = _unit(tmp_path), Facts(unit="c:cand")
    executor = LocalExecutor()
    config = {"root": str(tmp_path), "toolchain": {"cc": "cc"}, "runs": 2}
    ref = ExecutableGoldenOracle().materialize(unit, facts, tmp_path / "ws", executor, config)
    verifier = ProbeVerifier()
    good = verifier.verify(unit, _candidate(tmp_path), ref, tmp_path / "ws", executor, config)
    assert good.confidence is Confidence.TOLERANCED, good.detail
    assert good.metrics["checksums_compared"] == 1 and good.metrics["runs"] == 2
    bad = verifier.verify(
        unit, _candidate(tmp_path, mean="9"), ref, tmp_path / "ws", executor, config
    )
    assert bad.confidence is Confidence.FAILED and bad.metrics["gate"] == "stats"


def test_benchmark_wall_clock_and_parsers(tmp_path: Path) -> None:
    _plant(tmp_path)
    unit = _unit(tmp_path)
    config = {"root": str(tmp_path), "toolchain": {"cc": "cc"}, "profiler": "wall", "runs": 2}
    verdict = GpuTimeVerifier().verify(
        unit, _candidate(tmp_path), None, tmp_path / "ws", LocalExecutor(), config
    )  # type: ignore[arg-type]
    assert verdict.confidence is Confidence.SAMPLED and verdict.metrics["runs"] == 2
    csv_text = (
        "** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):\n"
        "Time (%),Total Time (ns),Instances,Name\n"
        '90.0,"2,000,000",2,k1\n10.0,"1,000,000",1,k2\n\n'
        "** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):\n"
        "Time (%),Total Time (ns),Count,Operation\n"
        '70.0,"700,000",3,[CUDA memcpy Host-to-Device]\n'
        '30.0,"300,000",1,[CUDA memcpy Device-to-Host]\n'
    )
    got = parse_nsys_csv(csv_text)
    assert (got["kernel_ms"], got["htod_ms"], got["dtoh_ms"]) == (3.0, 0.7, 0.3)
    assert (
        parse_nv_acc_time("device time(us): total=1,500 max=2\n device time(us): total=500\n")
        == 2.0
    )
    assert parse_nv_acc_time("nothing") is None


def test_render_settles_nested_placeholders_and_drops_empty_args() -> None:
    tc = Toolchain({"cc": "nvc++", "sm": "cc80", "flags": "-gpu={sm}", "device_flags": ""})
    assert tc.render("{flags}") == "-gpu=cc80"
    assert tc.render("{device_flags}") == ""
