"""``performance.benchmark``: how long the device worked, measured three ways.

The metric a GPU-offload paper reports is device time: every kernel's
execution plus every host/device transfer, API overhead excluded. Two tools
give it and both are here, chosen by ``profiler``:

``nsys``
    Nsight Systems: ``nsys profile`` around the run, then ``nsys stats`` for
    the ``cuda_gpu_kern_sum`` and ``cuda_gpu_mem_time_sum`` reports in CSV;
    the ``Total Time (ns)`` columns are summed.
``nv_acc_time``
    the NVIDIA HPC compilers' built-in timer: ``NV_ACC_TIME=1`` in the
    environment makes the runtime print per-region ``device time(us):
    total=`` lines, which are summed. What works where CUPTI cannot inject.
``wall``
    wall-clock around the run. No device timing; for machines without one.

It fills the ``performance.benchmark`` slot the ``port`` recipe declares
(opt in with ``benchmark: true``: it is a measurement of the candidate alone,
a ``StaticVerifier`` that needs no oracle -- the reference's time is the
reference's own run). It gates nothing: its confidence is ``SAMPLED``
whenever a measurement was taken and ``FAILED`` when none could be (no
profiler, no build spec, a crash), never a claim about correctness.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from recast.c.build import Toolchain, run
from recast.model import Candidate, Confidence, Unit, Verdict
from recast.plugins.executor import Executor, Job
from recast.plugins.verifier import StaticVerifier
from recast.verify.probes import stage_candidate

__all__ = ["GpuTimeVerifier", "factory", "parse_nsys_csv", "parse_nv_acc_time"]

_ACC_TOTAL = re.compile(r"device time\(us\):\s*total=([0-9,]+)")
_SECTION = re.compile(r"^\*\*\s*(.+?)\s*\((\w+)\):")
_EMPTY = {"kernel_ms": 0.0, "htod_ms": 0.0, "dtoh_ms": 0.0, "memset_ms": 0.0, "other_mem_ms": 0.0}


def parse_nsys_csv(text: str) -> dict[str, float]:
    """Sum the ``Total Time (ns)`` column of each report in ``nsys stats`` CSV output, in ms."""
    out = dict(_EMPTY)
    section: str | None = None
    rows: list[str] = []

    def flush() -> None:
        if section is None or not rows:
            return
        for row in csv.DictReader(io.StringIO("\n".join(rows))):
            try:
                ns = float((row.get("Total Time (ns)") or "0").replace(",", ""))
            except ValueError:
                continue
            ms = ns / 1e6
            if section == "cuda_gpu_kern_sum":
                out["kernel_ms"] += ms
            elif section == "cuda_gpu_mem_time_sum":
                op = (row.get("Operation") or "").lower()
                if "host-to-device" in op:
                    out["htod_ms"] += ms
                elif "device-to-host" in op:
                    out["dtoh_ms"] += ms
                elif "memset" in op:
                    out["memset_ms"] += ms
                else:
                    out["other_mem_ms"] += ms

    for line in text.splitlines():
        m = _SECTION.match(line.strip())
        if m:
            flush()
            section, rows = m.group(2), []
            continue
        if section is not None and line.strip():
            rows.append(line)
    flush()
    return out


def parse_nv_acc_time(text: str) -> float | None:
    """Total device time in ms from ``NV_ACC_TIME=1`` output, or None if absent."""
    total = 0
    found = False
    for m in _ACC_TOTAL.finditer(text):
        total += int(m.group(1).replace(",", ""))
        found = True
    return total / 1e3 if found else None


class GpuTimeVerifier(StaticVerifier):
    name = "performance.benchmark"
    provides = Confidence.SAMPLED

    def check(
        self,
        unit: Unit,
        candidate: Candidate,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        profiler = str(config.get("profiler", "nsys"))
        try:
            return self._measure(unit, candidate, workspace, executor, config, profiler)
        except Exception as error:  # a refusing executor: fail closed
            return self._verdict(
                candidate, Confidence.FAILED, {"profiler": profiler}, f"could not measure: {error}"
            )

    def _measure(
        self,
        unit: Unit,
        candidate: Candidate,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
        profiler: str,
    ) -> Verdict:
        root = Path(config["root"])
        toolchain = Toolchain.from_config(config)
        runs = int(config.get("runs", 3))
        spec, _staging, kernel_dir, _built = stage_candidate(
            unit, candidate, root, workspace, toolchain, executor
        )
        if not (kernel_dir / toolchain.render(spec.program)).exists():
            return self._verdict(
                candidate, Confidence.FAILED, {"profiler": profiler}, "candidate is not built"
            )
        args = tuple(config.get("run_args") or spec.args)

        samples: list[dict[str, float]] = []
        walls: list[float] = []
        for i in range(runs):
            label = f"{unit.uid}:bench{i + 1}"
            t0 = time.perf_counter()
            if profiler == "nsys":
                nsys = str(config.get("nsys", "nsys"))
                if shutil.which(nsys) is None:
                    return self._verdict(
                        candidate, Confidence.FAILED, {"profiler": profiler}, f"{nsys} not found"
                    )
                report = kernel_dir / f"bench{i + 1}"
                ran = run(
                    spec,
                    kernel_dir,
                    toolchain,
                    executor,
                    label,
                    args=args,
                    prefix=(
                        nsys,
                        "profile",
                        "--stats=false",
                        "--force-overwrite=true",
                        "-o",
                        str(report),
                    ),
                )
                walls.append(time.perf_counter() - t0)
                if ran.returncode != 0:
                    return self._verdict(
                        candidate,
                        Confidence.FAILED,
                        {"profiler": profiler, "run": i + 1, "returncode": ran.returncode},
                        f"profiled run {i + 1} exited {ran.returncode}: "
                        + (ran.stderr or ran.stdout)[-300:],
                    )
                rep = next(iter(sorted(kernel_dir.glob(f"bench{i + 1}.*rep"))), None)
                if rep is None:
                    return self._verdict(
                        candidate, Confidence.FAILED, {"profiler": profiler}, "nsys wrote no report"
                    )
                stats = executor.run(
                    Job(
                        argv=[
                            nsys,
                            "stats",
                            "--report",
                            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
                            "--format",
                            "csv",
                            str(rep),
                        ],
                        cwd=kernel_dir,
                        env=toolchain.env(),
                        timeout_s=toolchain.timeout_s,
                        label=f"{label}:stats",
                    )
                )
                (kernel_dir / f"bench{i + 1}.stats.csv").write_text(stats.stdout)
                samples.append(parse_nsys_csv(stats.stdout))
            elif profiler == "nv_acc_time":
                ran = run(
                    spec,
                    kernel_dir,
                    toolchain,
                    executor,
                    label,
                    args=args,
                    env={"NV_ACC_TIME": "1", "NVCOMPILER_ACC_TIME": "1"},
                )
                walls.append(time.perf_counter() - t0)
                if ran.returncode != 0:
                    return self._verdict(
                        candidate,
                        Confidence.FAILED,
                        {"profiler": profiler, "run": i + 1, "returncode": ran.returncode},
                        f"timed run {i + 1} exited {ran.returncode}",
                    )
                (kernel_dir / f"bench{i + 1}.acc.txt").write_text(ran.stdout + "\n" + ran.stderr)
                total = parse_nv_acc_time(ran.stdout + "\n" + ran.stderr)
                if total is None:
                    return self._verdict(
                        candidate,
                        Confidence.FAILED,
                        {"profiler": profiler, "run": i + 1},
                        "NV_ACC_TIME printed no device time (not an NVIDIA HPC offload build?)",
                    )
                samples.append({**_EMPTY, "device_ms": total})
            elif profiler == "wall":
                ran = run(spec, kernel_dir, toolchain, executor, label, args=args)
                walls.append(time.perf_counter() - t0)
                if ran.returncode != 0:
                    return self._verdict(
                        candidate,
                        Confidence.FAILED,
                        {"profiler": profiler, "run": i + 1, "returncode": ran.returncode},
                        f"run {i + 1} exited {ran.returncode}",
                    )
                samples.append({})
            else:
                return self._verdict(
                    candidate,
                    Confidence.FAILED,
                    {"profiler": profiler},
                    f"unknown profiler {profiler!r}",
                )

        metrics: dict[str, Any] = {
            "profiler": profiler,
            "runs": runs,
            "wall_s": [round(w, 4) for w in walls],
            "wall_s_mean": round(statistics.fmean(walls), 4) if walls else None,
        }
        if profiler == "wall":
            return self._verdict(
                candidate,
                Confidence.SAMPLED,
                metrics,
                f"wall time {metrics['wall_s_mean']:.3f} s over {runs} runs; "
                "no device timing on this host",
            )
        gpu = [s.get("device_ms", sum(s.get(k, 0.0) for k in _EMPTY)) for s in samples]
        metrics.update(
            {
                "gpu_ms": [round(g, 4) for g in gpu],
                "gpu_ms_mean": round(statistics.fmean(gpu), 4),
                "gpu_ms_stdev": round(statistics.pstdev(gpu), 4) if len(gpu) > 1 else 0.0,
                "kernel_ms_mean": round(
                    statistics.fmean(s.get("kernel_ms", 0.0) for s in samples), 4
                ),
                "htod_ms_mean": round(statistics.fmean(s.get("htod_ms", 0.0) for s in samples), 4),
                "dtoh_ms_mean": round(statistics.fmean(s.get("dtoh_ms", 0.0) for s in samples), 4),
            }
        )
        return self._verdict(
            candidate,
            Confidence.SAMPLED,
            metrics,
            f"GPU time {metrics['gpu_ms_mean']:.3f} ms (±{metrics['gpu_ms_stdev']:.3f}) "
            f"over {runs} runs, {profiler}",
        )

    def _verdict(
        self, candidate: Candidate, confidence: Confidence, metrics: dict[str, Any], detail: str
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics={"variant": candidate.notes.get("variant"), **metrics},
            detail=detail,
        )


def factory(**_config: Any) -> GpuTimeVerifier:
    return GpuTimeVerifier()
