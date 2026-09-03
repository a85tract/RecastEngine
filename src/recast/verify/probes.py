"""``differential.probes``: two programs, the same probes, compared.

Both sides are built from source in the workspace with the operator's
toolchain: the candidate from its files under the build spec in
``attrs["build"]``, and the reference from the ``executable-golden`` oracle's
staging with the candidate's probes carried onto it (``probe_inject``). The
candidate runs ``runs`` times (5 in the paper this comes from), the reference
once, and ``probe_protocol.compare`` decides.

Confidence: ``TOLERANCED`` when probe statistics agreed within the stated
tolerance (checksums are exact by definition, but the bar is the tolerance);
``SAMPLED`` when the candidate carries no probes and only the stdout
comparison could run; ``FAILED`` for everything else, including every way
the comparison could not run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from recast.c.build import (
    Spec,
    Toolchain,
    build,
    run,
    stage_directory,
    stage_siblings,
)
from recast.model import Candidate, Confidence, OracleRef, Unit, Verdict
from recast.plugins.executor import Executor, JobResult
from recast.plugins.verifier import Verifier
from recast.verify import probe_protocol as protocol
from recast.verify.probe_inject import ProbeInjectionError, extract_probes, has_probes, inject

__all__ = ["ProbeVerifier", "factory", "stage_candidate"]

STAGING_SCHEMA = "1"
"""Bumped when what gets staged beside a candidate changes, so a cached build
from an earlier layout is redone rather than trusted."""


def stage_candidate(
    unit: Unit,
    candidate: Candidate,
    root: Path,
    workspace: Path,
    toolchain: Toolchain,
    executor: Executor,
) -> tuple[Spec, Path, Path, JobResult | None]:
    """The candidate's files under ``workspace/candidate/<digest>/``, built.

    Returns ``(spec, staging_root, kernel_dir, build_result)``; the result is
    None when a build from this run is already there, which is how a second
    stage (the benchmark) finds it by digest.
    """
    spec = Spec.from_attrs(unit.attrs["build"])
    staging = workspace / "candidate" / candidate.digest()[:12]
    kernel_dir = staging / spec.dir
    marker = kernel_dir / ".built"
    stamp = f"{candidate.digest()}:{STAGING_SCHEMA}:{toolchain.identity()}"
    if marker.exists() and marker.read_text() == stamp:
        return spec, staging, kernel_dir, None
    if staging.exists():
        shutil.rmtree(staging)
    for rel, content in candidate.files.items():
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    for extra in unit.attrs["build"].get("stage", ()):
        item = root / str(extra)
        if item.is_dir():
            stage_directory(item, staging / item.relative_to(root))
    stage_siblings(root / spec.dir, kernel_dir)
    for extra in unit.attrs["build"].get("inputs", ()):
        item = root / str(extra)
        if item.is_symlink() and not item.exists():
            continue
        target = kernel_dir / item.name
        if item.is_dir() and not target.exists():
            stage_directory(item, target)
        elif item.is_file() and not target.exists():
            target.write_bytes(item.read_bytes())
    result = build(spec, kernel_dir, staging, toolchain, executor, label=f"{unit.uid}:candidate")
    (kernel_dir / "build.log").write_text(result.stdout + result.stderr)
    if result.ok and (kernel_dir / toolchain.render(spec.program)).exists():
        marker.write_text(stamp)
    return spec, staging, kernel_dir, result


def _counterpart(reference_dir: Path, candidate_rel: str, golden: Spec) -> Path | None:
    """The reference file a candidate file corresponds to: the one with the
    same name, else the reference's first source."""
    name = Path(candidate_rel).name
    hits = [p for p in reference_dir.rglob(name) if p.is_file()]
    if len(hits) == 1:
        return hits[0]
    return reference_dir / golden.sources[0] if golden.sources else None


class ProbeVerifier(Verifier):
    name = "differential.probes"
    provides = Confidence.TOLERANCED

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        if not isinstance(oracle.handle, dict) or "dir" not in oracle.handle:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "no_oracle"},
                "oracle handle carries no staged reference",
            )
        root = Path(config["root"])
        toolchain = Toolchain.from_config(config)
        runs = int(config.get("runs", 5))
        rtol = float(config.get("rtol", protocol.DEFAULT_RTOL))
        atol = float(config.get("atol", protocol.DEFAULT_ATOL))
        try:
            return self._verify(
                unit,
                candidate,
                oracle,
                workspace,
                executor,
                config,
                root,
                toolchain,
                runs,
                rtol,
                atol,
            )
        except Exception as error:  # a refusing executor, an unreadable staging: fail closed
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "could_not_run"},
                f"comparison could not run: {error}",
            )

    def _verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
        root: Path,
        toolchain: Toolchain,
        runs: int,
        rtol: float,
        atol: float,
    ) -> Verdict:
        spec, _staging, kernel_dir, built = stage_candidate(
            unit, candidate, root, workspace, toolchain, executor
        )
        program = toolchain.render(spec.program)
        if built is not None and (not built.ok or not (kernel_dir / program).exists()):
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "cand_build_failed", "returncode": built.returncode},
                "candidate did not build: " + (built.stderr or built.stdout)[-400:],
            )
        # The candidate runs with its own spec's arguments unless the operator
        # overrides them; the reference's may differ (it can take more).
        run_args = tuple(config.get("run_args") or spec.args)
        outputs: list[protocol.ProbeOutput] = []
        stdouts: list[str] = []
        for i in range(runs):
            ran = run(
                spec, kernel_dir, toolchain, executor, f"{unit.uid}:cand-run{i + 1}", args=run_args
            )
            if ran.returncode != 0:
                return self._verdict(
                    candidate,
                    Confidence.FAILED,
                    {"gate": "cand_nonzero_exit", "returncode": ran.returncode, "run": i + 1},
                    f"candidate exited {ran.returncode} on run {i + 1}: "
                    + (ran.stderr or ran.stdout)[-400:],
                )
            stdouts.append(ran.stdout)
            outputs.append(protocol.parse(ran.stdout))
        (kernel_dir / "run.stdout").write_text(stdouts[-1])

        candidate_sources = {
            rel: candidate.files[spec.dir / rel].decode(errors="replace")
            for rel in candidate.notes.get("candidate_files", spec.sources)
            if spec.dir / rel in candidate.files
        }
        if outputs and outputs[0].has_probes:
            return self._with_probes(
                unit,
                candidate,
                oracle,
                workspace,
                executor,
                toolchain,
                run_args,
                candidate_sources,
                outputs,
                rtol,
                atol,
            )
        if not config.get("fallback_stdout", True):
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "no_cand_gate_output"},
                "candidate printed no probes and the stdout fallback is off",
            )
        agree, how = protocol.stdout_agree(str(oracle.handle.get("stdout", "")), stdouts[0])
        deterministic = all(s == stdouts[0] for s in stdouts) or all(
            protocol.is_success_output(s) for s in stdouts
        )
        metrics = {"gate": "stdout", "stdout": how, "runs": runs, "deterministic": deterministic}
        if agree and deterministic:
            return self._verdict(
                candidate,
                Confidence.SAMPLED,
                metrics,
                f"no probes; stdout comparison {how} over {runs} runs",
            )
        return self._verdict(candidate, Confidence.FAILED, metrics, f"stdout comparison: {how}")

    def _with_probes(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        toolchain: Toolchain,
        run_args: tuple[str, ...],
        candidate_sources: dict[str, str],
        outputs: list[protocol.ProbeOutput],
        rtol: float,
        atol: float,
    ) -> Verdict:
        golden_spec = Spec.from_attrs(unit.attrs["golden"])
        staging = workspace / "gate" / candidate.digest()[:12]
        reference_dir = staging / "golden" / Path(str(oracle.handle["dir"])).name
        stage_directory(Path(str(oracle.handle["dir"])), reference_dir)
        gate_sdk = Path(str(oracle.handle["staging"])) / "gate_sdk"
        if gate_sdk.is_dir():
            shutil.copytree(gate_sdk, staging / "gate_sdk", dirs_exist_ok=True)
        stage_siblings(Path(str(oracle.handle["dir"])), reference_dir)

        probes = "published"
        injected: list[str] = []
        for rel, source in candidate_sources.items():
            _include, blocks = extract_probes(source)
            if not blocks:
                continue
            target = _counterpart(reference_dir, rel, golden_spec)
            if target is None or not target.exists():
                return self._verdict(
                    candidate,
                    Confidence.FAILED,
                    {"gate": "no_reference_counterpart", "file": rel},
                    f"no reference file corresponds to {rel}",
                )
            reference_source = target.read_text(errors="replace")
            if has_probes(reference_source):
                continue
            try:
                target.write_text(inject(reference_source, source))
            except ProbeInjectionError as error:
                return self._verdict(
                    candidate,
                    Confidence.FAILED,
                    {"gate": "probe_injection_refused", "file": rel},
                    f"could not carry the candidate's probes from {rel} to the reference: {error}",
                )
            injected.append(rel)
        if injected:
            probes = "injected"

        built = build(
            golden_spec,
            reference_dir,
            staging,
            toolchain,
            executor,
            label=f"{unit.uid}:golden-probed",
        )
        (reference_dir / "build.log").write_text(built.stdout + built.stderr)
        program = toolchain.render(golden_spec.program)
        if not built.ok or not (reference_dir / program).exists():
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "probe_build_failed", "probes": probes, "returncode": built.returncode},
                "reference with the candidate's probes did not build: "
                + (built.stderr or built.stdout)[-400:],
            )
        golden_args = tuple(oracle.handle.get("run_args") or golden_spec.args)
        if run_args != tuple(oracle.handle.get("run_args") or ()):
            # The candidate ran with other arguments; the reference follows,
            # keeping whatever it takes beyond them.
            golden_args = run_args + golden_args[len(run_args) :]
        ran = run(
            golden_spec,
            reference_dir,
            toolchain,
            executor,
            f"{unit.uid}:golden-probed-run",
            args=golden_args,
        )
        (reference_dir / "run.stdout").write_text(ran.stdout)
        if ran.returncode != 0:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"gate": "ref_nonzero_exit", "returncode": ran.returncode},
                f"instrumented reference exited {ran.returncode}",
            )
        result = protocol.compare(protocol.parse(ran.stdout), outputs, rtol=rtol, atol=atol)
        metrics = {
            **result.metrics(),
            "rtol": rtol,
            "atol": atol,
            "probes": probes,
            "probed_files": injected,
        }
        if result.passed:
            return self._verdict(
                candidate,
                Confidence.TOLERANCED,
                metrics,
                f"{result.checksums_compared} checksum(s) identical, "
                f"{result.stats_compared} statistic(s) within rtol={rtol}/atol={atol} "
                f"(max rel err {result.max_rel_err:.2e}), {result.runs} runs deterministic",
            )
        return self._verdict(
            candidate,
            Confidence.FAILED,
            metrics,
            f"[{result.reason}] " + "; ".join(result.failures[:5]),
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


def factory(**_config: Any) -> ProbeVerifier:
    return ProbeVerifier()
