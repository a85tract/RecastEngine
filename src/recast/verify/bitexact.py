"""``differential.bitexact``: the translation against the compiled truth.

Migrated from the comparison half of CESM-language-translator
``diff_driver.py`` and the tolerance ladder of its ``tests/test_diff.py``.
The candidate's emitted module and the oracle's compiled Fortran are called
side by side on the same generated inputs, and every output is compared bit
for bit. The ladder has two rungs and both are spelled in the Verdict:

* ``BIT_EXACT`` -- every compared value identical to the last bit. The
  strongest empirical claim there is, and the default acceptance bar.
* ``TOLERANCED`` -- everything agreed within an operator-stated ``rtol``.
  Only awarded when the operator *asked* for a tolerance; loosening the bar
  is a decision someone must make, never a default.

Anything else is ``FAILED``, including every way the comparison could not
run: the candidate does not import, the oracle handle is not a compiled
module, a subprogram raises. Fail closed -- a gate that cannot run did not
pass.

Inputs are generated deterministically per (subprogram, trial): shapes come
from the interface's dimensions resolved against the operator's ``dims``
table, values from per-name ``ranges``. The physical ranges that make CAM
kernels behave -- temperatures in kelvin, pressures in pascals -- are domain
knowledge and arrive in config; the engine's defaults are only wide, not
wise. Subprograms with deferred blocks are skipped and said so: their
translation raises ``NotImplementedError`` by construction, and the gate's
job is to judge translations, not queues.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from recast.model import Candidate, Confidence, OracleRef, Unit, Verdict
from recast.plugins.executor import Executor
from recast.plugins.verifier import Verifier
from recast.verify.ulp import ulp_audit

__all__ = ["BitexactVerifier", "factory"]

DEFAULT_RANGE = (-1000.0, 1000.0)
DEFAULT_INTEGER_RANGE = (1, 8)
DEFAULT_DIMENSION = 8


def _resolve_extent(text: str | None, dims: dict[str, int]) -> int:
    """A declared dimension's extent under the operator's table."""
    if text is None:
        return int(dims.get("default_dim", DEFAULT_DIMENSION))
    spelled = str(text).strip().lower()
    if spelled.isdigit():
        return int(spelled)
    resolved = spelled
    for name, value in dims.items():
        resolved = re.sub(rf"\b{re.escape(name.lower())}\b", str(value), resolved)
    try:
        return int(eval(resolved, {"__builtins__": {}}, {}))  # noqa: S307 -- digits and operators by now
    except Exception:
        return int(dims.get("default_dim", DEFAULT_DIMENSION))


class BitexactVerifier(Verifier):
    """Call both sides on the same inputs; count the bits that disagree."""

    name = "differential.bitexact"
    provides = Confidence.BIT_EXACT

    dominant_at: float | None = None
    """Fraction of a row's maximum above which an element is *dominant*.

    ``None`` here: bit-exactness has no use for the distinction, since every
    element has to agree. A gate that tolerates ULP drift does have one --
    see ``recast.verify.tolerance`` -- and sets it, which is what turns the
    per-element mask on in the comparison below.
    """

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        try:
            import numpy as np
        except ImportError:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                "numpy is not installed; install recast-engine[translate]",
            )

        handle = oracle.handle if isinstance(oracle.handle, dict) else {}
        truth = handle.get("module")
        wrappers = handle.get("wrappers", {})
        if truth is None:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                f"oracle {oracle.oracle!r} handed no compiled module to compare against",
            )

        try:
            translated = self._load_candidate(
                candidate, workspace, config.get("module_suffix", "_numpy.py")
            )
        except Exception as error:  # fail closed, whatever broke
            return self._verdict(
                candidate, Confidence.FAILED, {}, f"candidate does not import: {error}"
            )

        table = getattr(translated, "_SIGNATURES", None)
        if not isinstance(table, dict) or not table:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                "the emitted module carries no _SIGNATURES table to generate "
                "inputs from; the transform embeds one for exactly this harness",
            )

        deferred_subprograms = {entry.split("/", 1)[0] for entry in candidate.deferred}

        def generable(name: str) -> bool:
            """Whether this harness can produce every required input.

            Character arguments have no sampling story yet; a default that
            tried would fail the whole gate on an init routine's errstring.
            Explicit config still wins -- and then fails loudly.
            """
            return not any(
                a["dtype"] == "str" and not a.get("optional")
                for a in table[name]["args"]
                if a["intent"] != "OUT"
            )

        wanted = config.get("subprograms") or [
            name
            for name in wrappers
            if name in table and name not in deferred_subprograms and generable(name)
        ]
        skipped = sorted(set(wrappers) - set(wanted))

        trials = int(config.get("trials", 10))
        dims = dict(config.get("dims", {}))
        ranges = {str(k).lower(): tuple(v) for k, v in (config.get("ranges") or {}).items()}

        # Module state first: the emitted header says "call <init> before
        # use", and the Fortran side's SAVE variables need the same call with
        # the same constants, or the two sides are computing under different
        # physics and every mismatch is noise.
        try:
            self._run_setup(config.get("setup") or [], translated, truth, wrappers)
        except Exception as error:  # fail closed
            return self._verdict(candidate, Confidence.FAILED, {}, f"setup call failed: {error}")

        per_subprogram: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        worst_rel = 0.0
        totals = {"points": 0, "bit_exact": 0, "max_ulp": 0, "nan_mismatch": 0}
        for name in wanted:
            sub = table[name]
            translated_fn = getattr(translated, name, None)
            truth_fn = getattr(truth, wrappers.get(name, f"w_{name}"), None)
            if translated_fn is None or truth_fn is None:
                side = "candidate" if translated_fn is None else "oracle"
                failures.append(f"{name}: missing on the {side} side")
                continue
            outcome = self._compare_subprogram(
                np,
                name,
                sub,
                translated_fn,
                truth_fn,
                trials,
                dims,
                ranges,
                prepare=getattr(translated, "_PREPARE_INPUTS", None),
                dominant_at=config.get("dominant_at", self.dominant_at),
            )
            per_subprogram[name] = outcome
            if "error" in outcome:
                failures.append(f"{name}: {outcome['error']}")
                continue
            totals["points"] += outcome["points"]
            totals["bit_exact"] += outcome["bit_exact"]
            totals["max_ulp"] = max(totals["max_ulp"], outcome["max_ulp"])
            totals["nan_mismatch"] += outcome["nan_mismatch"]
            worst_rel = max(worst_rel, outcome["max_rel"])
            if "max_ulp_dominant" in outcome:
                totals["max_ulp_dominant"] = max(
                    totals.get("max_ulp_dominant", 0), outcome["max_ulp_dominant"]
                )
                totals["dominant_points"] = (
                    totals.get("dominant_points", 0) + outcome["dominant_points"]
                )

        metrics = {
            "subprograms": per_subprogram,
            "trials": trials,
            "skipped": skipped,
            "max_rel": worst_rel,
            **totals,
            **self._devices(translated, handle),
        }
        if failures:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(failures)} subprogram(s) could not be compared: " + "; ".join(failures[:3]),
            )
        if not per_subprogram:
            return self._verdict(
                candidate, Confidence.FAILED, metrics, "nothing was compared; that is not a pass"
            )
        if totals["nan_mismatch"]:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{totals['nan_mismatch']} point(s) where one side produced NaN "
                "and the other a number",
            )
        return self._award(candidate, totals, per_subprogram, metrics, config)

    def _award(
        self,
        candidate: Candidate,
        totals: dict[str, Any],
        per_subprogram: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> Verdict:
        """Which confidence the numbers earn.

        Everything above this point -- generating inputs, calling both sides,
        counting ULP -- is the same comparison whatever the gate. What differs
        between gates is only the policy, so that is the part a subclass
        overrides. Splitting it here is what lets a second differential gate
        exist without a second harness to keep in step with this one.
        """
        rtol = config.get("rtol")
        worst_rel = metrics["max_rel"]
        if totals["bit_exact"] == totals["points"]:
            return self._verdict(
                candidate,
                Confidence.BIT_EXACT,
                metrics,
                f"{totals['points']} points across {len(per_subprogram)} "
                f"subprogram(s), all bit-exact",
            )
        if rtol is not None and worst_rel <= float(rtol):
            return self._verdict(
                candidate,
                Confidence.TOLERANCED,
                metrics,
                f"{totals['bit_exact']}/{totals['points']} bit-exact, "
                f"max_rel={worst_rel:.3e} within rtol={rtol}",
            )
        return self._verdict(
            candidate,
            Confidence.FAILED,
            metrics,
            f"{totals['points'] - totals['bit_exact']}/{totals['points']} points differ "
            f"(max {totals['max_ulp']} ULP, max_rel={worst_rel:.3e}) and no rtol excuses them",
        )

    @staticmethod
    def _run_setup(
        setup: list[dict[str, Any]], translated: Any, truth: Any, wrappers: dict[str, str]
    ) -> None:
        from recast.transform.numpy.vocabulary import pysafe

        for call in setup:
            name = call["subprogram"]
            inputs = call.get("inputs", {})
            getattr(translated, pysafe(name))(**{pysafe(k): v for k, v in inputs.items()})
            getattr(truth, wrappers.get(name, f"w_{name}"))(**inputs)

    # -- one subprogram -------------------------------------------------------

    def _compare_subprogram(
        self,
        np: Any,
        name: str,
        sub: dict[str, Any],
        translated_fn: Any,
        truth_fn: Any,
        trials: int,
        dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        prepare: Any = None,
        dominant_at: float | None = None,
    ) -> dict[str, Any]:
        from recast.transform.numpy.vocabulary import pysafe

        required = [a for a in sub["args"] if not a.get("optional")]
        outs_all = [a for a in sub["args"] if a["intent"] in ("OUT", "INOUT")]
        outs_required = [a for a in outs_all if not a.get("optional")]

        # A scalar that names another argument's extent is not free data: it
        # must equal the extent the arrays are generated with, or every call
        # is a shape error rather than a comparison.
        dimension_names = {
            token
            for argument in sub["args"]
            for dim in argument.get("dims") or []
            for token in re.findall(r"[a-z_]\w*", str(dim.get("ub") or "").lower())
        }

        points = bit_exact = nan_mismatch = 0
        max_ulp = 0
        max_ulp_dominant = 0
        dominant_points = 0
        max_rel = 0.0
        for trial in range(trials):
            # hash() is salted per process; a seed must not be.
            rng = np.random.default_rng(int.from_bytes(f"{name}:{trial}".encode(), "big") % 2**32)
            inputs = {}
            for argument in required:
                if argument["intent"] == "OUT":
                    continue
                lowered = argument["name"].lower()
                if not argument.get("dims") and (lowered in dimension_names or lowered in dims):
                    inputs[argument["name"]] = np.int32(_resolve_extent(lowered, dims))
                else:
                    inputs[argument["name"]] = self._value(np, argument, dims, ranges, rng)

            if prepare is not None:
                # The candidate may carry a ``_PREPARE_INPUTS(name, inputs,
                # rng)`` hook, the way it carries ``_SIGNATURES``: per-name
                # ranges cannot express structure -- a pressure column must
                # be monotone, an interface field must bracket its levels --
                # and unphysical inputs drive both sides into error paths
                # the production model aborts out of. The hook shapes inputs
                # into the defined domain; it cannot bias the verdict,
                # because both sides receive the same shaped inputs.
                prepare(name, inputs, rng)

            # Keyword calls on both sides: f2py reorders inferred-dimension
            # scalars into trailing keywords, so positional order is not a
            # shared vocabulary -- names are.
            translated_kwargs = {
                pysafe(a["name"]): inputs[a["name"]] for a in required if a["intent"] != "OUT"
            }
            # f2py lowercases every dummy name, because Fortran is
            # case-insensitive and the source's spelling is not a fact about
            # the interface. A candidate that reports `sl_prePBL` still
            # reaches the same oracle argument.
            truth_kwargs = {
                a["name"].lower(): (
                    np.copy(v) if isinstance(v := inputs[a["name"]], np.ndarray) else v
                )
                for a in required
                if a["intent"] != "OUT"
            }
            truth_args = [truth_kwargs[a["name"].lower()] for a in required if a["intent"] != "OUT"]
            try:
                translated_out = translated_fn(**translated_kwargs)
            except Exception as error:
                return {"error": f"candidate raised: {type(error).__name__}: {error}"}
            try:
                truth_out = truth_fn(**truth_kwargs)
            except Exception as error:
                return {"error": f"oracle raised: {type(error).__name__}: {error}"}

            pairs = self._paired_outputs(
                sub, outs_all, outs_required, required, translated_out, truth_out, truth_args
            )
            if isinstance(pairs, str):
                return {"error": pairs}
            for label, ours, theirs in pairs:
                shaped_ours = np.asarray(ours, dtype=np.float64)
                shaped_theirs = np.asarray(theirs, dtype=np.float64)
                if shaped_ours.shape != shaped_theirs.shape:
                    return {"error": f"{label}: shape {shaped_ours.shape} vs {shaped_theirs.shape}"}
                a = shaped_ours.ravel()
                b = shaped_theirs.ravel()
                audit = ulp_audit(
                    a.tolist(),
                    b.tolist(),
                    dominant=self._dominance(np, shaped_theirs, dominant_at),
                )
                points += audit["total_points"]
                bit_exact += audit["bit_exact"]
                nan_mismatch += audit["nan_mismatch"]
                max_ulp = max(max_ulp, audit["max_ulp"])
                if "max_ulp_dominant" in audit:
                    max_ulp_dominant = max(max_ulp_dominant, audit["max_ulp_dominant"])
                    dominant_points += audit["dominant_points"]
                if audit["bit_exact"] != audit["total_points"]:
                    scale = np.maximum(np.abs(b), 1e-300)
                    with np.errstate(invalid="ignore"):
                        rel = float(np.nanmax(np.abs(a - b) / scale))
                    max_rel = max(max_rel, rel)
        outcome = {
            "points": points,
            "bit_exact": bit_exact,
            "max_ulp": max_ulp,
            "max_rel": max_rel,
            "nan_mismatch": nan_mismatch,
        }
        if dominant_at is not None:
            outcome["max_ulp_dominant"] = max_ulp_dominant
            outcome["dominant_points"] = dominant_points
        return outcome

    @staticmethod
    def _devices(translated: Any, handle: dict[str, Any]) -> dict[str, str]:
        """Which device each side ran on, when either side says.

        Every rung of the ladder is a claim about an environment rather than
        about the code -- the same candidate and the same oracle can agree to
        the bit on one machine and differ on the next -- and for an accelerator
        backend the device is the half of that environment most likely to move.
        A verdict that does not record it cannot be re-argued later.

        Asked for rather than detected, and by the same convention as
        ``_SIGNATURES`` and ``_PREPARE_INPUTS``: the emitted module declares
        ``_DEVICE`` if it knows, and an Oracle puts one on its handle. Reaching
        for ``jax.devices()`` here instead would put an accelerator import in
        the core, which is the one thing the core does not do.
        """
        found = {
            "candidate_device": getattr(translated, "_DEVICE", None),
            "reference_device": handle.get("device"),
        }
        return {name: str(value) for name, value in found.items() if value}

    @staticmethod
    def _dominance(np: Any, reference: Any, dominant_at: float | None) -> list[bool] | None:
        """Which elements a ULP bound is allowed to be held to.

        ``|v| >= fraction * the maximum along the last axis``, so an element is
        judged against its own row: a column of small values is not excused by
        a large value somewhere else in the array. The *reference* side decides,
        because whether an element matters is a fact about what it should have
        been, not about the candidate being judged.
        """
        if dominant_at is None:
            return None
        magnitude = np.abs(reference)
        scale = magnitude.max(axis=-1, keepdims=True) if magnitude.ndim > 1 else magnitude.max()
        mask: list[bool] = (magnitude >= dominant_at * scale).ravel().tolist()
        return mask

    @staticmethod
    def _paired_outputs(
        sub: dict[str, Any],
        outs_all: list[dict[str, Any]],
        outs_required: list[dict[str, Any]],
        required: list[dict[str, Any]],
        translated_out: Any,
        truth_out: Any,
        truth_args: list[Any],
    ) -> list[tuple[str, Any, Any]] | str:
        """Match the two sides' outputs by argument.

        The translation returns every OUT/INOUT argument, optional ones
        included, in declaration order (a function returns its result). f2py
        returns the wrapper's ``intent(out)`` arguments and mutates the
        ``inout`` ones in place, so INOUT values are read back from the
        arrays that were passed.
        """
        if sub["kind"] == "function":
            return [(sub.get("result") or "result", translated_out, truth_out)]

        ours = list(translated_out) if isinstance(translated_out, tuple) else [translated_out]
        if len(ours) != len(outs_all):
            return (
                f"candidate returned {len(ours)} value(s) for "
                f"{len(outs_all)} out-intent argument(s)"
            )
        by_name = dict(zip([a["name"] for a in outs_all], ours, strict=True))

        theirs_out = (
            list(truth_out)
            if isinstance(truth_out, tuple)
            else ([truth_out] if truth_out is not None else [])
        )
        pure_out = [a for a in outs_required if a["intent"] == "OUT"]
        if len(theirs_out) != len(pure_out):
            return (
                f"oracle returned {len(theirs_out)} value(s) for "
                f"{len(pure_out)} intent(out) argument(s)"
            )
        theirs = dict(zip([a["name"] for a in pure_out], theirs_out, strict=True))
        passed_in = [a["name"] for a in required if a["intent"] != "OUT"]
        for argument in outs_required:
            if argument["intent"] == "INOUT":
                theirs[argument["name"]] = truth_args[passed_in.index(argument["name"])]

        return [(a["name"], by_name[a["name"]], theirs[a["name"]]) for a in outs_required]

    def _value(
        self,
        np: Any,
        argument: dict[str, Any],
        dims: dict[str, int],
        ranges: dict[str, tuple[float, float]],
        rng: Any,
    ) -> Any:
        name = argument["name"].lower()
        dtype = {
            "float64": np.float64,
            "float32": np.float32,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
        }.get(argument["dtype"], np.float64)
        shape = None
        if argument.get("dims"):
            shape = tuple(_resolve_extent(d.get("ub"), dims) for d in argument["dims"])
        if dtype in (np.float64, np.float32):
            low, high = ranges.get(name, DEFAULT_RANGE)
            if shape is None:
                return dtype(rng.uniform(low, high))
            return np.asfortranarray(rng.uniform(low, high, size=shape).astype(dtype))
        if dtype in (np.int32, np.int64):
            low, high = ranges.get(name, DEFAULT_INTEGER_RANGE)
            if shape is None:
                return dtype(rng.integers(int(low), int(high) + 1))
            return np.asfortranarray(
                rng.integers(int(low), int(high) + 1, size=shape).astype(dtype)
            )
        if shape is None:
            return np.bool_(rng.integers(0, 2))
        return np.asfortranarray(rng.integers(0, 2, size=shape).astype(np.bool_))

    # -- loading --------------------------------------------------------------

    @staticmethod
    def _load_candidate(candidate: Candidate, workspace: Path, suffix: str = "_numpy.py") -> Any:
        """Write the candidate's files and import its generated module.

        The candidate is self-contained by design -- module, constants,
        use-constants -- so importing it needs nothing but its own files on
        the path. Companion modules, when a scheme has them, must already be
        importable; supplying them is the run's business, not this gate's.

        ``suffix`` picks which of those files is the one under judgement. A
        port carries more than one: the JAX module, and the NumPy module it
        host-delegates to and imports. Both are staged, only one is imported
        as the candidate, and the recipe says which by setting
        ``config["module_suffix"]``. Defaulting to the NumPy module keeps
        every existing config meaning what it did.
        """
        staged = workspace / "candidate"
        staged.mkdir(parents=True, exist_ok=True)
        module_path = None
        for path, content in candidate.files.items():
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            if str(path).endswith(suffix):
                module_path = target
        if module_path is None:
            raise FileNotFoundError(f"candidate carries no *{suffix} module")

        sys.path.insert(0, str(staged))
        try:
            for name in list(sys.modules):
                if name == module_path.stem or name.endswith("_constants"):
                    del sys.modules[name]
            spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path.stem] = module
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(staged))
        return module

    def _verdict(
        self, candidate: Candidate, confidence: Confidence, metrics: dict[str, Any], detail: str
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics=metrics,
            detail=detail,
        )


def factory(**_config: Any) -> BitexactVerifier:
    return BitexactVerifier()
