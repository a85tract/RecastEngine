"""``differential.bitexact``: the translation against the compiled truth.

Migrated from the comparison half of the source pipeline
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
table, values from per-name ``ranges``. The physical ranges that make a model
kernels behave -- temperatures in kelvin, pressures in pascals -- are domain
knowledge and arrive in config; the engine's defaults are only wide, not
wise. Subprograms with deferred blocks are skipped and said so: their
translation raises ``NotImplementedError`` by construction, and the gate's
job is to judge translations, not queues.
"""

from __future__ import annotations

import ast
import importlib.util
import operator
import re
import sys
from collections.abc import Callable
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
SUPPORTED_DTYPES = frozenset({"float32", "float64", "int32", "int64", "bool"})


def _extent(dim: dict[str, Any], dims: dict[str, int]) -> int:
    """An axis's extent: ``ub - lb + 1`` when a lower bound is declared
    (CLUBB's ``lhs(-2:2, ...)`` has five rows, not two), ``ub`` otherwise."""
    upper = _resolve_extent(dim.get("ub"), dims)
    lower = str(dim.get("lb") or "1").strip()
    if lower == "1" or dim.get("ub") is None:
        return upper
    return upper - _resolve_extent(lower, dims) + 1


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
        return int(_arithmetic(resolved))
    except Exception:
        return int(dims.get("default_dim", DEFAULT_DIMENSION))


# Split by arity rather than kept in one table. A single dict of both is a
# dict whose value type is the join of a two-argument and a one-argument
# callable, which is a type nothing can call -- the checker says so, and the
# result was an ``Any`` leaking out of a function declared to return ``float``.
_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _arithmetic(text: str) -> float:
    """Integer arithmetic over the operators a Fortran extent can use.

    This was ``eval`` with empty builtins, on text that came from a declared
    dimension in the source under verification -- and the source under
    verification is the input this engine exists to take from other people.
    An empty ``__builtins__`` does not make ``eval`` safe, and a dimension
    expression has no business reaching anything but arithmetic. Anything
    that is not a number or one of the operators above is a ``ValueError``,
    which the caller turns into the default extent exactly as before.
    """

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError(f"not arithmetic: {ast.dump(node)}")

    return walk(ast.parse(text, mode="eval"))


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
        verdict = self._compare_all(unit, candidate, oracle, workspace, executor, config)
        # An oracle that could not spell every subprogram lists the rest on
        # its handle; a module that passes with three of its eleven
        # subprograms compared must say so where the evidence is read. This
        # neither weakens the verdict for what was compared nor strengthens
        # it for what was not.
        handle = oracle.handle if isinstance(oracle.handle, dict) else {}
        # ... and so must an operator who declared one ungated in config; the
        # comparison put those it found in the table on the metrics.
        ungated = {
            **dict(handle.get("ungated") or {}),
            **dict(verdict.metrics.get("ungated") or {}),
        }
        if not ungated:
            return verdict
        return Verdict(
            unit=verdict.unit,
            candidate=verdict.candidate,
            verifier=verdict.verifier,
            confidence=verdict.confidence,
            metrics={**verdict.metrics, "ungated": ungated},
            detail=f"{verdict.detail}; {len(ungated)} subprogram(s) ungated, no reference: "
            + ", ".join(f"{name} ({why})" for name, why in sorted(ungated.items())),
        )

    def _compare_all(
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
        # Which direction this comparison runs. Every reference that *computes*
        # answers inputs the harness chose, so the harness generates them. A
        # reference that only *replays* cannot be asked anything it was not
        # already asked -- the inputs are whatever the recorded run used -- so
        # it supplies them, and says so here rather than being detected.
        recorded = handle.get("input_source") == "recorded"
        samples = list(handle.get("samples") or []) if recorded else []
        if recorded and not samples:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {},
                f"oracle {oracle.oracle!r} supplies the inputs and handed over no samples",
            )
        if truth is None and not recorded:
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
        if recorded:
            # A recording is text and parses as float64 throughout; the
            # signature says which of its inputs are integers.
            self._type_recorded(np, samples, table)

        def judged(name: str) -> bool:
            """Not deferred -- and not a flat adapter around a subprogram
            that is, since the adapter would call into a NotImplementedError
            and the skip has to map the adapter's name back to its own."""
            if name in deferred_subprograms:
                return False
            return not (name.endswith("_flat") and name[: -len("_flat")] in deferred_subprograms)

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

        if recorded:
            # A recording names what it is a recording of, so the set to
            # compare is the set that was captured -- not every subprogram the
            # module exports. ``generable`` does not apply: nothing is
            # generated, and a character argument that was recorded can be
            # replayed.
            by_subprogram: dict[str, list[dict[str, Any]]] = {}
            for sample in samples:
                by_subprogram.setdefault(str(sample.get("subprogram", "")), []).append(sample)
            offered = sorted(by_subprogram)
            wanted = config.get("subprograms") or [
                name for name in offered if name in table and judged(name)
            ]
            skipped = sorted(set(offered) - set(wanted))
        else:
            by_subprogram = {}
            wanted = config.get("subprograms") or [
                name for name in wrappers if name in table and judged(name) and generable(name)
            ]
            skipped = sorted(set(wrappers) - set(wanted))

        trials = int(config.get("trials", 10))
        # The transform may have read the tree for the value of every name
        # that sizes a dummy array (``Candidate.notes["dims"]``); the
        # operator's table wins where both speak.
        dims = {**(candidate.notes.get("dims") or {}), **dict(config.get("dims", {}))}
        ranges = {str(k).lower(): tuple(v) for k, v in (config.get("ranges") or {}).items()}

        # Module state first: the emitted header says "call <init> before
        # use", and the Fortran side's SAVE variables need the same call with
        # the same constants, or the two sides are computing under different
        # physics and every mismatch is noise.
        try:
            self._run_setup(
                config.get("setup") or [],
                translated,
                truth,
                wrappers,
                str(handle.get("arg_naming", "lower")),
            )
        except Exception as error:  # fail closed
            return self._verdict(candidate, Confidence.FAILED, {}, f"setup call failed: {error}")

        per_subprogram: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        worst_rel = 0.0
        totals = {
            "points": 0,
            "bit_exact": 0,
            "max_ulp": 0,
            "nan_mismatch": 0,
            "integer_points": 0,
            "integer_mismatch": 0,
        }
        for name in wanted:
            sub = table[name]
            translated_fn = getattr(translated, name, None)
            truth_fn = None if recorded else getattr(truth, wrappers.get(name, f"w_{name}"), None)
            if translated_fn is None or (truth_fn is None and not recorded):
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
                dominant_axis=config.get("dominant_axis", -1),
                rel_scale=str(config.get("rel_scale", "element")),
                arg_naming=str(handle.get("arg_naming", "lower")),
                convention=str(handle.get("return_convention", "f2py")),
                samples=by_subprogram.get(name) if recorded else None,
            )
            per_subprogram[name] = outcome
            if "error" in outcome:
                failures.append(f"{name}: {outcome['error']}")
                continue
            totals["points"] += outcome["points"]
            totals["bit_exact"] += outcome["bit_exact"]
            totals["max_ulp"] = max(totals["max_ulp"], outcome["max_ulp"])
            totals["nan_mismatch"] += outcome["nan_mismatch"]
            totals["integer_points"] += outcome["integer_points"]
            totals["integer_mismatch"] += outcome["integer_mismatch"]
            worst_rel = max(worst_rel, outcome["max_rel"])
            if "max_ulp_dominant" in outcome:
                totals["max_ulp_dominant"] = max(
                    totals.get("max_ulp_dominant", 0), outcome["max_ulp_dominant"]
                )
                totals["dominant_points"] = (
                    totals.get("dominant_points", 0) + outcome["dominant_points"]
                )

        # Policy gate: a public subprogram the module declares (its
        # _SIGNATURES), that is not deferred, and that no comparison attempt
        # covered, is a translation claim with no evidence. Every silent-
        # narrowing filter -- oracle-side wrapper drops, generability skips,
        # config subsets -- lands here by construction, because coverage is
        # judged against what was TRANSLATED, not against whatever survived
        # the filters. Three things are not silence: a private subprogram,
        # which no wrapper can reach and every public caller exercises; a
        # subprogram compared through its ``<name>_flat`` adapter, which calls
        # it on both sides; and one the oracle listed as ungated, whose
        # reason ``verify`` carries onto the verdict; and one the operator
        # declared ungated in config, ``{name: why}``, for a reference this
        # oracle cannot hold -- a routine a recording never reached because it
        # is compared inside another unit's replay, say. The reason goes on
        # the verdict with the oracle's, and a name not in this unit's table
        # is not this unit's to report.
        declared = {
            name: str(why) for name, why in (config.get("ungated") or {}).items() if name in table
        }
        compared = set(wanted)
        ungated = set(handle.get("ungated") or {}) | set(declared)
        uncovered = sorted(
            name
            for name, sub in table.items()
            if sub.get("public", True)
            and judged(name)
            and name not in compared
            and f"{name}_flat" not in compared
            and name not in ungated
        )
        metrics = {
            "subprograms": per_subprogram,
            "trials": trials,
            "skipped": skipped,
            "uncovered": uncovered,
            "max_rel": worst_rel,
            **({"ungated": declared} if declared else {}),
            **totals,
            **self._devices(translated, handle),
        }
        if uncovered:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(uncovered)} translated subprogram(s) were never compared: "
                + ", ".join(uncovered[:5])
                + " -- defer them or drop them from the unit; silence is not a pass",
            )
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
        if totals["points"] == 0:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                "zero numerical points were compared; that is not a pass",
            )
        if totals["integer_mismatch"]:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{totals['integer_mismatch']}/{totals['integer_points']} integer point(s) "
                "differ exactly; integer mismatches cannot be tolerance-excused",
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
        setup: list[dict[str, Any]],
        translated: Any,
        truth: Any,
        wrappers: dict[str, str],
        arg_naming: str = "lower",
    ) -> None:
        from recast.transform.numpy.vocabulary import pysafe

        spell = pysafe if arg_naming == "pysafe" else str.lower
        for call in setup:
            name = call["subprogram"]
            inputs = call.get("inputs", {})
            getattr(translated, pysafe(name))(**{pysafe(k): v for k, v in inputs.items()})
            if truth is None:
                # A replayed reference has no state to set: whatever the
                # production run's module state was is already folded into the
                # numbers it recorded. The candidate still needs the call, and
                # an operator whose ``setup`` does not match the run's own
                # initialization gets a difference rather than a silent pass --
                # which is the correct outcome and worth naming, because it is
                # the one thing about a replay that cannot be checked from
                # here.
                continue
            getattr(truth, wrappers.get(name, f"w_{name}"))(
                **{spell(k): v for k, v in inputs.items()}
            )

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
        dominant_axis: Any = -1,
        rel_scale: str = "element",
        arg_naming: str = "lower",
        convention: str = "f2py",
        samples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from recast.transform.numpy.vocabulary import pysafe

        if convention not in {"f2py", "emitted", "recorded"}:
            return {"error": f"oracle declares unsupported return convention {convention!r}"}

        declared_dtypes = [
            (f"argument {a.get('name', '<unnamed>')!r}", a.get("dtype")) for a in sub["args"]
        ]
        if sub["kind"] == "function":
            declared_dtypes.append(("function result", sub.get("result_dtype")))
        unsupported = [
            f"{place}={dtype!r}"
            for place, dtype in declared_dtypes
            if not isinstance(dtype, str) or dtype not in SUPPORTED_DTYPES
        ]
        if unsupported:
            return {
                "error": "unsupported declared dtype(s) "
                f"{', '.join(unsupported)}; supported dtypes are "
                f"{', '.join(sorted(SUPPORTED_DTYPES))}"
            }

        required = [a for a in sub["args"] if not a.get("optional")]
        outs_all = [a for a in sub["args"] if a["intent"] in ("OUT", "INOUT")]
        outs_required = [a for a in outs_all if not a.get("optional")]
        unknown_intents = [a["name"] for a in sub["args"] if a["intent"] == "UNKNOWN"]
        if unknown_intents:
            return {
                "error": "argument(s) "
                f"{', '.join(unknown_intents)} have UNKNOWN intent; this verifier cannot "
                "know whether their post-call values are outputs"
            }
        if sub["kind"] == "function" and outs_all:
            names = ", ".join(a["name"] for a in outs_all)
            return {
                "error": f"function {name!r} declares OUT/INOUT dummy argument(s) "
                f"{names}; this verifier cannot pair both its result and side effects"
            }
        logical_inouts = [
            a["name"] for a in outs_all if a["intent"] == "INOUT" and a.get("dtype") == "bool"
        ]
        if convention == "f2py" and logical_inouts:
            return {
                "error": "f2py LOGICAL INOUT dummy argument(s) "
                f"{', '.join(logical_inouts)} have no portable Python buffer ABI; "
                "refusing to guess the compiler's raw true representation"
            }

        # A scalar that names another argument's extent is not free data: it
        # must equal the extent the arrays are generated with, or every call
        # is a shape error rather than a comparison.
        dimension_names = {
            token
            for argument in sub["args"]
            for dim in argument.get("dims") or []
            for token in re.findall(
                r"[a-z_]\w*", f"{dim.get('lb') or ''} {dim.get('ub') or ''}".lower()
            )
        }

        points = bit_exact = nan_mismatch = 0
        integer_points = integer_mismatch = 0
        max_ulp = 0
        max_ulp_dominant = 0
        dominant_points = 0
        max_rel = 0.0
        # Replayed samples are the trials, and there are as many as were
        # recorded. ``trials`` is a sampling parameter and does not apply: a
        # recording cannot be asked for more points than it holds, and
        # truncating it to a count chosen here would silently narrow the
        # evidence.
        rounds: list[Any] = list(samples) if samples is not None else list(range(trials))
        for round_index, round_item in enumerate(rounds):
            # hash() is salted per process; a seed must not be.
            rng = np.random.default_rng(
                int.from_bytes(f"{name}:{round_index}".encode(), "big") % 2**32
            )
            if samples is not None:
                bound = self._recorded_inputs(np, required, round_item)
                if isinstance(bound, str):
                    return {"error": bound}
                inputs = bound
            else:
                inputs = {}
                for argument in required:
                    if argument["intent"] == "OUT" and not argument.get("buffer"):
                        continue
                    # An intent(out) buffer is the caller's storage: generated
                    # like an input, handed to the candidate, and compared
                    # after the call the way any output is.
                    lowered = argument["name"].lower()
                    if not argument.get("dims") and (lowered in dimension_names or lowered in dims):
                        inputs[argument["name"]] = np.int32(_resolve_extent(lowered, dims))
                    else:
                        inputs[argument["name"]] = self._value(np, argument, dims, ranges, rng)

            recorded_outputs = None
            if samples is not None:
                recorded_outputs = round_item.get("outputs")
                if not isinstance(recorded_outputs, dict):
                    return {"error": f"{round_item.get('source', 'sample')} has no OUTPUT mapping"}
                required_output_names = (
                    [sub.get("result") or "result"]
                    if sub["kind"] == "function"
                    else [a["name"] for a in outs_required]
                )
                missing_outputs = [
                    output
                    for output in required_output_names
                    if output.lower() not in recorded_outputs
                ]
                if missing_outputs:
                    return {
                        "error": "the recorded sample carries no value for required output(s) "
                        f"{', '.join(missing_outputs)}; partial output evidence is not a pass"
                    }

            if prepare is not None and samples is None:
                # The candidate may carry a ``_PREPARE_INPUTS(name, inputs,
                # rng)`` hook, the way it carries ``_SIGNATURES``: per-name
                # ranges cannot express structure -- a pressure column must
                # be monotone, an interface field must bracket its levels --
                # and unphysical inputs drive both sides into error paths
                # the production model aborts out of. The hook shapes inputs
                # into the defined domain; it cannot bias the verdict,
                # because both sides receive the same shaped inputs.
                #
                # It is skipped for a replay, and that is the important half:
                # the hook exists to drag *generated* inputs into the physical
                # domain, and recorded inputs are already there by
                # construction. Running it would let the candidate edit the
                # production run's own numbers before being judged on them,
                # which is the one thing a hook supplied by the artifact under
                # test must never be able to do.
                prepare(name, inputs, rng)

            # Keyword calls on both sides: f2py reorders inferred-dimension
            # scalars into trailing keywords, so positional order is not a
            # shared vocabulary -- names are.
            translated_kwargs = {
                pysafe(a["name"]): inputs[a["name"]]
                for a in required
                if a["intent"] != "OUT" or (a.get("buffer") and a["name"] in inputs)
            }
            # How the reference spells an argument is the reference's business,
            # and it declares which on its handle. f2py lowercases every dummy
            # name, because Fortran is case-insensitive and the source's
            # spelling is not a fact about the interface -- a candidate that
            # reports `sl_prePBL` still reaches the same oracle argument. An
            # anchor emitted by this engine's own backend spells names the
            # emitted way instead, because both sides of that comparison came
            # out of the same emitter.
            spell = pysafe if arg_naming == "pysafe" else str.lower
            try:
                truth_kwargs = {
                    spell(a["name"]): self._truth_input(np, a, inputs[a["name"]], convention)
                    for a in required
                    if a["intent"] != "OUT"
                }
            except Exception as error:
                return {
                    "error": f"oracle input preparation failed: {type(error).__name__}: {error}"
                }
            truth_args = [truth_kwargs[spell(a["name"])] for a in required if a["intent"] != "OUT"]
            try:
                translated_out = translated_fn(**translated_kwargs)
            except Exception as error:
                return {"error": f"candidate raised: {type(error).__name__}: {error}"}
            if samples is not None:
                # Nothing to call: the reference already ran, in production,
                # and what it produced is the recording.
                truth_out = recorded_outputs
            else:
                try:
                    truth_out = truth_fn(**truth_kwargs)
                except Exception as error:
                    return {"error": f"oracle raised: {type(error).__name__}: {error}"}

            pairs = self._paired_outputs(
                sub,
                outs_all,
                outs_required,
                required,
                translated_out,
                truth_out,
                truth_args,
                convention,
            )
            if isinstance(pairs, str):
                return {"error": pairs}
            output_dtypes = (
                {sub.get("result") or "result": sub.get("result_dtype")}
                if sub["kind"] == "function"
                else {a["name"]: a.get("dtype") for a in outs_all}
            )
            for label, ours, theirs in pairs:
                declared_dtype = output_dtypes.get(label)
                if declared_dtype in {"int32", "int64"}:
                    shaped_ours = self._integer_output(
                        np, ours, declared_dtype, label=label, side="candidate"
                    )
                    if isinstance(shaped_ours, str):
                        return {"error": shaped_ours}
                    shaped_theirs = self._integer_output(
                        np, theirs, declared_dtype, label=label, side="oracle"
                    )
                    if isinstance(shaped_theirs, str):
                        return {"error": shaped_theirs}
                    if shaped_ours.shape != shaped_theirs.shape:
                        return {
                            "error": f"{label}: shape {shaped_ours.shape} vs {shaped_theirs.shape}"
                        }
                    exact = shaped_ours == shaped_theirs
                    compared = int(exact.size)
                    agreed = int(np.count_nonzero(exact))
                    points += compared
                    bit_exact += agreed
                    integer_points += compared
                    integer_mismatch += compared - agreed
                    continue
                if declared_dtype == "bool":
                    # f2py exposes Fortran LOGICAL as a C int.  A true value
                    # need only be nonzero: gfortran commonly emits 1/-1/-2,
                    # and another compiler may choose a different bit pattern.
                    # Compare the declared logical meaning, not that private
                    # representation.
                    shaped_ours = np.asarray(np.asarray(ours) != 0, dtype=np.float64)
                    shaped_theirs = np.asarray(np.asarray(theirs) != 0, dtype=np.float64)
                else:
                    shaped_ours = np.asarray(ours, dtype=np.float64)
                    shaped_theirs = np.asarray(theirs, dtype=np.float64)
                if shaped_ours.shape != shaped_theirs.shape:
                    return {"error": f"{label}: shape {shaped_ours.shape} vs {shaped_theirs.shape}"}
                a = shaped_ours.ravel()
                b = shaped_theirs.ravel()
                audit = ulp_audit(
                    a.tolist(),
                    b.tolist(),
                    dominant=self._dominance(np, shaped_theirs, dominant_at, dominant_axis),
                )
                points += audit["total_points"]
                bit_exact += audit["bit_exact"]
                nan_mismatch += audit["nan_mismatch"]
                max_ulp = max(max_ulp, audit["max_ulp"])
                if "max_ulp_dominant" in audit:
                    max_ulp_dominant = max(max_ulp_dominant, audit["max_ulp_dominant"])
                    dominant_points += audit["dominant_points"]
                if audit["bit_exact"] != audit["total_points"]:
                    # ``rel_scale``: each element against itself (the default),
                    # or ``"array"`` -- against the array's largest magnitude,
                    # for a layout where a cancellation residual of 1e-17 sits
                    # beside values of order one and its own relative error
                    # says nothing about the translation.
                    if rel_scale == "array":
                        scale = np.maximum(float(np.abs(b).max()) if b.size else 0.0, 1e-300)
                    else:
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
            "integer_points": integer_points,
            "integer_mismatch": integer_mismatch,
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
    def _dominance(
        np: Any, reference: Any, dominant_at: float | None, axis: Any = -1
    ) -> list[bool] | None:
        """Which elements a ULP bound is allowed to be held to.

        ``|v| >= fraction * the maximum along the last axis``, so an element is
        judged against its own row: a column of small values is not excused by
        a large value somewhere else in the array. The *reference* side decides,
        because whether an element matters is a fact about what it should have
        been, not about the candidate being judged.

        ``axis`` is the operator's (``dominant_axis``): the last axis by
        default, or ``"all"`` for the whole array -- for a layout whose last
        axis is not a row of comparable values (a two-element sun/shade pair,
        say), where a cancellation residual of 1e-17 would otherwise be the
        maximum of its own row and judged at the ULP tier.
        """
        if dominant_at is None:
            return None
        magnitude = np.abs(reference)
        if axis in ("all", None) or magnitude.ndim <= 1:
            scale = magnitude.max()
        else:
            scale = magnitude.max(axis=int(axis), keepdims=True)
        mask: list[bool] = (magnitude >= dominant_at * scale).ravel().tolist()
        return mask

    @staticmethod
    def _type_recorded(np: Any, samples: list[dict[str, Any]], table: dict[str, Any]) -> int:
        """Cast each recorded sample's inputs to the dtypes its signature declares.

        The cast changes no value -- an integer written as ``3`` is 3 -- and
        a scalar recorded as a one-element array is shaped back to a scalar,
        on the output side as well, so it compares against a scalar."""
        kinds = {
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
            "float32": np.float32,
            "float64": np.float64,
        }
        cast = 0
        for sample in samples:
            sig = table.get(str(sample.get("subprogram", "")))
            if not sig:
                continue
            for argument in sig["args"]:
                key = argument["name"].lower()
                value = sample.get("outputs", {}).get(key)
                if isinstance(value, np.ndarray) and not argument.get("dims") and value.size == 1:
                    sample["outputs"][key] = value.reshape(-1)[0]
            for argument in sig["args"]:
                key = argument["name"].lower()
                dtype = kinds.get(str(argument["dtype"]))
                if key not in sample.get("inputs", {}) or dtype is None:
                    continue
                value = sample["inputs"][key]
                if isinstance(value, np.ndarray) and not argument.get("dims") and value.size == 1:
                    sample["inputs"][key] = dtype(value.reshape(-1)[0])
                    cast += 1
                elif isinstance(value, np.ndarray):
                    if value.dtype != dtype:
                        sample["inputs"][key] = np.asfortranarray(value.astype(dtype))
                        cast += 1
                elif not isinstance(value, dtype):
                    sample["inputs"][key] = dtype(value)
                    cast += 1
        return cast

    @staticmethod
    def _recorded_inputs(
        np: Any, required: list[dict[str, Any]], sample: dict[str, Any]
    ) -> dict[str, Any] | str:
        """Bind one recorded sample's INPUT sections to the declared arguments.

        By exact name, lowercased, and nothing else. The script this oracle
        came from matched fuzzily -- exact, then with ``in``/``out`` stripped,
        then any substring either way -- and filled whatever was left with
        zeros. That is defensible in a one-shot investigation and not in a
        gate: a substring match binds ``t`` to ``theta``, a zero fill invents
        an input the run never had, and either one produces numbers that can
        be compared and mean nothing. So a required argument the recording does
        not name is a refusal, which is what a verifier that fails closed owes
        its reader.
        """
        recorded = sample.get("inputs", {})
        inputs: dict[str, Any] = {}
        missing = []
        for argument in required:
            if argument["intent"] == "OUT":
                continue
            key = argument["name"].lower()
            if key not in recorded:
                missing.append(argument["name"])
                continue
            value = recorded[key]
            inputs[argument["name"]] = np.copy(value) if isinstance(value, np.ndarray) else value
        if missing:
            return (
                f"{sample.get('source', 'sample')} records no value for "
                f"{', '.join(missing)}; a replay does not invent one"
            )
        return inputs

    @staticmethod
    def _truth_input(
        np: Any,
        argument: dict[str, Any],
        value: Any,
        convention: str,
    ) -> Any:
        """Give the reference an independent input with its required ABI shape.

        f2py represents a scalar ``intent(inout)`` dummy as an in/output
        rank-0 array.  It accepts a NumPy scalar too, but that object is
        immutable: the wrapper updates a temporary and Python observes the
        original value.  A writable zero-dimensional ndarray is therefore
        part of the f2py calling convention, not a change to the sampled
        value.  Array INOUTs already arrive as independent writable copies;
        emitted and recorded references retain their own conventions.
        """
        if convention == "f2py" and argument["intent"] == "INOUT" and not argument.get("dims"):
            buffered = np.asarray(value).copy()
            if buffered.ndim != 0:
                raise ValueError(f"scalar INOUT {argument['name']!r} became rank {buffered.ndim}")
            return buffered
        return np.copy(value) if isinstance(value, np.ndarray) else value

    @staticmethod
    def _integer_output(
        np: Any,
        value: Any,
        declared_dtype: str,
        *,
        label: str,
        side: str,
    ) -> Any | str:
        """Validate and preserve one declared integer output exactly.

        Casting through float64 aliases adjacent int64 values above 2**53.
        Casting a float *to* an integer is no safer: it lets a candidate that
        violated its declared interface masquerade as one that did not.  Only
        actual integer values in the declared signed range enter the exact
        comparison.
        """
        try:
            raw = np.asarray(value)
        except Exception as error:
            return (
                f"{label}: {side} {declared_dtype} output cannot be represented as an "
                f"array: {type(error).__name__}: {error}"
            )

        if raw.dtype.kind not in {"i", "u", "O"}:
            return (
                f"{label}: {side} declared {declared_dtype} but produced non-integer "
                f"dtype {raw.dtype}"
            )
        if raw.dtype.kind == "O":
            for item in raw.flat:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                    return (
                        f"{label}: {side} declared {declared_dtype} but produced "
                        f"non-integer value of type {type(item).__name__}"
                    )

        target = np.dtype(np.int32 if declared_dtype == "int32" else np.int64)
        limits = np.iinfo(target)
        if raw.size:
            if raw.dtype.kind == "O":
                smallest = min(int(item) for item in raw.flat)
                largest = max(int(item) for item in raw.flat)
            else:
                smallest = int(raw.min())
                largest = int(raw.max())
            if smallest < int(limits.min) or largest > int(limits.max):
                offending = smallest if smallest < int(limits.min) else largest
                return (
                    f"{label}: {side} {declared_dtype} output value {offending} is outside "
                    f"[{int(limits.min)}, {int(limits.max)}]"
                )
        return raw.astype(target, copy=False)

    @staticmethod
    def _paired_outputs(
        sub: dict[str, Any],
        outs_all: list[dict[str, Any]],
        outs_required: list[dict[str, Any]],
        required: list[dict[str, Any]],
        translated_out: Any,
        truth_out: Any,
        truth_args: list[Any],
        convention: str = "f2py",
    ) -> list[tuple[str, Any, Any]] | str:
        """Match the two sides' outputs by argument.

        The translation returns every OUT/INOUT argument, optional ones
        included, in declaration order (a function returns its result). What
        the *reference* returns depends on what kind of reference it is, and it
        says which on its handle rather than being guessed at here.

        ``f2py`` returns the wrapper's ``intent(out)`` arguments and mutates
        the ``inout`` ones in place, so INOUT values are read back from the
        independent arrays that were passed (including rank-0 buffers for
        scalar INOUTs). ``emitted`` is a reference this engine's own backend
        produced -- a NumPy anchor for a port -- and returns exactly what the
        candidate does, because the same emitter wrote both.
        """
        if convention == "recorded":
            # ``truth_out`` is not a return value here -- it is the recorded
            # OUTPUT section, keyed by the name the probe wrote. The match is
            # therefore by exact name on both sides. Every required output was
            # preflighted before the candidate call; keep the same check here
            # as a fail-closed local invariant for direct callers.
            mine = list(translated_out) if isinstance(translated_out, tuple) else [translated_out]
            names = (
                [sub.get("result") or "result"]
                if sub["kind"] == "function"
                else [a["name"] for a in outs_all]
            )
            if len(mine) != len(names):
                return (
                    f"candidate returned {len(mine)} value(s) for {len(names)} "
                    "out-intent argument(s)"
                )
            ours_by_name = dict(zip(names, mine, strict=True))
            wanted = names if sub["kind"] == "function" else [a["name"] for a in outs_required]
            missing = [name for name in wanted if name.lower() not in truth_out]
            if missing:
                return (
                    "the recorded sample carries no value for required output(s) "
                    f"{', '.join(missing)}; partial output evidence is not a pass"
                )
            pairs: list[tuple[str, Any, Any]] = []
            for name in wanted:
                ours = ours_by_name[name]
                theirs = truth_out[name.lower()]
                # The probe format has no rank-0 section: a scalar result is
                # written as a section holding exactly one value and parses as
                # shape (1,). When the candidate returned a scalar, read the
                # recording back as the scalar it is. The one value is still
                # compared bit for bit, so the reshape hides nothing; without
                # it every recorded scalar function was "shape () vs (1,)".
                import numpy as np

                if getattr(theirs, "shape", None) == (1,) and np.ndim(ours) == 0:
                    theirs = theirs.reshape(())
                pairs.append((name, ours, theirs))
            return pairs

        if sub["kind"] == "function":
            # Both sides return the result, whatever kind of reference this is.
            return [(sub.get("result") or "result", translated_out, truth_out)]

        if convention == "emitted":
            mine = list(translated_out) if isinstance(translated_out, tuple) else [translated_out]
            yours = list(truth_out) if isinstance(truth_out, tuple) else [truth_out]
            names = [a["name"] for a in outs_all]
            if len(mine) != len(names) or len(yours) != len(names):
                return (
                    f"candidate returned {len(mine)} and reference {len(yours)} value(s) "
                    f"for {len(names)} out-intent argument(s)"
                )
            ours_by_name = dict(zip(names, mine, strict=True))
            theirs_by_name = dict(zip(names, yours, strict=True))
            return [
                (a["name"], ours_by_name[a["name"]], theirs_by_name[a["name"]])
                for a in outs_required
            ]

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
            shape = tuple(_extent(d, dims) for d in argument["dims"])
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
