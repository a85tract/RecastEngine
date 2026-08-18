"""``symbolic.notary``: every rewritten expression proves itself equivalent.

Migrated from CESM-language-translator ``pipeline/notary.py`` and the
expression half of ``pipeline/highprec_verify.py``, which were the same idea
written twice: sample two expressions over their physical input ranges at 50
significant digits -- about 166 bits, three times a double's mantissa -- and
call them equivalent only if every sample agrees to within 1e-45 relative.

The mechanical translation itself never needs this: it prints directly from
the parse tree, fully parenthesized, order preserved, and the emission
differential holds it to the pipeline byte for byte. What needs it is every
expression somebody *rewrote* -- a SymPy simplification, an agent's patch, a
hand reorder for a GPU -- because a rewrite that is algebra in exact
arithmetic can still be a different program in float64, and the only honest
classifications are "equivalent in exact arithmetic" and "an algorithmic
change someone must justify separately".

A Transform that rewrites records each site in ``Candidate.notes["rewrites"]``
as ``{"site", "old", "new", "ranges"}``, with ``ranges`` naming the physical
interval of every free symbol. A candidate with no rewrites passes with that
fact in its metrics -- the pipeline's production log recorded "zero rewrites"
explicitly rather than leaving it to be assumed, and so does this.

Sampling is deterministic (a fixed-seed LCG), because a notarization that
cannot be reproduced is testimony, not evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.model import Candidate, Confidence, Unit, Verdict
from recast.plugins.executor import Executor
from recast.plugins.verifier import StaticVerifier

__all__ = ["NotaryVerifier", "factory", "notarize"]

EQUIVALENCE_THRESHOLD = "1e-45"
"""Worst allowed relative difference at 50 digits. Far below anything float64
can express, far above the sampling precision's own noise floor."""

SEED = 20260611
"""The pipeline's seed, kept: its notarizations replay under this one."""


def _require_sympy() -> None:
    try:
        import mpmath  # noqa: F401
        import sympy  # noqa: F401
    except ImportError as exc:
        if (exc.name or "").split(".")[0] not in ("sympy", "mpmath"):
            raise
        raise ConfigError(
            "the notary needs sympy and mpmath, which are not installed. "
            "Install them with: pip install 'recast-engine[verify]'"
        ) from exc


def notarize(
    expression_old: str,
    expression_new: str,
    ranges: dict[str, tuple[float, float]],
    *,
    samples: int = 1000,
    digits: int = 50,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare two expression strings over sampled inputs at high precision.

    Returns ``{"verdict": "EQUIVALENT" | "ALGORITHMIC", ...}`` with the worst
    relative and absolute differences and the point that produced them. A
    free symbol with no range raises: sampling an unphysical interval proves
    nothing about the program, and guessing one would launder that nothing
    into a verdict.
    """
    _require_sympy()
    import mpmath
    import sympy

    with mpmath.workdps(digits):
        old = sympy.sympify(expression_old, evaluate=False)
        new = sympy.sympify(expression_new, evaluate=False)
        symbols = sorted(old.free_symbols | new.free_symbols, key=str)
        missing = [str(s) for s in symbols if str(s) not in ranges]
        if missing:
            raise ConfigError(f"no physical range for symbols {missing}")

        evaluate_old = sympy.lambdify(symbols, old, modules="mpmath")
        evaluate_new = sympy.lambdify(symbols, new, modules="mpmath")

        state = seed
        worst_rel = mpmath.mpf(0)
        worst_abs = mpmath.mpf(0)
        worst_point: dict[str, float] | None = None
        for _ in range(samples):
            point = []
            for symbol in symbols:
                state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
                unit = mpmath.mpf(state) / mpmath.mpf(2**64)
                low, high = ranges[str(symbol)]
                point.append(mpmath.mpf(low) + (mpmath.mpf(high) - mpmath.mpf(low)) * unit)
            try:
                value_old = evaluate_old(*point)
                value_new = evaluate_new(*point)
            except (ZeroDivisionError, ValueError):
                continue  # a pole in the sampled range; the next point decides
            difference = abs(value_old - value_new)
            relative = difference / max(abs(value_old), mpmath.mpf("1e-300"))
            worst_abs = max(worst_abs, difference)
            if relative > worst_rel:
                worst_rel = relative
                worst_point = {str(s): float(v) for s, v in zip(symbols, point, strict=True)}

        equivalent = worst_rel < mpmath.mpf(EQUIVALENCE_THRESHOLD)
        return {
            "verdict": "EQUIVALENT" if equivalent else "ALGORITHMIC",
            "worst_rel": float(worst_rel),
            "worst_abs": float(worst_abs),
            "worst_point": worst_point,
            "samples": samples,
            "digits": digits,
            "threshold": EQUIVALENCE_THRESHOLD,
        }


class NotaryVerifier(StaticVerifier):
    """Fail any recorded rewrite that does not prove out in exact arithmetic."""

    name = "symbolic.notary"
    provides = Confidence.SYMBOLIC

    def check(
        self,
        unit: Unit,
        candidate: Candidate,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        rewrites = candidate.notes.get("rewrites") or []
        judged: list[dict[str, Any]] = []
        failures: list[str] = []
        for rewrite in rewrites:
            site = rewrite.get("site", "<unnamed>")
            try:
                result = notarize(
                    rewrite["old"],
                    rewrite["new"],
                    {k: tuple(v) for k, v in rewrite["ranges"].items()},
                    samples=int(config.get("samples", 1000)),
                    digits=int(config.get("digits", 50)),
                )
            except Exception as error:  # fail closed, whatever broke
                judged.append({"site": site, "verdict": "FAILED", "reason": str(error)})
                failures.append(f"{site}: {error}")
                continue
            judged.append({"site": site, **result})
            if result["verdict"] != "EQUIVALENT":
                failures.append(
                    f"{site}: worst_rel={result['worst_rel']:.2e} at {result['worst_point']}"
                )

        metrics = {"rewrites": len(rewrites), "judged": judged}
        if failures:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(failures)}/{len(rewrites)} rewrites are algorithmic changes, "
                "not equivalences: " + "; ".join(failures[:3]),
            )
        detail = (
            f"all {len(rewrites)} rewrites equivalent at {config.get('digits', 50)} digits"
            if rewrites
            # Zero is a finding, not an absence: the translation printed
            # straight from the parse tree and nothing was rewritten.
            else "no rewrites to notarize; the translation is print-order faithful"
        )
        return self._verdict(candidate, Confidence.SYMBOLIC, metrics, detail)

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


def factory(**_config: Any) -> NotaryVerifier:
    return NotaryVerifier()
