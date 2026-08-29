"""The behaviour hook, and the rule about who is allowed to supply one.

``Transform.deterministic`` is read at plan time, off the plugin the recipe
names, to decide whether the run needs a hard gate. So the dangerous shape is
not a transform that consults a model -- it is one that consults a model while
declaring that it does not, because then the gate rule looks at the
declaration, sees a rule-driven transform, and lets the recipe through without
one. These tests are about that declaration holding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.errors import ConfigError
from recast.fortran.frontend import FortranFrontend
from recast.model import Candidate, Facts, Unit
from recast.transform.numpy.agentic import DeferredSite
from recast.transform.numpy.translate import NumpyTranslation

SOURCE = """\
module hook_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine mechanical(x, y)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: y
    y = 2.0_r8 * x + 1.0_r8
  end subroutine mechanical

  subroutine refused(x, y)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: y
    character(len=32) :: buffer
    ! A formatted internal write with a D edit descriptor is refused -- the
    ! runtime does not implement it -- so this block is what the hook is offered.
    y = 2.0_r8 * x
    write(buffer, '(D8.2)') y
  end subroutine refused
end module hook_demo
"""

FILL = ["buffer = '%8.2f' % y"]


@pytest.fixture
def subject(tmp_path: Path) -> tuple[Unit, Facts, dict[str, Any]]:
    (tmp_path / "hook_demo.f90").write_text(SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:hook_demo")
    return unit, frontend.analyze(unit, tmp_path), {"root": str(tmp_path)}


def _apply(subject: Any, transform: NumpyTranslation, **extra: Any) -> Candidate:
    unit, facts, config = subject
    return transform.apply(unit, facts, {**config, **extra})


def test_without_a_handler_the_site_is_deferred(subject: Any) -> None:
    """The baseline the hook changes, and the behaviour every other transform
    still gets: refused means refused, and the run stays reproducible."""
    first = _apply(subject, NumpyTranslation())
    second = _apply(subject, NumpyTranslation())
    assert first.deferred == [
        "refused/B002: formatted internal write: unsupported edit descriptor in '(D8.2)'"
    ]
    assert first.digest() == second.digest()


def test_a_deterministic_transform_refuses_the_handler(subject: Any) -> None:
    """The mechanism behind the rule, rather than a sentence in a document.

    Without this, an operator could make the reference transform agentic from
    the outside while it went on declaring ``deterministic = True``, and the
    recipe rule that requires a hard gate for agentic work would never see it.
    """
    with pytest.raises(ConfigError, match="deterministic = True"):
        _apply(subject, NumpyTranslation(), deferred_handler=lambda site: None)


def test_a_transform_that_declared_otherwise_may_fill_the_site(subject: Any) -> None:
    seen: list[DeferredSite] = []

    def handler(site: DeferredSite) -> dict[str, Any]:
        seen.append(site)
        return {
            "python": FILL,
            "reason": "edit descriptor spelled out",
            "model": "a-model",
            "prompt_digest": "sha256:deadbeef",
        }

    candidate = _apply(subject, NumpyTranslation(deterministic=False), deferred_handler=handler)

    assert candidate.deferred == [], "the site was filled, so nothing is left for the queue"
    assert [s.block for s in seen] == ["B002"]
    assert seen[0].subprogram == "refused"
    assert "WRITE" in seen[0].fortran.upper(), "the handler sees the source it was asked about"
    assert seen[0].reason == "formatted internal write: unsupported edit descriptor in '(D8.2)'"

    emitted = candidate.files[Path("hook_demo_numpy.py")].decode()
    assert "buffer = '%8.2f' % y" in emitted
    assert "NotImplementedError" not in emitted, "a filled site does not also raise"


def test_the_provenance_the_handler_returns_reaches_the_notes(subject: Any) -> None:
    """A non-deterministic transform's reproducibility is by provenance, and
    provenance nobody recorded is not provenance."""

    def handler(site: DeferredSite) -> dict[str, Any]:
        return {"python": FILL, "reason": "r", "model": "a-model", "prompt_digest": "sha256:d"}

    candidate = _apply(subject, NumpyTranslation(deterministic=False), deferred_handler=handler)
    filled = [b for b in candidate.notes["blocks"] if b["status"] == "agent_filled"]
    assert len(filled) == 1
    assert filled[0]["model"] == "a-model"
    assert filled[0]["prompt_digest"] == "sha256:d"


def test_a_handler_that_raises_leaves_the_site_deferred_and_says_so(subject: Any) -> None:
    """One block's handler failing is not the run's death, and not a silence
    either: the site goes back to the queue it would have been in, carrying
    what went wrong."""

    def handler(site: DeferredSite) -> dict[str, Any]:
        raise RuntimeError("model timed out")

    candidate = _apply(subject, NumpyTranslation(deterministic=False), deferred_handler=handler)
    assert candidate.deferred == [
        "refused/B002: formatted internal write: unsupported edit descriptor in '(D8.2)'"
    ]
    queued = [b for b in candidate.notes["blocks"] if b["status"] == "agent_queue"]
    assert queued[0]["handler_error"] == "handler raised RuntimeError: model timed out"


def test_a_handler_that_answers_with_nonsense_is_treated_as_a_refusal(subject: Any) -> None:
    candidate = _apply(
        subject,
        NumpyTranslation(deterministic=False),
        deferred_handler=lambda site: {"reason": "forgot the code"},
    )
    queued = [b for b in candidate.notes["blocks"] if b["status"] == "agent_queue"]
    assert "no 'python' list" in queued[0]["handler_error"]


def test_a_handler_may_decline(subject: Any) -> None:
    """Declining is a normal answer -- the model had nothing useful to say --
    and it must be distinguishable from the model having failed."""
    candidate = _apply(
        subject, NumpyTranslation(deterministic=False), deferred_handler=lambda site: None
    )
    queued = [b for b in candidate.notes["blocks"] if b["status"] == "agent_queue"]
    assert "handler_error" not in queued[0]
    assert candidate.deferred == [
        "refused/B002: formatted internal write: unsupported edit descriptor in '(D8.2)'"
    ]


def test_a_handler_from_a_config_file_is_rejected(subject: Any) -> None:
    """Config arrives as JSON, so a string is the shape this mistake takes."""
    with pytest.raises(ConfigError, match="not callable"):
        _apply(
            subject,
            NumpyTranslation(deterministic=False),
            deferred_handler="yourpkg.handlers:fill",
        )
