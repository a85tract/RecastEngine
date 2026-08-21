"""The behaviour hook: what a plugin supplies for the blocks the rules refuse.

The rules translate what they have a rule for and put everything else on
``Candidate.deferred``. Two ways to fill those sites, and they are not
redundant -- they differ in *when* they arrive, and that decides what each can
do.

**Data, before the run.** ``config["patches"]`` maps ``"subprogram/block"`` to
a replacement worked out ahead of time and audited by an operator. It is known
before a line is rendered, so it may add module-level imports. The run stays
deterministic: the same patches in, the same bytes out, and whatever produced
them -- a model, a person, a second pass -- is outside the run entirely.

**Behaviour, during the run.** ``config["deferred_handler"]`` is a callable the
transform consults at the moment a block is refused, with the refusal in hand.
It sees what the rules saw, which is the whole reason to prefer it: filling a
site out of band means reconstructing that context from a previous run's
report. It cannot add imports, because the module header is assembled before
the subprograms are rendered -- a filled body works with what the emitted
module already has (``np``, the star-imported constants, ``_ext``, and the
``_RUNTIME`` side channel).

**A handler makes the transform non-deterministic, and the transform that says
so has to be the one the recipe names.** ``Transform.deterministic`` is read at
plan time, off the registered plugin, to decide whether the recipe needs a hard
gate -- so a transform that claims determinism and then quietly consults a
model would slip past that rule. ``NumpyTranslation`` therefore refuses a
handler unless it was constructed having said otherwise, which means the class
that accepts one is a class somebody wrote and registered:

    class AgenticTranslation(Transform):
        name = "yourpkg.translate.agentic"
        deterministic = False                       # the claim the gate reads

        def __init__(self) -> None:
            self._engine = NumpyTranslation(deterministic=False)

        def apply(self, unit, facts, config):
            filled = {**config, "deferred_handler": self._fill}
            candidate = self._engine.apply(unit, facts, filled)
            candidate.transform = self.name
            return candidate
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DeferredHandler", "DeferredSite"]


@dataclass(frozen=True)
class DeferredSite:
    """One block the rules refused, with everything the transform knew about it."""

    subprogram: str
    block: str
    """Block id, e.g. ``B002``. ``f"{subprogram}/{block}"`` is the key used in
    ``Candidate.deferred`` and in the ``patches`` table."""

    fortran: str
    """The source statement, as the parser rendered it back."""

    src_span: tuple[int, int]
    """1-based inclusive line range in the source file."""

    reason: str
    """Why the rules refused -- the ``NoRule`` or ``Unanalyzable`` message."""

    names: dict[str, str] = field(default_factory=dict)
    """Emitted name -> source name for this subprogram, so a filled body spells
    variables the way the rest of the emitted module spells them."""


DeferredHandler = Callable[[DeferredSite], "dict[str, Any] | None"]
"""Fill a refused block, or decline it.

Return ``None`` to leave the site deferred -- declining is a normal answer, and
the site goes to ``Candidate.deferred`` exactly as it would have.

Return a mapping to fill it:

``python``
    The body, as a list of un-indented source lines. Required.
``reason``
    One line on what was done, for the block report.

Any other keys ride along into the block's entry in ``Candidate.notes``, which
is where a non-deterministic transform's provenance belongs: record at least
``model`` and ``prompt_digest`` there, because an artifact that cannot be traced
to the model that produced it replays to nothing.
"""
