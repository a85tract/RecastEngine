"""Which emitted symbol a Fortran symbol becomes.

Small, and worth its own module for two reasons. It is where Fortran's scoping
rules have to be honoured exactly -- a subprogram's own dummies and locals
shadow anything at module level, and a translation that binds a local to a
module constant of the same name does not read the wrong value, it *writes
through* to it. And it is the table the read/write cross-check needs in order
to undo the renaming, so producing it is part of the Transform's obligation
rather than an internal detail.

Two kinds of renaming happen here and they are different in kind. A collision
rename -- ``lambda`` to ``lambda_`` -- is forced by the target language and is
reversible. A constant rename -- ``pi`` to ``PI``, a hoisted literal to
``F_273P15`` -- is a decision the zero-literal rule made earlier, and the map
back to the source name is the only record of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recast.fortran.semantics import Semantics
from recast.transform.numpy.vocabulary import (
    WHITELIST_INT,
    WHITELIST_REAL,
    pysafe,
)
from recast.transform.rules import NoRule

__all__ = ["Names", "for_subprogram"]


@dataclass
class Names:
    """Symbol resolution for one subprogram."""

    semantics: Semantics
    module_parameters: dict[str, str] = field(default_factory=dict)
    """Module-level parameter -> its emitted (upper-case) constant name."""

    use_parameters: dict[str, str] = field(default_factory=dict)
    """Constant imported from a module that is not being translated."""

    companion_globals: dict[str, str] = field(default_factory=dict)
    """Global reached through a sibling translated module's alias."""

    literals: dict[str, str] = field(default_factory=dict)
    """Literal text -> the hoisted constant standing for it, in this subprogram."""

    local_constants: dict[str, str] = field(default_factory=dict)
    """Local parameter -> its emitted constant name, e.g. ``WORK__ALPHA``.

    Taken from the frontend's constants record rather than rebuilt from the
    convention. The pipeline this came from spelled the convention out in two
    places, and a cross-check that rebuilds a name the emitter also builds is
    checking its own arithmetic rather than the translation.
    """

    def symbol(self, name: str) -> str:
        """The emitted name for a Fortran symbol.

        Innermost scope first, and that order is the whole point. A local
        called ``tmin`` must not resolve to a companion module's ``TMIN``: the
        translation would read the wrong value and, on assignment, write
        through to a constant the rest of the program shares.
        """
        lowered = name.lower()
        subprogram = self.semantics.subprogram
        if any(p["name"] == lowered for p in subprogram["local_parameters"]):
            return pysafe(lowered)
        if any(a["name"] == lowered for a in subprogram["args"]):
            return pysafe(lowered)
        if any(loc["name"] == lowered for loc in subprogram.get("locals") or ()):
            return pysafe(lowered)
        if lowered in self.module_parameters:
            return self.module_parameters[lowered]
        if lowered in self.use_parameters:
            return self.use_parameters[lowered]
        if lowered in self.companion_globals:
            return self.companion_globals[lowered]
        return pysafe(lowered)

    def literal(self, node: Any) -> str:
        """The emitted text for a numeric literal.

        A whitelisted value is written out; anything else is the hoisted
        constant that the frontend named for it. Refusing when there is no
        entry is deliberate -- emitting the bare number would put a magic
        number back into code the zero-literal rule had just cleaned, and the
        gate that would notice runs much later than this does.
        """
        from recast.fortran._parse import f03

        text = str(node)
        base = text.split("_")[0]
        if isinstance(node, f03.Real_Literal_Constant):
            value = float(base.lower().replace("d", "e"))
            if value in WHITELIST_REAL:
                return repr(value)
        elif base in WHITELIST_INT:
            return base
        hoisted = self.literals.get(text)
        if hoisted is None:
            raise NoRule(f"literal {text!r} was never hoisted, so it has no name to emit")
        return hoisted

    def as_protocol_table(self) -> dict[str, str]:
        """Emitted name -> source name, for ``Candidate.notes``.

        Only the constant renames. A collision rename is reversible from the
        target language's own rules, and the cross-check undoes those itself.
        """
        table = {emitted: source for source, emitted in self.module_parameters.items()}
        table.update({emitted: source for source, emitted in self.use_parameters.items()})
        table.update({emitted: source for source, emitted in self.local_constants.items()})
        return table


def for_subprogram(
    semantics: Semantics,
    constants: dict[str, Any] | None = None,
    *,
    use_parameters: dict[str, str] | None = None,
    companion_globals: dict[str, str] | None = None,
) -> Names:
    """Build ``Names`` from the frontend's two records plus the operator's tables.

    ``constants`` is a whole-file ``constants.extract`` record; only this
    subprogram's entries are kept from it, so a value hoisted in one routine
    cannot leak into another that never mentioned it, and the constant names
    come from the frontend that chose them rather than from a second copy of
    the convention.
    """
    record = constants or {}
    name = semantics.subprogram["name"]
    return Names(
        semantics=semantics,
        module_parameters={
            p["name"]: p["name"].upper() for p in semantics.module["module_parameters"]
        },
        use_parameters=dict(use_parameters or {}),
        companion_globals=dict(companion_globals or {}),
        literals=dict(record.get("literal_map", {}).get(name, {})),
        local_constants={
            p["name"]: p["const"]
            for p in record.get("local_parameters", ())
            if p["subprogram"] == name
        },
    )
