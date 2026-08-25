"""``translate.numba``: the Transform, and the file it assembles.

The recipe already declared this slot -- ``TranslateRecipe`` accepts
``target: numba`` and plans a ``translate.numba`` stage -- and until now
``recast plan`` answered ``[MISS]``. This fills it.

The product is a second module beside the NumPy one rather than a replacement
for it, which is the whole shape of the backend: the NumPy translation is
where initialization, module state, admin routines and everything outside the
kernel subset stay, and it is the one the bit-exact gate has already passed.
This file imports it as ``_host``, re-exports what it delegates, and replaces
only the numeric bodies with compiled kernels. Nothing is duplicated, so
nothing can drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform

# Nothing heavier is imported at module scope. The registry loads every
# entry-point factory on a bare install -- no fparser, no numpy -- so that
# ``recast doctor`` works before the extras are there, and a missing extra has
# to surface on first use with its install line rather than at registration.

__all__ = ["NumbaTranslation", "factory"]

DOCSTRING = '''"""Machine-translated from {source} by recast -- stage 2 of 3.

``@njit`` kernels (``cache=True``, ``fastmath=False`` -- never enable
fastmath) taking their module-state read closure as explicit parameters.
Admin, initialization and module state live in the validated NumPy module
beside this one; the wrappers here keep the locked public signatures.
DO NOT hand-edit mechanical blocks -- fix the engine instead.
"""'''


class NumbaTranslation(Transform):
    """Fortran to compiled NumPy kernels, anchored on the NumPy translation."""

    name = "recast.translate.fortran-to-numba"
    requires = ("interface", "constants", "effects")
    deterministic = True

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return (
            unit.kind in ("module", "program")
            and "parse_error" not in unit.attrs
            and bool(facts.interface)
        )

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        from recast.transform.numba.backend import Kernels
        from recast.transform.numba.emitter import Emission, NumbaSubprograms
        from recast.transform.numpy.translate import NumpyTranslation, companion_tables
        from recast.transform.profiles import DEFAULT, PROFILES

        source = NumpyTranslation._verified_source(unit, facts, config)
        declared = config.get("companions")
        if declared is None:
            declared = facts.provenance.get("companions", [])
        records, remotes, companion_globals, companion_imports = companion_tables(declared)
        aliased = _aliased(companion_imports, records)

        kernels = Kernels(
            record=facts.interface,
            companions=aliased,
            remotes=remotes,
            externals=facts.provenance.get("externals", {}),
        )
        emission = Emission(kernels=kernels)
        use = config.get("use_constants")
        use_parameters = (
            {e["name"]: e["name"].upper() for e in use["resolved"] if e["requested"]} if use else {}
        )
        assembler = NumbaSubprograms(
            record=facts.interface,
            constants=facts.constants,
            profile=PROFILES[config.get("profile", DEFAULT)],
            companions=records,
            use_parameters=use_parameters,
            companion_globals=companion_globals,
            externals=facts.provenance.get("externals", {}),
            remotes=remotes,
            function_stubs=config.get("function_stubs", {}),
            statement_stubs=config.get("statement_stubs", {}),
            intrinsics=config.get("intrinsic_overrides", {}),
            runtime_imports=tuple(config.get("runtime_imports", ())),
            call_transforms=config.get("call_transforms", {}),
            function_transforms=config.get("function_transforms", {}),
            handle_producers=frozenset(config.get("handle_producers", ())),
            type_bound_procedures=frozenset(config.get("type_bound_procedures", ())),
            patches=config.get("patches", {}),
            emission=emission,
        )

        nodes = _subprogram_nodes(source)
        body, report, delegated = self._kernels(assembler, emission, nodes)
        body.extend(self._wrappers(assembler, emission))
        body.extend(f"{name} = _host.{name}" for name, _ in delegated)
        body.append("")
        body.append(f"_NJIT_KERNELS = {sorted(emission.kernels.names)!r}")
        body.append("")

        module = facts.interface["module"]
        text = self._header(module, source, companion_imports, config) + _adapt(
            "\n".join([*self._prelude(emission), "", *body])
        )
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={Path(f"{module}_njit.py"): text.encode()},
            deferred=[f"{name}: {reason}" for name, reason in delegated],
            notes={
                "blocks": report,
                "kernels": sorted(emission.kernels.names),
                "host_delegated": [name for name, _ in delegated],
                "anchor": f"{module}_numpy.py",
                "profile": assembler.profile.name,
            },
        )

    # -- assembly -------------------------------------------------------------

    def _kernels(
        self, assembler: Any, emission: Any, nodes: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]], list[tuple[str, str]]]:
        """Every eligible subprogram, or the reason it is delegated instead.

        A refusal anywhere in a body delegates the *whole* subprogram and
        removes it from the kernel set, which matters for what comes after: a
        caller emitted later must not go on referring to a kernel that does not
        exist. That is why the kernel set is mutable and why this loop follows
        the interface's order, as the pipeline's does.
        """
        from recast.transform.numba.backend import ineligible_reason
        from recast.transform.numpy.statements import REFUSED

        lines: list[str] = []
        report: list[dict[str, Any]] = []
        delegated: list[tuple[str, str]] = []
        for record in assembler.record["subprograms"]:
            name = record["name"]
            if name not in emission.kernels.names:
                delegated.append((name, ineligible_reason(record, assembler.externals)))
                continue
            node = nodes.get(name)
            if node is None:
                emission.kernels.names.discard(name)
                delegated.append((name, "[emit] no definition in the source"))
                continue
            try:
                emitted, blocks = assembler.render(node, name)
            except (*REFUSED, KeyError) as refusal:
                emission.kernels.names.discard(name)
                delegated.append((name, f"[emit] {refusal}"))
                continue
            lines.extend(emitted)
            report.extend(blocks)
        return lines, report, delegated

    @staticmethod
    def _wrappers(assembler: Any, emission: Any) -> list[str]:
        lines: list[str] = []
        for record in assembler.record["subprograms"]:
            if record["name"] in emission.kernels.names:
                lines.extend(assembler.wrapper(record))
        return lines

    @staticmethod
    def _prelude(emission: Any) -> list[str]:
        """The two module-level tables the kernels index into."""
        lines = [f"_ERRMSGS = [' '] + {emission.messages!r}"]
        for (name, arguments), constant in sorted(emission.folds.items(), key=lambda kv: kv[1]):
            lines.append(f"{constant} = _host._f_cfold('{name}', {arguments})")
        return lines

    @staticmethod
    def _header(
        module: str, source: Path, companion_imports: tuple[str, ...], config: dict[str, Any]
    ) -> str:
        stem = config.get("constants_stem", f"{module}_constants")
        pieces = [DOCSTRING.format(source=source.name), ""]
        pieces.append(_runtime_text())
        pieces.append("")
        pieces.append(f"from {stem} import *  # noqa: F401,F403")
        if config.get("use_constants"):
            use_stem = config.get("use_constants_stem", f"{module}_use_constants")
            pieces.append(f"from {use_stem} import *  # noqa: F401,F403")
        pieces.append(f"import {module}_numpy as _host")
        for line in companion_imports:
            # ``import micro_mg_utils_numpy as _mgu`` -> the host module the
            # wrapper reads state off, and the njit module its kernels call.
            imported, _, alias = line.partition(" as ")
            imported = imported[len("import ") :]
            prefix = alias.lstrip("_")
            pieces.append(f"import {imported} as _host_{prefix}")
            pieces.append(f"import {imported.replace('_numpy', '_njit')} as _{prefix}_nj")
        pieces.append("")
        return "\n".join(pieces) + "\n"


def _subprogram_nodes(source: Path) -> dict[str, Any]:
    """``{name: definition node}`` for the module being translated."""
    from recast.fortran._parse import f03, parse, walk
    from recast.fortran.frontend import _subprograms_of

    tree = parse(source)
    found = walk(tree, f03.Module)
    return dict(_subprograms_of(found[0] if found else tree))


def _runtime_text() -> str:
    """The shim library's source, read from disk rather than imported.

    Importing it would import numba, and emitting Numba code must not require
    numba to be installed -- the emitter is pure AST work. Same rule, and same
    reason, as ``recast.transform.jax.backend.emit_runtime``.
    """
    text = Path(__file__).with_name("runtime.py").read_text()
    return text[text.index("import math") :].rstrip() + "\n"


def _aliased(
    companion_imports: tuple[str, ...], records: tuple[dict[str, Any], ...]
) -> dict[str, dict[str, Any]]:
    """``alias -> record``, paired off the import lines the NumPy side built."""
    out: dict[str, dict[str, Any]] = {}
    for line, record in zip(companion_imports, records, strict=True):
        _, _, alias = line.partition(" as ")
        out[alias.strip()] = record
    return out


def _adapt(body: str) -> str:
    """The two text passes the pipeline applies to a finished njit module.

    Both are about what numba cannot compile rather than about the
    translation. ``_ext.gamma`` is the one external with a compiled
    counterpart, and ``Ellipsis`` indexing has no nopython form -- ``[:]``
    keeps the whole-array semantics the NumPy backend meant by ``[...]``.
    """
    return body.replace("_ext.gamma(", "math.gamma(").replace("[...]", "[:]")


def factory(**config: Any) -> NumbaTranslation:
    del config
    return NumbaTranslation()
