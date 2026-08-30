"""Python stand-ins for the modules a candidate imports but never ports.

The emitter binds every ``use`` of a non-companion module to an import --
``import abortutils_numpy as _abortutils`` -- so that a USE'd name never
comes out bare. When such a module is a *stub* under the frontend, nothing
translates it and nothing would put ``abortutils_numpy`` on the path; the
bit-exact gate then fails on the import before it compares a number. This
module writes the files those imports name, into the candidate itself, so
the candidate stays what the gate assumes it is: self-contained.

What a stand-in carries:

* every initialized module-level entity of the Fortran module, resolved
  from the tree with ``recast.fortran.use.resolve`` and rendered by the same
  ``use_constants_module`` the candidate's own use-constants come from --
  so ``clm_varcon_numpy.tfrz`` is the same parsed expression as ``TFRZ``,
  in both spellings;
* a ``_Record`` for every module variable of derived type, for whoever
  drives the translation to fill;
* for a framework module, whatever the caller's ``framework`` table says a
  standalone run answers -- ``endrun`` raises, the kinds are NumPy dtypes.
  That table is the one piece of knowledge here, and it is the caller's.

A name the tree initializes with something ``resolve`` cannot evaluate --
``selected_real_kind(12)``, a namelist default read at run time -- is
skipped, and the skip is recorded on the candidate. The stand-in is for the
import to succeed and for constants to agree; it is not a translation of
the framework, and it does not pretend to be one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from recast.fortran.tree import MODULE_DEFINITION, module_sources

__all__ = ["stand_ins"]

STUB_IMPORT = re.compile(r"^\s*import (?P<module>\w+)_numpy as (?P<alias>_\w+)\s*$", re.MULTILINE)


def _derived_state(path: Path, kinds: dict[str, str]) -> list[str]:
    """Module variables of derived type, by the engine's own interface
    record -- a regex over the text trips on the ``contains`` a type-bound
    procedure section puts inside the type definition."""
    from recast.fortran import interface

    try:
        record = interface.extract(path, kind_assumptions=kinds)
    except Exception:  # an unparsable sibling is not this stand-in's failure
        return []
    return [
        str(entry["name"]).lower()
        for entry in record.get("module_state", ())
        if "TYPE(" in str(entry.get("dtype", "")).upper()
    ]


def _module_file(module: str, root: Path) -> Path | None:
    for path in module_sources(root, frozenset({module})):
        if any(m.group(1).lower() == module for m in MODULE_DEFINITION.finditer(path.read_text())):
            return path
    return None


def _resolved_entities(path: Path, files: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    from recast.fortran.use import UnresolvedConstant, harvest, resolve

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for name in harvest(path):
        try:
            records = resolve([name], files)
        except UnresolvedConstant:
            skipped.append(name)
            continue
        except Exception as error:  # an initializer shape build() has no rule for
            skipped.append(f"{name} ({type(error).__name__})")
            continue
        for entry in records:
            if entry["name"] not in seen:
                seen.add(entry["name"])
                resolved.append(entry)
    return resolved, skipped


def stand_ins(
    emitted: str,
    root: Path,
    existing: set[str],
    *,
    modules: frozenset[str],
    framework: dict[str, str] | None = None,
    kind_assumptions: dict[str, str] | None = None,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    """Files for every ``import X_numpy`` in ``emitted`` that ``existing`` lacks.

    ``modules`` are the tree's stub and constants modules, whose files are
    the resolver's search path; ``framework`` maps a module name to the
    Python text a standalone run answers its calls with. Returns
    ``(files, report)``; the report says per module what was resolved and
    what was skipped, for ``Candidate.notes``.
    """
    from recast.transform.numpy.constants import use_constants_module

    framework = framework or {}
    kinds = kind_assumptions or {}
    files: dict[Path, bytes] = {}
    report: dict[str, Any] = {}
    search = module_sources(root, modules)
    for match in STUB_IMPORT.finditer(emitted):
        module = match.group("module").lower()
        filename = f"{module}_numpy.py"
        if filename in existing or filename in {str(p) for p in files}:
            continue
        path = _module_file(module, root)
        pieces = [
            f'"""Stand-in for Fortran module {module}, written by RecastEngine.',
            "",
            "Not a translation: the module is a stub under the frontend. Its",
            "initialized entities are resolved from the source tree; framework",
            'calls are answered the way a standalone run answers them."""',
            "",
            "import numpy as np  # noqa: F401",
            "",
            "",
            "class _Record:",
            '    """A module variable of derived type: components are set by whoever',
            '    drives the translation (the flat adapters, a harness)."""',
            "",
            "    def __init__(self, **fields):",
            "        self.__dict__.update(fields)",
            "",
        ]
        entry: dict[str, Any] = {"source": None, "resolved": [], "skipped": []}
        if path is not None:
            entry["source"] = path.name
            deps = [path, *[s for s in search if s != path]]
            resolved, skipped = _resolved_entities(path, deps)
            entry["resolved"] = [e["name"] for e in resolved]
            entry["skipped"] = skipped
            if resolved:
                body = use_constants_module(resolved, module).splitlines()
                # The shared renderer's header (docstring, numpy import) is
                # already above; keep only the constant lines after it.
                start = body.index("import numpy as np") + 1
                pieces.extend(ln for ln in body[start:] if ln)
                pieces.append("")
                pieces.extend(f"{e['name']} = {e['name'].upper()}" for e in resolved)
                pieces.append("")
            state = _derived_state(path, kinds)
            if state:
                pieces.extend(f"{name} = _Record()" for name in state)
                pieces.append("")
                entry["state"] = state
        if module in framework:
            pieces.append(framework[module])
            entry["framework"] = True
        files[Path(filename)] = ("\n".join(pieces) + "\n").encode()
        report[module] = entry
    return files, report
