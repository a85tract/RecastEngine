"""The Python side of a flattened interface: ``<name>_flat`` in the emitted module.

Built from the same ``FlatPlan`` the oracle's Fortran adapter is
(``recast.oracle.flat``): the adapter takes the object's touched components
as flat arrays, builds the object out of them, sets the module state the
plan names, calls the translation, and returns the written arrays in the
flat signature's order. The gate then compares the two sides the way it
compares any flat subprogram, and a component whose translation reads the
wrong index is a difference it sees.
"""

from __future__ import annotations

import keyword

from recast.fortran.flatten import FlatPlan, signature

__all__ = ["python_adapter"]


def _py(name: str) -> str:
    """The emitted spelling of a dummy: a Python keyword gets its underscore."""
    return name + "_" if keyword.iskeyword(name) else name


def python_adapter(plans: list[FlatPlan]) -> str:
    """The ``<name>_flat`` functions and their ``_SIGNATURES`` entries,
    appended to the emitted module."""
    lines = [
        "",
        "",
        "# Flattened adapters for the differential gate (recast.transform.numpy.flat).",
        "class _Record:",
        "    def __init__(self, **fields):",
        "        self.__dict__.update(fields)",
        "",
    ]
    for plan in plans:
        args = plan.flat_args
        sub = plan.subprogram
        # The emitted convention: an OUT argument is returned, not taken.
        taken = [a["name"] for a in args if a["intent"] != "OUT"]
        lines.append(f"def {plan.name}({', '.join(taken)}):")
        for obj in plan.objects:
            fields = ", ".join(f"{c.name}={c.flat}" for c in obj.components)
            if obj.kind == "dummy":
                lines.append(f"    {obj.name} = _Record({fields})")
            else:
                # The module's own alias may not be in this file's header --
                # the state can be reached only through a callee -- so bind
                # it here; and a translated module may spell an unset state
                # variable as None, which cannot carry components.
                lines.append(f"    import {obj.module}_numpy as _{obj.module}")
                lines.append(
                    f"    if not hasattr(getattr(_{obj.module}, {obj.name!r}, None), '__dict__'):"
                )
                lines.append(f"        _{obj.module}.{obj.name} = _Record()")
                for comp in obj.components:
                    lines.append(f"    _{obj.module}.{obj.name}.{comp.name} = {comp.flat}")
        for state in plan.states:
            lines.append(f"    import {state.module}_numpy as _{state.module}")
            lines.append(f"    _{state.module}.{state.name} = {state.flat}")
        # The translation takes IN and INOUT dummies (optionals absent) and
        # returns every OUT/INOUT one in declaration order -- the engine's
        # emitted convention -- so call it by keyword and unpack by name.
        passed = [a for a in sub["args"] if a["intent"] != "OUT" and not a.get("optional")]
        # An intent(out) array of the original is the caller's buffer under
        # the frontend's convention: it is handed in as INOUT storage and
        # comes back in the return tuple like every output.
        passed += [a for a in sub["args"] if a["intent"] == "OUT" and a.get("dims")]
        outs = [a["name"] for a in sub["args"] if a["intent"] in ("OUT", "INOUT")]
        call_kwargs = ", ".join(f"{_py(a['name'])}={a['name']}" for a in passed)
        lines.append(f"    _out = {sub['name']}({call_kwargs})")
        if len(outs) == 1:
            lines.append("    _out = (_out,)")
        if outs:
            lines.append(f"    {', '.join(_py(n) + '_' for n in outs)}, = _out")
        for obj in plan.objects:
            if obj.kind == "dummy":
                for comp in obj.components:
                    if comp.written:
                        lines.append(f"    {comp.flat} = {obj.name}.{comp.name}")
        for state in plan.states:
            if state.written:
                lines.append(f"    {state.flat} = _{state.module}.{state.name}")
        returned = []
        for a in args:
            if a["intent"] in ("OUT", "INOUT"):
                if a["name"] in outs:
                    returned.append(_py(a["name"]) + "_")
                else:
                    returned.append(a["name"])  # a written component
        lines.append(f"    return {', '.join(returned) if returned else 'None'}")
        lines.append("")
    lines.append("_SIGNATURES.update({")
    for plan in plans:
        lines.append(f"    {plan.name!r}: {signature(plan)!r},")
    lines.append("})")
    return "\n".join(lines) + "\n"
