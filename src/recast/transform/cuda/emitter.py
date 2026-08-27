"""The CUDA floors, subclassed from the Numba backend's.

Relayed from ``pipeline/cudaize.py``, whose ``CudaEmitter`` subclasses the
Numba emitter and changes four things. The parallel axis is the *input array
index* -- one thread per element -- because the module this was built for is a
library of scalar functions, so a device function stays at scalar rank and the
launcher above it does the mapping.

**Compile-time specialization is the whole idea.** A GPU kernel cannot branch
on whether an optional argument was supplied: there is no ``present()`` at run
time and nothing to test. So every subprogram with optional inputs is emitted
*twice* -- ``_<name>_kp`` with them and ``_<name>_ka`` without -- and
``present(x)`` folds to the literal ``True`` or ``False`` in each. numba's
dead-branch pruning then removes the unreachable side, and the absent variant
does not even have the parameter. An intra-module call picks the variant that
matches what it is forwarding: an actual naming an argument the absent variant
removed selects the absent variant of the callee.

Three smaller differences, each a consequence of the same thing:

* the state closure is **not expanded** here. A derived object never reaches a
  device function, so ``sorted(closure)`` is the whole rule and there is no
  ``own__obj__c`` flattening;
* there is no determinizing prologue, no error-code table and no patch
  handling. A device function is the numeric core and nothing else;
* the decorator is ``@cuda.jit(device=True, inline=True)``.

This backend cannot be bit-exact and does not pretend to be: ``math.*`` lowers
to CUDA's libdevice, which is one to two ULP from libm, so the gate is a
tolerance one -- the same ceiling, and for the same reason, as the JAX backend.

**Two places this does not follow upstream, both reported rather than copied.**
``cudaize.py`` was only ever run on one module, and ``tools/cuda_diff.py``
points it at twenty-seven:

1. *It raises on any module with an ``allocate``.* ``emit_kernel_variant``
   omits the per-subprogram state its ``emit_kernel`` sibling sets up --
   ``alloc_lb``, ``stmt_funcs``, ``cur_elemental``, the break-label scan --
   and the first subscript of a module allocatable then reads an attribute
   that does not exist. Reproduced from their own command line:
   ``python pipeline/cudaize.py src_fortran/mo_airmas.F90`` ends in
   ``AttributeError: 'CudaEmitter' object has no attribute 'alloc_lb'``. The
   differential counts these apart from differences, because a crash is not
   something an emitter can disagree with. Here the state is per-subprogram by
   construction, so there is nothing to omit.

2. *A generic call inside a device function emits a name the file does not
   define.* ``CudaEmitter.funcref`` tests the *generic* name against the
   kernel set, which never holds it, and falls through to a base branch that
   formats ``f"{name}(...)"`` directly instead of going through
   ``emit_local_fncall`` -- so ``rising_factorial(x, n)`` comes out as
   ``rising_factorial_r8(x, n)`` while the only thing emitted is
   ``_rising_factorial_r8_k``. Checked against their own output: the file
   defines ``_<name>_k*``, ``k_<name>_k*`` and ``<name>_v``, and never a bare
   ``<name>``. This is the shape of their open issue #4, and the resolution
   here is the same -- resolve the generic first, then call the kernel.
"""

from __future__ import annotations

from typing import Any

from recast.fortran.chunk import chunk_subprogram
from recast.fortran.interface import emit_name, subprogram_key
from recast.transform.numba.emitter import (
    NumbaExpressions,
    NumbaStatements,
    NumbaSubprograms,
)
from recast.transform.numpy.expressions import Expressions

__all__ = ["CudaExpressions", "CudaStatements", "CudaSubprograms", "optionals_of"]


def optionals_of(subprogram: dict[str, Any]) -> list[str]:
    """The optional *input* arguments a variant is specialized on."""
    return [
        a["name"]
        for a in subprogram["args"]
        if a["optional"] and a["intent"] in ("IN", "INOUT", "UNKNOWN")
    ]


def kernel_name(name: str, variant: str | None) -> str:
    """``_<name>_k``, or ``_kp``/``_ka`` when there is something to specialize."""
    return f"_{name}_k" if variant is None else f"_{name}_k{variant}"


class CudaExpressions(NumbaExpressions):
    """``present()`` as a literal, and calls that pick a variant."""

    def _present(self, argument: str) -> str:
        """Folded, not tested. Which variant is being emitted is the answer."""
        del argument
        assert self.emission is not None
        return "True" if self.emission.variant == "p" else "False"

    def _call(self, name: str, items: list[Any], arguments: list[str]) -> str | None:
        """An intra-module call, forwarding to the callee's matching variant.

        Deliberately skips the Numba floor's rule rather than extending it:
        upstream calls ``super(NjitEmitter, self).funcref``, one level past its
        own parent, so none of the closure-forwarding, companion or
        derived-flattening logic there applies. There are no companions in this
        backend and no derived arguments to flatten.
        """
        assert self.emission is not None
        kernels = self.emission.kernels
        if name not in kernels.names or name not in {
            s["name"] for s in kernels.record["subprograms"]
        }:
            return Expressions._call(self, name, items, arguments)

        callee = {s["name"]: s for s in kernels.record["subprograms"]}[name]
        formals = [a for a in callee["args"] if a["intent"] in ("IN", "INOUT", "UNKNOWN")]
        current = self.emission.subprogram or {}
        removed = set(optionals_of(current)) if self.emission.variant == "a" else set()
        positional: list[str] = []
        optional: list[str] = []
        variant = "a"
        for formal, actual in zip(formals, arguments, strict=False):
            if not formal["optional"]:
                positional.append(actual)
            elif actual in removed:
                variant = "a"  # forwarding an argument this variant does not have
            else:
                variant = "p"
                optional.append(actual)
        state = sorted(kernels.state_closure(name))
        callee_name = kernel_name(name, variant if optionals_of(callee) else None)
        return f"{callee_name}({', '.join(positional + state + optional)})"


class CudaStatements(NumbaStatements):
    """The Numba statement rules; a device function needs no others."""


class CudaSubprograms(NumbaSubprograms):
    """One device function per specialization."""

    expressions_class = CudaExpressions
    statements_class = CudaStatements

    def floors(self, name: str) -> Any:
        """The Numba floors, with ELEMENTAL turned off.

        A device function is at scalar rank whatever the Fortran said: one
        thread handles one element and the launcher above it does the mapping.
        The NumPy backend widens an ELEMENTAL body's transcendentals and its
        power operator to their array spellings, because *there* the body runs
        over array actuals -- here it never does, and upstream emits the scalar
        spelling too, though by omission rather than by decision:
        ``emit_kernel_variant`` never sets the flag its ``emit_kernel`` sibling
        does.
        """
        statements = super().floors(name)
        statements.expressions.elemental = False
        return statements

    def variants(self, subprogram: dict[str, Any]) -> list[str | None]:
        return ["p", "a"] if optionals_of(subprogram) else [None]

    def render(self, node: Any, name: str) -> tuple[list[str], list[dict[str, Any]]]:
        """Every variant of one subprogram, and the blocks they came from."""
        assert self.emission is not None
        lines: list[str] = []
        report: list[dict[str, Any]] = []
        for variant in self.variants({s["name"]: s for s in self.record["subprograms"]}[name]):
            emitted, blocks = self._variant(node, name, variant)
            lines.extend(emitted)
            report.extend(blocks)
        return lines, report

    def _variant(
        self, node: Any, name: str, variant: str | None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        assert self.emission is not None
        self.emission.variant = variant
        statements = self.floors(name)
        statements.scan(node)
        subprogram = statements.semantics.subprogram
        self.emission.subprogram = subprogram
        self.emission.error_args = set()
        self.emission.extra = set()

        lines = ["@cuda.jit(device=True, inline=True)", self.signature(subprogram)]
        lines.extend(self._result_initializer(subprogram, statements.semantics, statements))

        report: list[dict[str, Any]] = []
        for block, statement, span in chunk_subprogram(node):
            before = len(lines)
            lines.append(f"    # {block} <- L{span[0]}-L{span[1]}")
            lines.extend(statements.render(statement, 1))
            report.append(
                {
                    "subprogram": emit_name(subprogram),
                    "key": subprogram_key(subprogram),
                    "block": block,
                    "variant": variant,
                    "src_span": [int(span[0] or 0), int(span[1] or 0)],
                    "status": "mechanical",
                    "py_lines": [before, len(lines)],
                }
            )
        tail = statements.returned_value()
        lines.append(f"    return {tail}" if tail else "    return")
        lines += ["", ""]
        return lines, report

    def signature(self, subprogram: dict[str, Any]) -> str:
        """``def _<name>_k<v>(inputs..., state..., optionals...)``.

        The closure is ``sorted()`` and not expanded: no derived object reaches
        a device function, so there is nothing to flatten.
        """
        assert self.emission is not None
        positional = [
            a["name"]
            for a in subprogram["args"]
            if a["intent"] in ("IN", "INOUT", "UNKNOWN") and not a["optional"]
        ]
        state = sorted(self.emission.kernels.state_closure(subprogram["name"]))
        optionals = optionals_of(subprogram)
        variant = self.emission.variant
        supplied = optionals if variant == "p" else []
        name = kernel_name(subprogram["name"], variant if optionals else None)
        return f"def {name}({', '.join(positional + state + supplied)}):"
