"""The three floors, subclassed from the NumPy backend's.

Relayed from ``NjitEmitter`` in ``pipeline/numbaize.py``. Upstream it is one
subclass overriding thirteen methods of a single translator class; here those
thirteen land on three floors, because that is where this repository put them.
The rules are the same rules and ``tools/numba_diff.py`` holds them to the
original byte for byte.

Everything below is one of two kinds of change, and it is worth keeping them
apart when reading:

**Spelling, because the code is compiled.** ``np.size``'s axis argument does
not compile, so a kernel reads the shape tuple. Boolean setitem on an N-D
array does not compile, so a masked assignment becomes a full-slice
``np.where`` blend. ``mpmath`` cannot run inside a kernel, so a compile-time
fold is hoisted to a module-level constant the host evaluates at import --
numba freezes scalar float globals at compile time, which is exactly what is
wanted for a constant and exactly what is not wanted for module state.

**Structure, because a kernel cannot see module state.** Every call to another
kernel has to forward that callee's state closure, so a call site is rewritten
to ``_callee_k(actuals..., closure..., keywords...)`` -- and the closure sits
*between* the positional and keyword parts, which is why an optional actual
must be passed by keyword or the two would slide over each other. A derived
type has no nopython representation, so its arguments, locals and module state
are flattened one parameter per component. A CHARACTER ``intent(out)``
argument cannot be written in a kernel at all, so writes to one become an
integer error code and the host wrapper decodes it back.

Anything outside that subset raises ``NoRule``, which ``NumbaSubprograms``
catches for the whole subprogram: the unit is host-delegated to the validated
NumPy module rather than half-emitted. That is a coarser granularity than the
NumPy backend's per-block deferral, and deliberately so -- half a kernel is
not a thing that can run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03, walk
from recast.fortran.chunk import chunk_subprogram
from recast.fortran.interface import emit_name, subprogram_key
from recast.transform.numba.backend import DERIVED, Kernels, derived_components
from recast.transform.numpy.expressions import Expressions
from recast.transform.numpy.names import Names
from recast.transform.numpy.statements import Statements
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.rules import NoRule

__all__ = [
    "Emission",
    "NumbaExpressions",
    "NumbaNames",
    "NumbaStatements",
    "NumbaSubprograms",
]

COMPANION_PARAMETER = re.compile(r"(_\w+)\.([A-Z][A-Z0-9_]*)")
COMPANION_STATE = re.compile(r"(_\w+)\.([a-z]\w*)")

NUMBA_DTYPES = {"float64": "f8", "int32": "i4"}
"""Numba's signature spelling for the scalar types a ufunc may be built over."""


@dataclass
class Emission:
    """What the three floors share for the length of one module.

    The floors are rebuilt per subprogram -- that is the NumPy backend's design
    and this one keeps it -- but a kernel is not a self-contained thing the way
    a translated function is. Its callers need to know its closure, its
    compile-time folds are module-level constants numbered in emission order,
    and its error codes index one module-level table. All of that outlives the
    subprogram, so it lives here and every floor is handed the same instance.
    """

    kernels: Kernels

    folds: dict[tuple[str, str], str] = field(default_factory=dict)
    """``(intrinsic, rendered arguments)`` -> ``_CF_<n>``.

    A constant fold is done by ``mpmath`` at the precision the reference
    compiler folds at, which cannot happen inside a compiled kernel. The value
    is identical either way; what changes is who computes it and when, so it
    becomes a module-level name the host evaluates once at import.
    """

    messages: list[str] = field(default_factory=list)
    """CHARACTER values written to an ``intent(out)`` argument, in the order
    they were seen. A kernel writes the 1-based index; the wrapper looks it up
    in ``_ERRMSGS``, whose zeroth entry is the blank string."""

    subprogram: dict[str, Any] | None = None
    """The record being emitted, for the floors that need to ask about it."""

    extra: set[str] = field(default_factory=set)
    """Closure entries for the current subprogram that its ``calls`` list does
    not mention: companion module state and companion procedures reached by
    name rather than by call, found by walking the body for names."""

    error_args: set[str] = field(default_factory=set)
    """This subprogram's CHARACTER ``intent(out)`` arguments."""

    variant: str | None = None
    """Which compile-time specialization is being emitted, for a backend that
    has them. ``None`` for Numba, which does not; ``"p"``/``"a"`` for CUDA,
    where ``present()`` folds to a literal rather than being tested."""

    per_subprogram_extra: dict[str, set[str]] = field(default_factory=dict)
    """``extra`` for each subprogram already emitted, because the wrapper pass
    runs after every kernel and would otherwise see only the last one's."""

    def fold(self, name: str, arguments: str) -> str:
        key = (name, arguments)
        if key not in self.folds:
            self.folds[key] = f"_CF_{len(self.folds)}"
        return self.folds[key]

    def closure_of(self, name: str) -> list[str]:
        """The full closure for one subprogram, expanded into parameters."""
        return self.kernels.expand(
            set(self.kernels.state_closure(name)) | self.per_subprogram_extra.get(name, set())
        )


# --------------------------------------------------------------- expressions


@dataclass
class NumbaExpressions(Expressions):
    """Expression spellings a compiled kernel needs, and kernel call sites."""

    emission: Emission | None = None

    _constructor_depth: int = 0
    """Whether rendering is inside a reference fparser read as a constructor.

    The pipeline has two entry points here and overrides only one of them; the
    split this repository made funnels both into ``_call``, so the entry has to
    be recorded to tell them apart. See ``_call``.
    """

    def _structure_constructor(self, node: Any) -> str:
        self._constructor_depth += 1
        try:
            return super()._structure_constructor(node)
        finally:
            self._constructor_depth -= 1

    # -- spelling -------------------------------------------------------------

    def extent_along(self, name: str, axis: int) -> str:
        """``np.size`` compiles; its ``axis`` argument does not."""
        return f"{name}.shape[{axis}]"

    def axis_reduction(self, spelling: str, array: str, dimension: str) -> str:
        if spelling == "np.size":
            return f"{array}.shape[({dimension}) - 1]"
        return super().axis_reduction(spelling, array, dimension)

    def _constant_fold(self, name: str, items: list[Any]) -> str | None:
        folded = super()._constant_fold(name, items)
        if folded is None:
            return None
        # ``_f_cfold('exp', x)`` -> the module-level constant standing for it.
        inner = folded[len(f"_f_cfold('{name}', ") : -1]
        assert self.emission is not None
        return self.emission.fold(name, inner)

    # -- names ----------------------------------------------------------------

    def _data_ref(self, node: Any) -> str:
        """A derived argument or local reaches a kernel already flattened."""
        if len(node.children) == 2 and isinstance(node.children[0], f03.Name):
            root = str(node.children[0]).lower()
            flattened = self._flattened_locals()
            if root in flattened:
                tail = node.children[1]
                if isinstance(tail, f03.Name):
                    return f"{root}__{str(tail).lower()}"
                if isinstance(tail, f03.Part_Ref):
                    component = str(tail.children[0]).lower()
                    subscripts = self._subscripts(tail.children[1])
                    return f"{root}__{component}[{subscripts}]"
        return super()._data_ref(node)

    def _subscripts(self, arglist: Any) -> str:
        from recast.transform.rules import indexing

        positions = indexing.describe(arglist, None, rank_of=self.semantics.rank)
        return ", ".join(self._position(p) for p in positions)

    def _flattened_locals(self) -> dict[str, list[str]]:
        """Derived-type arguments and locals, each with its component names."""
        assert self.emission is not None
        record = self.emission.kernels.record
        subprogram = self.semantics.subprogram
        out = dict(derived_components(record, subprogram))
        types = record.get("types", {})
        for local in subprogram.get("locals") or ():
            match = DERIVED.match(str(local.get("dtype")))
            if match:
                out[local["name"]] = list(types.get(match.group(1).lower(), {}))
        return out

    # -- calls ----------------------------------------------------------------

    def _call(self, name: str, items: list[Any], arguments: list[str]) -> str | None:
        """A call to another kernel carries that kernel's state closure.

        Except on the constructor path, where an ELEMENTAL callee with an
        array actual broadcasts and the ordinary ``_f_ecall`` is emitted
        instead.

        The two paths really do differ upstream, and reproducing that is the
        point. A plainly-parsed reference reaches ``funcref``, which the Numba
        emitter overrides outright, so the kernel wins with no broadcast test.
        A reference fparser could not tell from a derived-type constructor
        reaches a *different* branch, which is not overridden -- and that one
        tests ``_needs_ecall`` before calling ``emit_local_fncall``. So
        ``var_coef_r8(relvar(:), 2.47_r8)`` broadcasts while
        ``var_coef_integer(relvar, 2)`` does not, in the same module, and the
        reason is which node the parser built.
        """
        assert self.emission is not None
        kernels = self.emission.kernels

        if self._constructor_depth and self._broadcast_target(name, items):
            return super()._call(name, items, arguments)

        if name in self.semantics.companion_generics:
            resolved = self.semantics.dispatch(name, items)
            remote = self.remotes.get(resolved)
            if remote is not None and resolved in kernels.companion_kernels(remote.alias):
                return self._companion_reference(remote, items)
            raise NoRule(f"companion generic function {name!r} is not njit-eligible")

        remote = self.remotes.get(name)
        if remote is not None:
            if remote.name in kernels.companion_kernels(remote.alias):
                return self._companion_reference(remote, items)
            raise NoRule(f"companion {name!r} is not njit-eligible")

        if name in kernels.names and name in {
            s["name"] for s in kernels.record["subprograms"]
        }:
            return self._kernel_reference(name, items)
        return super()._call(name, items, arguments)

    def _broadcast_target(self, name: str, items: list[Any]) -> bool:
        """Whether this call is an ELEMENTAL one over an array actual."""
        assert self.emission is not None
        record = self.semantics.procedures.get(name)
        if record is None:
            remote = self.remotes.get(name)
            if remote is not None:
                record = self.emission.kernels.companion_subprogram(remote.alias, remote.name)
        return record is not None and self._broadcasts(record, items)

    def _companion_reference(self, remote: Any, items: list[Any]) -> str:
        prefix = remote.alias.lstrip("_")
        assert self.emission is not None
        positional = [
            self.render(item) for item in items if not isinstance(item, f03.Actual_Arg_Spec)
        ]
        closure = self.emission.kernels.expand(
            {
                f"{prefix}__{n}"
                for n in self.emission.kernels.companion_closure(remote.alias, remote.name)
            }
        )
        return f"_{prefix}_nj._{remote.name}_k({', '.join(positional + closure)})"

    def _kernel_reference(self, name: str, items: list[Any]) -> str:
        """``f(a, b)`` -> ``_f_k(a, b, <state...>, optional=...)``.

        The closure sits between the positional and keyword parts, which is
        why an optional actual has to be passed by keyword: it would otherwise
        land on a state parameter.
        """
        assert self.emission is not None
        kernels = self.emission.kernels
        callee = {s["name"]: s for s in kernels.record["subprograms"]}[name]
        callee_derived = derived_components(kernels.record, callee)
        # Arguments only -- deliberately not ``_flattened_locals()``. The
        # pipeline builds this set from the subprogram's *arguments* here and
        # from arguments *plus locals* in the two other places it asks the
        # same question (a call statement, a component reference). So a
        # derived local passed to a kernel from an expression is refused and
        # the subprogram delegates, where the same actual in a call statement
        # is accepted. Reproduced rather than tidied: it is their rule, and
        # widening it here would emit a kernel call they never emit.
        mine = derived_components(kernels.record, self.semantics.subprogram)
        formals = [
            a
            for a in callee["args"]
            if a["intent"] in ("IN", "INOUT", "UNKNOWN")
            and not Statements.is_optional_output(a)
        ]
        positional: list[str] = []
        keyword: list[str] = []
        # Not strict: a trailing optional formal with no actual is normal, and
        # the pipeline stops at the shorter list here too.
        for formal, actual in zip(formals, items, strict=False):
            if formal["name"] in callee_derived:
                positional.extend(
                    self._derived_actual(actual, callee_derived[formal["name"]], mine)
                )
                continue
            rendered = self.render(actual)
            if formal["optional"]:
                keyword.append(f"{formal['name']}={rendered}")
            else:
                positional.append(rendered)
        closure = kernels.expand(kernels.state_closure(name))
        return f"_{name}_k({', '.join(positional + closure + keyword)})"

    def _derived_actual(
        self, actual: Any, components: list[str], mine: dict[str, list[str]]
    ) -> list[str]:
        """A derived actual is passed as the components it was flattened into."""
        assert self.emission is not None
        root = str(actual).lower()
        if root in mine:
            return [f"{root}__{c}" for c in components]
        if any(f"own__{root}__" in entry for entry in self.emission.extra):
            return [f"own__{root}__{c}" for c in components]
        raise NoRule(f"derived actual {root!r} is not a flattened kernel argument")


# --------------------------------------------------------------------- names


@dataclass
class NumbaNames(Names):
    """Companion references, as a kernel has to spell them.

    A translated module reaches a sibling's globals through its import alias --
    ``_wv.omeps``. A kernel cannot: the alias is a module object, and reading
    an attribute off one at run time is exactly the frozen-global problem the
    closure exists to avoid. So a companion's *parameters* inline as bare
    constants (its constants file is star-imported into the generated module,
    so the name is already in scope) and its *state* becomes a tagged closure
    parameter the host wrapper fills.
    """

    companion_aliases: frozenset[str] = frozenset()

    def symbol(self, name: str) -> str:
        out = super().symbol(name)
        parameter = COMPANION_PARAMETER.fullmatch(out)
        if parameter is not None and parameter.group(1) in self.companion_aliases:
            return parameter.group(2)
        state = COMPANION_STATE.fullmatch(out)
        if state is not None and state.group(1) in self.companion_aliases:
            return f"{state.group(1).lstrip('_')}__{state.group(2)}"
        return out


# ---------------------------------------------------------------- statements


@dataclass
class NumbaStatements(Statements):
    """Statement rules a compiled kernel needs."""

    emission: Emission | None = None

    def render(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        if isinstance(node, f03.Assignment_Stmt):
            lhs, _, rhs = node.children
            assert self.emission is not None
            if isinstance(lhs, f03.Name) and str(lhs).lower() in self.emission.error_args:
                return [self._error_code(pad, str(lhs).lower(), rhs)]
            if isinstance(lhs, f03.Name) and isinstance(rhs, f03.Name):
                copied = self._derived_copy(pad, str(lhs).lower(), str(rhs).lower())
                if copied is not None:
                    return copied
        return super().render(node, indent)

    def _error_code(self, pad: str, name: str, rhs: Any) -> str:
        """A write to a CHARACTER ``intent(out)`` argument becomes an index.

        Nothing in nopython mode can hold the string, so the kernel writes the
        1-based position of the message in a module-level table and the host
        wrapper looks it up. A blank message is code zero, which is why that
        table's zeroth entry is a single space.
        """
        assert self.emission is not None
        if not isinstance(rhs, f03.Char_Literal_Constant):
            raise NoRule(f"non-literal write to the CHARACTER out-argument {name!r}")
        text = str(rhs)[1:-1]
        if text.strip() == "":
            index = 0
        else:
            self.emission.messages.append(text)
            index = len(self.emission.messages)
        return f"{pad}_errflag_{name} = {index}  # str-OUT -> error code"

    def _derived_copy(self, pad: str, left: str, right: str) -> list[str] | None:
        """``a = b`` between derived types is a componentwise deep copy.

        Fortran's intrinsic assignment on a derived type copies every
        component. The NumPy backend can lean on the namespace object; a kernel
        has only the flattened components, so the copy is spelled out -- and an
        array component copies *into* its storage rather than rebinding the
        name, because the caller shares that storage.
        """
        assert self.emission is not None
        record = self.emission.kernels.record
        subprogram = self.semantics.subprogram
        types = record.get("types", {})
        locals_derived: dict[str, list[str]] = {}
        specification: dict[str, Any] = {}
        for local in subprogram.get("locals") or ():
            match = DERIVED.match(str(local.get("dtype")))
            if not match:
                continue
            locals_derived[local["name"]] = list(types.get(match.group(1).lower(), {}))
            if local["name"] == left:
                specification = types.get(match.group(1).lower(), {})
        flattened = {**derived_components(record, subprogram), **locals_derived}
        if left not in locals_derived or right not in flattened:
            return None
        lines = []
        for component in locals_derived[left]:
            if specification.get(component, {}).get("dims"):
                lines.append(f"{pad}{left}__{component}[...] = {right}__{component}")
            else:
                lines.append(f"{pad}{left}__{component} = {right}__{component}")
        return lines

    def _masked_assignment(self, mask: str, node: Any, indent: int) -> str:
        """WHERE, without N-D boolean setitem.

        numba has no ``a[mask] = v`` for a multidimensional array, so the
        assignment becomes a whole-slice blend. Both sides are evaluated, which
        is what Fortran's WHERE means for a pure right-hand side -- and every
        right-hand side that reaches here is pure.
        """
        pad = "    " * indent
        lhs, _, rhs = node.children
        target = self.target(lhs)
        value = self.expressions.render(rhs)
        if target.endswith("[...]"):
            base = target[: -len("[...]")]
            written = f"{base}[:]"
        else:
            base = written = target
        return f"{pad}{written} = np.where({mask}, {value}, {base})"

    def returned_value(self) -> str:
        base = super().returned_value()
        assert self.emission is not None
        if not self.emission.error_args:
            return base
        parts = [p.strip() for p in base.split(",")] if base else []
        return ", ".join(f"_errflag_{p}" if p in self.emission.error_args else p for p in parts)

    # -- call statements ------------------------------------------------------

    def _call(self, node: Any, indent: int) -> list[str]:
        assert self.emission is not None
        kernels = self.emission.kernels
        name = str(node.children[0]).lower()
        items = self._actual_items(node)
        if name in self.semantics.generics:
            name = self.semantics.dispatch(name, items)

        if name in self.semantics.companion_generics:
            resolved = self.semantics.dispatch(name, items)
            remote = self.expressions.remotes.get(resolved)
            if remote is None or resolved not in kernels.companion_kernels(remote.alias):
                raise NoRule(f"companion generic {name!r} is not njit-eligible")
            return self._companion_call(node, indent, remote)

        remote = self.expressions.remotes.get(name)
        if remote is not None:
            if remote.name not in kernels.companion_kernels(remote.alias):
                raise NoRule(f"companion {name!r} is not njit-eligible")
            return self._companion_call(node, indent, remote)

        if name not in kernels.names or name not in {
            s["name"] for s in kernels.record["subprograms"]
        }:
            return super()._call(node, indent)

        callee = {s["name"]: s for s in kernels.record["subprograms"]}[name]
        inputs, outputs = self._bind_actuals(node, callee, kernels.record, remote=None)
        closure = kernels.expand(kernels.state_closure(name))
        return self._call_line(indent, f"_{name}_k", inputs, closure, outputs)

    def _companion_call(self, node: Any, indent: int, remote: Any) -> list[str]:
        assert self.emission is not None
        kernels = self.emission.kernels
        callee = kernels.companion_subprogram(remote.alias, remote.name)
        if callee is None:
            raise NoRule(f"companion {remote.name!r} has no interface record")
        inputs, outputs = self._bind_actuals(node, callee, kernels.record, remote=remote)
        prefix = remote.alias.lstrip("_")
        closure = kernels.expand(
            {f"{prefix}__{n}" for n in kernels.companion_closure(remote.alias, remote.name)}
        )
        return self._call_line(indent, f"_{prefix}_nj._{remote.name}_k", inputs, closure, outputs)

    @staticmethod
    def _call_line(
        indent: int, callee: str, inputs: list[str], closure: list[str], outputs: list[str]
    ) -> list[str]:
        """The state parameters sit between the positional and keyword parts."""
        keyword = [a for a in inputs if "=" in a.split("(")[0]]
        positional = [a for a in inputs if a not in keyword]
        call = f"{callee}({', '.join(positional + closure + keyword)})"
        pad = "    " * indent
        if outputs:
            return [f"{pad}{', '.join(outputs)} = {call}"]
        return [f"{pad}{call}"]

    @staticmethod
    def _actual_items(node: Any) -> list[Any]:
        arglist = node.children[1]
        if arglist is None:
            return []
        return list(arglist.children) if hasattr(arglist, "children") else [arglist]

    def _bind_actuals(
        self, node: Any, callee: dict[str, Any], record: dict[str, Any], remote: Any
    ) -> tuple[list[str], list[str]]:
        """Pair actuals to formals, then split them the way a kernel wants.

        Fortran passes everything by reference, so an ``intent(out)`` actual is
        a *target* rather than an argument -- the kernel returns it and the call
        site assigns it back. An ``intent(inout)`` is both. Keyword actuals are
        placed by name first, because a kernel's parameters are not in the
        Fortran's order.
        """
        assert self.emission is not None
        formals = callee["args"]
        actuals: list[Any] = [None] * len(formals)
        names = [f["name"] for f in formals]
        position = 0
        for item in self._actual_items(node):
            if isinstance(item, f03.Actual_Arg_Spec):
                keyword = str(item.children[0]).lower()
                if keyword not in names:
                    raise NoRule(f"keyword actual {keyword!r} names no dummy")
                actuals[names.index(keyword)] = item.children[1]
            else:
                actuals[position] = item
                position += 1

        callee_derived = derived_components(record, callee)
        inputs: list[str] = []
        outputs: list[str] = []
        for formal, actual in zip(formals, actuals, strict=True):
            optional_output = Statements.is_optional_output(formal)
            if actual is None:
                if not formal["optional"]:
                    raise NoRule("a required actual is missing")
                if optional_output:
                    outputs.append("_")
                continue
            if optional_output:
                inputs.append(f"want_{formal['name']}=True")
            if formal["intent"] in ("IN", "INOUT", "UNKNOWN") and not optional_output:
                inputs.extend(self._input_actual(formal, actual, callee_derived, remote))
            if formal["intent"] in ("OUT", "INOUT"):
                outputs.append(self._output_actual(actual))
        return inputs, outputs

    def _input_actual(
        self, formal: dict[str, Any], actual: Any, callee_derived: dict[str, list[str]], remote: Any
    ) -> list[str]:
        """One actual, as the parameters the kernel declared for it."""
        if remote is None:
            if formal["name"] in callee_derived:
                return self._flattened_local(actual, callee_derived[formal["name"]])
        elif "UNKNOWN(TYPE" in str(formal["dtype"]):
            return self._flattened_remote(actual, callee_derived.get(formal["name"], []), remote)
        rendered = self.expressions.render(actual)
        return [f"{formal['name']}={rendered}" if formal["optional"] else rendered]

    def _flattened_local(self, actual: Any, components: list[str]) -> list[str]:
        """A derived actual to an in-module kernel.

        Its own flattened arguments and locals first, then this module's
        derived state -- which reaches the kernel as ``own__obj__c`` closure
        parameters, so an actual naming one is already in scope under that
        spelling. Either way the *callee's* component list decides the order,
        because those are its parameters.
        """
        assert self.emission is not None
        root = str(actual).lower()
        if root in self._flattened_here():
            return [f"{root}__{c}" for c in components]
        if any(f"own__{root}__" in entry for entry in self.emission.extra):
            return [f"own__{root}__{c}" for c in components]
        raise NoRule(f"derived actual {root!r} is not a flattened kernel argument")

    def _flattened_remote(
        self, actual: Any, components: list[str], remote: Any
    ) -> list[str]:
        """The same, to a companion's kernel.

        The order of the three tests is the pipeline's and is not the order
        above: this module's own derived state is checked *first* here. A
        companion global is the third case and the one that has no counterpart
        in the in-module path -- ``mg_liq_props`` living in the companion, read
        by a kernel in this module, passed on to another of the companion's.
        """
        assert self.emission is not None
        kernels = self.emission.kernels
        root = str(actual).lower()
        if any(f"own__{root}__" in entry for entry in self.emission.extra):
            return [f"own__{root}__{c}" for c in components]
        mine = self._flattened_here()
        if root in mine:
            return [f"{root}__{c}" for c in mine[root]]
        if root in self.names.companion_globals:
            prefix = remote.alias.lstrip("_")
            companion = kernels.companion_derived_state(remote.alias).get(root)
            if companion is None:
                raise NoRule(f"companion global {root!r} has no type information")
            return [f"{prefix}__{root}__{c}" for c in companion]
        raise NoRule(f"derived actual {root!r} is not flattened")

    def _flattened_here(self) -> dict[str, list[str]]:
        assert self.emission is not None
        record = self.emission.kernels.record
        subprogram = self.semantics.subprogram
        out = dict(derived_components(record, subprogram))
        types = record.get("types", {})
        for local in subprogram.get("locals") or ():
            match = DERIVED.match(str(local.get("dtype")))
            if match:
                out[local["name"]] = list(types.get(match.group(1).lower(), {}))
        return out

    def _output_actual(self, actual: Any) -> str:
        """Where a returned out-argument is written back."""
        if isinstance(actual, f03.Name):
            name = str(actual).lower()
            emitted = self.names.symbol(name)
            return f"{emitted}[...]" if self.semantics.is_array(name) else emitted
        if isinstance(actual, f03.Part_Ref):
            base = str(actual.children[0]).lower()
            if self.semantics.is_array(base):
                return self.expressions.subscript(base, actual.children[1])
        raise NoRule("an out-argument actual that is not a variable")


# --------------------------------------------------------------- subprograms


@dataclass
class NumbaSubprograms(Subprograms):
    """One kernel, and the host wrapper that keeps its public signature.

    Assembly here is not the NumPy backend's, and the difference is not
    stylistic. That one renders every block, turning a refusal into a
    ``raise NotImplementedError`` the agent layer later fills; this one lets a
    refusal escape, because half a kernel cannot run and the honest answer is
    to delegate the whole subprogram to the NumPy module. There are also no
    goto regions and no DATA blocks: a kernel that needed either is outside the
    subset by the time it gets here.
    """

    expressions_class = NumbaExpressions
    statements_class = NumbaStatements

    emission: Emission | None = None

    def floors(self, name: str) -> Statements:
        statements = super().floors(name)
        statements.emission = self.emission  # type: ignore[attr-defined]
        statements.expressions.emission = self.emission  # type: ignore[attr-defined]
        aliases = frozenset(self.emission.kernels.aliases()) if self.emission else frozenset()
        statements.names = _as_numba_names(statements.names, aliases)
        statements.expressions.names = statements.names
        return statements

    # -- the kernel -----------------------------------------------------------

    def render(self, node: Any, name: str) -> tuple[list[str], list[dict[str, Any]]]:
        """The compiled kernel for one subprogram, and its block report."""
        assert self.emission is not None
        statements = self.floors(name)
        statements.scan(node)
        semantics = statements.semantics
        subprogram = semantics.subprogram

        self.emission.subprogram = subprogram
        self.emission.error_args = {
            a["name"] for a in subprogram["args"] if a["dtype"] == "str" and a["intent"] == "OUT"
        }
        self.emission.extra = self._referenced_closure(node)
        self.emission.per_subprogram_extra[name] = set(self.emission.extra)

        lines = [self._decorator(subprogram), self.signature(subprogram)]
        span = subprogram["line_span"]
        lines.append(
            f'    """@njit kernel for {subprogram["name"]} '
            f'(L{span[0]}-L{span[1]}); state closure passed explicitly."""'
        )
        lines.extend(self._result_initializer(subprogram, semantics, statements))
        lines.extend(self._prologue(subprogram, semantics, statements))
        for argument in sorted(self.emission.error_args):
            lines.append(f"    _errflag_{argument} = 0")

        report: list[dict[str, Any]] = []
        for block, statement, block_span in chunk_subprogram(node):
            entry: dict[str, Any] = {
                "subprogram": emit_name(subprogram),
                "key": subprogram_key(subprogram),
                "block": block,
                "src_span": [int(block_span[0] or 0), int(block_span[1] or 0)],
            }
            before = len(lines)
            patch = self.patches.get(f"{subprogram['name']}/{block}")
            if patch is not None:
                lines.append(
                    f"    # {block} <- L{block_span[0]}-L{block_span[1]} "
                    f"AGENT-PATCHED (njit-adapted)"
                )
                lines.extend("    " + line for line in _njit_patch(patch["python"]))
                entry["status"] = "agent_patched"
            else:
                lines.append(f"    # {block} <- L{block_span[0]}-L{block_span[1]}")
                lines.extend(statements.render(statement, 1))
                entry["status"] = "mechanical"
            entry["py_lines"] = [before, len(lines)]
            report.append(entry)

        tail = statements.returned_value()
        lines.append(f"    return {tail}" if tail else "    return")
        lines += ["", ""]
        return lines, report

    def _result_initializer(
        self, subprogram: dict[str, Any], semantics: Any, statements: Statements
    ) -> list[str]:
        """A function result's determinizing value, which is not the NumPy
        backend's.

        That one allocates: an array-valued result gets ``np.zeros`` at its
        declared shape, a LOGICAL gets ``False``, a derived type gets its
        factory. This one writes a scalar zero for the three numeric kinds and
        nothing at all for anything else -- so an array-valued function starts
        its result as ``0.0`` and a LOGICAL one starts it undefined.

        Relayed as it stands. It is the pipeline's rule, the kernels it
        produces are what the head-to-head arms have been run against, and the
        standing rule here is that a difference from it is a bug in the
        migration until shown otherwise.
        """
        del semantics, statements
        if subprogram["kind"] != "function":
            return []
        if subprogram["result_dtype"] in ("float64", "float32"):
            return [f"    {subprogram['result']} = 0.0"]
        if subprogram["result_dtype"] == "int32":
            return [f"    {subprogram['result']} = 0"]
        return []

    def _decorator(self, subprogram: dict[str, Any]) -> str:
        """``@vectorize`` where Fortran said ELEMENTAL, ``@njit`` otherwise.

        An ELEMENTAL function over scalars with an empty state closure is
        exactly a ufunc: numba builds one that takes scalar or array actuals,
        which is what ELEMENTAL means. Anything else is a plain kernel.
        """
        if not self._vectorizable(subprogram):
            return '@njit(cache=True, fastmath=False, error_model="numpy")'
        result = NUMBA_DTYPES[subprogram["result_dtype"]]
        arguments = ", ".join(NUMBA_DTYPES[a["dtype"]] for a in subprogram["args"])
        signatures = [f'"{result}({arguments})"']
        if "i4" in arguments:
            # A Python int literal arrives as i8, so the i4 signature alone
            # would fail to match at the call site.
            signatures.append(f'"{result}({arguments.replace("i4", "i8")})"')
        return f'@vectorize([{", ".join(signatures)}], nopython=True)'

    def _vectorizable(self, subprogram: dict[str, Any]) -> bool:
        assert self.emission is not None
        if subprogram["kind"] != "function":
            return False
        if not any("ELEMENTAL" in str(p).upper() for p in (subprogram.get("prefixes") or ())):
            return False
        if self.emission.kernels.state_closure(subprogram["name"]):
            return False
        if subprogram["result_dtype"] not in NUMBA_DTYPES:
            return False
        return all(
            a["dtype"] in NUMBA_DTYPES and not a.get("dims") and not a["optional"]
            for a in subprogram["args"]
        )

    def signature(self, subprogram: dict[str, Any]) -> str:
        """``def _<name>_k(actuals..., state..., optionals..., wants...)``."""
        assert self.emission is not None
        derived = derived_components(self.emission.kernels.record, subprogram)
        positional: list[str] = []
        keyword: list[str] = []
        wants: list[str] = []
        for argument in subprogram["args"]:
            if Statements.is_optional_output(argument):
                wants.append(f"want_{argument['name']}=False")
            elif argument["intent"] in ("IN", "INOUT", "UNKNOWN"):
                if argument["name"] in derived:
                    positional.extend(f"{argument['name']}__{c}" for c in derived[argument["name"]])
                else:
                    (keyword if argument["optional"] else positional).append(argument["name"])
        state = self.emission.kernels.expand(
            set(self.emission.kernels.state_closure(subprogram["name"])) | self.emission.extra
        )
        parts = positional + state + [f"{k}=None" for k in keyword] + wants
        return f"def _{subprogram['name']}_k({', '.join(parts)}):"

    def _prologue(
        self, subprogram: dict[str, Any], semantics: Any, statements: Statements
    ) -> list[str]:
        """The NumPy prologue, with derived locals flattened rather than made.

        A ``_make_<type>()`` factory returns a namespace object, which cannot
        enter nopython mode -- so those lines are dropped and one slot per
        component takes their place.
        """
        lines = super()._prologue(subprogram, semantics, statements)
        assert self.emission is not None
        record = self.emission.kernels.record
        types = record.get("types", {})
        flattened: dict[str, str] = {}
        for local in subprogram.get("locals") or ():
            match = DERIVED.match(str(local.get("dtype")))
            if match is not None:
                flattened[local["name"]] = match.group(1).lower()
        lines = [
            line
            for line in lines
            if not any(
                f"{name} = _make_" in line or f"{name} = _new_" in line for name in flattened
            )
        ]
        for name, type_name in flattened.items():
            specification = types.get(type_name, {})
            for component, spec in specification.items():
                if spec.get("dims"):
                    if any(d.get("ub") is None for d in spec["dims"]):
                        # A deferred or assumed extent: there is no shape to
                        # give np.empty. The NumPy backend can defer the
                        # component to the agent queue, a kernel cannot -- so
                        # the whole subprogram delegates rather than guess.
                        raise NoRule(
                            f"derived local {name!r} has a component with no static extent"
                        )
                    shape = ", ".join(
                        statements.expressions.bound(d["ub"]) for d in spec["dims"]
                    )
                    lines.append(
                        f"    {name}__{component} = np.empty(({shape},), dtype=np.float64)"
                    )
                else:
                    lines.append(f"    {name}__{component} = 0.0")
        return lines

    # -- the host wrapper -----------------------------------------------------

    def wrapper(self, subprogram: dict[str, Any]) -> list[str]:
        """The interpreted function that keeps the locked public signature.

        It is the only thing that knows the kernel needs state: it reads each
        closure entry off the validated NumPy module (``_host``) or the
        companion's (``_host_<prefix>``) and passes it positionally. Everything
        the Fortran declared stays where the Fortran put it.
        """
        assert self.emission is not None
        derived = derived_components(self.emission.kernels.record, subprogram)
        positional: list[str] = []
        keyword: list[str] = []
        actuals: list[str] = []
        for argument in subprogram["args"]:
            if argument["intent"] not in ("IN", "INOUT", "UNKNOWN"):
                continue
            if argument["name"] in derived:
                positional.append(argument["name"])
                actuals.extend(f"{argument['name']}.{c}" for c in derived[argument["name"]])
            elif argument["optional"]:
                keyword.append(argument["name"])
            else:
                positional.append(argument["name"])
                actuals.append(argument["name"])
        wants = [
            f"want_{a['name']}" for a in subprogram["args"] if Statements.is_optional_output(a)
        ]
        parameters = (
            positional + [f"{k}=None" for k in keyword] + [f"{w}=False" for w in wants]
        )
        state = self.emission.closure_of(subprogram["name"])
        call = ", ".join(
            actuals + [_host_reference(s, self.emission) for s in state] + keyword
            + [f"{w}={w}" for w in wants]
        )
        errors = [
            a["name"] for a in subprogram["args"] if a["dtype"] == "str" and a["intent"] == "OUT"
        ]
        head = f"def {subprogram['name']}({', '.join(parameters)}):"
        if not errors:
            return [
                head,
                '    """Host wrapper: state from validated numpy module."""',
                f"    return _{subprogram['name']}_k({call})",
                "",
                "",
            ]
        outs = [a["name"] for a in subprogram["args"] if a["intent"] in ("OUT", "INOUT")]
        unpacked = ", ".join(f"_r_{o}" for o in outs)
        lines = [
            head,
            '    """Host wrapper (str-OUT decoded from error codes)."""',
            f"    {unpacked} = _{subprogram['name']}_k({call})",
        ]
        lines.extend(f"    _r_{e} = _ERRMSGS[int(_r_{e})]" for e in errors)
        lines.append(f"    return {unpacked}")
        return [*lines, "", ""]

    # -- the pre-scan ---------------------------------------------------------

    def _referenced_closure(self, node: Any) -> set[str]:
        """Closure entries the ``calls`` list does not mention.

        The interface records only in-module callees, and a derived state
        object is read by naming it rather than by calling anything. So the
        body is walked for names, and three things it can turn up become
        closure entries: a companion's module state (flattened if derived), a
        companion procedure referenced by name, and this module's own derived
        state.
        """
        assert self.emission is not None
        kernels = self.emission.kernels
        used = {str(name).lower() for name in walk(node, f03.Name)}
        extra: set[str] = set()

        for alias in kernels.aliases():
            prefix = alias.lstrip("_")
            derived = kernels.companion_derived_state(alias)
            for obj, components in derived.items():
                if obj in used:
                    extra |= {f"{prefix}__{obj}__{c}" for c in components}
            for obj in kernels.companion_state(alias):
                if obj in used and obj not in derived:
                    extra.add(f"{prefix}__{obj}")

        for obj, components in kernels.own_derived_state().items():
            if obj in used:
                extra |= {f"own__{obj}__{c}" for c in components}

        for name in used:
            remote = self.remotes.get(name)
            if remote is not None and remote.name in kernels.companion_kernels(remote.alias):
                prefix = remote.alias.lstrip("_")
                extra |= {
                    f"{prefix}__{n}"
                    for n in kernels.companion_closure(remote.alias, remote.name)
                }
        return extra


def _as_numba_names(names: Names, aliases: frozenset[str]) -> NumbaNames:
    """The same ``Names``, answering companion references a kernel's way."""
    return NumbaNames(
        semantics=names.semantics,
        module_parameters=names.module_parameters,
        use_parameters=names.use_parameters,
        companion_globals=names.companion_globals,
        state_names=names.state_names,
        use_bindings=names.use_bindings,
        literals=names.literals,
        local_constants=names.local_constants,
        companion_aliases=aliases,
    )


def _host_reference(entry: str, emission: Emission) -> str:
    """Where the wrapper reads one closure entry from."""
    if entry.startswith("own__"):
        return "_host." + entry[len("own__") :].replace("__", ".", 1)
    if "__" in entry:
        prefix, rest = entry.split("__", 1)
        if prefix in {alias.lstrip("_") for alias in emission.kernels.aliases()}:
            return f"_host_{prefix}.{rest.replace('__', '.', 1)}"
    return f"_host.{entry}"


SCALAR_VIEW = re.compile(r"np\.(int64|float64)\(([A-Za-z_]\w*)\)\.view\(np\.(float64|int64)\)")


def _njit_patch(lines: list[str]) -> list[str]:
    """An operator's patch, adapted where numba differs.

    Only one adaptation, and it is mechanical: a scalar bit-view roundtrip has
    no nopython form, so it goes through a one-element array. The patch is
    otherwise applied verbatim, because a patch is an audited answer and this
    backend has no standing to rewrite it.
    """
    return [
        SCALAR_VIEW.sub(
            lambda m: f"np.array([{m.group(2)}], dtype=np.{m.group(1)}).view(np.{m.group(3)})[0]",
            line,
        )
        for line in lines
    ]
