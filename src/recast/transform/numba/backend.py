"""Kernel eligibility and the state closure, which is what Numba changes.

Relayed from ``pipeline/numbaize.py``, whose ``NjitEmitter`` subclasses the
pipeline's one-class translator and overrides thirteen of its methods. The
split this repository made means those thirteen land on three different floors,
so the emitter here is three subclasses rather than one -- but they are the
same thirteen rules, and ``tools/numba_diff.py`` holds them to the original.

**What the backend is actually for.** ``@njit`` compiles a function ahead of
its first call, and a compiled function cannot see a module-level global the
way an interpreted one can: numba resolves globals at compile time and freezes
them. A translated module keeps its Fortran module state as module-level
names, so a kernel that reads ``omeps`` would be compiled against whatever
``omeps`` held when the kernel was first called, and would go on using that
value after the model changed it. The whole design follows from that one fact:

  - every subprogram's **transitive** module-state read set is computed and
    appended to its kernel's signature as explicit positional parameters, so
    the state arrives as an argument and is never frozen;
  - a thin **host wrapper** keeps the public signature, reads the state off
    the validated NumPy module, and calls the kernel with it -- which is also
    why no logic is duplicated: initialization, admin and everything
    ineligible stay in the NumPy module and are re-exported from here;
  - a subprogram outside the subset is **host-delegated**, ``name = _host.name``,
    rather than guessed at.

Eligibility is a property of the Fortran, not of numba's mood: a subprogram
that writes module state cannot take it as a parameter, a CHARACTER argument
has no nopython representation, and a call into an external shim has no
compiled body to call. The JAX backend mirrors this rule deliberately -- see
``recast.transform.jax.backend.eligible`` -- so the two accelerators delegate
the same set and a coverage number means the same thing on both.

**One deliberate divergence, in how a companion's kernel set is learned.** The
pipeline prefers to read ``_NJIT_KERNELS`` out of the companion's
already-emitted ``<module>_njit.py``, and computes the set from the interface
only when no such file is on disk. This backend always computes it, because at
transform time there is no emitted file to read -- a Transform returns a
Candidate rather than writing into a ``translated/`` directory, and the
companion it is told about is an interface record.

That is not only a mechanical necessity, it is the safer of the two. Checked
against the translator's own tree: ``translated/micro_mg_utils_njit.py`` lists
``ice_deposition_sublimation`` in ``_NJIT_KERNELS`` and defines a kernel for
it, and ``micro_mg2_0_njit.py`` calls that kernel -- but the subprogram writes
the module state ``mg_ice_props``, so the pipeline's *own current* rule makes
it ineligible, and its own fallback computation agrees. The emitted file is
older than the rule. Reading a kernel set off disk means a stale companion can
put a call to a kernel in a file that a fresh run of the same emitter does not
define; computing it cannot. A full run in dependency order converges to the
same answer either way, which is why this is a note rather than an issue filed
against them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recast.transform.numpy.expressions import Remote

__all__ = [
    "NJIT_EXTERNALS",
    "Kernels",
    "derived_components",
    "eligible",
    "ineligible_reason",
]

DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")

NJIT_EXTERNALS = {"gamma": "math.gamma"}
"""Externals numba can compile after all, and what it calls them.

The externals module is a file of audited Python shims the interpreted
translation calls into; a kernel cannot, because there is no compiled body
behind the name. ``gamma`` is the exception only because numba supports
``math.gamma`` directly, so the call is rewritten rather than delegated.
"""


def derived_components(record: dict[str, Any], subprogram: dict[str, Any]) -> dict[str, list[str]]:
    """``{argument: [component names]}`` for this subprogram's derived-type
    arguments.

    They are flattened into one scalar kernel parameter per component, because
    numba's nopython mode has no representation for the ``SimpleNamespace`` the
    NumPy backend uses for a derived type. The host wrapper unpacks
    ``obj.component`` on the way in, which is why the wrapper is what keeps the
    locked public signature and the kernel is free to spell its arguments
    however the compiler needs them.
    """
    types = record.get("types", {})
    out = {}
    for argument in subprogram["args"]:
        match = DERIVED.match(str(argument["dtype"] or ""))
        if match:
            out[argument["name"]] = list(types.get(match.group(1).lower(), {}))
    return out


def _has_character(subprogram: dict[str, Any]) -> bool:
    """A CHARACTER anywhere numba would have to represent it.

    An ``intent(out)`` CHARACTER argument is exempt, and not by oversight: the
    emitter turns writes to one into an integer error code and decodes it back
    to a string in the host wrapper, so the kernel never holds the string. An
    ``in`` argument has no such escape, and neither does a CHARACTER result.
    """
    if any(a["dtype"] == "str" and a["intent"] != "OUT" for a in subprogram["args"]):
        return True
    result = str(subprogram.get("result_dtype") or "")
    return result == "str" or "UNKNOWN(TYPE" in result


def eligible(subprogram: dict[str, Any], externals: dict[str, Any] | None = None) -> bool:
    """Whether this subprogram can become a compiled kernel."""
    if subprogram["module_state_written"]:
        return False
    if _has_character(subprogram):
        return False
    return not any(
        call in (externals or {}) and call not in NJIT_EXTERNALS for call in subprogram["calls"]
    )


def ineligible_reason(subprogram: dict[str, Any], externals: dict[str, Any] | None = None) -> str:
    """Why not, in the categories a coverage report should keep apart.

    Same three families the JAX survey reports, and worth keeping apart for the
    same reason: ``[elig]`` is by design and will not fall, while the emitter's
    own refusals are the number that should.
    """
    if subprogram["module_state_written"]:
        return "[elig] module-state write"
    if _has_character(subprogram):
        return "[elig] character argument or result"
    shimmed = [
        call
        for call in subprogram["calls"]
        if call in (externals or {}) and call not in NJIT_EXTERNALS
    ]
    if shimmed:
        return f"[elig] calls external shim {shimmed[0]!r}"
    return "[elig] not eligible"


@dataclass
class Kernels:
    """The kernel set for one module, and the state each kernel has to be handed.

    Built once per module and shared by the three floors, because the closure is
    a property of the call graph rather than of any one subprogram: a kernel's
    signature has to hold the state its callees read, transitively, or the
    callee is compiled against a frozen global. Cross-module entries are tagged
    ``alias__name`` and resolved by the wrapper against the companion's own
    validated NumPy module.
    """

    record: dict[str, Any]
    """The module's ``interface.extract`` record."""

    companions: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Emitted import alias -> that companion's interface record.

    Keyed by alias rather than a list, because every cross-module closure entry
    is tagged with the alias and has to be resolved back to the record that
    explains it. Deriving the alias here would mean a second copy of
    ``translate._aliases``, whose rule is the pipeline's and whose collision
    handling is this repository's; the caller has both and passes them.
    """

    remotes: dict[str, Remote] = field(default_factory=dict)
    externals: dict[str, dict[str, Any]] = field(default_factory=dict)

    names: set[str] = field(init=False, default_factory=set)
    """Subprograms that are still kernels. The emitter *removes* from this as
    it goes: a subprogram whose body turns out to be outside the subset is
    delegated, and every caller must then stop referring to its kernel."""

    def __post_init__(self) -> None:
        self._subprograms = {s["name"]: s for s in self.record["subprograms"]}
        self._closures: dict[str, set[str]] = {}
        self._companion_closures: dict[str, dict[str, set[str]]] = {}
        self._companion_subprograms: dict[str, dict[str, Any]] = {}
        self._companion_kernels: dict[str, set[str]] = {}
        self._companion_state: dict[str, dict[str, Any]] = {}
        self._companion_types: dict[str, dict[str, Any]] = {}
        self.names = {
            s["name"] for s in self.record["subprograms"] if eligible(s, self.externals)
        }
        for alias, companion in self.companions.items():
            self._companion_subprograms[alias] = {s["name"]: s for s in companion["subprograms"]}
            self._companion_state[alias] = {
                m["name"]: m.get("dtype") for m in companion.get("module_state", [])
            }
            self._companion_types[alias] = companion.get("types", {})
            # A companion's eligibility is computed from its own interface with
            # the same rules, minus the externals test: this module does not
            # have its externals table, and the pipeline does not consult one
            # here either.
            self._companion_kernels[alias] = {
                s["name"]
                for s in companion["subprograms"]
                if not s["module_state_written"] and not _has_character(s)
            }

    # -- the closure ----------------------------------------------------------

    def state_closure(self, name: str) -> set[str]:
        """Every module variable this subprogram reads, transitively.

        Only through callees that are *themselves* kernels: a call that is
        host-delegated goes back to the NumPy module, which reads its own
        globals in the ordinary way and needs nothing passed to it.
        """
        if name in self._closures:
            return self._closures[name]
        self._closures[name] = set()  # cycle guard: recursion sees an empty set
        subprogram = self._subprograms.get(name)
        if subprogram is None:
            return self._closures[name]
        state = set(subprogram["module_state_read"])
        for call in subprogram["calls"]:
            if call in self.names:
                state |= self.state_closure(call)
                continue
            remote = self.remotes.get(call)
            if remote is not None and remote.name in self._companion_kernels.get(remote.alias, ()):
                state |= {
                    f"{remote.alias.lstrip('_')}__{n}"
                    for n in self.companion_closure(remote.alias, remote.name)
                }
        self._closures[name] = state
        return state

    def companion_closure(self, alias: str, name: str) -> set[str]:
        """The companion's own transitive closure for one of its procedures."""
        memo = self._companion_closures.setdefault(alias, {})
        subprograms = self._companion_subprograms.get(alias, {})

        def walk(procedure: str) -> set[str]:
            if procedure in memo:
                return memo[procedure]
            memo[procedure] = set()
            record = subprograms.get(procedure)
            if record is None:
                return set()
            state = set(record["module_state_read"])
            for call in record["calls"]:
                if call in self._companion_kernels.get(alias, ()):
                    state |= walk(call)
            memo[procedure] = state
            return state

        return walk(name)

    def companion_derived_state(self, alias: str) -> dict[str, list[str]]:
        """``{state name: [components]}`` for a companion's derived module state."""
        types = self._companion_types.get(alias, {})
        out = {}
        for name, dtype in self._companion_state.get(alias, {}).items():
            match = DERIVED.match(str(dtype))
            if match:
                out[name] = list(types.get(match.group(1).lower(), {}))
        return out

    def own_derived_state(self) -> dict[str, list[str]]:
        """The same, for this module's own derived module state."""
        types = self.record.get("types", {})
        out = {}
        for state in self.record.get("module_state", []):
            match = DERIVED.match(str(state.get("dtype")))
            if match:
                out[state["name"]] = list(types.get(match.group(1).lower(), {}))
        return out

    def expand(self, names: set[str]) -> list[str]:
        """Closure entries in signature order, derived objects flattened.

        A derived state object cannot cross into nopython mode whole, so it
        arrives one parameter per component: ``own__obj__c`` for this module's,
        ``alias__obj__c`` for a companion's. Scalars pass through. The order is
        sorted entries, each object's components in type-declaration order --
        and it has to be exactly this order on both sides, because these are
        positional parameters and the wrapper fills them by position.
        """
        own_derived = self.own_derived_state()
        out: list[str] = []
        seen: set[str] = set()

        def add(entry: str) -> None:
            if entry not in seen:
                seen.add(entry)
                out.append(entry)

        for entry in sorted(set(names)):
            if entry in own_derived:
                for component in own_derived[entry]:
                    add(f"own__{entry}__{component}")
                continue
            if "__" in entry and not entry.startswith("own__"):
                prefix, rest = entry.split("__", 1)
                alias = next(
                    (a for a in self._companion_subprograms if a.lstrip("_") == prefix), None
                )
                if alias is not None and "__" not in rest:
                    derived = self.companion_derived_state(alias)
                    if rest in derived:
                        for component in derived[rest]:
                            add(f"{prefix}__{rest}__{component}")
                        continue
            add(entry)
        return out

    def companion_state(self, alias: str) -> dict[str, Any]:
        """``{state name: dtype}`` for one companion's module state."""
        return self._companion_state.get(alias, {})

    def aliases(self) -> tuple[str, ...]:
        return tuple(self._companion_subprograms)

    def companion_kernels(self, alias: str) -> set[str]:
        return self._companion_kernels.get(alias, set())

    def companion_subprogram(self, alias: str, name: str) -> dict[str, Any] | None:
        return self._companion_subprograms.get(alias, {}).get(name)
