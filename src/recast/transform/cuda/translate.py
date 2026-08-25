"""``translate.cuda``: device functions, elementwise launchers, host wrappers.

The third stage, and the second slot ``TranslateRecipe`` declared and nothing
filled. Like the Numba backend it stands beside the validated NumPy module
rather than replacing it, and reads its state from there.

Three layers, and it is worth knowing which does what:

* a **device function** per specialization -- scalar rank, exactly the Fortran;
* a **launcher**, ``@cuda.jit``, one thread per element of the input arrays,
  calling the device function on that element;
* a **host wrapper** ``<name>_v(arrays..., idx=None)`` that allocates the
  outputs, moves everything to the device, picks the present or absent
  launcher by whether ``idx`` was given, and copies the results back.

**Two tables are configuration rather than rules**, because upstream has them
hard-coded to the one module it was built for. ``launchers_exclude`` names
subprograms that get device functions but no launcher -- the interchangeable
implementations behind a dispatch, and a function whose result is not a float
array. ``integer_state`` names closure entries the wrapper must pass as
``np.int32`` rather than ``np.float64``. Which names those are is knowledge
about a particular module, so it arrives the way every other such table does
and the engine ships neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform

__all__ = ["CudaTranslation", "factory"]

DOCSTRING = '''"""Machine-translated from {source} by recast -- stage 3 of 3.

``@cuda.jit`` device functions with compile-time present/absent
specialization, and one elementwise launcher per public numeric function.
Tolerance applies: CUDA's libdevice is one to two ULP from libm, so this
backend is gated on a ULP bound and never on bit-exactness.
DO NOT hand-edit mechanical blocks -- fix the engine instead.
"""'''

LAUNCHER = """@cuda.jit
def {launcher}({arrays}, {state}, {optional}{outputs}):
    i = cuda.grid(1)
    if i < {first}.size:
        {assign}


"""

WRAPPER = '''def {name}_v({arrays}, idx=None):
    """Vectorized GPU entry; state read from the validated numpy module."""
    n = {first}.size
    {allocate}
    d_in = [cuda.to_device(np.ascontiguousarray(a)) for a in ({arrays},)]
    d_out = [cuda.to_device(o) for o in ({output_names},)]
    blocks = (n + TPB - 1) // TPB
    if idx is None:
        {absent}[blocks, TPB](*d_in, {state}, *d_out)
    else:
        {present}[blocks, TPB](*d_in, {state}, np.int32(idx), *d_out)
    res = [d.copy_to_host() for d in d_out]
    return res[0] if len(res) == 1 else tuple(res)


'''


class CudaTranslation(Transform):
    """Fortran to CUDA device functions, anchored on the NumPy translation."""

    name = "recast.translate.fortran-to-cuda"
    requires = ("interface", "constants", "effects")
    deterministic = True

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return (
            unit.kind in ("module", "program")
            and "parse_error" not in unit.attrs
            and bool(facts.interface)
        )

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        from recast.transform.cuda.emitter import CudaSubprograms
        from recast.transform.numba.backend import Kernels, ineligible_reason
        from recast.transform.numba.emitter import Emission
        from recast.transform.numba.translate import _subprogram_nodes
        from recast.transform.numpy.statements import REFUSED
        from recast.transform.numpy.translate import NumpyTranslation
        from recast.transform.profiles import DEFAULT, PROFILES

        source = NumpyTranslation._verified_source(unit, facts, config)
        kernels = Kernels(record=facts.interface, externals=facts.provenance.get("externals", {}))
        emission = Emission(kernels=kernels)
        assembler = CudaSubprograms(
            record=facts.interface,
            constants=facts.constants,
            profile=PROFILES[config.get("profile", DEFAULT)],
            externals=facts.provenance.get("externals", {}),
            function_stubs=config.get("function_stubs", {}),
            statement_stubs=config.get("statement_stubs", {}),
            intrinsics=config.get("intrinsic_overrides", {}),
            call_transforms=config.get("call_transforms", {}),
            function_transforms=config.get("function_transforms", {}),
            emission=emission,
        )

        nodes = _subprogram_nodes(source)
        body: list[str] = []
        report: list[dict[str, Any]] = []
        delegated: list[tuple[str, str]] = []
        for record in facts.interface["subprograms"]:
            name = record["name"]
            if name not in kernels.names or name not in nodes:
                kernels.names.discard(name)
                delegated.append((name, ineligible_reason(record, assembler.externals)))
                continue
            try:
                emitted, blocks = assembler.render(nodes[name], name)
            except (*REFUSED, KeyError) as refusal:
                kernels.names.discard(name)
                delegated.append((name, f"[emit] {refusal}"))
                continue
            body.extend(emitted)
            report.extend(blocks)

        body.extend(
            self.launchers(
                facts.interface,
                emission,
                exclude=tuple(config.get("launchers_exclude", ())),
                integer_state=frozenset(config.get("integer_state", ())),
            )
        )

        module = facts.interface["module"]
        text = self._header(module, source, config) + "".join(
            line if line.endswith("\n") else line + "\n" for line in body
        )
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={Path(f"{module}_cuda.py"): text.encode()},
            deferred=[f"{name}: {reason}" for name, reason in delegated],
            notes={
                "blocks": report,
                "device_functions": sorted(kernels.names),
                "host_delegated": [name for name, _ in delegated],
                "anchor": f"{module}_numpy.py",
                "profile": assembler.profile.name,
            },
        )

    @staticmethod
    def launchers(
        record: dict[str, Any],
        emission: Any,
        exclude: tuple[str, ...] = (),
        integer_state: frozenset[str] = frozenset(),
    ) -> list[str]:
        """A launcher per variant, and one host wrapper, per public function."""
        from recast.transform.cuda.emitter import kernel_name, optionals_of

        out: list[str] = []
        for subprogram in record["subprograms"]:
            name = subprogram["name"]
            if name not in emission.kernels.names or name in exclude:
                continue
            arrays = [
                a["name"]
                for a in subprogram["args"]
                if a["intent"] in ("IN", "INOUT", "UNKNOWN") and not a["optional"]
            ]
            if not arrays:
                continue
            outputs = (
                [subprogram["result"]]
                if subprogram["kind"] == "function"
                else [a["name"] for a in subprogram["args"] if a["intent"] == "OUT"]
            )
            state = sorted(emission.kernels.state_closure(name))
            optionals = optionals_of(subprogram)
            variants: list[tuple[str | None, list[str]]] = (
                [("p", optionals), ("a", [])] if optionals else [(None, [])]
            )
            for variant, supplied in variants:
                device = kernel_name(name, variant)
                call = f"{device}({', '.join([f'{a}[i]' for a in arrays] + state + supplied)})"
                if len(outputs) == 1:
                    assign = f"out_{outputs[0]}[i] = {call}"
                else:
                    unpacked = ", ".join(f"_t{at}" for at in range(len(outputs)))
                    assign = f"{unpacked} = {call}\n        " + "\n        ".join(
                        f"out_{o}[i] = _t{at}" for at, o in enumerate(outputs)
                    )
                out.append(
                    LAUNCHER.format(
                        launcher=f"k{device}",
                        arrays=", ".join(arrays),
                        state=", ".join(state),
                        optional=(", ".join(supplied) + ", ") if supplied else "",
                        outputs=", ".join(f"out_{o}" for o in outputs),
                        first=arrays[0],
                        assign=assign,
                    ).replace(", ,", ",")
                )
            out.append(
                WRAPPER.format(
                    name=name,
                    arrays=", ".join(arrays),
                    first=arrays[0],
                    allocate="\n    ".join(
                        f"out_{o} = np.empty(n, dtype=np.float64)" for o in outputs
                    ),
                    output_names=", ".join(f"out_{o}" for o in outputs),
                    absent=f"k{kernel_name(name, 'a' if optionals else None)}",
                    present=f"k{kernel_name(name, 'p' if optionals else None)}",
                    state=", ".join(
                        f"np.int32(_host.{s})" if s in integer_state else f"np.float64(_host.{s})"
                        for s in state
                    ),
                ).replace("(*d_in, , ", "(*d_in, ")
            )
        return out

    @staticmethod
    def _header(module: str, source: Path, config: dict[str, Any]) -> str:
        stem = config.get("constants_stem", f"{module}_constants")
        pieces = [
            DOCSTRING.format(source=source.name),
            "",
            "import math",
            "",
            "import numpy as np",
            "from numba import cuda",
            "",
            f"from {stem} import *  # noqa: F401,F403",
            "",
            f"import {module}_numpy as _host",
            "",
            "TPB = 64",
            "",
        ]
        return "\n".join(pieces) + "\n"


def factory(**config: Any) -> CudaTranslation:
    del config
    return CudaTranslation()
