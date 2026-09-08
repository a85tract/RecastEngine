"""``f2py-golden``: the untouched Fortran, compiled, as the reference.

Migrated from the build half of the source pipeline's ``diff_driver.py``
and the wrapper rules of ``gen_wrapper.py``. f2py does not translate
anything: it compiles the original source with a real Fortran compiler and
generates glue so Python can call the resulting machine code directly. That
is what makes it a reference -- the thing being compared against *is* the
original program, and bit-exact agreement with it is a meaningful claim.

The wrapper layer exists because f2py's own Fortran parser is shallow. It
cannot resolve use-imported kinds (a silent ``real(4)`` truncation, learned
the hard way), stumbles on derived types, and handles optional arguments
badly. So every subprogram under test gets a flat wrapper subroutine: raw
``real(8)``/``integer`` declarations that need no kind resolution, optional
arguments dropped, dimensions spelled in terms of the other arguments so
f2py can size the outputs.

The cache key folds in everything that can move the reference: the source
digest, every extra source's digest, the compiler's identity and version,
and the flags. Two builds with the same key must behave identically, and a
compiler upgrade changes the key rather than silently invalidating every
downstream Verdict.

The compile itself goes through the executor -- on a laptop that is a
subprocess, on a batch system it is a job -- and the environment is passed
whole plus the compiler overrides, because f2py needs a real PATH to find
its toolchain.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from recast import references
from recast.errors import ConfigError, OracleUnavailable, RecastError
from recast.fortran.intrinsics import ALL as INTRINSICS
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor, Job
from recast.plugins.oracle import Oracle

__all__ = [
    "F2pyGoldenOracle",
    "derived_components",
    "factory",
    "flattened_dummies",
    "unspellable",
    "wrappers_for",
]

FORTRAN_TYPES = {
    "float64": "real(8)",
    "float32": "real(4)",
    "int32": "integer",
    "int64": "integer(8)",
    "bool": "logical",
    "complex128": "complex(8)",
    "complex64": "complex(4)",
    # Fixed width because f2py cannot size len=* dummies; 128 covers every
    # message and name in the corpus, and Fortran comparison semantics pad
    # the shorter operand with blanks anyway.
    "str": "character(len=128)",
}
"""Raw type spellings, no kind parameters: nothing here needs f2py's
crackfortran to resolve a use-imported kind, which it cannot."""

DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)", re.I)
"""How the frontend spells a dummy of derived type: ``UNKNOWN(TYPE(name))``."""

DEFINED_ZERO = {"bool": ".false.", "str": "''"}
"""What an intent(out) dummy is set to before the call, by dtype.

A character dummy given ``0`` is a type error the compiler rejects
("Cannot convert INTEGER(4) to CHARACTER(128)"), which cost every module
with a character output its whole reference.
"""

DEFAULT_FLAGS = "-O1 -fno-fast-math -ffp-contract=off -fcheck=bounds"
"""Conservative by default. The reference must round the way the production
build rounds, and aggressive optimization is a second variable nobody asked
to test.

``-fcheck=bounds`` because a reference that reads outside its arrays is not
a reference: what it returns then is whatever memory sat beside the array
in *this* process, which the next process will not repeat (#42: PCHIP's
``dpchkt`` drawn with ``n = 1`` reads ``x(0)``; the reference read the byte
before the buffer, the translation's ``x[-1]`` wrapped to the last element,
and the verdict's numbers changed with the process the reference ran in).
With the check on, the reference ends its process on that draw with the
array and index named, and the gate declines the draw and says so instead
of comparing two undefined values. The checks do not touch the arithmetic,
so the rounding is the production build's still."""

_BUILD_LOG_TAIL_CHARS = 3000
"""How much of a failed build's output the error itself quotes. The tail,
because that is where crackfortran and the compiler report what they could
not parse; small enough that the error stays a message, not a log."""

_SAFE_SOURCE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}\Z")
"""A suffix that is safe to reproduce on a canonical staging filename.

f2py infers the source language and fixed/free form from the suffix, so the
staged copy must retain it.  The original basename is deliberately *not*
retained: NumPy's Meson backend joins and splits source arguments internally,
which turns whitespace (or a flag-looking basename) into additional tokens.
"""


def _resolved_root(value: str | os.PathLike[str]) -> Path:
    """Resolve and validate the project root before trusting provenance."""
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ConfigError(
            f"f2py project root {value!r} does not exist or cannot be resolved"
        ) from exc
    if not root.is_dir():
        raise ConfigError(f"f2py project root {root} is not a directory")
    return root


def _regular_file(path: Path, *, label: str) -> Path:
    """Return one canonical regular file or reject it fail-closed."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"{label} {path} does not exist or cannot be resolved") from exc
    if not resolved.is_file():
        raise ConfigError(f"{label} {resolved} is not a regular file")
    return resolved


def _source_under_root(root: Path, value: object, *, label: str) -> Path:
    """Resolve a provenance path and prove its target remains in ``root``."""
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"{label} must be a filesystem path, got {type(value).__name__}")
    resolved = _regular_file(root / Path(value), label=label)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(
            f"{label} {value!r} resolves outside the configured project root {root}"
        ) from exc
    return resolved


def _extra_sources(config: dict[str, Any]) -> list[Path]:
    """Validate explicitly configured sources (which may live outside root)."""
    values = config.get("extra_sources", []) or []
    if isinstance(values, (str, bytes, os.PathLike)):
        raise ConfigError("config['extra_sources'] must be a list of filesystem paths")
    resolved: list[Path] = []
    for index, value in enumerate(values):
        if not isinstance(value, (str, os.PathLike)):
            raise ConfigError(
                f"extra source {index} must be a filesystem path, got {type(value).__name__}"
            )
        resolved.append(_regular_file(Path(value), label=f"extra source {index}"))
    return resolved


def _staged_suffix(source: Path) -> str:
    suffix = source.suffix
    if not _SAFE_SOURCE_SUFFIX.fullmatch(suffix):
        raise ConfigError(
            f"source {source} has suffix {suffix!r}, which cannot be represented by a safe "
            "f2py staging filename"
        )
    return suffix


MODULE_DIR = "includes/mods"
"""Where the compiler leaves the reference's ``.mod`` files, and where the
wrapper's ``use`` finds them. A short relative name for the same reason every
other token here is one."""


def _stage_build_inputs(
    stage: Path, sources: list[Path], wrapper_text: str
) -> tuple[list[str], str, list[str]]:
    """Copy sources and expose include directories through controlled names.

    Returns the staged reference sources, the staged wrapper, and the include
    arguments -- the reference and the wrapper separately, because only the
    wrapper is ever handed to f2py.

    NumPy currently performs ``' '.join(...).split()`` both while parsing
    f2py sources and while parsing include options.  Consequently *every*
    token given to it is a short relative name we generated here.  Source
    parents are reachable to the compiler only through ``includes/dNNNN``
    aliases; their original spellings never enter a flags string or Meson
    template.
    """
    source_dir = stage / "sources"
    include_dir = stage / "includes"
    backend_include_dir = stage / "f2py-build" / "includes"
    source_dir.mkdir()
    include_dir.mkdir()
    backend_include_dir.mkdir(parents=True)
    (stage / MODULE_DIR).mkdir()
    # Meson resolves an include directory against f2py's build directory, and
    # crackfortran and the compiler resolve it against the job's cwd. The
    # same relative name has to name the same directory from both.
    (backend_include_dir / "mods").symlink_to(stage / MODULE_DIR, target_is_directory=True)

    staged: list[str] = []
    for index, source in enumerate(sources):
        relative = Path("sources") / f"source_{index:04d}{_staged_suffix(source)}"
        shutil.copyfile(source, stage / relative)
        staged.append(relative.as_posix())

    wrapper = Path("sources") / "wrappers.f90"
    (stage / wrapper).write_text(wrapper_text)

    include_args: list[str] = []
    parents = dict.fromkeys(source.parent for source in sources)
    for index, parent in enumerate(parents):
        alias = f"d{index:04d}"
        # One alias is used by crackfortran from the job cwd; the identical
        # alias below f2py's explicit build directory is used by Meson.
        (include_dir / alias).symlink_to(parent, target_is_directory=True)
        (backend_include_dir / alias).symlink_to(parent, target_is_directory=True)
        include_args.append(f"-Iincludes/{alias}")
    return staged, wrapper.as_posix(), include_args


INTENT_SPELLING = {"IN": "in", "OUT": "out", "INOUT": "inout", "UNKNOWN": "inout"}
"""Declared intent -> what a wrapper dummy is spelled with."""

CALLBACK_INTENT = {"IN": "in", "OUT": "out", "INOUT": "in,out"}
"""Declared intent -> what f2py calls it on a call-back argument.

f2py's call-back convention is the emitted translation's own: the arguments
the Fortran side supplies are passed to the Python function, and the ones it
expects back come out of the return. ``intent(in,out)`` is both.
"""


def _callback_declarations(
    argument: dict[str, Any], interfaces: dict[str, Any]
) -> tuple[list[str], str]:
    """The ``!f2py`` lines that teach f2py what a call-back argument is.

    f2py works a call-back's signature out from a call to it in the body of
    the routine being wrapped, and there is no such call here: the wrapper
    hands the procedure straight on. So the call is written for f2py alone --
    ``!f2py`` lines are Fortran comments and reach no compiler -- together
    with a declaration per call-back argument, which is where the intents
    come from. Without them the generated glue calls Python with no arguments
    and copies nothing back.

    The call-back's own arguments are the only names in scope for its
    dimensions, so an extent naming one of them is renamed with it; an extent
    naming anything else is refused rather than resolved against the
    wrapper's scope, where it would mean a different variable.

    A subroutine call-back is written as a CALL and a function call-back as an
    assignment, because that is how crackfortran tells the two apart, and the
    dummy carries the result's type so the ``implicit none`` wrapper compiles.
    """
    name = argument["name"]
    # An interface record names its interface; a signature entry carries the
    # record itself, because the harness on the other side has no module
    # record to look it up in. Either arrives here.
    interface = argument.get("interface")
    if isinstance(interface, str):
        interface = interfaces.get(interface)
    if not interface:
        raise ConfigError(
            f"procedure argument {name!r} carries no interface; this wrapper cannot say "
            "what calling it means -- wrap it by hand or drop the subprogram from the gate"
        )
    result_type = None
    if interface["kind"] == "function":
        # A function call-back answers through its result, so there is a type
        # to spell twice: on the dummy itself, because the wrapper is
        # ``implicit none`` and an EXTERNAL alone leaves it untyped, and on
        # the variable f2py's own call takes the result in.
        result_type = FORTRAN_TYPES.get(interface.get("result_dtype"))
        if result_type is None:
            raise ConfigError(
                f"call-back {name!r} returns dtype {interface.get('result_dtype')!r}, "
                "which this wrapper cannot spell"
            )
        if interface.get("result_dims"):
            raise ConfigError(
                f"call-back {name!r} returns an array; this wrapper spells scalar function "
                "call-backs only"
            )
        written = [a["name"] for a in interface["args"] if a["intent"] != "IN"]
        if written:
            # f2py hands a function call-back's written arguments back beside
            # its result, and which comes first is a convention this wrapper
            # would be inventing rather than sharing with the translation.
            raise ConfigError(
                f"call-back {name!r} is a function that writes argument(s) "
                f"{', '.join(written)}; this wrapper spells function call-backs that only "
                "read theirs"
            )
    spelled = {a["name"].lower(): f"cb_{name}_{a['name']}" for a in interface["args"]}
    sized = {
        token.lower()
        for a in interface["args"]
        for d in a.get("dims") or []
        for token in re.findall(r"[A-Za-z_]\w*", str(d.get("ub") or ""))
    }
    lines = []
    for a in interface["args"]:
        base = FORTRAN_TYPES.get(a["dtype"])
        intent = CALLBACK_INTENT.get(a["intent"])
        if base is None or intent is None:
            raise ConfigError(
                f"call-back {name!r} argument {a['name']!r} is dtype {a['dtype']!r} "
                f"intent {a['intent']!r}, which this wrapper cannot spell"
            )
        attributes = [f"intent({intent})"]
        if a["name"].lower() in sized:
            # f2py makes a dimension-determining integer optional and moves it
            # to the end of the call-back's argument list. The translation
            # calls the same object positionally, in declaration order, so it
            # has to stay where the interface put it.
            attributes.append("required")
        dims = ""
        if a.get("dims"):
            axes = []
            for d in a["dims"]:
                axis = str(d.get("ub") or "").strip()
                if axis.lower() not in spelled:
                    raise ConfigError(
                        f"call-back {name!r} argument {a['name']!r} has extent {axis!r}, "
                        "which is not one of the call-back's own arguments"
                    )
                axes.append(spelled[axis.lower()])
            dims = ", dimension(" + ", ".join(axes) + ")"
        lines.append(
            f"!f2py  {base}{dims}, {', '.join(attributes)} :: {spelled[a['name'].lower()]}"
        )
    arguments = ", ".join(spelled[a["name"].lower()] for a in interface["args"])
    if result_type is not None:
        # f2py reads a *function* call-back off an assignment whose right-hand
        # side calls it -- a bare call is a subroutine to crackfortran -- and
        # takes the result's type from the assigned variable, which therefore
        # has to be declared before the line that assigns it.
        assigned = f"cb_{name}_res"
        lines.append(f"!f2py  {result_type} :: {assigned}")
        lines.append(f"!f2py  {assigned} = {name}({arguments})")
        return lines, f"  {result_type}, external :: {name}"
    lines.append(f"!f2py  call {name}({arguments})")
    return lines, f"  external {name}"


_FREE_FORM_COLUMNS = 100
"""Where a generated wrapper line is folded.

Free-form Fortran's limit is 132 columns and gfortran makes overrunning it an
error, not a warning -- a subprogram with two dozen arguments writes an
argument list longer than that. Folded well short of the limit so the
continuation marker itself always fits.
"""


def _fold(line: str) -> list[str]:
    """One generated line as as many continued lines as it needs.

    Broken after a comma, which is the only place a wrapper's long lines have
    a boundary, and never inside a ``!f2py`` directive: those are comments to
    every compiler, and f2py's own parser does not read a continuation in one.
    """
    if len(line) <= _FREE_FORM_COLUMNS or line.lstrip().startswith("!"):
        return [line]
    indent = " " * (len(line) - len(line.lstrip())) + "    "
    pieces = line.split(", ")
    folded: list[str] = []
    current = pieces[0]
    for piece in pieces[1:]:
        if len(current) + len(piece) + 4 > _FREE_FORM_COLUMNS:
            folded.append(current + ", &")
            current = indent + piece
        else:
            current = f"{current}, {piece}"
    folded.append(current)
    return folded


def _logical_through_integer(argument: dict[str, Any]) -> bool:
    """A scalar LOGICAL INOUT dummy: the wrapper carries it as an integer."""
    return bool(
        argument.get("dtype") == "bool"
        and argument.get("intent") == "INOUT"
        and not argument.get("dims")
    )


def _passed_buffer(argument: dict[str, Any]) -> bool:
    """An OUT array that is the caller's buffer and has an axis of no
    declared extent: passed in and written in place, never allocated by
    the wrapper."""
    return bool(
        argument.get("intent") == "OUT"
        and argument.get("buffer")
        and any(not d.get("ub") for d in argument.get("dims") or ())
    )


def _extent(dim: dict[str, Any]) -> str:
    """The axis as the wrapper declares it: ``lb:ub`` when the lower bound is
    not one (CLUBB's ``lhs(-2:2, ngrdcol, ndim)``), so the callee sees the
    layout it was written for and f2py sizes the axis ``ub - lb + 1``."""
    if dim.get("ub"):
        lower = str(dim.get("lb") or "1").strip()
        return f"{lower}:{dim['ub']}" if lower != "1" else str(dim["ub"])
    return "*" if dim.get("assumed_size") else ":"


def _hide(
    extents: str, argument_names: list[str], parameters: dict[str, int] | None, hidden: list[str]
) -> None:
    """An extent naming neither an argument nor a local parameter is a hidden
    integer dummy the caller supplies; recorded once, in order of first use.

    An intrinsic call is not such a name. ``b(size(a))`` computes its extent
    from an argument already being passed, and hiding ``size`` declared a
    dummy of that name beside it -- ``integer, intent(in) :: size`` next to
    ``res(size(a))`` -- which gfortran rejects twice over, as a PROCEDURE
    attribute conflicting with INTENT and as a call to something not PURE.

    Neither is a name the argument list already carries in another case.
    Fortran does not distinguish ``N`` from ``n``, and the extent keeps the
    source's spelling while the argument names arrive lowercased from the
    frontend: ``real(dp) :: mesh(N+1)`` over ``integer, intent(in) :: N``
    hid an ``N`` beside the wrapper's own ``n``, which gfortran rejects as a
    duplicate formal argument -- the mesh module's three exponential-mesh
    functions, and every array-valued function whose extent names an
    argument in capitals.
    """
    known = {name.lower() for name in argument_names} | {
        name.lower() for name in (parameters or {})
    }
    for token, call in re.findall(r"([A-Za-z_]\w*)\s*(\(?)", extents):
        lowered = token.lower()
        if call and lowered in INTRINSICS:
            continue
        if lowered not in known:
            if lowered not in {name.lower() for name in hidden}:
                hidden.append(token)


def _allocatable_shim(
    argument: dict[str, Any], spelled: str
) -> tuple[str, list[str], list[str], list[str]]:
    """Pass an ALLOCATABLE dummy the allocatable actual Fortran requires.

    ``call loadtxt(filename, d)`` does not compile with ``d`` a plain
    assumed-shape dummy -- "Actual argument for 'd' must be ALLOCATABLE" --
    and f2py has no allocatable of its own to offer, because the extent the
    callee chooses is not known when it builds the array it hands back. So
    the wrapper keeps its caller-side buffer and calls through a local
    allocatable: the buffer's values go in, the callee's array comes back as
    far as the buffer reaches, and the rest of the buffer is left defined.

    Returns ``(actual, declarations, before, after)``: what to pass at the
    call site, the locals to declare, and the copies either side of the call.
    Truncation is why the differential harness does not call one of these --
    an array the callee sized is not the caller's buffer, and comparing the
    two would be comparing shapes nobody chose (see ``BitexactVerifier``).
    """
    name = argument["name"]
    local = f"{name}_alloc"
    rank = len(argument.get("dims") or ())
    hands_in = argument["intent"] in ("IN", "INOUT", "UNKNOWN")
    hands_back = argument["intent"] != "IN"
    if not rank:
        # A scalar allocatable dummy: no extent to reconcile, so the local is
        # allocated from the buffer and read back whole.
        declarations = [f"  {spelled}, allocatable :: {local}"]
        before = [f"  allocate({local}, source={name})"] if hands_in else []
        after = [f"  if (allocated({local})) {name} = {local}"] if hands_back else []
        return local, declarations, before, after
    colons = ", ".join([":"] * rank)
    fits = f"{name}_n"
    declarations = [f"  {spelled}, allocatable :: {local}({colons})"]
    before = [f"  allocate({local}, source={name})"] if hands_in else []
    if not hands_back:
        return local, declarations, before, []
    declarations.append(f"  integer :: {fits}({rank})")
    section = ", ".join(f":{fits}({axis})" for axis in range(1, rank + 1))
    after = [
        f"  {fits} = 0",
        f"  if (allocated({local})) {fits} = min(shape({name}), shape({local}))",
        f"  {name} = {DEFINED_ZERO.get(argument['dtype'], '0')}",
        f"  if (allocated({local})) {name}({section}) = {local}({section})",
    ]
    return local, declarations, before, after


def derived_components(
    record: dict[str, Any], argument: dict[str, Any], taken: set[str] | None = None
) -> list[dict[str, Any]] | str:
    """One flat scalar dummy per component of a derived-type argument, or why not.

    f2py cannot marshal a derived type, and a module whose only public
    subprogram takes one -- SLSQP's ``slsqp`` carries its reverse-communication
    state in ``type(slsqpb_data)`` and ``type(linmin_data)`` -- had no
    reference at all, so its gate never ran. A type made of scalar
    components *can* be spelled: the wrapper takes each component as a
    dummy of its own, ``<argument>_<component>``, copies them into a local
    of the type before the call and back out after it, and the candidate
    side does the same with the object it takes (``BitexactVerifier``,
    through the plan ``flattened_dummies`` puts on the oracle's handle).

    Returned entries carry ``name`` (the flat dummy), ``component``,
    ``dtype`` and ``spelled`` (the Fortran declaration type). A string is the
    reason there is no such spelling: a type the record does not define, one
    the module does not export (the wrapper has to ``use`` it), a component
    that is an array, allocatable or pointer, or one of a dtype this wrapper
    cannot spell either -- and a flat name that collides with another dummy.
    """
    derived = DERIVED.match(str(argument["dtype"]))
    if derived is None:
        return f"dtype {argument['dtype']!r} is not a derived type"
    type_name = derived.group(1).lower()
    components = (record.get("types") or {}).get(type_name)
    if components is None:
        return f"type {type_name!r} is not defined in this module's record"
    exported = {str(name).lower() for name in record.get("public_types") or ()}
    if type_name not in exported:
        return f"type {type_name!r} is not public, so a wrapper cannot use it"
    if not components:
        return f"type {type_name!r} has no components"
    flat: list[dict[str, Any]] = []
    names = set(taken or ())
    for component, spec in components.items():
        if spec.get("dims") or spec.get("allocatable") or spec.get("pointer"):
            return f"component {type_name}%{component} is not a scalar"
        spelled = FORTRAN_TYPES.get(str(spec.get("dtype")))
        if spelled is None or spec.get("dtype") == "str":
            return f"component {type_name}%{component} has dtype {spec.get('dtype')!r}"
        name = f"{argument['name']}_{component}".lower()
        if name in names:
            return f"flat name {name!r} for {type_name}%{component} collides with another dummy"
        names.add(name)
        flat.append(
            {
                "name": name,
                "component": str(component).lower(),
                "dtype": str(spec["dtype"]),
                "spelled": spelled,
            }
        )
    return flat


def flattened_dummies(
    record: dict[str, Any], subprograms: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{subprogram: {argument: {"type": name, "components": [...]}}}`` for
    every derived-type dummy ``wrappers_for`` spells component by component.

    The verifier reads this off the oracle's handle to split the candidate's
    own derived-type argument the same way: same flat names, same order.
    Only subprograms with at least one flattened dummy appear.
    """
    table = {s["name"]: s for s in record["subprograms"]}
    plan: dict[str, dict[str, dict[str, Any]]] = {}
    for name in subprograms:
        sub = table.get(name)
        if sub is None:
            continue
        arguments = [a for a in sub["args"] if not a.get("optional")]
        taken = {str(a["name"]).lower() for a in arguments}
        entries: dict[str, dict[str, Any]] = {}
        for argument in arguments:
            derived = DERIVED.match(str(argument["dtype"]))
            if derived is None:
                continue
            components = derived_components(record, argument, taken)
            if isinstance(components, str):
                continue
            taken.update(c["name"] for c in components)
            entries[argument["name"]] = {
                "type": derived.group(1).lower(),
                "components": [{k: v for k, v in c.items() if k != "spelled"} for c in components],
            }
        if entries:
            plan[name] = entries
    return plan


def unexercisable(subprogram: dict[str, Any]) -> str | None:
    """Why the differential cannot exercise this reference, or ``None``.

    The wrapper compiles either way; what this answers is whether calling it
    means anything. Four shapes it does not:

    *A character value.* ``FORTRAN_TYPES`` spells every character dummy
    ``character(len=128)`` because f2py cannot size a ``len=*`` one, so the
    reference's interface is not the source's, and the harness has no draw
    for a string in the first place. One character dummy is exercisable
    anyway: a path an OPEN in the body *creates* (``path: "created"``). Any
    name works there -- the subprogram makes the file rather than finding one
    -- so the harness draws a scratch path per side and compares the files,
    and the wrapper passes ``trim()`` of its padded dummy so the callee's
    ``len=*`` is the length the caller chose. A path the body opens
    ``STATUS='OLD'`` stays ungated: the draw would have to be a file that
    already holds something, which nothing here can produce.

    *An array the callee allocates.* An ALLOCATABLE intent(out) dummy is
    sized by the callee; f2py can only hand back the buffer the caller
    passed, and comparing a buffer against an allocation compares two shapes
    nobody chose (see ``_allocatable_shim``).

    *A LOGICAL INOUT array dummy.* f2py exposes a scalar LOGICAL INOUT as a
    writable rank-0 array and marshals it through its own Python-object
    conversion, so writing 0/1 for false/true is enough -- Fortran's own
    truthiness test is "nonzero", the same convention already relied on when
    reading a LOGICAL OUT back (see ``BitexactVerifier``'s bool comparison).
    That is not true of a LOGICAL INOUT *array*: f2py hands one back as an
    in-place buffer whose element size must match the compiler's native
    LOGICAL storage exactly (4 bytes for the default kind), while this
    harness draws LOGICAL arrays with NumPy's 1-byte ``bool_`` dtype, which
    f2py rejects (``failed to initialize intent(inout) array``). Left
    ungated here instead of reaching that refusal (``BitexactVerifier``'s
    ``logical_inouts`` check, which draws the same scalar/array line) blocks
    every other subprogram in the same unit's differential gate along with
    it.

    *A function with OUT/INOUT dummies.* f2py returns a FUNCTION's result
    and its OUT/INOUT dummies in one tuple, the same as a SUBROUTINE's, but
    ``BitexactVerifier._paired_outputs`` only ever pairs a function's single
    result -- it has no side-effect leg for a function to fall into the way
    a subroutine's OUT/INOUT dummies do. Left ungated here instead of
    reaching that refusal (``BitexactVerifier._compare_subprogram``'s own,
    matching check) blocks every other subprogram in the same unit's
    differential gate along with it.

    Named rather than silently skipped: the verifier counts an uncompared
    public subprogram as silence unless the oracle says why, which is what
    ``OracleRef.handle["ungated"]`` carries. The flat oracle answers the same
    question in ``recast.oracle.flat.unspellable``; the two differ because
    what each can build differs.
    """
    for argument in subprogram["args"]:
        if argument.get("optional"):
            continue  # dropped from both calls, so it decides nothing here
        if str(argument["dtype"]) == "str":
            if argument.get("path") == "created":
                continue
            if argument.get("path") == "existing":
                return (
                    f"{argument['name']}: names a file the body opens STATUS='OLD', "
                    "which no generated draw can put there"
                )
            return f"{argument['name']}: character dummy, fixed at len=128 by the wrapper"
        if (
            argument.get("allocatable")
            and argument.get("dims")
            and argument["intent"] in ("OUT", "INOUT")
        ):
            return f"{argument['name']}: allocatable array the callee sizes"
        if (
            argument["intent"] == "INOUT"
            and str(argument["dtype"]) == "bool"
            and argument.get("dims")
        ):
            return (
                f"{argument['name']}: LOGICAL INOUT array dummy, f2py's in-place buffer "
                "requires the compiler's native LOGICAL element size, which this harness's "
                "1-byte bool draw does not provide"
            )
    if subprogram["kind"] == "function" and str(subprogram.get("result_dtype")) == "str":
        return "character result, fixed at len=128 by the wrapper"
    if subprogram["kind"] == "function":
        outs_required = [
            argument["name"]
            for argument in subprogram["args"]
            if argument["intent"] in ("OUT", "INOUT") and not argument.get("optional")
        ]
        if outs_required:
            return (
                "declares OUT/INOUT dummy argument(s) "
                f"{', '.join(outs_required)}; this verifier cannot pair both its "
                "result and side effects"
            )
    return None


INTERFACE_BLOCK = re.compile(
    r"^[ \t]*interface\b.*?^[ \t]*end[ \t]*interface\b", re.I | re.M | re.S
)
"""An INTERFACE block, for the text scan below: what it holds is declared,
not defined, which is the whole distinction ``undefined_externals`` draws."""

SUBPROGRAM_DEFINITION = re.compile(
    r"^[^!\n]*?\b(?:subroutine|function)\s+([A-Za-z_]\w*)", re.I | re.M
)
"""A line that opens (or closes) a subprogram definition. Deliberately loose:
this only ever *suppresses* a stub, so a name too many costs a build the
diagnosis it already gives today, and a name too few costs a duplicate symbol."""


def build_records(facts: Facts) -> list[dict[str, Any]]:
    """Every interface record the reference build compiles from source.

    The unit's own, its companions', and the companions' own dependencies --
    the same three groups ``companion_sources`` hands the compiler, because
    the question here is what that build defines.
    """
    records = [facts.interface]
    for group in ("companions", "companion_dependencies"):
        for entry in facts.provenance.get(group) or []:
            record = entry.get("record") if isinstance(entry, dict) else None
            if isinstance(record, dict) and record.get("subprograms") is not None:
                records.append(record)
    return records


def undefined_externals(records: list[dict[str, Any]], extras: list[Path]) -> list[str]:
    """Procedures the build declares an INTERFACE for and defines nowhere.

    ``use lapack, only: dgesv`` names a module whose whole content is
    interface blocks: the bodies are in a compiled library the original
    program linked, and the reference build links nothing but the sources
    staged for it. The declarations are real -- they are what the compiler
    checked the call against -- and the definitions are absent, so the
    extension links with ``dgesv_`` undefined and the *import* fails, taking
    every subprogram in the module with it, including the ones that never go
    near LAPACK.

    Named here so the build can answer for them. ``recast.references`` holds a
    reference implementation for a few, and those get a body that computes --
    the same one the translation gets, so the call rounds alike on both sides.
    For the rest ``unresolved_stubs`` gives a body that refuses, and
    ``reaching`` says which subprograms must not be exercised because they
    would reach one.

    ``extras`` are scanned as text rather than as records: a source the
    operator added from outside the tree may be the very definition this is
    looking for, and stubbing a name that build already defines is a
    duplicate symbol where there was a working reference.
    """
    declared: set[str] = set()
    defined: set[str] = set()
    for record in records:
        defined |= {str(s["name"]).lower() for s in record["subprograms"]}
        for entry in (record.get("interfaces") or {}).values():
            if isinstance(entry, dict) and entry.get("kind") in ("subroutine", "function"):
                declared.add(str(entry["name"]).lower())
    for extra in extras:
        try:
            text = extra.read_text(errors="replace")
        except OSError:
            continue
        defined |= {
            match.group(1).lower()
            for match in SUBPROGRAM_DEFINITION.finditer(INTERFACE_BLOCK.sub("", text))
        }
    return sorted(declared - defined)


def reaching(records: list[dict[str, Any]], targets: set[str]) -> dict[str, str]:
    """Subprogram name -> the undefined procedure it reaches, directly or not.

    ``spline3`` calls ``spline3pars``, which calls ``dgesv``; neither can be
    run against a reference whose ``dgesv`` is a refusal, and only this
    closure says so about the first one.
    """
    edges: dict[str, set[str]] = {}
    for record in records:
        for subprogram in record["subprograms"]:
            name = str(subprogram["name"]).lower()
            edges.setdefault(name, set()).update(
                str(callee).lower()
                for callee in (*subprogram["calls"], *subprogram.get("external_calls", ()))
            )
    found: dict[str, str] = {}
    for name in edges:
        seen: set[str] = set()
        pending = [name]
        while pending:
            current = pending.pop()
            for callee in sorted(edges.get(current, ())):
                if callee in targets:
                    found.setdefault(name, callee)
                    pending = []
                    break
                if callee not in seen:
                    seen.add(callee)
                    pending.append(callee)
    return found


def unresolved_stubs(names: list[str]) -> str:
    """A definition for each undefined external: one that refuses.

    The reference exists to say what the original program computes, and for a
    call into a library this build does not have it cannot say. A body that
    stops is the honest form of that: the symbol resolves, so the extension
    loads and the subprograms that never reach the library are compared as
    usual, and anything that does reach it stops where the missing library is
    rather than returning a number nobody computed. Nothing should reach one
    -- ``reaching`` leaves every caller ungated -- and if something does, this
    says which name was missing.

    No argument list: a Fortran external is resolved by name, and the callers
    were compiled against the interface the tree declared, not against this.
    """
    lines = [
        "! Machine-generated by recast for the reference build.",
        "! Procedures this build declares an INTERFACE for and defines nowhere:",
        "! the library the original program linked is not part of it.",
    ]
    for name in names:
        lines.extend(
            [
                f"subroutine {name}()",
                f'  error stop "recast reference build: {name} has no definition in this build"',
                f"end subroutine {name}",
            ]
        )
    return "\n".join(lines) + "\n"


def _generic_reach(record: dict[str, Any]) -> set[str]:
    """The specific procedures a public generic name reaches.

    ``record["public"]`` names the module's public entities and
    ``record["generics"]`` maps each generic to its specifics; a specific of a
    public generic is callable from outside the module even though its own
    name is private, which is the whole reason ``wrappers_for`` calls one
    through the generic.
    """
    public = {str(name).lower() for name in record.get("public") or ()}
    return {
        specific
        for generic, specifics in (record.get("generics") or {}).items()
        if str(generic).lower() in public
        for specific in specifics
    }


def unspellable(
    record: dict[str, Any],
    names: list[str],
    *,
    parameters: dict[str, Any] | None = None,
    dims_override: dict[str, str] | None = None,
) -> dict[str, str]:
    """Subprogram name -> why ``wrappers_for`` cannot write its wrapper.

    Answered by writing each one alone. What cannot be spelled -- a dtype with
    no Fortran form (COMPLEX has none here), a call-back whose interface the
    frontend did not resolve, an array result of deferred extent -- is decided
    in ``wrappers_for``, and a second copy of those rules here would be a
    second implementation to disagree with the first. The reason is the
    wrapper's own refusal, less the name it already prefixes.

    A public subprogram in this table is left *ungated* rather than failing
    the build: the reference is compiled for the rest of the module, and the
    verdict carries the name and the reason (``OracleRef.handle["ungated"]``),
    which is what the flat oracle already does (``recast.oracle.flat.unspellable``)
    and what ``unexercisable`` does for a wrapper that compiles but cannot be
    called. Failing instead cost every real-valued subprogram of a module its
    reference for the sake of one COMPLEX overload nobody required, and the
    coverage policy still refuses to count an ungated subprogram verified.
    """
    refused: dict[str, str] = {}
    for name in names:
        try:
            wrappers_for(record, [name], parameters=parameters, dims_override=dims_override)
        except ConfigError as error:
            reason = str(error)
            prefix = f"{name}: "
            refused[name] = reason[len(prefix) :] if reason.startswith(prefix) else reason
    return refused


def _wrappable(record: dict[str, Any], name: str) -> bool:
    """Whether ``wrappers_for`` can write this subprogram's wrapper; see
    ``unspellable`` for what decides it."""
    try:
        wrappers_for(record, [name])
    except ConfigError:
        return False
    return True


def wrappers_for(
    record: dict[str, Any],
    subprograms: list[str],
    parameters: dict[str, int] | None = None,
    dims_override: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Flat wrapper subroutines for the named subprograms of one module.

    Returns the wrapper source text and the wrapper names, one ``w_<name>``
    per subprogram. Optional arguments are dropped -- the translation spells
    them as keyword sentinels and the differential compares the required
    surface; a specific procedure of a generic is called through the generic
    name, because the specifics are private.

    ``parameters`` are integer constants the argument dimensions name but the
    file use-imports -- a grid's ``pcols`` and ``pver``, say -- emitted as local
    PARAMETERs so f2py can fold the declared shapes (the pipeline's
    ``gen_wrapper --local-params``). A file of bare subprograms gets no
    ``use`` line at all: the callee is an external, and the borrowed module
    name would not compile.
    """
    generic_of = {
        specific: generic
        for generic, specifics in record.get("generics", {}).items()
        for specific in specifics
    }
    table = {s["name"]: s for s in record["subprograms"]}
    interfaces = record.get("interfaces") or {}
    module = record["module"]
    is_module = record.get("is_module", True)
    # A submodule cannot be USEd; its procedures are reached through the
    # parent module whose interface declares them (#29).
    module = record.get("submodule_of") or module
    parameter_lines = [
        f"  integer, parameter :: {name} = {int(value)}"
        for name, value in (parameters or {}).items()
    ]
    pieces = ["! Machine-generated by recast (f2py-golden oracle) -- DO NOT EDIT.", ""]
    names = []
    for name in subprograms:
        sub = table[name]
        call_name = generic_of.get(name, name)
        arguments = [a for a in sub["args"] if not a.get("optional")]
        argument_names = [a["name"] for a in arguments]
        # What the call passes, which is the dummy's own name except where an
        # ALLOCATABLE dummy forces a local allocatable actual.
        actuals = list(argument_names)
        before: list[str] = []
        after: list[str] = []
        declarations = []
        hidden: list[str] = []
        converted: list[str] = []  # scalar LOGICAL INOUTs, through an integer
        # A derived-type dummy is spelled component by component: the
        # wrapper's dummies are the flat scalars, in the argument's place,
        # and the call passes a local of the type they were copied into.
        dummies: list[str] = []
        used_types: list[str] = []
        taken = {str(a["name"]).lower() for a in arguments}
        for argument in arguments:
            if argument["dtype"] == "PROCEDURE":
                dummies.append(argument["name"])
                try:
                    directives, external = _callback_declarations(argument, interfaces)
                except ConfigError as error:
                    raise ConfigError(f"{name}: {error}") from error
                declarations.append(external)
                declarations.extend(directives)
                continue
            spelled = FORTRAN_TYPES.get(argument["dtype"])
            derived = DERIVED.match(str(argument["dtype"])) if spelled is None else None
            if derived is not None:
                components = (
                    derived_components(record, argument, taken)
                    if is_module
                    else "a derived type of a file of bare subprograms cannot be used"
                )
                if isinstance(components, str):
                    raise ConfigError(
                        f"{name}: argument {argument['name']!r} has dtype "
                        f"{argument['dtype']!r}, which this wrapper cannot spell "
                        f"({components}); wrap it by hand or drop the subprogram from the gate"
                    )
                type_name = derived.group(1).lower()
                if type_name not in used_types:
                    used_types.append(type_name)
                taken.update(c["name"] for c in components)
                hands_in = argument["intent"] in ("IN", "INOUT", "UNKNOWN")
                hands_back = argument["intent"] != "IN"
                intent = "in" if not hands_back else "in out"
                declarations.append(f"  type({type_name}) :: {argument['name']}")
                for component in components:
                    dummies.append(component["name"])
                    declarations.append(
                        f"  {component['spelled']}, intent({intent}) :: {component['name']}"
                    )
                    if hands_in:
                        before.append(
                            f"  {argument['name']}%{component['component']} = {component['name']}"
                        )
                    if hands_back:
                        after.append(
                            f"  {component['name']} = {argument['name']}%{component['component']}"
                        )
                continue
            dummies.append(argument["name"])
            if spelled is None:
                raise ConfigError(
                    f"{name}: argument {argument['name']!r} has dtype "
                    f"{argument['dtype']!r}, which this wrapper cannot spell; "
                    "wrap it by hand or drop the subprogram from the gate"
                )
            intent = INTENT_SPELLING[argument["intent"]]
            if _logical_through_integer(argument):
                # A scalar LOGICAL INOUT has no portable Python buffer ABI
                # (the compiler's raw true is its own). The wrapper takes an
                # integer, 0 or 1, and converts on the way in and out; the
                # callee sees the logical it declared (PCHIP's ``skip``).
                declarations.append(f"  integer, intent(inout) :: {argument['name']}")
                declarations.append(f"  logical :: {argument['name']}_l")
                converted.append(argument["name"])
                continue
            if _passed_buffer(argument) or (
                argument["intent"] == "OUT" and argument.get("allocatable") and argument.get("dims")
            ):
                # A caller-buffer OUT array of no declared extent (``fe(*)``,
                # PCHIP's evaluators), or an OUT allocatable array: the array
                # is the caller's storage on both sides, so the reference takes
                # it the way the candidate does -- an argument, updated in
                # place. Spelling it intent(out) asks f2py to allocate a result
                # whose extent the wrapper never states -- ``dy(*)``,
                # ``x2(:, :)`` -- and every call died on "must have defined
                # dimensions but got (-1, -1)" (``meshgrid`` of the mesh
                # module, ``dcopy`` of SLSQP).
                #
                # An allocatable OUT array is shimmed through a local
                # allocatable (see ``_allocatable_shim``); its own dummy is the
                # caller's buffer, spelled ``in out`` as the shimmed dummies
                # are. A plain caller buffer (``dy(*)``) keeps ``inout``.
                intent = "in out" if argument.get("allocatable") else "inout"
            dims = ""
            override = (dims_override or {}).get(argument["name"])
            if override and argument.get("dims"):
                # An explicit override wins: a use-imported extent is
                # invisible to f2py, and the operator names it instead (the
                # pipeline's ``gen_wrapper --dims-override``).
                dims = f"({override})"
                _hide(override, argument_names, parameters, hidden)
            elif argument.get("dims"):
                # Spelled as the source declares it: an explicit extent, ``*``
                # for an assumed-size dummy, ``:`` for an assumed-shape one.
                # f2py's interface carries either alone; what it cannot carry
                # is the mix ``(incfd, :)`` that spelling ``*`` as ``:`` made.
                dims = "(" + ", ".join(_extent(d) for d in argument["dims"]) + ")"
            declarations.append(f"  {spelled}, intent({intent}) :: {argument['name']}{dims}")
            if argument["dtype"] == "str" and argument.get("path"):
                # The wrapper's dummy is a fixed width and the callee's is
                # ``len=*``: passed straight through, the callee would see 128
                # characters whatever the caller wrote. TRIM restores the
                # caller's own length, which is what the source's callers pass.
                # Only where the harness supplies the value -- a path it draws
                # -- because everywhere else the subprogram is ungated and the
                # actual would be changing a wrapper nothing calls.
                actuals[argument_names.index(argument["name"])] = f"trim({argument['name']})"
            if argument.get("allocatable"):
                actual, locals_, opening, closing = _allocatable_shim(argument, spelled)
                actuals[argument_names.index(argument["name"])] = actual
                declarations.extend(locals_)
                before.extend(opening)
                after.extend(closing)
        # An intent(out) dummy is undefined on entry, and a subprogram that
        # returns early -- a guard rejecting its own arguments -- never
        # assigns it. What f2py then hands back is whatever the buffer it
        # allocated happened to hold, which is not a fact about the Fortran
        # and not something any translation can be held to. Defined here,
        # once, so the reference's output buffers start where the emitted
        # translation's do (see ``undefined_array``) and an output neither
        # side wrote compares equal instead of comparing two heaps.
        #
        # A caller-buffer array (``buffer``) is not the wrapper's to define:
        # it is the caller's storage on both sides, so the gate generates it
        # and hands the same values to the reference and the candidate, and
        # zeroing it here would leave every cell the callee never writes at
        # 0 against the candidate's generated value. An assumed-size dummy,
        # which is always a buffer, cannot be assigned whole anyway.
        defined = [
            f"  {a['name']} = {DEFINED_ZERO.get(a['dtype'], '0')}"
            for a in arguments
            if a["intent"] == "OUT" and a["dtype"] != "PROCEDURE" and not a.get("buffer")
        ]
        wrapper = f"w_{name}"
        names.append(wrapper)
        # A scalar LOGICAL INOUT: the integer in, the logical to the callee,
        # the integer out again. The conversions join the copies the
        # derived-type and allocatable dummies already queued either side of
        # the call, and a converted argument's actual is its logical local.
        for a in converted:
            actuals[argument_names.index(a)] = f"{a}_l"
        before += [f"  {a}_l = ({a} /= 0)" for a in converted]
        after += [f"  {a} = merge(1, 0, {a}_l)" for a in converted]
        use_line = (
            [f"  use {module}, only: {', '.join([call_name, *used_types])}"] if is_module else []
        )
        external_line = [] if is_module else [f"  external {call_name}"]
        result_dims = sub.get("result_dims") or [] if sub["kind"] == "function" else []
        if result_dims:
            # An array-valued function result needs its extents too, spelled
            # the way a dummy's are, and f2py cannot wrap an array-valued
            # *function*: the wrapper becomes a subroutine whose ``res`` is
            # an intent(out) dummy. An extent naming neither an argument nor
            # a local parameter becomes a hidden integer dummy (#17).
            if any(d.get("ub") is None for d in result_dims):
                raise ConfigError(
                    f"{name}: array-valued result with a deferred/assumed extent "
                    "is not wrappable; wrap it by hand or drop the subprogram from the gate"
                )
            result = FORTRAN_TYPES.get(sub["result_dtype"], "real(8)")
            extents = [str(d["ub"]) for d in result_dims]
            for extent in extents:
                _hide(extent, argument_names, parameters, hidden)
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"subroutine {wrapper}({', '.join([*dummies, *hidden, 'res'])})",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *declarations,
                f"  {result}, intent(out) :: res({', '.join(extents)})",
                *([f"  {result}, external :: {call_name}"] if not is_module else []),
                *before,
                f"  res = {call_name}({', '.join(actuals)})",
                *after,
                f"end subroutine {wrapper}",
                "",
            ]
        elif sub["kind"] == "function":
            result = FORTRAN_TYPES.get(sub["result_dtype"], "real(8)")
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"function {wrapper}({', '.join([*dummies, *hidden])}) result(res)",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *declarations,
                f"  {result} :: res",
                *([f"  {result}, external :: {call_name}"] if not is_module else []),
                *before,
                f"  res = {call_name}({', '.join(actuals)})",
                *after,
                f"end function {wrapper}",
                "",
            ]
        else:
            declarations += [f"  integer, intent(in) :: {token}" for token in hidden]
            pieces += [
                f"subroutine {wrapper}({', '.join([*dummies, *hidden])})",
                *use_line,
                "  implicit none",
                *parameter_lines,
                *external_line,
                *declarations,
                *defined,
                *before,
                f"  call {call_name}({', '.join(actuals)})",
                *after,
                f"end subroutine {wrapper}",
                "",
            ]
    return "\n".join(line for piece in pieces for line in _fold(piece)) + "\n", names


def companion_sources(facts: Facts, root: Path) -> list[Path]:
    """The sibling files this unit ``use``s, dependencies first.

    A module that takes its working precision from a kinds module one file
    over does not compile alone: gfortran wants the ``.mod``, and the file
    that would produce it was never handed to the build. The frontend already
    resolved those siblings -- ``Facts.provenance['companions']`` names them
    -- so the reference build asks the facts rather than the operator.

    ``provenance['companion_dependencies']`` is the rest of that closure: what
    the companions themselves ``use``. None of it is visible in this unit, so
    none of it is a companion, but a companion is compiled from source here
    and a compiler wants the ``.mod`` under it too -- without them the build
    stops at "cannot open module file" on a file the unit never named.

    ``config['extra_sources']`` stays what it always was: files from outside
    the tree, which nothing in the tree can name. Ordering is a topological
    sort over the companions' own ``use`` statements, because a Fortran
    compiler cannot read a module it has not compiled yet, and the ones that
    depend on nothing here come first.
    """
    root = _resolved_root(root)
    companions = facts.provenance.get("companions") or []
    if not isinstance(companions, list):
        raise ConfigError("Facts.provenance['companions'] must be a list")
    dependencies = facts.provenance.get("companion_dependencies") or []
    if not isinstance(dependencies, list):
        raise ConfigError("Facts.provenance['companion_dependencies'] must be a list")
    by_module: dict[str, tuple[dict[str, Any], Path]] = {}
    for index, companion in enumerate([*companions, *dependencies]):
        if not isinstance(companion, dict):
            raise ConfigError(f"companion {index} must be an object")
        module = str(companion.get("module", "")).lower()
        if module in by_module:
            raise ConfigError(f"duplicate companion module {module!r} in Facts provenance")
        path = _source_under_root(
            root,
            companion.get("source"),
            label=f"companion {module or index} source",
        )
        by_module[module] = (companion, path)
    ordered: list[Path] = []
    placed: set[str] = set()

    def place(name: str, stack: frozenset[str]) -> None:
        item = by_module.get(name)
        if item is None or name in placed or name in stack:
            # A cycle is not this build's to resolve -- Fortran allows mutual
            # use only through submodules, and stopping keeps the order total.
            return
        companion, path = item
        for statement in companion.get("record", {}).get("use_statements", ()):
            match = re.match(r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", statement.strip(), re.I)
            if match:
                place(match.group(1).lower(), stack | {name})
        if name in placed:
            return
        placed.add(name)
        ordered.append(path)

    for name in sorted(by_module):
        place(name, frozenset())
    return ordered


def _log_tail(output: str, limit: int = _BUILD_LOG_TAIL_CHARS) -> str:
    """The last ``limit`` characters of ``output``, cut at a line boundary and
    saying how much came before."""
    text = output.strip()
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    newline = tail.find("\n")
    if 0 <= newline < limit // 2:
        tail = tail[newline + 1 :]
    return f"… [{len(text) - len(tail)} earlier characters of the build output omitted]\n{tail}"


def _compiler_version(compiler: str) -> str:
    """The compiler's own version line. A metadata query, not a build --
    which is why it does not go through the executor: the *key* has to fold
    the version in before anyone decides whether to build at all."""
    try:
        out = subprocess.run(  # noqa: S603
            [compiler, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError as exc:
        raise ConfigError(
            f"Fortran compiler {compiler!r} is not runnable ({exc}); "
            "install gfortran or point config['fc'] at one"
        ) from exc
    if out.returncode != 0:
        raise ConfigError(f"{compiler!r} --version failed: {out.stderr.strip()[:200]}")
    return out.stdout.splitlines()[0].strip()


class F2pyGoldenOracle(Oracle):
    """Compile the untouched Fortran and hand back a callable truth module."""

    name = "f2py-golden"
    cost = "build"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        compiler = config.get("fc", "gfortran")
        digest = hashlib.sha256()
        digest.update(str(facts.provenance.get("digest")).encode())
        root = _resolved_root(config.get("root", "."))
        # Trust the bytes, not only a digest carried in mutable Facts.  This
        # also makes the key validation exercise the same root boundary as the
        # materializer before it queries a compiler or creates a workspace.
        digest.update(self._main_source_digest(facts, root).encode())
        dependencies = [*companion_sources(facts, root), *_extra_sources(config)]
        for path in sorted(dependencies, key=str):
            digest.update(str(path).encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(_compiler_version(compiler).encode())
        digest.update(config.get("fflags", DEFAULT_FLAGS).encode())
        digest.update(str(sorted((config.get("wrapper_parameters") or {}).items())).encode())
        digest.update(str(sorted((config.get("wrapper_dims") or {}).items())).encode())
        digest.update(",".join(self._subprograms(facts, config)).encode())
        return f"f2py:{facts.interface.get('module', unit.uid)}:{digest.hexdigest()[:16]}"

    def _main_source(self, facts: Facts, root: Path) -> Path:
        """The unit's own source, proven to lie under ``root``.

        An oracle that writes the source it wraps (``F2pyFlatOracle``)
        overrides this with the file it wrote.
        """
        return _source_under_root(root, facts.provenance.get("source"), label="main source")

    def _main_source_digest(self, facts: Facts, root: Path) -> str:
        """sha256 of the main source's bytes, read for the key."""
        return hashlib.sha256(self._main_source(facts, root).read_bytes()).hexdigest()

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        # The key of exactly the facts this build is of. A subclass that keys
        # a wider plan (``F2pyFlatOracle``) hands facts of its own making in
        # here and files the ref under its own key once the build is done.
        key = F2pyGoldenOracle.key(self, unit, facts, config)
        build = workspace / f"oracle-{key.rsplit(':', 1)[-1]}"
        build.mkdir(parents=True, exist_ok=True)

        root = _resolved_root(config.get("root", "."))
        source = self._main_source(facts, root)
        companions = companion_sources(facts, root)
        extras = _extra_sources(config)
        subprograms = self._subprograms(facts, config)
        if not subprograms:
            # Nothing callable to wrap -- a module of kind parameters, or of
            # abstract interfaces. f2py is happy to build an extension with an
            # empty ``only:`` list, and importing the result segfaults the
            # interpreter, which no ``except`` can catch and which takes the
            # whole run with it. A reference to nothing is not a reference.
            raise OracleUnavailable(
                f"{unit.uid}: no public subprogram to wrap; there is no reference to build"
            )
        # A public subprogram the wrapper cannot spell is left ungated, name
        # and reason on the handle, and the reference is built for the rest.
        # An operator's explicit list is different: a name they wrote is a
        # name they meant, and refusing it loudly is the answer to "wrap it
        # by hand or drop the subprogram from the gate".
        refused: dict[str, str] = {}
        if not config.get("subprograms"):
            refused = unspellable(
                facts.interface,
                subprograms,
                parameters=config.get("wrapper_parameters"),
                dims_override=config.get("wrapper_dims"),
            )
            subprograms = [name for name in subprograms if name not in refused]
            if not subprograms:
                raise OracleUnavailable(
                    f"{unit.uid}: no public subprogram this wrapper can spell; "
                    + "; ".join(f"{n} ({why})" for n, why in sorted(refused.items()))
                )
        wrapper_text, wrapper_names = wrappers_for(
            facts.interface,
            subprograms,
            parameters=config.get("wrapper_parameters"),
            dims_override=config.get("wrapper_dims"),
        )
        # ``only:`` names the wrappers to build -- and drops f2py's own
        # ``__user__routines`` module with them, leaving the generated C
        # referring to a call-back type nothing declared. The wrapper file
        # holds nothing but these wrappers, so the list is a restriction to
        # everything, and saying nothing builds the same set.
        selection = [] if "!f2py" in wrapper_text else ["only:", *wrapper_names, ":"]

        compiler = config.get("fc", "gfortran")
        module_name = f"ref_{facts.interface['module']}"
        # Companions first, then whatever the operator added, then the unit's
        # own source: gfortran compiles in argument order and a ``use`` of a
        # module later in the list is a fatal "cannot open module file".
        original_sources = [*companions, *extras, source]
        stage = Path(tempfile.mkdtemp(prefix="f2py-stage-", dir=build))
        sources, wrapper, include_args = _stage_build_inputs(stage, original_sources, wrapper_text)
        # A procedure the tree declares an interface for and defines nowhere
        # is a call into a library this build does not link. Left alone, the
        # extension links with the symbol undefined and *importing* it fails,
        # which costs the module's every other subprogram its reference too.
        records = build_records(facts)
        unresolved = undefined_externals(records, extras)
        # A few of them recast can define rather than refuse, and defines the
        # same way on the other side (``recast.references``): those are not
        # blocked, because there *is* a reference for a subprogram that
        # reaches one -- one that stood in for the library, which the verdict
        # says. The rest keep the body that refuses, and keep disclaiming
        # their callers.
        substituted = references.supported(unresolved)
        missing = [name for name in unresolved if name not in set(substituted)]
        blocked = reaching(records, set(missing)) if missing else {}
        if substituted:
            supplied = Path("sources") / "recast_references.f90"
            (stage / supplied).write_text(references.fortran_for(substituted))
            sources.append(supplied.as_posix())
        if missing:
            stub = Path("sources") / "unresolved.f90"
            (stage / stub).write_text(unresolved_stubs(missing))
            sources.append(stub.as_posix())
        # fflags remains the operator's compiler-flags string.  Source/include
        # paths never join it: NumPy splits this value internally, so appending
        # an original directory here would let whitespace and flag-looking
        # path components become compiler options.
        flags = config.get("fflags", DEFAULT_FLAGS)
        objects = self._compile_reference(
            unit, sources, stage, build, compiler, flags, include_args, executor, config
        )
        job = Job(
            argv=[
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                "--build-dir",
                "f2py-build",
                wrapper,
                *objects,
                *include_args,
                f"-I{MODULE_DIR}",
                "-m",
                module_name,
                *selection,
                f"--f90flags={flags}",
                f"--f77flags={flags}",
                "--backend",
                "meson",
            ],
            cwd=stage,
            # The whole environment plus the compiler overrides: f2py needs a
            # real PATH, and the local executor passes exactly what it is given.
            # The interpreter's own bin directory rides in front so the build
            # backend (meson, ninja) installed beside numpy is found.
            env={
                **os.environ,
                "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
                "FC": compiler,
                "F90": compiler,
            },
            timeout_s=float(config.get("build_timeout", 600)),
            label=f"f2py {module_name}",
        )
        try:
            result = executor.run(job)
        except RecastError:
            raise
        except Exception as error:
            # An executor that refuses -- it cannot honestly supply what the job
            # asked for -- is the case ``OracleUnavailable`` exists for, and it
            # has to arrive as one. The runner catches ``RecastError`` and marks
            # this unit's oracle stage failed; anything else escapes it and takes
            # the whole run down, so a refusal nobody wrapped costs the other
            # units their verdicts as well as this one.
            raise OracleUnavailable(
                f"executor {getattr(executor, 'name', type(executor).__name__)!r} did not "
                f"run the f2py build for {unit.uid}: {type(error).__name__}: {error}"
            ) from error
        if not result.ok:
            log = build / "f2py.log"
            output = result.stdout + "\n" + result.stderr
            log.write_text(output)
            # The log lives in a workspace that is gone once the run returns,
            # so the error carries the end of it -- where crackfortran and the
            # compiler say what they could not parse -- not just its path.
            raise ConfigError(
                f"f2py build for {unit.uid} failed (exit {result.returncode}); log at {log}\n"
                + _log_tail(output)
            )

        # The reference runs in a process of its own, so that an ``error
        # stop`` in it is an answer rather than the end of the run (#21).
        # Not when a wrapped subprogram takes a procedure: an f2py call-back
        # is a Python object of *this* process that the reference calls, and
        # it cannot be handed across. Those references stay in-process, and
        # the handle says so.
        takes_callbacks = sorted(
            sub["name"]
            for sub in facts.interface.get("subprograms", ())
            if sub["name"] in subprograms
            and any(a.get("dtype") == "PROCEDURE" for a in sub.get("args", ()))
        )
        asked = str(config.get("reference_isolation", "process"))
        if asked == "in-process":
            isolation = "in-process (configured)"
        elif takes_callbacks:
            isolation = f"in-process (call-back arguments: {', '.join(takes_callbacks)})"
        else:
            isolation = "process"
        if isolation == "process":
            from recast.oracle.isolated import IsolatedModule

            module: Any = IsolatedModule(stage, module_name, log_dir=build)
        else:
            sys.path.insert(0, str(stage))
            try:
                module = importlib.import_module(module_name)
            finally:
                sys.path.remove(str(stage))
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=key,
            handle={
                "module": module,
                "wrappers": dict(zip(subprograms, wrapper_names, strict=True)),
                # Derived-type dummies the wrappers spell component by
                # component; the verifier splits the candidate's the same way.
                "flattened": flattened_dummies(facts.interface, subprograms),
                "ungated": {
                    **{
                        s["name"]: reason
                        for s in facts.interface["subprograms"]
                        if s["name"] in set(subprograms)
                        and (
                            reason := (
                                unexercisable(s)
                                or (
                                    f"reaches {blocked[s['name']]}, which no source in this "
                                    "build defines"
                                    if s["name"] in blocked
                                    else None
                                )
                            )
                        )
                        is not None
                    },
                    **refused,
                },
                # What this build stood in for rather than linked. Not a
                # narrowing of the comparison -- both sides ran it -- but a
                # condition on it, and one the evidence has to carry: the
                # numbers a subprogram reaching one of these was compared at
                # are not the numbers the library would have produced.
                "substituted": {name: references.reason(name) for name in substituted},
                "build_dir": stage,
                "isolation": isolation,
            },
            cost=self.cost,
        )

    def _compile_reference(
        self,
        unit: Unit,
        sources: list[str],
        stage: Path,
        build: Path,
        compiler: str,
        flags: str,
        include_args: list[str],
        executor: Executor,
        config: dict[str, Any],
    ) -> list[str]:
        """Compile the reference sources, and hand f2py the objects.

        f2py's own parser only ever sees the generated wrapper. The reference
        is compiled by the Fortran compiler, which is the only thing here that
        actually understands Fortran: crackfortran does not know BLOCK
        constructs, submodules, or a dozen other things a modern file is
        written with, and it fails the *whole* build on one of them -- an
        oracle refused over a construct nobody was asking it to wrap. The
        objects ride into ``f2py -c`` as extra objects, which it passes
        straight to the linker, and the ``.mod`` files the wrapper's ``use``
        needs are left in one directory both halves of the build can name.

        Sources are compiled in the order they were staged, because a ``use``
        of a module compiled later is a fatal "cannot open module file".
        """
        objects: list[str] = []
        for index, relative in enumerate(sources):
            # A bare name in the job's own directory: NumPy's Meson backend
            # copies an extra object into the build directory but names it in
            # ``meson.build`` by the path it was given, so only a basename
            # resolves from both places.
            obj = f"object_{index:04d}.o"
            job = Job(
                argv=[
                    compiler,
                    "-c",
                    "-fPIC",
                    *flags.split(),
                    f"-J{MODULE_DIR}",
                    f"-I{MODULE_DIR}",
                    *include_args,
                    relative,
                    "-o",
                    obj,
                ],
                cwd=stage,
                env={**os.environ},
                timeout_s=float(config.get("build_timeout", 600)),
                label=f"{compiler} {relative}",
            )
            try:
                result = executor.run(job)
            except RecastError:
                raise
            except Exception as error:
                raise OracleUnavailable(
                    f"executor {getattr(executor, 'name', type(executor).__name__)!r} did not "
                    f"run the reference compile for {unit.uid}: {type(error).__name__}: {error}"
                ) from error
            if not result.ok:
                log = build / "reference.log"
                output = result.stdout + "\n" + result.stderr
                log.write_text(output)
                raise ConfigError(
                    f"reference compile of {relative} for {unit.uid} failed "
                    f"(exit {result.returncode}); log at {log}\n" + _log_tail(output)
                )
            objects.append(obj)
        return objects

    @staticmethod
    def _subprograms(facts: Facts, config: dict[str, Any]) -> list[str]:
        named = config.get("subprograms")
        if named:
            return list(named)
        record = facts.interface
        # Public only, because the wrappers `use` the module: a private
        # symbol is not importable and the build fails on the whole file.
        # A specific procedure of a public generic is the exception
        # ``wrappers_for`` already makes -- it calls one through the generic
        # name, which *is* importable -- so it is reachable too. Without it a
        # module that publishes nothing but generics has no public subprogram
        # at all and gets no reference: the corpus's sorting module declares
        # `public sort, sortpairs, argsort` over twelve private specifics, and
        # every one of them was skipped here while the wrapper stood ready to
        # write it.
        reached = _generic_reach(record)
        return [
            s["name"]
            for s in record["subprograms"]
            # Reached, not exported: a public name whose interface this
            # wrapper cannot spell stays in the list, because the module says
            # it is part of its surface -- ``materialize`` then leaves it
            # ungated by name and reason (``unspellable``) -- but one that is
            # merely reachable is dropped instead: a complex-valued overload
            # of a generic must not cost its ten siblings their reference.
            if s.get("public", True) or (s["name"] in reached and _wrappable(record, s["name"]))
        ]


def factory(**_config: Any) -> F2pyGoldenOracle:
    return F2pyGoldenOracle()
