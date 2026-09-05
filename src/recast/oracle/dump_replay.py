"""``dump-replay``: inputs and outputs recorded from a real run, as the reference.

The cheapest of the three oracles and the only one that computes nothing. An
``f2py-golden`` reference is compiled and called; a ``numpy-anchor`` is derived
and called; this one is *read*. A probe injected into a production run wrote
what went into a subprogram and what came out of it, and replaying that is the
reference.

That difference is not a detail of cost. Every other oracle answers "what would
the original do with these inputs", so the harness may choose the inputs. This
one cannot be asked a question it was not already asked: the inputs are
whatever the model actually ran on, and the outputs are what it actually
produced. **So the reference supplies the inputs**, which is the inverse of the
gate's usual direction, and it says so on its handle with ``input_source``.

What that buys is the one thing generated inputs cannot give: the points are
the model's own operating point rather than a sampling of the declared domain.
A range that never occurs in a real run is not tested, and a correlation
between arguments that a sampler would never produce is.

What it costs is stated too, because a replay verdict is weaker than it looks:

* **It covers what was recorded and nothing else.** Coverage is a property of
  the run that produced the dumps, not of this oracle, and no amount of
  re-running it widens the set.
* **It cannot re-derive.** If the recorded outputs are wrong -- a probe placed
  after the wrong statement, a run whose build differs from the one under
  test -- nothing here can tell. The other two oracles recompute and would.
* **The recording carries no build identity.** The dump format is values and
  extents; it does not say which compiler or flags produced them. The cache key
  can only fold what the files contain, so an operator who re-records under a
  different build gets a different digest, and one who does not gets the same
  key for a different reference.

The parser is relayed from ``pipeline/dump_verify.py``'s ``parse_dump_file``
and holds to its behaviour rather than tidying it: the probes in the field emit
this format, and a parser that reads it *nearly* the same way is worse than one
that reads it exactly the same way. ``tools/dump_diff.py`` is what says it
does.

Nothing here knows CESM. ``dump_verify.py`` also carried CAM's init constants,
a fuzzy dump-name-to-argument matcher and a hard-coded preload of two
translated modules; the first is the domain extension's, and the other two are
the verifier's half of that script, which this engine already has.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from recast.errors import ConfigError, OracleUnavailable
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.plugins.oracle import Oracle

__all__ = ["DumpReplayOracle", "factory", "parse_dump"]

_SECTION = re.compile(r"#\s*(INPUT|OUTPUT):\s*(\w+)(?:\(([^)]*)\))?")
"""``# INPUT: qv(mgncol,nlev)`` -- the section header the probes write."""

_SCALAR = re.compile(r"#\s*(\w+)\s*=\s*(.+)")
"""``# mgncol = 10`` -- a header scalar, which is also an input."""

_PROBE = re.compile(r"#\s*PROBE\s+(\w+)\.(\w+)\s*:")
"""``# PROBE micro_mg_utils.size_dist_param_liq: call=    1`` -- the line the
generated probes write first, naming what they are a recording *of*.

**Upstream's parser drops this line**: neither its section regex nor its
scalar regex matches it, so the subprogram's identity is lost and the script
that consumes the dump has to guess, trying every subprogram in the module and
binding dump variables to arguments by substring match with zeros for whatever
is left over. Reading the line the probe already writes is additive -- the
inputs and outputs parse identically either way, which is what
``tools/dump_diff.py`` checks -- and it retires the guessing rather than
relaying it, because a gate that binds the wrong array by substring and fills
the rest with zeros can report a pass it has not earned."""

_IMPLICIT_EXPONENT = re.compile(r"([+-]?\d+\.\d+)([+-]\d+)$")
_INTEGER_LITERAL = re.compile(r"^[+-]?\d+$")
"""Fortran ``G`` editing drops the ``E`` when the exponent needs three digits,
so ``1.0701116457083034-114`` is ``1.07...e-114`` and not a subtraction. Python
cannot read it and the probes emit it, so it is read here."""

_NOT_THE_REFERENCE = frozenset({"root", "workspace", "store_root", "dumps"})
"""Config keys that locate rather than describe. ``dumps`` is excluded for the
reason the others are and one more: the files' *content* is folded in below, so
folding the path as well would give two machines different keys for the same
recording."""


def parse_dump(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """One probe dump file into ``(inputs, outputs)``, each ``{name: value}``.

    Relayed from ``dump_verify.parse_dump_file``, including the parts that
    would be written differently today. Three of them are load-bearing and are
    kept deliberately:

    * a header scalar lands in ``metadata`` *and* in ``inputs``, because an
      extent like ``mgncol`` is both the shape of the arrays below it and an
      argument the subprogram takes;
    * an extent that names a scalar not in the header falls back to the value
      count, so a rank-1 array is read as its own length rather than refused;
    * a value that parses as neither a float nor an implicit-exponent literal
      is skipped rather than raising, which is how a probe's own trailing
      diagnostics survive being in the middle of a data section.

    The reshape is ``order="F"``: the probe walks the array in Fortran order.
    """
    import numpy as np

    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    target: str | None = None
    name: str | None = None
    dims_text: str | None = None
    values: list[float] = []
    all_integers = True  # every value of the section so far is an integer literal

    def flush() -> None:
        if not name or not values:
            return
        # The recorder writes integer arrays with ``i0`` and reals with an
        # exponent, so a section whose every value is an integer literal is an
        # integer array: read it as int32, which is what the signature declares
        # and what the exact comparison of a declared-integer output expects.
        array = np.array(values, dtype=np.int32 if all_integers else np.float64)
        if dims_text:
            shape: list[int] = []
            for token in dims_text.split(","):
                extent = token.strip().lower()
                if extent in metadata:
                    shape.append(int(metadata[extent]))
                else:
                    try:
                        shape.append(int(extent))
                    except ValueError:
                        shape.append(len(values))
            if shape:
                try:
                    array = array.reshape(tuple(shape), order="F")
                except ValueError:
                    # The declared extents do not multiply out to what was
                    # written. Upstream keeps the flat array rather than
                    # refusing, and so does this: the shape is the probe's
                    # claim about the data, and the data is the data.
                    pass
        (inputs if target == "INPUT" else outputs)[name] = array

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        section = _SECTION.match(line)
        if section:
            flush()
            target, name, dims_text = section.group(1), section.group(2).lower(), section.group(3)
            values = []
            all_integers = True
            continue

        if line.startswith("#"):
            scalar = _SCALAR.match(line)
            if scalar:
                key = scalar.group(1).lower()
                text_value = scalar.group(2).strip()
                try:
                    if "." in text_value or "E" in text_value.upper() or "D" in text_value.upper():
                        number = float(text_value.replace("D", "E").replace("d", "e"))
                        metadata[key] = number
                        inputs[key] = np.float64(number)
                    else:
                        whole = int(text_value)
                        metadata[key] = whole
                        inputs[key] = np.int32(whole)
                except ValueError:
                    # Not a number: a character scalar the probe wrote as
                    # text (``# phase = sun``), which is the value it is.
                    # Upstream swallowed this with a bare ``except``.
                    metadata[key] = text_value
                    inputs[key] = text_value
            continue

        try:
            values.append(float(line.replace("D", "E").replace("d", "e")))
            all_integers = all_integers and _INTEGER_LITERAL.match(line) is not None
        except ValueError:
            implicit = _IMPLICIT_EXPONENT.match(line)
            if implicit:
                values.append(float(f"{implicit.group(1)}E{implicit.group(2)}"))
                all_integers = False

    flush()
    return inputs, outputs


def parse_probe_header(text: str) -> tuple[str, str] | None:
    """``(module, subprogram)`` the recording is of, or ``None`` if it says.

    Separate from ``parse_dump`` rather than folded into it, so that what the
    differential compares stays exactly what upstream's parser returns.
    """
    for raw in text.splitlines():
        probe = _PROBE.match(raw.strip())
        if probe:
            return probe.group(1).lower(), probe.group(2).lower()
    return None


class DumpReplayOracle(Oracle):
    """Recorded samples from a production run, replayed as the reference."""

    name = "dump-replay"
    cost = "cheap"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        """Fold what the recording *is*, which is the bytes of its files.

        A dump carries no build identity -- no compiler, no flags, no revision
        -- so unlike ``f2py-golden`` there is nothing here to fold about how it
        was produced. The bytes of the files are the reference: two recordings
        that are byte-identical are the same reference, and one re-recorded
        under a different build is a different one whether or not anybody says
        so.

        **The source digest is folded in as well, though the recording does not
        depend on it.** That looks redundant and is not. A recording is of a
        particular revision of the code; when the source moves and the
        recording does not, replaying it compares the new candidate against
        what the *old* code produced -- a stale reference, which is the failure
        mode this key exists to prevent and the one that leaves no trace when
        it happens. Folding the digest means the Verdict's oracle key says
        which source the recording was a recording of, and re-recording is
        visible as the key moving rather than as nothing at all.
        """
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        for path in self._dump_files(config):
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(str(facts.provenance.get("digest")).encode())
        digest.update(_stable(config).encode())
        module = facts.interface.get("module", unit.uid)
        return f"dump-replay:{module}:{digest.hexdigest()[:16]}"

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        """Read the recording. Nothing is built and nothing is submitted.

        No ``executor`` work, deliberately: there is no reference run to make,
        which is the whole reason this oracle is ``cheap``. It is in the
        signature because the contract has one, not because a replay needs one.
        """
        files = self._dump_files(config)
        if not files:
            raise OracleUnavailable(
                f"{self.name} found no dump files under {config.get('dumps')!r}; "
                "a replay oracle has no way to derive a reference it was not given"
            )

        samples = []
        unnamed = 0
        for path in files:
            text = path.read_text()
            try:
                recorded_in, recorded_out = parse_dump(text)
            except Exception as error:  # a malformed recording is not a pass
                raise OracleUnavailable(f"{path.name} does not parse: {error}") from error
            if not recorded_in:
                # A file with no inputs cannot be replayed against anything.
                # Upstream counts these as skips and goes on; here it is only
                # a sample that is not offered, and the count is reported so
                # the difference between "recorded nothing" and "recorded a
                # pass" stays visible in the Verdict's metrics.
                continue
            probe = parse_probe_header(text)
            if probe is None:
                # Which subprogram this is a recording of is the one thing a
                # replay cannot infer, and guessing is what this oracle
                # declines to do. Counted rather than dropped silently.
                unnamed += 1
                continue
            samples.append(
                {
                    "source": path.name,
                    "module": probe[0],
                    "subprogram": probe[1],
                    "inputs": recorded_in,
                    "outputs": recorded_out,
                }
            )

        if not samples:
            raise OracleUnavailable(
                f"{self.name} parsed {len(files)} file(s) under {config.get('dumps')!r} "
                f"and none of them yielded a replayable sample "
                f"({unnamed} carried no '# PROBE <module>.<sub>:' header)"
            )

        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=self.key(unit, facts, config),
            handle={
                # The declaration that inverts the gate. Every other reference
                # answers questions the harness asks; this one only replays
                # questions that were already asked, so the harness must take
                # its inputs from here rather than generating them.
                "input_source": "recorded",
                "samples": samples,
                "unreadable": len(files) - len(samples),
                "unnamed": unnamed,
                # There is no module to call. Said explicitly rather than left
                # absent, because "no reference module" is the normal state of
                # this oracle and not a failure to build one.
                "module": None,
                "wrappers": {},
                # The recorded outputs are looked up by argument name, so
                # neither of the two calling conventions applies.
                "return_convention": "recorded",
                "arg_naming": "lower",
                # What machine produced the recording is not in the recording.
                # An operator who knows may say so; guessing would put a false
                # fact in the Verdict.
                "device": config.get("reference_device"),
            },
            cost=self.cost,
        )

    @staticmethod
    def _dump_files(config: dict[str, Any]) -> list[Path]:
        """The recording's files, in a fixed order.

        Sorted, because the samples' order reaches the Verdict's metrics and a
        verdict that depends on directory iteration order is not reproducible.
        """
        location = config.get("dumps")
        if not location:
            raise ConfigError(
                "the dump-replay oracle requires 'dumps': the directory of "
                "captured inputs/outputs to replay"
            )
        root = Path(location)
        if not root.is_dir():
            raise OracleUnavailable(f"{location!r} is not a directory of dumps")
        return sorted(p for p in root.iterdir() if p.suffix == ".txt" and p.is_file())


def _stable(config: dict[str, Any]) -> str:
    """A canonical string for the part of a config that describes the reference."""
    described = {k: v for k, v in sorted(config.items()) if k not in _NOT_THE_REFERENCE}
    parts = []
    for name, value in described.items():
        try:
            parts.append(f"{name}={json.dumps(value, sort_keys=True)}")
        except TypeError:
            parts.append(f"{name}=<present>")
    return ";".join(parts)


def factory(**_config: Any) -> DumpReplayOracle:
    return DumpReplayOracle()
