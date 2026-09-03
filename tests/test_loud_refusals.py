"""Policy: an incomplete translation is loud, recorded, and cannot pass.

Every site here once did something quieter -- a comment where a refusal
belonged, a ``pass`` past a DATA statement it could not honour, an I/O stub
that dropped the writes it stood in for, a subprogram silently missing from
the emitted module, a differential gate that judged coverage against whatever
survived its filters. Each test pins the loud behaviour at the layer that owns
it, so that a later change which makes a result look better by saying less
fails here first.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import constants, interface, rwset
from recast.fortran._parse import f03, parse, walk
from recast.model import Confidence
from recast.transform.numpy.modules import Modules
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.profiles import PROFILES

KINDS = {"wp_r8": "float64"}

SOURCE = """\
module loud_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8) :: folded = sin(1.0_r8)

  type wide_t
    real(r8) :: rows(nowhere)
  end type wide_t

contains

  subroutine prologue_refusals(n, out1, w)
    integer, intent(in) :: n
    real(r8), intent(out) :: out1(max(n, 2))
    type(wide_t), intent(out) :: w
    integer, parameter :: grid(2, 2) = reshape((/ 1, 2, 3, 4 /), (/ 2, 2 /))
    real(r8) :: scr(max(n, 2))
    out1 = 0.0_r8
    scr = 0.0_r8
    w%rows = 0.0_r8
  end subroutine prologue_refusals

  subroutine folded_parameters(x, label)
    real(r8), intent(inout) :: x
    character(len=6), intent(out) :: label
    real(r8), parameter :: twist = sin(0.5_r8)
    real(r8), parameter :: top = huge(1.0_r8)
    character(len=*), parameter :: tag = 'abc' // 'def'
    x = x * twist + top
    label = tag
  end subroutine folded_parameters

  subroutine short_data(x)
    real(r8), intent(inout) :: x
    real(r8) :: a, b
    data a, b / 1.0_r8 /
    x = x + a + b
  end subroutine short_data

  subroutine reads_input(unit, x)
    integer, intent(in) :: unit
    real(r8), intent(out) :: x
    read(unit, *) x
  end subroutine reads_input

  subroutine scoped(n, total)
    integer, intent(in) :: n
    real(r8), intent(out) :: total
    total = 0.0_r8
    inner: block
      real(r8) :: acc
      acc = real(n, r8)
      total = total + acc
    end block inner
  end subroutine scoped
end module loud_mod
"""


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("loud") / "loud_mod.f90"
    path.write_text(SOURCE)
    return path


@pytest.fixture(scope="module")
def renderer(source: Path) -> Modules:
    return Modules(
        subprograms=Subprograms(
            record=interface.extract(source, kind_assumptions=KINDS),
            constants=constants.extract(source),
            profile=PROFILES["ifx"],
        ),
        companion_imports=(),
    )


def _node(source: Path, name: str) -> Any:
    return next(
        sub
        for sub in walk(parse(source), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )


def _deferred(report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in report if entry["status"] == "agent_queue"]


# --- the prologue -------------------------------------------------------------


def test_every_prologue_refusal_is_recorded_and_raises(source: Path, renderer: Modules) -> None:
    """Three refusal sites share one prologue: an out-arg whose extent nothing
    resolves, a local parameter whose initializer is not renderable, a local
    array with the same unresolvable extent, and a derived out-arg whose
    component has no static shape. Each is an agent_queue entry AND a raise
    at the site; none is a comment the code runs past."""
    node = _node(source, "prologue_refusals")
    lines, report = renderer.subprograms.render(node, "prologue_refusals")
    body = "\n".join(lines)
    deferred = _deferred(report)
    prologue = [entry for entry in deferred if entry["block"].startswith("P")]
    assert [entry["block"] for entry in prologue] == ["P001", "P002", "P003", "P004"]
    reasons = "\n".join(entry["reason"] for entry in prologue)
    assert "out-arg out1: allocation refused (dim expr 'MAX(n, 2)')" in reasons
    assert "out-arg w: INTENT(OUT) derived-type dummy not materialized" in reasons
    assert "local parameter grid" in reasons
    assert "local array scr: extent not resolvable (dim expr 'MAX(n, 2)')" in reasons
    # The old wording is gone, and every refusal raises.
    assert "allocation skipped" not in body
    assert "prologue skipped" not in body
    assert body.count("raise NotImplementedError(") >= len(prologue)
    # Each entry points at the lines it emitted, ending on its raise.
    for entry in prologue:
        low, high = entry["py_lines"]
        assert "# AGENT_QUEUE:" in lines[low]
        assert lines[high - 1].lstrip().startswith("raise NotImplementedError(")


def test_a_parameter_initializer_the_token_pass_only_uppercased_is_reparsed(
    source: Path, renderer: Modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sin(0.5_r8)`` came out as ``SIN(0.5)``: valid Python, a NameError
    the first time the subprogram ran, and no entry anywhere saying why.
    ``'abc' // 'def'`` came out as ``'ABC' // 'DEF'`` -- floor division of
    two strings, with the literal's case changed. Python parsing the text is
    not the text meaning the Fortran; what the token pass only spelled is
    read again as an expression, and refused if that fails."""
    lines, report = renderer.subprograms.render(
        _node(source, "folded_parameters"), "folded_parameters"
    )
    assert "    twist = math.sin(0.5)" in lines
    assert "    top = _f_huge(1.0)" in lines
    assert "    tag = ('abc' + 'def')" in lines
    assert not _deferred(report)
    # The whole module still imports with those spellings. Its companions
    # (the constants module, the used precision module) are not rendered
    # here, so empty stand-ins take their names.
    for companion in ("constants", "precision_mod_numpy"):
        monkeypatch.setitem(sys.modules, companion, types.ModuleType(companion))
    text, _ = renderer.render(source)
    assert "import math" in text.splitlines()
    namespace: dict[str, Any] = {}
    exec(compile(text, "loud_mod_numpy.py", "exec"), namespace)
    assert callable(namespace["folded_parameters"])


def test_prologue_entries_land_at_their_lines_in_the_finished_file(
    source: Path, renderer: Modules
) -> None:
    text, report = renderer.render(source)
    lines = text.splitlines()
    prologue = [entry for entry in report if entry["block"].startswith("P")]
    assert prologue
    for entry in prologue:
        first, last = entry["py_lines"]
        assert last >= first
        assert "# AGENT_QUEUE:" in lines[first - 1]
        assert lines[last - 1].lstrip().startswith("raise NotImplementedError(")


# --- DATA ---------------------------------------------------------------------


def test_a_refused_data_statement_raises_instead_of_passing(
    source: Path, renderer: Modules
) -> None:
    """Fewer values than objects is a DATA the translation cannot honour. It
    was already deferred; it also executed past a bare ``pass`` on zeroed
    state, which is a wrong number with nothing to say so at runtime."""
    lines, report = renderer.subprograms.render(_node(source, "short_data"), "short_data")
    entry = next(e for e in report if e["block"] == "D001")
    assert entry["status"] == "agent_queue"
    assert "fewer values" in entry["reason"]
    low, high = entry["py_lines"]
    emitted = lines[low:high]
    assert emitted[0].lstrip().startswith("# AGENT_QUEUE: DATA deferred:")
    assert emitted[1].lstrip().startswith("raise NotImplementedError(")
    assert not any("pass  # DATA" in line for line in lines)


# --- READ ---------------------------------------------------------------------


def test_read_is_refused_not_stubbed(source: Path, renderer: Modules) -> None:
    """READ writes its item list. A ``pass`` in its place dropped the writes
    silently -- the same hazard INQUIRE already refused over."""
    lines, report = renderer.subprograms.render(_node(source, "reads_input"), "reads_input")
    deferred = _deferred(report)
    assert len(deferred) == 1
    assert "READ writes its item list" in deferred[0]["reason"]
    assert not any("pass  # READ" in line for line in lines)
    assert any("raise NotImplementedError(" in line for line in lines)


# --- module state -------------------------------------------------------------


def test_module_state_the_renderer_cannot_initialize_is_deferred(
    source: Path, renderer: Modules
) -> None:
    body, report = renderer.body(renderer._subprogram_nodes(source))
    assert any(line.startswith("folded = None  # AGENT_QUEUE: ") for line in body)
    entry = next(e for e in report if e["key"] == "module-state:folded")
    assert entry["status"] == "agent_queue" and entry["block"] == "S001"


# --- a hole in the module -----------------------------------------------------


def test_a_subprogram_with_a_record_but_no_node_crashes_the_transform(
    source: Path, renderer: Modules
) -> None:
    """The interface record and the parse pass disagreeing is a broken
    invariant. It used to drop the subprogram and ship a file whose coverage
    note said it was attempted."""
    nodes = renderer._subprogram_nodes(source)
    nodes.pop("scoped")
    with pytest.raises(RuntimeError, match="'scoped' has an interface record but no parse node"):
        renderer.body(nodes)


# --- BLOCK constructs in the read/write analysis ------------------------------


def test_a_block_construct_reports_the_writes_inside_it(source: Path) -> None:
    """``inner: block ... end block inner`` fell into the conservative
    fallback, which counted the construct label as a read and dropped every
    write inside -- so a static gate compared against a wrong truth."""
    record = interface.extract(source, kind_assumptions=KINDS)
    blocks = rwset.block_rwsets(_node(source, "scoped"), rwset.scope_for(record, "scoped"))
    inner = next(b for b in blocks if "acc" in b["writes"])
    assert "total" in inner["writes"]
    assert "inner" not in inner["reads"]
    assert "n" in inner["reads"]


# --- the differential gate ----------------------------------------------------


def _toy_physics() -> Path:
    root = Path(__file__).resolve().parent.parent / "examples" / "toy_physics"
    if not (root / "dumps").is_dir():  # pragma: no cover - wheel install
        pytest.skip("examples/ is not present in an installed wheel")
    return root


def _replay(tmp_path: Path, dumps: Path) -> Any:
    from recast.executors.local import LocalExecutor
    from recast.oracle.dump_replay import DumpReplayOracle
    from recast.registry import REGISTRY
    from recast.verify.bitexact import BitexactVerifier

    root = _toy_physics()
    frontend = REGISTRY.get("frontend", "fortran")()
    unit = min(frontend.discover(root), key=lambda u: len(u.uid))
    facts = frontend.analyze(unit, root)
    config = {"root": str(root)}
    candidate = REGISTRY.get("transform", "translate.numpy")().apply(unit, facts, config)
    workspace = tmp_path / "candidate"
    for name, blob in candidate.files.items():
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_bytes(blob)
    executor = LocalExecutor()
    ref = DumpReplayOracle().materialize(
        unit, facts, workspace, executor, {**config, "dumps": str(dumps)}
    )
    return BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)


def test_a_translated_subprogram_nobody_compared_fails_the_unit_by_name(tmp_path: Path) -> None:
    """toy_physics translates ``settle`` and ``column_mass``. A recording of
    only ``settle`` used to be a bit-exact pass for the whole unit; the gate
    now judges coverage against what was TRANSLATED and names the gap."""
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    for path in (_toy_physics() / "dumps").glob("settle_*.txt"):
        shutil.copy(path, dumps / path.name)
    verdict = _replay(tmp_path, dumps)
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["uncovered"] == ["column_mass"]
    assert "column_mass" in verdict.detail
    assert "silence is not a pass" in verdict.detail


def test_a_recorded_scalar_result_compares_as_the_scalar_it_is(tmp_path: Path) -> None:
    """The probe format has no rank-0 section; ``column_mass`` records its
    result as one value under ``# OUTPUT: m``. Read as shape (1,) against a
    scalar candidate it was "shape () vs (1,)" -- every recorded scalar
    function was uncomparable, which the uncovered gate then reported as the
    unit's failure. Full coverage of the example replays bit-exact."""
    verdict = _replay(tmp_path, _toy_physics() / "dumps")
    assert verdict.confidence is Confidence.BIT_EXACT
    assert verdict.metrics["uncovered"] == []
    assert verdict.metrics["subprograms"]["column_mass"]["points"] == 3


# --- what the coverage gate does not mistake for silence ----------------------


def _gate(tmp_path: Path, module: str, handle: dict[str, Any]) -> Any:
    """Judge a hand-written ``cover_numpy.py`` against a faked oracle handle."""
    from types import SimpleNamespace

    from recast.executors.local import LocalExecutor
    from recast.model import Candidate, OracleRef, Unit
    from recast.verify.bitexact import BitexactVerifier

    candidate = Candidate(
        unit="fortran:cover",
        transform="test.cover",
        files={Path("cover_numpy.py"): module.encode()},
    )
    (tmp_path / "cover_numpy.py").write_bytes(module.encode())
    truth = handle.pop("truth")
    oracle = OracleRef(
        unit=candidate.unit,
        oracle="test.python-truth",
        key="k",
        handle={
            "module": SimpleNamespace(**{f"w_{name}": fn for name, fn in truth.items()}),
            "wrappers": {name: f"w_{name}" for name in truth},
            **handle,
        },
    )
    unit = Unit(uid=candidate.unit, kind="module")
    return BitexactVerifier().verify(
        unit, candidate, oracle, tmp_path, LocalExecutor(), {"trials": 3}
    )


SCALAR = {
    "kind": "function",
    "result": "r",
    "result_dtype": "float64",
    "args": [
        {"name": "v", "dtype": "float64", "intent": "IN", "optional": False},
    ],
}

COVER = """_SIGNATURES = {{
    "scale": {scale!r},
    "{other}": {other_signature!r},
}}


def twice(v):
    return 2.0 * v


def scale(v):
    return twice(v)


{other_body}
"""


def test_a_private_helper_is_covered_through_the_public_subprogram_that_calls_it(
    tmp_path: Path,
) -> None:
    """f2py cannot wrap a private procedure -- the wrappers ``use`` the
    module -- so no oracle will ever offer one. It is exercised through every
    public caller, and the gate judging it "never compared" failed every
    module with a helper."""
    verdict = _gate(
        tmp_path,
        COVER.format(
            scale=SCALAR,
            other="twice",
            other_signature={**SCALAR, "public": False},
            other_body="",
        ),
        {"truth": {"scale": lambda v: 2.0 * v}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["uncovered"] == []


def test_a_subprogram_compared_through_its_flat_adapter_is_covered(tmp_path: Path) -> None:
    """A derived-type interface is compared as ``<name>_flat`` on both sides;
    the adapter calls the subprogram, so the comparison is of it."""
    verdict = _gate(
        tmp_path,
        COVER.format(
            scale=SCALAR,
            other="scale_flat",
            other_signature=SCALAR,
            other_body="def scale_flat(v):\n    return scale(v)\n",
        ),
        {"truth": {"scale_flat": lambda v: 2.0 * v}, "flattened": ["scale"]},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["uncovered"] == []


def test_an_ungated_subprogram_is_named_on_the_verdict_not_counted_as_silence(
    tmp_path: Path,
) -> None:
    """The flat oracle lists what its wrapper cannot spell, with the reason,
    and the verdict carries the list. That is the audit tag, not a gap."""
    verdict = _gate(
        tmp_path,
        COVER.format(
            scale=SCALAR,
            other="takes_proc",
            other_signature=SCALAR,
            other_body="def takes_proc(v):\n    return v\n",
        ),
        {"truth": {"scale": lambda v: 2.0 * v}, "ungated": {"takes_proc": "procedure dummy"}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["uncovered"] == []
    assert verdict.metrics["ungated"] == {"takes_proc": "procedure dummy"}
    assert "takes_proc (procedure dummy)" in verdict.detail


def test_a_public_subprogram_no_wrapper_reached_still_fails_by_name(tmp_path: Path) -> None:
    """The gate's purpose: a public subprogram dropped on the oracle side with
    no reason given is exactly the narrowing it exists to refuse."""
    verdict = _gate(
        tmp_path,
        COVER.format(
            scale=SCALAR,
            other="dropped",
            other_signature=SCALAR,
            other_body="def dropped(v):\n    return v\n",
        ),
        {"truth": {"scale": lambda v: 2.0 * v}},
    )
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["uncovered"] == ["dropped"]
    assert "silence is not a pass" in verdict.detail


PRIVATE_MODULE = """\
module privmod
  implicit none
  private
  public :: scale_all
contains
  subroutine scale_all(n, x, y)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = twice(x(i))
    end do
  end subroutine scale_all

  function twice(v) result(r)
    real(8), intent(in) :: v
    real(8) :: r
    r = 2.0d0 * v
  end function twice
end module privmod
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_a_module_with_a_private_helper_is_bit_exact_end_to_end(tmp_path: Path) -> None:
    """The transform marks the helper private in ``_SIGNATURES``, the f2py
    oracle wraps only the public subprogram, and the gate agrees."""
    from recast.recipes import BUILTIN
    from recast.run import RunStatus, run_recipe

    root = tmp_path / "privmod"
    root.mkdir()
    (root / "privmod.f90").write_text(PRIVATE_MODULE)
    run = run_recipe(
        BUILTIN["translate"](),
        root,
        {
            "output": str(tmp_path / "out"),
            "stages": {"differential.bitexact": {"trials": 3, "dims": {"n": 4}}},
        },
    )
    assert run.status is RunStatus.PASSED, [
        (o.plugin, o.status, o.detail) for u in run.units for o in u.outcomes
    ]


# --- the numba path ------------------------------------------------------------


def test_the_numba_emitter_records_prologue_refusals_too(source: Path) -> None:
    """The numba emitter re-derives its prologue from the NumPy one and used to
    call it without a refusal list: the same refused out-arg raised at run time
    but no ``P###`` entry existed, so the agent queue never heard of it."""
    from recast.transform.numba.backend import Kernels
    from recast.transform.numba.emitter import Emission, NumbaSubprograms

    record = interface.extract(source, kind_assumptions=KINDS)
    assembler = NumbaSubprograms(
        record=record,
        constants=constants.extract(source),
        profile=PROFILES["ifx"],
        emission=Emission(kernels=Kernels(record=record)),
    )
    lines, report = assembler.render(_node(source, "prologue_refusals"), "prologue_refusals")
    prologue = [entry for entry in _deferred(report) if entry["block"].startswith("P")]
    reasons = [entry["reason"] for entry in prologue]
    assert any("out-arg out1" in reason for reason in reasons)
    assert any("out-arg w" in reason for reason in reasons)
    assert any("local parameter grid" in reason for reason in reasons)
    assert any("local array scr" in reason for reason in reasons)
    for entry in prologue:
        low, high = entry["py_lines"]
        span = lines[low:high]
        assert any("# AGENT_QUEUE:" in line for line in span), (entry, span)
        assert span[-1].lstrip().startswith("raise NotImplementedError("), (entry, span)
