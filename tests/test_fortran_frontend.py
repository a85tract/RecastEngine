"""Tests for the in-tree Fortran frontend.

Two things are being guarded. The contract ones -- a Unit is stably
addressable, ``analyze`` is deterministic, the optional dependency is named
when it is missing -- apply to any Frontend and are the reason this plugin
ships in-tree as the reference. The rest pin the specific promises this
frontend makes about what it reports and, more importantly, about what it
refuses to guess.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from recast import OUTPUT_DIRNAME, WORKSPACE_DIRNAME
from recast.errors import ConfigError
from recast.model import Facts, Unit
from recast.plugins.frontend import Frontend

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import UnparsableSource, factory
from recast.fortran import frontend as frontend_mod
from recast.fortran.interface import IntentConflict, UnknownOverride

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module micro_mg2_0
  use precision_mod, only: r8 => wp_r8
  implicit none
  private
  public :: micro_mg_tend, mg_init
  real(r8), parameter :: pi = 3.14159265358979_r8
  real(r8) :: cached_dt
  integer :: ncall
contains
  subroutine mg_init(dt)
    real(r8), intent(in) :: dt
    cached_dt = dt
    ncall = 0
  end subroutine mg_init

  subroutine micro_mg_tend(ncol, t, qc, tend, err)
    integer, intent(in) :: ncol
    real(r8), intent(in) :: t(ncol), qc(ncol)
    real(r8), intent(out) :: tend(ncol)
    character(len=*), intent(out), optional :: err
    integer :: i
    do i = 1, ncol
      tend(i) = pi * qc(i) / (t(i) + 273.15_r8)
      if (tend(i) > 1.0e12_r8) then
        write(6,*) 'overflow', i
        stop
      end if
    end do
    ncall = ncall + 1
    call mg_init(cached_dt)
    call mpi_barrier(ierr)
    if (present(err)) err = ''
  end subroutine micro_mg_tend

  ! Declares no intent for either dummy -- the case roughly a third of CAM's
  ! arguments are in, and the reason intent_overrides exists.
  subroutine legacy_tend(n, x, y)
    integer :: n
    real(r8) :: x(n)
    real(r8) :: y(n)
    y = x * 2.0_r8
  end subroutine legacy_tend
end module micro_mg2_0
"""

BROKEN = "module broken\n  this is not fortran ((((\nend module\n"


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "micro_mg.F90").write_text(SOURCE)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "broken.f90").write_text(BROKEN)
    return tmp_path


@pytest.fixture
def fe():
    return factory(kind_assumptions={"r8": "float64"})


# --- the Frontend contract ---------------------------------------------------


def test_the_entry_point_factory_builds_a_frontend(fe) -> None:
    assert isinstance(fe, Frontend)
    assert fe.name == "fortran"
    assert "fortran" in fe.languages


def test_discover_is_deterministic(fe, tree) -> None:
    """Ordering is not significant to the engine, but instability would still
    make a cached Facts table impossible to compare between runs."""
    first = [u.uid for u in fe.discover(tree)]
    second = [u.uid for u in fe.discover(tree)]
    assert first == second


def test_units_are_addressable_at_both_granularities(fe, tree) -> None:
    units = {u.uid: u for u in fe.discover(tree)}
    module = units["fortran:micro_mg2_0"]
    kernel = units["fortran:micro_mg2_0/micro_mg_tend"]
    assert module.kind == "module" and module.parent is None
    assert kernel.kind == "subprogram" and kernel.parent == module.uid
    # Relative to the root, so a Unit survives the tree being moved or copied.
    assert not kernel.sources[0].is_absolute()


def test_analyze_is_reproducible_for_one_revision(fe, tree) -> None:
    unit = next(u for u in fe.discover(tree) if u.uid == "fortran:micro_mg2_0")
    a, b = fe.analyze(unit, tree), fe.analyze(unit, tree)
    assert isinstance(a, Facts)
    assert (a.interface, a.constants, a.callgraph, a.effects) == (
        b.interface,
        b.constants,
        b.callgraph,
        b.effects,
    )
    assert a.provenance["digest"] == b.provenance["digest"]


def test_provenance_records_what_was_assumed(fe, tree) -> None:
    """A kind assumption changes the dtypes analysis reports, so a Facts record
    that does not carry it cannot be checked by anyone reading it later."""
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    prov = fe.analyze(unit, tree).provenance
    assert prov["kind_assumptions"] == {"r8": "float64"}
    assert prov["parser"] == "fparser2" and prov["standard"]


# --- what it reports ---------------------------------------------------------


def test_callgraph_addresses_units_and_keeps_unresolved_names(fe, tree) -> None:
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    calls = fe.analyze(unit, tree).callgraph["fortran:micro_mg2_0/micro_mg_tend"]
    assert "fortran:micro_mg2_0/mg_init" in calls, "a resolved callee is a Unit uid"
    assert "mpi_barrier" in calls, "an unresolved external stays visible as a name"


def test_side_channels_are_reported(fe, tree) -> None:
    """A routine that writes to a unit and can halt is not a pure kernel, and a
    Verifier that treats it as one will produce a passing Verdict that lies."""
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    effects = fe.analyze(unit, tree).effects
    tend = effects["fortran:micro_mg2_0/micro_mg_tend"]
    assert tend["io"] == ["write"] and tend["halts"] and tend["mpi"] == ["mpi_barrier"]
    init = effects["fortran:micro_mg2_0/mg_init"]
    assert {k: v for k, v in init.items() if k != "blocks"} == {
        "reads": [],
        "writes": ["cached_dt", "ncall"],
        "optional_args": [],
        "io": [],
        "halts": False,
        "mpi": [],
        "allocates": False,
    }


def test_read_write_sets_are_reported_per_block(fe, tree) -> None:
    """A Verifier comparing a translation against the source has to be able to
    say *which* block disagrees. Block ids come from the same chunking every
    other stage uses, so the answer lines up with theirs."""
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    blocks = fe.analyze(unit, tree).effects["fortran:micro_mg2_0/mg_init"]["blocks"]
    assert [b["id"] for b in blocks] == ["B001", "B002"]
    assert blocks[0] == {"id": "B001", "reads": ["dt"], "writes": ["cached_dt"]}
    assert blocks[1] == {"id": "B002", "reads": [], "writes": ["ncall"]}


def test_undeclared_intent_is_not_guessed(fe, tree) -> None:
    """``ierr`` aside, every argument here declares an intent. The point is that
    the reported value comes from the source, never from a default."""
    unit = next(u for u in fe.discover(tree) if u.uid.endswith("/micro_mg_tend"))
    args = {a["name"]: a for a in fe.analyze(unit, tree).interface["subprograms"][0]["args"]}
    assert args["tend"]["intent"] == "OUT"
    assert args["err"]["optional"] is True


def test_subprogram_facts_are_narrowed_to_that_subprogram(fe, tree) -> None:
    """A Transform handed one kernel must not have to filter whole-file facts."""
    unit = next(u for u in fe.discover(tree) if u.uid.endswith("/micro_mg_tend"))
    facts = fe.analyze(unit, tree)
    assert [s["name"] for s in facts.interface["subprograms"]] == ["micro_mg_tend"]
    assert list(facts.constants["literal_map"]) == ["micro_mg_tend"]
    assert all(
        loc.startswith("micro_mg_tend:")
        for entry in facts.constants["hoisted_literals"].values()
        for loc in entry["locations"]
    )
    # Module parameters stay: the kernel reads pi, so its constants module must
    # still define it.
    assert any(p["name"] == "pi" for p in facts.constants["module_parameters"])


def test_every_non_whitelisted_literal_is_hoisted(fe, tree) -> None:
    """The zero-literal rule. A translated routine contains no bare magic
    numbers, so both sides of a differential check name the same constant."""
    unit = next(u for u in fe.discover(tree) if u.uid.endswith("/micro_mg_tend"))
    values = {e["value"] for e in fe.analyze(unit, tree).constants["hoisted_literals"].values()}
    assert {"273.15", "1.0e12"} <= values
    assert "1" not in values, "loop bounds of 1 are structural, not magic"


# --- what it refuses ---------------------------------------------------------


def test_an_unparsable_file_is_reported_not_dropped(fe, tree) -> None:
    """Discovery that silently skips what it choked on reports coverage it
    never had."""
    units = {u.uid: u for u in fe.discover(tree)}
    assert "fortran:broken" in units
    assert units["fortran:broken"].attrs["parse_error"]
    with pytest.raises(UnparsableSource):
        fe.analyze(units["fortran:broken"], tree)


def test_a_unit_from_another_frontend_is_refused(fe, tree) -> None:
    with pytest.raises(ConfigError):
        fe.analyze(Unit(uid="c:foo", kind="module"), tree)
    with pytest.raises(ConfigError):
        fe.analyze(Unit(uid="fortran:gone", kind="module", sources=(tree / "nope.F90",)), tree)


def test_a_stale_subprogram_unit_is_refused(fe, tree) -> None:
    """Analyzing against a source that no longer defines the subprogram must
    fail rather than quietly return facts about the whole file."""
    unit = Unit(
        uid="fortran:micro_mg2_0/deleted_routine",
        kind="subprogram",
        sources=((tree / "micro_mg.F90").relative_to(tree),),
        parent="fortran:micro_mg2_0",
    )
    with pytest.raises(UnparsableSource):
        fe.analyze(unit, tree)


# --- the optional dependency -------------------------------------------------


def test_a_missing_parser_names_the_extra(monkeypatch) -> None:
    """Registration must not need fparser, so this is where the operator finds
    out -- and the message has to say what to install."""
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "recast.fortran._parse":
            raise ImportError("No module named 'fparser'", name="fparser")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ConfigError, match=r"recast-engine\[fortran\]"):
        frontend_mod._require_fparser()


def test_an_unrelated_import_error_is_not_disguised(monkeypatch) -> None:
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "recast.fortran._parse":
            raise ImportError("boom", name="recast.fortran._typo")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="boom"):
        frontend_mod._require_fparser()


def test_importing_the_package_does_not_import_the_parser() -> None:
    """Registering must be free. If this fails, ``recast doctor`` on an
    installation without the extra stops listing the plugin instead of
    listing it and saying what it needs."""
    code = (
        "import sys, recast.fortran, recast.fortran.frontend;"
        "print([m for m in sys.modules if m.split('.')[0] == 'fparser'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# --- intent overrides --------------------------------------------------------


def _args(facts, sub="legacy_tend"):
    record = next(s for s in facts.interface["subprograms"] if s["name"] == sub)
    return {a["name"]: a for a in record["args"]}


def test_undeclared_intent_stays_unknown_by_default(fe, tree) -> None:
    """No table, no guess. This is the state a third of CAM is in."""
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    args = _args(fe.analyze(unit, tree))
    assert args["x"]["intent"] == "UNKNOWN"
    assert "intent_override" not in args["x"], "an untouched argument grows no extra keys"


def test_an_override_fills_unknown_in_and_says_so(tree) -> None:
    """The record must keep both halves: what the file says, and the fact that
    a human supplied the rest. A Transform that cannot tell them apart cannot
    decide how much to trust the answer."""
    fe = factory(intent_overrides={"legacy_tend": {"x": "IN", "y": "OUT"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    facts = fe.analyze(unit, tree)
    args = _args(facts)
    assert args["x"]["intent"] == "IN"
    assert args["x"]["intent_declared"] == "UNKNOWN"
    assert args["x"]["intent_override"] is True
    assert args["n"]["intent"] == "UNKNOWN", "an argument the table omits is untouched"
    # Recorded once, not repeated on every argument.
    assert facts.provenance["intent_overrides"] == {"legacy_tend": {"x": "IN", "y": "OUT"}}


def test_an_override_may_not_contradict_the_source(tree) -> None:
    """If the file and the operator disagree, one of them is wrong, and which
    is not a question analysis gets to answer."""
    fe = factory(intent_overrides={"micro_mg_tend": {"tend": "IN"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    with pytest.raises(IntentConflict, match="tend"):
        fe.analyze(unit, tree)


def test_an_override_that_agrees_with_the_source_is_harmless(tree) -> None:
    fe = factory(intent_overrides={"micro_mg_tend": {"tend": "OUT"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    args = _args(fe.analyze(unit, tree), "micro_mg_tend")
    assert args["tend"]["intent"] == "OUT"
    assert "intent_override" not in args["tend"], "the source already said it"


def test_a_misspelled_argument_is_an_error(tree) -> None:
    """An override the operator believes is in force but that matches nothing
    is worse than no override at all."""
    fe = factory(intent_overrides={"legacy_tend": {"xx": "IN"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    with pytest.raises(UnknownOverride, match="xx"):
        fe.analyze(unit, tree)


def test_a_bad_intent_value_is_an_error(tree) -> None:
    fe = factory(intent_overrides={"legacy_tend": {"x": "INPUT"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    with pytest.raises(UnknownOverride, match="INPUT"):
        fe.analyze(unit, tree)


def test_one_table_may_cover_a_whole_tree(tree) -> None:
    """Subprograms this file does not define are somebody else's -- ignoring
    them is what lets a project keep one table rather than one per file."""
    fe = factory(
        intent_overrides={
            "_provenance": "observed at the call sites in zm_conv_intr.F90",
            "legacy_tend": {"x": "IN"},
            "defined_in_another_file": {"whatever": "OUT"},
        }
    )
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    assert _args(fe.analyze(unit, tree))["x"]["intent"] == "IN"


def test_overrides_are_case_insensitive(tree) -> None:
    """Fortran is. A table that has to match the source's capitalisation would
    be a table that silently stops applying when someone reformats."""
    fe = factory(intent_overrides={"LEGACY_TEND": {"X": "in"}})
    unit = next(u for u in fe.discover(tree) if u.kind == "module")
    assert _args(fe.analyze(unit, tree))["x"]["intent"] == "IN"


# --- accessibility -----------------------------------------------------------


def test_visibility_follows_the_default_and_the_explicit_lists(tmp_path: Path) -> None:
    """CAM's convention is a bare `private` with an explicit public list, and
    a wrapper that `use`s a private symbol does not compile -- so who is
    public is a fact consumers genuinely need recorded."""
    from recast.fortran import interface

    source = tmp_path / "vis.f90"
    source.write_text(
        """\
module vis_mod
  implicit none
  private
  public seen
contains
  subroutine seen()
  end subroutine seen
  subroutine hidden()
  end subroutine hidden
end module vis_mod
"""
    )
    record = interface.extract(source, kind_assumptions=KINDS)
    visibility = {s["name"]: s["public"] for s in record["subprograms"]}
    assert visibility == {"seen": True, "hidden": False}

    open_source = tmp_path / "open.f90"
    open_source.write_text(
        """\
module open_mod
  implicit none
  private shy
contains
  subroutine bold()
  end subroutine bold
  subroutine shy()
  end subroutine shy
end module open_mod
"""
    )
    record = interface.extract(open_source, kind_assumptions=KINDS)
    visibility = {s["name"]: s["public"] for s in record["subprograms"]}
    assert visibility == {"bold": True, "shy": False}


@pytest.mark.parametrize("dirname", [OUTPUT_DIRNAME, WORKSPACE_DIRNAME])
def test_the_engines_own_output_is_not_source(tree, fe, dirname: str) -> None:
    """A previous run's generated code is output, not input.

    The f2py oracle leaves compilable wrappers in the run's workspace.
    Discovering those turns the engine's output into its own input: the same
    tree yields a different unit set before and after a run, and the second run
    offers to translate the scaffolding the first one generated.

    ``output/`` normally sits outside the tree, which is the real fix; both
    names are pinned because ``config["output"]`` can be pointed back inside,
    and because a tree carrying a run from before the move still has a
    ``.recast/``.
    """
    before = {u.uid for u in fe.discover(tree)}

    build = tree / dirname / "translate" / "oracle-deadbeef"
    build.mkdir(parents=True)
    (build / "wrappers.f90").write_text(SOURCE)

    assert {u.uid for u in fe.discover(tree)} == before


def test_a_vendored_tree_can_be_left_out_of_discovery(tmp_path: Path) -> None:
    """Two libraries a repository vendors may each define a module of the same
    name; that is their business, not a collision this run has to resolve.
    Without a way to say so, discovery over such a tree yields one uid twice
    and the run refuses -- which is right, but leaves nothing to do about it."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "own.f90").write_text("module own\nend module own\n")
    for vendor in ("a", "b"):
        directory = tmp_path / "vendor" / vendor
        directory.mkdir(parents=True)
        (directory / "shared.f90").write_text("module shared\nend module shared\n")

    everything = [u.uid for u in factory().discover(tmp_path)]
    assert everything.count("fortran:shared") == 2, "the collision is real"

    narrowed = [u.uid for u in factory(exclude=["vendor"]).discover(tmp_path)]
    assert narrowed == ["fortran:own"]


def test_every_program_unit_of_a_file_is_a_unit_analyzed_under_its_own_name(tmp_path: Path) -> None:
    """One file, two modules and the program that uses the second: the
    frontend read only the first module, so ``use second`` resolved to the
    file and came back with ``first``'s record, and the program was a Unit
    of kind ``program`` whose body no record described."""
    (tmp_path / "pair.f90").write_text(
        "module first\n"
        "  implicit none\n"
        "  real :: a = 1.0\n"
        "contains\n"
        "  subroutine one(x)\n"
        "    real, intent(out) :: x\n"
        "    x = a\n"
        "  end subroutine one\n"
        "end module first\n"
        "\n"
        "module second\n"
        "  implicit none\n"
        "  real :: b = 2.0\n"
        "contains\n"
        "  subroutine two(y)\n"
        "    real, intent(out) :: y\n"
        "    y = b\n"
        "  end subroutine two\n"
        "end module second\n"
        "\n"
        "program driver\n"
        "  use second, only: two\n"
        "  implicit none\n"
        "  real :: out\n"
        "  call two(out)\n"
        "end program driver\n"
    )
    from recast.fortran.frontend import FortranFrontend

    frontend = FortranFrontend()
    units = {u.uid: u for u in frontend.discover(tmp_path)}
    assert {"fortran:first", "fortran:second", "fortran:driver"} <= set(units)
    assert units["fortran:second"].kind == "module"
    assert units["fortran:driver"].kind == "program"
    assert "fortran:second/two" in units and units["fortran:second/two"].parent == "fortran:second"

    second = frontend.analyze(units["fortran:second"], tmp_path)
    assert second.interface["module"] == "second"
    assert [s["name"] for s in second.interface["subprograms"]] == ["two"]
    assert [s["name"] for s in second.interface["module_state"]] == ["b"]

    driver = frontend.analyze(units["fortran:driver"], tmp_path)
    body = driver.interface["subprograms"][0]
    assert body["name"] == "driver" and body["kind"] == "program"
    assert body["args"] == [] and [entry["name"] for entry in body["locals"]] == ["out"]
    # the call resolves through the companion: the module the program asked
    # for, in the same file, under its own record
    assert [c["module"] for c in driver.provenance["companions"]] == ["second"]
    companion = driver.provenance["companions"][0]["record"]
    assert [s["name"] for s in companion["subprograms"]] == ["two"]
    assert any(c.endswith("two") for c in driver.callgraph["fortran:driver/driver"])
