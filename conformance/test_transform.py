"""Transform: refuses out loud, and repeats itself.

The two rules with teeth pull in opposite directions. A Transform must not
raise on input it cannot handle -- a partial Candidate with a populated
``deferred`` list is the normal, useful result, and it is what the agent layer
consumes next -- and it must not quietly approximate one either, which is why
``deferred`` being non-empty on hard input is checked rather than assumed. A
transform with an empty deferred list on a hard input is usually hiding
something rather than handling it.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from recast.model import Facts, Unit
from recast.plugins.transform import Transform
from recast.registry import REGISTRY

DEGENERATE = "conformance-degenerate"


@pytest.fixture
def transform(transform_case: Any) -> Transform:
    missing = [m for m in transform_case.requires if importlib.util.find_spec(m) is None]
    missing += [c for c in transform_case.requires_commands if shutil.which(c) is None]
    if missing:
        pytest.skip(f"case {transform_case.name!r} needs {missing}, which are not available here")
    built: Transform = (
        transform_case.build()
        if transform_case.build
        else REGISTRY.get("transform", transform_case.name)()
    )
    return built


def test_requires_names_real_facts_fields(transform: Transform) -> None:
    """A typo here disables the transform and says something else.

    The runner reports a name it cannot find on ``Facts`` as a missing
    prerequisite, so ``requires = ("effect",)`` does not fail loudly -- every
    unit is skipped with a message blaming the frontend for not producing
    something no frontend was ever asked for.
    """
    fields = set(Facts.__dataclass_fields__)
    unknown = [name for name in transform.requires if name not in fields]
    assert not unknown, (
        f"{transform.name!r} requires {unknown}, which are not fields of Facts "
        f"({sorted(fields)}); the runner will skip every unit and blame the frontend"
    )


def test_applicable_never_raises(transform_case: Any, transform: Transform, tmp_path: Path) -> None:
    """It is a cheap pre-filter over whatever a frontend produced, including
    the units it produced by giving up. Returning False is the answer; raising
    ends the run for every unit behind this one."""
    subject = transform_case.subject(tmp_path)
    unit = subject.unit
    hostile: list[tuple[str, Unit, Facts]] = [
        ("empty facts", unit, Facts(unit=unit.uid)),
        ("a foreign kind", Unit(uid=DEGENERATE, kind=DEGENERATE), Facts(unit=DEGENERATE)),
        (
            "a unit the frontend could not parse",
            Unit(uid=DEGENERATE, kind="file", attrs={"parse_error": "SyntaxError: contrived"}),
            Facts(unit=DEGENERATE),
        ),
        ("an interface with no subprograms", unit, Facts(unit=unit.uid, interface={"module": "m"})),
        ("facts describing another unit", unit, Facts(unit="conformance:someone-else")),
    ]
    for label, hostile_unit, hostile_facts in hostile:
        try:
            transform.applicable(hostile_unit, hostile_facts)
        except Exception as error:
            pytest.fail(
                f"{transform.name!r}.applicable raised on {label}: {type(error).__name__}: {error}"
            )


def test_it_is_applicable_to_its_own_subject(
    transform_case: Any, transform: Transform, tmp_path: Path
) -> None:
    """Without this, "never raises" is satisfied by ``return False``."""
    subject = transform_case.subject(tmp_path)
    unit, facts = subject.unit, subject.facts
    assert transform.applicable(unit, facts), (
        "the case's own subject is not applicable, so nothing below transforms anything"
    )
    unmet = [name for name in transform.requires if not getattr(facts, name, None)]
    assert not unmet, f"the subject's Facts do not satisfy the declared requires {unmet}"


def test_a_deterministic_transform_repeats_its_artifact(
    transform_case: Any, transform: Transform, tmp_path: Path
) -> None:
    """``deterministic = True`` is a claim about bytes, and the digest is how
    the claim is stated. Evidence is filed against it."""
    if not transform.deterministic:
        pytest.skip(f"{transform.name!r} declares deterministic = False")

    subject = transform_case.subject(tmp_path)
    unit, facts, config = subject.unit, subject.facts, _config(transform_case, subject)
    first = transform.apply(unit, facts, dict(config))
    second = transform.apply(unit, facts, dict(config))
    assert first.digest() == second.digest(), (
        "two applications of a deterministic transform produced different artifacts"
    )
    assert first.transform == transform.name, (
        f"the Candidate names {first.transform!r} and the plugin is {transform.name!r}; "
        "the name in the Evidence record has to be the one that can be looked up"
    )
    assert first.unit == unit.uid


def test_an_agentic_transform_records_how_to_reconstruct_it(
    transform_case: Any, transform: Transform, tmp_path: Path
) -> None:
    """``deterministic = False`` cannot promise the same bytes, so it promises
    provenance instead: the model that answered and the prompt it answered.
    Those two names are ``AgentResult``'s own, not this suite's invention."""
    if transform.deterministic:
        pytest.skip(f"{transform.name!r} is deterministic; its contract is the digest")

    subject = transform_case.subject(tmp_path)
    notes = transform.apply(subject.unit, subject.facts, _config(transform_case, subject)).notes
    flat = _flatten(notes)
    for key in ("model", "prompt_digest"):
        assert key in flat, (
            f"Candidate.notes records no {key!r}; an artifact that cannot be traced to "
            f"the model that produced it replays to nothing. Recorded: {sorted(flat)}"
        )


def test_what_it_cannot_handle_is_deferred_and_not_raised(
    transform_case: Any, transform: Transform, tmp_path: Path
) -> None:
    if transform_case.defers is None:
        pytest.skip(
            f"{transform_case.name!r} declares no input its rules refuse -- unexercised, not passed"
        )
    subject = transform_case.defers(tmp_path)
    candidate = transform.apply(subject.unit, subject.facts, _config(transform_case, subject))
    assert candidate.deferred, (
        "the case says this input has a site the rules cannot handle, and the "
        "Candidate defers nothing; an empty deferred list on hard input is usually "
        "something hidden rather than something handled"
    )
    assert candidate.files or candidate.patches, (
        "a partial Candidate is still a Candidate: the mechanical part is what the "
        "agent layer patches into"
    )


def test_what_it_calls_mechanical_is_at_least_well_formed(
    transform_case: Any, transform: Transform, tmp_path: Path
) -> None:
    """A Candidate's own files have to parse in the language they are written in.

    Weaker than "correct" on purpose -- nothing here can judge the numbers --
    and it is still the check that four separate defects would have been
    caught by. Each of them emitted a file that no interpreter would load:
    an untranslatable initializer passed through as upper-cased Fortran, an
    integer constant too wide for the ``np.int32`` it was spelled with, a
    ``DIMENSION``-only array read as a statement function and emitted as
    ``def pm(i, i)``, an ``import`` of a companion module that cannot exist.
    Every one of them was reported *mechanical*, which is the part that
    matters: a transform that refuses out loud is behaving, and a transform
    that hands back a file which cannot be loaded has not refused at all.

    A deferred site is a comment or a raise in an otherwise loadable file, so
    this holds for partial candidates too, and skipping when ``deferred`` is
    non-empty would excuse exactly the runs most likely to fail it.
    """
    subject = transform_case.subject(tmp_path)
    candidate = transform.apply(subject.unit, subject.facts, _config(transform_case, subject))
    checked = 0
    for path, content in sorted(candidate.files.items()):
        if path.suffix != ".py":
            continue
        checked += 1
        try:
            compile(content, str(path), "exec")
        except SyntaxError as bad:
            raise AssertionError(
                f"{transform.name!r} emitted {path} and it does not parse as Python: "
                f"{bad.msg} at line {bad.lineno}. A file the candidate wrote itself is "
                "the transform's own claim about its output, and a syntax error in it "
                "is a refusal that was never made."
            ) from bad
    if not checked:
        pytest.skip(
            f"{transform_case.name!r} emitted no Python of its own -- unexercised, not passed"
        )


def _config(case: Any, subject: Any) -> dict[str, Any]:
    """The subject's own config over the case's: where the source lives is
    decided when the subject is planted, not when the case is declared."""
    return {**dict(case.config), **dict(subject.config)}


def _flatten(notes: dict[str, Any]) -> set[str]:
    """Keys at any depth. Where provenance is nested is the transform's choice;
    that it is recorded is not."""
    seen: set[str] = set()
    stack: list[Any] = [notes]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            seen |= set(current)
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return seen
