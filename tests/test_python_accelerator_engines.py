"""First-class Python/NumPy to Numba and JAX engine tests."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from recast.engines import python_jax_engine, python_numba_engine
from recast.errors import ConfigError
from recast.executors.local import LocalExecutor
from recast.model import Confidence, Facts, OracleRef, Unit
from recast.oracle.python_source import PythonSourceOracle
from recast.phases import transform_recipe, verify_recipe_candidates
from recast.plugins.executor import Executor, Job, JobResult
from recast.python.frontend import PythonNumpyFrontend
from recast.recipes import PythonToJaxRecipe, PythonToNumbaRecipe
from recast.run import RunStatus, run_recipe
from recast.transform.python_accelerators import PythonNumpyToJax, PythonNumpyToNumba
from recast.verify.python_accelerators import (
    PythonJaxDifferentialVerifier,
    PythonNumbaDifferentialVerifier,
    _compare,
)

SOURCE = """\
from __future__ import annotations

import numpy as np

__all__ = ["blend"]

def _square(x: np.ndarray) -> np.ndarray:
    return x * x

def blend(x: np.ndarray, scale: float) -> np.ndarray:
    return np.sin(x) * scale + _square(x)
"""


class _RecordingLocalExecutor(LocalExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[Job] = []

    def submit(self, job: Job) -> str:
        self.jobs.append(job)
        return super().submit(job)


class _BrokenWorkerExecutor(Executor):
    name = "broken-worker"

    def __init__(self, behavior: str) -> None:
        self.behavior = behavior
        self.job: Job | None = None

    def submit(self, job: Job) -> str:
        self.job = job
        return "broken-worker-job"

    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        del handle, timeout_s
        assert self.job is not None
        if self.behavior == "timeout":
            return JobResult(124, "", "timeout")
        Path(self.job.argv[-1]).write_bytes(b'{"ok": true}')
        return JobResult(0, "", "")


def _subject(root: Path) -> tuple[Unit, Facts]:
    (root / "kernel.py").write_text(SOURCE)
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(root)))
    return unit, frontend.analyze(unit, root)


def test_python_frontend_is_deterministic_and_skips_recast(tmp_path: Path) -> None:
    unit, facts = _subject(tmp_path)
    hidden = tmp_path / ".recast" / "generated.py"
    hidden.parent.mkdir()
    hidden.write_text(SOURCE)

    frontend = PythonNumpyFrontend()
    assert tuple(frontend.discover(tmp_path)) == (unit,)
    assert facts.interface["exports"] == ["blend"]
    assert facts.interface["module"] == "kernel"
    assert facts.callgraph["python:kernel.py/blend"] == ["_square"]
    assert facts.provenance["digest"]


def test_python_source_oracle_uses_v2_inert_handle_identity(tmp_path: Path) -> None:
    unit, facts = _subject(tmp_path)
    oracle = PythonSourceOracle()
    config = {"root": tmp_path}
    reference = oracle.materialize(unit, facts, tmp_path / "oracle", LocalExecutor(), config)
    source = str(facts.provenance["source"])
    digest = str(facts.provenance["digest"])
    expected = hashlib.sha256(f"python-source-v2\0{source}\0{digest}".encode()).hexdigest()

    assert reference.key == f"python-source:{expected[:24]}"
    assert reference.handle == {
        "root": str(tmp_path.resolve()),
        "source": "kernel.py",
        "module_name": "kernel",
        "functions": ("blend",),
    }
    assert all(not hasattr(value, "__dict__") for value in reference.handle.values())


def test_numba_transform_decorates_reachable_closure(tmp_path: Path) -> None:
    unit, facts = _subject(tmp_path)
    candidate = PythonNumpyToNumba().apply(unit, facts, {"root": tmp_path})
    source = candidate.files[Path("kernel_numba.py")].decode()
    tree = ast.parse(source)

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_square", "blend"):
        decorator = functions[name].decorator_list[0]
        assert isinstance(decorator, ast.Call)
        assert ast.unparse(decorator) == ("__recast_numba.njit(cache=False, fastmath=False)")
    assert candidate.deferred == []
    assert candidate.notes["translated_functions"] == ["_square", "blend"]
    assert candidate.notes["coverage"] == {"subprograms": ["_square", "blend"]}


def test_jax_transform_lowers_numpy_and_defers_mutation(tmp_path: Path) -> None:
    unit, facts = _subject(tmp_path)
    candidate = PythonNumpyToJax().apply(unit, facts, {"root": tmp_path})
    source = candidate.files[Path("kernel_jax.py")].decode()

    assert "import jax.numpy as __recast_jnp" in source
    assert "@__recast_jax.jit\ndef blend" in source
    assert "__recast_jnp.sin(x)" in source
    assert candidate.deferred == []

    (tmp_path / "kernel.py").write_text(
        "import numpy as np\n\n"
        "def mutate(x: np.ndarray) -> np.ndarray:\n"
        "    x[0] = 9.0\n"
        "    return x\n"
    )
    frontend = PythonNumpyFrontend()
    changed = next(iter(frontend.discover(tmp_path)))
    changed_facts = frontend.analyze(changed, tmp_path)
    partial = PythonNumpyToJax().apply(changed, changed_facts, {"root": tmp_path})
    assert partial.deferred == [
        "mutate: in-place array or attribute mutation is not JAX-functional"
    ]


def test_accelerator_generated_bindings_are_reserved(tmp_path: Path) -> None:
    (tmp_path / "kernel.py").write_text(
        "import numpy as np\n\n"
        "def blend(__recast_jnp: np.ndarray) -> np.ndarray:\n"
        "    return __recast_jnp\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)

    with pytest.raises(ConfigError, match="reserved accelerator binding"):
        PythonNumpyToJax().apply(unit, facts, {"root": tmp_path})


@pytest.mark.parametrize(
    ("target", "recipe_type", "gate"),
    [
        ("numba", PythonToNumbaRecipe, "differential.python-numba"),
        ("jax", PythonToJaxRecipe, "differential.python-jax"),
    ],
)
def test_engine_manifests_pin_independent_recipes_and_contracts(
    target: str, recipe_type: type[PythonToNumbaRecipe | PythonToJaxRecipe], gate: str
) -> None:
    manifest = python_numba_engine() if target == "numba" else python_jax_engine()
    recipe = recipe_type()

    assert manifest.id == f"recast.python-numpy.{target}"
    assert manifest.default_recipe == f"python-to-{target}"
    assert manifest.input_artifact_contract.id == "recast.source-tree.python.numpy"
    assert manifest.input_artifact_contract.profile == "numpy"
    assert manifest.output_artifact_contract.profile == target
    assert manifest.default_config == {
        "target": target,
        "frontend": "python-numpy",
        "executor": "local",
    }
    assert manifest.config_schema["additionalProperties"] is False
    assert manifest.required_gates == ("static.complete", gate)
    assert recipe.resolved_engine_id(dict(manifest.default_config)) == manifest.id
    assert [stage.plugin for stage in recipe.stages({}) if stage.gate] == [
        "static.complete",
        gate,
    ]


@pytest.mark.parametrize(
    ("dependency", "transform", "verifier"),
    [
        ("numba", PythonNumpyToNumba, PythonNumbaDifferentialVerifier),
        ("jax", PythonNumpyToJax, PythonJaxDifferentialVerifier),
    ],
)
def test_real_accelerator_numeric_verification(
    dependency: str,
    transform: type[PythonNumpyToNumba | PythonNumpyToJax],
    verifier: type[PythonNumbaDifferentialVerifier | PythonJaxDifferentialVerifier],
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    unit, facts = _subject(tmp_path)
    config = {"root": tmp_path}
    candidate = transform().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit,
        facts,
        tmp_path / "oracle-workspace",
        LocalExecutor(),
        config,
    )
    verdict = verifier().verify(
        unit,
        candidate,
        oracle,
        tmp_path / "verify-workspace",
        LocalExecutor(),
        config,
    )

    assert verdict.confidence in {Confidence.BIT_EXACT, Confidence.TOLERANCED}, verdict.detail
    assert verdict.metrics["backend"] == dependency
    assert verdict.metrics["functions"] == ["blend"]
    assert verdict.metrics["points"] > 0


def test_accelerator_verifier_runs_oracle_and_candidate_as_executor_jobs(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    unit, facts = _subject(tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )
    executor = _RecordingLocalExecutor()

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", executor, config
    )

    assert verdict.passed, verdict.detail
    assert [job.label for job in executor.jobs] == [
        "python-numba-oracle",
        "python-numba-candidate",
    ]
    assert all(job.argv[0] == sys.executable for job in executor.jobs)
    assert all(
        job.argv[1:4] == ("-I", "-m", "recast.verify.python_accelerators_worker")
        for job in executor.jobs
    )
    assert all("PYTHONPATH" not in job.env for job in executor.jobs)


@pytest.mark.parametrize(
    ("behavior", "detail"),
    [
        ("timeout", "exited with status 124"),
        ("noncanonical", "not canonical"),
    ],
)
def test_accelerator_worker_failures_are_fail_closed(
    behavior: str, detail: str, tmp_path: Path
) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    unit, facts = _subject(tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit,
        candidate,
        oracle,
        tmp_path / "verify",
        _BrokenWorkerExecutor(behavior),
        config,
    )

    assert verdict.confidence is Confidence.FAILED
    assert detail in verdict.detail


def test_package_relative_import_survives_candidate_staging(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text("BIAS = 0.25\n")
    (package / "kernel.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from .helpers import BIAS\n\n"
        "def shifted(x: np.ndarray) -> np.ndarray:\n"
        "    return x + BIAS\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    candidate = PythonNumpyToNumba().apply(unit, facts, {"root": tmp_path})
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), {"root": tmp_path}
    )
    verdict = PythonNumbaDifferentialVerifier().verify(
        unit,
        candidate,
        oracle,
        tmp_path / "verify",
        LocalExecutor(),
        {"root": tmp_path},
    )

    assert verdict.passed, verdict.detail


def test_same_named_helpers_do_not_leak_between_project_roots(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")

    def prepare(root: Path, factor: float) -> tuple[Unit, Facts]:
        root.mkdir()
        (root / "helper.py").write_text(f"FACTOR = {factor!r}\n")
        (root / "kernel.py").write_text(
            "from __future__ import annotations\n"
            "import numpy as np\n"
            "from helper import FACTOR\n\n"
            "def scale(x: np.ndarray) -> np.ndarray:\n"
            "    return x * FACTOR\n"
        )
        frontend = PythonNumpyFrontend()
        unit = next(iter(frontend.discover(root)))
        return unit, frontend.analyze(unit, root)

    root_a = tmp_path / "project-a"
    unit_a, facts_a = prepare(root_a, 2.0)
    candidate_a = PythonNumpyToNumba().apply(unit_a, facts_a, {"root": root_a})
    oracle_a = PythonSourceOracle().materialize(
        unit_a, facts_a, tmp_path / "oracle-a", LocalExecutor(), {"root": root_a}
    )
    first = PythonNumbaDifferentialVerifier().verify(
        unit_a,
        candidate_a,
        oracle_a,
        tmp_path / "verify-a",
        LocalExecutor(),
        {"root": root_a},
    )
    assert first.passed, first.detail

    root_b = tmp_path / "project-b"
    unit_b, facts_b = prepare(root_b, 3.0)
    candidate_b = PythonNumpyToNumba().apply(unit_b, facts_b, {"root": root_b})
    output = Path("kernel_numba.py")
    generated = candidate_b.files[output].decode()
    forged = generated.replace("return x * FACTOR", "return x * 2.0")
    assert forged != generated
    candidate_b.files[output] = forged.encode()
    oracle_b = PythonSourceOracle().materialize(
        unit_b, facts_b, tmp_path / "oracle-b", LocalExecutor(), {"root": root_b}
    )

    second = PythonNumbaDifferentialVerifier().verify(
        unit_b,
        candidate_b,
        oracle_b,
        tmp_path / "verify-b",
        LocalExecutor(),
        {"root": root_b},
    )

    assert second.confidence is Confidence.FAILED
    assert "differs" in second.detail


def test_numba_gate_rejects_backend_spoof_from_shared_helper(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    (tmp_path / "helper.py").write_text(
        "import numba\n"
        "from numba.core import registry\n\n"
        "class FakeDispatcher:\n"
        "    signatures = ('forged',)\n"
        "    def __init__(self, function):\n"
        "        self.function = function\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self.function(*args, **kwargs)\n\n"
        "def fake_njit(**_options):\n"
        "    return lambda function: FakeDispatcher(function)\n\n"
        "numba.njit = fake_njit\n"
        "registry.CPUDispatcher = FakeDispatcher\n"
        "MARKER = True\n"
    )
    (tmp_path / "kernel.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from helper import MARKER\n\n"
        "def blend(x: np.ndarray) -> np.ndarray:\n"
        "    return np.sin(x) if MARKER else x\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "not a genuine Numba CPU dispatcher" in verdict.detail


def test_jax_gate_rejects_backend_spoof_from_shared_helper(tmp_path: Path) -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")
    (tmp_path / "helper.py").write_text(
        "import jax\n\n"
        "class Lowered:\n"
        "    def __init__(self, function):\n"
        "        self.function = function\n"
        "    def compile(self):\n"
        "        return self.function\n\n"
        "class FakeJitted:\n"
        "    def __init__(self, function):\n"
        "        self.function = function\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self.function(*args, **kwargs)\n"
        "    def lower(self, *_args, **_kwargs):\n"
        "        return Lowered(self.function)\n\n"
        "def fake_jit(function):\n"
        "    return FakeJitted(function)\n\n"
        "jax.jit = fake_jit\n"
        "MARKER = True\n"
    )
    (tmp_path / "kernel.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from helper import MARKER\n\n"
        "def blend(x: np.ndarray) -> np.ndarray:\n"
        "    return np.sin(x) if MARKER else x\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToJax().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonJaxDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "not a genuine JAX jitted function" in verdict.detail


def test_matching_nan_values_are_valid_numerical_points(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    (tmp_path / "kernel.py").write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n\n"
        "def mask(x: np.ndarray) -> np.ndarray:\n"
        "    return np.where(x > 1.0, np.nan, x)\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )
    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.passed, verdict.detail
    assert verdict.metrics["points"] > 0


def test_equal_non_numerical_result_cannot_satisfy_numeric_gate(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    (tmp_path / "kernel.py").write_text(
        "import numpy as np\n\ndef label() -> str:\n    return 'hello'\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "nothing numerical was compared" in verdict.detail


@pytest.mark.parametrize(
    ("reference", "translated"),
    [
        (2**53 + 1, 2**53),
        (2**63 - 1, 2**63 - 2),
        (True, False),
    ],
)
def test_discrete_mismatches_are_never_tolerance_excused(
    reference: int | bool, translated: int | bool
) -> None:
    np = pytest.importorskip("numpy")

    comparison = _compare(np, reference, translated, rtol=1e100, atol=1e100)

    assert comparison is not None
    points, exact, _max_abs, _max_rel, discrete_mismatch = comparison
    assert points == 1
    assert exact == 0
    assert discrete_mismatch


def test_empty_arrays_do_not_manufacture_a_numerical_point() -> None:
    np = pytest.importorskip("numpy")

    comparison = _compare(np, np.array([]), np.array([]), rtol=1e-12, atol=1e-12)

    assert comparison == (0, 0, 0.0, 0.0, False)


def test_jax_gate_rejects_globally_disabled_jit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")
    unit, facts = _subject(tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToJax().apply(unit, facts, config)
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )
    monkeypatch.setenv("JAX_DISABLE_JIT", "1")
    verdict = PythonJaxDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "JAX JIT execution is disabled" in verdict.detail


def test_numba_gate_rejects_duck_typed_dispatcher_spoof(tmp_path: Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    (tmp_path / "kernel.py").write_text(
        "import numpy as np\n\n"
        "__all__ = ['blend']\n\n"
        "def njit(**_options):\n"
        "    def decorate(function):\n"
        "        function.signatures = ('forged',)\n"
        "        return function\n"
        "    return decorate\n\n"
        "def blend(x: np.ndarray) -> np.ndarray:\n"
        "    return np.sin(x)\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToNumba().apply(unit, facts, config)
    path = Path("kernel_numba.py")
    forged = (
        candidate.files[path]
        .decode()
        .replace(
            "@__recast_numba.njit(cache=False, fastmath=False)",
            "@njit(cache=False, fastmath=False)",
        )
    )
    candidate.files[path] = forged.encode()
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "not a genuine Numba CPU dispatcher" in verdict.detail


def test_jax_gate_rejects_duck_typed_lower_compile_spoof(tmp_path: Path) -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")
    (tmp_path / "kernel.py").write_text(
        "import numpy as np\n\n"
        "__all__ = ['blend']\n\n"
        "def jit(function):\n"
        "    class Lowered:\n"
        "        def compile(self):\n"
        "            return function\n"
        "    function.lower = lambda *args, **kwargs: Lowered()\n"
        "    return function\n\n"
        "def blend(x: np.ndarray) -> np.ndarray:\n"
        "    return np.sin(x)\n"
    )
    frontend = PythonNumpyFrontend()
    unit = next(iter(frontend.discover(tmp_path)))
    facts = frontend.analyze(unit, tmp_path)
    config = {"root": tmp_path}
    candidate = PythonNumpyToJax().apply(unit, facts, config)
    path = Path("kernel_jax.py")
    forged = candidate.files[path].decode().replace("@__recast_jax.jit", "@jit")
    candidate.files[path] = forged.encode()
    oracle = PythonSourceOracle().materialize(
        unit, facts, tmp_path / "oracle", LocalExecutor(), config
    )

    verdict = PythonJaxDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), config
    )

    assert verdict.confidence is Confidence.FAILED
    assert "not a genuine JAX jitted function" in verdict.detail


def test_backend_verifiers_fail_closed_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit, facts = _subject(tmp_path)
    candidate = PythonNumpyToNumba().apply(unit, facts, {"root": tmp_path})
    oracle = OracleRef(
        unit=unit.uid,
        oracle="python-source",
        key="source",
        handle={"module": object(), "functions": ("blend",)},
    )
    real_find_spec = importlib.util.find_spec

    def absent(name: str) -> ModuleSpec | None:
        return None if name == "numba" else real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", absent)
    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path, LocalExecutor(), {"root": tmp_path}
    )
    assert verdict.confidence is Confidence.FAILED
    assert "numba is not installed" in verdict.detail


def test_accelerator_verifier_rejects_legacy_live_module_oracle_handle(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    unit, facts = _subject(tmp_path)
    candidate = PythonNumpyToNumba().apply(unit, facts, {"root": tmp_path})
    oracle = OracleRef(
        unit=unit.uid,
        oracle="python-source",
        key="legacy-source",
        handle={"module": object(), "functions": ("blend",)},
    )

    verdict = PythonNumbaDifferentialVerifier().verify(
        unit, candidate, oracle, tmp_path / "verify", LocalExecutor(), {"root": tmp_path}
    )

    assert verdict.confidence is Confidence.FAILED
    assert "inert source handle" in verdict.detail


@pytest.mark.parametrize(
    ("dependency", "recipe"),
    [
        ("numba", PythonToNumbaRecipe()),
        ("jax", PythonToJaxRecipe()),
    ],
)
def test_first_class_recipe_runs_real_numeric_pipeline(
    dependency: str,
    recipe: PythonToNumbaRecipe | PythonToJaxRecipe,
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    source_root = tmp_path / "source"
    source_root.mkdir()
    _subject(source_root)
    run = run_recipe(
        recipe,
        source_root,
        {
            "target": dependency,
            "frontend": "python-numpy",
            "executor": "local",
            "output": tmp_path / "output",
        },
    )

    assert run.status is RunStatus.PASSED, run.summary()
    assert len(run.units) == 1
    assert run.units[0].candidate is not None
    assert run.units[0].candidate.deferred == []
    assert run.units[0].verdicts[-1].verifier == f"differential.python-{dependency}"
    assert run.units[0].verdicts[-1].metrics["points"] > 0


@pytest.mark.parametrize(
    ("dependency", "recipe"),
    [
        ("numba", PythonToNumbaRecipe()),
        ("jax", PythonToJaxRecipe()),
    ],
)
def test_phased_verification_accepts_required_python_subprogram(
    dependency: str,
    recipe: PythonToNumbaRecipe | PythonToJaxRecipe,
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    source_root = tmp_path / "source"
    source_root.mkdir()
    _subject(source_root)
    config = {
        "target": dependency,
        "frontend": "python-numpy",
        "executor": "local",
    }
    source_digest = "sha256:" + "7" * 64
    bundle = transform_recipe(
        recipe,
        source_root,
        config,
        source_artifact_digest=source_digest,
        output=tmp_path / "transform-output",
        workspace=tmp_path / "transform-workspace",
    )
    report = verify_recipe_candidates(
        recipe,
        source_root,
        bundle,
        config,
        expected_source_artifact_digest=source_digest,
        expected_engine=bundle.engine,
        required_units=("python:kernel.py",),
        required_subprograms=("blend",),
        require_no_deferred=True,
        output=tmp_path / "verify-output",
        workspace=tmp_path / "verify-workspace",
    )

    assert report.accepted, report.reason_codes
    blend = next(item for item in report.subprograms if item.selector == "blend")
    assert blend.required and blend.observed and blend.gates_passed and blend.accepted
