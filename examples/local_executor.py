"""What the ``local`` executor does, and what it refuses to do.

P1 ships no driver -- ``recast plan`` shows a recipe's stages but nothing walks
them yet -- so the executor is one of the few pieces that can be exercised end
to end today. Run it to see the contract behave:

    python examples/local_executor.py

The interesting half is [6]. A batch gate silently downgraded to one rank would
return a passing Verdict that means nothing, so the executor declines rather
than pretending. ``RefactorRecipe`` gates on a 512-rank pinned run and therefore
cannot complete here -- that is a property of the work, not a wiring problem.
"""

from __future__ import annotations

from pathlib import Path

from recast.plugins.executor import Job
from recast.registry import REGISTRY


def main() -> None:
    # Through the registry rather than a direct import: this is the same entry
    # point path an out-of-tree PBS or Slurm executor arrives by, so swapping
    # backends is a name change and nothing else.
    executor = REGISTRY.get("executor", "local")()
    cwd = Path.cwd()

    result = executor.run(Job(argv=["python3", "-c", "print(6 * 7)"], cwd=cwd, label="answer"))
    print(f"[1] rc={result.returncode} out={result.stdout.strip()!r} ok={result.ok}")

    # submit() only records the Job; the subprocess starts in wait(). The ABC
    # asks that submit not block, and this implementation satisfies that by not
    # starting anything -- which also means two submits do not run concurrently.
    probe = cwd / "_probe"
    handle = executor.submit(Job(argv=["touch", str(probe)], cwd=cwd))
    print(f"[2] ran after submit? {probe.exists()}")
    executor.wait(handle)
    print(f"    ran after wait?   {probe.exists()}")
    probe.unlink(missing_ok=True)

    # Shell conventions: 124 for a timeout, 127 for a command that is not there.
    timed_out = executor.run(Job(argv=["sleep", "5"], cwd=cwd, timeout_s=0.3))
    print(f"[3] timeout rc={timed_out.returncode} {timed_out.stderr!r}")
    missing = executor.run(Job(argv=["no-such-bin"], cwd=cwd))
    print(f"[4] missing rc={missing.returncode}")

    # A non-empty env replaces the environment; it does not merge into it. Note
    # PATH collapsing to the shell's built-in default on the second line.
    show_path = ["/bin/sh", "-c", "echo $PATH"]
    inherited = executor.run(Job(argv=show_path, cwd=cwd, env={})).stdout.strip()
    replaced = executor.run(Job(argv=show_path, cwd=cwd, env={"A": "b"})).stdout.strip()
    print(f"[5] env={{}}       PATH={inherited[:40]}...")
    print(f"    env={{'A':'b'}}  PATH={replaced}")

    try:
        executor.submit(
            Job(argv=["true"], cwd=cwd, resources={"ranks": 512}, label="fullmodel-gate")
        )
    except RuntimeError as exc:
        print(f"[6] refused: {exc}")


if __name__ == "__main__":
    main()
