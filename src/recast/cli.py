"""The ``recast`` command line.

Introspection first -- ``recast doctor`` and ``recast plan`` check that a
plugin set is wired correctly without burning HPC allocation to find out --
and then ``recast run``, which walks a recipe's stages over a source tree
and leaves Evidence behind.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from recast import __version__
from recast.errors import RecastError
from recast.plugins.recipe import Recipe
from recast.recipes import BUILTIN
from recast.registry import KINDS, REGISTRY


def _recipe(name: str) -> Recipe:
    """A recipe by name: the four builtins, then anything a plugin registered.

    A domain package's recipe attaches through the same entry-point group as
    everything else, and a CLI that only knew the builtins would make that
    attachment decorative.
    """
    cls = BUILTIN.get(name)
    if cls is None and name in REGISTRY.names("recipe"):
        cls = REGISTRY.get("recipe", name)
    if cls is None:
        known = sorted(set(BUILTIN) | set(REGISTRY.names("recipe")))
        raise RecastError(f"unknown recipe {name!r}; known: {', '.join(known)}")
    return cls()


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"recast {__version__}")
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    kinds = [args.kind] if args.kind else list(KINDS)
    for kind in kinds:
        names = REGISTRY.names(kind)
        print(f"{kind}:")
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (none registered)")
    broken = REGISTRY.broken()
    if broken:
        print("\nfailed to load:", file=sys.stderr)
        for key, reason in sorted(broken.items()):
            print(f"  {key}: {reason}", file=sys.stderr)
    return 0


def _cmd_recipes(_args: argparse.Namespace) -> int:
    names = set(BUILTIN) | set(REGISTRY.names("recipe"))
    width = max(map(len, names), default=0)
    for name, cls in sorted(BUILTIN.items()):
        print(f"{name:{width}s} {cls.summary}")
    for name in REGISTRY.names("recipe"):
        if name not in BUILTIN:
            summary = getattr(REGISTRY.get("recipe", name), "summary", "")
            print(f"{name:{width}s} {summary} (plugin)")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    """Show the stages a recipe would run, and which plugins are missing.

    Dry only. This is the cheap check that a config is coherent before a batch
    oracle costs hours.
    """
    config = json.loads(args.config) if args.config else {}
    recipe = _recipe(args.recipe)

    problems = recipe.validate(config)
    for problem in problems:
        print(f"config: {problem}", file=sys.stderr)

    from recast.run import missing_tools

    stages = recipe.stages(config)
    # The preflight: a scanner that wraps a binary says so, and this asks for
    # the binary now rather than letting the run find out two stages in.
    # Reported beside the stage and counted as unavailable, because a plugin
    # whose tool is absent will be ``incomplete`` at run time, and the point of
    # ``plan`` is to say that before anything runs.
    tools = missing_tools(stages, config, registry=REGISTRY)

    missing = 0
    for i, stage in enumerate(stages, 1):
        available = stage.plugin in REGISTRY.names(stage.kind)
        tool = tools.get(stage.plugin)
        if tool:
            mark = "????"
        else:
            mark = "ok " if available else ("opt" if stage.optional else "MISS")
        flags = " ".join(f for f, on in (("gate", stage.gate), ("optional", stage.optional)) if on)
        note = f"  {tool}" if tool else ""
        print(f"{i:2d}. [{mark}] {stage.kind:12s} {stage.plugin:28s} {flags}{note}")
        if (not available or tool) and not stage.optional:
            missing += 1

    if missing or problems:
        print(f"\n{missing} required plugin(s) unavailable, {len(problems)} config problem(s)")
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from recast.run import RunStatus, run_recipe

    recipe = _recipe(args.recipe)
    config = {}
    if args.config:
        path = Path(args.config)
        if not path.is_file():
            raise RecastError(
                f"config file {path} does not exist"
                + ("" if path.is_absolute() else f" (relative to {Path.cwd()})")
            )
        if path.suffix == ".toml":
            import tomllib

            config = tomllib.loads(path.read_text())
        else:
            config = json.loads(path.read_text())
    if args.unit:
        config["units"] = list(args.unit)
    if getattr(args, "range", None):
        config["range"] = args.range

    root = Path(args.root)
    if not root.is_dir():
        raise RecastError(
            f"source tree {root} does not exist"
            + ("" if root.is_absolute() else f" (relative to {Path.cwd()})")
        )
    run = run_recipe(recipe, root, config)
    if args.summary:
        import json as _json

        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(_json.dumps(run.summary(), indent=2, sort_keys=True) + "\n")
    if args.summary:
        print(f"summary: {args.summary}")
    for unit_run in run.units:
        print(f"{unit_run.unit.uid}")
        for outcome in unit_run.outcomes:
            mark = {"ok": "ok ", "failed": "FAIL", "skipped": "skip", "incomplete": "????"}[
                outcome.status
            ]
            detail = f"  {outcome.detail}" if outcome.detail else ""
            print(f"  [{mark}] {outcome.kind:10s} {outcome.plugin:26s}{detail}")
        for uri in unit_run.evidence:
            print(f"  evidence: {uri}")
    verdicts = [v for u in run.units for v in u.verdicts]
    print()
    # Three words for three states, and two distinct non-zero exits. A caller
    # that only checks for zero keeps behaving as it did; one that wants to
    # tell "checked and did not like it" from "did not check" now can, without
    # parsing this line.
    said = {
        RunStatus.PASSED: "all passed",
        RunStatus.INCOMPLETE: "INCOMPLETE -- something could not run, so nothing here is a pass",
        RunStatus.FAILED: "FAILED",
    }[run.status]
    print(f"{len(run.units)} unit(s), {len(verdicts)} verdict(s), {said}")
    if run.status is RunStatus.INCOMPLETE:
        for unit_run in run.units:
            for outcome in unit_run.outcomes:
                if outcome.status == "incomplete" and not outcome.waived:
                    print(f"  could not run: {outcome.kind} {outcome.plugin}")
    if getattr(args, "report_only", False):
        # hpc-devsecops's default: say everything, block nothing, exit 0. The
        # engine's default is the other way round, because `recast run` is what
        # CI and the pre-push hook call and both need the exit status to mean
        # something. The hook passes nothing and gets the blocking form.
        return 0
    return {RunStatus.PASSED: 0, RunStatus.FAILED: 1, RunStatus.INCOMPLETE: 2}[run.status]


def _cmd_doctor(_args: argparse.Namespace) -> int:
    print(f"recast {__version__}  python {sys.version.split()[0]}")
    total = sum(len(REGISTRY.names(k)) for k in KINDS)
    print(f"{total} plugin(s) registered across {len(KINDS)} kinds")
    broken = REGISTRY.broken()
    for key, reason in sorted(broken.items()):
        print(f"BROKEN {key}: {reason}", file=sys.stderr)
    return 1 if broken else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recast", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version").set_defaults(func=_cmd_version)

    plugins = sub.add_parser("plugins", help="list registered plugins")
    plugins.add_argument("--kind", choices=KINDS)
    plugins.set_defaults(func=_cmd_plugins)

    sub.add_parser("recipes", help="list available recipes").set_defaults(func=_cmd_recipes)

    plan = sub.add_parser("plan", help="show a recipe's stages without running it")
    plan.add_argument("recipe")
    plan.add_argument("--config", help="JSON object of recipe config")
    plan.set_defaults(func=_cmd_plan)

    run = sub.add_parser("run", help="walk a recipe's stages over a source tree")
    run.add_argument("recipe")
    run.add_argument("root", help="source tree to run over")
    run.add_argument("--config", help="operator config, .json or .toml")
    run.add_argument(
        "--unit", action="append", help="unit uid to run (repeatable; default: top-level units)"
    )
    run.add_argument(
        "--range",
        metavar="REV..REV",
        help="scope history-reading scanners to a revision range; what the pre-push hook passes",
    )
    run.add_argument(
        "--report-only",
        action="store_true",
        help="print the outcome and exit 0 regardless; the default exits 2 for incomplete, "
        "1 for failed",
    )
    run.add_argument(
        "--summary",
        help="write the run's verification status here -- one entry per unit and verifier, "
        "stable across runs over the same revisions, meant to be committed",
    )
    run.set_defaults(func=_cmd_run)

    sub.add_parser("doctor", help="check the installation").set_defaults(func=_cmd_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RecastError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
