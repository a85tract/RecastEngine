"""The ``recast`` command line.

P1 ships the introspection surface only -- enough to see what is registered and
what a recipe would do, before any of it can actually run. ``recast doctor`` and
``recast plan`` are how you check that a plugin set is wired correctly without
burning HPC allocation to find out.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from recast import __version__
from recast.errors import RecastError
from recast.recipes import BUILTIN
from recast.registry import KINDS, REGISTRY


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
    for name, cls in sorted(BUILTIN.items()):
        print(f"{name:10s} {cls.summary}")
    for name in REGISTRY.names("recipe"):
        if name not in BUILTIN:
            print(f"{name:10s} (plugin)")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    """Show the stages a recipe would run, and which plugins are missing.

    Dry only. This is the cheap check that a config is coherent before a batch
    oracle costs hours.
    """
    config = json.loads(args.config) if args.config else {}
    cls = BUILTIN.get(args.recipe)
    if cls is None:
        raise RecastError(f"unknown recipe {args.recipe!r}; try `recast recipes`")
    recipe = cls()

    problems = recipe.validate(config)
    for problem in problems:
        print(f"config: {problem}", file=sys.stderr)

    missing = 0
    for i, stage in enumerate(recipe.stages(config), 1):
        available = stage.plugin in REGISTRY.names(stage.kind)
        mark = "ok " if available else ("opt" if stage.optional else "MISS")
        flags = " ".join(f for f, on in (("gate", stage.gate), ("optional", stage.optional)) if on)
        print(f"{i:2d}. [{mark}] {stage.kind:12s} {stage.plugin:28s} {flags}")
        if not available and not stage.optional:
            missing += 1

    if missing or problems:
        print(f"\n{missing} required plugin(s) unavailable, {len(problems)} config problem(s)")
        return 1
    return 0


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
