#!/usr/bin/env bash
#
# Install the recast pre-push hook into a target repository.
#
# Usage: install-hooks.sh [--force] [TARGET_REPO]
#
# The hook is symlinked, so updating this checkout updates the hook everywhere
# it is installed. Uninstall: rm <repo>/.git/hooks/pre-push
#
# From hpc-devsecops's tools/install-hooks.sh (CESM-CC-Test, Chien-Wei Huang),
# including the refusal to overwrite a hook that is not ours without --force.

set -euo pipefail

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; shift; fi
[ $# -le 1 ] || { echo "usage: install-hooks.sh [--force] [TARGET_REPO]" >&2; exit 2; }
REPO="${1:-$PWD}"
REPO="$(cd "$REPO" && git rev-parse --show-toplevel)"
HOOKS_DIR="$(git -C "$REPO" rev-parse --absolute-git-dir)/hooks"
SELF="$(cd "$(dirname "$0")" && pwd)"
HOOK="$SELF/pre-push"
DEST="$HOOKS_DIR/pre-push"

mkdir -p "$HOOKS_DIR"
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  current="$(readlink -f "$DEST" 2>/dev/null || true)"
  desired="$(readlink -f "$HOOK")"
  if [ "$current" != "$desired" ] && [ "$FORCE" -ne 1 ]; then
    echo "refusing to overwrite existing hook: $DEST" >&2
    echo "Re-run with --force after reviewing the existing hook." >&2
    exit 2
  fi
fi
ln -sfn "$HOOK" "$DEST" 2>/dev/null || {
  [ "$FORCE" -eq 1 ] || { echo "cannot symlink hook; use --force to install a copy" >&2; exit 2; }
  cp "$HOOK" "$DEST"
}
chmod +x "$HOOK" "$DEST" 2>/dev/null || true

echo "installed pre-push hook -> $DEST"
echo "'git push' from $REPO now runs 'recast run audit' over the pushed range first."
