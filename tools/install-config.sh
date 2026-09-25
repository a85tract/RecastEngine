#!/usr/bin/env bash
#
# Install the audit recipe's config files into a target repository.
#
# Usage: install-config.sh [OPTIONS] [TARGET_REPO]   (defaults to the current repo)
#
# Options:
#   --force    overwrite files that already exist
#   -h, --help this help
#
# Installs three files `recast run audit` looks for in the TARGET repo:
#
#   .gitleaks.toml         secret-scan rules and allowlist (gitleaks reads it
#                          from the scanned tree)
#   .vex/openvex.json      CVE suppressions, passed to grype (starts empty)
#   .recast-audit.json     the run config the pre-push hook hands to recast
#
# Without the first two each check degrades quietly: gitleaks falls back to
# default rules with no project allowlist, and grype reports every CVE with no
# way to mark one not-affected. Copies are real files, not symlinks, because
# the target repo has to commit them so CI sees the same config the local gate
# does -- and because each repo should be free to tune its own rules.
#
# From hpc-devsecops's tools/install-config.sh (CESM-CC-Test, Chien-Wei Huang).
# The third file that script installed, .github/scripts/ai_audit.py, is a
# Scanner now (the domain extension's `llm-audit`) and needs no copy in the target.

set -euo pipefail

FORCE=0
REPO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force)   FORCE=1 ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)        echo "unknown option: $1" >&2; exit 2 ;;
    *)         REPO="$1" ;;
  esac
  shift
done

REPO="${REPO:-$PWD}"
REPO="$(cd "$REPO" && git rev-parse --show-toplevel)"
SELF="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$SELF/templates"

echo "recast audit: installing config into $REPO"
echo

INSTALLED=0
SKIPPED=0

install_one() {
  local rel="$1"
  local from="$SRC/$rel"
  local to="$REPO/$rel"

  if [ -e "$to" ] && [ "$FORCE" != 1 ]; then
    echo "  skip  $rel already exists (use --force to overwrite)"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  mkdir -p "$(dirname "$to")"
  cp "$from" "$to"
  echo "  ok    $rel"
  INSTALLED=$((INSTALLED + 1))
}

install_one ".gitleaks.toml"
install_one ".vex/openvex.json"
install_one ".recast-audit.json"

echo
echo "installed=$INSTALLED skipped=$SKIPPED"
echo
echo "Next:"
echo "  1. Commit these files -- CI reads the same config the local gate does."
echo "  2. .vex/openvex.json ships with no statements, so it suppresses nothing"
echo "     yet. Add one statement per CVE you have assessed; an empty document"
echo "     means every finding is reported, which is the safe default."
echo "  3. Tune .gitleaks.toml against a full-history scan before trusting it:"
echo "       gitleaks dir \"$REPO\" --config \"$REPO/.gitleaks.toml\""
echo "  4. Install the hook: tools/install-hooks.sh \"$REPO\". Every push then"
echo "     runs 'recast run audit' over the pushed range and blocks on a finding"
echo "     or an incomplete check."
