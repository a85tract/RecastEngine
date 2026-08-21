#!/usr/bin/env bash
#
# Run the CI workflow's jobs on this machine, before spending a runner on them.
#
# The commands are not written here. They are read out of
# .github/workflows/ci.yml at the moment you run this, because a local mirror
# of CI that is maintained by hand stops being a mirror the first time someone
# edits the workflow and not the script -- and it stops silently, which is the
# failure that matters. If a step here is wrong, the workflow is wrong.
#
#   tools/ci_local.sh              # every job that can run locally
#   tools/ci_local.sh spine        # one or more jobs by name
#   tools/ci_local.sh --list       # what is there, and what will be skipped
#
# Two differences from the real runner are deliberate, and both are printed
# rather than assumed:
#
#   --isolated is added to every `uv run`. A working .venv here usually has a
#   domain extension installed in it; the CI runner never does, and P4's "the
#   engine passes with the extension uninstalled" is checked by nothing else.
#   A plain local pytest is the *other* configuration, and passing there
#   proves less.
#
#   apt-get steps are not run. The prerequisite they install is checked for on
#   PATH instead, and its local version is reported next to the runner's --
#   they will differ, and whether that matters is a property of the job.
#
# What this cannot answer: anything about the runner's architecture. This is
# whatever machine you are on. The port side's ULP counts are the known case --
# see the port-spine job's comment in ci.yml.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKFLOW=".github/workflows/ci.yml"
[ -f "$WORKFLOW" ] || { echo "no $WORKFLOW here" >&2; exit 1; }

# A `run: |` block would be silently truncated to its first line by the reader
# below, which is worse than not running it at all.
if grep -qE '^\s+- run: \|' "$WORKFLOW"; then
    echo "$WORKFLOW has a multi-line 'run: |' block; this reader handles only" >&2
    echo "single-line steps. Extend it before trusting what it reports." >&2
    exit 1
fi

# job<TAB>kind<TAB>value, in file order. `kind` is `if` for a job-level
# condition and `run` for a step.
read_workflow() {
    awk '
        /^jobs:/            { injobs = 1; next }
        !injobs             { next }
        /^  [a-z][a-z0-9_-]*:[[:space:]]*$/ {
            job = $1; sub(/:$/, "", job); next
        }
        /^    if:/          { $1 = ""; sub(/^[[:space:]]+/, ""); print job "\tif\t" $0; next }
        /^[[:space:]]+- run:/ {
            sub(/^[[:space:]]+- run:[[:space:]]*/, "")
            print job "\trun\t" $0
        }
    ' "$WORKFLOW"
}

# The matrix is one line of YAML and expanding it in general is a project. The
# jobs here have one axis, so take its first value and say so.
MATRIX_PYTHON=$(sed -n 's/^ *python: \[\"\([0-9.]*\)\".*/\1/p' "$WORKFLOW" | head -1)
MATRIX_ALL=$(sed -n 's/^ *python: \[\(.*\)\]/\1/p' "$WORKFLOW" | head -1 | tr -d '" ')

JOBS=$(read_workflow | cut -f1 | awk '!seen[$0]++')

if [ "${1:-}" = "--list" ]; then
    echo "jobs in $WORKFLOW:"
    for job in $JOBS; do
        cond=$(read_workflow | awk -F'\t' -v j="$job" '$1==j && $2=="if" {print $3}')
        if [ -n "$cond" ]; then
            echo "  $job  -- skipped: runs only when [$cond]"
        else
            echo "  $job"
        fi
    done
    exit 0
fi

# Unquoted on purpose: $JOBS arrives newline-separated, and the membership
# tests below are written against a single-space list.
# shellcheck disable=SC2116,SC2086
WANTED=$(echo ${*:-$JOBS})
for job in $WANTED; do
    echo "$JOBS" | grep -qx "$job" || { echo "no job '$job' in $WORKFLOW" >&2; exit 1; }
done

failed=""
ran=""
for job in $WANTED; do
    cond=$(read_workflow | awk -F'\t' -v j="$job" '$1==j && $2=="if" {print $3}')
    if [ -n "$cond" ]; then
        # Not a pass. There is no event here, so inventing one would check
        # something other than what CI checks.
        printf '\n=== %s: skipped, runs only when [%s]\n' "$job" "$cond"
        continue
    fi

    printf '\n=== %s\n' "$job"
    job_failed=""
    while IFS=$'\t' read -r j kind cmd; do
        [ "$j" = "$job" ] && [ "$kind" = "run" ] || continue

        # Before the dispatch below, so a skipped step reports the command it
        # would have been rather than the template.
        cmd=${cmd//'${{ matrix.python }}'/$MATRIX_PYTHON}

        case "$cmd" in
            *apt-get*)
                # Check the prerequisite instead of installing it.
                for tool in gfortran; do
                    case "$cmd" in *"$tool"*)
                        if command -v "$tool" >/dev/null 2>&1; then
                            v=$("$tool" --version 2>&1 | head -1)
                            echo "  [have] $tool -- $v"
                            echo "         (the runner installs its distribution's; versions will differ)"
                        else
                            echo "  [MISS] $tool is not on PATH; this job needs it"
                            job_failed=1
                        fi
                    esac
                done
                continue ;;
            "uv python install"*)
                echo "  [skip] $cmd -- uv resolves interpreters here as it needs them"
                continue ;;
        esac

        # The rewrite this script exists to make.
        local_cmd=${cmd/uv run /uv run --isolated }

        echo "  [run ] $local_cmd"
        if ! eval "$local_cmd"; then
            job_failed=1
            break
        fi
    done < <(read_workflow)

    if [ -n "$job_failed" ]; then
        echo "  --> $job FAILED"
        failed="$failed $job"
    else
        echo "  --> $job ok"
        ran="$ran $job"
    fi
done

printf '\n'
case " $WANTED " in
    *" test "*) echo "note: the test matrix is [$MATRIX_ALL]; only $MATRIX_PYTHON ran here." ;;
esac

# The port summary is committed but deliberately not gated on, so a change in
# it must be visible without being fatal. See ci.yml's port-spine job.
if ! git diff --quiet -- examples/toy_physics/port-verification.json 2>/dev/null; then
    echo "note: examples/toy_physics/port-verification.json changed in the working tree."
    echo "      ULP counts are not device-independent, so this is a question rather"
    echo "      than a failure. git diff it, and decide."
fi

if [ -n "$failed" ]; then
    echo "FAILED:$failed"
    exit 1
fi
echo "ok:$ran"
echo "This machine only. It says nothing about the runner's architecture."
