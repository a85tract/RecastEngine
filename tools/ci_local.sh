#!/usr/bin/env bash
#
# Run the workflows' jobs on this machine, before spending a runner on them.
#
# The commands are not written here. They are read out of .github/workflows/ at
# the moment you run this, because a local mirror of CI that is maintained by
# hand stops being a mirror the first time someone edits a workflow and not the
# script -- and it stops silently, still reporting green for a set of steps
# that is no longer the set CI runs. If a step here is wrong, the workflow is
# wrong.
#
#   tools/ci_local.sh                    # every job in every workflow
#   tools/ci_local.sh spine secrets      # one or more jobs by name
#   tools/ci_local.sh --list             # what is there, and what will be skipped
#   tools/ci_local.sh --workflow ci.yml  # one workflow's jobs
#
# Three rewrites, all printed rather than assumed:
#
#   --isolated is added to every `uv run`. A working .venv here usually has a
#   domain extension installed and the runner never does, which makes an
#   ordinary local pytest the *other* configuration -- and the one P4's "the
#   engine passes with the extension uninstalled" is not checked in.
#
#   A bare `python` becomes `python3`, for the same reason and a sharper one:
#   the runner's `python` is what actions/setup-python put there, while here it
#   is whatever is first on PATH, which in this repository is the .venv.
#
#   Steps that install a tool install nothing. The tool is looked for on PATH
#   instead and its local version reported next to the observation that the
#   runner's will differ -- which for a scanner pinned to a version is a real
#   difference, not a formality.
#
# Three exits, because "would CI pass" has three answers. 0 is yes, 1 is no,
# and 2 is that a job could not be run here at all -- a tool it installs is
# absent, so nothing was checked. 2 is not a pass with a caveat: a pre-flight
# that reports success for a check it never ran is the exact failure it exists
# to prevent.
#
# What this cannot do is its last line: this is one machine, and it says
# nothing about the runner's architecture. The port side's ULP counts are the
# case that matters, and are why port-verification.json is reported as changed
# rather than treated as a failure.
set -euo pipefail

cd "$(dirname "$0")/.."
WORKFLOW_DIR=".github/workflows"
[ -d "$WORKFLOW_DIR" ] || { echo "no $WORKFLOW_DIR here" >&2; exit 1; }

WORKFLOWS=()
MODE=run
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --list)     MODE=list; shift ;;
        --workflow) WORKFLOWS+=("$WORKFLOW_DIR/${2#"$WORKFLOW_DIR/"}"); shift 2 ;;
        -*)         echo "unknown option $1" >&2; exit 1 ;;
        *)          ARGS+=("$1"); shift ;;
    esac
done
if [ ${#WORKFLOWS[@]} -eq 0 ]; then
    while IFS= read -r f; do WORKFLOWS+=("$f"); done < <(find "$WORKFLOW_DIR" -name '*.yml' | sort)
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Reads one workflow into $WORK: an index of job<TAB>kind<TAB>value in file
# order, with each step's script written to its own file so that a `run: |`
# block survives intact. A block flattened to its first line would be worse
# than not running it -- it would pass.
read_workflow() {
    awk -v work="$WORK" -v wf="$1" -v tag="$(basename "$1" .yml)" '
        function flush_block() {
            if (blockfile != "") { close(blockfile); blockfile = ""; contentindent = 0 }
        }
        function step_file() {
            # Tagged by workflow: awk restarts its counter per file, and two
            # workflows sharing a step number would overwrite one another,
            # which shows up as one job running a different job steps.
            nstep++
            return work "/step." tag "." nstep ".sh"
        }
        /^jobs:/ { injobs = 1; next }
        !injobs { next }

        # Inside a `run: |` block: anything indented deeper than the `run:` key
        # belongs to it, a blank line included.
        blockfile != "" {
            if ($0 ~ /^[[:space:]]*$/) { print "" >> blockfile; next }
            match($0, /^[[:space:]]*/)
            if (RLENGTH > blockindent) {
                # A block scalar is dedented by the indent of its first
                # content line, not by a guess at the nesting step.
                if (contentindent == 0) contentindent = RLENGTH
                print substr($0, contentindent + 1) >> blockfile
                next
            }
            flush_block()
        }

        /^  [a-z][a-z0-9_-]*:[[:space:]]*$/ {
            job = $1; sub(/:$/, "", job)
            print job "\tjob\t" wf
            next
        }
        /^    if:/ { $1 = ""; sub(/^[[:space:]]+/, ""); print job "\tif\t" $0; next }

        # A step is `- run:` or, when it carries a `name:`, a bare `run:`.
        /^[[:space:]]+-?[[:space:]]*run:[[:space:]]*\|[[:space:]]*$/ {
            match($0, /^[[:space:]]*/); blockindent = RLENGTH
            blockfile = step_file(); contentindent = 0
            printf "" > blockfile
            print job "\trun\t" blockfile
            next
        }
        /^[[:space:]]+-?[[:space:]]*run:/ {
            line = $0
            sub(/^[[:space:]]+-?[[:space:]]*run:[[:space:]]*/, "", line)
            f = step_file()
            print line > f
            close(f)
            print job "\trun\t" f
            next
        }
        END { flush_block() }
    ' "$1"
}

INDEX="$WORK/index"
: > "$INDEX"
for wf in "${WORKFLOWS[@]}"; do read_workflow "$wf" >> "$INDEX"; done

field() { awk -F'\t' -v j="$1" -v k="$2" '$1==j && $2==k {print $3; exit}' "$INDEX"; }
JOBS=$(awk -F'\t' '$2=="job" {print $1}' "$INDEX" | awk '!seen[$0]++')

# The matrix is one line of YAML and expanding it in general is a project. The
# jobs here have one axis, so take its first value and say so.
MATRIX_PYTHON=$(sed -n 's/^ *python: \[\"\([0-9.]*\)\".*/\1/p' "${WORKFLOWS[@]}" | head -1)
MATRIX_ALL=$(sed -n 's/^ *python: \[\(.*\)\]/\1/p' "${WORKFLOWS[@]}" | head -1 | tr -d '" ')

if [ "$MODE" = list ]; then
    for job in $JOBS; do
        cond=$(field "$job" if)
        printf '%-14s %-28s' "$job" "$(field "$job" job)"
        [ -n "$cond" ] && printf 'skipped: runs only when [%s]' "$cond"
        printf '\n'
    done
    exit 0
fi

WANTED="${ARGS[*]:-}"
[ -n "$WANTED" ] || WANTED=$(echo $JOBS)
for job in $WANTED; do
    echo "$JOBS" | grep -qx "$job" || { echo "no job '$job' in ${WORKFLOWS[*]}" >&2; exit 1; }
done

# A step whose purpose is to put a tool on the runner. Locally the tool is
# either already there or the job cannot run; installing one behind the user's
# back is not this script's business.
provided_tool() {
    # `|| true`: an ordinary step provides no tool, and grep saying so is not
    # an error -- but under `set -e` an empty pipeline would end the run.
    sed -n 's/.*tar -xz -C \/usr\/local\/bin \([a-z0-9_-]*\).*/\1/p; s/.*apt-get install -y \(.*\)/\1/p' "$1" \
        | tr ' ' '\n' | grep -v '^$' | sort -u || true
}

failed="" ran="" unrun=""
for job in $WANTED; do
    cond=$(field "$job" if)
    if [ -n "$cond" ]; then
        # Not a pass. There is no event here, so inventing one would check
        # something other than what CI checks.
        printf '\n=== %s (%s): skipped, runs only when [%s]\n' "$job" "$(field "$job" job)" "$cond"
        continue
    fi
    printf '\n=== %s (%s)\n' "$job" "$(field "$job" job)"

    job_failed="" job_unrun=""
    while IFS=$'\t' read -r j kind step; do
        [ "$j" = "$job" ] && [ "$kind" = "run" ] || continue

        # Before anything reads the step, so a skipped one reports the command
        # it would have been rather than the template.
        sed -i '' "s/\${{ matrix.python }}/$MATRIX_PYTHON/g" "$step" 2>/dev/null \
            || sed -i "s/\${{ matrix.python }}/$MATRIX_PYTHON/g" "$step"

        tools=$(provided_tool "$step")
        if [ -n "$tools" ]; then
            for tool in $tools; do
                if command -v "$tool" >/dev/null 2>&1; then
                    echo "  [have] $tool -- $("$tool" --version 2>&1 | head -1 | tr -d '\r')"
                    echo "         (the runner installs its own; versions will differ)"
                else
                    echo "  [MISS] $tool is not on PATH"
                    job_unrun=1
                fi
            done
            [ -n "$job_unrun" ] && break
            continue
        fi

        if grep -q '^uv python install' "$step"; then
            echo "  [skip] $(head -1 "$step") -- uv resolves interpreters here as it needs them"
            continue
        fi

        sed -i '' -e 's|uv run |uv run --isolated |g' -e 's|^python |python3 |g' "$step" 2>/dev/null \
            || sed -i -e 's|uv run |uv run --isolated |g' -e 's|^python |python3 |g' "$step"

        sed 's/^/  [run ] /' "$step"
        if ! bash -euo pipefail "$step"; then
            job_failed=1
            break
        fi
    done < "$INDEX"

    if [ -n "$job_unrun" ]; then
        echo "  --> $job NOT RUN -- install what it needs, or let the runner do it"
        unrun="$unrun $job"
    elif [ -n "$job_failed" ]; then
        echo "  --> $job FAILED"; failed="$failed $job"
    else
        echo "  --> $job ok"; ran="$ran $job"
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
    [ -n "$unrun" ] && echo "NOT RUN:$unrun"
    exit 1
fi
echo "ok:$ran"
if [ -n "$unrun" ]; then
    # Exit 2 rather than 0: nothing failed, but the question this script exists
    # to answer -- would CI pass -- has no answer for these. A pre-flight that
    # reports success for a check it could not run is the failure it is for.
    echo "NOT RUN:$unrun"
    echo "Nothing failed, but these were not checked. Exit 2 says so."
    echo "This machine only. It says nothing about the runner's architecture."
    exit 2
fi
echo "This machine only. It says nothing about the runner's architecture."
