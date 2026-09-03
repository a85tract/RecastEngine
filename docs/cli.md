# The CLI

```console
$ recast doctor
recast 0.0.1.dev0  python 3.11.16
```

`doctor` then prints the live plugin count and registry inventory. The count is
intentionally not fixed here: installing an out-of-tree frontend, engine, or
other plugin changes it without changing RecastEngine.

| | |
|---|---|
| `recast doctor` | version, interpreter, how much is registered |
| `recast recipes` | the six shipped recipes |
| `recast engines [--json]` | translation engine manifests for people or an outer pipeline builder |
| `recast plugins` | what is registered, by kind |
| `recast plan <recipe>` | the stages a run would walk, and which slots nothing fills |
| `recast run <recipe> <tree>` | walk them over a source tree |
| `recast version` | the version alone |

`recast engines --json` emits the canonical catalog and its digest. Engine
manifests describe artifact contracts and select a default recipe; they do not
run work themselves. See [translation-engines.md](translation-engines.md).

## The six recipes

```console
$ recast recipes
audit           Secret scan and SBOM/CVE/VEX, gating the way hpc-devsecops does.
port            Retarget a kernel to an accelerator; gate on captured production dumps.
python-to-jax   Lower Python/NumPy functions to JAX and verify against the source.
python-to-numba Compile Python/NumPy functions with Numba and verify against the source.
refactor-todo   Restructure architecture without touching numerics; gate on a full run.
translate       Translate a source language to a target language, gated bit-exact.
```

Six recipes, one spine — every recipe is the same five steps
(`discover → analyze → transform → verify → record`) with different plugins in
the slots, which is [architecture.md](architecture.md)'s subject.

They are not equally far along. `translate` runs end to end, gated bit-exact.
`port` and `audit` plan clean on the plugins shipped here. `refactor-todo` declares
four slots nothing fills yet. [`roadmap.md`](roadmap.md) is where each is
going.

## plan

`plan` resolves a recipe against the registry and compiles nothing, which
makes it the cheap check before an oracle costs hours:

```console
$ recast plan translate --config '{"target": "numba"}'
 1. [ok ] executor     local
 2. [ok ] frontend     fortran
 3. [ok ] transform    translate.numba
 4. [ok ] verifier     static.rwset                 gate
 5. [ok ] oracle       f2py-golden
 6. [ok ] verifier     differential.bitexact        gate
 7. [ok ] verifier     symbolic.notary              optional
 8. [ok ] store        fs-evidence
```

`[MISS]` means nothing is registered under the name the recipe asked for. What
that does and does not tell you is in
[`writing-a-plugin.md`](writing-a-plugin.md); how far each `translate`
target's evidence actually reaches is in [`roadmap.md`](roadmap.md).

## Configuration

`plan --config` takes a JSON object on the command line. `run --config` takes
a path to a `.json` or `.toml` file. Same object either way, and
`examples/toy_physics/recast.json` uses all three kinds of key:


```json
{
  "units": ["fortran:toy_physics"],
  "stages": {
    "differential.bitexact": {
      "trials": 5,
      "dims": {"n": 8},
      "ranges": {"rho": [0.1, 2.0], "dz": [10.0, 100.0], "w": [-5.0, 5.0]}
    }
  }
}
```

**Recipe keys** are the recipe's own, and each recipe validates its own:

| Recipe | Keys of its own |
|---|---|
| `translate` | `target` — `numpy` (default), `numba`, `cuda`, or `tree`: the NumPy translation of a unit that `use`s sibling modules in the same tree, with their translations bundled into the candidate so the gate can import it |
| `port` | `backend` — `jax` (default), `numba` or `cuda`. `oracle` — what to gate against, `numpy-anchor` by default; `dump-replay` also requires `dumps` |
| `refactor-todo` | `reference_commit`, required. And an `executor` that is not `local`, because the gate is a batch oracle |
| `audit` | none |
| `python-to-numba` | fixed `target=numba`, `frontend=python-numpy`; local executor |
| `python-to-jax` | fixed `target=jax`, `frontend=python-numpy`; local executor |

The four legacy recipes also read `executor` and `frontend`, defaulting to
`local` and `fortran`. The two Python accelerator engine manifests pin both
values so an outer launcher cannot silently change their artifact contract.

`plan` reports a value a recipe rejects and a required key that is absent,
before anything runs. A key no recipe knows is ignored rather than flagged, so
check the spelling — the validators answer for the keys they own, and there is
no schema over the rest.

**`units`** selects what to walk; the default is every top-level unit. `--unit
fortran:basic` does the same from the command line and is repeatable.

**`stages`** passes settings through to one plugin, by the name the registry
knows it under. One worth knowing about:

```json
{"stages": {"translate.numpy": {"poison_undefined": true}}}
```

A Fortran local is undefined until assigned, and this backend gives every one
of them an initializer — which makes a read-before-write *reproducible*, not
*visible*. `np.empty` usually lands on an OS zero page, so the answer looks
deterministic and drifts later, once the heap is dirty. `poison_undefined`
NaN-fills those float arrays instead of merely allocating them, so the read
propagates into the outputs the gate compares and `differential.bitexact`
counts it as `nan_mismatch`. Scalars are untouched.

`poison_integers` is the second arm and is off even when the first is on: an
integer array is filled with `INT32_MIN + 1`, an impossible index, and nothing
propagates to a NaN scan — a read of an unwritten cell either crashes on the
subscript or shifts the run's outputs, so its detector is an A/B diff against
the unpoisoned run rather than the gate already running.

It is a separate run, not a mode of the shipped one — the candidate is
different and its digest says so.

**`output`** names where the run writes. The default is
`output/<project>/` under the working directory, where `<project>` is the
source tree's own directory name — `examples/toy_physics` gives
`output/toy_physics/`, holding `translate/<unit>/candidate/` and
`evidence/<unit>/` for each unit the run walks. `<unit>` is that unit's id
with `:` and `/` made path-safe, so the one unit of the shipped example,
`fortran:toy_physics`, writes to `translate/fortran_toy_physics/`. Nothing is
written into the source tree: generated code sitting in a checkout is one
`git add -A` from being committed as source, and a second run would discover
the first one's output as input.
`RECAST_OUTPUT_HOME` moves only the base, keeping the per-project segment;
`workspace` overrides just the `<recipe>/` half, for a run that has to build
somewhere specific.

## The flags on `run`

| | |
|---|---|
| `--config PATH` | the object above, `.json` or `.toml` |
| `--unit UID` | one unit, repeatable |
| `--summary PATH` | write the verification status per unit and verifier — stable across runs over the same revisions, meant to be committed. CI gates on `git diff --exit-code` over it |
| `--report-only` | print the outcome and exit 0 regardless |
| `--range REV..REV` | scope history-reading scanners to a revision range; what the pre-push hook passes |

## Exit codes

| | |
|---|---|
| 0 | everything passed |
| 1 | a gate failed |
| 2 | a stage could not reach a verdict — or the command itself errored |

`incomplete` is 2 and not 0 on purpose. A verifier whose build failed or whose
scheduler rejected the job has not passed; it has not run, and `recast run` is
what CI and the pre-push hook call, so the exit status has to tell them apart.
`--report-only` is the opt-out, for when you want the report without the block.
