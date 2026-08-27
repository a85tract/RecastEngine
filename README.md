# RecastEngine

**The modernization engine of [SciRecast](https://github.com/a85tract/SciRecast).**

LLM agents do the labor of modernizing legacy scientific software; RecastEngine
is the part that is reusable across every such effort, and the part that refuses
to let anything through that has not been checked against the original.

> **Status: pre-alpha.** The plugin contract, the CLI, and the whole
> `translate` recipe exist and run end to end — Fortran in, verified NumPy
> and evidence manifests out, gated bit-exact against the compiled original.
> The other three recipes have their contracts and await their plugins
> (P3–P4) — see [`docs/roadmap.md`](docs/roadmap.md).

## Four workloads, one spine

RecastEngine is not a translator. Translation is one recipe of four, and all
four are the same five steps with different plugins in the slots:

```
discover  ->  analyze  ->  transform  ->  verify  ->  record
  Unit        Facts        Candidate     Verdict    Evidence
```

| Recipe | What it does | Example product |
|---|---|---|
| `translate` | Fortran → NumPy / Numba / CUDA | [PyCAM5](https://github.com/a85tract/PyCAM5) |
| `refactor` | carve a Python control plane into a Fortran monolith, numerics untouched | [freeCAM](https://github.com/a85tract/freeCAM) |
| `port` | retarget a kernel to an accelerator | [JaxCAM6](https://github.com/a85tract/CESM-jax-kernels) |
| `audit` | secret scan, SBOM+CVE+VEX, LLM source audit, sanitizer builds | CC-Test (cyber half, restricted access) |

## Quick start

The engine needs Python 3.11 or newer. Make a virtual environment and activate
it once; every command after that is a bare `recast`, in this shell and any
later one you activate.

```bash
uv venv --python 3.11                              # fetches a 3.11 if you have none
source .venv/bin/activate                          # Windows: .venv\Scripts\activate
uv pip install -e ".[fortran,translate,verify]"
```

Without `uv`, the same three lines are `python3 -m venv .venv` (a `python3` of
3.11 or newer — the one Apple ships is older than that) and `pip install -e`
with the same extras.

```console
$ recast doctor
recast 0.0.1.dev0  python 3.11.16
21 plugin(s) registered across 10 kinds
```

`recast recipes` lists the workloads. To run the whole `translate` recipe over
the shipped example you also need a `gfortran` on PATH — the reference really
is compiled:

```console
$ recast run translate examples/toy_physics --config examples/toy_physics/recast.json
fortran:toy_physics
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy
  [ok ] verifier   static.rwset                sampled: 4 blocks match
  [ok ] oracle     f2py-golden                 f2py:toy_physics:3f3e0f78...
  [ok ] verifier   differential.bitexact       bit_exact: 85 points across 2 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

One command, eight stages: the module is translated, its dataflow
cross-checked against the source's, the untouched Fortran compiled as the
reference, every output compared bit for bit, and three evidence manifests
written under `examples/toy_physics/.recast/evidence/` — because a candidate
without evidence is a draft, however good it looks. The generated Python
itself lands beside them under `.recast/translate/.../candidate/`.

`recast plan` dry-runs a recipe and names the plugin filling every slot before
anything costs compute. It is also how a backend is chosen — the same recipe,
retargeted from NumPy to Numba by one config key:

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

`numpy`, `numba`, and `cuda` all resolve. `[MISS]` is a slot nothing is
registered for — a name the config got wrong, a plugin an extension was meant
to supply — reported before a run has spent a compile on it. It is a
registration check and not an import, so a backend whose Python dependency is
absent still reads `[ok]` here.

What three `[ok]`s do not say is that the three carry the same evidence. Only
`numpy` records the dataflow protocol the `static.rwset` gate on line 4 reads,
and that gate fails closed, so a `translate` run retargeted to `numba` or
`cuda` stops there. Both of those are relays of an upstream backend and are
checked by comparing emitted bytes against it rather than by this recipe;
[`docs/roadmap.md`](docs/roadmap.md) records how far each one's evidence
reaches.

The core installs with zero dependencies and stays importable without a
compiler, a GPU, or a model provider; everything heavier is an extra
(`fortran`, `translate`, `verify`, `numba`, `jax`, `agents`).

To extend it, implement one of the ten interfaces in
[`src/recast/plugins/`](src/recast/plugins/) and register an entry point —
see [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md).

## The same recipe on code we did not write

`examples/toy_physics` is ours, which makes it the weaker demo.
[`corpus/`](corpus/) pins twelve open-source Fortran libraries as submodules and
holds the engine — with no domain extension installed — to translating them.
One unit of one of them, end to end, on your machine:

```bash
git submodule update --init --depth 1 corpus/numfor
python tools/corpus.py stage numfor
recast run translate corpus/.build/numfor --unit fortran:basic
```

`stage` reads [`corpus/cases.json`](corpus/cases.json) for what belongs to the
case and lays it out somewhere the engine is free to write. For `numfor` that
is the 133 `.f90` and `.inc` files under `src/`, its test tree left out,
flattened into a fresh `corpus/.build/numfor/` — flat because an
`#include "qtrs1d.inc"` names no directory. A case carrying `.F90` sources goes
through `gfortran -E -P -cpp` on the way. The submodule is only ever read.

```console
fortran:basic
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             5 deferred block(s)
  [ok ] verifier   static.rwset                sampled: 55 blocks match
  [ok ] oracle     f2py-golden                 f2py:basic:f4038505...
  [ok ] verifier   differential.bitexact       bit_exact: 10 points across 1 subprogram(s), all bit-exact
  [ok ] verifier   symbolic.notary             symbolic: no rewrites to notarize; the translation is print-order faithful
  [ok ] store      fs-evidence                 3 verdict(s) recorded

1 unit(s), 3 verdict(s), all passed
```

`basic` is [numfor](https://github.com/numericfor/numfor)'s 354-line utility
module — kinds, timers, a date stamp, `is_inf` — at the commit the submodule
pins. Two things to open under `corpus/.build/numfor/.recast/`:

| | |
|---|---|
| `translate/fortran_basic/candidate/basic_numpy.py` | the generated Python, every block carrying the source lines it came from |
| `evidence/fortran_basic/*.json` | one manifest per verdict — artifact digest, oracle key, metrics |

The `5 deferred block(s)` are the rules declining to guess: two `cpu_time`
calls, a `date_and_time`, and two formatted internal writes, each left standing
as a `raise NotImplementedError` for a human to answer. Everything else in the
module is translated, and checked.

The other eleven cases are listed in [`corpus/README.md`](corpus/README.md);
`corpus/baseline.json` records how far the engine gets on each.

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | the spine, the ten interfaces, where the boundaries fall |
| [`docs/writing-a-plugin.md`](docs/writing-a-plugin.md) | how to extend the engine |
| [`docs/roadmap.md`](docs/roadmap.md) | phases P0–P6 |
| [`examples/`](examples/) | source trees the shipped recipes run over end to end |
| [`corpus/`](corpus/) | twelve third-party Fortran libraries the engine alone is held to, and the record of how far it gets |
| [`conformance/`](conformance/) | what a plugin must satisfy, as a suite that runs |

## Contact

| | |
|---|---|
| Bugs, features | issues in this repository |
| Vulnerabilities | [`SECURITY.md`](SECURITY.md) — private advisory, never a public issue |
| Collaboration, licensing | **Yueqi Chen**, University of Colorado Boulder — <yueqi.chen@colorado.edu> |

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).
Improving the engine improves it for everyone using it — the plugin contract is
the same one every extension uses, so a plugin you write is not second-class.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
