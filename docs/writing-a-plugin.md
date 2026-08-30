# Writing a plugin

Everything the engine can do arrives this way, in-tree parts included. There is
no privileged path. The ten interfaces are in
[`../src/recast/plugins/`](../src/recast/plugins/); implement one, register an
entry point under `recast.<kind>s`, and `recast plugins` shows it.

## 1. Pick a kind

`frontend` `transform` `oracle` `verifier` `scanner` `adjudicator` `executor`
`store` `agent` `recipe` `engine` — see [architecture.md](architecture.md) for
what each one may and may not do. An `engine` is an immutable declaration, not
an ABC or runnable Stage; its separate contract is documented in
[translation-engines.md](translation-engines.md).

## 2. Implement the ABC

```python
from recast.model import Candidate, Facts, Unit
from recast.plugins import Transform


class KernelToJax(Transform):
    name = "port.jax"
    requires = ("interface", "effects")  # checked before apply() is called
    deterministic = True

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return unit.kind == "subprogram" and not facts.effects.get("io")

    def apply(self, unit, facts, config) -> Candidate:
        ...
        return Candidate(
            unit=unit.uid, transform=self.name, files=files, deferred=unhandled
        )  # not an error — the agent queue
```

Two rules that are easy to get wrong:

- **Return, don't raise, on sites you cannot handle.** Put them in
  `Candidate.deferred`. A partial candidate is a useful result; an exception
  loses the 90% that did translate.
- **Never decide you are correct.** No `Transform` returns a `Verdict`.

## 3. Register it

```toml
[project.entry-points."recast.transforms"]
"port.jax" = "yourpkg.transforms:KernelToJax"
```

Group name is `recast.<kind>s`. Install the package and `recast plugins` shows
it. The registry records a stable, path-free origin for each installed plugin:
the normalized distribution name/version and exact entry-point group, name and
value. These fields identify installed metadata; they are not a package
signature or a safety verdict.

For tests and deliberately in-process extensions,
`recast.registry.register(kind, name, factory)` remains available, but it is not
allowed to impersonate an installed package. Its origin is explicitly
`source="local"`, `verification="unverified"`, with no distribution or
entry-point value:

```python
from recast.registry import Registry

registry = Registry(discover_installed=False)  # isolated unit-test inventory
registry.register("transform", "example.fake", FakeTransform)
assert registry.origin("transform", "example.fake").as_dict() == {
    "schema": "recast.plugin-origin.v1",
    "source": "local",
    "verification": "unverified",
    "distribution_name": None,
    "distribution_version": None,
    "group": "recast.transforms",
    "name": "example.fake",
    "value": None,
}
```

Use `registry.origin(kind, name)` for one immutable record or
`registry.origins(kind)` for a defensive name-to-origin snapshot. The
process-wide equivalents are `recast.registry.origin` and
`recast.registry.origins`.

Plugin addresses are unique, not ordered preferences. If two installed
distributions publish the same `(kind, entry-point name)`, discovery raises
before loading either. Discovery also raises when an installed address collides
with a plugin explicitly registered earlier in the process. `replace=True`
remains an explicit in-process override *after* discovery, and changes the
recorded origin to local/unverified; package installation order and
`setdefault()` are never used to select a winner.

Registration is also all `recast plan` checks. A slot reads `[MISS]` when no
plugin is registered under the name the recipe asked for -- a typo in the
config, an extension that was meant to be installed and is not -- and `[ok]`
otherwise. It does not import your factory, so a plugin whose own dependency
is missing plans clean and fails at run time. The one exception is a scanner
or adjudicator that declares a `tool`: `plan` looks for that binary on PATH
and reports it beside the stage. See *Scanners specifically* below.

## 4. Prove it

`conformance/` holds the suite every plugin kind must satisfy. Run it before
publishing.

It cannot find your plugins by itself, and for most kinds it could not check
them if it did: asking whether a Verifier fails closed means handing it a
candidate, an oracle and a workspace, and only you know what a valid one looks
like. So you declare a `PluginSet` -- one case per plugin, carrying the least
material its checks need -- and name it:

```python
# yourpkg/conformance.py
from recast.conformance import ExecutorCase, PluginSet

PLUGIN_SET = PluginSet(
    name="your-extension",
    executors=(ExecutorCase(name="pbs", unsatisfiable={"nodes": 4096}),),
)
```

```toml
[project.entry-points."recast.conformance"]
your-extension = "yourpkg.conformance:PLUGIN_SET"
```

```bash
uv run pytest conformance/ --plugin-set your-extension
```

A kind you do not declare is reported as unexercised rather than passing, which
is the answer you want: it says the suite was given nothing to check, instead of
leaving you to believe it found nothing wrong. `recast/conformance/builtin.py`
is the set the engine holds itself to, and the example worth copying.

## Naming

Two names, and they are allowed to differ.

The **entry-point name** is the address: what a recipe or a config asks for.
Dotted and namespaced -- `translate.numpy`, `port.jax`, `differential.bitexact`,
`scan.mytool` -- and unique per kind, which the registry enforces by refusing a
silent override.

The **`name` attribute** is the identity of the implementation: what answered.
It lands in `Verdict.verifier`, `Candidate.transform` and `Facts.provenance`, so
treat a rename as a breaking change.

One implementation may sit behind more than one address, and then the two names
differ on purpose. The CESM extension's `cesm` frontend is not a new analysis --
it is the engine's `FortranFrontend` with CAM's kind table preloaded -- so it
answers to `fortran`, and the difference between the two is recorded as the
configuration it is. What you may not do is the reverse: two different
implementations under one `name`. `conformance/` checks that one.

## Non-deterministic transforms

If your transform calls an `AgentProvider`, set `deterministic = False` and
record the model, prompt digest, and sampling parameters in `Candidate.notes`.
Reproducible does not mean deterministic — it means someone can reconstruct what
produced this artifact. A transform that cannot be reconstructed cannot be
trusted no matter what the verifier says, because the next run is a different
artifact.

**Declare it on a class of your own.** `deterministic` is read at plan time, off
the plugin the recipe names, and it is what decides whether the recipe needs a
hard gate. So agenticness cannot be switched on from a config file: a transform
that claimed determinism and then consulted a model would go through the gate
rule unseen. Wrapping the engine's transform is the shape — and is what the CESM
extension already does for its tables:

```python
class AgenticTranslation(Transform):
    name = "yourpkg.translate.agentic"
    deterministic = False  # the claim the gate reads

    def __init__(self) -> None:
        self._engine = NumpyTranslation(deterministic=False)

    def apply(self, unit, facts, config):
        candidate = self._engine.apply(unit, facts, {**config, "deferred_handler": self._fill})
        candidate.transform = self.name
        return candidate
```

`NumpyTranslation` refuses a handler unless it was constructed that way, so the
rule is a mechanism rather than a convention.

## Filling what the rules refused

Two ways in, and they are not duplicates — they differ in when they arrive.

**Data, before the run.** `config["patches"]` maps `"subprogram/block"` to a
replacement worked out ahead of time. The model, if there was one, ran out of
band; its output is now input, so the run stays reproducible bit for bit. Known
before rendering starts, so it may add module-level imports.

**Behaviour, during the run.** `config["deferred_handler"]` is a callable
consulted at the moment a block is refused. It receives a `DeferredSite` — the
Fortran, the refusal, the line span, the subprogram's name table — so it sees
what the rules saw rather than reconstructing it from a previous run's report.
It returns the body to emit plus whatever provenance it wants recorded, or
`None` to leave the site deferred. It cannot add imports, because the module
header is assembled before the subprograms are.

A handler that raises, or answers with something that is not source lines,
leaves its site deferred with the reason recorded on the block. One block's
handler failing is not the run's death, and not a silence either.

See `recast/transform/numpy/agentic.py` for the full contract.

## Gating a candidate for promotion

A populated `Candidate.deferred` is a successful discovery result, not a
complete deliverable. The shipped `translate` recipe deliberately preserves
that distinction: it remains useful for discovering and measuring unsupported
sites and does **not** include a completeness gate by default.

A domain or promotion recipe that intends to merge its output should put the
in-tree `static.complete` verifier immediately after its transform, before an
expensive oracle:

```python
from recast.plugins.recipe import Recipe, Stage


class PromoteTranslation(Recipe):
    name = "yourpkg.promote"

    def stages(self, config):
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("transform", "yourpkg.translate"),
            Stage("verifier", "static.complete", gate=True),
            Stage("verifier", "static.rwset", gate=True),
            Stage("oracle", "yourpkg.golden"),
            Stage("verifier", "yourpkg.differential", gate=True),
            Stage("store", "fs-evidence"),
        ]
```

`static.complete` fails when even one entry remains in the deferred ledger the
Transform returned, and it has no waiver configuration. Its Evidence records
the entry count and a canonical digest of that ordered ledger, not the entries
themselves; the full queue may contain private symbol names and stays in the
project's access-controlled store or workspace rather than public Evidence.

The scope of that claim is deliberately precise: **the Transform declared no
deferred work**. The verifier does not rediscover source sites independently,
so it cannot catch a Transform that forgot to put one in the ledger. A
promotion policy still needs an independently derived coverage gate from its
project specification, plus behavioural/numerical gates. Likewise, a project
that proves an untranslated boundary unreachable expresses that different
claim through a separately named project verifier with its own coverage
evidence; it must not make `static.complete` mean “no declared work except for
this project.”

## Scanners specifically

`Finding.access` defaults to `EMBARGOED` and `disclosure` to `PLAUSIBLE`. Do not
override either downward. Emit uncertain findings freely — precision is the
`Adjudicator`'s job, and a finding suppressed at scan time is lost, while a
false positive costs one adjudication pass.

**Raise `ScannerUnavailable` when you could not run, and never return an empty
iterable for it.** Empty means "I ran and found nothing", and the run is entitled
to report a clean scan on the strength of it. The tool is not on PATH, the API
refused, the sanitizer build is missing: all of those are `ScannerUnavailable`,
the stage becomes `incomplete`, and the run cannot report `passed`. The same
applies to an `Adjudicator`, where it matters more — it is usually the recipe's
gate.

If your tool emits SARIF — gitleaks, most static analyzers, an LLM audit —
use `recast.sarif.findings_from` rather than writing the translation again. It
raises `ScannerUnavailable` on a report that will not parse or that records a
failed invocation, which are the two ways a crashed tool otherwise arrives
looking like a clean scan.

Declare what you scan. `subject = "unit"` is walked once per Unit with its
Facts; `subject = "repository"` is walked once per run against a Unit that stands
for the tree — history scanners and dependency scanners are the second kind,
and walking them per Unit is N identical scans. Declare the binary you wrap in
`tool`, so `recast plan` can ask for it before the run. Run it through the
`executor` you are handed; the contract gives you one for the same reason it
gives Oracles one.

Then declare a `ScannerCase` in your conformance plugin set. It takes the
binary from your scanner's `tool`, or from its own `tool` if you need to
override it. The suite fakes it on PATH — present and clean, present and
garbage, absent entirely — and checks that your scanner can tell those apart.
See `recast.conformance.fake_tool`.

Read [SECURITY.md](../SECURITY.md) before writing one.
