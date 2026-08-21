# Writing a plugin

Everything the engine can do arrives this way, in-tree parts included. There is
no privileged path.

## 1. Pick a kind

`frontend` `transform` `oracle` `verifier` `scanner` `adjudicator` `executor`
`store` `agent` `recipe` — see [architecture.md](architecture.md) for what each
one may and may not do.

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
it. For tests and in-process use, `recast.registry.register(kind, name, factory)`
is equivalent.

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
`audit.llm` -- and unique per kind, which the registry enforces by refusing a
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

## Scanners specifically

`Finding.access` defaults to `EMBARGOED` and `disclosure` to `PLAUSIBLE`. Do not
override either downward. Emit uncertain findings freely — precision is the
`Adjudicator`'s job, and a finding suppressed at scan time is lost, while a
false positive costs one adjudication pass.

Read [SECURITY.md](../SECURITY.md) before writing one.
