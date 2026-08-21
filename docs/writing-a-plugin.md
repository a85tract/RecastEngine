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

Dotted and namespaced: `translate.numpy`, `port.jax`, `differential.bitexact`,
`audit.llm`. The name goes into every Evidence record, so treat a rename as a
breaking change.

## Non-deterministic transforms

If your transform calls an `AgentProvider`, set `deterministic = False` and
record the model, prompt digest, and sampling parameters in `Candidate.notes`.
Reproducible does not mean deterministic — it means someone can reconstruct what
produced this artifact. A transform that cannot be reconstructed cannot be
trusted no matter what the verifier says, because the next run is a different
artifact.

## Scanners specifically

`Finding.access` defaults to `EMBARGOED` and `disclosure` to `PLAUSIBLE`. Do not
override either downward. Emit uncertain findings freely — precision is the
`Adjudicator`'s job, and a finding suppressed at scan time is lost, while a
false positive costs one adjudication pass.

Read [SECURITY.md](../SECURITY.md) before writing one.
