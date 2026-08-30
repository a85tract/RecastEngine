# Observing a run

`run_recipe` accepts an optional `RunObserver`. It is a read-only boundary for
a UI, audit recorder, or outer orchestrator that needs to know what one recipe
attempt did without putting orchestration into the engine.

```python
from recast import RunEvent, RunObserver
from recast.run import run_recipe


class Recorder(RunObserver):
    def observe(self, event: RunEvent) -> None:
        append_event(event.to_record())


result = run_recipe(recipe, root, config, observer=Recorder())
```

The observer sees ordered start and finish edges for the run; executor and
frontend initialization; frontend discovery; each selected Unit and its
frontend analysis; each walked stage; and every Candidate, Verdict, and
Evidence lifecycle. Run-scoped stage events omit `unit_id`. `run_id` identifies
one attempt and `sequence` is its authoritative order; wall-clock timestamps
are display metadata only. `stage_index` is the declaration's occurrence in
the Recipe, so it remains unambiguous even when a Recipe reuses the same
`Stage` object. `reason_code` is stable enough for automation, while `reason`
explains the decision to an operator.

A Verdict finish carries its Candidate digest, verifier, confidence, and
pass/fail status before any store is invoked. A UI therefore does not have to
infer a gate result from human text, and a Recipe without a store still has a
complete verification trace. Metrics remain in Evidence rather than events so
the observation stream does not become another route for sensitive case data.

A gate which stops a Unit produces `skipped` finish events for the declared
stages it suppressed, with `reason_code="upstream_stop"`. An unexpected plugin
exception closes its open Candidate, stage, Unit, and run lifecycles as
`aborted` before the exception is re-raised. A normal failed Verdict remains a
normal `failed` outcome, not an exception.

Events intentionally omit source text, configuration, Facts, Candidate notes,
and Finding bodies. They still carry `access="embargoed"` by default because a
reason can reveal information about undisclosed code. Consumers must preserve
that label when storing or projecting events.

Delivery is synchronous and ordered. An observer exception aborts the run:
continuing would make the resulting audit trail look complete when it has a
hole. Durable storage, retry, and replay are responsibilities of the caller.
Retrying work means starting a new run and therefore a new `run_id`; a later
Verdict is never fed back into the Transform inside the attempt it judged.

With no observer, no run identifier, timestamp, event, or Candidate digest is
created for observation. The existing execution and return value are unchanged.
