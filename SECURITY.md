# Security Policy

## Reporting a vulnerability in RecastEngine

Do **not** open a public issue. Use a
[private security advisory](https://github.com/a85tract/RecastEngine/security/advisories/new),
or email **Yueqi Chen** <yueqi.chen@colorado.edu>.

Expect an acknowledgement within five working days and a coordinated-disclosure
window agreed case by case.

## Vulnerabilities *found by* RecastEngine

The `audit` recipe exists to find defects in other software, so this repository
sits upstream of a stream of unpatched findings. The rules that govern them:

- Findings default to `Access.EMBARGOED` and `Disclosure.PLAUSIBLE`. A scanner
  can only classify downward through an explicit, human-reviewed decision.
- Embargoed findings go to a `FindingStore` — Sec-Track — and nowhere else.
  `FindingStore.guard()` raises `AccessViolation` on any attempt to write a
  record into a store not cleared to hold it, and CI asserts this.
- Nothing reaches the public evidence store, a CI log, or an issue until
  disclosure is complete: `Access.PUBLIC` **and** `Disclosure.PUBLISHED`.
- Reproducers are kept to the pattern and capability level. This project does
  not develop or publish weaponized exploits against production systems.

If you are extending the engine with a new `Scanner`, treat those defaults as
load-bearing. The failure mode — a 0-day landing in a public log — is not
recoverable by editing it out afterwards.

## Scope

Report against this repository only. Findings in a modernized product belong to
that product's repository; findings in NCAR or upstream HPC software follow
Sec-Track's coordinated-disclosure process, not this one.
