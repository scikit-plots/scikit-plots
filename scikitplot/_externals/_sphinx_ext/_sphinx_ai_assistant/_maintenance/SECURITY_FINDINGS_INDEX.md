# `_sphinx_ai_assistant` Security Finding Index

Status: **REVALIDATION REQUIRED at B02**. This file is a durable index, not proof
that each finding remains at the same line number after the source anchor
changes.

The previous review identified the following high-priority families. B02 must
reproduce or close each against the exact current source before implementation
work claims success.

| ID | Priority | Finding family | Target invariant |
|---|---|---|---|
| `SEC-P0-01` | P0 | server credential attached to configurable/unbound backend destination | credentials are destination-bound |
| `SEC-P0-02` | P0 | wildcard/permissive CORS in proxy/edge paths | explicit least-privilege origins with path parity |
| `SEC-P0-03` | P0 | share locator/UUID doubles as edit authority | read and write capabilities are distinct |
| `SEC-P0-04` | P0 | alternate/direct inference relay bypasses primary controls | all service paths enforce equivalent policy |
| `SEC-P0-05` | P0 | caller/browser can supply authoritative `system` content | server exclusively owns system policy |
| `SEC-P0-06` | P0 | feedback/training contribution poisoning path | contributions remain untrusted with provenance/review |
| `SEC-P0-07` | P0 | consent/version provenance insufficient or disabled | stored contribution carries consent version/state |
| `SEC-P0-08` | P1/P0 by deployment | untrusted forwarded client identity | only trusted proxy boundary can assert forwarding identity |
| `SEC-P0-09` | P1/P0 by resource | body limit applied after excessive buffering/allocation | enforce protective limits early |
| `SEC-P0-10` | P1 | floating/root/container/supply-chain hardening gaps | reproducible least-privilege service environment |
| `SEC-P0-11` | P0 target | credential/token values can reach generated/browser config escape hatches | production secrets never client-visible/persisted |

## Closure format

For each finding, B02/B03+ must record:

```text
current source anchor
exact owner/path
reproduction/exploit precondition
current status: CONFIRMED | DISPROVED | PARTIAL | DEFERRED
fix checkpoint
regression test that bypasses the browser when service-level
rollback/compatibility impact
```

Do not close `SEC-P0-05` with client prompt sanitization alone; direct service
callers must be unable to set authoritative system policy.
