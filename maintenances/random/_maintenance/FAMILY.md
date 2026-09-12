# Family contract for `scikitplot.random`

`scikitplot.cexternals._annoy` is the owner of shared native sources. `scikitplot.random` is a direct Cython consumer of `kissrandom.h`. The dependency is expressed by an explicit relative `cdef extern from` path, not by ownership transfer.

Current edge:

```text
scikitplot/cexternals/_annoy/src/kissrandom.h
    -> scikitplot/random/_kiss/kiss_random.pxd
```

Rules:

- Do not duplicate `kissrandom.h` under `scikitplot/random`.
- A shared-header behavior change is an upstream change even when discovered by this module.
- Recheck Cython declarations whenever the upstream header changes; textual path resolution does not establish ABI/semantic parity.
- Keep this module's public API, typing and tests independently owned here.
- Sibling consumers have independent maintenance state; do not copy findings/status between them merely to keep files identical.

The upstream owner may observe this module as a consumer, but its gate does not prove this module's public API or runtime behavior. Conversely, this module's gate does not prove the upstream native implementation.
