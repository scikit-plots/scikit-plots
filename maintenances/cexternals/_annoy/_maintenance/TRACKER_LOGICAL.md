# Logical tracker

| Contract | Verification boundary |
|---|---|
| Shared headers have one source owner | Resolved Cython/C++ paths and declared consumer sets |
| `Annoy` native type differs from Cython `Index` | Both dependency routes are recorded; clean extension builds still required |
| Corpus retains backend selection | Both index imports required in runtime evidence |
| MCP delegates to Corpus | `RetrievalIndex` import required; direct Annoy imports rejected |
| Runtime stays independent of maintenance | Structured import checks over upstream and six consumers |
| Recorded state describes this input | Input archive provenance is distinct from live runtime inventory |
| Prior test success cannot cover changed input | Input digest and log hashes checked on every review |

Native invariants remain behavior obligations: exception/error ownership,
no-fail destruction, dtype precision, bounds, RNG stream continuity, mmap/file
lifecycle, atomic save/load and persistence compatibility. Static dependency
checks establish none of those by themselves. See [VERIFICATION.md](VERIFICATION.md).
