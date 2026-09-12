# Source ownership and consumer boundaries

This is the _annoy owner's contract, not a document to copy into six sibling
maintenance directories. The machine-readable edges are in
[MAINTENANCE.json](../MAINTENANCE.json); observed paths and line evidence are in
[DEPENDENCY_GRAPH.json](DEPENDENCY_GRAPH.json).

| Module | Relationship | Evidence / ownership |
|---|---|---|
| `cexternals/_annoy` | Maintained upstream | Shared headers, native `Annoy`, local helpers and tests |
| `annoy` | Direct source and native API | Cython templates include `annoylib.h`, `kissrandom.h`, `annoy_type_support.h`; Python imports native `Annoy`/`AnnoyIndex` and `_plotting` |
| `random` | Direct source | `kiss_random.pxd` includes `kissrandom.h` |
| `memmap` | Direct source | `mem_map.pxd` and `.pyx` include `mman.h` |
| `impute` | Compiled index | Imports `scikitplot.annoy._annoy.Index`; also has an external Annoy fallback |
| `corpus` | Selectable index | Selects `scikitplot.annoy.Index` or `scikitplot.annoy._annoy.Index` |
| `mcp` | Transitive index | Uses `scikitplot.corpus.RetrievalIndex`; direct Annoy imports bypass this boundary |

The low-level native `Annoy` type and the Cython `Index` are distinct builds.
The upstream directory also has Python wrappers and plotting helpers; it is not
accurate to call the entire subtree dependency-free C++.

Source changes route by actual consumers. A KISS stream change affects both
index construction and the public RNG; an mmap change affects memory mapping
and index persistence. A header's existence does not establish ABI compatibility.
Rebuild the consuming extensions and exercise behavior after changing headers.

Runtime must not import maintenance or skill code. Upstream implementation must
not import the six consumers. Test imports may exercise public APIs; they are
excluded from architectural runtime edges, but retained in inventory/evidence.
Cython `.in` templates are authoritative; generated `.pyx`/`.pxd` are build
outputs. Do not duplicate shared headers in a consumer to repair an include path.

Static checks resolve Python imports (including nested and literal absolute dynamic
imports) and Cython extern paths. They cannot prove computed dynamic imports,
Meson include search semantics, full ABI equivalence, or runtime correctness.
