# Annoy family boundary

The name “Annoy” appears in two independently owned compiled submodules.

| Domain | Owner | Compilation model | Public/runtime role |
|---|---|---|---|
| `scikitplot/cexternals/_annoy` | `maintenances/cexternals/_annoy` | hand-written C++/pybind11 extension + shared headers | native `Annoy` / `AnnoyIndex` and source headers |
| `scikitplot/annoy` | `maintenances/annoy` | Tempita-generated Cython `.pyx/.pxd` -> generated C++ -> extension, plus Python facade | private Cython `annoy._annoy.Index` and public high-level `annoy.Index` |

The Cython layer declares directly from three upstream headers:

```text
annoylib.pxd.in  -> ../../cexternals/_annoy/src/annoylib.h
annoylib.pxd.in  -> ../../cexternals/_annoy/src/kissrandom.h
annoylib.pyx.in  -> ../../cexternals/_annoy/src/annoy_type_support.h
```

The public high-level layer also imports/inherits the separately compiled
`cexternals._annoy.Annoy` type. A change to shared C++ declarations can therefore
break the Cython extension even when the native cexternals extension still
builds, and vice versa.

`memmap` and `random` are sibling direct-header consumers. They have their own
maintenance/skill owners and must not be edited merely to make the Annoy gate
green.
