# The annoy family — one C++ source, four submodules

**Identical in all four `_maintenance/` folders.** If you edit it, edit all four.

---

## 1. The dependency graph

```text
scikitplot/cexternals/_annoy/src/        <- the single shared C++ source
    annoylib.h            ──► annoy/_annoy/annoylib.pxd.in      (4 extern sites)
    kissrandom.h          ──► annoy/_annoy/annoylib.pxd.in
                          ──► random/_kiss/kiss_random.pxd
    annoy_type_support.h  ──► annoy/_annoy/annoylib.pyx.in
    mman.h                ──► memmap/_memmap/mem_map.pxd
                          ──► memmap/_memmap/mem_map.pyx
```

```text
cexternals/_annoy        standalone C++; imports nothing above it
        │
        ├──► annoy       Cython bindings + Python mixins   (annoylib.h, kissrandom.h,
        │                                                   annoy_type_support.h)
        ├──► memmap      mmap portability                  (mman.h)
        └──► random      KISS RNG                          (kissrandom.h)
```

`kissrandom.h` has **two** consumers. A change to it affects index construction
*and* user-facing randomness at once.

---

## 2. How the coupling is expressed

**How the coupling is expressed, and why it matters.**  Consumers reach the
headers by a *relative path written into Cython source*:

```cython
cdef extern from "../../cexternals/_annoy/src/kissrandom.h" namespace "Annoy" nogil:
```

Not through `include_directories`.  In `memmap/_memmap/meson.build` and
`random/_kiss/meson.build` the `'src'` entry is present but **commented out**,
so the relative path is the only mechanism.

Two consequences worth stating plainly:

* moving or renaming anything under `cexternals/_annoy/src/` breaks three
  submodules at *Cython compile time*, with an error that names a path rather
  than a contract;
* the coupling is invisible to any tool that reads `meson.build` alone.

This is the family's defining structural fact.  Every maintenance decision below
follows from it.

---

## 3. The family contract

> `cexternals/_annoy` is **upstream**. It owns C++ and knows nothing about
> Python. The other three own Python-level behaviour and must not fork,
> duplicate or patch the C++ they consume.

Concretely:

| Rule | Why |
|---|---|
| `cexternals` imports nothing from `annoy`, `memmap`, `random` | It must stay standalone; that is its stated design |
| The three consumers do not vendor their own copy of a shared header | Two copies of `mman.h` is two behaviours |
| A `cdef extern` block is a **mirror** of a header, and must be re-checked when that header changes | Nothing checks it automatically |
| A change under `src/` is a three-submodule change | Whether or not it is treated as one |
| Generated `.pyx`/`.pxd` are outputs; the `.in` templates are the sources | Editing the output produces a build that works until it is built clean |

---

## 4. What each submodule owns

| Submodule | Owns | Depends on |
|---|---|---|
| `cexternals/_annoy` | the C++ headers, `annoymodule.cc`, the `Annoy` C-extension type | nothing |
| `annoy` | Cython bindings from tempita templates, Python mixins, `supported_dtypes()` | `annoylib.h`, `kissrandom.h`, `annoy_type_support.h` |
| `memmap` | `mem_map.pyx` and its typed surface | `mman.h` |
| `random` | KISS RNG, NumPy-compatible surface | `kissrandom.h` |

---

## 5. Where the rest of the project meets this family

The Corpus campaign completed (review R00–R16, implementation IMPL-01–18) and
changed what Annoy is expected to provide. Two contracts now exist that did not
when the ANNOY review guide was written:

| Corpus contract | What it means here |
|---|---|
| `VectorIndexBackend` (renamed from `ANNBackend`) | Declares eleven capability members including `metric`, `score_semantics`, `supports_persistence`, `memory_profile`. Annoy's backend must answer all of them truthfully. |
| `ANNIndexArtifact` + ordinal→doc_id sidecar | **Built and tested.** Corpus can now persist an index directory with a versioned sidecar and an embedding manifest. |

Review run R06 established that Annoy could fill only **one** of its four stated
roles — static read-heavy ANN, mmap-backed serving, multi-process sharing,
semantic seed generation — because three required an on-disk index Corpus had no
way to describe. **The artifact now exists.** Wiring Annoy's native `save`/`load`
into `write_native`/`open_native` is therefore a contained change rather than an
architectural one, and it is the highest-value work this family has available.

That is run **A15**'s content, and it is now concrete rather than speculative.

One constraint runs the other way: `supports_persistence` is currently declared
`True` by Corpus on the strength of the artifact format. If Annoy's native
persistence turns out not to round-trip, that declaration is wrong and must
change — a declaration that outruns the implementation is exactly the defect
class this project has spent two campaigns removing.

---

## 6. The review sequence

Guide: `ANNOY/ANNOY_UNIFIED_DEEP_REVIEW_EVOLUTION_GUIDE.md` in the review kit,
runs **A00–A21**.

```text
A00  clean-room source/build baseline      <- NEXT
A01  public API and low/high-level ownership
A02  explicit lifecycle/state machine
A03  type and specialization support matrix
A04  numeric boundaries
A05  sparse ID and allocation behaviour
A06  error channel and ownership
A07  file descriptors, mmap and persistence      <- memmap's stake
A08  on-disk build
A09  pickle/state/bundle contracts
A10  generated source and ABI drift              <- the template/ABI question
A11  RNG and reproducibility                     <- random's stake
A12  concurrency and free-threaded semantics
A13  CPU/platform portability
A14  WASM/restricted platform profile
A15  Corpus VectorIndexBackend integration       <- now concrete; see §5
A16  immutable generations
A17  delta/freshness strategy
A18  memory architecture
A19  quality/performance benchmark
A20  security and hostile persistence
A21  public docs/stubs/maintenance truth
```

**Do not begin implementation before the runs close.** The Corpus campaign's
value came from 55 findings and **23 disproofs** established before any code,
which is why 18 implementation increments ran without a red suite.

Three runs have a stake outside `annoy` itself and should be read by whoever
maintains the consumer: **A07** (memmap), **A10** (all three), **A11** (random).

---

## 7. Project sequence

```text
Corpus   review COMPLETE (R00–R16) | implementation COMPLETE (IMPL-01–18)
MCP      maintenance set ready; run M00 pending
ANNOY    maintenance set ready; run A00 next    <- this family
CLI      after ANNOY
         ↓
   cross-module consolidation → final implementation DAG → code
```
