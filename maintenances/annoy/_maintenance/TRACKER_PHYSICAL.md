# Physical Tracker — `scikitplot.annoy`

Re-derived from the live tree. **Do not hand-edit the numbers**:

```console
$ python scikitplot/annoy/_maintenance/check_trackers.py
```

Machine-readable mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files     17      source LOC    13535
test files       46      test LOC       6134
markdown files   48
```

## 2. Inventory

| Area | src | src LOC | tests | test LOC |
|---|---:|---:|---:|---:|
| `(root)` | 3 | 740 | 0 | 0 |
| `_annoy` | 6 | 9333 | 26 | 3954 |
| `_maintenance` | 1 | 51 | 0 | 0 |
| `_mixins` | 7 | 3411 | 2 | 160 |
| `tests` | 0 | 0 | 18 | 2020 |

## 3. Largest source files

| LOC | File |
|---:|---|
|   4208 | `_annoy/annoylib.pyx.in` |
|   4007 | `_annoy/_backup/annoylib_pyx.in` |
|    878 | `_mixins/_vectors.py` |
|    774 | `_mixins/_ndarray.py` |
|    604 | `_annoy/annoylib.pxd.in` |
|    527 | `_mixins/_meta.py` |
|    434 | `_annoy/annoymodule.cpp` |
|    426 | `_mixins/_pickle.py` |

## 4. Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| markdown files | 48 | the goal is < 15 after archival |
| checked-in generated sources | 1 (`annoymodule.cpp`) | > 1 |
| `cdef extern` sites against `cexternals` | 6 | any new one without an ABI note |

## 5. Known physical debt

### 1. 48 markdown files, ~35 of them `RUN*.md` from a **completed** campaign

Per-run checkpoints for finished work. They are history, not guidance. See §3.

### 2. `_annoy/_backup/` duplicates `cexternals/_annoy/_backup/`

Two copies of the same vendored provenance.

### 3. ABI declarations are unchecked against the headers they describe

`cdef extern` blocks are a hand-maintained mirror of another submodule's C++.

### 4. CY-015 (`annoymodule.cc` modernization) deferred

Recorded as deferred by the prior campaign; still open.
