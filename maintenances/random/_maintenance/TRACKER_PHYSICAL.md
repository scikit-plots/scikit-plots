# Physical Tracker — `scikitplot.random`

Re-derived from the live tree. **Do not hand-edit the numbers**:

```console
$ python scikitplot/random/_maintenance/check_trackers.py
```

Machine-readable mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files      5      source LOC     4555
test files        4      test LOC       1435
markdown files    4
```

## 2. Inventory

| Area | src | src LOC | tests | test LOC |
|---|---:|---:|---:|---:|
| `(root)` | 1 | 45 | 0 | 0 |
| `_kiss` | 4 | 4510 | 4 | 1435 |

## 3. Largest source files

| LOC | File |
|---:|---|
|   3985 | `_kiss/kiss_random.pyx` |
|    237 | `_kiss/kiss_random.pxi` |
|    198 | `_kiss/kiss_random.pxd` |
|     90 | `_kiss/__init__.py` |
|     45 | `__init__.py` |

## 4. Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| `cdef extern` sites against `cexternals` | 1 | any new one without an ABI note |
| documented stream-stability guarantee | absent | must exist before 1.0 |

## 5. Known physical debt

### 1. Shares `kissrandom.h` with `annoy` by relative path

A change to the header is a change to *both* index construction and user-facing randomness.

### 2. Three markdown files beside the source

`KISSRANDOM.md`, `KISSRANDOM_NUMPY_COMPATIBLE_FINAL.md`, `README_KISSRANDOM.md` — overlapping, none in `_maintenance/`.

### 3. No recorded stream-stability guarantee

Whether a seeded stream is stable across releases is undocumented.
