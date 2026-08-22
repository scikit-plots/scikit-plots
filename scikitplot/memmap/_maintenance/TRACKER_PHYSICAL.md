# Physical Tracker — `scikitplot.memmap`

Re-derived from the live tree. **Do not hand-edit the numbers**:

```console
$ python scikitplot/memmap/_maintenance/check_trackers.py
```

Machine-readable mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files      4      source LOC     2140
test files        2      test LOC        710
markdown files    2
```

## 2. Inventory

| Area | src | src LOC | tests | test LOC |
|---|---:|---:|---:|---:|
| `(root)` | 1 | 40 | 0 | 0 |
| `_memmap` | 3 | 2100 | 2 | 710 |

## 3. Largest source files

| LOC | File |
|---:|---|
|   1622 | `_memmap/mem_map.pyx` |
|    258 | `_memmap/mem_map.pxd` |
|    220 | `_memmap/__init__.py` |
|     40 | `__init__.py` |

## 4. Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| `cdef extern` sites against `cexternals` | 2 | any new one without an ABI note |
| test : source LOC | see TRACKER.json | < 0.50 |

## 5. Known physical debt

### 1. Depends on `mman.h` by relative path; `'src'` commented out of its include dirs

See §2. The dependency is invisible to `meson.build` readers.

### 2. `MMAN.md` sits beside the source rather than in `_maintenance/`

Documentation is not co-located with the other maintenance material.
