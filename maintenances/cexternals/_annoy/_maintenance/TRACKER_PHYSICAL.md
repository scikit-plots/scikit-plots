# Physical Tracker — `scikitplot.cexternals._annoy`

Re-derived from the live tree. **Do not hand-edit the numbers**:

```console
$ python scikitplot/cexternals/_maintenance/check_trackers.py
```

Machine-readable mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files     29      source LOC    23846
test files       14      test LOC       1001
markdown files    5
```

## 2. Inventory

| Area | src | src LOC | tests | test LOC |
|---|---:|---:|---:|---:|
| `(root)` | 1 | 33 | 0 | 0 |
| `_annoy` | 28 | 23813 | 14 | 1001 |

## 3. Largest source files

| LOC | File |
|---:|---|
|  11520 | `_annoy/src/annoymodule.cc` |
|   5132 | `_annoy/src/annoylib.h` |
|   1597 | `_annoy/_backup/src/annoylib.h` |
|   1361 | `_annoy/_plotting.py` |
|    685 | `_annoy/_backup/src/annoymodule.cc` |
|    434 | `_annoy/_backup/fastannoy/annoymodule.cpp` |
|    346 | `_annoy/src/annoyluamodule.cc` |
|    321 | `_annoy/_backup/src/annoyluamodule.cc` |

## 4. Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| files under `src/` | 13 | any addition without a downstream check |
| downstream consumers | 3 | a fourth without a documented contract |
| unbuilt source files in `src/` | 1 (`.cu`) | > 1 |

## 5. Known physical debt

### 1. `_backup/` holds a full vendored upstream tree

CMakeLists, debian/, rockspec, Go and Lua bindings, a second `annoy/` package. None is built. It is provenance, not source — it belongs in `history/` or out of the tree.

### 2. Downstream coupling is by relative path, not include dir

See §2. `'src'` is commented out of two consumers' include dirs.

### 3. `annoymodule.cu` (CUDA) is present but unbuilt

Neither referenced by `meson.build` nor tested. Either wire it or archive it.
