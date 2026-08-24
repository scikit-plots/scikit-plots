# Submodule Structure — `scikitplot.memmap`

> **Read `FAMILY.md` first** if the change touches `cexternals/_annoy/src/`.

---

## 1. Role

**memory-mapping over the shared `mman.h` shim.**

## 2. Where does a new thing go?

| You are adding | Put it |
|---|---|
| A C++ header or template | `cexternals/_annoy/src/` — **and check all three consumers** |
| A Cython binding | the consuming submodule's `_<name>/` directory |
| Python-level convenience over the C type | `annoy`'s mixins, never `annoymodule.cc` |
| A dtype or specialization | `annoy`'s support matrix + a test |
| Anything importing a Python layer from `cexternals` | **nowhere** — it must stay standalone |

## 3. Structural debt and disposition

### Documentation placement

`_memmap/MMAN.md` sits beside the source rather than in `_maintenance/`. It is
good content in the wrong place: a fresh session reading `_maintenance/` will not
find it.

| Verdict | File |
|---|---|
| **MOVE** → `_maintenance/` | `_memmap/MMAN.md` |

## 4. Review checklist

```text
[ ] Does the change touch cexternals/_annoy/src/?   -> all 3 consumers rebuilt
[ ] Does it edit a generated .pyx/.pxd?             -> edit the template instead
[ ] Does it add a cdef extern?                      -> record the ABI it mirrors
[ ] Does cexternals import anything above it?       -> reject
[ ] Does every new public surface have a test?
[ ] python _maintenance/check_trackers.py           -> exit 0
[ ] Clean build from a fresh checkout               -> green
```

## 5. Directions, with prerequisites

| Direction | Needs first | Value |
|---|---|---|
| Move `MMAN.md` into `_maintenance/` | nothing | Discoverable by a fresh session |
| Add a resource-lifetime test suite | nothing | An unclosed mapping does not raise where it happened |
| Add `src/` to `include_directories` | coordinated with `cexternals` | Removes the relative-path coupling |
