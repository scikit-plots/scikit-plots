# Submodule Structure — `scikitplot.cexternals._annoy`

> **Read `FAMILY.md` first** if the change touches `cexternals/_annoy/src/`.

---

## 1. Role

**the shared C++ source of truth.**

## 2. Where does a new thing go?

| You are adding | Put it |
|---|---|
| A C++ header or template | `cexternals/_annoy/src/` — **and check all three consumers** |
| A Cython binding | the consuming submodule's `_<name>/` directory |
| Python-level convenience over the C type | `annoy`'s mixins, never `annoymodule.cc` |
| A dtype or specialization | `annoy`'s support matrix + a test |
| Anything importing a Python layer from `cexternals` | **nowhere** — it must stay standalone |

## 3. Structural debt and disposition

### `_backup/` — provenance, not source

`_backup/` holds a complete vendored upstream tree: `CMakeLists.txt`, `debian/`,
a rockspec, Go and Lua bindings, a second `annoy/` Python package, and
`fastannoy/annoymodule.cpp`. **None of it is built.**

| Verdict | Files |
|---|---|
| **ARCHIVE** → `_maintenance/history/` or out of the tree | the whole `_backup/` directory |

It is upstream provenance and should be reachable, but it should not sit inside
an installed Python package where it doubles the apparent source size and
duplicates `annoy/_annoy/_backup/`.

### `annoymodule.cu`

Present, unbuilt, unreferenced by `meson.build`, untested. Either wire it with a
capability declaration or archive it. An unbuilt source file in `src/` is
indistinguishable from a forgotten one.

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
| Move `_backup/` out of the installed package | nothing | Halves apparent source size; removes a duplicate |
| Add `src/` to consumers' `include_directories` and drop the relative paths | coordinated 3-submodule change | Makes the coupling visible to `meson.build` readers |
| Decide `annoymodule.cu`: wire or archive | a CUDA capability declaration | Removes an unbuilt file from `src/` |
