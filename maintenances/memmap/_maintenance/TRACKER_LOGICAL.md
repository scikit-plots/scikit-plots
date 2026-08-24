# Logical Tracker — `scikitplot.memmap`

What the code **promises**. Not re-derivable from the tree; maintained by hand.

> **Family context:** read `FAMILY.md` first. This submodule is
> a consumer of `cexternals/_annoy/src/`.

---

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| `mem_map.pyx` / `.pxd` | `_memmap/` | The mmap binding over `mman.h`. Resource lifetime is the contract: an unclosed mapping does not raise where it happened. |
| `mem_map.pyi` | `_memmap/` | The typed surface. Must track the `.pyx`; nothing checks it. |
| POSIX/Windows parity | (inherited) | Depends on `mman.h`'s *behaviour*, not only its signatures — a portability shim can compile and still differ. |

---

## 2. Cross-cutting invariants

| Invariant | Enforced by |
|---|---|
| Every `cdef extern` names a header that exists in `cexternals/_annoy/src/` | `check_trackers.py` |
| `cexternals` imports nothing from `annoy`/`memmap`/`random` | `check_trackers.py` |
| Generated `.pyx`/`.pxd` are outputs, never edited directly | **nothing — convention only** |
| The `cdef extern` signatures match the C++ they mirror | **nothing — existence is checked, not shape** |

The last two are the family's real exposure. Existence is cheap to check and
shape is not, so a signature drift compiles into a wrong call rather than an
error.

---

## 3. Known logical debt

See `SUBMODULE_STRUCTURE.md` §3 for the full disposition and `FAMILY.md` §5 for
what the completed Corpus campaign now expects of this family.
