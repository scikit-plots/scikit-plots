# Logical Tracker — `scikitplot.cexternals._annoy`

What the code **promises**. Not re-derivable from the tree; maintained by hand.

> **Family context:** read `FAMILY.md` first. This submodule is
> the shared C++ source three others depend on.

---

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| `Annoy` C-extension type | `src/annoymodule.cc` | The low-level, stable type. Python-level convenience belongs in `annoy`'s mixins, not here. |
| `annoylib.h` template API | `src/annoylib.h` | A C++ template header. Its signatures are mirrored by hand in `annoy/_annoy/annoylib.pxd.in`; nothing checks the mirror but `check_trackers.py`'s existence test. |
| `kissrandom.h` RNG | `src/kissrandom.h` | **Two consumers.** A stream change here silently alters both index construction and `scikitplot.random`. |
| `mman.h` portability shim | `src/mman.h` | POSIX/Windows mmap parity. `memmap` depends on the *behaviour*, not just the signatures. |
| standalone build | `meson.build` | Builds without any Python layer above it. Losing this makes the vendored source unusable on its own. |

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
