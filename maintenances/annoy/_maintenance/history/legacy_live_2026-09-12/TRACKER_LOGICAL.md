# Logical Tracker — `scikitplot.annoy`

What the code **promises**. Not re-derivable from the tree; maintained by hand.

> **Family context:** read `FAMILY.md` first. This submodule is
> a consumer of `cexternals/_annoy/src/`.

---

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| tempita templates | `_annoy/annoylib.pxd.in`, `.pyx.in` | **The real sources.** `annoylib.pyx`/`.pxd` are generated at build time; editing a generated file produces a build that works until it is built clean. |
| `cdef extern` ABI mirror | `_annoy/annoylib.pxd.in` | Six sites declaring C++ signatures owned by `cexternals`. A hand-maintained mirror of another submodule's header. |
| `supported_dtypes()` | Python layer | The dtype/specialization surface. float80 wired; 256/512 gated behind `ANNOY_ENABLE_MULTIPRECISION`. |
| Python mixins | Python layer | Extend the low-level `Annoy` type. This is where higher-level behaviour belongs — not in `annoymodule.cc`. |
| `VectorIndexBackend` conformance | (Corpus-facing) | Corpus declares eleven capability members. Annoy's backend must answer all truthfully — including `supports_persistence`, which Corpus currently declares `True`. |

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
