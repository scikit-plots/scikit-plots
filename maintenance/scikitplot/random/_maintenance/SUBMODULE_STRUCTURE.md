# Submodule Structure — `scikitplot.random`

> **Read `FAMILY.md` first** if the change touches `cexternals/_annoy/src/`.

---

## 1. Role

**KISS RNG over the shared `kissrandom.h`.**

## 2. Where does a new thing go?

| You are adding | Put it |
|---|---|
| A C++ header or template | `cexternals/_annoy/src/` — **and check all three consumers** |
| A Cython binding | the consuming submodule's `_<name>/` directory |
| Python-level convenience over the C type | `annoy`'s mixins, never `annoymodule.cc` |
| A dtype or specialization | `annoy`'s support matrix + a test |
| Anything importing a Python layer from `cexternals` | **nowhere** — it must stay standalone |

## 3. Structural debt and disposition

### Three overlapping markdown files

| Verdict | File | Why |
|---|---|---|
| **FOLD** → `_maintenance/TRACKER_LOGICAL.md` | `_kiss/KISSRANDOM_NUMPY_COMPATIBLE_FINAL.md` | Describes the NumPy-compatible contract — that is a contract, not a note |
| **MOVE** → `_maintenance/` | `_kiss/KISSRANDOM.md` | Reference material |
| **FOLD** → `README` | `_kiss/README_KISSRANDOM.md` | Overlaps the above two |

### The missing contract

**No file records whether a seeded stream is stable across releases.** For an
RNG that is the primary contract: a stream that changes is not a crash and not a
wrong answer, it is a silently different experiment, and nothing downstream can
detect it.

Recorded as a finding for run **A11**.

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
| Document the stream-stability guarantee | a decision, not code | The primary RNG contract, currently absent |
| Add a golden-stream regression test | the decision above | Makes stability checkable rather than assumed |
| Consolidate the three markdown files | nothing | One contract, one file |
