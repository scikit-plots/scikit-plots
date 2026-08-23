# Submodule Structure — `scikitplot.annoy`

> **Read `FAMILY.md` first** if the change touches `cexternals/_annoy/src/`.

---

## 1. Role

**the Cython + Python layer over the shared C++ source.**

## 2. Where does a new thing go?

| You are adding | Put it |
|---|---|
| A C++ header or template | `cexternals/_annoy/src/` — **and check all three consumers** |
| A Cython binding | the consuming submodule's `_<name>/` directory |
| Python-level convenience over the C type | `annoy`'s mixins, never `annoymodule.cc` |
| A dtype or specialization | `annoy`'s support matrix + a test |
| Anything importing a Python layer from `cexternals` | **nowhere** — it must stay standalone |

## 3. Structural debt and disposition

### 48 markdown files — the disposition

A completed campaign left ~35 per-run checkpoints in `_maintenance/`. They are
**history, not guidance**, and they bury the four files a fresh session actually
needs.

| Verdict | Files | Why |
|---|---|---|
| **ARCHIVE** → `_maintenance/history/` | `RUN0*`–`RUN25*`, `RUN_CY0*`, `RUN_F16C_FIX.md`, `RUN_SAVELOAD_SCOPEDERROR.md`, `RUN_CONTAINS_CONSISTENCY.md` (~35 files) | Per-run checkpoints for **finished** work. Evidence, not instruction. |
| **ARCHIVE** → `history/` | `BURNDOWN.md`, `CONTINUATION.md`, `CAMPAIGN_SUMMARY.md` | Point-in-time campaign state, now superseded by `STATE.json` |
| **FOLD** into `HISTORY.md` | `lessons.md` | Its lessons are durable; the file is a fragment |
| **FOLD** into `TRACKER_LOGICAL.md` | `SUPPORTED_DTYPES_AND_MULTIPRECISION.md`, `FLOAT80_DTYPE.md`, `FLOAT_TYPE_ANALYSIS_AND_DESIGN.md` | Three files describing one contract: the dtype support matrix |
| **KEEP** | `DEFERRED_FUTURE_WORK.md`, `CY015_ANNOYMODULE_DESIGN.md` | Open work with no home elsewhere |
| **DROP** | `ANNOY_REVIEW_PLAYBOOK.md`, `todo.md` | The playbook is superseded by the unified review kit; `todo.md` is a scratch file |

**48 markdown files become about 12**, without losing a decision — the run
checkpoints keep their evidence in `history/`.

The Corpus set solved the same problem the same way. A `_maintenance/` folder
that a fresh session cannot read in ten minutes has stopped being maintenance
material and become an archive.

### `_annoy/_backup/` duplicates `cexternals/_annoy/_backup/`

Two copies of the same vendored provenance. Keep at most one, in `cexternals`,
where the vendored source actually lives.

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
| Archive the ~35 RUN checkpoints | nothing | A fresh session can read `_maintenance/` in ten minutes |
| Wire native `save`/`load` into Corpus's `write_native`/`open_native` | **nothing — `ANNIndexArtifact` is built and tested** | Unlocks 3 of Annoy's 4 stated roles; the highest-value work available |
| Assert the `cdef extern` mirror against the headers | a parser or a compile-time static assert | Turns a hand-maintained mirror into a checked one |
| Fold the three dtype documents into one contract | nothing | One support matrix instead of three descriptions |
