# Hub inventory — one tracker for the whole `learn/` hub

`hub_inventory.py` + `hub_inventory.json` are the hub-level generalisation of a
course's `_discovered.json`: a single source of truth for **every** course's
structure and state, so gaps and drift are found by a scan instead of by eye.

## Where it sits — a tracker, not a generator

This is the **top** layer of a three-layer pipeline, and it only **reports** — it
never authors or generates lesson content:

```
you author:    *_inventory.tsv  +  *_content.py           (per course, by hand)
course build:  build_<course>.py  →  lesson .rst           (per course, automated)
hub track:     hub_inventory.py   →  dashboard / gaps / diff / mismatch   (read-only)
```

- Your `*_inventory.tsv` and the content behind it stay **hand-maintained** —
  nothing here changes that workflow.
- Lesson `.rst` is produced by each course's own `build_<course>.py` from its
  `*_content.py`, exactly as before. `hub_inventory.py` **does not write lesson
  `.rst`** and never edits your content.
- Its job is the **cross-check**: it counts the authored tsv rows *and* the built
  `.rst`, flags where they've drifted apart (`⚠ tsv≠rst`), and lists which roadmap
  sections are still empty.

The one thing it can write is `--scaffold`, and that is a *gap reminder*, not
lesson content — see Guarantees below.

## What it answers
- **Dashboard** — course → section → count for the whole hub, in one view.
- **Gaps** — roadmap sections that exist in the plan but have no content yet
  (e.g. Deep Learning courses 2–5), so "what to build next" is explicit.
- **Mismatches** — where a course's authored `*_inventory.tsv` and its built
  `.rst` tree disagree (a lost, extra, or unbuilt lesson).
- **Diff** — what changed since the previous run (per-course lesson/page deltas).

## The two halves of `hub_inventory.json`
- **`roadmap`** — HAND-AUTHORED, the only part you edit. Per course: a `kind`
  and, for `sectioned` courses, the list of `sections`. Add a course or a
  planned-but-empty section here and it immediately shows up as a gap.
- **`actual` / `gaps` / `diff` / `generated`** — GENERATED. Overwritten on every
  scan. The scanner never touches `roadmap`.

### Course kinds
| kind | how lessons live | counted from |
|------|------------------|--------------|
| `flat` | `NN-title.rst` at the course root | tsv rows ↔ root `.rst` (minus `index.rst`) |
| `sectioned` | one folder per section | per section: tsv `section`-column rows ↔ section-folder `.rst` |
| `mirror` | verbatim page mirror | `pages` in `<course>/_sources_cache/_discovered.json` |

A `sectioned` section maps to a tsv `section` key and/or a `dir`. `"whole_course":
true` means the section *is* the whole flat course (used for Deep Learning
section 1, whose 17 flat files are course 1 of a 5-course roadmap). A section with
none of these is roadmap-only → a gap.

## Usage
```
python3 hub_inventory.py              # scan, rewrite the generated blocks, print the dashboard
python3 hub_inventory.py --check      # exit non-zero if any gap or mismatch (CI gate)
python3 hub_inventory.py --scaffold   # also stub each gap under <course>/_roadmap/ (safe, idempotent)
python3 hub_inventory.py --json-only  # update JSON without printing
```

## Guarantees
- **Read-only by default.** The only thing `--scaffold` writes is a **gap-reminder
  placeholder** — a titled "content pending" marker, *not* a lesson skeleton to
  flesh out — under a course's `_roadmap/` staging folder. It never overwrites,
  never touches the generated lesson tree (which each `build_*.py` clears and
  rebuilds), and you delete the stub once the real section exists.
- **Deterministic & idempotent** — same tree → same JSON apart from the timestamp.
- **No dependencies** — stdlib only.
- **Run history** — each scan appends a one-line summary to
  `hub_inventory_log.jsonl` and snapshots the prior file to
  `hub_inventory.prev.json`, so old runs correspond.

## Adding a new course
1. Drop its folder in the hub with a `*_inventory.tsv` (and the usual
   `*_content.py` / `build_*.py` / `_buildcheck.py`).
2. Add one `roadmap` entry (pick the `kind`; list `sections` if sectioned).
3. Run `python3 hub_inventory.py` — it appears in the dashboard, with any empty
   sections flagged as gaps.

## See Also
- https://insightful-data-lab.com/
- https://labs.bilimedtech.com/index.html
- https://lpn-doc-sphinx-primer-devel.readthedocs.io/index.html
