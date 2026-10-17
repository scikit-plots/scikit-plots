# `_repo` maintenance: active tasks

Repository-wide work that no single `scikitplot/<submodule>` owns: CI
workflows, the `libs/` partial distributions, labeling, release tooling, and
the record planes themselves. Completed tasks are archived verbatim under
[`../_maintenance/history/`](../_maintenance/history/).

## Task: increment 4R - records by plane (tasks/ split into maintenances/, skills/, docs/tasks/)

### Context
- Goal: `tasks/todo.md` (197 KB, 2989 lines, 31 sections) and
  `tasks/lessons.md` (108 KB, 123 rules) are read whole by every session;
  most of both is history or concerns one plane. Move each record to the
  plane that owns it, keep active work small, and keep every rule
  reachable from one index.
- Precedent in the tree: `maintenances/<sub>/` (state and evidence),
  `skills/<sub>/` (fresh-session routing), `maintenances/<sub>/todo/`
  (`_sphinx_ext` family; its checker already retires a `tasks/` folder).
- Decisions (maintainer, 2026-10-16): lessons to `skills/_engineering/` by
  category; `tasks/` keeps only a router README; completed rounds archived
  per plane; done now, before 4c.

### Design (as built, after the independent review)
- Planes and owners (a record goes to the area owning most of what it
  changed; every section has an explicit destination in the migration map):
  - `maintenances/corpus/todo/todo.md` - round 33 (AI plane, lead area
    corpus): the active Run 4, Runs 5 and 6 scope (copied from the review
    bundle, which is outside the repository), open decisions D1, D4-D8;
    Runs 1, 2, 3 and 4-A archived to `maintenances/corpus/_maintenance/history/`.
  - `maintenances/cleanprompt/_maintenance/history/` - cleanprompt rounds 25, 26.
  - `maintenances/_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/`
    - rounds 29, 29b, 30 (page feedback); the family's
    `_maintenance/history/` - round 9 (extension stack). First mapped to
    `docs/tasks/`; review 7 showed they change the extensions, not the docs.
  - `maintenances/_repo/` - CI, `libs/`, labeling: partial distributions,
    rounds 2-8, 31, 32, the old `tasks/removed-files.txt`.
  - `docs/tasks/` - the documentation site as a whole (active tasks, pointers).
  - `skills/_engineering/` - the rules in 12 category files, numbers
    unchanged; generated `INDEX.md`; `PATTERNS.md`.
  - `tasks/README.md` - router only.
- Two commits (Rule 124): `4R-1` is the pure move, proved lossless (the
  manifest's parts rebuild the three files byte for byte); `4R-2` routes,
  edits and checks.
- Checker `_maintenance/tools/check_records.py`, run by
  `.github/workflows/pr_check_records.yml`: router only in `tasks/`; rule
  headings strict (malformed ones fail), numbered once, each category in one
  file; `INDEX.md` and `HISTORY_INDEX.md` (every archive with the areas its
  text names) current; archived parts unchanged; links resolve inside the
  repository (fences skipped); active `todo.md` files within budget.

### Steps
- [x] migration script (outside the repo), explicit section and category maps
- [x] `4R-1`: generated history parts, rules files, `PATTERNS.md`, manifest,
      the corpus active file; lossless proof
- [x] router, `SKILL.md`, `docs/tasks/`, this area's `MAINTAINING.md`;
      pointers in `docs/maintenances/MAINTAINING.md`, `docs/skills/README.md`,
      `maintenances/__init__.py`, `maintenances/corpus/MAINTAINING.md`,
      `skills/corpus/SKILL.md`, `skills/cleanprompt/SKILL.md`, cleanprompt
      `RESUME.md` and `FRESH_CHAT_HANDOFF.md`, `libs/_tools/registry.py`
- [x] six rules repointed from `tasks/todo.md` to the archives (29, 31, 35,
      66, 69, 100); Rule 124
- [x] checker, both indexes, mutation tests (38), workflow
- [x] labeler `sync --apply` (pre-existing mlflow drift since the uploaded tree)
- [x] independent review (13 findings), fixes
- [x] re-review of the fixes (6 findings + 2 carried over), fixes; Rule 125
- [x] harness, delivery

### Acceptance criteria
- `4R-1` rebuilds the three source files byte for byte.
- `tasks/` holds only `README.md`; rules 1-125 appear exactly once.
- The corpus, mcp and cleanprompt checkers and the `_sphinx_ext` family gate
  report the same as before; the labeler check passes.
- No file under `scikitplot/` changes.

### Risks
- An agent following the old convention writes to `tasks/todo.md` again ->
  the router translates the old names; the checker fails in CI.
- Archived history edited by accident -> archived parts are hash-checked.
- This checkout's `.git/info/exclude` ignores `README.md`: the new READMEs
  are added with `git add -f` (review 1).

### Results Review - 4R (2026-10-17)
- Commits: `4R-1` (8f64311) pure move; `4R-2` routing, edits, checker.
- Lossless: `check_records.py --prove 8f64311` rebuilds `tasks/todo.md`
  (31 parts), `tasks/lessons.md` (127 parts) and `tasks/removed-files.txt`
  byte for byte from 471e46c; the reviewer's own rebuild agrees.
- Reviews: review 1 found 13 (2 High: the new READMEs were ignored by this
  checkout's `.git/info/exclude`; the workflow's test step needed Sphinx),
  re-review 6 more (a regression: archive tests on absolute paths, Rule 125;
  an index that missed nested areas and named false ones; rule headings and
  open fences still unseen; link parser corners; nested link coverage; a
  proof outside the repository). All fixed with a test each.
- Checker tests: 52 passed on Python 3.12 (full environment), 3.12 with only
  pytest (as in CI) and 3.9.
- Unchanged: corpus, mcp and cleanprompt `check_trackers.py --json` output,
  byte for byte, at 471e46c and after; `_sphinx_ext` `check_all.py` GREEN
  (2 subsystems) before and after; no file under `scikitplot/` changed.
- Labeler: regenerated twice (the pre-existing mlflow drift; then the new
  `_sphinx_feedback` maintenance folder); PASS. Path lengths: 0 failing.
  zizmor on the new workflow: no findings.
- Harness on `snap4r` (identical to `4R-2`): 203 passed, 4 failed, 2
  skipped; the 4 are the corpus legs (4986 passed each) with the same 14 DNS
  failures as 3b.
- Accepted and documented: active files are not hash-checked (a deleted
  active section passes); an archived part copied into a second archive
  passes; `todo/lessons.md` files have no budget.
