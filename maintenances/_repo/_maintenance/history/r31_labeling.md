<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 31 - PR/issue labeling: one policy, generated configs, semantic checks

### Context (facts, grounded in the pinned action sources)
- `actions/labeler@bf12e9b` (v7): a label's bare match list is `any`; inside one
  `changed-files` list every option is a separate check, OR-ed
  (`src/changedFiles.ts::checkAnyChangedFiles`). Globs use minimatch, `dot: true`.
- `github/issue-labeler@ccca1e1` (v3.5): each config value is a JS regex,
  case-sensitive unless written `/re/flags`; several entries for one label are
  AND-ed; title text is `title + "\n\n"`; works for issues and PRs.
- `gerrymanoim/pr-prefix-labeler@c806232`: `title.startsWith(key)`, first match
  only, no word boundary, input `configuration-path` (read via `with:`), throws
  on non-PR events.
- GitHub label names: at most 50 characters (github/docs#32156).

### Root causes
1. Module labels hand-maintained beside the towncrier owner policy: 20 of 45
   packages unlabeled (cleanprompt, corpus, mcp, mlflow, rank_bm25, ...), no
   libs/ or tools/ labels, `scikitplot/visualkeras.py` glob dead (now a package).
2. `labeler.yml` semantics: `documentation` OR-s an all-negated
   `all-globs-to-any-file` -> nearly every PR is labelled `documentation`;
   `lang:cython` matches `*.pyi` stubs; `build:templates` matches `*.tp`
   (none) while templates are `*.in`; `release` names workflows that do not
   exist and a `maintainers*` file glob that misses the directory; tests label
   misses every `*/tests/` except `scikitplot/tests`.
3. Three title vocabularies (issue-labeler `defect`/`Build issues`/...,
   prefix labeler `00 - Bug`/`36 - Build`/..., dead `label_title_regex.py`
   `Documentation`/`Build / CI`): one prefix gives two labels; substring regexes
   without boundaries (`DOC` in `DOCKER`); `[Bug]:` template titles get no label.
4. `pr_labeler.yml`: also triggered by `issues: opened` (PR checklist posted on
   every new issue; PR-only steps fail silently; duplicates issue_labeler.yml);
   concurrency keyed by head branch (fork PRs named `main` cancel each other);
   draft comment on every push; prefix labeler config passed via `env:`;
   checklist names labels that do not exist (`whatsnew-needed`, `Extra CI`);
   title edits never re-label (no `edited`).
5. No generator, no CI gate -> drift (towncrier check also failing:
   `scikitplot.levenshtein` missing).

### Plan
- [x] `tools/maint_tools/generate_pr_labeler.py` (check/sync/explain/list),
      policy `[tool.scikitplot.maintenance.labeler]`, templates `*.yml.in`
- [x] module labels from towncrier owners + satellites + distribution ownership
- [x] title labels from one prefix/keyword registry (issue-labeler regexes)
- [x] curated `labeler.yml`: schema, glob subset, negation, unmatched checks; fix rules
- [x] workflow fixes; drop pr-prefix-labeler + dead script; issue templates labels;
      `issue_labeler.yml` without checkout; changelog comment names the real labels
- [x] CI workflow `pr_check_labeler.yml`; 41 tests incl. minimatch cross-check;
      towncrier sync (104 sections, test expectation 103 -> 104)

### Results Review - 2026-10-11
- `generate_pr_labeler.py check`: PASS (curated 12, module 78 (was 33), title 19);
  warnings only for root files absent from the uploaded archive (verify on a
  real checkout with `check --strict`).
- `test_generate_pr_labeler.py`: 41 tests OK on Python 3.12 and 3.8 (minimatch
  cross-check runs when Node + minimatch are installed: OK with minimatch 10).
- Earlier cross-checks: 356 globs x 6069 paths equal to minimatch; 52 titles
  equal between Node and Python regex engines.
- `generate_towncrier_sections.py check`: PASS (104); its tests: 7 OK.
- All workflow YAML parses; test module names unique; path lengths 0 failing.
- Found while writing tests: `label_matches` stopped at the first key of an
  `any:` object holding several keys (elif chain); labeler v7 checks each key.
  Fixed; `test_one_any_object_with_several_keys_checks_each_key`.
- Removed: `.github/pr-prefix-labeler.yml`, `.github/scripts/label_title_regex.py`
  (listed in `REMOVED_FILES.txt`).
