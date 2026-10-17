<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 29 - page feedback on any site, with or without the AI assistant

### Context
- Goal: https://scikit-plots.github.io/dev/ feedback works like
  https://scikit-plots-learn.readthedocs.io/en/latest/; any site can use
  `_sphinx_feedback`; add `_sphinx_feedback/_example_conf.py`; check
  `docs/source/conf.py`.
- Root cause (reproduced): the library site sets `feedback_site_id = "scikit-plots"`.
  (1) Build: the one packaged snapshot `_static/page-feedback-aggregate.json`
  carries `site_id` `scikit-plots-learn` -> `load_aggregate` raises
  "feedback aggregate site_id does not match feedback_site_id" at
  `config-inited`. (2) Submit: proxy default `FEEDBACK_ALLOWED_SITE_IDS` was
  `scikit-plots-learn` -> `422 site_not_allowed`. CORS was not a cause: both
  origins are in `_DEFAULT_ALLOWED_ORIGINS`. The JS and the Python package
  import nothing from the AI assistant.
- Found on the way: `SQLiteFeedbackStore` never closed connections
  (`with conn` only ends the transaction); the suite failed at random tests
  under `-W error` on Python 3.13.

### Implementation Steps
- [x] 1. `_config.load_aggregate(source_root=)` + `_resolve_aggregate_path`: no
      leading `/` = file inside the site's `confdir`, confined (`..`, `\`, `:`,
      `?`, `#`, empty segments, symlink escape refused); `/` form unchanged
- [x] 2. `_sphinx._configure` passes `app.confdir`
- [x] 3. Proxy default allowlist `scikit-plots-learn,scikit-plots`
- [x] 4. `docs/source/conf.py`: `feedback_aggregate_file = "_page_feedback/aggregate.json"`;
      new complete V3 snapshot (`site_id` `scikit-plots`, no rows); corrected the
      AI Learn comment (this site does not enable AI Learn)
- [x] 5. `_example_conf.py` (every `feedback_*` value; AI assistant, two-site,
      service variants)
- [x] 6. SQLite `closing()` in store and tests; mirror resynced; version 0.7.0
- [x] 7. Tests: `test_feedback_site_builds.py` (10 real builds incl. with the AI
      assistant), loader unit test, example-imports-nothing, integration test
      parses allowlist and checks the library conf
- [x] 8. README, assistant HISTORY, fragment `864.fix.rst`, lessons 64-65

### Acceptance Criteria
- [x] Library conf values validate and its snapshot loads (`scikit-plots`, complete)
- [x] Proxy accepts `scikit-plots-learn` and `scikit-plots`, refuses others
- [x] Any `site_id` builds with no snapshot; another site's packaged snapshot fails closed
- [x] Feedback suite green under `-W error`; proxy + integration suites green

### Risks & Mitigation
- Live HF Space keeps the old default until redeployed -> set
  `FEEDBACK_ALLOWED_SITE_IDS=scikit-plots-learn,scikit-plots` on the Space.
- `complete: true` for `scikit-plots` rests on the code default never having
  allowed that site; if the Space ever set the variable to include it,
  regenerate the snapshot from the reviewed events before publishing.

### Results Review - 2026-10-10
- Files: 17 (4 new: `_example_conf.py`, `tests/test_feedback_site_builds.py`,
  `docs/source/_page_feedback/aggregate.json`, fragment `864.fix.rst`).
- Evidence (CPython 3.13, Sphinx 9.1.0):
  - `pytest scikitplot/_externals/_sphinx_ext/_sphinx_feedback --confcutdir ... -W error`
    -> 314 passed (before: 297 passed + random ResourceWarning errors).
  - `pytest .../_sphinx_ai_assistant/tests/_hf_spaces_proxy .../tests/_integration -n 8`
    -> 1361 passed, 2 skipped (the two learn-checkout-only conf tests).
  - Proxy imported with no `FEEDBACK_ALLOWED_SITE_IDS` set: allowlist
    `('scikit-plots-learn', 'scikit-plots')`; both accepted, `someone-else`
    refused `site_not_allowed`.
  - Library `docs/source/conf.py` feedback values validate; snapshot loads
    `complete: True` for `scikit-plots`.
  - Python 3.8: changed modules compile; loader accepts the source form and
    refuses `../`, `_f/../../`, `C:` selectors.
  - `ruff check` clean on changed production modules and new tests; new paths
    <= 77 characters (budget 131).
- Deviation: SQLite connection leak fixed in scope (it made the suite
  nondeterministic under `-W error`).
- Not done here: redeploying the HF Space (or setting its variable).
