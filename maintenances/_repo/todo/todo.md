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

## Task: TP-1 - test-performance evidence (recorder, analyzer, bundle, CI gate)

### Context
- Goal: for every submodule, know which tests, fixtures, collections and
  functions cost the time and memory, locally and on every CI shard, from
  measurements only; and replace the manual steps (`--setup-show`, editing a
  fixture to time its calls, `python -m cProfile` + `pstats` + `zip`) with one
  recorder, one analyzer and one self-describing archive any reader (a person,
  or an AI session with no chat history) can analyse.
- Evidence on the uploaded tree (2026-10-23):
  - CI, single job, 5 October: 1 482 s of tests, of which 403 s are the 531
    full collections at module boundaries (`scikitplot/conftest.py`,
    `_test_gc_policy`); `_sphinx_ai_assistant` is 484 s of the 1 367 s in
    `.github/ci/test_durations.json`.
  - Your `scikitplot/_externals/` profile (cProfile, 6 540 tests, 2 493 s):
    filesystem 35 %, subprocess waits 24 %, JSON 10 %; `_sphinx_ai_assistant`
    is 1 661 s of the 1 974 s of test calls.
  - Several security modules are executed more than once in one process:
    tests put `tests/` and `_hf_spaces_proxy/security/` on `sys.path` and
    import test and source modules as top-level names, and the
    `_load(name, path)` helpers execute the same files under further names.
    cProfile keeps one copy's record per `(file, line, name)` and adds up the
    edges of all copies (`cProfile.Profile.snapshot_stats`, reproduced), so
    cumulative times of those functions are wrong (`source_tree_sha256`:
    0.01 s recorded, 198 s in its edges).
  - `test_run168_preserves_three_generation_run167_history`, here: setup
    15.6 s, call 2.3 s; under cProfile the module fixture `base` (23.4 s) is
    `t167.base` 7.6 s, three `t167._advance` 2.7-4 s each, and four
    `t166._advance_run163` about 1.5-2 s each (two inside `t167.base`): each
    module rebuilds its predecessors' chain (Runs 160 to 167) from scratch.
  - `ci_codecov_test_coverage.yml`: `report` runs `if: success()`, so a failed
    shard skips the required check, which GitHub then reports as passing; the
    shard upload is `if: success()`, so a failed or timed-out shard leaves no
    JUnit or coverage data.

### Design
- Recorder `scikitplot/_testing/_perf_telemetry.py`, registered by
  `scikitplot/conftest.py` only when `SKPLT_TEST_PERF_DIR` is set (off by
  default, conftest rule 4). One JSON line per test, flushed as it ends, so a
  killed shard keeps what it measured: wall and CPU per phase, child CPU
  (`RUSAGE_CHILDREN`), RSS and peak growth, GC per generation, fixture setups
  with scope and location, file / subprocess / network / module-execution
  counts from an audit hook, collection time per test file.
  `SKPLT_TEST_PERF_PROFILE=<regex>` writes a cProfile per selected test, and
  is skipped with a reason when another profiler is active (Python 3.11
  replaces it silently, 3.12+ refuses). `SKPLT_TEST_PERF_TRACEMALLOC=1` adds
  Python allocation peaks. Never raises into a test, never warns, never
  collects garbage.
- Analyzer `.github/scripts/ci_test_perf.py`, standard library only, one file:
  `report` takes files, directories and archives (nested GitHub artifact zips
  included, read in memory with size and depth limits) holding recorder
  files, JUnit, cProfile, Cobertura `coverage.xml` or `test_durations.json`;
  writes `report.md`, `summary.json`, `ANALYZE_ME.md` and, with `--bundle`, a
  reproducible zip carrying them, the inputs and the analyzer itself.
  `compare` (asv-style ratio and absolute threshold) and `profile` (edge-exact
  callers and callees, re-executed modules flagged). Units are the planner's
  (`Planner.owner` rule, containers recorded by the recorder).
- Workflow: `report` becomes a real gate (`if: always()`); shard artifacts are
  kept on failure and timeout; the recorder runs on every shard; a
  `profile_tests` input profiles chosen tests in CI; the report job publishes
  the analysis (`test-perf` artifact, step summary).
- Not in this task: changing what the security tests build or verify. The
  chain recomputation is routed to `upcoming_changes/` with its measurements.

### Steps
- [x] recorder + conftest registration + tests (floor Python 3.8 for the
      CI scripts; the recorder's floor pytest 7.1.2 is not yet run)
- [x] analyzer + tests (unittest, standard library, as the plan job runs them)
- [x] dogfood: record the run168 module and `security/` here; bundle; compare
- [x] workflow: gate, uploads, recorder env, `profile_tests`, report step
- [x] `.github/ci/README.md` sections 12-13; `skills/_repo/SKILL.md`; Makefile targets
- [x] rules for the lessons (131-137); `upcoming_changes/` notes; run168
      instrumentation removed
- [x] analyzer hardening: every field it reads is type-checked (bad lines are
      skipped and counted, rule `invalid-lines`); XML is read only as UTF-8
      without NUL or entity declarations (Rule 136); malformed Cobertura,
      JUnit, durations and summaries are refused with the file named
- [x] package benchmarks (added after Allen's second request, 2026-10-23):
      `benchmarks/` in asv conventions, `.github/scripts/ci_bench.py` (run,
      pair, compare without asv), `make bench*`, tests
- [x] on-demand workflow `ci_performance.yml`: tests of chosen paths without
      coverage; benchmarks base vs head on one runner
- [x] opt-in `SKPLT_TEST_GC=frozen` (`_FreezeSurvivors`), its tests, README
      section 8, `test_gc` choice in both workflows
- [x] recorder on its floor: Python 3.8 with pytest 7.1.2, `-W error`
- [x] quiet A/B/B/A of `young` against `frozen` on the same selection
- [x] independent review (three reviewers: analyzer, benchmark runner and
      workflow, recorder and GC policy); every finding fixed with a test
- [ ] gates on the final tree

### Acceptance criteria
- Disabled: no plugin registered, no audit hook, no file written.
- Enabled: `cost of every test` (the conftest monitor) rises by under 1 ms on
  the security suite; a test that fails, errors, skips or xfails is recorded
  with its outcome; a session killed mid-test leaves parseable lines.
- The analyzer gives byte-identical output for the same inputs, refuses
  unsafe archive members, and reproduces the run168 breakdown above from a
  per-test profile without editing the test.
- `check_records.py` reports no failure that the uploaded tree did not have.

### Risks
- An audit hook cannot be removed: it is installed only when the recorder is
  on, does no I/O, and counts its own errors instead of raising.
- `filterwarnings = error`: the recorder must not warn; its tests run the
  child session with `-W error`.
- The workflow cannot be run here: the YAML is checked by parsing and by
  `zizmor` when available; each changed condition is named in the results.

### Results Review - TP-1 (2026-10-23)
- Built: recorder (`scikitplot/_testing/_perf_telemetry.py`), analyzer
  (`.github/scripts/ci_test_perf.py`), package benchmarks (`benchmarks/`, asv
  conventions) with a standard-library runner (`.github/scripts/ci_bench.py`),
  on-demand workflow `ci_performance.yml`, opt-in `SKPLT_TEST_GC=frozen`,
  `make test-perf*` / `make bench*`, README sections 8, 12, 13, the repo
  skill, rules 131-137, three `upcoming_changes/` notes.
- Recorder cost: 0.22 ms per test over 4 134 tests (`recorder_seconds`, now
  the recorder thread's CPU time, with the once-per-session start-up apart).
  Floors: Python 3.8 with pytest 7.1.2 and `-W error`: outcomes, fixtures,
  profiles correct, 0 recorder errors.
- Run168, profiled without editing the test: `base` 25.73 s = three
  `t167._advance` 11.78 s + `t167.base` 7.95 s + two `t166._advance_run163`
  6.00 s (`upcoming_changes/.../security-tests-rebuild-their-chain.md`).
- GC, five submodules, 4 134 tests, 93 files, young / frozen / frozen /
  young on an idle machine: full collections 21.0-21.4 s against 1.1 s;
  sessions 165-172 s against 144-151 s; peak RSS 402-403 MiB against
  373-375 MiB; the same 7 failures in every run (4 in
  `scikitplot/_utils/tests/test_timeout.py`: `signal.alarm(0.1)` raises
  `TypeError`; 3 cleanprompt timeout tests). Runs made while other work
  shared the machine showed tests "slower" that the quiet runs did not
  (Rule 135).
- Benchmarks: whole suite, `pair` with one environment on both sides
  (4 rounds a side): 0 slower, 0 faster, 5 inconclusive, 20 unchanged. The
  same real samples with head scaled offline: 2x detected for 19 of 19
  timings, 1.5x for 17, 1.25x for 9 (the rest `x`). `asv check
  --python=same`: no problems; `asv run --python=same --quick` runs the suite.
- Independent review, three reviewers with reproductions: analyzer 17
  findings (damaged zip members, huge integers, deep JSON, a 5-byte `.prof`
  that makes `marshal` allocate gigabytes, Markdown injection into
  `ANALYZE_ME.md`, outputs re-read as inputs, absolute paths, platform bytes
  in the bundle, unbounded lists, JUnit-against-recorder compares that
  compared nothing, ...); runner and workflow 12 (an outlier sample hid a 2x
  slowdown, relative interpreters, `--quick` false changes, both-sides
  failures shown as unchanged, asv differences: inheritance, module
  `setup`, `setup_cache`, GC on while timing, dedent); recorder and GC 5 (a
  forked child could wait forever for the audit lock; wall-clock self-cost;
  two full collections per module under `frozen`; doctest files never ended
  a module; paths inside arguments). All fixed, each with a test. A first
  fix of the benchmark rule (a rank test on all samples) failed its own
  A/A check (6 of 25 marked); the rule compares whole rounds instead.
- Tests: `.github/scripts/tests` pass on Python 3.8, 3.9, 3.11, 3.12, 3.13;
  recorder and GC policy tests (95) pass, also under `-n 2`.
- Gates: black and ruff clean on the new files; `scikitplot/conftest.py`
  unchanged in ruff count; codespell, ASCII (sources), whitespace, path
  lengths, test module names: clean; zizmor: no findings;
  `check_records.py` and `test_records.py`: exactly the failures of the
  uploaded tree (`tasks/r4r_migrate.py`, the empty `numpydocstr.md`, the
  archived parts one character short of their hashes).
- Not verified: no workflow ran on GitHub (the pip cache, the on-demand
  build of base and head, `download-artifact` with nothing to download);
  free-threaded Python; Windows and macOS (`ru_maxrss` units, the shell
  wrappers of the runner's tests are POSIX-only); `frozen` on the whole
  CI suite.
- Environment-only (not a repository issue): a Debian-patched `site.py`
  here processes a venv's `.pth` twice, so each editable import ran ninja.
- Next: run `test_gc: frozen` once in CI and `compare` it with a `young`
  run before making it the default; fix the security chain rebuild and the
  `scikitplot.__getattr__` probes (`upcoming_changes/`).

## Task: TF-1 - submodule test failures, fixed under Rules 138-151 (handoff)

### Context
- Goal: every submodule's tests pass on their own and in the full suite,
  each failure fixed at its root cause under `skills/_engineering/rules/`
  (logging.md 138-141, module_submodule_actual_mock_patch.md 142-146,
  numpydocstr.md 147-151; logging.md also 152 output and 153 negative log
  assertions). One submodule per session; a session starts here
  and needs no earlier chat.
- State on 2026-10-24: TP-1 and Rules 138-153 delivered; records check
  `ok rules` (router and archives failures pre-existing).

### Known failures and their causes
- `scikitplot/annoy/tests/test_bundle_directory.py:413,469`: log capture
  depends on the pytest release (passes 9.1.1, fails 8.3.5 and 9.0.0)
  because `"scikitplot"` has `propagate = False` (Rule 139). Same file:
  absolute imports at 23, 191, 222 (Rule 142); string logger names at
  410, 466 (Rule 140).
- Annoy benchmarks (`benchmarks/bench_annoy.py`): failures reported by the
  user on 2026-10-23, output not yet collected.
- `scikitplot/_utils/tests/test_timeout.py`: 4 failures, `signal.alarm(0.1)`
  raises `TypeError` (the function takes whole seconds); see
  `upcoming_changes/scikitplot/_utils/run-with-timeout-tests-and-float-seconds.md`.
- cleanprompt: 3 timeout tests fail (cause not yet analysed).
- `scikitplot/_externals/_sphinx_ext`: `sys.path` edits and loads by path
  in tests (Rule 143); one file executed 29 times per session.

### Implementation Steps
- [ ] 0. The shared capture (Rule 139): `scikitplot/logging/_capture.py`
  (a factory `captured_logger(caplog)` returning a context manager taking
  `(level, logger_or_name)`; no pytest import), its tests in
  `scikitplot/logging/tests/`, registration in `scikitplot/conftest.py` and
  in the top-level conftest of each submodule with a partial distribution;
  the corpus fixture becomes that registration (its 30 calls unchanged).
  Confirm `libs/_tools` counts the new file as `scikit-plots-skinny`'s.
- [ ] 1. annoy: tests (Rules 139, 140, 142, 144, 153), then the benchmarks.
- [ ] 2. `_utils` timeout (fractional seconds, Rule 149 for the docstring).
- [ ] 3. cleanprompt timeout tests.
- [ ] 4. `_sphinx_ext`: remove `sys.path` edits and path loads (Rule 143),
  submodule by submodule.
- [ ] 5. Remaining submodules' Rule 142/144 violations, when their tests are
  next touched (survey: 624 absolute self-imports, 208 string patch targets
  in tests).

### Acceptance Criteria
- [ ] The submodule's tests pass as one file, as the directory, under
  `-n auto`, and under pytest 9.0.x and the latest 9.x.
- [ ] Each rule's "Verified by" command is clean for the changed files.
- [ ] A rule found wrong or missing while fixing is amended, or the next
  number added, in the same change, with the failure as its root cause.

- Negative log assertions that pass under pytest 9.0 whatever the code
  logs (Rule 153): `_sphinx_youtube_gallery/tests/test_sync.py:827`,
  `annoy/tests/test__threads.py:157`, `tests/test__distributions.py:598`.
- Output defects (Rule 152): `_sphinx_youtube_gallery/sync.py:727-731`
  debugging prints on every `resolve_source` call; `utils/_serialize.py:177,
  205,237` print an exception and return `None`;
  `api/metrics/_regression/_residuals.py:216-270` print fitted parameters.

### Side issues found, not changed
- `SKPLT_LOGGING_AUTO_CONFIG` (`_ENV_AUTO_CONFIG`) is defined and unused.
- `sanitize_log_message` docstring writes a doubled backslash inside
  `r"""` (Rule 48).
- `scikitplot/_cli/logging.py` uses `getLogger("scikitplot")` (allowed by
  Rule 138 for CLI entry points).
