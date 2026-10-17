<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 30 - CI collision (again), options regression, forge engine, GitLab, Bitbucket, live-Space check, logging flake

### Context
- CI (PR 864, 2026-10-11): shard and nine partial legs still fail on
  `_sphinx_feedback/tests/test_sphinx_build.py`: the uploaded tree still has
  it (round 29b's archive could not delete it).
- Uploaded tree also carries a lint edit in `_service/_config.py` that made
  numeric custom options raise `TypeError` (`len()` before the type check).
- Maintainer asks to continue the open items: native forge storage and the
  live Space.

### Implementation Steps
- [x] 1. Delete the stale file; `tools/maint_tools/check_test_module_names.py`
      (pytest prepend-mode names) + tests + `pr_check_test_module_names.yml`;
      `REMOVED_FILES.txt` in the archive
- [x] 2. Options: type-first validation, finite floats; 11 tests (7 fail on
      the uploaded code)
- [x] 3. Proxy `/` and `/health`: `page_feedback` status; one default constant
- [x] 4. `_service/_forge.py`: engine extracted from GitHub (GitHub's 27 tests
      unchanged and green); `GitHubOps`
- [x] 5. `_gitlab.py` (docs-checked calls), `_bitbucket.py` (Cloud);
      config (api_url, nested GitLab groups, Bitbucket Cloud only); core
      dispatch; 31 parity tests over both; mutation checks caught
- [x] 6. CLI `check-provider` (read-only); test asserts zero writes
- [x] 7. Logging flake: probe found two offenders; autouse restore fixture
- [x] 8. README, `_example_conf.py`, tools README, ledger narrowed to HF +
      Bitbucket DC, fragment, lessons 67-69, assistant HISTORY, mirror, 0.9.0

### Verification
- `check_test_module_names.py scikitplot`: ok (and CLASH when the stale file
  is put back).
- `_sphinx_feedback` suite `-W error`: 422 passed.
- AI-assistant whole tree `-n 8`: 2815 passed, 2 skipped; proxy+integration+
  architecture `-n 8` x3: green after the fix (was 1 failure in 2).
- Probe: each of 80 test files before `test_root_logging_isolation.py`:
  offenders before = 2, after = 0.
