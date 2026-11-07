# Lessons: `_repo` (CI, `libs/`, labeling, release and records tooling)

## Active Rules

### Rule REPO-L1: A generator's output changed - run the generator's own suite
- **When:** a change alters what a generator writes (`libs/_tools`,
  `tools/maint_tools/generate_usage_skills.py`, `build_skills_repo.py`,
  `generate_pr_labeler.py`, `check_records.py --write-index`), even through a
  function it calls from elsewhere (`scikitplot/_distributions.py`).
- **Then:** regenerate, run the generator's `check`, and run its test suite
  from the directory CI runs it in: `cd libs && python -m pytest _tools/tests`
  (it has its own `pytest.ini`), `python -m pytest tools/maint_tools/tests`,
  `python -m pytest maintenances/_repo/_maintenance/tests`. Then search the
  tests for the text that changed (`grep -rl "<old text>" --include=test_*.py`)
  and run every match.
- **Verified by:** each of those commands in the change's verification, and
  the grep listing every matching test module.
- **Added:** 2026-11-07. Root cause: PR 911 changed the install command of the
  `libs/` READMEs through `_distributions.pip_install` and ran `libs._tools
  check` (files current) but not `libs/_tools/tests`, whose
  `test_package_description_states_how_to_install_it` pinned the old bare
  `pip install <name>`; CI failed on 8 of the 9 parts. The ninth passed only
  because its staging pair's second line happened to contain the old text:
  the test now pins the exact command from the map and allows a bare name
  only as the second half of the staging pair (a mutation that prints the
  bare line fails all 9).
