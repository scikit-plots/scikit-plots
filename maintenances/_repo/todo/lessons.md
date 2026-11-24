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

### Rule REPO-L2: A helper that produced evidence is committed with the evidence
- **When:** a pull request or ledger note cites numbers produced by a script
  (a sweep, a benchmark, a timer, a sampler).
- **Then:** the script is in the tree in the same pull request or the one
  before it, under `tools/` with tests (engineering Rule 173).
- **Verified by:** every command quoted in the pull request description runs
  from a fresh clone.
- **Added:** 2026-11-24. Root cause: PRs 955 and the sweep that followed used
  helpers kept in a scratch folder; the maintainer asked twice to keep them.

### Rule REPO-L3: Tests never reach the network, the resolver or a data host
- **When:** a test mocks HTTP, resolves a host, or loads NLTK, spaCy or model
  data.
- **Then:** answer DNS with the corpus `fake_dns` fixture, use a fake corpus
  or `pytest.skip` when data is absent, and never pass `allow_download=True`
  outside a test that fakes `nltk`. In `scikitplot.corpus` autouse guards
  refuse real DNS and `nltk.download`, so a violation fails everywhere.
- **Verified by:** the corpus suite passes in a sandbox without a resolver
  (`socket.getaddrinfo` raises), with `nltk` installed and no data.
- **Added:** 2026-11-24. Root cause: 22 tests mocked HTTP but not DNS and
  failed on every offline machine (PR 970); then two stopword tests passed
  `allow_download=True` and failed in CI, which cannot reach the NLTK host.

### Rule REPO-L4: Log assertions use the package's own capture, not caplog
- **When:** a test asserts that a submodule logged something.
- **Then:** use the `captured_logger` fixture or patch the module's `logger`,
  and run the test once with propagation off
  (`logging.getLogger("scikitplot").propagate = False`).
- **Verified by:** the partial-distribution job (`scikit-plots-<part>`)
  passes, not only the full-package job.
- **Added:** 2026-11-24. Root cause: two CX-53 tests (PR 965) used `caplog`;
  in the corpus partial distribution the package logger does not propagate
  to the root logger, so they passed locally and failed in that job.

### Rule REPO-L5: Checks gate the commit; real data gates the merge
- **When:** committing, and before asking for a merge of a reader change.
- **Then:** commit only after `pre-commit` exits 0 (no `;` between them),
  run `generate_pr_labeler.py check` when a folder under `upcoming_changes/`
  appears or empties, and benchmark readers on real files, not only on
  generated ones.
- **Verified by:** the pull request quotes each check's result.
- **Added:** 2026-11-24. Root cause: a commit landed while ruff failed
  (PR 969); the labeler check failed twice in CI (PRs 959, 961); and a PDF
  fix passed every synthetic test but spaced every letter of a real file
  whose font size lived in the text matrix (PR 969).

### Rule REPO-L6: A new public name regenerates the usage skills
- **When:** a change adds, removes or renames a name in a submodule's
  `__all__`, or edits its first docstring line.
- **Then:** install the `usage-skills` extra and run
  `python tools/maint_tools/generate_usage_skills.py sync`, then `check`, and
  commit `plugins/` with the source change; also rebuild the API reference
  (pre-commit hook `apis-reference-coverage`).
- **Verified by:** `generate_usage_skills.py check` exits 0 and
  `scikitplot/tests/test_usage_skills.py` passes.
- **Added:** 2026-11-24. Root cause: PR 972 exported `ReadCoverage`; the
  API reference hook caught it locally, but the corpus usage skill was only
  checked in CI's full test shard, which failed.

### Rule REPO-L7: Run tests with unclosed-resource warnings as errors
- **When:** a change opens files, sockets or responses, or changes who closes
  them.
- **Then:** run the affected tests with `python -X dev -W error::ResourceWarning
  -m pytest -W error::pytest.PytestUnraisableExceptionWarning`, as CI turns an
  unclosed file into a failure when the garbage collector happens to run.
- **Verified by:** that command passes for the changed submodule.
- **Added:** 2026-11-24. Root cause: PR 974 moved the datasets loaders'
  file opening into helpers; three files opened by `get_file_data` were never
  closed (a refused pickle, a `.npy` read), which passed locally and failed in
  CI's test shard.

### Rule REPO-L8: A dependency change regenerates the partial-distribution files
- **When:** a change edits an extra or a dependency in `pyproject.toml` or
  `libs/*/pyproject.toml`.
- **Then:** run `python -m libs._tools generate` and commit what it writes
  (the `libs/*/README_*.md` install lines list each extra's requirements),
  with `python tools/requirements/generate_requirements.py` for
  `requirements/`.
- **Verified by:** `python -m libs._tools check` exits 0.
- **Added:** 2026-11-25. Root cause: PR 975 added `beautifulsoup4` to the
  `corpus` extra by hand; `libs/corpus/README_CORPUS.md` went stale on
  `main`, and the next pull request (976) failed "Generated packaging files
  are up to date".
