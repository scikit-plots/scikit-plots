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

### Rule REPO-L9: A syntax check does not prove an API exists on the floor
- **When:** new code, test helpers included, calls the standard library.
- **Then:** besides `ast.parse(..., feature_version=(3, 8))`, check each
  call's keyword arguments against the documentation's "Changed in version"
  notes; the partial distributions' CI runs the tests on Python 3.9.
  Known traps: `Path.write_text(newline=)` (3.10), `zip(strict=)` (3.10)
  and `str.removeprefix` (3.9).
  Static checkers such as `vermin` miss keyword arguments.
- **Verified by:** the "Verify · py3.9" job of
  `ci_libs_partial_distributions.yml` passes.
- **Added:** 2026-11-25. Root cause: PR 977's hostile sample wrote with
  `Path.write_text(..., newline="")`; the syntax check passed and the
  py3.9 job failed with `TypeError`.

### Rule REPO-L10: Check what a commit holds before pushing it
- **When:** any commit, and above all after `git stash` / `git stash pop`
  between staging and committing (a plain `pop` restores changes as
  unstaged, so an earlier `git add` is lost).
- **Then:** stage right before committing, compare against a base with a
  copy (`git show main:path > /tmp/x`) rather than a stash, and read
  `git show --stat HEAD` before `git push`: every changed source, test and
  record file must be listed.
- **Verified by:** the pushed commit's file list matches the pull
  request's description.
- **Added:** 2026-11-25. Root cause: PR 982 was pushed holding only its
  release note; a stash and pop used for a before/after comparison had
  unstaged the code and tests.

### Rule REPO-L11: Tests assert the contract, not one Python's limit
- **When:** a test exercises a limit set by the interpreter or a library
  (recursion depth, a parser's nesting limit, float formatting).
- **Then:** assert what holds on every supported version (a clean read
  or the documented `ValueError`, never a crash), and force the limit's
  own path with a mock to test its message.
- **Verified by:** the partial distributions' newest-Python job passes.
- **Added:** 2026-11-25. Root cause: PR 981's test expected `json` to
  raise `RecursionError` at 100 000 levels; Python 3.15 parses that deep
  (recursion is bounded by stack size since 3.14), so the file was
  refused for its shape instead and the message differed.

### Rule REPO-L12: Write invisible characters as escapes
- **When:** source text needs a byte-order mark, a no-break space or
  another invisible or look-alike character.
- **Then:** write the escape (`"﻿"`, `" "`), never the
  character; some editing tools turn an escape into the character, so
  search new files for them before committing
  (`grep -rP '\x{feff}|\x{a0}' --include=*.py`).
- **Verified by:** ruff's ambiguous-character checks (RUF001-RUF003) pass
  and the search finds nothing.
- **Added:** 2026-11-25. Root cause: `_table.py` (PR 978) and
  `_notebook.py` (PR 981) were merged with a literal byte-order mark in a
  string; ruff does not flag that one. PR 983 replaced them with escapes.

### Rule REPO-L13: Check that a pull request is still open before pushing to it
- **When:** pushing a follow-up commit to a pull request's branch.
- **Then:** read its state first (`gh api repos/OWNER/REPO/pulls/N --jq
  .merged`); a branch pushed after its pull request merged changes
  nothing on `main`. If it merged, open a new pull request from `main`
  with the commit, and correct the old description.
- **Verified by:** after merging, `git log origin/main` holds the commit.
- **Added:** 2026-11-25. Root cause: PR 983 merged at its first commit;
  its review commit (fast companion lookup, companion options, linear
  markup) was pushed minutes later and never reached `main`. PR 984
  carries it.

### Rule REPO-L14: Do not treat an unchanged file time as "nothing changed"
- **When:** a cache is keyed on a file or folder modification time.
- **Then:** reuse an entry only when the time is settled, at least 2 s
  older than the moment the entry was taken; otherwise recompute.
  File times move in steps (about 15.6 ms on Windows, 2 s on FAT), so
  two changes in one step leave the time unchanged.
- **Verified by:** a test that writes, sets the folder time back to its
  old value with `os.utime`, and still sees the new file. Tests set
  times in whole seconds: Windows stores 100 ns units and FAT 2 s
  units, so `ns=(1, 2)` stores the same time twice there.
- **Added:** 2026-11-25. Root cause: PR 984's folder-listing cache
  served a stale listing on Windows CI, because files created within
  one clock step left the folder time unchanged.
