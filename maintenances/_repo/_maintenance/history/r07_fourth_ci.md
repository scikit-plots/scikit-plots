<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 7 - fourth CI run: two tests left on Windows

### Context
- Goal: the two failures of the Windows verify leg of CI run 37694585156 (260 rows passed,
  1 failed; sphinx-ext: 2 failed, 6165 passed, 99 skipped). Every other row is green.
- Ground truth: the uploaded job log; the pull request's head (`refs/pull/849/head`,
  42fe2af) = the round 6 delivery plus one line of yours (`atomic_write`'s docstring made
  raw). The drop-in is a diff against that commit.

### Findings and root causes
| Test | Symptom | Root cause | Kind | Evidence |
| --- | --- | --- | --- | --- |
| `test_mutant_is_caught[invisible-chars-narrowed]` | `TypeError: argument of type 'NoneType' is not iterable` | the harness reports the failing case with a zero-width joiner in it (UTF-8 `e2 80 8d`); `subprocess.run(text=True)` reads in the locale's encoding, cp1252 on Windows, where `0x8d` is undefined; the reader stops and `stdout` is `None` | test | the harness output captured as bytes: decodes as UTF-8, fails as cp1252 at `0x8d` in `got "a\xe2\x80\x8db"`; under the `enc` switch 1 failed before (this test), 0 after |
| `_sphinx_collection` `test_final_collection_asset_integrity_fails_closed_on_stale_output` | `'examples/index.html'` not in `...status sibling: examples\index.html` | `verify_collection_assets` named a stale page with `str(path)`: backslashes on Windows. Two tests disagreed about that message: this one expected `/`, `test_verify_lists_stale_pages_relative_and_sorted` expected `os.path.join` | **product** | the label is `as_posix()` now: one message on every platform; both tests expect `/` |

Why round 6 did not have them:
- The first was hidden behind the line-ending failure of the same test: with CRLF the harness
  stopped before it printed the joiner. Round 6 said one fix can uncover another; this is it.
- The `enc` switch of the emulation was silent, and wrongly so: Python 3.11+ resolves the
  default encoding before it builds the pipe's reader, and the switch only replaced the
  reader's default. It had never been shown a known positive (Rule 28). Repaired; it now
  fails exactly this test on the old file.
- The second was the 49th failure that no switch produced: path separators are not emulated.

### Also in this drop-in
- `test_js_harnesses.py` starts the same Node harnesses the same way; it gets the same named
  encoding, so a failing harness's report cannot be lost on Windows.
- `cleanprompt/_files.py`: after your change to a raw docstring, `%LOCALAPPDATA%\\cleanprompt`
  showed two backslashes; one now.

### Observation, not changed
- 40 other `subprocess` calls with `text=True` and no `encoding` remain in the sphinx-ext
  stack (2 product files, 18 test files). They pass on Windows today. Each has the same
  exposure the day its child prints a character outside cp1252.

### Acceptance Criteria
- [x] Both failures reproduced or demonstrated on Linux, and shown passing.
- [x] `python -m libs._tools verify` green on Linux for 3.8 and 3.13.
- [ ] Windows leg green (CI only).

### State of the emulation switches (Rule 47)
| Switch | Known positive |
| --- | --- |
| `nl`, `mode`, `defpath`, `home`, `zip`, `held` | round 6 calibration table |
| `enc` | this round: `test_mutant_is_caught[invisible-chars-narrowed]` on the old file |
| `clock` | round 5: `assert 'deadline_seconds' in []` in corpus |
| `names` | untested: no failure on Windows has needed it |
| path separators, drive letters | not emulated |

## Results Review - Round 7 - 2026-10-08

### Implementation Summary
- Files changed: 5 modified, none new, none removed or renamed. Product: 1 line in
  `_sphinx_collection/setup.py`, 1 docstring character in `cleanprompt/_files.py`. Tests: 3.
- Shared with the documentation repository (`_sphinx_ext`): `_sphinx_collection/setup.py`,
  `_sphinx_collection/tests/test_setup.py`, and `test_mutation.py`, `test_js_harnesses.py`
  under `_sphinx_ai_assistant/tests/_architecture/`.

### Verification Evidence
- [x] `python -m libs._tools verify`, Linux, final tree: 3.13 261/0/0, 3.8 164/0/3
      (passed / failed / skipped). sphinx-ext from the wheel: 6186 passed.
- [x] sphinx-ext from the checkout: 6310 passed; the same under the repaired `enc` switch:
      6310 passed.
- [x] `enc`, old `test_mutation.py`: 1 failed (`invisible-chars-narrowed`), 266 passed.
- [x] `precheck status=0` on the 5 files.
- [ ] Windows: not run here.

### Deviations from Plan
- None. 3.9, 3.10 and 3.15 were not rerun: the change is two keyword arguments, one
  `as_posix()` and one test literal.

### Approval Checklist
- [x] Both failures have a demonstrated cause and a passing run
- [x] No test deleted or weakened
- [ ] Windows leg green - next CI run
