<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 6 - third CI run: the Windows leg only

### Context
- Goal: every failure of the Windows verify leg of CI run 37668804532 fixed at its cause, or
  stated as a fact of the platform. Every other leg of that run is green.
- Ground truth: the uploaded job log and log archive, and the pull request's head
  (`refs/pull/849/head`, 7ffb50d), cloned read-only. It is the round 5 delivery plus your
  commit "doc" (formatting in `verify.py`, a spelling in `test__threads.py`, the blank last
  line of two test files, one README line). The drop-in is a diff against that commit.
- Not available: Windows itself, and the artifact with the complete test logs. The job log
  names 40 of the 49 sphinx-ext failures; the other three suites are named completely.

### Where the run stood
| Leg | Result |
| --- | --- |
| every ubuntu leg, macOS, Pyodide | green |
| Windows 3.12 | 256 rows passed, 5 failed: sphinx-ext 49 tests, cleanprompt 9 (twice: alone and together), cython 3, mlflow 1 |

### Method: the Windows rules, one at a time, on Linux
`/home/claude/work/winemu` (not part of the delivery) changes one documented behaviour of
Python per switch, in the test process and every Python it starts:

| Switch | What it changes | Source of the rule |
| --- | --- | --- |
| `nl` | text files, stdout and text pipes written without `newline` end lines with CRLF | CPython `textio.c`, `pylifecycle.c` (`MS_WINDOWS`) |
| `mode` | permission bits: files 0o666 or 0o444, directories 0o777, whatever is asked | `os.chmod` documentation |
| `defpath` | nothing is found through `os.defpath` | `ntpath.defpath` is `'.;C:\\bin'` |
| `home` | `~` expands from `USERPROFILE`, never `HOME` | `ntpath.expanduser` |
| `zip` | `zipfile.ZipInfo(name)` turns `os.sep` into `/` | `zipfile.ZipInfo.__init__` |
| `held` | a file this process has open or mapped cannot be removed or replaced | sharing violation |
| `clock`, `names`, `enc` | 15.6 ms clock; names Windows rejects; cp1252 as default encoding | - |

Calibration (Rule 28), on the wheels of the failing commit:

| Suite | Windows named | Reproduced on Linux |
| --- | --- | --- |
| sphinx-ext, mutants | 29 | the same 29, by name |
| sphinx-ext, others named | 11 | 11 |
| cleanprompt | 9 | 8 with `nl,mode,home`; the 9th with `held` |
| cython | 3 | 0: these set `os.name` itself (read, not emulated) |
| mlflow | 1 | 0: a path printed with backslashes (read, not emulated) |

The switches also failed tests Windows skips by `os.name` (5 in cleanprompt, 1 in cython) and
two that are artefacts of the emulation (a POSIX shared-memory name in cython under `held`;
a parent that decodes UTF-8 while the child writes cp1252 under `enc`). Those are read and
set aside, not counted.

### Findings and root causes
| Where | Tests | Root cause | Kind | Evidence |
| --- | --- | --- | --- | --- |
| sphinx-ext `test_mutation.py` | 29 | the mutated copy of `ai-assistant.js` was written with `write_text`: on Windows every line ends CRLF, and five harnesses that read the script line by line stop before an assertion ("crashed rather than failing") | test | `nl`: 29 failed before, 0 after (537 passed); the four harnesses exit 1 with no summary on a CRLF copy |
| sphinx-ext release gate | 3 + 12 | the gate records permission bits (`new file mode 100755`, `0o755` after replay) and finds Git through `os.defpath`; Windows has no executable bit and its `os.defpath` holds no Git (`GIT_REQUIRED`) | platform | `defpath`,`mode`: the same 15 fail; marked `POSIX_RELEASE_GATE` |
| sphinx-ext native status | 2 | the test starts `verifier.py` through its `#!` line | platform | `mode`: both fail; marked `RUNS_A_SCRIPT_BY_ITS_FIRST_LINE` |
| sphinx-ext `test_redis_chaos_attestation.py` | 1 | the last statement flips the executable bit and expects another digest; `chmod` cannot set that bit on Windows | platform | that statement runs on POSIX only; the content and cache statements run everywhere |
| sphinx-ext `test__zip_workspace.py` | 1 | `ZipInfo("a\\evil")` becomes `a/evil` on Windows before anything is written: the archive under test held no backslash | test | `zip`: fails before; after, the stored name is asserted and the archive is refused. The product refuses a real backslash name on Windows through its raw-name comparison |
| cleanprompt `test__mcp.py`, `test__runtime.py` | 4 | fixtures written with `write_text`; the tools keep a file's line endings, so the answers ended CRLF | test | `nl`: 4 failed before, 0 after |
| cleanprompt `_catalog.write_compiled` | (1) | the generated, committed `_compiled.json` was written in text mode: CRLF on Windows, so the same definitions compile to another file there | **product** | `nl`: `test_compiling_twice_writes_the_same_bytes` fails before, passes after. Not in the CI list: the test needs PyYAML, which the verify environment does not install, so it is skipped there |
| cleanprompt vault modes | 3 | the tests assert `0o600` / `0o700`; Windows has no permission bits | platform | marked "POSIX modes only", as two neighbours already were; README and `atomic_write` now say what holds on Windows |
| cleanprompt `test_a_user_path_is_expanded` | 1 | the test sets `HOME`; Windows expands `~` from `USERPROFILE` | test | `home`: fails before, passes after |
| cleanprompt `atomic_write` | 1 | `os.replace` is refused while a reader has the vault open (`[WinError 5]`) | **product** | bounded repetition as in corpus (15 attempts, about 1.2 s), Windows only; the reader test now states the contract: a refusal is allowed on Windows, a partial file or a scratch file never |
| cython `_default_cache_dir` | 3 | the tests set `os.name` for the whole process, which breaks `pathlib` on the platform it is not; the function read three globals | **product** (testability) | platform, environment and home are parameters; every branch, the Windows ones included, now runs on every platform |
| mlflow `test__facade.py` | 1 | `str(Path)` compared with a POSIX literal | test | `as_posix()` |

### What the reader test taught
With open files made unreplaceable (`held`): without repetition 30 to 35 of 40 writes were
refused; with the full pauses 14 or 15 were still refused and the test took 20 s. Repetition
therefore helps against a reader that lets go, and cannot beat one that never does. The test
says so (short pauses, refusals counted on Windows, forbidden on POSIX), and `atomic_write`
documents `PermissionError` for that case.

### Not found
- One sphinx-ext failure of the 49. The log names 40; the emulation predicts 8 of the other 9
  (6 release gate, 2 native status); none of the nine switches produces a ninth. It sorts
  after `test_promote_release.py::test_run149_finalize_rejects_stale_signature_record`.
  With at most a handful of failures left, the next job log names it.

### Observations, not changed (yours to decide)
1. `promote_release._trusted_git_executable` searches `os.defpath`. On Windows that is
   `.;C:\bin`: no Git there, and the first entry is the current directory. The search is the
   documented design of the release tools (`RELEASE_PROCESS_HERMETICITY_GUIDE.md`, and a test
   that pins the source line), and 12 modules give adapters `PATH=os.defpath`. Whether the
   release tools should refuse Windows outright, or require the pin there, is a decision.
2. `corpus` leaves SQLite connections for the interpreter to close (63
   `ResourceWarning: unclosed database` lines at exit in each corpus run here). Not a failure
   in any job.
3. `cleanprompt._runtime._write_zip` calls `os.replace` directly, without the repetition.
4. `cython/tests/test__profiles.py` still sets `os.name` for the process in four tests; they
   pass on both platforms.

### Implementation Steps
- [x] 1. Emulation, calibrated against the named failures.
- [x] 2. sphinx-ext: mutated copy as bytes; `tests/_platform.py` with two markers and their
       reasons; 17 tests marked; one statement POSIX-only; zip entries with exact names.
- [x] 3. cleanprompt: fixtures as bytes; `write_compiled` as bytes; `_replace` with bounded
       repetition; reader test as a contract; three mode tests marked; `~` test; README.
- [x] 4. cython: `_default_cache_dir(os_name=, environ=, home=)`; tests pass arguments.
- [x] 5. mlflow: `as_posix()`.
- [x] 6. Local run of the repository's hooks on changed files (end of file, trailing
       whitespace, codespell 2.4.3, black 26.5.1).

### Acceptance Criteria
- [x] Every named Windows failure is reproduced failing and shown passing on Linux, or is
      marked with the fact of the platform that stops it.
- [x] No test deleted; 20 tests skip on Windows only, each with its reason.
- [x] `python -m libs._tools verify` green on Linux for 3.8, 3.9, 3.10, 3.13, 3.15.
- [ ] Windows leg green (CI only; one failure not yet named).

## Results Review - Round 6 - 2026-10-07

### Implementation Summary
- Files changed: 18 (17 modified, 1 new: `_sphinx_ai_assistant/tests/_platform.py`); nothing
  removed or renamed. 366 lines added, 46 removed in the modified files.
- Product code: 3 files (`cleanprompt/_files.py`, `cleanprompt/_catalog.py`,
  `cython/_cache.py`), 1 README. Tests: 13 files and the new helper. Tooling: none.
- Shared with the documentation repository (`_sphinx_ext`), to be mirrored there: the six
  test files under `_sphinx_ai_assistant/tests/` and `tests/_platform.py`.

### Verification Evidence
- [x] `python -m libs._tools verify`, Linux, final tree (passed / failed / skipped):
      3.8 164/0/3, 3.9 194/0/3, 3.10 247/0/0, 3.13 261/0/0, 3.15 261/0/0.
- [x] Suites from the wheels (3.13): sphinx-ext 6186, cleanprompt 2365 (2360 before: five
      new tests), cython 1304 (1302 before, two new), mlflow 663, corpus 3537, annoy 793 and
      13, mcp 282, rank_bm25 174, `_cli` 153, logging 171.
- [x] sphinx-ext from the checkout, with its repository-only layout tests: 6310 passed.
- [x] The same wheels under `nl,mode,defpath,home,zip`: the 24 tests that still fail are
      exactly those that read `os.name` to skip on Windows (18 sphinx-ext, 5 cleanprompt,
      1 cython); the 29 mutants, the zip test, the 4 line-ending tests and the `~` test pass.
- [x] Tooling tests 696 passed; `python -m libs._tools check`: generated files up to date.
- [x] `precheck status=0` on the 18 files (end of file, trailing whitespace, codespell
      2.4.3, black 26.5.1; test files that black already flags at the head are not reformatted).
- [ ] Windows: not run here.

### Deviations from Plan
- A run number was written from memory into five files and corrected from the log header
  before delivery (Rule 42).
- No change to `promote_release.py`: its Git search through `os.defpath` is documented
  design and pinned by a test; reported as an observation instead.

### Technical Debt Created
- 20 tests do not run on Windows (15 release gate, 2 native status, 3 vault modes). Each
  names the fact that stops it. If the release tools are to work on Windows, that is a
  design task, not a test repair.
- The emulation lives outside the repository. It can become a `verify --emulate` option if
  you want it kept.

### Approval Checklist
- [x] Meets the acceptance criteria that can be met without Windows
- [x] No test deleted; no assertion weakened on POSIX
- [x] Documentation updated (README, `atomic_write`, `_default_cache_dir`, `_platform.py`)
- [x] Tests added (retry, cache directory branches, stored zip names)
- [ ] Windows leg green - next CI run
