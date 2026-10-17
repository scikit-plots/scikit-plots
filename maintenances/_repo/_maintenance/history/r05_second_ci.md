<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 5 - second CI run: what the Windows leg still names, threads at run time, bundle manifest

### Context
- Goal: (1) every failure named by CI run 37637526602 (Windows verify leg, ubuntu 3.9 and
  3.10 verify legs, the full test job) fixed or explained; (2) your four answers: a run-time
  single / multi / auto mode for Annoy threads with build switches in `setup.py` and Meson,
  documented and logged; a bundle manifest that follows best practice; (3) and (4) closed.
- Ground truth: the uploaded job logs, and the pull request's head (`refs/pull/849/head`,
  957bf77), cloned read-only. It equals the round 4 delivery except two comment lines in
  `ci_wheels_(test)pypi.yml`. The drop-in is a diff against that commit.
- Not available: the artifact `libs-windows-latest-py3.12` with the complete test logs
  (`dist/libs/test-logs/`); it needs a GitHub login. The job log names one failure per run.

### Where the run stood
| Leg | Result |
| --- | --- |
| ubuntu 3.8, 3.11, 3.12, 3.13, 3.14, 3.15; macOS 3.12 | green |
| ubuntu 3.9, 3.10 | 1 failed each (sphinx-ext, one Node harness) |
| Windows 3.12 | 250 passed, 11 rows failed (was 13 rows with far more tests) |
| full test job | 2 failed of 23 116 |

### Findings and root causes
| Where | Symptom | Root cause | Evidence |
| --- | --- | --- | --- |
| harness | "50 failed" and one test named | on a CI runner pytest prints the whole failure message in the short summary; the reader stopped at the first line that named no test (my round 4 fix for the opposite mistake) | unit test with a multi-line reason; the log says "50 failed" and names one |
| harness | a test passes here and fails in the 3.9 and 3.10 jobs | tests that start `python` got the machine's Python (3.11 here), not the environment's | the Node harness below fails here once Python 3.10 is first on `PATH`; the harness now puts the environment first |
| sphinx-ext, ubuntu 3.9 / 3.10 | `test_ai_assistant__large_batch_manager_export_provenance.mjs` exit 1 | the harness runs `python -c "import tomllib"`; that module is 3.11+ | 29 passed, 1 failed with Python 3.10 first on `PATH`; 30 passed after (falls back to `tomli`) |
| sphinx-ext, Windows | `test_ai_assistant__git_patch_export.mjs` exit 1 | byte-exact comparison after `git am`; Git for Windows converts line endings by default | with `core.autocrlf=true` in a scratch `HOME`: 28 passed, 2 failed before; 30 passed after |
| full job | `test_test_layout`: "use tests._paths authorities" | my `test_dev_proxy.py` located the script with `parents[1]`; this stack forbids that. The test that says so needs the repository and is not in the wheel, so the harness never ran it | the eight repository-only modules now run here from the clone: 124 passed |
| full job | `test_failed_replace_keeps_a_loaded_index_usable`: `ResourceWarning: unclosed file` | `open(path).read()` in my test; the repository's `pytest.ini` turns warnings into errors, the harness does not use it | passes with `-W error` |
| `_cli`, Windows | `Internal error: OSError: [Errno 22]` on a closed reader | my round 4 proof was "flushing stdout fails again". A large write goes to the pipe unbuffered; nothing is pending afterwards and the flush succeeds | new rule: shape of the error (no file name, no Windows error code) and stdout is a pipe; unit tests for each branch. An inference, stated as one in the code |
| corpus, Windows | `PermissionError: [WinError 5]` in `test_contended_target_stays_consistent` | `os.replace` is refused while another process is replacing the same target | bounded repetition (15 attempts, about 1.2 s), Windows only; unit tests with injected refusals |
| corpus, Windows | `assert 'deadline_seconds' in []` | `elapsed > deadline` with a deadline of 0.0; the Windows monotonic clock ticks every 15.6 ms, so elapsed was 0.0 | with a 15.6 ms clock on Linux: 1 failed before, 24 passed after (`>=`) |
| annoy, Windows | `[WinError 32]` removing a file in `test_save_load` | the test removes a file two indexes still have mapped | indexes unloaded first |
| annoy, Windows | `Unable to open: Invalid argument (22)` in every second case of `test_int32_overflow_raises_overflow_error` | all cases share `HERE/on_disk.ann` inside the installed package; the previous case's index, kept alive by its traceback, still has it mapped | one file per test under `tmp_path`, unloaded in `finally` |
| cleanprompt, Windows (13) | `['llm', '-m', "'gpt 4o'"]` | `shlex.split(posix=False)` keeps the quotes of a quoted argument and splits `--opt="a b"`: every model command with a quoted argument was passed wrongly on Windows | with the Windows split forced on Linux: 22 failed on the old code, 2436 passed on the new |
| cython, Windows (16) | `cl.exe ... returned non-zero exit status 1` | the object file path is the cache path twice (339 characters in the log); MSVC's limit is 260 | objects are now written directly into `build/`; 1302 passed on Linux. The compiler's message is not in the log: the limit is the premise |
| mlflow, Windows | `Expected /.dockerenv ..., got ['\\.dockerenv']` | the test compares `str(Path)` | `as_posix()` |

### Not named by the log (need the artifact or the next run)
- sphinx-ext on Windows: 49 more failures; cleanprompt: any of the 13 that is not the split;
  cython: any of the 16 that is not the path; mlflow: 1 more.
- The next run names up to 40 per suite in the job log itself.

### Threads: what you asked for
- **Run time** (`scikitplot/annoy/_threads.py`): `SKPLT_ANNOY_THREADS` = `auto` (default),
  `single` or `multi`. One wheel compiled with threads serves all three; a wheel without
  runs `auto` and `single` on one thread and refuses `multi` with what to do.
  `threads_info()` reports the state. `build`, `fit`, `fit_transform` and `rebuild` of
  `Index` go through the rule.
- **Default kept**: a build without `n_jobs`, or with `-1`, runs on one thread in every
  wheel, as before.
- **Build time**: `SKPLT_BUILD_THREADS=1` (`libs/annoy/setup.py`) and the new Meson option
  `-Dannoy-threads=true` for the full distribution; both default to off and a test keeps the
  two defaults equal. `annoylib.MULTITHREADED_BUILD` says what a module was compiled with.
- **Reproducibility, measured** (2 CPUs, 60 000 x 64, 24 trees): one thread gives the same
  saved file, byte for byte, in a wheel with threads and in one without; two threads take
  1.1 s instead of 2.1 s, build other trees, and the file differs from run to run.
- ANNOY-MT-002 root cause: `_make_tree` copied a stack-allocated node, padding included,
  into the index. The padding (4 bytes after `a` for 8-byte ids) held stack contents:
  one constant per binary on the main thread, changing values on a worker thread. Fixed by
  zeroing the node; also keeps stack bytes out of saved files. A one-thread build in a
  module with threads now runs on the calling thread.
- ANNOY-MT-003: `n_jobs=-1` is no longer left to the compiled default: `auto` passes 1,
  `multi` passes the CPU count.

### Bundle manifest
- The manifest records no location (`on_disk_path` is null under `params` and `info`) and
  names its index member: `"bundle": {"format": 1, "index": "index.ann"}`.
- Why not the final path instead: a bundle is relocatable, and `on_disk_path` in metadata
  configures an on-disk *build* at that path, which truncates the file; a manifest naming
  its own index there invites `from_json` to destroy it.
- `load_bundle(index_filename=None)` takes the member name from the manifest; member names
  must be plain file names, so a manifest cannot point outside the bundle; an older manifest
  (no `bundle` section, a stale path) loads, with an INFO line.
- Logging on `scikitplot.annoy._mixins._io`: DEBUG per step, WARNING when a failed
  publication is undone.

### Implementation Steps
- [x] 1. Reporter: every summary line that names a test; `PATH` of the environment.
- [x] 2. Full job: `test_dev_proxy.py` through `tests._paths`; files closed.
- [x] 3. Node harnesses: `tomli` fallback; `core.autocrlf=false` in the scratch repository.
- [x] 4. `_cli`, corpus (two), annoy tests (two), mlflow test, cleanprompt split, cython
       object paths.
- [x] 5. `_threads.py`, `Index` methods, `annoylib.MULTITHREADED_BUILD`, Meson option,
       README and generator texts, workflow assertions.
- [x] 6. `annoylib.h`: zeroed split node; one thread on the calling thread.
- [x] 7. Bundle manifest, `load_bundle`, logging.

### Acceptance Criteria
- [x] Every failure the logs name is reproduced failing and shown passing, or its premise is
      stated.
- [x] One thread: identical file from both kinds of wheel.
- [x] Both annoy suites pass on a wheel with threads, without, with threads in `multi` mode,
      and with the Windows orderings forced.
- [x] `python -m libs._tools verify` green on Linux for 3.8, 3.9, 3.10, 3.13, 3.15.
- [ ] Windows leg green (CI only).
- [ ] Full build with `-Dannoy-threads=true` (CI only; the Meson lines were checked in a
      miniature project and parse in place, the full build was not run here).

## Results Review - Round 5 - 2026-10-07

### Verification Evidence
`python -m libs._tools verify --python X.Y`, Linux x86-64, final tree, each run started after
the last edit (a lock file now guards the tree, Rule 22):

| Python | passed | failed | skipped |
| --- | --- | --- | --- |
| 3.8 | 164 | 0 | 3 |
| 3.9 | 194 | 0 | 3 |
| 3.10 | 247 | 0 | 0 |
| 3.13 | 261 | 0 | 0 |
| 3.15 | 261 | 0 | 0 |

- Own suites (3.13): `annoy` 793 (+100), `cexternals._annoy` 13, `corpus` 3537, `cleanprompt`
  2360, `cython` 1302, `mcp` 282, `mlflow` 663, `rank_bm25` 174, `_cli` 153, `logging` 171,
  `_externals._sphinx_ext` 6186.
- Tooling tests: 696 passed. zizmor 1.30.1 (offline) on `ci_wheels_conda_libs.yml`: no
  findings.
- Annoy, both suites on Python 3.10, 806 passed in each of four builds: without threads;
  with threads; with threads and `SKPLT_ANNOY_THREADS=multi` (every default build on all
  CPUs); with threads and both Windows orderings forced.
- One thread, 60 000 x 64, 24 trees: the saved file has the same SHA-256 in the wheel with
  threads and in the wheel without, in 8 builds of 8.
- From a checkout, with the repository's warning policy: cleanprompt 2436, corpus 3538,
  mlflow 663, cython 1302 passed; the eight repository-only sphinx-ext modules 124 passed.
- Emulations: Windows command splitting (cleanprompt: 22 failed before, 2436 passed after);
  a 15.6 ms clock (corpus graph: 1 failed before, 24 passed after); `core.autocrlf=true`
  (Node harness: 2 failed before, 30 passed after); Python 3.10 first on `PATH` (Node
  harness: 1 failed before, 30 passed after).
- Meson: the option, the dependency on threads and both ways the arguments are passed were
  configured and compiled in a miniature project (meson 1.12.1), off and on; the three
  changed build files parse.

### Not verified here
- Windows: as in round 4, nothing has run there. New premises: MSVC's 260-character limit is
  why the 16 cython compiles fail; a refused `os.replace` on Windows is transient; `fstat`
  reports a pipe as a FIFO there.
- The full Meson build, with or without `-Dannoy-threads=true`, was not run here.
- Threads were measured on Linux with GCC only.
- About 60 Windows failures are not named by the log (see above).

### Deviations from Plan
- The artifact with the complete Windows logs could not be fetched (GitHub access was not
  granted to this session), so the unnamed failures stay unnamed for one more run.
- ANNOY-MT-002 was planned as "avoid the worker thread for one thread". That removed the
  run-to-run variation but the two kinds of wheel still wrote different files; the dump
  showed the cause (stack padding), which is what was fixed.
- The Windows quoting fix was first "strip enclosing quotes"; its own test showed that
  `--opt="a b"` still broke. The rule is now the POSIX lexer without an escape character.

### Technical Debt / decisions for the maintainer
- Flip both build defaults to "with threads" when you want releases to have them: nothing
  changes for callers who do not name `n_jobs`, and one thread gives the same file either
  way. The remaining cost is one more build variant to test on Windows and macOS.
- A build on several threads is not byte-reproducible (node order follows scheduling).
- `scikitplot.annoy.Index` now overrides `build`, `fit`, `fit_transform` and `rebuild`; the
  Cython binding (`scikitplot.annoy._annoy`) is not routed through the rule.
- 14 more calls in the vendored annoy tests save files next to the tests.
- `_providers/artifact_output.py` (`SpooledTemporaryFile`) and `dev_proxy.py`
  (`logging.basicConfig` at import) are as in round 4.
- Shared Sphinx-extension stack, files changed this round:
  `_sphinx_ai_assistant/tests/test_dev_proxy.py`,
  `_sphinx_ai_assistant/tests/_static/ai_assistant/test_ai_assistant__git_patch_export.mjs`,
  `_sphinx_ai_assistant/tests/_static/ai_assistant/test_ai_assistant__large_batch_manager_export_provenance.mjs`.
