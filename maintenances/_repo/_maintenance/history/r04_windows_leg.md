<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 4 - Windows leg, core/parts compatibility number, staging wheels for every platform

### Context
- Goal: (1) make the Windows verify leg green or explained (CI run 37555382019: 214 passed,
  13 failed, all in the parts' own tests); (2) judge a mix of versions instead of always
  flagging it; (3) build every distribution for every platform and upload it to the Anaconda.org
  staging index, the way `ci_wheels_conda.yml` does for the full distribution; (4) decide how
  users get a multithreaded Annoy build; (5) the five decisions and three next steps of round 3.
- Ground truth: the uploaded Windows job log, and the tree delivered in round 3.
- No Windows machine here. Every Windows cause below was reproduced on Linux by creating the
  condition that differs (Rule 18), or, for the two C++ orderings, by forcing the Windows order
  in a Linux build; what could not be created is marked.

### Findings and root causes
| Where | Symptom on Windows | Root cause | Evidence |
| --- | --- | --- | --- |
| corpus (28 errors, 2 legs + together) | `PermissionError: [Errno 13]` in `_atomic.py` | a directory was opened to `fsync` it; Windows cannot open a directory | emulated: 28 errors before, 30 passed after |
| sphinx-ext, cleanprompt, mcp, cython, corpus | `UnicodeDecodeError` / `UnicodeEncodeError` (cp1252) | `read_text()` / `write_text()` / `open()` without `encoding` use the locale's encoding (PEP 597) | static scan: 831 lines changed in the owned trees, 0 findings after; now a check of the harness |
| cython | a rooted path (`\\x`, `C:x`) accepted as "safe" | `is_safe_path` asked `os.path.isabs`, whose answer depends on the platform the code runs on | unit tests with both path flavours on Linux |
| cython | toolchain test picks up the runner's compiler | the test read the real toolchain | fake toolchain; 1297 passed |
| mlflow | `test_posix_*` fail; `'/tmp/x'` vs `'\\tmp\\x'` | tests of POSIX process groups ran on Windows; expected paths written as literals | 663 passed; 6 skipped under emulation |
| cleanprompt (46) | vault found under `%LOCALAPPDATA%`; empty output afterwards | fixtures isolated `XDG_STATE_HOME` only; on Windows the vault lives under `LOCALAPPDATA`, so every test shared the runner's real vault and the rest cascaded | the first failure names the real path |
| `_cli` | `OSError: [Errno 22]` when the reader closes the pipe | Windows reports a closed pipe as `EINVAL`, not `BrokenPipeError` | unit tests of `_is_closed_reader` |
| annoy, bundle | `PermissionError: [WinError 5]` on `os.replace(candidate, target)` | `save` memory-maps the file it wrote; a directory holding a mapped file cannot be renamed or removed on Windows | emulation of that rule: 5 failed on the old code, 15 passed on the new |
| annoy, C++ `save(p); save(p)` | `Unable to atomically replace target file: No error (0)` | the first save leaves the index mapped from `p`; a mapped file cannot be replaced. The text says errno 0 because `MoveFileEx` reports through `GetLastError` (ANNOY-WIN-002) | Linux build with `-DANNOY_REPLACE_REQUIRES_UNMAP=1`: 706 passed |
| annoy, C++ on-disk build | `Unable to truncate: Input/output error (5)` | `build()` shrank the file while a view of it was mapped (ANNOY-WIN-001). Before the `ftruncate` shim checked its result this failed silently and left the file too long | Linux build with `-DANNOY_SHRINK_REQUIRES_UNMAP=1`: 706 passed |

Found on the way, not Windows:
| Where | Symptom | Root cause | Evidence |
| --- | --- | --- | --- |
| sphinx-ext on 3.8, 3.9 | `_sphinx_youtube_gallery` cannot be imported | `VideoRecord | ChannelRecord` evaluated at import | hidden by an earlier `import yaml` failure in the base-dependencies probe; a second probe with the optional packages now runs |
| sphinx-ext on 3.8 to 3.10 | `import tomllib` | 3.11+ only | 25 files now fall back to `tomli` |
| sphinx-ext on 3.9, 3.10 | 18 tests: `'SpooledTemporaryFile' object has no attribute 'seekable'` | before Python 3.11 the spooled file lacks the queries `zipfile` asks | 18 failed on 3.10 in the harness; 51 passed on 3.10 after |
| sphinx-ext on 3.9 | 46 tests: `There is no current event loop in thread 'MainThread'`; 2: `name 'anext' is not defined` | the hosted services create `asyncio` locks in constructors, which 3.9 binds to a loop at creation; `anext` is 3.10 | with the service tests gated below 3.10: 4958 passed on 3.9 |
| sphinx-ext on 3.8 | four extensions raise `AttributeError: 'str' object has no attribute 'removeprefix'` | Python 3.9 API in `_sphinx_ai_assistant`, `_sphinx_youtube_core`, `_sphinx_llm` (3 places) | 63 failures in their suites on 3.8 before, 1313 passed after. Round 3 said "the Sphinx extensions themselves are not affected" on 3.8; that was an inference from "every module imports" and it was wrong |
| sphinx-ext, any Python | importing `dev_proxy.py` ends the interpreter (`SystemExit: 1`) | its start-up check for `HF_TOKEN` was a module-level statement | found by the probe with optional packages; now in `main()` |
| annoy tests | `Unable to write: No space left on device` | `test_very_large_index` wrote a 3.2 GB file into the installed package and never removed it; three environments at once filled the disk | 5 such files found; the test now uses `tmp_path` and removes the file |
| harness | a log record `ERROR    pkg.app: ...` named as a failed test | every line starting with `ERROR ` was read, not only pytest's short summary | unit test |
| annoy | the multithreaded code did not compile | `assert` used without `<cassert>` (ANNOY-MT-001) | GCC 13: `'assert' was not declared in this scope`; compiles and passes 706 tests after |
| annoy | `n_jobs` has no effect | no build defines `ANNOYLIB_MULTITHREADED_BUILD` | 4.4 to 4.7 s for n_jobs 1, 2, -1; identical files |
| `_distributions.report` | every mix of versions was a problem | equal versions was the only test | replaced by `CORE_API`, see below |

### Decisions taken (yours, from the round 3 questions)
1. NumPy: `scikit-learn>=1.3.0rc1` stays; NumPy 1.x installed afterwards is supported silently.
   Measured: annoy + rank_bm25 + corpus suites with NumPy 1.26.4 and scikit-learn 1.3.2 on
   Python 3.10: 4396 passed, 37 skipped, 1 xfailed. `report()` adds a *note* for NumPy 1 (why
   `pip check` complains) and a *problem* for the one combination that cannot work (NumPy 2 with
   scikit-learn < 1.4.2). `log_report()` writes both to the `scikitplot` logger.
2. Annoy threads: see "Threads" below.
3. sphinx-ext keeps Python 3.8. Its suite runs from 3.9, and the tests of the hosted
   services from 3.10 (new registry field `test_gated`): 3.9 runs 4958 tests instead of none
   or of 48 failures. On 3.8 the suite still cannot run as a whole (125 failures are the
   tests' use of Sphinx's `pathlib` fixtures, which the newest Sphinx for 3.8 does not have);
   3.8 is verified by importing every module, and by the suites of the four fixed extensions
   run by hand (1313 passed).
4. The eight AI-assistant test modules stay out; `libs/README.md` says which and why.
5. sphinx-ext has one extra per extension (`own_extras`), no catch-all.

### Core API number
- `CORE_API = 1` in `scikitplot/_distributions.py` names the contract between the core and
  the parts (what a part uses from the core, how the core reaches a part, the ownership map).
- Each part states it as an entry-point group, `scikitplot.parts.api1`, written by the
  generator; the core reads it from installed metadata without importing the part.
- `report()`: same number and different versions -> note; other number -> problem with the
  command that fixes it; no statement -> trusted only at the core's own version.
- The harness checks the statement in every wheel and again after installation.
- Chosen over "same 0.4 series" (a breaking change can land inside a series, and a series
  can pass without one) and over a NumPy-style C-API hash (there is no C API between the
  parts; the contract is three Python-level rules, so a number a person raises is the honest
  form).

### Threads (plan, and what is implemented of it)
Measured, Linux x86_64, 2 CPUs, 60 000 x 64, 24 trees, same seed:
| wheel | n_jobs | build | result |
| --- | --- | --- | --- |
| without the macro | 1, 2, -1 | 4.4 to 4.7 s | identical saved file every time |
| with the macro | 1 | 4.2 s | same neighbours as without |
| with the macro | 2 | 2.15 s | other trees (seed + thread number), stable across runs |
| with the macro | -1 | 4.6 s | as n_jobs=1 |

- Phase 1 (done): `SKPLT_BUILD_THREADS=1` compiles the multithreaded code into
  `scikit-plots-annoy` (`libs/annoy/setup.py`); unset or 0 leaves it out, as the Meson build of
  `scikit-plots` does; a WebAssembly target refuses 1. A test fails when a `meson.build` starts
  defining the macro, so the two builds cannot drift apart silently.
- Phase 2 (done, not run): nightly wheels are built with threads, staging (release) wheels
  without; each wheel's test asserts which it is. Users try it with the nightly index.
- Phase 3 (open, needs your decision): make it the default for both builds in one change:
  one Meson option defining the macro plus `dependency('threads')`, and the default of
  `SKPLT_BUILD_THREADS` flipped. Before that:
  - ANNOY-MT-002: with the macro and n_jobs=1, two files saved by one process differed in 36
    of 51 376 960 bytes (9 runs of 4 bytes inside nodes); queries were equal. Cause not found.
  - ANNOY-MT-003: `n_jobs=-1` is documented as "all cores" and ran on one thread: the C++
    `build()` reads -1 as "use the stored parameter". Changing it changes what a seeded
    default build produces on a multi-core machine.
  - Expose the flag without building an index (a module constant in `annoylib`).

### Implementation Steps
- [x] 1. Windows causes reproduced and fixed part by part (table above).
- [x] 2. Harness: full pytest output per run under `<outdir>/test-logs/`, every failed test
       named in the report; second import probe with the optional packages installed;
       static check for text I/O without an encoding; core API statement checked in the wheel
       and after installation.
- [x] 3. `CORE_API`, `parts_entry_points`, `declared_core_api`, `log_report`; the root package
       names installation problems when a present part fails to import.
- [x] 4. Generator: `scikitplot.parts.api<N>` entry points, `own_extras`, threads option,
       README sections (mixing versions, threads, tests not run).
- [x] 5. `annoylib.h`: `<cassert>`; unmap-first shrink and replace where a mapping forbids
       them; Windows error code in the replace message.
- [x] 6. `save_bundle`: release, swap, map; back into memory on failure.
- [x] 7. `ci_wheels_conda_libs.yml`: pure job, eight native platforms in two tiers, four
       Pyodide rows, one upload job. `ci_wheels_(test)pypi.yml`: `distribution` input.

### Acceptance Criteria
- [x] Every Windows failure of the log has a named cause and a reproduction or a forced-order
      build that passes.
- [x] `python -m libs._tools verify` green on Linux for 3.8, 3.9, 3.10, 3.13, 3.15.
- [x] zizmor: no finding on both workflows.
- [ ] Windows leg green (CI only). The next run writes full logs, so whatever is left is named.
- [ ] `ci_wheels_conda_libs.yml` run once by dispatch (CI only).

## Results Review - Round 4 - 2026-10-07

### Verification Evidence
`python -m libs._tools verify --python X.Y`, Linux x86-64, final tree, each run started after
the last edit:

| Python | passed | failed | skipped |
| --- | --- | --- | --- |
| 3.8 | 164 | 0 | 3 |
| 3.9 | 194 | 0 | 3 |
| 3.10 | 247 | 0 | 0 |
| 3.13 | 261 | 0 | 0 |
| 3.15 | 261 | 0 | 0 |

- Skipped: the compiled wheel on 3.8 and 3.9 (annoy needs 3.10); the mcp suite below 3.10;
  the sphinx-ext suite on 3.8.
- Own suites (3.13): `annoy` 693, `cexternals._annoy` 13, `corpus` 3530, `cleanprompt` 2341,
  `cython` 1297, `mcp` 282, `mlflow` 663, `rank_bm25` 174, `_cli` 146, `logging` 171,
  `_externals._sphinx_ext` 6186. sphinx-ext on 3.9: 4958 (services gated); on 3.10: 6186.
- Tooling tests: 691 passed. zizmor 1.30.1 (offline) on `ci_wheels_conda_libs.yml` and
  `ci_wheels_(test)pypi.yml`: no findings.
- Annoy, Linux, both orderings forced (`-DANNOY_SHRINK_REQUIRES_UNMAP=1
  -DANNOY_REPLACE_REQUIRES_UNMAP=1`) and the multithreaded code compiled in: 706 passed.
- NumPy 1.26.4 with scikit-learn 1.3.2, Python 3.10: annoy + rank_bm25 + corpus 4396 passed.

### Not verified here
- Windows. The causes are named and each fix was exercised on Linux (emulated condition or
  forced order), but no fix has run on Windows. `annoy_shrink_unmapped` and the unmap-first
  `save()` rest on one statement about Windows that this environment cannot test: that a
  file with no mapped view can be truncated and replaced. The next run keeps complete logs
  (`<outdir>/test-logs/`), so whatever is left is named test by test.
- 16 more calls in the vendored annoy tests save files next to the tests
  (`grep -n 'HERE}/' scikitplot/annoy/tests/*.py`); on Windows any of them may meet the
  mapped-file rule. Not touched: none failed in the uploaded log's visible part.
- `ci_wheels_conda_libs.yml` has not run. `tools/wheels/upload_wheels.sh` is not in the
  uploaded tree; the workflow assumes it uploads every file in `ARTIFACTS_PATH`, as
  `ci_wheels_conda.yml` does. Experimental rows (musllinux, Windows ARM64, Pyodide) are
  expected to show what fails.
- The multithreaded wheel was built and tested on Linux only (GCC 13).
- macOS leg: not rerun since round 3.

### Deviations from Plan
- The matrix was started three times. First run: two failures found (disk full from
  `test_big.annoy`; `dev_proxy` exiting on import). Second run: sphinx-ext on 3.10 and 3.9,
  whose suite ran there for the first time. One fix was applied while the second run was
  alive (Rule 22, third time); that run was discarded and the rule became a script.
- The default of `SKPLT_BUILD_THREADS` was first "on where the platform has threads" and was
  changed to "off" after measuring: parity with the Meson build, and two open findings.

### Technical Debt / decisions for the maintainer
- Threads, phase 3: ANNOY-MT-002, ANNOY-MT-003, a module constant for the flag, then the
  default for both builds.
- ANNOY-BUNDLE-PATH: a bundle's manifest records `on_disk_path` of the candidate directory,
  which no longer exists after publication. `load_bundle` does not read it;
  `from_json(manifest)` with `load=True` would fail on it.
- `save_bundle` now ends with the index mapped from the published file, and a failed save
  ends with the index in memory (`on_disk_path` None) even if it was file-backed before.
- `_pydata_component_list` still calls `importlib.resources.files` and `str.removesuffix`;
  its dependency `pydata-sphinx-theme` does not support Python 3.8, so the extension cannot
  be used there in any case.
- `_providers/artifact_output.py` also uses `SpooledTemporaryFile`; no test fails on 3.9 or
  3.10, so it was not changed.
- `dev_proxy.py` still calls `logging.basicConfig` when imported.
- A part requires `scikit-plots-skinny>=<its own version>`, so an installer upgrades the core
  with a part; a part newer than the core therefore needs `--no-deps` to exist. The core API
  check covers that case too.
- Shared Sphinx-extension stack, files changed this round (keep the documentation repository
  in step): every file under `scikitplot/_externals/_sphinx_ext/` in the delivery list. By
  kind (76 changed files and 1 new): `encoding="utf-8"` (64 files, mechanical); `tomllib` with `tomli` fallback (25);
  and by hand `_sphinx_ai_assistant/__init__.py`, `_sphinx_ai_assistant/dev_proxy.py`,
  `_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py`,
  `_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_workspace.py`,
  `_sphinx_llm/core/generator.py`, `_sphinx_llm/sphinx_llm/docref.py`,
  `_sphinx_llm/sphinx_llm/txt.py`, `_sphinx_youtube_core/video_options.py`,
  `_sphinx_youtube_gallery/model.py`, and the new
  `_sphinx_ai_assistant/tests/test_dev_proxy.py`.
