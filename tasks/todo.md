# Active Tasks

## Task: Partial distributions of scikit-plots (`libs/<name>`), in the manner of mlflow-skinny

### Context
- Goal: publish focused parts of the one `scikitplot` import package as small distributions
  (`scikit-plots-skinny`, `-rank-bm25`, `-corpus`, `-annoy`, `-mcp`, `-cleanprompt`, `-cython`, `-mlflow`)
  that install alone or in any combination and form one coherent package.
- Rationale: `scikit-plots` needs a compiler toolchain and heavy dependencies; a user of one
  submodule should need neither unless that submodule does.
- Affected systems: `scikitplot/__init__.py`, `scikitplot/_cli`, new `scikitplot/_distributions.py`,
  `libs/`, `Makefile`, one new CI workflow, root `pyproject.toml` (four extras).

### Design (what / why)
- One owner per file. `scikitplot/_distributions.py` maps every shipped path to exactly one
  distribution. `scikit-plots-skinny` (core) owns the root `__init__.py`, `_cli`, `logging`;
  every other partial distribution owns only its subtree and requires the core. Because no file
  has two owners, any combination installs and any one uninstalls without touching the others.
- One source tree. `libs/<name>/setup.py` copies the owned files in for the build and removes
  them when the build exits (copy, not symlink: works on Windows without Developer Mode, and in
  an sdist). Editable installs are refused with an explanation.
- Generated metadata. `libs/_tools` writes each `pyproject.toml`, `setup.py`, `MANIFEST.in` and
  README from the ownership map, `libs/_tools/registry.py` and the root `pyproject.toml`
  (version, dependency specifiers by name, extras). `python -m libs._tools check` fails on drift.
- Runtime awareness. `scikitplot._distributions.report()` and `scikitplot doctor` say which
  flavor is installed (full / partial / source), which parts are present, and whether the
  installation is coherent; a missing part names the `pip install` that provides it.
- Names. Directory `libs/rank-bm25`, import `scikitplot.rank_bm25`, project
  `scikit-plots-rank-bm25`; every spelling (`scikit_plots_rank_bm25`, `scikit-plots-rank_bm25`,
  `Scikit.Plots.RANK.BM25`) is canonicalised (PEP 503) by the tooling and verified to install
  the same project.

### Implementation Steps
- [x] 1. Reproduce what breaks when only part of `scikitplot` is installed (baseline).
- [x] 2. Ownership map `scikitplot/_distributions.py` + tests.
- [x] 3. Root package and CLI work without the full distribution (lazy NumPy, no false warning,
       `dir()` and attribute access, actionable errors, `doctor`).
- [x] 4. `libs/_tools`: staging, registry, generator, CLI + tests.
- [x] 5. Generate the eight lib directories; remove placeholders.
- [x] 6. Compiled distribution (`scikit-plots-annoy`): both extension modules, same flags as meson
       minus `-march=native`.
- [x] 7. Verification harness (`python -m libs._tools verify`) + opt-in pytest plugin.
- [x] 8. Measure Python floors and lowest dependency versions; declare what was measured.
- [x] 9. Lint/format: library, tooling and generated files clean under the repository's ruff config.
- [x] 10. Makefile targets, CI workflow.
- [x] 11. Full matrix, Python 3.8 to 3.14.

### Acceptance Criteria
- [x] Each distribution installs alone into an empty environment and imports every module it ships.
- [x] All eight install together; uninstalling any one leaves the rest working.
- [x] No file is in two wheels; each wheel ships exactly the files its distribution owns.
- [x] Wheel built from the tree equals wheel built from the sdist; a build leaves nothing behind.
- [x] Pure-Python distributions need no compiler; only `scikit-plots-annoy` is compiled.
- [x] Each part's own test suite passes against the installed wheel.
- [x] Generated files are reproducible and lint-clean.

## Task: Round 2 - version-free declarations, scikit-learn 1.3+, Python 3.15, Sphinx extensions, WebAssembly

### Context
- Goal (maintainer's follow-up): declare requirements version-free; keep `scikit-learn>=1.3.0rc1`
  and make `scikitplot.annoy` work with it; confirm Python 3.14 and 3.15; say what happens when
  `scikit-plots` and partial distributions are mixed; make the Sphinx extensions installable;
  check Pyodide.
- Affected systems: `scikitplot/annoy/_mixins`, `scikitplot/_externals/_sphinx_ext` (3 files),
  `scikitplot/_distributions.py`, `libs/_tools`, `libs/sphinx-ext`, root `pyproject.toml`, CI.

### Implementation Steps
- [x] 1. `cleanprompt-*` extras and the tooling's own requirements version-free; tests state the
       rule (no pin, no cap; a floor only where a failure was measured).
- [x] 2. `annoy/_mixins/_vectors.py`: resolve `validate_data` (scikit-learn >= 1.6, else
       `scikitplot.utils.validation`, else none) and the `ensure_all_finite`/`force_all_finite`
       keyword once at import; remove the 1.6 floor override.
- [x] 3. Lowest-version run: only requirements that declare a floor; `lowest_constraints` for the
       one measured case where two root floors cannot be combined.
- [x] 4. Python 3.15 leg; CI matrix extended.
- [x] 5. Mixed installation measured (`scikit-plots` 0.4.0.post11 + partial 0.5.dev0).
- [x] 6. `scikit-plots-sphinx-ext`: ownership, registry, generated directory, test needs.
- [x] 7. Pyodide probe script and experimental CI job.
- [x] 8. Full matrix 3.8 to 3.15 on the final tree.

### Acceptance Criteria
- [x] No version range in anything this work declares, except lower bounds inherited from the root.
- [x] `scikit-plots-annoy` imports and passes its tests with scikit-learn 1.4.2 and with the newest.
- [x] Python 3.15: every distribution installs, imports and passes its tests.
- [x] `scikit-plots-sphinx-ext` installs with Sphinx alone and imports every module, 3.8 to 3.15.
- [x] Pure wheels import in Pyodide with no failure.

## Results Review - 2026-10-06

### Implementation Summary
- Files: 19 files under `scikitplot/` modified, 3 added, `libs/_tools` (15 Python files with
  tests) added, 8 lib directories regenerated, placeholder files removed (`tasks/removed-files.txt`), Makefile, 1 workflow.
- Deviation from the "simple" thresholds is deliberate: this is an architectural task.

### Verification Evidence
`python -m libs._tools verify --python X.Y` (build with X.Y, install into clean environments,
run every check), Linux x86-64:

| Python | passed | failed | skipped | note |
| --- | --- | --- | --- | --- |
| 3.8 | 120 | 0 | 2 | corpus, cython, mlflow refused by the installer; annoy not built (floor 3.10); mcp's tests need its `mcp` extra (3.10) |
| 3.9 | 145 | 0 | 3 | cython, mlflow refused; annoy not built; mcp's tests skipped as on 3.8 |
| 3.10 | 191 | 0 | 0 | mlflow refused |
| 3.11 | 203 | 0 | 0 | |
| 3.12 | 203 | 0 | 0 | |
| 3.13 | 203 | 0 | 0 | |
| 3.14 | 203 | 0 | 0 | |

Own test suites against the installed wheels (Python 3.13/3.14): `_cli` 141 passed (23 skipped by
rule: they need parts of the full distribution), `logging` 171, `rank_bm25` 174, `corpus` 3527,
`annoy` 647 + `cexternals._annoy` 11, `mcp` 282, `cleanprompt` 2341, `cython` 1240, `mlflow` 663.
Tooling tests: `pytest libs/_tools/tests` 477 passed. `scikitplot/tests/test__distributions.py`
102 passed. `ruff check` and `ruff format --check`: clean on every changed and generated file.

No regression: `scikitplot/tests/test___init__.py` in an environment without matplotlib fails 10
tests before and after (same 10; they need the full distribution's dependencies); `_cli` +
`logging` from the source tree went from 24 failed / 289 passed to 23 failed / 312 passed.

### Defects found by the verification and fixed at the root
- `import scikitplot` imported NumPy and warned "BOOM" whenever the compiled parts were absent.
- `dir(scikitplot)` and every lazy attribute raised when `scikitplot.api` was absent.
- CLI: `@dataclass(slots=True)` crashed on Python 3.8/3.9; a missing part was "Internal error".
- `scikitplot.rank_bm25`: `dict | dict` (Python 3.9) under a declared floor of 3.8.
- `scikitplot.annoy`: imported `scikitplot.externals._packaging` and `scikitplot.config` (other
  parts) from the Cython module; `X.__doc__ = ...` on a `typing.Union` fails on Python 3.14, so
  **`scikitplot.annoy` could not be imported on 3.14 at all** (full distribution included).
- CLI hints named four extras (`cleanprompt-ner`, `-nltk`, `-web`, `-crypto`) that did not exist.

### Deviations from Plan
- `libs/sphinx-ext/` was a placeholder with no owned files; removed rather than generated.
  (Superseded in round 2: it is now a real distribution.)
- corpus/cython/mlflow/annoy declare the Python floor that was measured, instead of being patched
  down to 3.8 (a first attempt showed further failures behind each fix).

### Technical Debt / decisions for the maintainer
- (Superseded in round 2.) Root `scikit-learn>=1.3.0rc1` was below what `scikitplot.annoy`
  imported; annoy now works from the oldest scikit-learn that runs with NumPy 2.
- Root `requires-python = ">=3.8"` is not true for corpus (3.9), annoy and cython (3.10), mlflow (3.11).
- Root extras (`mlflow`, ...) and `typing_extensions` (used by annoy, undeclared) have no ranges.
- Licence expressions in `registry.py` (annoy and rank-bm25 `BSD-3-Clause AND Apache-2.0`,
  cleanprompt `BSD-3-Clause AND MIT`) are read from the vendored LICENSE files; confirm.
- `requirements/*.txt` must be regenerated for the four new extras.
- Windows and macOS legs of the workflow are marked experimental: not run here.
- `scikit-plots-annoy` is not built under Python 3.8/3.9, so the installer's refusal of it there
  is declared (`Requires-Python`) but not exercised.
- Nothing is published: no release job was added for the partial distributions.

### Approval Checklist
- [x] Meets all acceptance criteria (Linux; Windows/macOS pending CI)
- [x] No new failures in existing tests
- [x] Documentation: generated READMEs, NumPyDoc in every new module
- [x] Tests added
- [ ] Maintainer CI run (final proof)

## Results Review - Round 2 - 2026-10-06

### Verification Evidence
`python -m libs._tools verify --python X.Y`, Linux x86-64, final tree, nine distributions:

| Python | passed | failed | skipped | note |
| --- | --- | --- | --- | --- |
| 3.8 | 142 | 0 | 3 | corpus, cython, mlflow refused; annoy not built; mcp and sphinx-ext suites skipped (stated reasons) |
| 3.9 | 167 | 0 | 4 | cython, mlflow refused; annoy not built; same suites skipped |
| 3.10 | 213 | 0 | 1 | mlflow refused; sphinx-ext suite skipped (runs from 3.11) |
| 3.11 | 226 | 0 | 0 | |
| 3.12 | 226 | 0 | 0 | |
| 3.13 | 226 | 0 | 0 | |
| 3.14 | 226 | 0 | 0 | |
| 3.15.0b4 | 226 | 0 | 0 | numpy 2.5.3, scikit-learn 1.9.1 |

- Own suites (3.15): `annoy` 665 (+18 new), `_externals._sphinx_ext` 6183, others unchanged.
- `scikit-plots-annoy [lowest]` on 3.10 to 3.12: numpy 2.0.0 + scikit-learn 1.4.2, all tests pass
  (the branch without `validate_data`).
- Tooling tests: 539 passed. `test__distributions.py`: 104 passed. ruff check/format: clean.
- Pyodide 314.0.7 (Python 3.14.2, emscripten/wasm32, no threads), eight pure wheels unpacked:
  `import scikitplot` works, flavor `partial`, 209 modules imported, 0 failed; the remainder
  need third-party packages that could not be downloaded here (numpy, docutils, sphinx, ...).
  `scikitplot --version` and `scikitplot doctor` run there with `click` installed.

### Measured: mixing the full and a partial distribution
Python 3.12: `pip install scikit-plots` (0.4.0.post11 from the index), then
`scikit-plots-annoy` 0.5.dev0 from the local build.
- The installer accepts it without a warning. 95 files under `scikitplot/` are then claimed by
  both. `scikitplot doctor` reports the overlap and the command that repairs it.
- `pip uninstall scikit-plots-annoy scikit-plots-skinny` removes `scikitplot/__init__.py` and
  `scikitplot/annoy/` although `scikit-plots` is still installed.
- Conclusion: full + partial is never a supported combination; nothing in package metadata can
  forbid it, so it is detected at run time.

### Defects found by the verification and fixed at the root
- Root floors `numpy>=2.0.0` and `scikit-learn>=1.3.0rc1` cannot be used together (see
  `lowest_constraints` in `libs/_tools/registry.py` for the measurements). Not changed in the
  root; reported.
- `scikitplot.annoy` required scikit-learn 1.6 at import (`validate_data`) and at call time
  (`ensure_all_finite`). Now works from 1.4.2 (the oldest that runs with NumPy 2).
- `annoy/_annoy/tests/test_sklearn_tags.py` asserted the 1.6 tags API unconditionally.
- Sphinx extensions: an example `conf.py` excerpt raised NameError when imported; one proxy
  module used `dataclass(slots=True)` (Python 3.10); one test depended on the `fork` start
  method and failed on Python 3.14+/Linux default (`forkserver`), macOS and Windows.

### Deviations from Plan
- First attempt at the lowest-version rule ("test the final release of a pre-release floor")
  rested on an unmeasured assumption and was removed (lessons, Rule 12).
- The first full matrix of this round was stopped and restarted from the first leg after a test
  file was edited while it ran (lessons, Rule 14).

### Technical Debt / decisions for the maintainer
- Root: the two floors above. Either `scikit-learn>=1.4.2` where NumPy 2 is required, or accept
  that the declared floor is unreachable on Python >= 3.9.
- `scikit-plots-sphinx-ext` declares only `sphinx`. What single extensions need beyond it is not
  declared as extras yet.
- Its test suite runs from Python 3.11; on 3.8 to 3.10 only imports are verified.
- Eight repository-bound test modules of the AI assistant are left out of the wheel's test run
  (`test_ignore`), by their own statement that they need `maintenances/`.
- annoy is compiled without `ANNOYLIB_MULTITHREADED_BUILD` in the full build and in the partial
  build alike (no meson file defines it), so index building is single-threaded everywhere.
- Mixed versions of partial distributions are reported by `doctor` as a problem; there is no
  compatibility contract between the core and the parts yet that would allow saying "fine".
- The Pyodide CI job is experimental: loading numpy etc. there was not possible to verify here.
- Files of the shared Sphinx-extension stack changed (keep the documentation repository in step):
  `_sphinx_jinja_render/_example_conf.py`,
  `_sphinx_ai_assistant/_hf_spaces_proxy/deduplicate_dataset.py`,
  `_sphinx_ai_assistant/tests/test___init__.py`.

### Approval Checklist
- [x] Meets all acceptance criteria (Linux; Windows/macOS/Pyodide-with-numpy pending CI)
- [x] No new failures in existing tests
- [x] Tests added (annoy compatibility, tooling)
- [ ] Maintainer CI run (final proof)

## Task: Round 3 - first CI run: zizmor, verify legs, full test job, Cython annotation report

### Context
- Goal: make every finding of the first CI run (run 37529331213, PR 849) green or explained:
  zizmor on the new workflow; verify legs ubuntu 3.8/3.9, windows 3.12, macos 3.12; the full
  test job (23 failed); and the Cython annotation report whose recorded path does not exist.
- Ground truth: `scikit-plots_1.zip` (the tree the CI ran) and the uploaded job logs.

### Findings and root causes
| Where | Symptom | Root cause | Evidence |
| --- | --- | --- | --- |
| zizmor | `adhoc-packages` on `npm install pyodide` | package installed outside a lockfile | zizmor 1.30.1 offline: 1 low before, none after |
| verify 3.8, 3.9 | `left: ['libs/annoy/scikitplot']` | the path is in the checkout before any build; it is only replaced and removed where annoy is built (3.10+). `/*/scikitplot/` in `libs/.gitignore` does not ignore a symlink, so the old layout's link can be committed | git demo: a link at that path is listed with the slash rule, ignored without; needs `git ls-files -s libs` on the real repository to confirm |
| verify windows | `annoylib.h(235): #error ... requires at least C++17` with `/std:c++17` | MSVC reports `__cplusplus` as 199711L unless `/Zc:__cplusplus` is given | compiler command line in the log; not runnable here |
| verify macos, sphinx-ext | `ValueError: '/private/var/..' is not in the subpath of '/var/..'` | `promote_release._release_files` returns resolved paths; four callers subtract an unresolved root | reproduced on Linux with `TMPDIR` behind a symlink: 1 failed before, 527 passed after |
| verify macos, cython | `_dedup_paths(..., drop=)` keeps the dropped path | `drop` compared as written against resolved entries | same symlinked-`TMPDIR` reproduction |
| verify macos, cython | "changing CC did not change the cache key" | the test "changes" `CC` to `clang`, which is already the value there | key unchanged when `CC` is preset to `clang`, changed with a fake value |
| verify macos, annoy | `test_float80_dtype_end_to_end[angular]`: first neighbour is 1 | test data: item 0 is the zero vector and the rest are colinear; all 12 angular distances from item 0 are 1.414214 (0 under `dot`), so the first of 12 ties is asserted | measured on Linux |
| verify macos, annoy | `('int8', 'float128')`: 2 neighbours instead of 3 | the leaf capacity was cast to the index type unchecked: 136 ids for int8 became -120; building then puts everything into one leaf and querying never recognises it (ANNOY-K-001). Not a macOS matter: there "float128" has the size of float64, and on Linux the same happens for int8 with float64 at f = 15..30 | reproduced on Linux: 6 of 18 new test cases fail on the old extension, 18 pass on the new |
| full tests (21) | `imported the wrong copy` / `module 'numpy' has no attribute 'int8'` | the child interpreter ran `site`; the editable install's import hook answers `import scikitplot` with the checkout before `sys.path` is searched | reproduced with a stand-in hook: 21 failed before, 24 passed after `-S` |
| full tests (1) | `module.PickleMode is literal` | the test compared by identity with a fresh `Literal[...]`; identity depends on a cache inside `typing` | identity is False after `typing._cleanups` on 3.11, 3.13, 3.14 |
| full tests (1) | `_get_feature_names(polars frame)` is `None` | polars 2.0.0 has no `DataFrame.__dataframe__` | measured with polars 2.0.0: 1 failed before, 11 passed after |
| cython | `meta["annotate_html"]` names a file that does not exist | recorded as an absolute path while the entry is still in its staging directory, which is renamed right after | reproduced: path under `.staging-...`, file in the final entry |

### Implementation Steps
- [x] 1. New baseline from `scikit-plots_1.zip`; logs read job by job.
- [x] 2. Pyodide: one directory per version with `package.json` + `package-lock.json`
       (0.28.3, 0.29.4, 314.0.0), `npm ci`, probe resolves Pyodide from the working directory.
- [x] 3. `libs/.gitignore` rule without the trailing slash; `node_modules/` ignored;
       verify reports a path that exists before the build as such, with the `git rm` to run.
- [x] 4. `/Zc:__cplusplus` for MSVC in the generated `setup.py`.
- [x] 5. `promote_release.py`: resolve the root wherever resolved paths are made relative.
- [x] 6. `cython._public._dedup_paths`: one normalisation for paths and `drop`.
- [x] 7. Tests corrected to their contract: toolchain key, float80 data, alias `__doc__`,
       isolated child interpreter.
- [x] 8. `utils.validation._get_feature_names`: polars branch.
- [x] 9. `annoylib.h`: leaf capacity capped at the largest value of the index type.
- [x] 10. Annotation report: entry-relative in `meta` for module and package builds alike;
       `BuildResult.annotation_html` / `PackageBuildResult.annotation_html` give absolute paths.

### Acceptance Criteria
- [x] zizmor reports no finding on the workflow.
- [x] Every failure with a root cause above is reproduced failing and shown passing.
- [x] A fresh build, a forced rebuild and a cache hit report the same, existing report file;
      a package build reports one per module.
- [ ] Windows and macOS legs green (CI only).

## Results Review - Round 3 - 2026-10-07

### Verification Evidence
`python -m libs._tools verify --python X.Y`, Linux x86-64, final tree:

| Python | passed | failed | skipped |
| --- | --- | --- | --- |
| 3.8 | 143 | 0 | 3 |
| 3.10 | 214 | 0 | 1 |
| 3.13 | 227 | 0 | 0 |
| 3.15.0b4 | 227 | 0 | 0 |

- Own suites (3.13): `annoy` 685 (+20), `cython` 1263 (+23), `_externals._sphinx_ext` 6183.
- Tooling tests: 544 passed. zizmor 1.30.1 (offline) on the workflow: no findings.
- Pyodide probe with the locked versions, standard library only: 0.28.3 and 0.29.4 (Python
  3.13.2) 194 modules imported, 314.0.0 (Python 3.14.2) 208; none failed. In CI, 314.0.7 with
  numpy, click, pyyaml, pydantic and docutils loaded: 306 imported, none failed.
- Full-test failures: 21 + 1 + 1 reproduced failing and passing (see the table above).

### Not verified here
- Windows: `/Zc:__cplusplus` removes the reported error; whether MSVC then compiles the rest
  is for CI to show.
- macOS: every failure of that leg was reproduced on Linux through the condition that differs
  and fixed there; the leg itself has not been rerun.
- That `libs/annoy/scikitplot` is committed is an inference; `git ls-files -s libs` shows it.

### Technical Debt / decisions for the maintainer
- ANNOY-K-001 changes the leaf capacity for narrow index types (uint8 with large vectors gets
  larger leaves than before), so an index file written by an older build with such a type must
  be rebuilt, not loaded.
- `meta["annotate_html"]` / `meta["annotation_html"]` are now relative to the cache entry, and
  `annotation_html` is a mapping for module builds too. A module entry written earlier holds an
  absolute string there and reports no annotation until it is rebuilt or hit through the build
  call (which rewrites its metadata).
- `test_add_build_query_new_index_types` still accepts fewer results than asked for
  (`0 < len(neighbors) <= N_RESULTS`), which is how ANNOY-K-001 stayed unnoticed.
- `docs/source/auto_examples/cython/plot_02_build_profiles.*` are generated copies of the
  example and were not edited.
- Files of the shared Sphinx-extension stack changed this round:
  `_sphinx_ai_assistant/_hf_spaces_proxy/security/promote_release.py`.

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

## Task: Round 8 - the annoy wheels: the test step cannot find scikit-plots-skinny

### Context
- Symptom (your log, `build_annoy`, manylinux x86_64, cp310): the wheel builds and is
  repaired; `CIBW_BEFORE_TEST` then runs `pip install --no-index --find-links
  /project/dist/libs scikit-plots-skinny` and pip answers "Location '/project/dist/libs' is
  ignored: it is either a non-existing path" and "No matching distribution".
- The pure job (`build --outdir dist/libs`, `twine check --strict`) is green.
- Head: `refs/pull/849/head` b745a35; the drop-in is a diff against it.

### Root cause
`{project}` is not the repository in this job. `package-dir` is the annoy sdist, and
cibuildwheel 4.1.0 (the pinned release, read from its source):
1. `__main__.py`: a `package_dir` that is a `.tar.gz` is extracted into a temporary
   directory, and the build runs after `contextlib.chdir(project_dir)` into it;
2. `platforms/linux.py`: `container.copy_into(Path.cwd(), "/project")`; the other platforms
   pass `project=Path.cwd()` / `"."` for the test step.

So `{project}` is the extracted sdist everywhere, and the sdist holds no `dist/libs` (its
top level: `LICENSE.txt MANIFEST.in PKG-INFO README.md README_ANNOY.md pyproject.toml
scikit_plots_annoy.egg-info scikitplot setup.cfg setup.py`). The workflow's comment said the
container "sees the repository ... as {project}": that premise was mine, and unverified.

### Fix
- `build_annoy`: `--find-links` names the host path of `dist/libs`. On Linux the build and
  test run in a container where cibuildwheel mounts the host's root at `/host`
  (`oci_container.py`: `--volume=/:/host` unless `disable_host_mount`), so the path is
  `/host${{ github.workspace }}/dist/libs`; on macOS and Windows the test runs on the host:
  `${{ github.workspace }}/dist/libs`.
- `build_annoy_pyodide`: same cause, same fix (`${{ github.workspace }}/dist/libs`); Pyodide
  builds and tests on the host.

### Evidence
- The cibuildwheel 4.1.0 source lines above.
- cibuildwheel's flow reproduced by hand: extract the sdist, change into it, install from
  `<extracted>/dist/libs`: "Failed to read --find-links directory ... No such file or
  directory"; from the host path of the built wheels: `scikit-plots-skinny 0.5.dev0`
  installed.
- The YAML as GitHub reads it (one line after folding): `pip install --no-index
  --find-links ${{ (matrix.buildplat[1] == 'manylinux' || matrix.buildplat[1] ==
  'musllinux') && format('/host{0}/dist/libs', github.workspace) || format('{0}/dist/libs',
  github.workspace) }} scikit-plots-skinny`.
- actionlint (expressions and contexts): no findings, before and after. zizmor 1.30.1: no
  findings.
- Not run: a container build (no Docker daemon here) and macOS, Windows, Pyodide.

### Acceptance Criteria
- [x] Cause read in the pinned tool's source and reproduced.
- [ ] `build_annoy` test step installs scikit-plots-skinny on every row (CI only).
