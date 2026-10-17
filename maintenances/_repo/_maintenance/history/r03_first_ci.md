<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

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
