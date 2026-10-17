<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

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
