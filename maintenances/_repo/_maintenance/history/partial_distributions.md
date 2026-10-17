<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

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
