# `scikitplot.annoy` maintenance: active tasks

Round 33 work that changes `scikitplot.annoy`. The campaign's status and
open decisions are in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md).
Completed tasks are archived under [`../_maintenance/history/`](../_maintenance/history/).

Rules that apply to every increment: [`skills/_engineering/`](../../../skills/_engineering/SKILL.md).

## Task: round 33 / Run 6 / 6c - the Annoy tests write nothing into the package (R6-4)

### Context
- Rule 30 names the class: "16 more calls in the vendored suite save small
  files next to the tests (`grep -n 'HERE}/' scikitplot/annoy/tests/*.py`)".
- Measured on 6a (`0a7dcec`), Python 3.12, editable build:
  - the suite without `index_test.py`: 199 passed, 1 skipped; it leaves
    `blah.ann`, `idx.ann`, `on_disk.ann`, `test_serialize.tree` in the
    package;
  - `index_test.py` downloads `test.tree` from GitHub **at import**, into
    the package; no test reads it (the reads are commented out). Without the
    network (HTTP 403 here) the module is a collection error, and pytest
    stops the whole directory ("Interrupted: 1 error during collection");
  - with the fixtures present, `index_test.py`: 23 passed, and it leaves 8
    files (`i/j/k.tree`, `t1/t2.ann`, `test.ann`, `test.annoy`,
    `something.annoy`);
  - `accuracy_test.py` downloads ann-benchmarks HDF5 files (hundreds of MB)
    into the package and builds `.annoy` files there; every test is
    `xfail("Test data unavailable.")`, which also hides a real failure.
- Every file under `scikitplot/` reaches the partial wheel
  (`libs/annoy/setup.py` `_package_data`), so a residue is shipped too.
- `tests/.gitignore` lists `!test.tree`, `!test64.tree`, `!test_v2.tree`
  **before** `*.tree`; the last matching pattern wins, so the negations do
  nothing and the fixture `test64.tree` (read by 9 tests, in the uploaded
  tree) was never tracked here.
- Base: 6a (`0a7dcec`).

### Design
- Writes go to `tmp_path`; reads of the committed fixture stay `HERE`.
- The import-time download is removed with its function: nothing reads
  `test.tree`, and a test module does no network access at import.
- `accuracy_test.py`: opt-in by `SKPLT_ANNOY_ACCURACY_DATA=<directory>`, the
  cache of the downloaded data and the built indexes (the convention of
  `scikitplot/conftest.py`: a costly test is opt-in, off by default).
  Unset: every accuracy test is skipped with that reason. Set (`~` and
  relative paths resolved): the data is downloaded once into that directory
  and the accuracy is asserted; the `xfail` marks go, except a strict one
  on `test_lastfm_angular`, which builds a 65-dimensional index from
  64-dimensional rows (`ValueError`). A cache file is written to a partial
  file of its own (process id and a random suffix; created by the writer,
  so it keeps the umask's mode) and renamed into place, so an interrupted
  or parallel write never stands for the data; the HDF5 file is closed.
- `.gitignore`: the one negation that names a committed fixture,
  `!test64.tree`, moves after the patterns; `test64.tree` is committed,
  byte-identical to the uploaded tree. `!test.tree` and `!test_v2.tree` go:
  neither is committed or read by a test, and the gallery writes
  `test_v2.tree` here.
- Guard, in a new `tests/conftest.py`: a snapshot of the test directory
  (name, size, modification time from `os.lstat`; a `.py` file by presence
  only, so editing a test during a run is not residue; `__pycache__`
  ignored) before and after each module (an autouse module fixture), and
  against the snapshot taken when pytest imported the conftest, before any
  test module (an autouse session fixture: writes at import time; it skips
  what a module guard already reported). A new or changed file fails the
  teardown with its name. One listing per module, never per test (the cost
  rule of `scikitplot/conftest.py`).
- `HERE` and `import os` go from the four modules that no longer read a
  fixture: a reintroduced `f"{HERE}/..."` write is a `NameError` there.

### Out of scope (recorded)
- `galleries/examples/annoy/plot_annoy_to_NPY_CSV.py` writes `test_v2.tree`
  into `scikitplot/annoy/tests`; the gallery examples write their outputs
  next to themselves. A gallery, not a test; a later task.

### Steps
- [x] `.gitignore` order; commit `test64.tree`
- [x] `index_test.py`: drop the download; 8 write sites to `tmp_path`
- [x] `angular`, `hamming`, `on_disk_build`, `serialize`: write sites to `tmp_path`
- [x] `accuracy_test.py`: opt-in data directory
- [x] `conftest.py`: the residue guard
- [x] fragment; review; negative proof; suites; gates; harness

### Acceptance
- `grep -n 'HERE}/' scikitplot/annoy/tests/*.py | grep -v ':\s*#'` names
  reads only (`test64.tree`, and `nonexists.tree`, which must not exist).
- After the whole annoy suite, `git status --short --ignored
  scikitplot/annoy` lists `__pycache__` only.
- Without the network, `index_test.py` collects and passes.
- The guard fails a module that writes into the directory (negative proof).

### Results review - 6c (2026-10-17)
- Annoy suite (Python 3.12, editable): 232 passed, 1 skipped (`h5py`
  absent); before: 199 passed, 1 skipped, plus `index_test.py` a
  collection error without the network. `git status --short --ignored
  scikitplot/annoy` after the run: `__pycache__` only (before: 12 files).
- Review 1 (11 findings, 0 High): Med - the guard missed writes at import
  time and rewrites of existing files; `test_lastfm_angular` fails on
  opt-in (65-dimensional index, 64-dimensional rows); two dead negations
  exposed a gallery output to `git add`. Low - a shared `.part` under
  xdist; the HDF5 file not closed; `~` not expanded; no end-to-end test of
  the fixtures; unused `HERE`; the todo's base hash. All fixed with tests,
  except the error attribution to a module's last test (by design: no file
  access per test).
- Re-review: 8 fixed, 3 accepted as designed or recorded; 7 new Low, fixed:
  one write reported by both guards (the session skips what a module
  reported); a test source edited during a run read as residue (`.py` by
  presence); `mkstemp` made the cache 0600 (the writer creates the
  partial); stale listing status on Windows (`os.lstat`); the inner pytest
  run not hermetic (empty `pytest.ini`, no plugin autoload); two record
  texts.
- Negative proof: the old test modules under the new guard: 5 modules
  error, each naming its files (55 passed, 5 errors). The final guard
  tests on the reviewed guard: 4 of 10 fail; on the first fixes: 1 fails
  (`.py` by presence), and 1 more without the single report.
- The opt-in accuracy path, with a stub `h5py` (no data here): unset ->
  4 skipped; not a directory -> `NotADirectoryError`; set -> download to a
  partial and rename, index built once and reused; an interrupted download
  leaves nothing; the cached files are 0644; the length mismatch raises
  `ValueError` and closes the file. The real ann-benchmarks data was not
  run (no network).
- The guard alone under Python 3.9, 3.13 and 3.13t: both kinds of write
  reported, 3 passed, 2 errors.
- Suites: 10015 passed + the baseline 15. Gates against 6a: clean (the
  binary fixture trips the text-only whitespace check; codespell `NotIn`
  pre-existing); labeler PASS; records ok.

- CI harness on an untouched snapshot (`snap6c`, identical to the commit):
  203 passed, 4 failed, 3 skipped; the failing ids are those of 6a (the 14
  DNS tests of the corpus legs). The annoy wheel ships 127 files (124 + the
  guard, its tests and `test64.tree`), each identical to the source tree.
  The harness builds the annoy wheel for 3.12 only, so the annoy tests ran
  in the editable build (above), not in the harness.
