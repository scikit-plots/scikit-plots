<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: Round 9 - `_sphinx_ext`: one stack, two import roots, 15 failures

### Context
- Symptom (your log): 15 failed, 21563 passed. 13 in the namespace guard
  (`_extension_setup.check_namespace`) and its three callers' `TestSetup`, 1 in the lazy
  registry, 1 in `test_submodule_independence` (`_validation.py`).
- Two supported layouts: `scikitplot._externals._sphinx_ext` (pip install) and a copy of the
  same tree at `docs/source/_sphinx_ext` (quick local testing). A third one in use,
  `docs/source/scikitplot/_externals/_sphinx_ext`, is examined below.

### Root causes
1. Guard disabled. `check_namespace` filtered its input with
   `[name for name in names if name in submodules]`, where `submodules` lists packages whose
   name starts with `"_sub"`: none exist, so every name was dropped, no root or retired name
   was ever found, and 13 tests that expect a refusal got none.
2. Why it had been disabled. The original guard took any name containing `_sphinx_ext.` as a
   stack root. `docs/source/conf.py` loads the docs' own helper package, also called
   `_sphinx_ext` (`_sphinx_ext.mpl_ext.*`, `.sklearn_ext.*`, `.skplt_ext.*`), next to the
   installed stack, so the original guard refused the real build: reproduced, "Mixed
   scikit-plots extension namespaces: _sphinx_ext, scikitplot._externals._sphinx_ext". The
   filter silenced that false positive by switching the guard off.
3. `_sphinx_rawfiles` was added on disk but not to `_OPTIONAL_PRIVATE_SUBMODULES`.
4. `scikitplot/mcp/_validation.py` had `__all__` exploded onto four lines; `corpus` and
   `rank_bm25` keep it on one. Both forms pass `ruff format`, so a hand edit, not the formatter.

### Fix
- `_extension_setup.py`: a name is a stack name only when the component right after a
  `_sphinx_ext` component is a declared member (`_CORE_PRIVATE_SUBMODULES |
  _OPTIONAL_PRIVATE_SUBMODULES`) or a retired one (`youtube_catalog`, `collection`,
  `_pydata_sphinx_theme`). New `split_stack_name()`. Messages unchanged. Module-level
  `pkgutil` scan and its import of the parent package removed.
- `__init__.py`: `_sphinx_rawfiles` added to `_OPTIONAL_PRIVATE_SUBMODULES`.
- `mcp/_validation.py`: back to byte-identical with the other two copies.
- `tests/test__extension_setup.py`: added the regression for cause 2 (docs helper package next
  to the installed root is accepted) and `TestSplitStackName`; file formatted, one composite
  assert split (both were already flagged in the original). No test removed or weakened.

### Evidence
- Before: the 14 extension failures reproduced in both layouts.
- After, installed layout, full `_sphinx_ext` suite against the untouched baseline: 105 -> 91
  failed, the 14 gone, no new failure. The 91 are this sandbox's (pytest-asyncio,
  pytest-regressions not installed; `_sphinx_ai_assistant` needs the repository root). With
  the plugins installed, `_sphinx_feedback`, `_sphinxcontrib_youtube`, `_sphinx_ai_learn`:
  416 passed, 71 skipped.
- `tests/` of the stack: 64 passed in both layouts; on Python 3.8: 39 passed (the two changed
  test modules); doctests 6/6 on 3.8 and 3.13.
- `test_submodule_independence.py`: 12 passed; the three sha256 copies match.
- Real `conf.py` extension list: accepted under the installed root; switched to `_sphinx_ext`:
  accepted; one entry left on the installed root: refused, naming both roots.
- `ruff check` / `ruff format --check`: clean on the 4 files.

### Layout C: `docs/source/scikitplot/_externals/_sphinx_ext` (not fixed in code)
`conf.py` puts `docs/source` first on `sys.path`, so `docs/source/scikitplot` competes with
the installed `scikitplot` for one name. Reproduced:
- without `__init__.py`: the installed copy is imported and the docs copy is ignored silently;
- with `__init__.py`: the docs stub becomes `scikitplot` and hides the whole library from
  autodoc.
The guard cannot see it: both copies have the same dotted name. Use `docs/source/_sphinx_ext`.

### Acceptance Criteria
- [x] All 15 have a demonstrated cause and a passing run
- [x] The docs build configuration that caused the band-aid passes the restored guard
- [x] No test deleted or weakened
- [ ] Full CI run green
