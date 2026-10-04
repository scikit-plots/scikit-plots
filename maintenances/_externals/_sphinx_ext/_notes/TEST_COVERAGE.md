# Test coverage of the Sphinx extension stack

Measured on 2026-10-04 against `scikit-plots_6.zip`, CPython 3.11.16, Sphinx 9,
with the project's pytest options (warnings as errors, strict markers, INFO
logging) and branch coverage. Library checkout: the site-dependent
`_sphinx_ai_learn` tests skip here and run in the documentation repository, so
that package's figure is a floor. Code run in a child process is not counted.

## Before and after this round

| Unit | Statements | Lines before | Lines after | Branches before | Branches after |
|---|---|---|---|---|---|
| `_sphinx_ai_assistant` | 32827 | 70.7% | 70.7% | 57.7% | 57.7% |
| `_sphinx_ai_learn` | 3375 | 38.1% | 38.1% | 26.4% | 26.4% |
| `_sphinx_llm` | 2514 | 4.0% | 4.0% | 3.4% | 3.4% |
| `_sphinx_feedback` | 1912 | 79.9% | 79.9% | 76.3% | 76.3% |
| `_sphinx_youtube_gallery` | 1311 | 0.0% | 99.2% | 0.0% | 99.3% |
| `_sphinx_collection` | 595 | 17.9% | 99.3% | 14.4% | 99.0% |
| `_sphinx_youtube_core` | 446 | 35.3% | 100.0% | 7.1% | 100.0% |
| `_sphinxcontrib_youtube` | 390 | 56.2% | 66.9% | 34.4% | 39.8% |
| `_sphinx_gallery_grid` | 272 | 0.0% | 100.0% | 0.0% | 100.0% |
| `_sphinx_jinja_render` | 144 | 66.7% | 66.7% | 73.7% | 73.7% |
| `_sphinx_gallery_jupyterlite` | 111 | 35.1% | 35.1% | 0.0% | 0.0% |
| `_pydata_component_list` | 47 | 0.0% | 100.0% | 0.0% | 100.0% |
| `_search_variant.py` | 33 | 90.9% | 100.0% | 77.8% | 94.4% |
| `_ansi_sanitizer` | 22 | 0.0% | 100.0% | 0.0% | 100.0% |
| `_extension_setup.py` | 20 | 75.0% | 100.0% | 68.8% | 100.0% |
| `__init__.py` | 16 | 68.8% | 75.0% | 50.0% | 50.0% |
| **total** | 44035 | 60.9% | 66.5% | 49.6% | 55.8% |

Tests: 3 338 passed before (with six `_sphinxcontrib_youtube` build tests
erroring for want of `pytest-regressions`, which CI installs); 6 299 passed,
80 skipped, 5 expected failures after.

## What had no tests in CI

- `_sphinx_youtube_gallery`, `_sphinx_gallery_grid`, `_pydata_component_list`,
  `_ansi_sanitizer`: no test module at all.
- `_sphinx_collection`: one test file; `select.py`, `sections.py`,
  `_browser.py`, `_yaml.py` and `_presentation.py` were never imported by a test.
- `_sphinx_youtube_core`: covered only incidentally.
- The ten YouTube gallery checks under `maintenances/` are scripts that run at
  import and need `myst_parser`; `testpaths` is `scikitplot`, so CI never ran them.
- `_sphinx_gallery_jupyterlite/tests/test___init__.py` is an empty file.

## Defects found by writing the tests, and fixed

| Where | Defect | Fix |
|---|---|---|
| `_ansi_sanitizer` | One pattern treated every escape as a control sequence: `ESC M` + `Hello` gave `ello`; a window-title sequence left `y title` in the document | Pattern follows the ECMA-48 forms; `strip_terminal_controls` is public |
| `_sphinx_ext/__init__.py` | `_ansi_sanitizer`, `_sphinx_gallery_jupyterlite`, `_sphinx_llm` missing from the lazy registry | Registered; a test compares the registry with the directory |
| `_sphinx_collection/_yaml.py` | An alias bomb (479 bytes describing 10^10 values) and nesting reached through aliases passed every limit | Size, text and depth are measured as expanded, in time linear in the document |
| `_sphinx_collection/select.py` | `stars>100` matched `stars: unknown` and `stars: [1, 2]`; a NaN in a column made the sort depend on input order | Non-comparable values compare false, as the docstring said; only finite numbers are numbers |
| `_sphinx_collection/_browser.py` | A valueless `:collection-id:` or field list raised `AttributeError`, a traceback instead of a directive error | `ValueError` |
| `_sphinx_collection/setup.py` | Published `sk-collection.css` and `.js` had mode 0600 | Ordinary file mode under the umask |
| `_sphinx_gallery_grid/directive.py` | `link: javascript:...` was written into `href`; a line break in `title`, `link` or an option injected markup, including `raw:: html` | Item values are kept on one line; only http, https, mailto and relative links |
| `_sphinx_gallery_grid/directive.py` | An empty gallery, `:limit: 0` or a filter matching nothing produced a sphinx-design `ERROR` | Empty groups are not rendered |
| `_sphinx_gallery_grid/directive.py` | `env.app`, deprecated in Sphinx 9 and removed in 11 | `confdir` is recorded on the environment at `env-before-read-docs` |
| `_sphinx_youtube_core/reference.py` | `''` raised `IndexError`; `t=²` raised `ValueError`; `%0A` survived into an id; `@john.doe` was rejected; `/redirect?q=//evil.example/...` was reported as a YouTube channel | Each fixed at its source |
| `_sphinx_youtube_gallery/model.py` | A clock duration with a non-ASCII digit raised a bare `ValueError`; a bad `url` beside a valid `id` lost the `record N:` prefix | `CatalogError` with the record index |
| assistant `tests/conftest.py` | Importing the proxy application put a URL-redacting handler on the root logger for the whole pytest session, every package included | Removed after collection, installed for the assistant package only |
| assistant Redis tests | `"6379" not in output` fails by chance about 1 run in 340 (timestamp and thread-id digits) | Port matched as a whole number; five-digit fixture ports |
| YouTube `check_trackers.py` | "Core imports a sibling" counted `from ..` inside `tests/` | Rule is relative to the file's depth |

## Left as expected failures, for a decision

These three are marked `xfail(strict=True)`; each states behaviour the docstring
promises and the code does not give. If the current behaviour is the intended
one, change the test and the docstring instead.

1. `_sphinx_youtube_gallery/query.py`: `match-regex` anchors apply to the title
   and description joined, so `^about` does not match a description starting
   with "about" and `\d$` does not match a title ending in a digit.
2. `_sphinx_youtube_gallery/sync.py`: `watch?v=ID&list=RD...` (a video opened
   from a Mix) is rejected as per-viewer content; the docstring says it stays an
   exact video.
3. `_sphinx_youtube_gallery/model.py`: for a handle with capitals, a later
   record's display name is ignored, but used if that record comes first.

## Not changed, reported

- `_sphinx_ai_learn` reads `env.app` in about thirty places. It works on
  Sphinx 9 with a deprecation warning that `sphinx -W` does not see and stops
  working on Sphinx 11.
- `str.removeprefix` / `removesuffix` (Python 3.9) are used in seven modules of
  the stack while `requires-python` is `>=3.8`.
- `_sphinx_youtube_core`: `ftp://`, `file://` and `javascript://www.youtube.com/...`
  references parse (the canonical URL emitted is always https);
  `youtube.com.br` is not recognised.
- `_sphinx_collection`: a bare presence filter treats `0` as absent; a
  `title: null` renders as the text `None`.

## Largest remaining gaps

| File | Statements | Missing | Lines |
|---|---|---|---|
| `_sphinx_ai_assistant/_hf_spaces_proxy/app.py` | 3268 | 994 | 66.5% |
| `_sphinx_ai_learn/_pages.py` | 741 | 615 | 12.9% |
| `_sphinx_ai_learn/_materialize.py` | 1033 | 552 | 41.4% |
| `_sphinx_llm/sphinx_llm/txt.py` | 523 | 523 | 0.0% |
| `_sphinx_ai_learn/_publication.py` | 539 | 383 | 24.9% |
| `_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py` | 1191 | 380 | 63.0% |
| `_sphinx_ai_assistant/_hf_spaces_model/app.py` | 353 | 353 | 0.0% |
| `_sphinx_ai_assistant/_hf_spaces_proxy/_page_feedback/_service/app.py` | 344 | 344 | 0.0% |
| `_sphinx_llm/core/generator.py` | 328 | 328 | 0.0% |
| `_sphinx_llm/sphinx_llm/docref.py` | 313 | 313 | 0.0% |
| `_sphinx_ai_assistant/dev_proxy.py` | 272 | 272 | 0.0% |
| `_sphinx_ai_assistant/_hf_spaces_proxy/security/govern_release_history.py` | 994 | 259 | 69.5% |
| `_sphinx_ai_assistant/__init__.py` | 1573 | 250 | 81.4% |
| `_sphinx_ai_assistant/_hf_spaces_proxy/deduplicate_dataset.py` | 854 | 249 | 67.6% |
| `_sphinx_ai_assistant/_hf_spaces_proxy/security/preserve_release_history.py` | 848 | 247 | 66.5% |
