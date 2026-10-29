# Documentation: active tasks

Documentation changes made inside a submodule increment are planned with
that increment (for round 33:
[`maintenances/corpus/todo/todo.md`](../../corpus/todo/todo.md)).

## Task: PR 891 - docs build imports, galleries, API reference, citation, CI root cause (2026-10-28)

### Context
- Goal: five requests from one issue list, each verified on its own.
- Lead area `_docs` (API reference, galleries, docs build); pointers:
  `_build_utils` (generator), `_testing` (telemetry tests), `tools/citation`.

### Plan and results
- [x] 4. CI-only failure of 8 `test__perf_telemetry.py` tests. Reproduced:
      pytest 9.2.0 + pytest-asyncio 1.4.1 fail, 9.0.2 passes. Root cause: the
      tests' child sessions (`-W error`, empty `pytest.ini`) autoloaded every
      installed plugin; pytest 9.2 applies `-W` to `pytest_configure`
      warnings; pytest-asyncio warns there. Fix: hermetic children
      (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-p xdist.plugin`), the analyzer
      test asserts its child ran. Second, latent root cause found on Python
      3.13: one `subprocess.Popen` raised two audit events (`subprocess.Popen`,
      `os.posix_spawn`), so programs were counted twice; fixed by caller
      (`_called_from_subprocess`), regression test. Rule 169 added.
- [x] 0. Importability check for the docs build:
      `generate_apis_reference imports` (`_importability.py`): submodule
      calls, modules, names, deprecated names; errors vs unavailable optional
      dependencies; `--strict`, `--json`, `--module`. Root cause of the old
      `check` failing on a good build: a simple name that is a submodule
      (`experimental.enable_ann_imputer`) was "stale"; `_resolve_name` now
      resolves it as autosummary does. Pre-push hook enabled; CircleCI
      `build_scikitplot` runs it right after installing the build.
- [x] 1. Galleries `galleries/examples/ipymagic` (4 examples, the empty files
      provided) and `galleries/examples/env` (3 examples). Each step shows
      the magic/CLI and runs the Python function; cell magics become runnable
      cells in the notebook download.
- [x] 2. `scikitplot._ipymagic` in `apis_reference.py.in`; `rebuild --apply`.
- [x] 3. `tools/citation/citation_ref.py`: render/check/sync; two BibTeX
      entries (all versions, historic key first; this version with kind,
      build, commit tree, release DOI); CFF with `version`, `commit`,
      identifiers; `pr_check_citation.yml`, pre-push hook, make targets.

### Verification evidence
- telemetry tests: 29 passed + 2 checkout-only skipped installed, on pytest
  9.2.0 and 9.0.2 with pytest-asyncio 1.4.1; 31 passed in a checkout.
- `generate_apis_reference imports`: checked=1118 errors=0 unavailable=10
  (flask, spacy, nltk, cryptography absent); `check`: stale=0;
  `rebuild --check` clean; build-utils tests 46 passed.
- galleries: each script exits 0; sphinx-gallery build clean; generated
  notebooks executed in a real kernel (nbclient), cell magics included.
- citation: unittest 11 OK; `check` ok; `cffconvert --validate` valid;
  BibTeX parsed (2 entries).
- Towncrier check PASS; labeler check PASS; path lengths 0 failing;
  test module names unique; `check_records` ok.

### Deviations / not verified
- No Windows run; CircleCI and GitHub workflows not run here (YAML parsed,
  zizmor clean). The root `CITATION.*` record commit `b656b70` (2026-10-28),
  the build `scikitplot -V` reported; run `sync --apply` in the checkout to
  record the full hash.
- `.github/workflows/pr_check_citation.yml` existed (cffconvert action) and
  was first overwritten; restored and extended (lesson DOCS-L1).
- Follow-ups: `upcoming_changes/tools/citation/zenodo-cache-script-is-matplotlibs.md`,
  `upcoming_changes/scikitplot/config/runtime-citation-names-latest-only.md`,
  `upcoming_changes/galleries/examples/gallery-folders-outside-labeler-rel.md`.

## Pointers: past work owned elsewhere that changed the docs build

| Round | Subject | Archive (owner plane) |
|---|---|---|
| 9 | `_sphinx_ext`: one extension stack, two import roots (`docs/source/conf.py`) | [`r09_sphinx_ext_stack.md`](../../_externals/_sphinx_ext/_maintenance/history/r09_sphinx_ext_stack.md) |
| 29 | Page feedback on any site (`docs/source/conf.py`, `_page_feedback/`) | [`r29_page_feedback.md`](../../_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/r29_page_feedback.md) |
| 29b | Feedback for any deployment and storage | [`r29b_feedback_storage.md`](../../_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/r29b_feedback_storage.md) |
| 30 | Feedback forges: GitLab, Bitbucket, live-Space check | [`r30_feedback_forges.md`](../../_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/r30_feedback_forges.md) |

Every other archive, with the areas it names:
[`HISTORY_INDEX.md`](../../_repo/_maintenance/HISTORY_INDEX.md).
