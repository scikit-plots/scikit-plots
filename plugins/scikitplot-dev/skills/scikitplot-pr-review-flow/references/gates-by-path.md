# Which checks a change needs, by the paths it touches

For stage 4 of `scikitplot-pr-review-flow`. Find each changed path's kind and
owning area first:

```sh
PR=123
python3 plugins/scikitplot-dev/skills/scikitplot-pr-walkthrough/scripts/pr_paths.py "$PR"
```

Then run every row whose paths the pull request touches. The authority is
section 5 of `.claude/new-chat-template.md`; this table only says which of
its checks apply. Run from the repository root.

| The pull request touches | Run |
|---|---|
| any file | `python -B maintenances/_repo/_maintenance/tools/check_records.py`; `python -B tools/maint_tools/check_path_lengths.py check` |
| `scikitplot/<area>/` (source) | that area's tests on the newest CPython and on Python 3.8 (`python -m pytest scikitplot/<area>`), with and without optional dependencies; its `_maintenance/check_trackers.py` when it has one; its probes under `maintenances/<area>/_maintenance/evidence/` |
| a test module (`test_*.py`) | `python -B tools/maint_tools/check_test_module_names.py` |
| a public name, signature or docstring of `scikitplot/` | `python tools/maint_tools/generate_usage_skills.py check` (needs the package with its optional dependencies; `sync` when stale) |
| `plugins/scikitplot/` or `plugins/_shared/` (published) | `python tools/maint_tools/build_skills_repo.py build --out DIR --unversioned`, then compare `DIR` with a clone of `scikit-plots/skills` to list what the merge publishes (Rules 166, 170) |
| `plugins/`, `maintenances/` | `check_records.py --write-index`, then `git checkout -- maintenances/_repo/_maintenance/HISTORY_INDEX.md` (gotcha G1) and confirm a second run changes nothing |
| `upcoming_changes/` (a folder appears or goes) | `python -B tools/maint_tools/generate_pr_labeler.py check`; `sync --apply` when it fails |
| `docs/source/whats_new/upcoming_changes/` (a fragment) | `python -B tools/maint_tools/generate_towncrier_sections.py --repo-root . check`; one bullet, named for the pull request that adds it |
| `libs/` | `python -B -m libs._tools check` |
| `scikitplot/cleanprompt/` | also `maintenances/cleanprompt/_maintenance/tools/generate_coverage_docs.py --check`, `python -m scikitplot.cleanprompt packs --check`, and `maintenances/cleanprompt/_maintenance/tools/output_digest.py --check` |
| `.github/workflows/` | after merge, the next scheduled run: `scikitplot-ci-failure-forecast` |

## Reading a result

- A check that also fails on the base branch is pre-existing: name it in the
  review, do not count it against the pull request.
- A generated file that differs from what its tool writes is a finding even
  when the tests pass: the source and the output have drifted.
- A gate that cannot run here (no compiled build, a missing optional
  dependency, the directory portal) is reported as not run, with the reason;
  it is not reported as passed.
