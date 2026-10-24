# Delivery, CI and generated configuration

Categories: Delivery, CI, Generated configuration, Labels, Labeler globs, Records

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 43: Delivery - run the repository's own hooks on the changed files before packaging
- **When:** a drop-in is about to be zipped.
- **Then:** run `precheck.sh`: one newline at the end of each file, no trailing blanks, codespell and black at the versions `.pre-commit-config.yaml` pins, on the files changed against the head.
- **Verified by:** `precheck status=0` in the results; the next commit on the branch does not touch the delivered files for formatting.
- **Added:** 2026-10-07. Root cause: round 5 was packaged without the hooks; your commit after it removed a blank last line from two files, parenthesised an expression black rewrites, and renamed a test value codespell read as a misspelling.

### Rule 49: CI - a tool's placeholder is read in the tool's source, for the mode in use
- **When:** a workflow uses a placeholder or variable a build tool fills in (`{project}`, `{package}`, `{wheel}`), and especially when the tool's input is not a plain checkout (an sdist, an archive, a subdirectory).
- **Then:** read what the pinned release sets it to in that mode, and write the comment from that reading; prefer a path the workflow itself controls (`github.workspace`) when the file comes from outside the tool's input.
- **Verified by:** the comment beside the placeholder names the source line it relies on.
- **Added:** 2026-10-08. Root cause: `{project}/dist/libs` assumed `{project}` was the repository; built from an sdist, cibuildwheel sets it to the extracted sdist, which has no `dist/libs`.

### Rule 67: Delivery - a rename ships with the deletion, and a check fails while the old file remains
- **When:** a change renames or removes a file and is delivered as an archive (a zip can add and replace, never delete).
- **Then:** list every removed path in `REMOVED_FILES.txt` at the archive root and in the reply; keep a fast check that fails while the stale file is present (`tools/maint_tools/check_test_module_names.py` for test modules).
- **Verified by:** `pr_check_test_module_names.yml` on every pull request; the archive's `REMOVED_FILES.txt`.
- **Added:** 2026-10-11. Root cause: round 29b renamed `_sphinx_feedback/tests/test_sphinx_build.py`; the archive could not delete the old file, it stayed in the repository, and the test shard and all nine partial-distribution legs failed at collection again.

### Rule 70: Generated configuration - derive it from the existing source of truth and gate it in CI
- **When:** a configuration file lists repository paths or names that another policy already owns (labels per package, release-note sections, distributions).
- **Then:** generate it from that policy with a `check`/`sync --apply` tool and a `.in` template; CI runs `check`, which fails on any stale file.
- **Verified by:** `pr_check_labeler.yml` (`generate_pr_labeler.py check`, `generate_towncrier_sections.py check`).
- **Added:** 2026-10-11. Root cause: `.github/labeler-by-module.yml` was written by hand beside the Towncrier owners; 20 of 45 packages had no label and one glob named a file that had become a package.

### Rule 71: Labeler globs - a negated glob is true for every other file
- **When:** writing `!glob` in an `actions/labeler` rule.
- **Then:** put it only next to a positive glob in `all-globs-to-any-file`, or alone in `all-globs-to-all-files` inside an `all:` group; never in `any-glob-*` or in an all-negated list in an `any` group.
- **Verified by:** `lint_label` in `generate_pr_labeler.py check`; `test_old_documentation_rule_matched_almost_everything`.
- **Added:** 2026-10-11. Root cause: the `documentation` rule OR-ed an all-negated option, so a change to `scikitplot/corpus/_base.py` was labelled `documentation`.

### Rule 72: Labels - one vocabulary, one registry, explicit grammar
- **When:** more than one tool maps titles or templates to labels.
- **Then:** keep one registry (`title_prefixes`) that generates every title rule; require an explicit terminator for case-insensitive prefixes; check that every label a template or workflow names exists.
- **Verified by:** `TitleTests` (36 titles, every token) and the referenced-label check in `generate_pr_labeler.py check`.
- **Added:** 2026-10-11. Root cause: three title vocabularies (`defect`, `00 - Bug`, `Documentation`) gave one prefix two labels; substring regexes labelled `DOCKER` as documentation; issue templates named labels nothing else used.

### Rule 92: Delivery - native, concurrent or provenance changes get an independent review before they ship
- **When:** a change touches native memory, thread interaction, a fallback state machine or provenance fields.
- **Then:**
  - Before delivery, a reviewer who has not seen the work probes the diff, read-only and with reproductions.
  - Each finding is fixed with a test that fails on the reviewed version, or is recorded with its reason.
- **Verified by:** the round's notes list the review, its 12 findings and each disposition.
- **Added:** 2026-10-12. Root cause: Run 1's own probes passed. The review then found 4 more crash or permanent-failure paths in the same change (`mprotect` race, `PROT_NONE` export, the chunker's load state, `float16` zeros) and 8 lesser issues.

### Rule 100: Delivery - a fragment's PR number is confirmed for the round, never inherited from the previous round
- **When:** creating `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst`.
- **Then:** use the PR number the user gave for this round; if none was given, ask once before the first fragment and record it in the round's active `todo/todo.md` (see `tasks/README.md`). Rename only this round's fragments (those absent from the uploaded tree); earlier rounds' fragments keep their number.
- **Verified by:** `find docs/source/whats_new/upcoming_changes -name "<PR>.*"` lists exactly this round's fragments, none of them in the uploaded tree; towncrier check PASS.
- **Added:** 2026-10-12. Root cause: round 33 reused round 32's PR 864 by assumption (it was recorded as a decision to reverse, not asked); the user corrected it to 865 before applying. Nine fragments renamed.

### Rule 124: Records - a work record lives in the plane of what it changed, and moves with a proof
- **When:** a plan, result or lesson is written, or records are moved between files.
- **Then:** write it where `tasks/README.md` says its owner keeps it (the active `todo/todo.md` of the plane that owns most of what it changes; a numbered rule in `skills/_engineering/rules/`); when a task is done, move it verbatim to that plane's `_maintenance/history/`. A move between files is its own commit: it records each part's source lines, length and sha256, and is proved by rebuilding the source from the parts byte for byte; edits come in a later commit.
- **Verified by:** `python -B maintenances/_repo/_maintenance/tools/check_records.py` (router only in `tasks/`, rules numbered once, indexes current, archives unchanged, links, budgets) and its mutation tests `maintenances/_repo/_maintenance/tests/test_records.py`; `check_records.py --prove <move commit>` rebuilds the moved sources from `git show` (for 4R, commit `4R-1` rebuilds all three files of 471e46c).
- **Added:** 2026-10-16. Root cause: all work went into one `tasks/todo.md` (197 KB) and `tasks/lessons.md` (108 KB) that every session read whole, while the per-submodule planes (`maintenances/<sub>/`, `skills/<sub>/`) were never updated: the corpus and mcp trackers still describe the runtime before round 33 (one says it is empty), and the Run 5 and 6 scope existed only outside the repository.

### Rule 132: CI - a required gate job runs whatever its needs did, and decides in its last step
- **When:** a workflow has a job that branch protection requires and that `needs:` other jobs (a report or result job over a matrix).
- **Then:** give it `if: always()` (plus any repository guard), make its last step fail unless every need succeeded, and let only the needed steps depend on success; never `if: success()` on the gate itself.
- **Verified by:** `report` in `.github/workflows/ci_codecov_test_coverage.yml` reads `if: always() && github.repository == ...`; its last step exits 1 when `needs.test.result` is not `success`.
- **Added:** 2026-10-23. Root cause: `report` was `if: success() && ...`; a failed shard made it *skipped*, and GitHub reports a job skipped by its condition as successful to a required check, so a failing run could pass branch protection. Its last step had the right check but could never run in that case; the commented-out `always()` above it shows the condition had been changed without that consequence.

### Rule 155: CI - a workflow runs only code its default branch already holds, never a ref typed into an input
- **When:** a workflow whose trigger can write the cache of the branch it runs on (`push`, `workflow_dispatch`, `schedule`) checks out, builds or runs a second revision: a base to compare with, a release to rebuild.
- **Then:** derive the revision from the repository (`git merge-base` with the default branch, the default branch's tip, a tag the workflow itself lists), offer the choice as a `type: choice` input, and check `git merge-base --is-ancestor <sha> <default tip>` before anything from it runs; never pass a `type: string` input to `ref:` or to `git fetch`/`checkout`. A job that builds a second revision keeps no cache (no `cache:` in setup actions, no `actions/cache`).
- **Verified by:** `grep -nE "ref: *\$\{\{ *(inputs|github\.event\.inputs)\." .github/workflows/*.yml` prints nothing; the CodeQL CLI with `codeql/actions-queries` (`actions-security-extended.qls`, a database of `.github/workflows`) reports no `actions/cache-poisoning/*` and no untrusted-checkout result, and reports alerts 231 and 232 on the tree before the fix (the check can fail).
- **Added:** 2026-10-24. Root cause: CodeQL alerts 231 and 232 on PR 867 (`actions/cache-poisoning/poisonable-step`): the benchmarks job of `.github/workflows/ci_performance.yml` checked out `inputs.base_ref`, any branch, tag or commit, a fork's `refs/pull/N/head` included, and ran `pip install ./base` and that revision's code, in a `workflow_dispatch` run that can write the cache of the branch it runs on, the default branch included, with `setup-python`'s pip cache in the same job. The repository's own workflow lint, zizmor 1.31.1, reports nothing on that file, so it could not catch this.
