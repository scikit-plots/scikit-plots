# Delivery, CI and generated configuration

Categories: Delivery, CI, Generated configuration, Labels, Labeler globs, Records, Skills

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
- **Then:** write it where `tasks/README.md` says its owner keeps it (the active `todo/todo.md` of the plane that owns most of what it changes; a numbered rule in `skills/scikitplot-engineering-rules/rules/`); when a task is done, move it verbatim to that plane's `_maintenance/history/`. A move between files is its own commit: it records each part's source lines, length and sha256, and is proved by rebuilding the source from the parts byte for byte; edits come in a later commit.
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

### Rule 164: Skills - a skill is `<skills dir>/<name>/SKILL.md`, its name its folder, prefixed `scikitplot`, derived from its package path
- **When:** adding, moving or renaming a skill, or a package or maintenance area that has one.
- **Then:** follow the [Agent Skills specification](https://agentskills.io/specification): one folder directly under a skills directory, and the frontmatter `name` equal to it (1-64 characters of `a-z`, `0-9` and single hyphens, not first or last; an underscore is not allowed, so `_cli` can never be a skill's folder). Keep the frontmatter to the specification's keys, one line per value, double quotes around a value holding `: ` or ` #`. Every name starts with the project, `scikitplot`, so it stays recognisable and unique next to other projects' skills in any tool (only vendored skills such as `skill-creator` keep their own names). There are two sets. A *development* skill (`skills/`) records the path it maintains below `scikitplot/` and `maintenances/` as `metadata:` / `  scikitplot-mirror: "<path>"` and is named `scikitplot-<derived>-dev`; the cross-cutting `scikitplot-engineering-rules` has no path. A *usage* skill (`plugins/scikitplot/skills/`) records the module it explains as `metadata.scikitplot-module` and is named `scikitplot` for the root package and `scikitplot-<derived>` otherwise. `<derived>` is the path's last part with underscores spelled `-`; when a path nearer the root (or a public twin at the same depth) already has that, the parent's part goes in front, then the grandparent's; a path that still collides gets `internal-` in front (`externals`, `_externals` → `externals`, `internal-externals`; `annoy`, `cexternals/_annoy` → `annoy`, `cexternals-annoy`). Names are global across both sets. Code finds a skill folder through the metadata (`check_records.skill_mirrors`, `usage_mirrors`), never by spelling a path.
- **Verified by:** `python -B maintenances/_repo/_maintenance/tools/check_records.py` prints `ok   skills` (layout, frontmatter subset, prefix and derived names in both sets, one skill per path or module, no name twice, agents and indexes current; regenerate with `--write-index`); `pip install skills-ref` then `agentskills validate <folder>` prints `Valid skill` for every folder of both sets; `python tools/maint_tools/generate_pr_labeler.py check` passes (its `{skill}` and `{usage}` satellites read the metadata).
- **Added:** 2026-10-26. Root cause: the skills mirrored the package tree as folders (`skills/_cli/`, `skills/_externals/_sphinx_ext/_sphinx_feedback/`) and named themselves freely (`cli-maintainer`), so `agentskills validate` failed 30 of 31 ("Directory name '_cli' must match skill name 'cli-maintainer'"), and the name cannot be set to such a folder (`'_cli' contains invalid characters`). `mlflow`'s unquoted description (`scikitplot.mlflow: ...`) did not parse as YAML, and skill-creator's `quick_validate.py` checked neither. Spelling names from paths collides (`externals`/`_externals`, `utils`/`_utils`, `api`/`_api` among 170 packages), hence the derivation. Amended the same day: `<x>-maintainer` named no project, so outside this checkout it was unclear and could collide with other projects' skills; every name now starts with `scikitplot`, and the usage set was added, which takes the plain names users type (`scikitplot-annoy`), development taking `-dev`.

### Rule 165: Skills - every skill with a path or module has an agent of its name, generated; usage skills are generated from the package
- **When:** adding or changing a skill, a public module's `__all__` or docstrings, or the usage policy.
- **Then:** never hand-edit an agent: `check_records.py --write-index` writes `agents/<name>.md` for each development skill with a path (preloading it and `scikitplot-engineering-rules`) and `plugins/scikitplot/agents/<name>.md` for each usage skill (preloading it), both as Claude Code subagents (`name`, `description`, `skills`; no `tools`, so every tool is inherited), and removes agents whose skill is gone. Usage skills are written by `python tools/maint_tools/generate_usage_skills.py sync` from each module listed in `[tool.scikitplot.maintenance.skills] usage_modules` of `pyproject.toml`: its docstring's first paragraph (or its partial distribution's summary), the install commands, `__all__` in its order with each name's kind, signature and first docstring sentence, and the first docstring example; a public subpackage without one is listed in `usage_excluded` with its reason. A module without `__all__` gets no usage skill: what it does not declare is not promised. The two plugins are `plugins/scikitplot/` (users, any project) and the repository root (`scikitplot-dev`, maintainers), listed by `.claude-plugin/marketplace.json`; a plugin always loads its whole `skills/` folder, so development skills never go into the users' plugin.
- **Verified by:** `check_records.py` (`ok   skills`); `generate_usage_skills.py check` prints `ok` in an environment with every optional dependency (it imports each module); `claude plugin validate ./plugins/scikitplot` and `claude plugin validate .` pass; `scikitplot/tests/test_usage_skills.py`.
- **Added:** 2026-10-26. Root cause: the skills were maintainer workflows that point at `maintenances/` files, so they could not help anyone using scikit-plots in another project, and a plugin at the repository root would have shipped them to every user. Hand-written usage skills would drift from the code and could claim what the package does not do; generating them from `__all__` and docstrings keeps every sentence traceable to the package. Two generator runs first differed: a default's repr held a memory address (`<function _accept_any at 0x...>`) and a `frozenset` default followed string hashing; signatures now print such defaults in a fixed form.
