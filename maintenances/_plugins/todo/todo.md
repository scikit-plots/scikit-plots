# `_plugins` maintenance: active tasks

Skills, agents, plugins and their release channels (`plugins/`,
`.claude-plugin/marketplace.json`, `.github/workflows/ci_claude_plugins_publish.yml`,
`tools/maint_tools/build_skills_repo.py`). The procedures are
[`../MAINTAINING.md`](../MAINTAINING.md); completed tasks are archived
verbatim under `../_maintenance/history/` (created with the first one).

## Task: CI-6 - two homes, one controller: centralize records, plugins and release (2026-10-27)

### Context
- Goal: one place to look for each kind of file, one control point for
  publishing, and room to grow by adding groups, not locations.
- Root cause (Rule 167): records lived in five places with three
  `MAINTAINING.md`; Claude material in five (root `skills/` and `agents/` as a
  plugin whose source was the whole repository, `plugins/scikitplot/`, a third
  Claude plugin in `scikitplot/mcp/_plugins/.claude-plugin/`, and
  `scikit-plots/skills`, whose pull workflow was copied there by hand and
  followed release tags only). A skill for both audiences
  (`scholarly-inquiry-cycle`) had no valid place: the checker failed it.
- Also found: the 20 `archives` problems were pre-commit's `end-of-file-fixer`
  trimming the final blank line of each history file, which changed the hash
  of the last archived part; not an edit of the archived text.

### Decisions (the maintainer's, 2026-10-27)
- Release: both mechanisms, the push from this repository the default;
  pinning this repository at a ref stays possible for Claude Code.
- MCP: both, an opt-in plugin `scikitplot-mcp` (default) and the bundles for
  other clients in the package.
- No redirects or stubs at the old paths: the retired locations fail the
  checker instead.

### Implementation
- [x] Records: `tasks/README.md` -> `maintenances/README.md` (the map);
  `docs/maintenances/MAINTAINING.md` -> `maintenances/MAINTAINING.md`;
  `docs/tasks/` -> area `maintenances/_docs/`; `plugins/maintenances/` -> area
  `maintenances/_plugins/`; `tasks/r4r_migrate.py` and the docs handoff into
  `maintenances/_repo/_maintenance/`; the towncrier check into `_docs`.
- [x] Plugins: `skills/` and `agents/` -> `plugins/scikitplot-dev/`;
  `docs/skills/` -> its README; `plugins/INDEX.md` for every plugin;
  `plugins/scikitplot-mcp/` from the package's Claude bundle (renamed from
  `scikitplot-docs`, no `version`); marketplace lists `./plugins/<name>` x3.
- [x] Shared skills: `plugins/_shared/skills/`, byte copies written and
  checked by `check_records.py`; the usage generator ignores them.
- [x] `scholarly-inquiry-cycle` reviewed: description says when to use it;
  `license`; toolkit moved to `references/toolkit.md` and linked; the false
  "Category / Count" table (architecture and design as question words)
  removed; an engineering mapping added.
- [x] Checker: `router` checks the retired locations; `archives` accepts a last
  part that lost only trailing newlines at end of file; new `plugins` check;
  budgets counted once per file. Contract checkers find the root by `plugins/`
  and still refuse runtime imports of `maintenances`, `skills` and `plugins`.
- [x] Release: `.github/workflows/ci_claude_plugins_publish.yml` (same-named
  refs, `stable` = newest release, tags immutable, App token, environment
  `claude-plugins`); `build_skills_repo.py --unversioned`.
- [x] Rules: 166 amended, 167 added; guides rewritten.
- [ ] Owner, once: GitHub App + environment `claude-plugins`; delete
  `.github/workflows/sync.yml` in `scikit-plots/skills` (its patch); dispatch
  a dry run; after v0.5.0 set that repository's default branch to `stable`.
- [ ] Tag v0.5.0; submit `scikit-plots/skills` in the portal.

### Acceptance criteria
- [x] `check_records.py`: every line `ok` (was 3 FAIL, 22 problems).
- [x] `test_records.py` and `test_build_skills_repo.py` pass on pytest 9.0.x and 9.1.x.
- [x] No other test file regresses against the baseline, run file by file.
- [x] `agentskills validate` on every skill folder; `claude plugin validate`
  on the marketplace, the three plugins, and a release build.
- [x] actionlint and zizmor clean; the publish step simulated against a local
  bare repository in seven cases.

### Results
- Evidence: `ci6-evidence.zip` of the delivery (command output for each
  criterion above).
- Not verified here: a run on GitHub (needs the App and environment), and
  `generate_usage_skills.py check` and `scikitplot/tests/test_usage_skills.py`
  (they import the built package); `current()` was checked with a stubbed
  import instead.

## Task: CI-8 - directory policy holds on scikit-plots/skills (2026-10-28)

- Observed (portal, `main` @ 8e0cad1): two policy holds,
  `MCP_FORWARDS_CREDENTIAL_ENV`: ".gitignore ... spells python-poetry.org and
  is given the installer's .pypirc", and the same read combined with the
  `scikitplot` skill's Python code; warnings `MANIFEST_VERSION_MISSING` and
  the `icon` notice.
- Root cause: `.gitignore` (and `LICENSE.txt`) in `scikit-plots/skills` were
  GitHub's repository template, outside what `build_skills_repo.py` owned
  ("left alone"); the publish step replaced only the managed paths, so files
  nobody built were published.
- Fix: the build owns a minimal `.gitignore`; `validate` rejects any
  top-level entry it did not build; the publish step replaces the whole tree
  but `.git`. Simulated over the real repository's files: `.github/`,
  `LICENSE.txt` and the template are gone, no file names `.pypirc` or
  `poetry`, `check` is `ok ... unchanged`, `claude plugin validate .` passes.
- The version warning is by design on `main`: submit with **Branch or tag**
  `stable` (MAINTAINING section 9).
- [x] Rerun the publish workflow on `main`, revalidate in the portal. Latest
  (24af6c0, 2026-11-09): the hold remains and is with a reviewer ("Content
  policy review"); its detail names two unrelated surfaces, Rule 77's
  `{**os.environ, ...}` in `rules/portability.md` and the `scikitplot`
  skill's example (`citation()`, which reads only local files). No change:
  the finding says a reviewer confirms unrelated parts.

## Task: CI-9 - the directory refused every build with LAUNCHER_REGISTRY_REDIRECT (2026-11-07 to 2026-11-09)

- Observed: the directory refused 246d056, 35a80cf, 05d1ab6, f43b8bd,
  8c8bd87, ae294df and 5b6faa3 (checked 7 to 9 Nov) with one blocking
  finding, `LAUNCHER_REGISTRY_REDIRECT` ("Package registry redirected") on
  `skills/scikitplot-corpus/SKILL.md`, each "no new findings, none fixed";
  8aee837 had passed on 30 Oct with nearly the same file.
- Root cause (Rule 170): the usage skills showed their docstring examples as
  `pycon` transcripts; the directory reads skill code as commands, and
  `>>> pipeline = f()` is a shell write to a file named `pipeline`. Why only
  corpus was named is not proven: its example is the only one that also sets
  `build_index=True` and calls `.source(...)`.
- Not the cause, though each removed a real defect: the install text with a
  staging index and an unpublished distribution (PR 912), the `<name>`
  placeholder in a shell block (PR 913), another marketplace's install
  commands in the README (PR 914), the icon (byte-identical in 8aee837).
- Fix: PR 915, `generate_usage_skills._as_python` (examples as plain Python,
  doctest's prompt rule, output as `# ` comments) and the `scikitplot.mlflow`
  docstring's CLI line moved to a bash code block (Rule 54). Guard:
  `build_skills_repo._check_code` refuses a code line that begins with `>`
  outside Python and a `python` block that does not parse (Rule 170).
- Evidence. Portal **Validate** (Rule 171 explains why it is not proof):

  | Branch, one change on 5b6faa3 | Validate | Directory scan |
  |---|---|---|
  | `main` @ 5b6faa3 (control) | passed | refused |
  | `probe/a-no-corpus-example` | passed | not run |
  | `probe/b-plain-python-examples` (= PR 915) | passed | 24af6c0: passed, 1 fixed |
  | `probe/c-no-corpus-api` | passed | not run |
  | `probe/d-no-index-source` | passed | not run |

  Validate and the scan read the same tree of 5b6faa3: 115 files in 52
  folders, 710.5 kB. 24af6c0 differs from 5b6faa3 only inside the 20 example
  blocks; `_check_code` refuses 178 lines of 5b6faa3 and none of 24af6c0.
- Results: 24af6c0 "Compared with 5b6faa3: no new findings, 1 fixed"; it is
  in review for the CI-8 hold. The ledger note
  `upcoming_changes/tools/maint_tools/claude-directory-refused-skills-build.md`
  is deleted (its acceptance criteria hold).
- Waiting items are ledger notes, followed from there:
  `upcoming_changes/tools/maint_tools/directory-review-credential-hold.md`
  and `upcoming_changes/tools/maint_tools/directory-listing-follows-main.md`.
- Open, for the maintainer: the submission tracks `main`, while Rule 166 and
  section 9 of `../MAINTAINING.md` intend `stable`, which exists only after
  the first release tag is built; change the tracked branch only when no
  publish request waits and the plugin is not with a reviewer (Rule 171).
- [ ] Delete the probe branches `probe/a-no-corpus-example`,
  `probe/b-plain-python-examples`, `probe/c-no-corpus-api` and
  `probe/d-no-index-source` in `scikit-plots/skills` (the session's GitHub
  access could not).

## Task: CI-10 - pull-request and CI workflow skills for developers (2026-11-09)

- Request (the maintainer): pull-request and CI skills, written fresh from
  the ideas of a public BSD-3 skill collection; then (same day) kept to the
  developers' plugin, since they serve contributors and maintainers, not
  people using the library.
- Done: five hand-written skills in `plugins/scikitplot-dev/skills/`, listed
  in `check_records.DEV_WORKFLOW_SKILLS` (no maintained path, no agent, not
  preloaded into the area agents): `scikitplot-pr-review-flow` (the process:
  orient, triage, understand, verify, findings, action, approved draft
  review, follow-up, with its gates), `scikitplot-pr-impact-brief`,
  `scikitplot-pr-walkthrough`, `scikitplot-pr-pending-review`,
  `scikitplot-ci-failure-forecast`. Markdown only, `gh` commands that run as
  written (Rule 170). Not in `plugins/_shared/` or the users' plugin, so
  `scikit-plots/skills` and its README are unchanged and the submission's
  Data handling answers stay true.
- Guard: `check_records.skills_using_github` derives from shell blocks which
  skills read or write GitHub, and the users' README states it; today the
  users' plugin has none ("No skill or agent makes a network call"). Tests
  in `test_records.py`, each failing under a mutation of the detection.
- Then (same day, the maintainer: newcomers and busy maintainers need the
  helpers too): four skills ship scripts, written fresh, Python 3.8, standard
  library plus `gh`, each optional beside the manual steps:
  `post_pending_review.py` (checks every comment against the diff, skips
  ones already posted, all or nothing, `--dry-run`, never submits),
  `worklist.py` (cached open pull requests, staged filter, seeded pick),
  `pr_context.py` (metadata, linked issues, public-API change from base to
  head, prior art), `ci_runs.py` (attempt-1 baseline, forecast scoring),
  `pr_paths.py` (owning area and kind of each changed file, generated files
  with their tool). References: the impact brief's `reference.md` (scales,
  counterfactual order, cost list) and the review flow's
  `references/gates-by-path.md` (which section-5 gates each path needs).
- Guard (Rule 167): `check_records.check_program_consent` fails a skill in
  any skills folder that ships a program and does not open with
  `## Programs (ask before each run)`. Tests: `test_records.py` (two),
  `test_workflow_skill_scripts.py` (25; nine mutations of the scripts each
  fail a test), run by `pr_check_records.yml`. The published build is
  byte-identical to main's.
- Found while testing: `pr_paths.area` gave `_plugins/todo` for a record
  under `maintenances/_plugins/todo/`; an area is now a top folder or the
  mirror of a package. Pre-existing, not changed here:
  `scikitplot-plugins-dev` has a shell block with `<skills checkout>`
  placeholders that `bash -n` rejects (developers' plugin, not published).
- Process slip, recovered: comparing builds, `git checkout origin/main -- .`
  overwrote the uncommitted edits in the working tree; restored from the
  dropped stash commit. Compare against main in `git worktree add`, never by
  checking main's files out over the working tree.
- Check against reality (same day, the maintainer asked for a last check):
  - `post_pending_review.py`: a rerun after a partial failure would have
    posted duplicates (comments in your pending review have no line, so the
    line-keyed dedupe missed them); the pending review and its comments are
    now read through REST and matched on file and text; a GraphQL failure
    stops with the list of comments already added; error numbers no longer
    shift.
  - `pr_context.py`: read the base and head with `git fetch origin`, wrong
    when `origin` is a fork; compared with the base branch tip instead of the
    merge base; reported "no API change" when the fetch failed; searched all
    of GitHub without `--repo`; looked up `other/name#4` in this repository;
    ignored changed defaults; and the search endpoint refuses a query
    without `is:pr` or `is:issue` (HTTP 422), which `check=False` hid. Now
    REST only (contents at the merge base and head, renames by their old
    name), `--repo` required, unreadable parts printed as "could not
    compute", merged pull requests marked.
  - `pr_paths.py`: `gh pr view --json files` stops at 100 files; now REST,
    paginated; a `.github/` file is kind `ci`.
  - `ci_runs.py`: an unfinished run counted its running jobs as failures and
    a cancelled job as a miss; now the newest completed run, cancelled jobs
    unscored, an unfinished run refused, a repeated job name keeps its worst
    conclusion; `--event` and `--branch`, because scikit-plots' Actions
    history held no scheduled run on 2026-11-09 (all 97 runs listed were
    pull_request, pull_request_target or push), though nine workflows
    declare weekly or monthly schedules.
  - `worklist.py`: a maintainer's own pull request is left out (the brief is
    for outside contributions); a pull request with more than 100 files is
    kept rather than dropped; a cache of another repository is refetched.
  - Examples named `nightly.yml`, which scikit-plots does not have; now
    `ci_wheels_conda.yml`.
  - Review flow: "Before you start" (first-review reading, quick or full
    depth, resume from each stage's visible trace).
  - Tests: 30; eleven more mutations, each failing a test. The REST paths
    were run against PRs 917 and 918 (`--dry-run` for the pending review);
    the GraphQL paths (adding to a pending review, the worklist query)
    cannot run in this session.
