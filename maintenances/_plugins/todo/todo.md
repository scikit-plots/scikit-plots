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
- [ ] Rerun the publish workflow on `main`, revalidate in the portal.
