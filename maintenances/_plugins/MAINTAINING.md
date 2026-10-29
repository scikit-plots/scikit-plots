# Maintaining the skills, agents and plugins of scikit-plots

Step-by-step procedures for every change to the Claude skills, agents and
plugins of this repository, written for someone doing it for the first time.
Read [`README.md`](README.md) first for what each piece is and why it is built
this way.

## TL;DR

```sh
# after changing a public API, a docstring, or the usage policy in pyproject.toml
python tools/maint_tools/generate_usage_skills.py sync           # users' skills
python -B maintenances/_repo/_maintenance/tools/check_records.py --write-index   # agents + indexes
git checkout -- maintenances/_repo/_maintenance/HISTORY_INDEX.md  # see gotcha G1
python tools/maint_tools/generate_pr_labeler.py sync --apply     # PR labels for new skills
# then validate everything: section 7
```

Never edit a generated file by hand: `plugins/scikitplot/skills/**`,
`plugins/scikitplot/agents/*.md`, `plugins/scikitplot/README.md`,
`plugins/scikitplot-dev/agents/*.md`, `plugins/INDEX.md`, the copies of a
shared skill (`plugins/*/skills/<shared>/`), `.github/labeler-by-module.yml`.
A shared skill is edited only in `plugins/_shared/skills/<name>/`.

## Contents

1. [One-time setup](#1-one-time-setup)
2. [Which procedure do I need?](#2-which-procedure-do-i-need)
3. [Change a public API or docstring](#3-change-a-public-api-or-docstring)
4. [Add, remove or exclude a public module](#4-add-remove-or-exclude-a-public-module)
5. [Add, rename or edit a developers' skill](#5-add-rename-or-edit-a-developers-skill)
6. [Try it locally in Claude Code](#6-try-it-locally-in-claude-code)
7. [Validate everything](#7-validate-everything)
8. [Test on claude.ai and Cowork](#8-test-on-claudeai-and-cowork)
9. [Release and submit to Anthropic's directory](#9-release-and-submit-to-anthropics-directory)
10. [The MCP server](#10-the-mcp-server)
11. [Gotchas](#11-gotchas)
12. [Troubleshooting](#12-troubleshooting)
13. [Review checklist for a pull request](#13-review-checklist-for-a-pull-request)
14. [References](#14-references)

## 1. One-time setup

| Tool | Why | Install | Check |
|---|---|---|---|
| A scikit-plots checkout | everything runs from its root | `git clone https://github.com/scikit-plots/scikit-plots` | `ls .claude-plugin/marketplace.json` |
| A Python environment with scikit-plots **built** and **all optional dependencies** | the generator imports every listed module | the project's development install | `python -c "import scikitplot; print(scikitplot.__version__)"` |
| `skills-ref` | the Agent Skills reference validator | `pip install skills-ref` | `agentskills --version` |
| Claude Code | `claude plugin validate`, local testing | <https://code.claude.com/docs/en/setup> | `claude --version` |

Run every command below from the repository root.

> **Tip.** Generate in one canonical environment (the CI test environment,
> Python 3.11): signatures and docstrings can print differently on another
> Python or with other dependency versions, and the drift test compares bytes.

## 2. Which procedure do I need?

| You changed | Do |
|---|---|
| A public function, class, `__all__` or docstring | [Section 3](#3-change-a-public-api-or-docstring) |
| Added or removed a public subpackage | [Section 4](#4-add-remove-or-exclude-a-public-module) |
| A `plugins/scikitplot-dev/skills/*-dev/SKILL.md`, or added a maintained area | [Section 5](#5-add-rename-or-edit-a-developers-skill) |
| A skill both audiences need (`plugins/_shared/skills/`) | Edit it there, then `check_records.py --write-index` (copies it into every plugin) |
| `plugin.json` or `marketplace.json` | [Section 7](#7-validate-everything), then [section 6](#6-try-it-locally-in-claude-code) |
| Nothing here, but CI says the skills are stale | [Section 3](#3-change-a-public-api-or-docstring), step 2 |
| Making a release | [Section 9](#9-release-and-submit-to-anthropics-directory) |

## 3. Change a public API or docstring

1. Change the code and its docstrings as usual. What the users' skill will
   say comes from: the module docstring's first paragraph, `__all__` (order
   matters: the first 40 names are shown), each name's first docstring
   sentence and signature, and the first `Examples` section.
2. Regenerate and check:

   ```sh
   python tools/maint_tools/generate_usage_skills.py sync
   python tools/maint_tools/generate_usage_skills.py check     # ok   usage skills (N files)
   ```

3. Regenerate agents and indexes (a description change can alter them):

   ```sh
   python -B maintenances/_repo/_maintenance/tools/check_records.py --write-index
   git checkout -- maintenances/_repo/_maintenance/HISTORY_INDEX.md
   ```

4. Review the diff of `plugins/scikitplot/` like code: it is what users'
   Claude will read.

## 4. Add, remove or exclude a public module

Every public top-level subpackage of `scikitplot` must be either listed or
excluded with a reason; `scikitplot/tests/test_usage_skills.py` fails
otherwise.

1. Edit `pyproject.toml`:

   ```toml
   [tool.scikitplot.maintenance.skills]
   usage_modules = [
     "scikitplot",
     "scikitplot.newmodule",        # gets plugins/scikitplot/skills/scikitplot-newmodule/
   ]

   [tool.scikitplot.maintenance.skills.usage_excluded]
   "scikitplot.oldmodule" = "declares no __all__"   # say why, in one line
   ```

   A listed module must import with its dependencies and declare `__all__`
   (an empty one is allowed for a package of subpackages).
2. Run section 3, steps 2 and 3. The generator writes the new skill and the
   records check its agent; a removed module's skill and agent are deleted.
3. Update the PR labels:

   ```sh
   python tools/maint_tools/generate_pr_labeler.py sync --apply
   ```

## 5. Add, rename or edit a developers' skill

Developers' skills are hand-written; their names are not.

1. Find the name the rule derives for the path (below `scikitplot/` or
   `maintenances/`):

   ```sh
   python - <<'PY'
   import runpy, pathlib
   r = runpy.run_path("maintenances/_repo/_maintenance/tools/check_records.py")
   derived = r["derived_skill_names"](r["mirror_keys"](pathlib.Path(".")))
   print(r["dev_skill_name"]("cexternals/_annoy", derived))   # scikitplot-cexternals-annoy-dev
   PY
   ```

2. Create `plugins/scikitplot-dev/skills/<that name>/SKILL.md`:

   ```markdown
   ---
   name: scikitplot-newmodule-dev
   description: "Maintain scikitplot.newmodule: what it owns and when to use this skill."
   metadata:
     scikitplot-mirror: "newmodule"
   ---

   # `newmodule` maintainer

   Read first: `maintenances/newmodule/MAINTAINING.md` ...
   ```

   Frontmatter rules: one `key: value` per line; only the specification's keys;
   **quote** any value that contains `: ` or ` #`.
3. Generate its agent and the index, restore the history index, update labels:

   ```sh
   python -B maintenances/_repo/_maintenance/tools/check_records.py --write-index
   git checkout -- maintenances/_repo/_maintenance/HISTORY_INDEX.md
   python tools/maint_tools/generate_pr_labeler.py sync --apply
   ```

4. Renaming: move the folder, set `name`, run step 3, and list the old paths
   in the change's removed files (Rule 67): a zip cannot delete them, and the
   records check fails while an old folder remains.

## 6. Try it locally in Claude Code

Two ways, both reading your working copy:

```sh
# a) one session only
claude --plugin-dir ./plugins/scikitplot      # the users' plugin
claude --plugin-dir ./plugins/scikitplot-mcp  # the opt-in MCP plugin
claude --plugin-dir ./plugins/scikitplot-dev  # the developers' plugin

# b) as an installed marketplace from this checkout: edits apply at the next
#    session or /reload-plugins, with no version change
claude plugin marketplace add "$PWD"
claude plugin install scikitplot@scikit-plots
claude plugin details scikitplot@scikit-plots   # inventory and context cost
```

Inside a session: `/plugin` (Installed, Errors), `/reload-plugins`, and ask
"which scikitplot skills do you have?". Messages use your plan's limits;
installs and `details` do not.

To try it as users will, from GitHub after pushing (in a throwaway home so
your own setup is untouched):

```sh
export HOME="$(mktemp -d)"
claude plugin marketplace add scikit-plots/scikit-plots --sparse .claude-plugin plugins
claude plugin install scikitplot@scikit-plots
claude plugin list        # Version: <12 characters of the commit>
```

## 7. Validate everything

Run before every pull request that touches skills, agents, plugins, a public
API or docstrings:

```sh
set -u; status=0
python tools/maint_tools/generate_usage_skills.py check                  || status=1
python -B maintenances/_repo/_maintenance/tools/check_records.py          || status=1
for d in plugins/*/skills/*/; do
  agentskills validate "$d" >/dev/null || { echo "invalid: $d"; status=1; }
done
claude plugin validate ./plugins/scikitplot --profile directory >/dev/null || status=1
for p in plugins/scikitplot-mcp plugins/scikitplot-dev; do
  claude plugin validate "./$p" >/dev/null                                || status=1
done
claude plugin validate . >/dev/null                                       || status=1   # the marketplace
python tools/maint_tools/generate_pr_labeler.py check | tail -1           || status=1
python -m pytest -q scikitplot/tests/test_usage_skills.py \
  tools/maint_tools/tests/test_generate_usage_skills.py \
  tools/maint_tools/tests/test_generate_pr_labeler.py                    || status=1
echo "overall: $status"     # 0 = everything passed
```

What each line proves:

| Check | Proves |
|---|---|
| `generate_usage_skills.py check` | users' skills equal what the installed package generates |
| `check_records.py` → every line `ok` | retired locations empty; layout, frontmatter, derived names, one skill per path or module, no name twice, agents, indexes and shared copies current (`skills`); marketplace entries match `plugins/<name>/` and set no `version` (`plugins`) |
| `agentskills validate` | each folder meets the open Agent Skills specification |
| `claude plugin validate --profile directory` | the manifest loads and the directory's local checks pass (README, license) |
| labeler check | PR labels know every skill and agent |
| pytest | drift test, policy completeness, generator unit tests |

Warnings you will see today and may keep: no `version` (on purpose, see
[section 9](#9-release-and-submit-to-anthropics-directory)) and no icon.

## 8. Test on claude.ai and Cowork

1. Build a zip of only the plugin folder:

   ```sh
   (cd plugins && zip -r ../scikitplot.zip scikitplot)
   ```

2. claude.ai → **Customize → Plugins → Add → Upload plugin** → choose the zip.
3. New chat: *"Which skills do you have from plugins?"*: expect the
   `scikitplot*` skills and the shared ones. Agents are not used in chat; that is expected.
4. A Cowork task: confirm the agents load.
5. Team testing: push, then each tester uses **Add marketplace** with
   `https://github.com/scikit-plots/scikit-plots`.

Source: [Plugin structure and testing](https://claude.com/docs/plugins/build).

## 9. Release and submit to Anthropic's directory

One workflow in this repository,
[`ci_claude_plugins_publish.yml`](../../.github/workflows/ci_claude_plugins_publish.yml),
publishes `scikitplot` to [`scikit-plots/skills`](https://github.com/scikit-plots/skills)
under the ref that was pushed here (Rule 166). That repository has no
workflow of its own and is never edited by hand outside `.gitignore`.

| Push here | Published in `scikit-plots/skills` | `version` |
|---|---|---|
| `main` | `main` | none (Claude Code versions it by commit) |
| `maintenance/X.Y.x` | `maintenance/X.Y.x` (created from `main` the first time) | none |
| tag `vX.Y.Z` or `vX.Y.Z.postN`, on `main` or on `maintenance/X.Y.x` | the same tag, never moved afterwards | `X.Y.Z[.postN]` |
| the same tag, when it is the newest release | also `stable` | `X.Y.Z[.postN]` |

The directory cannot validate this repository: its root `.gitattributes` and
its size (about 108 MiB archived, limit 50 MiB) both stop the portal
([pre-submission checklist](https://claude.com/docs/plugins/pre-submission-checklist.md)).
`scikit-plots/skills` is small, has neither, and is built, never edited. Its
default branch is `stable`, so the directory and a plain
`claude plugin marketplace add scikit-plots/skills` get the newest release,
while `@main` is the development build.

**One-time setup** (an organization owner):

1. Create a GitHub App owned by `scikit-plots` (Settings > Developer settings >
   GitHub Apps): no webhook; Repository permissions: **Contents: Read and
   write** (Metadata: read is implied), nothing else. Install it on
   `scikit-plots/skills` only.
2. In `scikit-plots/scikit-plots` > Settings > Environments, create
   `claude-plugins`; restrict it to the branches `main`, `maintenance/*` and the
   tags `v*`; add required reviewers if releases should wait for approval.
   In it set the variable `CLAUDE_PLUGINS_APP_CLIENT_ID` (the App's Client ID)
   and the secret `CLAUDE_PLUGINS_APP_PRIVATE_KEY` (a private key of the App).
3. In `scikit-plots/skills`: delete `.github/workflows/sync.yml` (the earlier
   pull workflow; the patch for that repository does this), and after the
   first release set the default branch to `stable` (Settings > General).
4. Actions > *CD 🧩 Claude plugins to scikit-plots/skills* > Run workflow on
   `main` with `publish` off: a dry run that builds and uploads the tree and
   pushes nothing.

**Each release:**

1. Merge everything, regenerate (TL;DR), and tag on `main` or on the release
   line's `maintenance/X.Y.x`: `git tag v0.5.0 && git push origin v0.5.0`.
   A tag on neither, or not of the form `vX.Y.Z[.postN]`, is refused.
2. The workflow builds that tag with `tools/maint_tools/build_skills_repo.py
   build --version 0.5.0`, pushes tag `v0.5.0` to `scikit-plots/skills`, and
   moves `stable` there when 0.5.0 is newer than what `stable` holds. A rerun
   with the same tree is a no-op; a tag that exists with another tree fails.
3. The directory picks up `stable`, scans it, and publishes it by the
   listing's publish setting ([submit a plugin](https://claude.com/docs/plugins/submit.md)).

Build and check by hand, from this checkout:

```sh
git clone https://github.com/scikit-plots/skills ../skills
python tools/maint_tools/build_skills_repo.py build --out ../skills --version 0.5.0   # a release
python tools/maint_tools/build_skills_repo.py build --out ../skills --unversioned     # a branch
python tools/maint_tools/build_skills_repo.py check --out ../skills --version 0.5.0   # ok, unchanged
(cd ../skills && claude plugin validate . && for d in skills/*/; do agentskills validate "$d"; done)
```

**Pin without `scikit-plots/skills`.** The development marketplace can be
added at any ref of this repository, for Claude Code only:
`claude plugin marketplace add scikit-plots/scikit-plots@v0.5.0 --sparse .claude-plugin plugins`.

**First submission** (once, after the first release is in `scikit-plots/skills`):

1. <https://claude.ai/directory/manage> > **Submit new** > **Plugin bundle**
   (Pro, Max, Team or Enterprise; on Team and Enterprise an Owner submits,
   [who can submit](https://claude.com/docs/directory/publish.md)).
2. **Repository** `scikit-plots/skills`; leave **Plugin path** empty; set
   **Branch or tag** to `stable` (or leave it empty once `stable` is the
   default branch); **Validate**. Following `main` gives the warning
   `MANIFEST_VERSION_MISSING`, by design: `main` is the unversioned
   development build. The `icon` notice (`UNKNOWN_KEY_CROSS_TOOL`) needs no
   action. A policy hold that names a file the build does not make means the
   tree was not replaced whole; rerun the publish workflow (CI-8).
3. Data handling: no personal data, nothing sent to other services, no
   retention, not aimed at people under 18 (the README states it).
4. Keep "GitHub push webhook", then **Submit for review**. The repository
   and folder cannot be changed after submitting.

## 10. The MCP server

The docs-search server `python -m scikitplot.mcp` is read-only; its tool
`search_docs` has a `title` and `readOnlyHint=True`.

- **Claude Code and local Cowork**: the opt-in plugin
  [`plugins/scikitplot-mcp/`](../../plugins/scikitplot-mcp/README.md) holds
  the `.mcp.json` that starts it. It is a plugin of its own so that the
  default `scikitplot` plugin stays Markdown only and runs nothing. Users need
  scikit-plots and the `mcp` package in the `python` on their `PATH`.
- **Other clients**: the bundles in `scikitplot/mcp/_plugins/` (Cursor,
  Windsurf, VS Code, Continue, Cline, Codex, OpenClaw) ship with the package.
- **claude.ai chat** (large change): host the server over HTTPS
  (`--transport streamable-http` behind an authenticated or rate-limited
  proxy; the server adds no authentication itself), add a second entry
  `{"type": "http", "url": "https://<host>/mcp"}`, and submit the server as a
  connector ([connector submission](https://claude.com/docs/connectors/building/submission.md)).

Never put keys or secrets in `.mcp.json`: every user receives the file.

## 11. Gotchas

| # | Gotcha | Handling |
|---|---|---|
| G1 | `check_records.py --write-index` also rewrites `HISTORY_INDEX.md`, and today that rewrite drops area names | Restore it: `git checkout -- maintenances/_repo/_maintenance/HISTORY_INDEX.md`, unless you meant to change it |
| G2 | The generator needs every optional dependency; a missing one stops it with "does not import" or "do not resolve here" | Install the extras; the drift test skips in such an environment rather than failing |
| G3 | Signatures differ across Python and dependency versions | Generate in the canonical environment ([section 1](#1-one-time-setup)) |
| G4 | Adding a package nearer the root can take a name a deeper skill holds (the naming rule) | The records check names the skill to rename; follow [section 5](#5-add-rename-or-edit-a-developers-skill), step 4 |
| G5 | Skill names are global across both sets and other projects | Keep the `scikitplot` prefix; the check rejects a name used twice |
| G6 | A plugin always loads its whole `skills/` folder | Never put developers' skills under `plugins/scikitplot/`; a skill for both goes in `plugins/_shared/skills/` |
| G7 | A top-level `bin/` in a plugin makes claude.ai and Cowork refuse it | Keep executables out of plugin roots |
| G8 | A pinned `version` freezes users on that copy | Only release tags have one: `build_skills_repo.py --version` refuses a development version, `--unversioned` (branches) sets none, and `check_records.py` (`plugins`) refuses one in this repository |
| G9 | A zip of the repository cannot delete renamed files | List them in `REMOVED_FILES.txt`, or apply with `git apply` |
| G10 | Unquoted `: ` in a description is a different YAML value (it broke the mlflow skill once) | Quote it; the records check rejects it |
| G11 | Agents' `skills:` preloading happens only in a live model session | Confirm by hand after changing agents: in a session, @-mention the agent (`@scikitplot:scikitplot-annoy`) and give it a task |
| G12 | Each enabled plugin adds context to every session (users' about 4,700 tokens) | Keep descriptions short; `claude plugin details` shows the cost |
| G13 | The portal says "Couldn't validate that repository" for scikit-plots/scikit-plots | Expected: its `.gitattributes` and size; submit `scikit-plots/skills` ([section 9](#9-release-and-submit-to-anthropics-directory)) |
| G14 | The publish job is skipped on a fork, on a branch other than `main` or `maintenance/*`, and on a dispatch with `publish` off | By design; push the ref to scikit-plots/scikit-plots, or dispatch with `publish` on ([section 9](#9-release-and-submit-to-anthropics-directory)) |

## 12. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `stale: plugins/scikitplot/skills/...` | API or docstring changed | [Section 3](#3-change-a-public-api-or-docstring) |
| `<module> declares no __all__` | Listed module without `__all__` | Declare `__all__`, or move it to `usage_excluded` |
| `plugins/scikitplot-dev/agents/<name>.md is out of date (run with --write-index)` | Skill changed, agent not regenerated | `check_records.py --write-index` (then G1) |
| `no skill has this agent` | Skill removed, agent left | `--write-index` removes it |
| `name '...' must equal its folder` | Folder and `name` differ | Rename one to match |
| `is named 'scikitplot-...' (Rule 164)` | Name not the derived one | Use the name in the message |
| `a folder here is a skill; no SKILL.md` | Old or empty folder under a skills directory | Delete it (`find skills -type d -empty -delete`) |
| `version: No version specified` | Expected on main | The released channel sets it ([section 9](#9-release-and-submit-to-anthropics-directory)) |
| `version '0.5.dev0' is not a release` | `build_skills_repo.py` given a development version | Build from a release tag with its version |
| `the checkout's generated plugin is not current` | Built from a stale checkout | Regenerate there (TL;DR), or build from the tag |
| `::notice::vX predates the released plugin` | The newest tag has no `build_skills_repo.py` (before 0.5.0) | Nothing to do until the next release |

## 13. Review checklist for a pull request

- [ ] No hand edits to generated files (TL;DR list)
- [ ] [Section 7](#7-validate-everything) prints `overall: 0`
- [ ] The diff of `plugins/scikitplot/` reads correctly: it is what users' Claude reads
- [ ] New or renamed skills: labels synced; removed paths listed (Rule 67)
- [ ] No secrets, keys or private URLs in any skill, agent or `.mcp.json`
- [ ] `HISTORY_INDEX.md` unchanged unless intended (G1)

## 14. References

| Topic | Source |
|---|---|
| Agent Skills specification | <https://agentskills.io/specification> |
| Plugin structure and testing | <https://claude.com/docs/plugins/build> |
| What each app loads | <https://claude.com/docs/plugins/platform-support> |
| Build for Claude, directory | <https://claude.com/docs/build/overview> |
| Connector submission | <https://claude.com/docs/connectors/building/submission.md> |
| Plugin manifest | <https://code.claude.com/docs/en/plugins-reference> |
| Marketplaces | <https://code.claude.com/docs/en/plugins/marketplace-reference.md> |
| Loading, versions, updates | <https://code.claude.com/docs/en/plugins/loading.md> |
| Skills in Claude Code | <https://code.claude.com/docs/en/skills.md> |
| Subagents | <https://code.claude.com/docs/en/sub-agents.md> |
| Install Claude Code | <https://code.claude.com/docs/en/setup> |
| Repository rules 164, 165, 67 | `plugins/_shared/skills/scikitplot-engineering-rules/rules/delivery.md` |
