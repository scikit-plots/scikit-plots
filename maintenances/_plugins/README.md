# Skills, agents and plugins of scikit-plots: developer overview

This page explains, from the beginning, how scikit-plots ships its Claude
skills, agents and plugins: what each piece is, where it lives, which file is
the source of truth, how files are generated and checked, and why it is built
this way. The step-by-step procedures are in
[`MAINTAINING.md`](MAINTAINING.md); the users' guide is
[`plugins/README.md`](../../plugins/README.md).

## TL;DR

- **Two homes.** Everything that ships to Claude is under `plugins/`, one
  folder per plugin; every record about it (plans, history, this guide) is
  under `maintenances/_plugins/` (Rule 167).
- **Three plugins, one marketplace.** `.claude-plugin/marketplace.json` (the
  `scikit-plots` marketplace) lists `scikitplot` (users, any project, Markdown
  only), `scikitplot-mcp` (users, opt-in: the local documentation MCP server)
  and `scikitplot-dev` (developers, this checkout), each as
  `./plugins/<name>`.
- **Names are derived, never chosen.** Users' skills are `scikitplot` and
  `scikitplot-<x>`; developers' are `scikitplot-<x>-dev`; each agent has its
  skill's name. `<x>` comes from the package path (Rule 164).
- **One source per file.** Users' skills come from each module's `__all__`
  and docstrings; agents and indexes from the skills; a skill every plugin
  carries lives once in `plugins/_shared/skills/` and is copied byte for byte.
- **Everything is checked.** `generate_usage_skills.py check`,
  `check_records.py` (`ok skills`, `ok plugins`), `agentskills validate`,
  `claude plugin validate`, the labeler check and the pytest files.
- **One controller, same-named refs.** `.github/workflows/ci_claude_plugins_publish.yml`
  here builds `scikitplot` and pushes it to
  [`scikit-plots/skills`](https://github.com/scikit-plots/skills) under the
  same ref: `main` → `main`, `maintenance/X.Y.x` → `maintenance/X.Y.x`, tag
  `vX.Y.Z` → tag `vX.Y.Z` (with `version`), and the newest release also to
  `stable` (Rule 166). That repository holds no workflow.

## Contents

1. [The pieces, explained](#1-the-pieces-explained)
2. [Repository map](#2-repository-map)
3. [How the files are made](#3-how-the-files-are-made)
4. [Naming](#4-naming)
5. [What loads where](#5-what-loads-where)
6. [Versions](#6-versions)
7. [Design: why it is built this way](#7-design-why-it-is-built-this-way)
8. [Roadmap](#8-roadmap)
9. [Rules that govern this](#9-rules-that-govern-this)
10. [Official references](#10-official-references)

## 1. The pieces, explained

| Piece | Format | Who reads it | Source of truth |
|---|---|---|---|
| **Skill** | Folder with `SKILL.md`: YAML frontmatter (`name`, `description`, optional `license`, `compatibility`, `metadata`, `allowed-tools`) and Markdown instructions ([Agent Skills specification](https://agentskills.io/specification)) | Any tool that reads Agent Skills; Claude loads the description always and the body when relevant | Users: the module's code. Developers: the hand-written `SKILL.md` |
| **Agent (subagent)** | One Markdown file with frontmatter `name`, `description`, `skills` (skills preloaded at start) and a system prompt ([subagents](https://code.claude.com/docs/en/sub-agents.md)) | Claude Code and Cowork; claude.ai chat ignores agents | Generated from its skill |
| **Plugin** | Folder with `.claude-plugin/plugin.json` and `skills/`, `agents/`, `.mcp.json` beside it ([plugin manifest](https://code.claude.com/docs/en/plugins-reference)) | Claude Code, Cowork, claude.ai | `plugin.json` (hand-written, small) |
| **Marketplace** | `.claude-plugin/marketplace.json` listing plugins and where they are ([marketplace reference](https://code.claude.com/docs/en/plugins/marketplace-reference.md)) | `claude plugin marketplace add`, claude.ai *Add marketplace* | Hand-written, small |
| **MCP server** | `python -m scikitplot.mcp` (stdio by default, `--transport streamable-http` available) | MCP clients: Claude through the `scikitplot-mcp` plugin, others through `scikitplot/mcp/_plugins/` | `scikitplot/mcp/` |

## 2. Repository map

```text
.claude-plugin/
  marketplace.json          marketplace "scikit-plots" (development channel): the three plugins below
plugins/                    everything that ships to Claude
  README.md                 users' guide (hand-written)
  INDEX.md                  generated: package path -> skills and agents of every plugin
  _shared/skills/           skills every plugin carries (one source; copied by --write-index)
    scholarly-inquiry-cycle/        structured inquiry on any question
    scikitplot-engineering-rules/   the numbered rules (no path, no agent)
    skill-creator/                  Anthropic's, plus a scikit-plots section: ask before each program run
  scikitplot/               plugin "scikitplot" (users; Markdown, plus skill-creator's programs, disclosed)
    .claude-plugin/plugin.json
    README.md               generated
    skills/scikitplot[-<x>]/SKILL.md, references/API.md    generated
    skills/<shared>/        generated copy
    agents/scikitplot[-<x>].md                              generated
  scikitplot-mcp/           plugin "scikitplot-mcp" (users; opt-in local MCP server)
    .claude-plugin/plugin.json, .mcp.json, README.md
  scikitplot-dev/           plugin "scikitplot-dev" (developers, this checkout)
    .claude-plugin/plugin.json
    README.md, sphinx_ext.md   how to pick a development skill
    skills/scikitplot-<x>-dev/       one per package or repository-wide area, metadata.scikitplot-mirror
    skills/<shared>/                 generated copy
    agents/scikitplot-<x>-dev.md     generated, one per dev skill with a path
maintenances/_plugins/      this guide, MAINTAINING.md, the plugins' todo and history
.github/workflows/ci_claude_plugins_publish.yml   builds and pushes scikit-plots/skills refs
pyproject.toml              [tool.scikitplot.maintenance.skills]: usage_modules, usage_excluded
tools/maint_tools/generate_usage_skills.py       writes the users' skills
tools/maint_tools/build_skills_repo.py           builds the scikit-plots/skills tree from a checkout
maintenances/_repo/_maintenance/tools/check_records.py   checks every plugin, writes agents, copies and indexes
tools/maint_tools/generate_pr_labeler.py          labels PRs, {skill} and {usage} satellites
scikitplot/tests/test_usage_skills.py             drift test against the installed package
```

## 3. How the files are made

```text
 sources                         generators                          artifacts                         checks
 ───────                         ──────────                          ─────────                         ──────
 module __all__ + docstrings ─┐
 scikitplot/_distributions.py ├─> generate_usage_skills.py sync ─> plugins/scikitplot/skills/ ─┐   generate_usage_skills.py check
 pyproject usage policy ──────┘                                                                │   test_usage_skills.py
                                                                                               │
 plugins/scikitplot-dev/skills/*-dev/SKILL.md (hand) ──────────────────────────────────────────┤
 plugins/_shared/skills/*/ (hand) ──────────────────────────────────────────────────────────────┤
                                                                                               ├─> check_records.py --write-index
                                                                                               │     plugins/scikitplot-dev/agents/*.md, plugins/scikitplot/agents/*.md,
                                                                                               │     plugins/INDEX.md, plugins/scikitplot/README.md,
                                                                                               │     plugins/*/skills/<shared>/ (byte copies)
                                                                                               │   check_records.py (ok skills)
 metadata of every skill ───────> generate_pr_labeler.py sync ──> .github/labeler-by-module.yml  generate_pr_labeler.py check
 plugin.json, marketplace.json (hand) ───────────────────────────────────────────────────────>  check_records.py (ok plugins),
                                                                                                  claude plugin validate, agentskills validate
 a checkout of main, maintenance/X.Y.x or vX.Y.Z ──> build_skills_repo.py ──> scikit-plots/skills, same ref   ci_claude_plugins_publish.yml
```

What a generated users' skill contains, all taken from the package:

- frontmatter: `name`, a `description` (the module's partial-distribution
  summary or its docstring's first sentence, then "Use when code imports
  ... or needs <first names>"), `license`, `metadata.scikitplot-module`;
- the docstring's first paragraph (reST titles, roles and footnotes made
  plain);
- install commands (`pip install scikit-plots`, plus the partial
  distribution that ships the module, from `_distributions.py`);
- "before writing code": check the version and the signature;
- the public API: names from `__all__` in order, kind, first docstring
  sentence (up to 40 rows; all rows with signatures in `references/API.md`);
- for a package with an empty `__all__`, its child packages, and whether each
  defines a Sphinx `setup(app)`;
- the first docstring example (`Examples` section).

## 4. Naming

| Set | Folder and `name` | Example | Metadata |
|---|---|---|---|
| Users | `scikitplot` (root), `scikitplot-<x>` | `scikitplot.annoy` → `scikitplot-annoy` | `scikitplot-module: "scikitplot.annoy"` |
| Developers | `scikitplot-<x>-dev` | `annoy` → `scikitplot-annoy-dev` | `scikitplot-mirror: "annoy"` |
| Cross-cutting | `scikitplot-engineering-rules` | | none |
| Agents | the skill's name, `.md` | `plugins/scikitplot-dev/agents/scikitplot-annoy-dev.md` | |

`<x>` is derived from the path below `scikitplot/` (or `maintenances/`):

1. the last part, leading and trailing `_` dropped, inner `_` spelled `-`
   (`feature_selection` → `feature-selection`);
2. if a path nearer the root already has that name, prepend the parent's part,
   then the grandparent's (`cexternals/_annoy` → `cexternals-annoy`, because
   `annoy` is taken);
3. if two paths differ only by a leading underscore, the private one gets
   `internal-` (`externals` → `externals`, `_externals` →
   `internal-externals`).

Hard limits from the specification: 1-64 characters, `a-z`, `0-9` and single
hyphens, not first or last, and the name equals its folder. That is why a
folder such as `plugins/scikitplot-dev/skills/_cli/` can never hold a valid skill.

## 5. What loads where

From [platform support](https://claude.com/docs/plugins/platform-support):

| Component | claude.ai chat | Cowork | Claude Code |
|---|---|---|---|
| Skills | loads | loads | loads |
| Agents | ignored | loads | loads |
| Remote MCP (`http`/`sse` URL) | Connectors tab, after the user connects | loads | loads |
| Local MCP (a command) | ignored | when the session runs on the user's computer | loads |
| Top-level `bin/` | refuses the plugin | refuses the plugin | loads |

Limits on claude.ai: 25 added marketplaces per account; 5,000 files and
200 MB per plugin. A plugin always loads its whole `skills/` folder, and a
manifest `skills` key only adds folders
([manifest reference](https://code.claude.com/docs/en/plugins-reference)).
That is why every plugin is its own folder under `plugins/`: a plugin at the
repository root would ship every skill to every user. A plugin also cannot
reach files outside its folder (installs copy only that folder, and a
relative `source` may not use `..`), which is why a shared skill is copied
into each plugin instead of referenced.

## 6. Versions

Resolution order
([loading reference](https://code.claude.com/docs/en/plugins/loading.md)):
`plugin.json` `version`, then the marketplace entry's `version`, then the git
commit of the installed folder. A pinned `version` keeps every user on the
cached copy until the string changes. Policy here:

| Channel (marketplace to add) | `version` | Users get |
|---|---|---|
| `scikit-plots/scikit-plots` (this repository, marketplace `scikit-plots`) | not set | each commit of the ref added (default `main`) |
| `scikit-plots/skills@main` | not set | the latest development build |
| `scikit-plots/skills@maintenance/X.Y.x` | not set | the release line `X.Y`, each fix as it lands |
| `scikit-plots/skills@vX.Y.Z` | `X.Y.Z` | exactly that release; never moves |
| `scikit-plots/skills` (default branch `stable`) and the directory | the newest release | the newest release; a newer tag moves it |

All `scikit-plots/skills` refs are written by one workflow in this
repository from the same-named ref here, so this repository is the single
point of control: a branch or tag is published by pushing it here.

A released plugin follows `pip install scikit-plots`: the newest PyPI release
(0.4.0.post11) has no `agent`, `cleanprompt`, `feature_selection`,
`levenshtein`, `mcp` or `rank_bm25`, so skills built from `main` would
describe modules those users cannot import.

## 7. Design: why it is built this way

**Goals.** Clear names that stay unique beside other projects' skills; one
skill per public module, for any project; developer skills that route to the
maintenance plane; no text that the code does not back; checks that fail
before a wrong file ships; installs that work in every Claude app that can
load them.

**Principles applied.**

1. *Single source of truth.* The package (`__all__`, docstrings,
   `_distributions.py`) for users' skills; the hand-written dev `SKILL.md` for
   developers'; everything else generated from those.
2. *Deterministic generation.* Same input, same bytes: set and frozenset
   defaults are printed sorted, memory addresses removed; a module without
   `__all__` is refused instead of guessed from `dir()`.
3. *Validate at three levels.* The repository's own checks (names, mirror,
   agents current), the open specification's validator (`agentskills`), and
   Claude Code's (`claude plugin validate`, `--profile directory`).
4. *Least privilege.* The users' plugin has no hooks, no MCP and no `bin/`:
   nothing runs when it loads. Its only programs are the shared
   `skill-creator`'s, disclosed in its README (derived from the files) and
   run only after Claude has told the user what each does and asked. Agents omit `tools`, so they inherit the tools available to
   subagents, still under the user's permission mode.
5. *Progressive disclosure.* Short descriptions always loaded; the API table
   when a skill loads; the full API only in `references/API.md`.
6. *Separate audiences.* Users' and developers' sets live apart, so neither
   ships to the wrong people; the MCP server is its own opt-in plugin, so the
   default plugin runs nothing.
7. *Centralized control, distributed groups.* One map (`maintenances/README.md`),
   one marketplace, one release workflow, one checker; inside them, one
   folder per plugin and one per maintenance area, each owning its files.
8. *Least privilege in CI.* The publish job runs no repository code and
   pushes with a GitHub App token scoped to `scikit-plots/skills`,
   `contents: write`, in a protected environment; a tag is never moved.

**Capabilities chosen, and those left out on purpose.**

| Capability | Used | Why |
|---|---|---|
| Skills | yes | Load on every surface; the right place for "how to use module X". |
| Agents | yes, generated | Specialists per module in Claude Code and Cowork, at no maintenance cost. |
| Commands | no | Skills already load everywhere; chat turns commands into skills anyway. |
| Hooks | no | Would run code on users' machines; nothing here needs it. |
| MCP server | yes, as the opt-in plugin `scikitplot-mcp` | Runs a local program, so it is never in the default plugin; works in Claude Code and local Cowork; chat needs a hosted HTTPS server ([roadmap](#8-roadmap)). |
| `bin/` | no | claude.ai and Cowork refuse a plugin that has it. |

## 8. Roadmap

| Next step | Effort | Reference |
|---|---|---|
| Done (CI-6): two homes, `plugins/<name>/` per plugin, shared skills, one publish workflow with same-named refs and `stable`, opt-in MCP plugin | - | [MAINTAINING section 9](MAINTAINING.md#9-release-and-submit-to-anthropics-directory) |
| Create the GitHub App and the `claude-plugins` environment; delete `sync.yml` from `scikit-plots/skills`; set its default branch to `stable` | small, once | [MAINTAINING section 9](MAINTAINING.md#9-release-and-submit-to-anthropics-directory) |
| Tag v0.5.0, let the workflow publish, submit `scikit-plots/skills` | small | [submit a plugin](https://claude.com/docs/plugins/submit.md) |
| Host the MCP server over HTTPS, then a remote `.mcp.json` entry and a connector submission | large | [connector submission](https://claude.com/docs/connectors/building/submission.md) |
| Evals: does a skill improve Claude's answers? `claude plugin eval` (each run is billed model use) | medium | [build overview](https://claude.com/docs/build/overview) |
| Declare `__all__` (and module docstrings) for `datasets`, `nc`, `stats`, `utils`, `neighbors`, `cexperimental`, `visualkeras` | per module | Rule 165 |

## 9. Rules that govern this

In `plugins/_shared/skills/scikitplot-engineering-rules/rules/delivery.md`:

- **Rule 164**: a skill is `<skills dir>/<name>/SKILL.md`, its name its
  folder, prefixed `scikitplot`, derived from its package path.
- **Rule 165**: every skill with a path or module has an agent of its name,
  generated; users' skills are generated from the package.
- **Rule 166**: one workflow here publishes `scikitplot` to
  `scikit-plots/skills` under the same ref; releases are versioned, tags never
  move, `stable` is the newest release.
- **Rule 167**: two homes, `maintenances/` for records and `plugins/` for what
  ships to Claude; retired locations stay empty; shared skills are copied.
- **Rule 67**: a rename ships with the deletion (archives cannot delete).

## 10. Official references

| Topic | Source |
|---|---|
| Agent Skills specification and validator | <https://agentskills.io/specification> |
| Build for Claude (plugins, skills, MCP, directory) | <https://claude.com/docs/build/overview> |
| Plugin structure and testing on each app | <https://claude.com/docs/plugins/build> |
| What each app loads | <https://claude.com/docs/plugins/platform-support> |
| Submit an MCP connector | <https://claude.com/docs/connectors/building/submission.md> |
| Plugin manifest reference | <https://code.claude.com/docs/en/plugins-reference> |
| Marketplace reference | <https://code.claude.com/docs/en/plugins/marketplace-reference.md> |
| Plugin loading, versions, updates | <https://code.claude.com/docs/en/plugins/loading.md> |
| Skills in Claude Code | <https://code.claude.com/docs/en/skills.md> |
| Subagents | <https://code.claude.com/docs/en/sub-agents.md> |
