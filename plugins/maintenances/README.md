# Skills, agents and plugins of scikit-plots: developer overview

This page explains, from the beginning, how scikit-plots ships its Claude
skills, agents and plugins: what each piece is, where it lives, which file is
the source of truth, how files are generated and checked, and why it is built
this way. The step-by-step procedures are in
[`MAINTAINING.md`](MAINTAINING.md); the users' guide is
[`../README.md`](../README.md).

## TL;DR

- **Two plugins, one marketplace.** `.claude-plugin/marketplace.json` (the
  `scikit-plots` marketplace) lists `scikitplot` (users, any project:
  `plugins/scikitplot/`) and `scikitplot-dev` (developers, this checkout: the
  repository root, `skills/` and `agents/`).
- **Names are derived, never chosen.** Users' skills are `scikitplot` and
  `scikitplot-<x>`; developers' are `scikitplot-<x>-dev`; each agent has its
  skill's name. `<x>` comes from the package path (Rule 164).
- **Almost nothing is hand-written.** Users' skills come from each module's
  `__all__` and docstrings; agents and indexes from the skills. Edit the
  sources, then regenerate.
- **Everything is checked.** `generate_usage_skills.py check`,
  `check_records.py` (`ok skills`), `agentskills validate`,
  `claude plugin validate`, the labeler check and two pytest files.
- **Two channels.** `plugins/scikitplot/` here is the development channel,
  unversioned, describing `main`. Each release tag is built into
  [`scikit-plots/skills`](https://github.com/scikit-plots/skills) with
  `version` set to the library version; that repository is the one the
  directory lists (Rule 166).

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
| **Plugin** | Folder with `.claude-plugin/plugin.json` and `skills/`, `agents/` beside it ([plugin manifest](https://code.claude.com/docs/en/plugins-reference)) | Claude Code, Cowork, claude.ai | `plugin.json` (hand-written, small) |
| **Marketplace** | `.claude-plugin/marketplace.json` listing plugins and where they are ([marketplace reference](https://code.claude.com/docs/en/plugins/marketplace-reference.md)) | `claude plugin marketplace add`, claude.ai *Add marketplace* | Hand-written, small |
| **MCP server** | `python -m scikitplot.mcp` (stdio by default, `--transport streamable-http` available) | MCP clients; bundles in `scikitplot/mcp/_plugins/` | `scikitplot/mcp/` |

## 2. Repository map

```text
.claude-plugin/
  marketplace.json          marketplace "scikit-plots": plugins scikitplot, scikitplot-dev
  plugin.json               plugin "scikitplot-dev" (the repository root)
skills/                     developers' skills (plugin scikitplot-dev)
  INDEX.md                  generated: package path -> skills and agents of both sets
  scikitplot-<x>-dev/       one per maintained path, metadata.scikitplot-mirror
  scikitplot-engineering-rules/   the numbered rules (no path, no agent)
  skill-creator/            vendored third-party skill, kept as published
agents/                     generated: scikitplot-<x>-dev.md, one per dev skill with a path
plugins/
  README.md                 users' guide (hand-written)
  maintenances/             this guide and MAINTAINING.md
  scikitplot/               plugin "scikitplot" (users)
    .claude-plugin/plugin.json
    README.md               generated
    skills/scikitplot[-<x>]/SKILL.md, references/API.md    generated
    agents/scikitplot[-<x>].md                              generated
pyproject.toml              [tool.scikitplot.maintenance.skills]: usage_modules, usage_excluded
tools/maint_tools/generate_usage_skills.py       writes the users' skills
maintenances/_repo/_maintenance/tools/check_records.py   checks both sets, writes agents and indexes
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
 skills/*-dev/SKILL.md (hand) ───────────────────────────────────────────────────────────────┤
                                                                                               ├─> check_records.py --write-index
                                                                                               │     agents/*.md, plugins/scikitplot/agents/*.md,
                                                                                               │     skills/INDEX.md, plugins/scikitplot/README.md
                                                                                               │   check_records.py (ok skills)
 metadata of every skill ───────> generate_pr_labeler.py sync ──> .github/labeler-by-module.yml  generate_pr_labeler.py check
 plugin.json, marketplace.json (hand) ───────────────────────────────────────────────────────>  claude plugin validate, agentskills validate
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
| Agents | the skill's name, `.md` | `agents/scikitplot-annoy-dev.md` | |

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
folder such as `skills/_cli/` can never hold a valid skill.

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
That is why the users' plugin is a separate folder: at the root it would ship
the developers' skills to every user.

## 6. Versions

Resolution order
([loading reference](https://code.claude.com/docs/en/plugins/loading.md)):
`plugin.json` `version`, then the marketplace entry's `version`, then the git
commit of the installed folder. A pinned `version` keeps every user on the
cached copy until the string changes. Policy here:

| Channel | `version` | Users get |
|---|---|---|
| main (development), `scikit-plots/scikit-plots` | not set | each commit; update with `marketplace update` then `plugin update` |
| released, `scikit-plots/skills` and the directory | the library version (`0.5.0`), set by `build_skills_repo.py` | that release; a new release raises it |

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
4. *Least privilege.* The users' plugin is Markdown only: no hooks, no MCP,
   no `bin/`. Agents omit `tools`, so they inherit the tools available to
   subagents, still under the user's permission mode.
5. *Progressive disclosure.* Short descriptions always loaded; the API table
   when a skill loads; the full API only in `references/API.md`.
6. *Separate audiences.* Users' and developers' sets live apart, so neither
   ships to the wrong people.

**Capabilities chosen, and those left out on purpose.**

| Capability | Used | Why |
|---|---|---|
| Skills | yes | Load on every surface; the right place for "how to use module X". |
| Agents | yes, generated | Specialists per module in Claude Code and Cowork, at no maintenance cost. |
| Commands | no | Skills already load everywhere; chat turns commands into skills anyway. |
| Hooks | no | Would run code on users' machines; nothing here needs it. |
| MCP server | not yet in the plugin | The local server works only in Claude Code and local Cowork; chat needs a hosted HTTPS server ([roadmap](#8-roadmap)). |
| `bin/` | no | claude.ai and Cowork refuse a plugin that has it. |

## 8. Roadmap

| Next step | Effort | Reference |
|---|---|---|
| Done: icon, release `version`, README data section, `scikit-plots/skills` and its sync workflow | - | [MAINTAINING section 9](MAINTAINING.md#9-release-and-submit-to-anthropics-directory) |
| Tag v0.5.0, let the workflow publish, submit `scikit-plots/skills` | small | [submit a plugin](https://claude.com/docs/plugins/submit.md) |
| Local MCP in the plugin (`.mcp.json` running `python -m scikitplot.mcp`) for Claude Code and local Cowork | small | [plugin structure](https://claude.com/docs/plugins/build) |
| Host the MCP server over HTTPS, then a remote `.mcp.json` entry and a connector submission | large | [connector submission](https://claude.com/docs/connectors/building/submission.md) |
| Evals: does a skill improve Claude's answers? `claude plugin eval` (each run is billed model use) | medium | [build overview](https://claude.com/docs/build/overview) |
| Declare `__all__` (and module docstrings) for `datasets`, `nc`, `stats`, `utils`, `neighbors`, `cexperimental`, `visualkeras` | per module | Rule 165 |
| A CI job that runs the checks in [MAINTAINING.md](MAINTAINING.md#7-validate-everything) on every pull request | small | |

## 9. Rules that govern this

In `skills/scikitplot-engineering-rules/rules/delivery.md`:

- **Rule 164**: a skill is `<skills dir>/<name>/SKILL.md`, its name its
  folder, prefixed `scikitplot`, derived from its package path.
- **Rule 165**: every skill with a path or module has an agent of its name,
  generated; users' skills are generated from the package.
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
