# Using scikit-plots with Claude: skills, agents and plugins

This guide is for anyone who wants Claude to help them **use** the
scikit-plots Python library in their own code. It assumes no prior knowledge
of Claude Code, plugins, skills or agents, and goes from nothing to a working
setup, then on to updates, removal and troubleshooting.

If you develop scikit-plots itself, read
[`maintenances/_plugins/README.md`](../maintenances/_plugins/README.md) instead.

## TL;DR

```sh
# 1. The library itself (any Python 3 environment)
pip install scikit-plots

# 2. Claude Code (needs a Claude Pro, Max, Team or Enterprise plan, or an API account)
#    https://code.claude.com/docs/en/setup  -> run `claude` once and sign in

# 3. The scikitplot plugin: a skill and an agent per public module
claude plugin marketplace add scikit-plots/scikit-plots --sparse .claude-plugin plugins
claude plugin install scikitplot@scikit-plots
claude plugin list          # expect: scikitplot@scikit-plots ... Status: enabled
```

Then start `claude` in your project and ask, for example: *"Using scikitplot,
plot a confusion matrix for a LogisticRegression on the iris data."*

- Installing and listing plugins uses **no** chat usage; only messages to
  Claude do.
- The plugin above follows the repository's main branch: each commit is a new
  version, and nothing updates until you run the update commands in
  [Versions and updates](#7-versions-and-updates). From scikit-plots 0.5.0,
  released versions that match `pip install scikit-plots` come from
  [`scikit-plots/skills`](https://github.com/scikit-plots/skills) and
  Anthropic's directory ([section 7](#7-versions-and-updates)).
- On claude.ai chat the skills work but the agents are ignored; in Claude Code
  and Cowork both work.

## Contents

1. [What you get](#1-what-you-get)
2. [Words you will meet](#2-words-you-will-meet)
3. [Before you start](#3-before-you-start)
4. [Choose where to use it](#4-choose-where-to-use-it)
5. [Path A: Claude Code in a terminal](#5-path-a-claude-code-in-a-terminal)
6. [Path B: claude.ai and Cowork](#6-path-b-claudeai-and-cowork)
7. [Versions and updates](#7-versions-and-updates)
8. [Path C: one skill on its own, any tool](#8-path-c-one-skill-on-its-own-any-tool)
9. [Using it well](#9-using-it-well)
10. [Manage, disable, remove](#10-manage-disable-remove)
11. [Gotchas](#11-gotchas)
12. [Troubleshooting](#12-troubleshooting)
13. [Security and privacy](#13-security-and-privacy)
14. [FAQ](#14-faq)
15. [Further reading (official sources)](#15-further-reading-official-sources)

## 1. What you get

The `scikitplot` plugin lives in [`plugins/scikitplot/`](scikitplot/README.md)
and holds two kinds of files for each public module of scikit-plots:

| Piece | Where | What it does |
|---|---|---|
| Skill | `plugins/scikitplot/skills/scikitplot-<module>/SKILL.md` | Instructions and reference Claude loads when your task matches: what the module is for, how to install it, its public API (from the module's own `__all__` and docstrings) and an example. |
| Agent | `plugins/scikitplot/agents/scikitplot-<module>.md` | A specialist Claude can hand a task to. It starts with its module's skill already loaded. |

Names always start with `scikitplot`: the root package is `scikitplot`, a
subpackage is `scikitplot-<name>` (for example `scikitplot.annoy` →
`scikitplot-annoy`, `scikitplot.feature_selection` →
`scikitplot-feature-selection`). The full list, with links, is in
[`scikitplot/README.md`](scikitplot/README.md).

The skills are **generated from the library's code**, so they say only what
the code declares. Modules that declare no public API (`__all__`) have no
skill yet.

## 2. Words you will meet

| Word | Meaning |
|---|---|
| **Claude Code** | Anthropic's coding tool for the terminal, IDEs and the desktop app. It can read your files and run commands. |
| **Skill** | A folder with a `SKILL.md` file. Claude reads its short description all the time and loads the rest only when a task needs it. Defined by the open [Agent Skills specification](https://agentskills.io/specification). |
| **Agent (subagent)** | A Markdown file describing a specialist with its own instructions; Claude can delegate a task to it. |
| **Plugin** | A folder that bundles skills, agents and other pieces, installed in one step. |
| **Marketplace** | A catalog of plugins, here the file `.claude-plugin/marketplace.json` in the scikit-plots repository. Its name is `scikit-plots`. |
| **`name@marketplace`** | How a plugin is named when you install it: `scikitplot@scikit-plots`. |
| **Scope** | Who an install applies to: `user` (you, every project; the default), `project` (everyone who clones the project), `local` (you, this project). |
| **MCP server / connector** | A program that gives Claude extra tools or data. This plugin has none; see the [FAQ](#14-faq). |

## 3. Before you start

| You need | Why | Check |
|---|---|---|
| A Claude plan with Claude Code (Pro, Max, Team or Enterprise), or an Anthropic Console (API) account | Claude Code signs in with one of these | `claude` shows `Claude Pro` (or your plan) under the logo |
| Claude Code | It installs and runs the plugin | `claude --version` |
| `git` | Claude Code clones the marketplace repository | `git --version` |
| Python and scikit-plots, in the environment your project uses | The skills tell Claude to check the installed version and signatures before writing calls | `python -c "import scikitplot; print(scikitplot.__version__)"` |

Install Claude Code by following its official
[setup page](https://code.claude.com/docs/en/setup). The first time you run
`claude`, it asks how to sign in:

```text
❯ 1. Claude account with subscription · Pro, Max, Team, or Enterprise
  2. Anthropic Console account · API usage billing
  3. 3rd-party platform · Amazon Bedrock, Microsoft Foundry, Google Vertex AI
```

Choose **1** if you use claude.ai with a paid plan, **2** if you have an API
account. A browser window opens; sign in, then return to the terminal. The
next screens (security notes, terminal setup, "trust this folder?") are
one-time: press Enter, and answer *Yes* to the trust question only for a
folder you trust.

> **Tip.** A free claude.ai plan does not include Claude Code. Check your plan
> on claude.ai under *Settings → Billing*.

## 4. Choose where to use it

Each Claude app loads a different part of a plugin
([platform support](https://claude.com/docs/plugins/platform-support)):

| | Skills | Agents | Install from |
|---|---|---|---|
| **Claude Code** (terminal, IDE, desktop Code tab) | yes | yes | the `claude plugin` commands, [Path A](#5-path-a-claude-code-in-a-terminal) |
| **Cowork** (desktop app tasks) | yes | yes | claude.ai *Customize → Plugins*, [Path B](#6-path-b-claudeai-and-cowork) |
| **claude.ai chat** (web, desktop, mobile) | yes | ignored, without an error | claude.ai *Customize → Plugins*, [Path B](#6-path-b-claudeai-and-cowork) |
| **Any other tool that reads Agent Skills** | yes (one folder at a time) | no | copy a folder, [Path C](#8-path-c-one-skill-on-its-own-any-tool) |

A plugin you add on claude.ai also appears in Claude Code (as
`scikitplot@synced`) when you sign in to Claude Code with the same account
([synced plugins](https://code.claude.com/docs/en/plugins/loading.md)).

## 5. Path A: Claude Code in a terminal

**Step 1. Add the marketplace.** In your shell (not inside a Claude session):

```sh
claude plugin marketplace add scikit-plots/scikit-plots --sparse .claude-plugin plugins
```

Expected output ends with:

```text
✔ Successfully added marketplace: scikit-plots (declared in user settings)
```

`--sparse .claude-plugin plugins` downloads only the two folders the plugin
needs instead of the whole repository
([marketplace sources](https://code.claude.com/docs/en/plugins/marketplace-reference.md)).
Running the command again is harmless; it refreshes the copy.

**Step 2. Install the plugin.**

```sh
claude plugin install scikitplot@scikit-plots
```

```text
Installing plugin "scikitplot@scikit-plots"...✔ Successfully installed plugin: scikitplot@scikit-plots (scope: user)
```

To share it with everyone who clones *your* project instead, add
`--scope project` (it writes the project's `.claude/settings.json`).

**Step 3. Check it.**

```sh
claude plugin list
claude plugin details scikitplot@scikit-plots
```

```text
Installed plugins:

  > scikitplot@scikit-plots
    Version: bdc727caf6f3
    Scope: user
    Status: ✔ enabled
```

`details` lists the 22 skills and 22 agents and the context they cost: about
4,700 tokens are added to every session (the skills' short descriptions);
each skill's full text loads only when used.

**Step 4. Use it.** Start `claude` inside your project folder and ask in
plain language. Claude picks the matching skill from its description:

```text
Using scikitplot, plot ROC curves for a RandomForestClassifier on the breast cancer data.
```

To be explicit, name the skill (`/scikitplot:scikitplot-api`) or the agent:
plugin agents appear in the `@` typeahead under their scoped name, such as
`@scikitplot:scikitplot-annoy`
([subagents](https://code.claude.com/docs/en/sub-agents.md)). To run a whole
session as one agent:

```sh
claude --agent scikitplot:scikitplot-annoy
```

> **Tip.** Inside a session, `/plugin` opens the plugin manager (installed
> plugins, marketplaces, errors), and `/reload-plugins` applies changes
> without restarting.

## 6. Path B: claude.ai and Cowork

**Option 1, from the marketplace.** In claude.ai (or the desktop app) go to
**Customize → Plugins → Add → Add marketplace** and enter
`https://github.com/scikit-plots/scikit-plots`, then install `scikitplot`.

**Option 2, upload a zip.** Build a zip of just the plugin folder:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/scikit-plots/scikit-plots
cd scikit-plots && git sparse-checkout set plugins/scikitplot
cd plugins && zip -r scikitplot.zip scikitplot
```

Then **Customize → Plugins → Add → Upload plugin** and choose
`scikitplot.zip` (about 120 KB; the limit is 200 MB and 5,000 files per
plugin). Check it by asking in a new chat: *"Which skills do you have from
plugins?"*

Only install `scikitplot` on claude.ai. The marketplace also lists
`scikitplot-mcp`, which starts a local program (claude.ai chat can only use
hosted servers), and `scikitplot-dev`, the developers' plugin, meant for
Claude Code in a checkout of the repository.

Steps and limits: [Plugin structure and testing](https://claude.com/docs/plugins/build),
[platform support](https://claude.com/docs/plugins/platform-support).

## 7. Versions and updates

**Channels.** Both repositories carry the same refs, built from one place
(scikit-plots/scikit-plots). Pick the one that matches the scikit-plots you
installed:

| You installed scikit-plots with | Add the marketplace | Its version |
|---|---|---|
| `pip install scikit-plots` (the newest release) | `scikit-plots/skills` (its `stable` branch: the newest release) | the library version, e.g. `0.5.0` |
| one exact release `X.Y.Z` | `scikit-plots/skills@vX.Y.Z` | `X.Y.Z` |
| a release line kept up to date, `X.Y.*` | `scikit-plots/skills@maintenance/X.Y.x` | the git commit |
| `pip install git+https://github.com/scikit-plots/scikit-plots` or a checkout | `scikit-plots/skills@main`, or `scikit-plots/scikit-plots` (the commands in [section 5](#5-path-a-claude-code-in-a-terminal)) | the git commit |

The `scikit-plots/skills` refs start with scikit-plots 0.5.0. Install, for example:

```sh
claude plugin marketplace add scikit-plots/skills            # or scikit-plots/skills@v0.5.0
claude plugin install scikitplot@scikit-plots-skills
```

Install one channel only: every channel provides a plugin named `scikitplot`,
so their skills would have the same names.

The rest of this section describes the development channel.

**How the version is chosen.** Claude Code computes a version for every
plugin
([versions and updates](https://code.claude.com/docs/en/plugins/loading.md)):
the `version` in the plugin's `plugin.json` first, then the one in the
marketplace entry, and when neither is set (the case here) the **git commit**
of the installed folder. So:

- you get the API of the commit you installed, a development version, not a
  release;
- every new commit on the main branch is a new version;
- **nothing updates by itself**: auto-update is off by default for
  marketplaces other than Anthropic's.

**Update** (uses no chat usage):

```sh
claude plugin marketplace update scikit-plots
claude plugin update scikitplot@scikit-plots
```

```text
✔ scikitplot is already at the latest version (bdc727caf6f3).
```

Or turn on automatic updates: inside `claude`, `/plugin` → **Marketplaces** →
`scikit-plots` → **Enable auto-update**. Run `/reload-plugins` (or restart)
after an update.

**Pin a fixed point.** Add the marketplace at a branch or tag:

```sh
claude plugin marketplace add scikit-plots/scikit-plots@<tag-or-branch> --sparse .claude-plugin plugins
```

Only tags made after the plugin was added to the repository contain it.

**Match the plugin to your installed library.** The skills describe the
commit they were generated from. Compare:

```sh
python -c "import scikitplot; print(scikitplot.__version__)"   # e.g. 0.5.dev0+git.20261026.bdc727c
claude plugin list                                                 # e.g. Version: bdc727caf6f3
```

When the commit parts agree, the skills describe exactly your library. When
they differ, the agents and skills still tell Claude to check signatures with
`inspect.signature` and `help()` before using a name.

## 8. Path C: one skill on its own, any tool

Every skill folder follows the open Agent Skills format, so it can be copied
on its own:

```sh
# Claude Code, for you in every project
mkdir -p ~/.claude/skills && cp -r plugins/scikitplot/skills/scikitplot-annoy ~/.claude/skills/
# Claude Code, for one project (commit it to share with the team)
mkdir -p .claude/skills && cp -r plugins/scikitplot/skills/scikitplot-annoy .claude/skills/
```

Other tools that read Agent Skills take the same folder; see their own docs
for where. Check any folder against the specification with its reference
validator:

```sh
pip install skills-ref
agentskills validate plugins/scikitplot/skills/scikitplot-annoy   # Valid skill: ...
```

A copied skill does not update; copy it again after updating the repository.

## 9. Using it well

- **Say "scikitplot" and the module.** Skill descriptions start with
  "Use scikitplot.<module> (scikit-plots) in any Python project", so a request
  that names the module matches it reliably.
- **Let Claude run code.** The skills ask Claude to check the installed
  version and signatures first; in Claude Code it can do that in your
  environment, in chat it relies on the skill's API tables.
- **One skill per module, plus the root.** `scikitplot` covers the package
  entry points (`get_logger`, `set_config`, `show_versions`, ...) and lists
  the other skills; `scikitplot-api` covers the plotting functions
  (`plot_confusion_matrix`, `plot_roc`, ...).
- **Big modules have a reference file.** A skill shows up to 40 names; the
  full list with signatures is in its `references/API.md`, which Claude opens
  when needed.
- **Watch your plan's limits.** Claude Code shows usage in its status line.
  `/model sonnet` uses less of a plan's limit than larger models.

## 10. Manage, disable, remove

```sh
claude plugin disable scikitplot@scikit-plots     # keep installed, stop loading
claude plugin enable  scikitplot@scikit-plots
claude plugin uninstall scikitplot@scikit-plots   # remove the plugin
claude plugin marketplace remove scikit-plots     # remove the catalog too
claude plugin marketplace list
```

On claude.ai, manage it in **Customize → Plugins**.

## 11. Gotchas

| Gotcha | What to do |
|---|---|
| `plugin update` says "already at the latest version" after a push | Run `claude plugin marketplace update scikit-plots` first; the catalog copy is refreshed separately. |
| Agents do nothing in claude.ai chat | By design: chat ignores agents. Use Claude Code or Cowork. |
| Skills mention a name your library doesn't have | Your library and the plugin come from different commits; compare them ([section 7](#7-versions-and-updates)) and update one. |
| Adding the marketplace without `--sparse` is slow | Without it the whole repository is cloned. |
| `scikitplot-dev` fails on claude.ai | It is the developers' plugin (the whole repository). Install `scikitplot`. |
| A module has no skill | Its module declares no `__all__`; see the list in `pyproject.toml` under `[tool.scikitplot.maintenance.skills]`. |
| Two plugins with the same skill names | Plugin skills are namespaced (`/scikitplot:...`), so they don't collide in Claude Code; a copied skill (Path C) is not namespaced. |
| Changes don't appear in a running session | Run `/reload-plugins` or restart `claude`. |

## 12. Troubleshooting

| You see | Likely cause | Fix |
|---|---|---|
| `claude: command not found` | Claude Code is not installed or not on `PATH` | Follow the [setup page](https://code.claude.com/docs/en/setup). |
| The sign-in offers no Claude Code access | Free plan | A paid plan or a Console account is needed. |
| `Plugin "scikitplot" not found in marketplace "..."` | The marketplace was not added, has another name, or its copy is old | `claude plugin marketplace list`; add it as in [Path A](#5-path-a-claude-code-in-a-terminal), or `claude plugin marketplace update scikit-plots`. |
| Clone timeout or network error | No access to github.com | Retry; behind a proxy, see Claude Code's [network configuration](https://code.claude.com/docs/en/network-config.md). |
| Upload says `plugin.json must be at .claude-plugin/plugin.json at the zip root` | The zip contains more than the plugin folder | Zip only `plugins/scikitplot` as shown in [Path B](#6-path-b-claudeai-and-cowork). |
| Errors in `/plugin` | Anything the loader rejected | Open the **Errors** tab; messages are explained in [Troubleshoot plugins](https://code.claude.com/docs/en/plugins/troubleshooting.md). |

## 13. Security and privacy

- The `scikitplot` plugin is **Markdown only**: skills, agents and a README.
  It has no hooks, no MCP servers and no executables, and it sends nothing
  anywhere by itself.
- The **agents inherit Claude Code's tools**, so they can read and edit files
  and run commands in your project, under your permission mode. Shift+Tab
  cycles the mode; choose one that asks before acting while you learn
  ([permission modes](https://code.claude.com/docs/en/permission-modes)).
- Only install marketplaces you trust; a plugin can carry code
  ([security](https://code.claude.com/docs/en/security)).

## 14. FAQ

**Do I need a release of scikit-plots?** No. The plugin tracks the
repository's commits; the library can be any version you have installed.
See [section 7](#7-versions-and-updates).

**Does it cost anything extra?** Installing does not use your plan. Sessions
use your plan's normal limits; the skills' descriptions add about 4,700
tokens of context to each session while the plugin is enabled.

**Is there an MCP server?** Yes, opt-in: the plugin `scikitplot-mcp` registers
the read-only, source-cited documentation search server
(`python -m scikitplot.mcp`), which runs on your machine in the Python where
scikit-plots is installed (`pip install "scikit-plots[mcp]"` if your release
has that extra). `scikitplot` itself stays Markdown only and runs nothing.
Install it from the development marketplace:
`claude plugin install scikitplot-mcp@scikit-plots`. Bundles for other
clients (Cursor, Windsurf, VS Code, Continue, Cline, Codex) are in
`scikitplot/mcp/_plugins/`. claude.ai chat can only use remote (hosted)
servers ([platform support](https://claude.com/docs/plugins/platform-support)).

**Can I use it without Claude?** Each skill is a plain Agent Skills folder
that other tools can read ([Path C](#8-path-c-one-skill-on-its-own-any-tool)).

**Where do I report a problem?** In the repository's issue tracker:
<https://github.com/scikit-plots/scikit-plots/issues>.

## 15. Further reading (official sources)

| Topic | Source |
|---|---|
| Build and add things to Claude | <https://claude.com/docs/build/overview> |
| What each Claude app loads from a plugin | <https://claude.com/docs/plugins/platform-support> |
| Plugin structure and testing on each app | <https://claude.com/docs/plugins/build> |
| Install Claude Code | <https://code.claude.com/docs/en/setup> |
| Plugin loading, versions and updates | <https://code.claude.com/docs/en/plugins/loading.md> |
| Marketplaces (sources, `@ref`, `--sparse`) | <https://code.claude.com/docs/en/plugins/marketplace-reference.md> |
| Plugin manifest | <https://code.claude.com/docs/en/plugins-reference> |
| Skills in Claude Code | <https://code.claude.com/docs/en/skills.md> |
| Subagents | <https://code.claude.com/docs/en/sub-agents.md> |
| Agent Skills specification | <https://agentskills.io/specification> |
| scikit-plots documentation | <https://scikit-plots.github.io> |
