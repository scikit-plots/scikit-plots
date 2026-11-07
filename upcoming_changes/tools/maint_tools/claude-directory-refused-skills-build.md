---
title: "The Claude plugin directory refused the scikit-plots/skills build"
status: open
kind: "packaging"
area: "tools/maint_tools"
discovered_during: "PR 909 assembly: directory review of scikit-plots/skills 246d056"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The Claude plugin directory refused the scikit-plots/skills build

## Summary

The `scikitplot` plugin, published to `scikit-plots/skills` by
`.github/workflows/ci_claude_plugins_publish.yml` from `plugins/scikitplot/`
through `tools/maint_tools/build_skills_repo.py`, was refused by the Claude
plugin directory for version `246d056` ("Build main from
scikit-plots/scikit-plots@583b6490"). The maintainer's directory page
reports "Validation failed, Directory policy failed, Security scan passed",
with three findings that block and three that do not.

## Why it matters

The plugin cannot be listed until each blocking finding is cleared, and every
push to `main` publishes a new build: a fix made by hand in
`scikit-plots/skills` is overwritten by the next build, so every fix belongs
in this repository's generators.

## Current evidence

Directory findings, as reported for `246d056` (blocking first):

1. `LAUNCHER_REGISTRY_REDIRECT`, `skills/scikitplot-corpus/SKILL.md`: "Name
   the output file in full, with a name that is not a package manager's
   configuration file (.npmrc, bunfig.toml, pip.conf, pnpm-workspace.yaml and
   the like); in documentation, link to the package manager's instructions
   instead."
2. `LOCAL_CODE_FILE_TYPE`: refused unless every file is Markdown, text, JSON,
   YAML or an allowed image, none starts with `#!`, all text is UTF-8
   without NUL; "a license, a readme, .gitignore, and the plugin's icon can
   stay".
3. `LOCAL_CODE_FRONTMATTER_KEY`: a skill, command, agent or output style has
   a frontmatter key, or a value under one, outside the allowed list; "give
   a key such as tags only text".
4. Not blocking: `MCP_FORWARDS_CREDENTIAL_ENV` (`.claude-plugin/plugin.json`;
   a reviewer checks it), `UNKNOWN_KEY_CROSS_TOOL` (`icon`; no action
   needed), `MANIFEST_VERSION_MISSING` (branch builds are unversioned by
   design; release tags carry a version).

Checked on the published tree (`git clone scikit-plots/skills`, `246d056`)
and on this checkout's build
(`build_skills_repo.py build --out DIR --unversioned`, 115 files):

- No file starts with `#!`, holds a NUL byte (other than the PNG icon) or is
  not UTF-8. Files that are not `.md`, `.json` or `.png`: `.gitignore`,
  `LICENSE`, and the extensionless license texts under `LICENSES/` (22 in the
  published tree; PR 909 adds `LICENSES/LICENSE_unicode`).
- Frontmatter keys in use: skills `name`, `description`, `license`,
  `metadata` (a map, for example `scikitplot-module:`); agents `name`,
  `description`, `skills` (a list naming the preloaded skill).
- `skills/scikitplot-corpus/SKILL.md` names no package-manager configuration
  file by a plain search; its install block is `pip install scikit-plots` and
  `pip install scikit-plots-corpus`. Eight other skills carry the same
  two-line install block and were not flagged, so the install block alone is
  not the trigger.

## Root cause / current understanding

Not established. Leads to verify against the directory's own finding detail
(the portal's "Documentation" link per finding), not to act on as facts:

- file type: the extensionless `LICENSES/LICENSE_*` files may not count as
  "a license" (only the root `LICENSE` may);
- frontmatter: the agents' `skills:` list or the skills' `metadata:` map may
  be outside the community allowed list;
- registry redirect: the trigger in the corpus skill is unknown; its text is
  generated from `scikitplot/corpus/__init__.py`'s docstring.

## Expected behavior

The build passes the portal's **Validate** with no blocking finding, and
`build_skills_repo.py`'s own `validate` refuses each case the directory
refuses, so a regression fails in this repository's CI before it is
published.

## Affected paths and ownership

`tools/maint_tools/build_skills_repo.py` (layout, licenses, validation),
`maintenances/_repo/_maintenance/tools/check_records.py` (agent files and
their frontmatter), `tools/maint_tools/generate_usage_skills.py` (skill
text), `scikitplot/corpus/__init__.py` (the docstring the corpus skill is
generated from). Maintenance area: `maintenances/_plugins/`.

## Constraints and non-goals

Generated files are never edited by hand, here or in `scikit-plots/skills`.
Third-party license texts stay complete and shipped. No change to the
`scikitplot-dev` plugin, which is not published to the directory.

## Edge cases to cover

A release tag build (`--version`) and a branch build (`--unversioned`); a
new vendored license (as `LICENSE_unicode` in PR 909).

## Proposed direction

Read each blocking finding's detail on the directory page, reproduce it in
`build_skills_repo.validate`, fix it in the generator, rebuild, and confirm
with the portal's Validate before the next publish.

## Verification / acceptance criteria

`build_skills_repo.py check` green; a new validation test per finding; the
portal's Validate passes for the next published commit.

## Documentation impact

`plugins/README.md` and `maintenances/_plugins/MAINTAINING.md` if the
published layout or frontmatter changes.

## Release-note promotion

Not required: the published plugin tree is not part of the Python package.
