---
title: "The Claude directory holds scikit-plots/skills for review: MCP_FORWARDS_CREDENTIAL_ENV"
status: blocked
kind: "packaging"
area: "tools/maint_tools"
discovered_during: "PR 916: directory review of scikit-plots/skills 24af6c0"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The Claude directory holds scikit-plots/skills for review: MCP_FORWARDS_CREDENTIAL_ENV

## Summary

Version `24af6c0` of `scikit-plots/skills` (the build of PR 915) passed the
directory's checks with no blocking finding and is "In review", held for
"Content policy review" by one policy hold, `MCP_FORWARDS_CREDENTIAL_ENV`
("Uses a credential from the user's machine"), reported on
`.claude-plugin/plugin.json`. Status `blocked`: the decision is Anthropic's
reviewer's. Nothing in the plugin reads or sends a credential.

## Why it matters

The listing cannot go live until the reviewer clears the version. A new push
to `main` publishes a new version while the review runs.

## Current evidence

- Portal, version `24af6c0` (2026-11-09): "Directory policy has warnings",
  "With a reviewer: Content policy review"; other findings are warnings
  (`UNKNOWN_KEY_CROSS_TOOL` for `icon`, "no action needed";
  `MANIFEST_VERSION_MISSING`, see `directory-listing-follows-main.md`).
- The hold's detail (portal Validate of the same tree): the plugin "reads the
  installer's environment (the whole environment object)" in
  `skills/scikitplot-engineering-rules/rules/portability.md`, and "can send
  data off the machine" in the `scikitplot` skill body's Python code.
- In the published tree the only whole-environment text is Rule 77's inline
  code `env = {**os.environ, ...}` (`rules/portability.md`, and its title row
  in `INDEX.md`). The `scikitplot` skill's example is `citation()`, which reads
  only local files (`scikitplot/config/_citation.py.in` imports `json`,
  `pathlib`, `re`). The mlflow skill sets one named variable,
  `SCIKITPLOT_PROJECT_MARKERS`, and reads no credential.
- The same hold was raised on earlier versions (CI-8 in
  `maintenances/_plugins/todo/todo.md`).

## Root cause / current understanding

Verified: the directory joins two unrelated documentation surfaces, a rule's
inline code and a skill's example, into one inferred data path. The plugin is
Markdown only: no MCP server, hook or script runs.

## Expected behavior

The reviewer clears the version, or names a change.

## Affected paths and ownership

- `plugins/_shared/skills/scikitplot-engineering-rules/rules/portability.md`
  (Rule 77), copied into each plugin by `check_records.py --write-index`.
- `plugins/scikitplot/skills/scikitplot/SKILL.md` (generated example).
- Maintenance area: `maintenances/_plugins/` (CI-8, CI-9).

## Constraints and non-goals

Correct engineering guidance is not rewritten to quiet a scanner unless the
reviewer asks. Do not push to `main` of this repository while the version is
with the reviewer unless the change is wanted in the review: every push
publishes a new version (Rule 171).

## Edge cases to cover

The reviewer asks for a change: then describe Rule 77 in words instead of
inline code (as Rule 170 does for its trigger), regenerate with
`check_records.py --write-index`, and check the published tree has no
whole-environment inline code before merging.

## Proposed direction

1. Wait for the decision on the plugin page. If asked, reply through the
   plugin's menu (Get help): the match is documentation; Rule 77 tells library
   code how to pass the environment to a subprocess on Windows; the plugin is
   Markdown only and `citation()` reads local files.
2. After approval: select Publish for the approved version (the old publish
   request is for `2ba48c0`); delete the four `probe/*` branches of
   `scikit-plots/skills` (open item in CI-9). PR 916 (the records and the
   guard) is already merged; its build is the newest version.

## Verification / acceptance criteria

- The plugin page shows the version approved and published.
- The build of PR 916 (merged) scanned with no blocking finding.
- No `probe/*` branch left in `scikit-plots/skills`.
Then delete this note.

## Documentation impact

None, unless the reviewer asks for the Rule 77 rewording above.

## Release-note promotion

Not required: the published plugin tree is not part of the Python package.
