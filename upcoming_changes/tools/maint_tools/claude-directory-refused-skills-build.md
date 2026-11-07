---
title: "The Claude directory refuses the corpus skill: LAUNCHER_REGISTRY_REDIRECT"
status: blocked
kind: "packaging"
area: "tools/maint_tools"
discovered_during: "PR 909 assembly: directory review of scikit-plots/skills 246d056"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The Claude directory refuses the corpus skill: LAUNCHER_REGISTRY_REDIRECT

## Summary

The Claude plugin directory refuses every build of `scikit-plots/skills` with
one blocking finding, `LAUNCHER_REGISTRY_REDIRECT` ("Package registry
redirected") on `skills/scikitplot-corpus/SKILL.md`. Latest refused build:
`05d1ab6`, from `scikit-plots/scikit-plots@70d915bf` (PR 910). The portal's
advice: "Name the output file in full, with a name that is not a package
manager's configuration file (.npmrc, bunfig.toml, pip.conf,
pnpm-workspace.yaml and the like); in documentation, link to the package
manager's instructions instead." Which text in the skill it reads is not yet
known; status `blocked` on the probe below.

## Why it matters

The plugin cannot be listed while a finding blocks, and every push to `main`
publishes a new build that the directory rescans. A fix made by hand in
`scikit-plots/skills` is overwritten by the next build (Rule 166), so the fix
belongs in this repository's generators.

## Current evidence

- Portal, version `05d1ab6`: "Directory policy refused this version. Reason:
  Package registry redirected"; one blocking finding, the one above.
- The finding names the corpus skill only. Every other skill of the same
  build, with the same generated sections, passes.
- Non-blocking findings of the same version, with their evidence:
  `MCP_FORWARDS_CREDENTIAL_ENV` on `.claude-plugin/plugin.json` (the plugin
  has no MCP server, hook or script, and no file reads or sends a token; the
  only matches are rule text in `scikitplot-engineering-rules`; a reviewer
  confirms), `UNKNOWN_KEY_CROSS_TOOL` (`icon`, "no action needed"),
  `MANIFEST_VERSION_MISSING` (branch builds are unversioned by design; a
  release tag sets `version`).
- Cleared earlier, history in PR 909 and its fragments: `LOCAL_CODE_FILE_TYPE`
  and `LOCAL_CODE_FRONTMATTER_KEY` (now enforced by
  `tools/maint_tools/build_skills_repo.py`, see Expected behavior).

## Root cause / current understanding

Verified:

- The trigger is inside `skills/scikitplot-corpus/` of the published tree.

Ruled out, each with the evidence that excludes it:

- The example block. PR 910 reduced it to executed doctest lines (no URL,
  path, prose or `+SKIP`); `05d1ab6` is still refused.
- The install block (`pip install scikit-plots`, `pip install
  scikit-plots-corpus`): eight unflagged skills carry the same two lines.
- The "Before writing code" block (`python -c ...m.<name>...`): every skill
  carries it.
- A package-manager configuration file, an index or registry option, a
  `npm|pip|uv config set` command or `VIRTUAL_ENV` (the triggers of the
  public cases of this finding): none occurs in the tree;
  `build_skills_repo.validate` now refuses each.

Unverified candidates (corpus-only content):

- `references/API.md`: downloaders that take `input_url` and write
  `output_path: Path | None = None` (an output file not named in full), and
  the embedding cache (`CachePolicy(unpinned=...)`, `DEFAULT_CACHE_DIR`,
  "the Hugging Face Hub commit a model id currently points to").
- The Public API rows of `SKILL.md` (`extract_archive`, "to a destination
  directory", and the rest of the corpus table).

## Expected behavior

The published build passes the directory with no blocking finding, and
`build_skills_repo.validate` refuses the identified trigger, so a regression
fails this repository's CI before it is published.

## Affected paths and ownership

- `tools/maint_tools/generate_usage_skills.py`: the skill text and its
  `references/API.md` (signatures and summaries from the installed package).
- `scikitplot/corpus/`: the docstrings and signatures those are read from
  (downloaders, embedding cache).
- `tools/maint_tools/build_skills_repo.py`: published layout and the
  directory checks; `.github/workflows/ci_claude_plugins_publish.yml`: the
  publish job that runs it.
- Maintenance area: `maintenances/_plugins/`.

## Constraints and non-goals

Generated files are never edited by hand, here or in `scikit-plots/skills`.
No change is made on a guess: the probe names the part first. The library's
API (downloaders, cache policy) is not changed to satisfy a listing rule; if
the trigger is its documentation, the skill's rendering of it changes.

## Edge cases to cover

A trigger in `references/API.md` rather than `SKILL.md` (the finding may be
reported on the skill's entry file); a trigger split across two sections (no
single ablation clears it: then pair the two candidates).

## Proposed direction

Run one portal **Validate** round on an ablation probe; do not submit it.

1. Build the probe from the latest refused build: one plugin folder per
   variant, each holding `.claude-plugin/plugin.json` (a distinct `name`),
   a README of at least 40 words, `LICENSE`, and
   `skills/scikitplot-corpus/` as listed:
   - `p9-full-tree`: the whole published tree (control);
   - `p0-control`: `SKILL.md` and `references/API.md` unchanged (control);
   - `p1-no-references`: `SKILL.md` only;
   - `p2-only-references`: `references/API.md`; `SKILL.md` reduced to
     frontmatter, title and intro;
   - `p3-no-example`, `p4-no-api-table`, `p5-no-install`, `p6-no-before`:
     `SKILL.md` without that one section, `references/API.md` unchanged;
   - `p7-head-only`: frontmatter, title and intro; no references.
2. Push it to a scratch public repository; in the developer portal open one
   draft (**Submit new > Plugin bundle**), validate each folder in turn
   (change the plugin path, **Re-validate**).
3. Both controls must report the finding; the variant that does not names
   the part. Narrow within it by halving the lines the same way.
4. Change the generator for that part, add the case to
   `build_skills_repo.validate` with a test, and delete this note once a
   published build passes.

## Verification / acceptance criteria

- The probe table (folder, finding yes/no) recorded here.
- `build_skills_repo.py check` green, with a test that refuses the trigger.
- The directory's scan of the next published `main` build reports no
  blocking finding.

## Documentation impact

`plugins/README.md` and `maintenances/_plugins/MAINTAINING.md` only if the
published layout or a skill's sections change.

## Release-note promotion

Not required: the published plugin tree is not part of the Python package.
Delete this note when the acceptance criteria hold; the PR is the record.
