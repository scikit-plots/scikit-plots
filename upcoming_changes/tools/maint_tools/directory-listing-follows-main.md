---
title: "The directory listing follows main, not the released stable channel"
status: blocked
kind: "packaging"
area: "tools/maint_tools"
discovered_during: "PR 916: directory review of scikit-plots/skills 24af6c0"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The directory listing follows main, not the released stable channel

## Summary

The directory submission of `scikit-plots/skills` tracks `main`, the
unversioned development build, so every version carries the warning
`MANIFEST_VERSION_MISSING` ("No version in plugin.json"). Rule 166 and
section 9 of `maintenances/_plugins/MAINTAINING.md` intend the listing to
follow `stable`, the newest release, which does not exist yet. Status
`blocked`: on the first release tag, and on the end of the current review.

## Why it matters

Following `main` lists development builds: users of the listing get modules
the newest PyPI release may lack, and every merge here becomes a new directory
version to scan and review.

## Current evidence

- Portal: every scanned version is a `Build main from ...` commit; the
  `version` warning on each.
- `scikit-plots/skills` has only `main` (no `stable`, no release tag): no
  release has been built since the publish workflow was added (CI-6).
- `.github/workflows/ci_claude_plugins_publish.yml`: a tag `vX.Y.Z` is built
  with `version` and pushed as that tag; the newest release is force-pushed to
  `stable`.

## Root cause / current understanding

Verified: the submission was made before any release existed, with the
tracked branch left on the default branch, `main`.

## Expected behavior

The submission tracks `stable`; each version has a `version`; the warning is
gone; `main` stays the development channel.

## Affected paths and ownership

- `.github/workflows/ci_claude_plugins_publish.yml`,
  `tools/maint_tools/build_skills_repo.py` (Rule 166).
- The submission's Settings tab, field "Tracked branch or tag" (portal).
- Maintenance area: `maintenances/_plugins/` (CI-9).

## Constraints and non-goals

No plugin manifest in this repository sets `version` (Rule 166); do not add
one to `main`. Change the tracked branch only when no publish request waits
and the plugin is not with a reviewer: changing it cancels a waiting request
(Rule 171).

## Edge cases to cover

The first `stable` build is refused or held: fix it on `main` and tag again
(tags never move); keep the listing on its last published version meanwhile.

## Proposed direction

1. After the current review ends (`directory-review-credential-hold.md`), tag
   the next release; let the publish workflow create the tag and `stable` in
   `scikit-plots/skills`.
2. Validate `stable` in the portal, then set the submission's tracked branch
   to `stable` and select Check for new commits.
3. Publish the passing release version.

## Verification / acceptance criteria

- `scikit-plots/skills` has `stable` and the release tag, `plugin.json` with
  `version`.
- The portal's newest version is the `stable` commit, with no
  `MANIFEST_VERSION_MISSING`.
Then delete this note.

## Documentation impact

`maintenances/_plugins/MAINTAINING.md` section 9 already describes this path;
none otherwise.

## Release-note promotion

Not required: the published plugin tree is not part of the Python package.
