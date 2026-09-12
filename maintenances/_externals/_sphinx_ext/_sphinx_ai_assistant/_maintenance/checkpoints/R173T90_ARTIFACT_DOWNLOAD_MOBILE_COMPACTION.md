# R173T90 — Artifact download mobile compaction threshold

Status: COMPLETE

Base: R173T89 mobile speak-toggle resting visibility.

## Reported UI issue

A per-file artifact card could keep the visible `Download` label on some mobile
widths even though the same control correctly compacted to its download glyph
when the desktop panel was manually made narrow. Long filenames then lost useful
space precisely on the mobile surface where readability mattered most.

## Root cause

The icon-only policy used one `22rem` `ai-artifact-surface` container threshold
for two different jobs:

- per-file Download controls that directly compete with filename text and an
  overflow menu;
- bulk footer controls whose labels do not compete with a filename.

The container-query architecture itself was correct. The threshold policy was
too conservative for per-file rows. A full-width mobile panel can expose a
message/artifact surface wider than the default desktop panel, so a phone can be
visually crowded while still sitting just above `22rem`.

## Repair

- Keep named-container responsiveness; do **not** replace it with a viewport
  media query.
- Compact per-file snippet/presented-file Download labels at `26rem`.
- Keep `Download all` / patch-series footer labels visible until the original
  tighter `22rem` threshold.
- Continue clipping labels rather than `display:none`, preserving the existing
  accessible names, titles, and touch-target geometry.
- No JavaScript or download behavior changed.

## Verification

- activity/latest-file artifact contract: **198/198**;
- registered Node/UI harnesses: **160/160**;
- mutation catalogue structure/anchors: **243/243**;
- mutation execution: **240/240 mutants caught**;
- targeted new/neighbor mutation slice: **8/8**;
- JavaScript syntax: **GREEN**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.

## Prevention

Responsive thresholds belong to the content competition they resolve, not to a
generic device category. A per-file action that steals width from the filename
may need to compact earlier than a footer action carrying useful descriptive
text, even when both share the same icon-button primitive.
