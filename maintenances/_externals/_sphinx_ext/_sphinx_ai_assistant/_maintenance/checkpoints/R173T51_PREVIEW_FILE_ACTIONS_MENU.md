# R173T51 — A chevron menu for file actions

Status: COMPLETE

Base: R173T50 footer layout, same package.

## The title bar

```
| title …                       | ⌄ | ⎽  ▢  ✕ |
```

The chevron opens actions on the **file**; the three controls beside it act on
the **window** showing it. Two groups rather than one row of five, because
merging them would put Close next to Download and make them read as peers —
one closes a view, the other writes to disk.

Download is the first item. The menu exists as much for what comes next: a
future action is a row in one array, which is why the chevron was worth adding
now rather than a second bare button.

## Built from the shared menu

`_buildOverflowMenu` gained an optional icon parameter, defaulted so the three
existing callers keep the ⋮ they were written against. Everything else is
inherited: Escape closes and returns focus to the trigger, outside clicks
dismiss, only one menu is open at a time, `role="menu"`/`menuitem`. A fourth
copy would have grown its own weaker version of each — the failure R173T19
already corrected once.

**Items resolve per open**, reading `_attachmentPreviewState.item` at that
moment. The menu is built once when the dialog is created, so resolving at
build time would bind it to whatever file was open then and every later preview
would download the first one. That is the R173T27 staleness bug in a new place,
and its mutant is here to prove it stays fixed.

The download filename goes through `_artifactNameSlugPreservingExtension`, the
same sanitiser every other download in the panel uses.

## Rotation from state, not from a class

The chevron rotates on `[aria-expanded="true"]`. Driven from a class instead,
the glyph and the announced state could disagree — the arrow saying open while
the trigger still reports closed to a screen reader. Motion is opt-out under
`prefers-reduced-motion`.

## Verification

- browser wrapper gate: **150/150** (57 assertions in the owning harness);
- architecture gates: **551/551**, two new mutants, both caught:
  `preview-file-menu-frozen-at-build-time` and
  `chevron-rotation-not-driven-by-aria`.

## A duplicate media block, merged

The menu button's touch sizing was first written as a second
`@media (max-width: 640px), (pointer: coarse)` block, which broke the
uniqueness of an existing mutant's anchor. Folded into the one that was already
there — the same one-place principle applied to `.ai-md-file-sheet` in R173T46,
and the same signal: when a mutant anchor stops being unique, the code has
usually grown a duplicate worth merging rather than an anchor worth rewording.
