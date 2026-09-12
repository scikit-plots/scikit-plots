# R173T19 — Snippet cards get the same card + ⋮ shape

Status: COMPLETE

Base: R173T18 timeline diff counts, same package.

## The inconsistency

The tracked-file card had settled on `preview | download | ⋮`. The snippet card
inside the answer body had not: it still carried a second full-width
"Save as file…" button beside its download card, so a two-snippet answer was
four buttons wide and the same gesture meant different things on the two
surfaces a reader meets minutes apart.

`.ai-md-artifact-row` is now `| card | ⋮ |` — the card downloads, the menu holds
everything else:

- **Save as a tracked file…** — gives it revisions, diffs and patch export
  (the old promote action).
- **Download as…** — download under a chosen name, nothing tracked.

## One menu builder, not two

The important part is not the button. Both surfaces now call one
`_buildOverflowMenu(ariaLabel, items, className)`; `_buildFileOverflow` is a
thin wrapper that supplies the tracked-file items.

A second near-identical menu is where Escape handling, focus return and
outside-click cleanup quietly go missing — the copy gets written for the happy
path and the keyboard support is not on it. The harness now asserts there is
exactly **one** keydown implementation in the file, and a mutant
(`snippet-card-grows-a-second-menu`) proves that assertion can fail.

Both surfaces therefore inherit, by construction: `role="menu"`/`menuitem`,
`aria-haspopup` and `aria-expanded`, Escape closing and returning focus to the
trigger, both capture-phase document listeners removed on close, and only one
menu open at a time.

## Verification

- browser wrapper gate: **147/147** (122 assertions in the owning harness);
- architecture gates: **442/442**, one new mutant caught.

## Four assertions relocated, not loosened

The menu mechanics moved from `_buildFileOverflow` into `_buildOverflowMenu`,
so four assertions were pointed at the shared builder while the item list stays
asserted on the wrapper. Nothing was weakened to make the refactor pass.
