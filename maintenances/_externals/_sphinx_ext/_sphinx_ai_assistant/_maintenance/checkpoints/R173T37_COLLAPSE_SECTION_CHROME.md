# R173T37 — Collapsible sections read as controls

Status: COMPLETE

Base: R173T36 share destination layout, same package.

## The inconsistency

`.ai-assistant-panel-review-content-privacy` was already a bordered card.
`.ai-assistant-conv-share-collapse` — every other collapsible section — got a
single `border-top` hairline.

So one control looked like a panel and its siblings looked like separators. A
reader scanning the share sheet for *where do I change what gets sent* had to
learn which rows were interactive by clicking them.

Every section is now the same card: border, radius, surface, and spacing
between adjacent ones. `:focus-within` puts keyboard focus on the card rather
than only on the button inside it, so the whole thing reads as one control.
Open sections carry a slightly heavier border than closed ones, so a reader can
see at a glance which they have already been through.

## The privacy section is not a peer

It governs what leaves the device, so it keeps the shared card **shape** — a
different shape would read as a different kind of thing — and takes an accent
leading edge, from the same `--ai-artifact-accent` token R173T24 introduced. It
is identifiable while scrolling and cannot be mistaken for a formatting option.

In forced-colours modes the accent is discarded, so the distinction falls back
to a heavier edge rather than disappearing entirely.

## What the border is and is not

A signpost, not a safeguard. What makes this section safe is that the snapshot
is built from the reviewed options — asserted in the share harnesses and by
`local-save-redacted-like-a-published-share` and
`share-export-leaks-the-source-page`, not by anything in this checkpoint.
Styling a control to look important does not make it enforce anything, and it
is worth being explicit about that: the value here is that a reader notices the
control, not that the control does more.

## Verification

- browser wrapper gate: **148/148** (133 assertions in the owning harness);
- architecture gates: **497/497**, two new **CSS-targeted** mutants, both
  caught: `privacy-section-looks-like-a-formatting-option` and
  `collapse-sections-are-bare-rows`.
