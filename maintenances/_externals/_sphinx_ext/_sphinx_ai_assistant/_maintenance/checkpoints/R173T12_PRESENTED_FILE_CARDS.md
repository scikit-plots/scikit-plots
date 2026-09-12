# R173T12 — Presented-file cards and timeline reporting

Status: COMPLETE

Base: R173T11 in-place file preview flow, same package.

## What changed

T11 collapsed the whole-file mirror into an in-place disclosure and renamed the
summary to `Presented N files`. This completes the flow the reference layout
describes: a card per file, then one full-width bulk action.

**Per card.** A type badge sits beside the filename — it does what an extension
does at a glance, and unlike the extension it survives truncation of a long
path, which is the case that actually needs it. `Preview` and `Download` own
the card's primary row. `Patch` and `Continue` drop to a quieter second line:
four equal-weight buttons made none of them primary, and most readers want the
first two. An empty secondary line collapses rather than leaving a gap.

**Footer.** Both bulk controls now share one strip, and `Download all N files`
spans its full width so it reads as covering every card above it rather than as
a fifth peer button. Collapsing the summary hides the whole strip.

**Timeline.** The presentation is now reported as a `file` activity step —
`Presented N files`, detailing each path at its content revision. File events
are already the timeline's job; a presentation that existed only in the answer
body was invisible to a reader who consults the timeline to see what a turn
actually did.

## Verification

- browser wrapper gate: **144/144**;
- architecture gates: **417/417**, one new mutant caught:
  `presented-files-absent-from-activity-timeline`;
- 94 assertions in the owning harness, including two presentation contracts
  checked against the paired stylesheet (download-all spans the strip; an empty
  secondary line collapses).

## Two stale expectations corrected

`bulk download re-resolves latest files` pinned the literal
`'Download all latest files'`. The label now names the count; the contract it
guards — resolving each file from the ledger at click time — is unchanged and
still asserted.

`collapsing the summary hides its series control with it` was superseded rather
than broken: the footer strip now holds both bulk controls, so the assertion
was rewritten against the footer it replaced.

## Still open

Unchanged: §18 branching and §20 virtualised transcripts remain deferred with
rationale in `STATE.json`.
