# R173T15 — Activity that survives a reload

Status: COMPLETE

Base: R173T14 file-editing sheet and data-raw dedup, same package.

## The gap

`section.ai-assistant-panel-activity` vanished on reload. A remembered
conversation came back as answers with no account of how they were produced —
what context was prepared, which files were presented, what was verified. That
is the part a sceptical reader most wants to re-read, and it was the only part
that did not survive.

## A summary, not the live state

What persists is a bounded step summary: `kind`, `state`, `label`, optional
`detail`. The live activity object owns budgets, cancellation and file byte
accounting; none of that means anything once the page is gone, and carrying it
would persist a controller that can no longer control anything.

The summary is read from the **rendered rows**, not the internal step map.
The rows are exactly what the reader saw; reconstructing from internal state
risks persisting something that was never displayed.

Capture happens at record time, when the rows are final — earlier would persist
a half-finished timeline, later would race the next turn reusing the live
object.

## Bounded twice, because the budget is shared

12 steps, 120-character labels, 240-character details. This rides in session
storage beside the transcript, and R173T10 established that the persistence
budget is shared rather than per-feature: a timeline that quietly grew would
push turns out of the very conversation it is describing.

## Restore re-validates

Session storage is same-origin but not trustworthy — any script on the page can
write to it. Every restored field is re-checked against the same enumerations
the live renderer uses, so a tampered record can only ever produce a shorter or
emptier timeline, never a different kind of one. An unknown `kind` or `state`
falls back to the safe default; a non-string label is rejected outright.

## The restored section is read-only

`data-state="done"`, `data-restored="true"`, and **no Stop button**. The turn is
over, and a control that cannot act invites a click that does nothing, which is
worse than no control. It starts collapsed so it does not bury the answer, and
its note says plainly that live details such as timings are not kept.

## Verification

- browser wrapper gate: **146/146** (one new harness, 26 assertions);
- architecture gates: **429/429**, two new mutants, both caught:
  `restored-activity-not-revalidated` and
  `restored-activity-offers-a-dead-stop-button`;
- the harness drives capture against a DOM stand-in, asserts both bounds fire,
  and feeds the restorer tampered records: unknown kinds and states, non-string
  labels and details, oversized arrays, whitespace-only labels.

## Two test corrections

`transcript replay uses common user-turn renderer...` pinned the entire replay
meta object as one literal. It gained a field; the two contracts it actually
guarded — legacy `attachments` fallback and memory-only runtime — are now
asserted individually rather than as one string.

One assertion of my own was a tautology: `!/…/.test(x) === false`, which is true
whenever the pattern matches and false otherwise, but reads as a negation and
would have passed a rewrite either way. Replaced with the direct form.

## Remaining

- §18 conversation branching / version navigation — deferred, blast radius
  exceeds value.
- §20 virtualised long transcripts — deferred pending a measured problem;
  R173T11 and T14 both reduced the pressure behind it.
- Server-side git — deferred by design; push-only courier in R173T5.
