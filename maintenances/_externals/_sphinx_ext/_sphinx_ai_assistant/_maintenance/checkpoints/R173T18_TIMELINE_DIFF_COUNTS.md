# R173T18 — Diff counts on timeline rows

Status: COMPLETE

Base: R173T17 activity actually persists, same package.

## The last named gap

`Updated file preview: docs/index.rst` told a reader that a file changed and
nothing about how much. Learning whether that meant a typo or a rewrite
required scrolling past the answer to the file card below it — which defeats
the point of a timeline you scan to see what a turn did.

Live rows now carry `+84 −85` beside the path, built by the **same**
`_diffStatElement` the file card uses, so the two can never disagree about the
numbers.

## Restored rows say the same thing

A DOM element cannot be serialized. Its `aria-label` can — and it is already
the sentence a screen reader gets, so nothing new had to be invented: the stat
folds into the persisted label as `… — 84 lines added, 85 lines removed`.

Four properties are gated: the path stays first so the row remains scannable; a
row with no stat gains no invented numbers; a label that already carries the
stat is not given it twice; and the folded label still respects the persisted
120-character bound.

## A stub that could hide bugs, tightened

The persistence harness's row stub answered *any* selector containing `label`
with the label node and everything else with the detail node. When the
summariser started querying `.ai-assistant-panel-diff-stat`, it was handed a
detail node with no `getAttribute` — a crash, but the underlying problem is
worse than the crash: a stub that answers every selector with whichever node it
has handy tests a DOM that cannot exist.

It is now selector-accurate, returning `null` for anything it does not model.

## Verification

- browser wrapper gate: **147/147** (41 assertions in the owning harness);
- architecture gates: **440/440**, one new mutant caught:
  `timeline-file-row-drops-its-diff-counts`.

## State of the work

Every item raised across this run is now closed except two, both recorded in
`STATE.json` as decisions rather than tasks:

- **§18 conversation branching** — a list-to-tree restructure reaching
  persistence, restore, export, share, feedback and contribution. A product
  judgement.
- **Server-side storage of uploaded files for in-situ patching** — moves the
  privacy boundary from "this message plus the context you selected" to "files
  the server holds on your behalf", bringing per-reader identity, retention and
  deletion with it. The bounded shape is the push-only courier in R173T5.

Awaiting local test results.
