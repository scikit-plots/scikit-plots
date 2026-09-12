# R173T16 — Transcript scale measured, and the file card reduced to one menu

Status: COMPLETE

Base: R173T15 activity persistence, same package.

## §20 closed on evidence, not deferred again

Virtualised transcripts were deferred three times "pending a measured
rendering problem". Deferring because measurement says it is unnecessary and
deferring because it looks expensive are indistinguishable from outside, so
this checkpoint supplies the measurement and keeps supplying it as a gate.

Measured at the ceiling the panel enforces (`_TRANSCRIPT_MAX_TURNS_DEFAULT`,
200), worst case every turn carrying a 600-line reStructuredText file:

| shape | elements | `_mdToHtml` |
|---|---|---|
| prose only | 2,400 | ~8 ms |
| file every 4th turn | 2,550 | ~11 ms |
| file every turn | 3,000 | ~29 ms |

**A fenced file is one `<pre><code>` however many lines it holds.** 200 files of
600 lines add 600 elements, not 120,000. Virtualising would buy little: the DOM
node count is not where the size goes.

Where it did go was persisted bytes, and R173T14 already acted:

```
persisted data-raw   before=1.42MB  after=0.10MB  (-92.7%)   ceiling=2.00MB
```

Both fit, so the elision was headroom rather than a fix for breakage — worth
stating precisely, because R173T15 spent that headroom on activity persistence
and the claim should rest on a number.

The gate asserts element count and persisted size, both deterministic. Timing
is printed for information only: a timing assertion in CI measures the runner's
load as much as the code's cost.

## The file card is now preview | download | ⋮

The card had reached five visible controls. Preview and Download are what
almost every reader wants; the rest are for readers who already know they want
them. Everything else moved behind one ⋮ menu — the same affordance the panel
subbar already uses, so a reader who has met it once has met it here. The
separate "Patch · Continue" disclosure row is gone.

Menu items, in one extensible list rather than more buttons:

- **Open in a sheet** — full view with line numbers.
- **Save as…** — download under a chosen name.
- **Download patch** — apply with `git am`.
- **Continue editing** — attach to the next message.

**Preview stays the default and the sheet is the escape hatch.** Quick preview
answers most questions; the sheet is for the cases it does not — a long file, a
multi-step review, a small screen. Offering both as equal peers would make the
reader choose before they know which they need.

Sheet mode reuses the same viewer with `{ sheet: true }` rather than adding a
second one. Two viewers could disagree about what a file contains, which is the
one thing a preview must never do.

Menu hygiene is asserted, not assumed: `role="menu"`/`menuitem`,
`aria-haspopup` and `aria-expanded` on the trigger, Escape closes and returns
focus to the trigger, both capture-phase document listeners are removed on
close, and only one file menu can be open at a time.

## data-raw: unchanged, and that is the right answer

Snippet answers keep their full `data-raw` — the fenced code is the answer, and
quick copy/paste is exactly what it is for. Only large path-bearing file bodies
elide (R173T14). The two modes were already split along the line this
checkpoint's review confirms.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **434/434**, two new mutants, both caught:
  `file-menu-leaks-document-listeners` and `file-menu-escape-strands-focus`;
- four assertions relocated rather than deleted: the controls moved into the
  menu, so each is now asserted at its new home.

## A loose assertion caught by its own mutant

`closing the menu removes its document listeners` checked only that the string
`document.removeEventListener` appeared somewhere in the closer. The mutant
deleted one of the two removals and the assertion still passed, because the
other remained. Both are now asserted by name, with their registration
asserted too.

## Open

**§18 conversation branching** is the only remaining item from the original
design note, and it stays open as a product judgement rather than a maintenance
task.

**Server-side storage of uploaded files** was raised again: keep a large
uploaded file on the server and apply patches in situ rather than resending it
each turn. That is the same privacy-boundary decision recorded in R173T5 — it
moves the boundary from "this message plus the context you selected" to "files
the server holds on your behalf", and needs per-reader identity, retention and
deletion. The bounded shape remains the push-only courier described there. It
should not be absorbed into a maintenance run.
