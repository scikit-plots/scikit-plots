# R173T45 — Every code block is numbered

Status: COMPLETE

Base: R173T44 section title contrast, same package.

## The remaining gap

R173T14 numbered collapsed files; R173T43 numbered the preview overlay. What
was still unnumbered was everything in between: short files below the collapse
threshold, and **every snippet that never declared a path** — which is what
most answers are actually made of.

So a reader could cite a line in a large file and not in the snippet beside it.

`_numberRemainingCodeBlocks` numbers whatever the collapse pass did not, using
the same `_buildLineNumberedSheet`, so a line number means the same thing on
every surface: inline snippet, inline file, and preview overlay.

## Two guards against nesting

The collapse pass runs first and already sheets the files it collapses. The
numbering pass therefore checks both a `data-ai-line-numbered` marker **and**
whether the block's parent is already a sheet. Either alone would be enough
today; both are cheap, and the failure they prevent — a gutter beside a gutter —
is the kind that looks like a rendering bug rather than a logic one.

Ordering is asserted, not assumed: numbering runs after the collapse, and the
mutant that removes the parent check is caught.

## Same timing rule as the collapse

Finalization only. On the per-chunk path it would wrap a block whose fence is
still arriving and re-wrap it on every chunk — the failure R173T11 established
for the collapse, asserted here for numbering by the same means: the per-chunk
sync function must not mention it.

## Presentation

A snippet is part of the prose around it, not a document in its own right, so
its sheet keeps the code block's own radius rather than the file sheet's
squared top edge, which exists to meet a disclosure header.

The gutter inherits `font-size` and `line-height` from the code in every sheet
variant. Restating them lets the two drift the moment either is returned, and
drift in a gutter is visible as numbers sliding out of line with their rows.

## Verification

- browser wrapper gate: **148/148** (174 assertions in the owning harness);
- architecture gates: **525/525**, two new mutants, both caught:
  `snippet-blocks-left-unnumbered` and
  `numbering-nests-a-sheet-inside-a-sheet`.

Copy remains clean by construction everywhere: the numbers are never in the
`<pre>`, so selecting a block yields the code and nothing else.
