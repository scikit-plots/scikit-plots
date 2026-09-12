# R173T20 — Snippet card previews; download is its own control

Status: COMPLETE

Base: R173T19 snippet card overflow, same package.

## The behaviour

Clicking anywhere on `.ai-md-artifact-card` downloaded the snippet. The only way
to see what a snippet contained was to put a file on disk and open it — which is
the wrong default, because a quick check before committing to a download is what
a reader wants most of the time, and it is the cheaper of the two actions to get
wrong.

The card now opens the same attachment preview a tracked file uses, badged
`SNIPPET` and marked turn-scoped so it never reads as a tracked file. The
`Download` label became a real button beside it.

Row shape, matching the tracked-file card:

```
| icon + name + type   (preview)   | Download | ⋮ |
```

## Why Download is a sibling, not a nested span

`<span class="ai-md-artifact-download-label">` sat *inside* the card button.
Making that span clickable would have nested a button inside a button, which is
invalid HTML — browsers resolve it by dropping one of the two click targets, and
which one varies by engine. Two independent actions need two independent
controls.

The sibling form also gives Download its own focus stop and its own accessible
name, so a keyboard or screen-reader user gets both actions instead of one
ambiguous one. Mutant `snippet-download-nested-inside-the-card` re-nests it and
is caught.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **446/446**, two new mutants, both caught:
  `snippet-download-nested-inside-the-card` and
  `snippet-card-click-downloads-instead-of-previewing`;
- eight assertions pin the split, including that the card no longer calls
  `_downloadBlob` on click and that the preview card takes the room the two
  controls leave.

## Consistency across the two surfaces

Snippet cards and tracked-file cards now share the same gesture — big target
previews, a named button downloads, ⋮ holds the rest — and, since R173T19, the
same menu implementation. A reader learns it once.
