# R173T36 — Four destinations, four equal cards

Status: COMPLETE

Base: R173T35 quick model list complete, same package.

## The layout said something the feature does not

`.ai-assistant-conv-share-destinations` used `repeat(3, minmax(0,1fr))` for
four destinations, so the row always broke 3 + 1:

```
| Save file | Local preview | Self-contained link |
| Global link                                      |
```

Three cards of one width, and a fourth alone across the full row. That reads as
a hierarchy the destinations do not have — Global link is a **peer** of the
other three, not a summary of them, and the widest card in a chooser is read as
the recommended one.

## Two columns, not four

```
| Save file           | Local preview |
| Self-contained link | Global link   |
```

Two columns divide four evenly, and stay even at three (one empty cell) or two,
so no count of destinations produces an orphan. Four columns would have fitted
the current count and broken the moment a destination was added or hidden —
which the build already does conditionally.

`grid-auto-rows: 1fr` plus `height: 100%` on the card keeps both rows the same
height when one description wraps and another does not. That was the other half
of the imbalance: even before the orphan, the cards were different heights.

The narrow-panel rule stays at one column and now releases `grid-auto-rows`, so
a single column sizes each card to its own content rather than to the tallest.

## Verification

- browser wrapper gate: **148/148** (125 assertions in the owning harness);
- architecture gates: **493/493**, one new mutant caught:
  `share-destinations-orphan-the-fourth`, a **CSS-targeted** mutant restoring
  the three-column track — the capability added in R173T24, now earning its
  keep on a purely presentational contract.

The gate asserts the track, the absence of any three-column rule, the equal-row
behaviour, the card height that makes it visible, and the narrow fallback.
