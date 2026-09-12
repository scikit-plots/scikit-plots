# R173T54 — The composer owns the first row

Status: COMPLETE

Base: R173T53 escape ladder, same package.

## A regression introduced by R173T50

On a wide panel the footer laid out as:

```
| input-group | footer-note | footer-credit |
```

all three on one line, with the composer beside its own disclaimer.

The cause is a dependency that was never stated. `.ai-assistant-panel-input-group`
carried `flex: 1 1 auto`, so it only had a row to itself because the note and
the credit each carried `width: 100%` and were pushed off it. R173T50 made
those two shrinkable so they could share a row **with each other** — and in
doing so removed the thing that had been keeping the composer alone.

The two-row structure was a side effect of the footnotes not fitting, not a
property anything asserted.

## The fix states the intent

`flex: 1 1 100%` on the composer. Its row is now independent of what the
elements after it happen to be sized at:

```
| input-group                                              |
| footer-note                        | footer-credit       |
```

`min-width: 0` stays, so the composer can still shrink internally and the
attachment strip scrolls rather than widening the footer.

## What this run keeps demonstrating

A layout that works because of what a *neighbour* cannot do will break when the
neighbour changes, and nothing will point at the change. R173T50's edit was
correct in isolation and correct against its own gate; the property it broke
belonged to a different element and was written down nowhere.

The gate now asserts both rows — the composer's basis and that the note and
credit still share the second line — so neither half can be changed without the
other being re-examined.

## Verification

- browser wrapper gate: **152/152** (20 assertions in the owning harness);
- architecture gates: **563/563**, one new CSS mutant caught:
  `composer-shares-a-row-with-its-own-small-print`, which restores
  `flex: 1 1 auto` and reproduces the reported layout exactly.
