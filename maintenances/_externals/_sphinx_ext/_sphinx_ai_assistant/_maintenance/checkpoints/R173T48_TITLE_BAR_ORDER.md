# R173T48 — Title, then the window controls

Status: COMPLETE

Base: R173T47 preview window, same package.

## What was wrong

The title bar rendered as `minimise maximise title close`. The window controls
were split around the thing they act on, and Close was separated from its two
peers.

The cause was ordering by construction site: `minBtn` and `maxBtn` were
appended to the header where they were built, which runs before
`header.appendChild(heading)`. Nothing was wrong with either control — the
order was simply a side effect of where in the function each one happened to be
created.

## Grouped, so it cannot recur

```
| title …                          | ⎽  ▢  ✕ |
```

All three now live in one `.ai-assistant-panel-attachment-preview-controls`
container appended after the heading. Order is stated in one place, and a
fourth control cannot land on the far side of the title by accident — which is
exactly what happened here.

The mutant `window-controls-split-around-the-title` restores direct header
appends and is caught, so the arrangement is guarded rather than merely tidy.

## Two details

Close takes the same 1.85rem box as its neighbours, so the group reads as one
row of equal controls rather than two shapes sitting together.

The heading shrinks and truncates. A long filename — and these are repository
paths — must never push Close off the edge; the control a reader reaches for
when something is wrong is the one that must always be reachable.

## Verification

- browser wrapper gate: **149/149** (32 assertions in the owning harness);
- architecture gates: **538/538**, one new mutant caught.

The order assertion checks the group membership *and* the sequence, and
separately asserts that no control is appended to the header directly — the
condition that caused the fault, rather than only its symptom.
