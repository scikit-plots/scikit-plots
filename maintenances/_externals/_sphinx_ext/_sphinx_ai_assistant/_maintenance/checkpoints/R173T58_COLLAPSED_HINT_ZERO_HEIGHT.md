# R173T58 — The collapsed hint comes back

Status: COMPLETE

Base: R173T57 collapsed hint leaves the flow, same package.

## What R173T57 broke

The pill vanished entirely when collapsed.

The cause was an assumption stated in that checkpoint's own comment and never
checked: *"the footer is the positioning context"*. It is not. The row is
`panel.appendChild(speakRow)` — a **sibling** of the footer and a child of the
panel. So `position: absolute; bottom: calc(100% + 0.35rem)` resolved `100%`
against the **panel's full height** and placed the pill above the top of the
panel. It did not move; it left.

`.ai-assistant-panel-footer { position: relative; }` was added in the same
checkpoint to serve a child the footer does not have, and did nothing.

## The fix needs no positioning ancestor

The row keeps its place in the panel's column and collapses its own height to
zero; its contents are lifted clear with `translateY(-100%)`.

```
| …the answer continues to the end of this line.        [ 🎤 › ] |
```

This is better than a corrected absolute position, not merely equivalent:
sitting between the transcript and the footer in normal flow, the pill lands
just above the composer **however tall the composer has grown**. That was the
property the absolute version was reaching for and computing wrongly.

`overflow: visible` keeps the lifted contents from being clipped by the
zero-height box, and the transcript's reserved end padding — unchanged from
R173T57 — is the space the pill is lifted into.

## Why the gate did not catch it

Every R173T57 assertion was about the stylesheet, and the stylesheet said
exactly what it was asserted to say. The wrong part was a claim about the
**DOM**: which element was the positioning ancestor. That is checkable —
`panel.appendChild(speakRow)` is one grep — and no assertion looked.

The harness now asserts the absence of the failed approach as well as the
presence of the working one: no `position: absolute`, no percentage resolved
against the wrong box, and no positioning context added for a child the footer
does not have.

## Verification

- browser wrapper gate: **152/152** (32 assertions in the owning harness);
- architecture gates: **571/571**, one mutant retargeted and one added:
  `collapsed-pill-not-lifted-out-of-its-box`.
