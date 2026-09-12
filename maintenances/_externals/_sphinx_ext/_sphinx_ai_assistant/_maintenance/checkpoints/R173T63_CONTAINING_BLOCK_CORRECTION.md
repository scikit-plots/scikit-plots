# R173T63 — `position: fixed` is not always the viewport

Status: COMPLETE

Base: R173T62 menu sits against its trigger, same package.

## The claim R173T62 rested on was false

That checkpoint said, in its own words: *"Fixed coordinates have one containing
block, the viewport, which is also the thing that clips the menu."*

They do not, when an ancestor has a `transform` — and `.ai-assistant-panel`
carries `transform: translateY(0) scale(1)` on its open state. A transformed
ancestor becomes the containing block for `position: fixed` descendants as well
as absolute ones.

So `getBoundingClientRect()` produced viewport coordinates, and those were
written into a **different** coordinate space. The menu landed offset by the
panel's own position, and the offset moved with the panel — the reported
symptom exactly: a menu whose bottom (968px) exceeded the viewport (911px).

R173T58 is the same fault. That one assumed a containing block the element did
not have; this one assumed the viewport was the containing block when an
ancestor had claimed it. Both were assumptions about layout written as
comments and never measured.

## The correction

The menu is pinned at `(0, 0)` and measured. That rect **is** its containing
block's origin in viewport terms, whatever the containing block turns out to
be. Every coordinate is computed in viewport space and converted once, at the
point of writing.

This is correct with a transform, without one, and if the panel gains or drops
one later — which matters more than either case, because nothing in the
stylesheet announces that a transform governs a menu three subtrees away.

## Measured, not read

The gate drives the routine twice with a containing block at the viewport
origin and at `(120, 80)`, and asserts the written coordinates differ by exactly
that offset while the resulting on-screen position is identical, and inside the
viewport. Its mutant removes the conversion.

## On the CSS Anchor Positioning suggestion

Not adopted, though the diagnosis behind it was right and is what fixed this.
`anchor-name` / `position-anchor` are not available in every browser this panel
ships to, so it would need the JavaScript path anyway as a fallback — leaving
two positioning systems where the bug was that one of them computed the wrong
number. The `!important` overrides would also have made the stale-inline-style
problem invisible rather than absent.

`box-sizing: border-box` from that patch **was** adopted: without it the
computed `max-height` clips the content by exactly the menu's own chrome.

## Verification

- browser wrapper gate: **152/152** (102 assertions in the owning harness);
- architecture gates: **585/585**, with the retargeted
  `menu-placed-against-an-assumed-containing-block` caught.
