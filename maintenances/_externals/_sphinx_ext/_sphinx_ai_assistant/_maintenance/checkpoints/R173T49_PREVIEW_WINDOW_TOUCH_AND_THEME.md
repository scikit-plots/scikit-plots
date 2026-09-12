# R173T49 — The preview window on a phone, and in both themes

Status: COMPLETE

Base: R173T48 title bar order, same package.

## A bug in R173T47's own clamp

Driving the shipped clamp at phone widths found it:

```
viewport 280x400, requested 200x200  ->  width 320   (overflows by 40px)
```

`Math.max(_PREVIEW_MIN_W, Math.min(geom.width, v.w))` applies the 320px floor
**after** the viewport cap, so on any display narrower than 320px the floor
wins and the window is sized wider than the screen it is on. The minimum that
exists to keep the controls usable was pushing them off the edge.

The floor is now itself capped: `max(min(MIN, v.w), min(width, v.w))`. Verified
at 280×400, 320×568 and 200×300 — and that the minimum still applies where the
viewport can hold it, so the fix did not simply remove the floor.

This is the second time this run that arithmetic has been checked by executing
it rather than reading it, and the second time that found something.

## Touch is not a small pointer

A movable, corner-resizable window is a pointer idea. On a phone no hover
reveals the resize corner, the corner is smaller than a fingertip, and a
finger-dragged window is easy to strand and hard to recover.

Below 640px — **or on any coarse pointer, since a touch laptop has the same
fingertip problem at any width** — the preview stops pretending to be a window:
it fills the viewport, cannot be dragged, cannot be resized, and the header
gets `touch-action: auto` back so it scrolls the page rather than swallowing
the gesture. Controls grow from 1.85rem to 2.5rem.

Minimise and maximise stay, docked to the bottom edge: collapsing to the title
bar is still useful for glancing at the page behind, and it costs no gesture
that touch handles badly.

## Both themes, without depending on the host

The control hover used `color-mix(… var(--pst-color-surface, #fff) …)`. On a
site that never defines `--pst-color-surface`, the `#fff` fallback produces a
near-white wash under a light glyph on a dark page. The dark rule now states
its own values rather than relying on a token the host may not set.

## Verification

- browser wrapper gate: **149/149** (45 assertions in the owning harness);
- architecture gates: **542/542**, two new mutants, both caught:
  `minimum-size-exceeds-a-small-viewport` and
  `preview-window-draggable-on-touch`.
