# R173T42 — Toggle parity investigated; forced-colours gap closed

Status: COMPLETE

Base: R173T41 format switcher grid, same package.

## The reported shift could not be reproduced from source

`.ai-assistant-pdf-toggle-thumb` was reported as shifted relative to
`.ai-assistant-panel-toggle-thumb`. Both were compared properly rather than
patched on a guess:

- **Markup**: identical. Both build
  `span.ai-assistant-mic-toggle-track .ai-assistant-<x>-toggle-track` with the
  thumb appended inside it, so both thumbs resolve against a `position:
  relative` track.
- **Cascade**: enumerated every rule reaching each thumb with its specificity.
  Identical in both cases — the shared mic base at (0,1,0), the per-switch
  override at (0,2,0) with `top:1px left:1px 14×14`, and the checked transform
  at (0,3,0) with `translateX(16px)`.
- **Geometry**: a 34px border-box track with 1px borders gives a 32px inner
  width; a 14px thumb at `left:1px` translated 16px leaves 1px on both sides.
  Symmetric for both switches.

No change was made on that basis. A CSS edit that cannot be traced to a cause
is a guess that happens to be checked in, and this run has already filed the
lesson about asserting defects from reading rather than measuring.

**What would settle it**: which state the shift appears in (checked or
unchecked), which theme, and whether the offset is vertical or horizontal.
Those three answers identify the rule; without them any edit is speculative.

## What the comparison did find

`@media (forced-colors: active)` covered the **PDF** switch and nothing else.
The panel switch and the shared mic switch had no forced-colours treatment at
all.

In those modes author colours are discarded, so a track drawn as
`rgba(0,0,0,.22)` and a thumb drawn `#fff` both collapse toward the system
background. The switch becomes a pill with no visible state — and on-or-off is
the single thing a switch exists to communicate.

Both now use system keywords: `ButtonText` border on `Canvas`, `Highlight` when
checked, `HighlightText` thumb. The thumb also gains a `Canvas` border, because
its drop shadow is discarded too and the shadow was what separated it from a
filled track.

## Verification

- browser wrapper gate: **148/148** (45 assertions in the owning harness);
- architecture gates: **515/515**, one new CSS mutant caught:
  `panel-toggle-invisible-in-forced-colours`.

## An assertion the mutant corrected

`the checked track uses a system colour` searched the joined forced-colours
blocks for `background: Highlight`. The PDF block has its own, so the
assertion passed while the panel toggle's rule had been deleted. It now names
each toggle's selector in the match.

Same shape as the T33 and T27 corrections: a search across combined text finds
another construct's copy of what it is looking for.
