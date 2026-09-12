# R173T55 — The speak hint collapses instead of disappearing

Status: COMPLETE

Base: R173T54 composer owns its row, same package.

## Correcting R173T52

That checkpoint made the hint dismissable and remembered the dismissal. It
recovered the row, and it was the wrong shape.

The row is onboarding for a shortcut. Once removed there was no way back to it
short of a new session — no control, no menu entry, nothing. **A hint that can
only ever be destroyed is one a reader will not risk putting away**, so the
affordance that was supposed to reclaim space went unused by exactly the
cautious readers who most wanted the space.

## Collapsed, not gone

```
expanded    [ 🎤  Speak with your assistant   Space   ‹ ]
collapsed   [ 🎤 › ]
```

Collapsed it keeps the glyph, the hit area and the accessible name, at the
width of one icon. Nothing is lost, so there is nothing to regret — and the row
costs almost no height either way, which is what the original request was
about.

The label follows: `Show the speak hint` / `Collapse the speak hint`. A control
that announces only its current state leaves a screen-reader user guessing what
activating it does, so the name states the action the next press performs while
`aria-expanded` carries the state.

The text is **clipped, not removed**, so the banner keeps its hit area and its
`aria-label` is untouched — collapsed, the button still announces *"Speak with
your assistant"*.

## The height rule went with it

R173T52 suppressed the row entirely below 620px of viewport height. That is no
longer needed: the reader can collapse it, and a rule that removed it on their
behalf would take away the same choice this checkpoint just gave them.

## Verification

- browser wrapper gate: **152/152** (one harness replaced, 17 assertions);
- architecture gates: **563/563**, two mutants replaced:
  `speak-hint-collapse-not-remembered` and
  `speak-toggle-label-states-only-its-state`.

The dismiss harness and its two mutants were deleted rather than left passing
against removed code — a gate for behaviour that no longer exists is noise that
looks like coverage.
