# R173T35 — The quick model list stops hiding half the models

Status: COMPLETE

Base: R173T34 file menu edge aware, same package.

## The report, and what was actually happening

Twelve configured models, six shown. Investigated before changing anything:
`_quickModelCandidates` filters only `disabled` and hidden-builtin entries —
**stubs and custom models are not filtered** — so all twelve reached the list
builder, which then applied a hardcoded `if (quickDisplayModels.length >= 6)
return`.

The truncation was silent, and that is the defect rather than the number. A
twelve-model configuration rendered identically to a six-model one, under a
heading reading *"Try a different model"* that was quietly answering *"these
six"*. Nothing distinguished the shown from the hidden: the cut fell wherever
the configured order happened to put it, so which six a reader could reach was
an accident of list order.

## Length is a scrolling problem, not a truncation problem

The cap is gone. Every candidate is listed, active model first and never
twice. The list is height-clamped to `min(50vh, 18rem)` and scrolls, so a long
configuration costs a scroll rather than six missing entries — and an ordinary
configuration never scrolls at all.

`overscroll-behavior: contain` stops a flick inside the list from scrolling the
transcript behind it, and the heading is sticky so a reader who has scrolled
still knows what the list is. The bubble menu was already edge-aware via
`_positionAnchoredPopupWithinPanelBody`, so a taller list is placed and clamped
by the same routine R173T34 gave the file menu.

## The gate that pinned the cap did its job

`quick menu is bounded to six choices` failed the moment the cap was removed —
correctly, since it asserted the old contract. It is rewritten to the corrected
one: the active model is listed first and not duplicated, **no** length bound
appears, and the scrolling behaviour is asserted against the stylesheet.

Two assertions were added for the report's own premise, so a future filter
cannot reintroduce the symptom from the other end: stubs are not filtered from
the candidate list, and custom models are candidates.

## Verification

- browser wrapper gate: **148/148** (34 assertions in the owning harness);
- architecture gates: **491/491**, one new mutant caught:
  `quick-model-menu-silently-truncates`, which restores the cap.
