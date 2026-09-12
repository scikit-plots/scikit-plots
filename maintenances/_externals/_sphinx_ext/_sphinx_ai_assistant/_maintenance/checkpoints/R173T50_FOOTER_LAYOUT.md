# R173T50 — The footer spends one row, not two

Status: COMPLETE

Base: R173T49 preview window touch and theme, same package.

## The waste

`.ai-assistant-panel-footer-note` and `.ai-assistant-panel-footer-credit` both
carried `width: 100%`, so each claimed a full line of the wrapping flex footer.
Two rows of small print at the bottom of a panel whose scarce dimension is
vertical.

They now share one line — the note takes the room left over (`flex: 1 1 15rem`),
the credit takes what it needs and sits at the trailing edge:

```
✨ The chatbot is an AI and can make mistakes.        Powered by scikit-plots
   Please double-check cited sources.
```

Below 26rem of **footer** width the note reclaims the line and the credit moves
underneath it, re-centred. Stacked because the two do not fit, not as a default
that then has to be undone.

Measured with a named container query rather than a viewport media query, for
the reason established in R173T39: this panel is resizable and can be docked or
maximized, so its width and the window's are different numbers.

## What I did not do, and why

The suggestion included revealing the full text on hover.

The disclaimer is the one piece of text in this panel a reader most needs to
have seen, and hover reveals nothing on a touch screen. Clipping a correctness
notice to one line and putting the remainder behind a pointer gesture would
make the footer look tidier and mean less — the same trade this run has
refused four times already under the invariant that the UI must not claim more
than it delivers.

So the note is never truncated and never hidden: `overflow: visible`,
`text-overflow: clip`, `white-space: normal`, all stated rather than inherited,
and the harness asserts each of them plus the absence of any hover rule that
would change its display, visibility or opacity. The mutant
`disclaimer-truncated-to-one-line` applies the abbreviated version and is
caught.

Where something had to give on a narrow panel, it is the **credit** that moves
— it is the less important of the two, and it is the one that can be shortened
without costing the reader anything.

## Verification

- browser wrapper gate: **150/150** (one new harness, 16 assertions);
- architecture gates: **547/547**, two new CSS mutants, both caught:
  `footer-spends-two-rows-on-small-print` and
  `disclaimer-truncated-to-one-line`.
