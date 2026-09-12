# R173T46 — One scroller per sheet, and no wrapping beside a gutter

Status: COMPLETE

Base: R173T45 number every code block, same package.

## Two faults, both introduced by R173T43

Putting a gutter beside a block that already scrolled itself broke in two ways
at once, and the 5543-line `.ipynb` preview showed both.

**The sheet stretched to the length of the file.** The `<pre>` carried
`max-height: min(70vh, 28rem)` and `overflow: auto`, so it scrolled inside a
fixed box. The gutter had neither, so it rendered all 5543 numbers at full
height. The sheet took the taller of the two, and the numbers slid out of step
with the code the moment either scrolled — two scroll positions for one
document.

The **sheet** is the scroll container now. Gutter and code grow to their
natural height inside it and move under one scrollbar, so they cannot
disagree. Inside the preview overlay the sheet defers entirely: the overlay
body is already a scroller, and a sheet scrolling within it would give the
reader two vertical scrollbars for one file.

**Wrapping desynchronised the numbers.** `white-space: pre-wrap` is wrong
wherever numbers are shown: a logical line that wraps to three visual rows
fills three line boxes in the code and one in the gutter, so every number after
the first wrapped line is off by the accumulated overflow. Numbered code does
not wrap — it scrolls horizontally, and the gutter stays put because that
scroll belongs to the code alone.

`overflow-wrap` and `word-break` are reset alongside `white-space`, because the
inherited rules set all three and changing one leaves the others breaking long
tokens.

## A duplicate rule, merged

The fix was first written as a second `.ai-md-file-sheet` block appended after
the original. Two rules for one selector meant the cascade had to be read to
know what the sheet does — and an assertion matched the *first* block and
passed against a rule with no overflow in it, which is exactly the confusion
that arrangement causes. Merged into one rule; the harness now asserts the
selector is defined once.

## Verification

- browser wrapper gate: **148/148** (181 assertions in the owning harness);
- architecture gates: **529/529**, two new CSS mutants, both caught:
  `numbered-code-scrolls-itself` and
  `numbered-code-wraps-and-desyncs-the-gutter`.

The wrap mutant needed its anchor widened twice: `white-space: pre;` appears
five times in the stylesheet and the three-declaration run appears twice, so
the anchor now includes `overflow-x: auto` to be unique to this rule.

## Not in this checkpoint

The requested preview **window controls** — minimise, maximise, move, resize
from the edges — are not built. The header already carries
`data-drag-handle="true"` with no drag behaviour behind it anywhere in the
file, so that attribute is currently a promise nothing keeps.

It is deliberately separate: the fault above made a long preview unusable and
was worth shipping on its own, whereas window management is a feature with its
own state (position, size, mode), its own persistence question, and its own
constraint that a dragged dialog must never be moved somewhere it cannot be
dragged back from.
