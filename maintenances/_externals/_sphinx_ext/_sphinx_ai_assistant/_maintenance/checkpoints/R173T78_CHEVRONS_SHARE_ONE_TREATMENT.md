# R173T78 — Both chevrons, one treatment

Status: COMPLETE

Base: R173T77 chevron shares picker tokens, same package.

## The same control, corrected in one place only

The model picker's chevron and the preview title bar's are the same kind of
thing: a small glyph button that opens a menu. R173T74 gave the first the
corrections a touch device needs — appearance reset, no tap delay, no grey
flash, a 44px target — and the second was left behind, so the same gesture
behaved differently depending on which surface a reader was on.

The shared corrections now live on a rule that **names both selectors**, and
the declarations were removed from the picker's own rule rather than copied.
Each property has one home, and a third chevron inherits them by being added to
that list instead of by someone remembering.

The preview chevron also gets the 44px coarse-pointer target: a thumb does not
aim differently because the control is in a dialog rather than a composer.

## What was deliberately not shared

The **segmented border** from R173T77 stays with the picker. That chevron is
joined to the model button and the two need one outline between them; the
preview chevron stands beside min/max/close and takes their treatment instead.

Sharing the parts that are genuinely common, and only those, is the difference
between a shared rule and a rule that has to be fought later — each control
matches the group it is actually in.

## Verification

- browser wrapper gate: **152/152** (91 assertions in the owning harness);
- architecture gates: **637/637**, two new CSS mutants, both caught:
  `preview-chevron-left-out-of-the-shared-treatment` and
  `preview-chevron-too-small-for-a-thumb`.

A grep suggested a duplicate rule for the preview chevron and there was none —
the match was the shared rule's second selector sitting at line start. Checked
rather than merged on the strength of a count, which after seven genuine
duplicates this run was the tempting move.
