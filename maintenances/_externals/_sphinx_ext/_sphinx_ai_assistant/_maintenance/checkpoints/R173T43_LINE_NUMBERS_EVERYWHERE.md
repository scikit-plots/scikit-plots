# R173T43 — Every preview is numbered, by one gutter

Status: COMPLETE

Base: R173T42 toggle forced colours, same package.

## The gap

R173T14 gave the **inline** file view a line-number gutter. The attachment
preview overlay — which is where a long file is actually read — had none.

A reader asking for a change at a particular line has to be able to read that
number off the thing they are looking at. Numbering the surface used for a
glance while leaving the surface used for reading unnumbered is the wrong way
round.

## One gutter, two surfaces

`_buildLineNumberedSheet(pre, text, sheetCls)` was extracted from the T14 code
and is now called by both. Two gutters would drift in exactly the properties
that stay invisible until they are wrong: the `aria-hidden`, the digit width,
the line counting. The harness asserts there is exactly one gutter
implementation in the file.

The builder wraps the `<pre>` **in place**, inserting the sheet where the block
already sat, so neither surface has to know where its block lives among its
siblings.

## The rule that made T14 worth reusing

Numbers live in their own element and never enter the `<pre>`. Prefixing each
line is the usual shortcut and it poisons every copy, every download and every
patch taken from the block — selecting the whole sheet must yield the file, not
the file with a number welded to each line. `aria-hidden` keeps the gutter out
of the accessibility tree, because a screen reader announcing *"one import sys
two import os"* is worse than no numbers at all.

For the overlay the gutter inherits the code's `font-size` and `line-height`
rather than restating them. Restated values drift the moment either side
changes, and drift in a gutter is visible as numbers sliding out of line with
their rows — the failure this whole arrangement exists to avoid.

## Verification

- browser wrapper gate: **148/148** (166 assertions in the owning harness);
- architecture gates: **517/517**, one new mutant caught
  (`overlay-preview-loses-its-line-numbers`) and
  `line-numbers-written-into-the-code` retargeted to the shared builder.

## The new mutant had to be rewritten first

It initially wrapped the overlay call in `if (false)`, and the assertion — a
substring search for the call — still matched a call that could never run. It
now deletes the statement.

That is the second time this run: the rule filed in R173T27 is *assert the
guard, not just the call*, and its corollary is that a mutation which disables
rather than deletes tests the assertion's spelling instead of the behaviour.
