# R173T77 — The chevron is drawn from the picker's tokens

Status: COMPLETE

Base: R173T76 crowding and floating hint, same package.

## One control, two sets of values

The chevron and the model picker are joined into a single segmented control,
and were drawn from different tokens: the picker took
`--pst-color-border` and `--pst-color-text-muted, --color-foreground-secondary`,
while the chevron was transparent with `#71717a` and had a border only on touch.

At rest on a desktop — where nothing hovers — the pair read as a bordered
button with a bare glyph stuck to its side.

Both now use the picker's tokens. `border-inline-start: 0` on the chevron and
`border-inline-end` supplied once by the picker give the pair **one outline**
rather than two meeting in the middle, and the separate pseudo-element divider
is dropped so the hairline is not doubled.

## Two pieces of debris removed

A `border: 0` left over from the original chevron rule sat above the new border
declaration. The cascade made it harmless and the file misleading, which is
worse in a stylesheet this size than a rule that is simply wrong — a reader
finds the first declaration and stops. Removed, and asserted absent.

The duplicate rule was merged in-turn again: **seventh** appearance of that
habit, and the second caught in the same turn rather than a checkpoint later.

## Verification

- browser wrapper gate: **152/152** (88 assertions in the owning harness);
- architecture gates: **633/633**, one new CSS mutant caught:
  `chevron-drawn-from-its-own-tokens`.

A harness fixture named `more` collided with an existing identifier in the same
file. Renamed to say what it holds — that harness has accumulated enough
fixtures that generic names are now a hazard.
