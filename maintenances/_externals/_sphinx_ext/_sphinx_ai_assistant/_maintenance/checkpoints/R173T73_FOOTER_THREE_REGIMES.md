# R173T73 — Three footer regimes, two of them side by side

Status: COMPLETE

Base: R173T72 one scroller per menu, same package.

## What R173T50 left

That checkpoint gave the footer two states: side by side, or stacked. The
stacked one arrived far too early, so a medium panel — the common case on a
tablet, or a desktop panel dragged narrower — dropped straight to three rows
while there was still room for two columns of wrapped text.

```
wide     | long text                    short text |
medium   | long   |  short |     (both wrapped into columns)
tiny     | long text |
         | short text |
```

## Two changes, both about what forces the stack

**The credit stopped being unbreakable.** `white-space: nowrap` made it a solid
~11rem block, so the pair failed as soon as the note could not also fit beside
it — the credit was deciding the layout for both. Allowed to wrap, "Powered by"
and the link sit on two lines and the pair survives to a much smaller panel.

The **link** keeps `nowrap`: breaking a project name across lines makes it read
as two words, and it is short enough never to need it.

**The note's basis dropped from 15rem to 10rem.** The basis is what decides
when two columns stop fitting; at 15rem the footer gave up while two narrow
columns were still viable.

Together these move the stack threshold from 26rem to 20rem, and the stack
becomes the last resort rather than the second option. Below 20rem two columns
of a few words each are narrower than the words themselves, and the note —
the part that must be read — would wrap to five or six lines to make room for a
credit.

## Verification

- browser wrapper gate: **152/152** (25 assertions in the owning harness);
- architecture gates: **621/621**, two new CSS mutants, both caught:
  `credit-forces-an-early-stack` and
  `footer-stacks-while-two-columns-still-fit`.

Both mutant anchors had to be widened: `flex: 1 1 10rem` appears elsewhere in
the stylesheet, and R173T50's anchor still named the old basis. The uniqueness
check caught both before either could sit in the catalogue matching nothing.
