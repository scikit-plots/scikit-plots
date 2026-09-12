# R173T21 — Preview and Download become one segmented control

Status: COMPLETE

Base: R173T20 snippet preview first, same package.

## The correction

R173T20 split the card into a preview target and a Download button, which was
the right behaviour and the wrong presentation: Download rendered as a second
peer button sitting next to the card, so the row read as two unrelated actions
rather than one artifact with two things you can do to it.

They are now one segmented control:

```
| icon + name + type (preview) │ Download | ⋮ |
```

The group owns the border and radius; each segment sheds its own chrome. The
divider is a real element rather than a border on a segment — a border would
move or disappear with whichever segment is hovered or focused.

## Still two buttons, and that is not negotiable

The joining is presentational only. Nesting Download inside the card button
would be invalid HTML, and browsers resolve that by dropping one of the two
click targets — which one varies by engine. Two independent actions need two
independent controls.

`role="group"` with the filename as its accessible name is what carries the
relationship to assistive technology, so a screen reader announces two buttons
belonging to one artifact rather than two unrelated ones. `:focus-within` puts
keyboard focus on the group, so the highlight matches what a sighted reader
sees as a single control. Mutant `snippet-group-loses-its-accessible-name`
strips the role and is caught.

## An assertion re-expressed, not loosened

`the download button is a sibling of the card, never nested inside it` pinned
the literal adjacency `cardRow.appendChild(card); cardRow.appendChild(dlBtn);`.
Both moved into the group wrapper, so the adjacency changed while the contract
did not. It now asserts the contract directly — both are appended to the group,
and `card.appendChild(dlBtn)` appears nowhere — which is what the assertion
always meant. Adjacency was only ever a proxy for it.

## Verification

- browser wrapper gate: **147/147** (42 assertions in the owning harness);
- architecture gates: **448/448**, one new mutant caught, one retargeted;
- four presentation contracts asserted against the paired stylesheet: the group
  owns the border, segments shed their chrome, the divider is its own element,
  and focus is visible on the group.
