# R173T80 — Menu items carry a glyph

Status: COMPLETE

Base: R173T79 workspace tabs, same package.

## What each item now shows

The shared menu builder takes an optional `icon` per item, following the export
menu's shape: glyph, then a text stack of label and hint. Each glyph matches
what its action produces rather than being decorative filler:

| item | glyph |
|---|---|
| Open in a sheet | document |
| Save as… / Download as… / Download | download arrow |
| Download patch | git mark |
| Continue editing | chevron |
| Stop continuing | close |

The last pair is the one worth noting: `Continue` and `Stop continuing` are the
same row in two states (R173T27), so they take **different** glyphs. An icon
that stayed the same while the label flipped would be the one part of the row
still describing the old state.

## Two details

**The glyph is aria-hidden.** The accessible name is the label text, so
removing an icon changes nothing announced — decoration in the strict sense. Its
mutant makes it announced.

**The gutter is reserved whether or not an item has one.** With the column
sized to content, items with a glyph indent and items without do not; a
half-indented list is harder to scan than one with no icons at all. The glyph
also aligns to the label's first line rather than the middle of the item, or a
two-line hint drags it to the centre of a block it is meant to head.

## Verification

- browser wrapper gate: **153/153** (195 assertions in the owning harness);
- architecture gates: **646/646**, two new mutants, both caught:
  `menu-glyph-is-announced` and `menu-icon-gutter-collapses`.

A heredoc note for future turns: a `python3 - <<'EOF'` block whose payload
contains the word `EOF` at line start closes early and the shell eats the rest.
Delimiters here are now distinctive.
