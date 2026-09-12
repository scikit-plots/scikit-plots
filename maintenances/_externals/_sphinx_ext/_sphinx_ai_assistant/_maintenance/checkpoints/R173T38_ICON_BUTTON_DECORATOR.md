# R173T38 — The bulk actions carry the download glyph

Status: COMPLETE

Base: R173T37 collapse section chrome, same package.

## What changed

The presented-files footer pair — `Download all N files` and
`Download patch series` / `Download patch` — now carries the same download
glyph the snippet `Download all` button has always had. Two bulk actions in the
same panel doing the same kind of thing looked like different kinds of thing.

## One decorator, three callers

`_decorateIconButton(btn, iconSvg, labelText)` builds the glyph and the label.
The snippet button's own two-line version was replaced by a call to it, so
there is one implementation rather than a original and a copy — the pattern
this run has applied to the segmented control, the overflow menu, the card
markup and the accent token.

Three properties the shared version pins that the inline version did not:

- the glyph is `aria-hidden`, and the accessible name comes from `aria-label`,
  so removing the icon changes nothing that is announced — decoration in the
  strict sense;
- `innerHTML` is used for the glyph, and only ever with an `ICONS` constant.
  The comment says so at the assignment, because that is the line a future
  reader has to justify;
- the button is cleared before decorating, so a second call replaces the label
  rather than appending a second glyph beside it. The footer's bulk control is
  relabelled on every queue change (R173T29), so this is reachable, not
  theoretical.

The glyph sizes in `em`, so it scales with its button's text rather than
holding a fixed pixel size across three differently sized surfaces.

## Verification

- browser wrapper gate: **148/148** (147 assertions in the owning harness);
- architecture gates: **501/501**, two new mutants, both caught:
  `icon-button-glyph-is-announced` and
  `icon-button-decorated-twice-duplicates-its-label`.

## One mutant retargeted

`one-file-export-called-a-series` anchored on `series.textContent = many ? …`,
which moved inside the decorator call. Retargeted to the ternary at its new
location; the contract it guards — a series of one file is called a patch — is
unchanged.
