# R173T39 — Every download control carries a glyph, and narrow panels use it alone

Status: COMPLETE

Base: R173T38 icon button decorator, same package.

## Icons on all three download controls

`.ai-md-artifact-download-label`, `.ai-assistant-panel-changed-file-download`
and `.ai-assistant-panel-changed-files-download-all` now go through the shared
`_decorateIconButton` with the download arrow.

## The Git mark, in one colour

The patch export carries git's logomark rather than the download arrow: the two
footer actions produce different kinds of artifact and should not look
interchangeable.

Three fixed-colour variants were supplied — white, black, orange. The shipped
mark uses **`fill="currentColor"`** and none of them. The button already
inherits `--ai-artifact-accent`, which has a light value, a dark value and a
`forced-colors` fallback, so one asset gives correct contrast in every theme.
Shipping three copies would mean choosing between them at runtime and getting
the forced-colours case wrong, since that mode discards author colours
entirely and a hardcoded `#fff` would vanish into the background.

The mark keeps a slightly larger optical size than the arrows: it is a brand,
not an operation, and matches their weight only when drawn a little bigger.

## Icon-only below 22rem — measured on the panel

Labels collapse via `@container ai-artifact-surface (max-width: 22rem)`, not a
media query. The panel is resizable, maximizable and embeddable, so its
rendered width and the viewport's are different numbers: a media query would
collapse labels on a wide panel inside a narrow window and keep them on a
narrow panel inside a wide one. The pattern already exists here — R173's share
export section uses a named container for the same reason.

This is safe **only** because every one of these buttons already carried a full
`aria-label` and `title`. The accessible name never depended on the visible
text, so a screen reader announces *"Download latest docs/index.rst under its
own name"* at every width and a pointer user gets the same sentence on hover.

Two details:

- the label is **clipped, not removed**. `display: none` would take it out of
  the box, shrinking the hit area to the glyph's width — bad on a touch screen
  — and dropping the text a `title` would otherwise carry.
- dropping the label beats letting it truncate. `"Downl…"` is a worse
  affordance than a recognisable glyph, and truncation would also squeeze the
  filename beside it, which is the part that identifies the row.

## Verification

- browser wrapper gate: **148/148** (159 assertions in the owning harness);
- architecture gates: **505/505**, two new CSS-targeted mutants, both caught:
  `narrow-panel-labels-collapse-by-viewport` and
  `icon-only-label-removed-not-clipped`.

## A comment defeated an assertion again

`the label is clipped, not removed` matched `display:none` inside the rule's own
comment, which explains why it is not used. Comments are now stripped before
matching. This is the third time this run — the TOML null convention, the retry
"as-is" label, and now this. The rule is stable enough to state plainly:
**assertions against raw source must strip comments first**, because a comment
that explains why something is absent contains the thing it is absent of.
