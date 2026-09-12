# R173T84 — Feedback workspace tabs share the proven icon + label composition

Status: COMPLETE

Base: R173T83 PDF prepared-mode thumb optical alignment.

## Reported UI issue

The `Feedback`, `Dataset contribution`, and `Activity` workspace tabs were
semantically valid but visually under-specified: each button contained only a
text node, while the nearby Conversation export format tabs use a stable
`icon + label` composition.  The mismatch made the three-button workspace read
like a separate, unfinished control family.

## Repair

The workspace tab factory now composes every tab exactly like the proven export
format buttons:

- one `ai-assistant-conv-share-format-icon` span (`aria-hidden=true`);
- one text-label span created with `textContent`;
- `Feedback` uses the internal `commentDiscussion` glyph;
- `Dataset contribution` uses the internal `dataset` glyph;
- `Activity` uses the internal `pulse` glyph.

The icons are trusted static entries from the existing `ICONS` registry and
inherit `currentColor`, so hover, selected, dark-theme, and forced-colour state
remain owned by the existing tab CSS rather than a new parallel palette.

T79's workspace-specific layout remains authoritative.  The three named tabs
stay content-sized/flex-wrapped and do **not** inherit the equal-width export
format grid merely because they now share the same button anatomy.

## Accessibility completion

While restructuring the button factory, the tab relationship is completed:

- stable tab IDs;
- matching panel IDs;
- `aria-controls` on each tab;
- `aria-labelledby` on each tabpanel;
- ArrowLeft/ArrowRight/Home/End roving keyboard selection and focus.

All selection paths converge on `_setWorkspaceTab`, so `aria-selected`,
`tabindex`, pane visibility, click selection, programmatic selection, and
keyboard selection do not develop separate state authorities.

## Verification

- T84 focused static composition/accessibility contract: **18/18**;
- existing T79 workspace layout contract: **11/11**;
- share-conversation static neighbor suite: **147/147**;
- JavaScript syntax: GREEN;
- T83 PDF-thumb regression: **8/8**;
- PDF layout neighbor: **16/16**;
- Panel trigger neighbor: **100/100**;
- Copy-mode neighbor: **13/13**;
- maintenance core: **35/35**;
- family maintenance gate: **2/2 GREEN**.
