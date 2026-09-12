# R173T40 — Three layout faults, three different causes

Status: COMPLETE

Base: R173T39 download icons and git mark, same package.

Two width faults and one vertical fault, fixed at their own levels rather than
with blanket overflow rules.

## 1. The tab switcher widened the sheet

`.ai-assistant-conv-share-format-switcher` already had `overflow-x: auto`, and
it did nothing.

A flex item's default `min-width: auto` refuses to shrink below its content, so
the switcher grew to fit all five format tabs and pushed the sheet wider
instead of scrolling. `min-width: 0` is the declaration that actually engages
the overflow; the `overflow-x` was never the missing piece.

Tabs are pinned `flex: 0 0 auto` so they scroll rather than squash — the format
name is the only thing distinguishing one tab from another, and a squashed tab
row truncates all five at once.

## 2. The artifact row overflowed instead of wrapping

`.ai-assistant-conv-share-artifact` is a flex line whose action buttons — Copy
link, Open, Download — do not shrink. With no wrap they pushed the row past its
container, and the part pushed out of view was the artifact **name**: the only
thing identifying which artifact the row is about.

The row now wraps and can shrink inside its own flex parent. Below the wrap
point the text claims the whole first line (`flex: 1 1 12rem`) so the buttons
land beneath it as a group rather than one per line.

## 3. The section title read as the preview's caption

`.ai-assistant-conv-share-section-title` carries `margin-top: .2rem`, tuned for
a title following ordinary text. After `.ai-assistant-conv-share-format-host` —
a bordered, full-height preview — that gap made "Destination" look like a
caption belonging to the preview rather than the heading of the next section.

An adjacent-sibling rule gives it `1rem` after the preview only. The base
spacing is unchanged everywhere else, and is asserted to still be `.2rem` so
this fix cannot quietly become a global change.

## Verification

- browser wrapper gate: **148/148** (143 assertions in the owning harness);
- architecture gates: **511/511**, three new CSS-targeted mutants, all caught:
  `tab-switcher-widens-the-sheet`,
  `artifact-row-overflows-instead-of-wrapping`, and
  `section-title-reads-as-the-preview-caption`.

The gate strips comments before matching, per the rule filed in R173T39.
