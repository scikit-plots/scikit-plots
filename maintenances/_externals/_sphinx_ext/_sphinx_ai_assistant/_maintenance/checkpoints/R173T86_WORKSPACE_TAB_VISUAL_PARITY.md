# R173T86 — Workspace tab visual parity with Conversation format tabs

Status: COMPLETE

Base: R173T85 PDF/Panel/Copy thumb geometry parity.

## Reported UI issue

The three `Feedback`, `Dataset contribution`, and `Activity` workspace tabs had
already adopted the export-format tab icon + label anatomy in T84, but their
font rhythm and selected colour treatment still did not look like the proven
Conversation format buttons (for example the JSON tab).

## Root cause

Two independent CSS-context leaks remained:

1. `.ai-assistant-conv-share-format-btn` used `font: inherit` and set size and
   weight but did not own its line-height. Conversation format tabs live inside
   `.ai-assistant-panel-privacy-body` (`line-height: 1.55`), while the workspace
   tablist is a direct sheet child. The same button class could therefore
   compute different vertical typography solely from placement.
2. T79 added a workspace-only selected inset edge using
   `--ai-artifact-accent`. The canonical format button already owns selected
   background, border, text/icon colour, hover, focus, dark mode, and forced
   colour behavior. The extra edge created a second colour authority and made
   the workspace look different from the reference component.

## Repair

- The shared `.ai-assistant-conv-share-format-btn` now owns
  `line-height: 1.55`, matching the previously correct Conversation-format
  computed rhythm while removing host-context dependence.
- The workspace-specific selected `box-shadow` and forced-colour override are
  removed.
- `.ai-assistant-panel-feedback-workspace-tabs` keeps only the layout divergence
  that is actually required: flex row/wrap and content-sized tabs.
- Font, neutral colour, selected background/border/text colour, hover, focus,
  dark mode, and icon colour now come from the exact same canonical button
  rules as JSON/HTML/Text/YAML/TOML.
- T84 icon composition and tab ARIA/keyboard behavior are unchanged.

This supersedes only T79's extra selected-edge treatment. T79's content-sized
workspace layout remains authoritative.

## Verification

- T86 visual parity contract: **14/14**;
- T79 workspace layout neighbor: **11/11**;
- T84 icon/ARIA composition neighbor: **18/18**;
- Share conversation neighbor: **147/147**;
- Feedback workspace parity: **19/19**;
- targeted workspace mutation controls: **4/4**;
- JavaScript syntax: GREEN.
