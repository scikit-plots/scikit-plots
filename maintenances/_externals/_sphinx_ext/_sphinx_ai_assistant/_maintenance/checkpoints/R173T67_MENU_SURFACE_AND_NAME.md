# R173T67 — One builder, four surfaces: bounds and name corrected

Status: COMPLETE

Base: R173T66 composer is a floor, same package.

Two faults, both from one builder serving four unrelated surfaces. Both were
reported from the live DOM, where two menus with the same class had visibly
different geometry rules.

## 1. Bounded by the wrong surface

```
aria-label="File actions"          max-width: 1193px
aria-label="Try a different model" max-width:  978px
```

The placement routine bounded every menu by `.ai-assistant-panel`. The preview
dialog is a **floating window of its own** — draggable, resizable, frequently
wider than the panel and placed elsewhere — so a menu opened from its title bar
was clamped to a box its trigger was not in. That 1193px ceiling is the panel's
width leaking into a dialog's menu.

The routine now takes the **nearest** enclosing surface: the preview dialog if
the trigger is inside one, otherwise the panel. Ordered nearest-first, so a
transcript menu still picks the panel.

## 2. Named after one of its callers

Every menu carried `ai-assistant-panel-changed-file-menu` — the class of the
first surface that needed one. Three later callers inherited it, so a rule
written for "the file menu" silently restyled the preview title bar and the
model picker, which is exactly the coupling reported: *"one of change affect
other"*.

Menus now carry `ai-assistant-menu` **as well**. The neutral name is what the
builder actually produces and is where new shared rules go; the legacy name is
kept because it is in the shipped DOM and in every existing rule, and removing
it would be a silent breaking change for anything selecting on it.

Surface-specific rules are scoped through the caller's own class
(`.ai-assistant-panel-attachment-preview .ai-assistant-menu`), so changing one
surface cannot reach the others.

## Why not rename outright

A rename would have been cleaner and is not worth the blast radius here: the
old name appears in shipped markup, in the stylesheet, in harness assertions
and in mutant anchors. Adding a name costs nothing and removes the coupling;
removing one costs a coordinated change across four surfaces to fix a problem
that is already fixed.

## Verification

- browser wrapper gate: **152/152** (130 assertions in the owning harness);
- architecture gates: **601/601**, two new mutants, both caught:
  `preview-menu-bounded-by-the-wrong-surface` and
  `menus-share-only-a-caller-specific-name`.
