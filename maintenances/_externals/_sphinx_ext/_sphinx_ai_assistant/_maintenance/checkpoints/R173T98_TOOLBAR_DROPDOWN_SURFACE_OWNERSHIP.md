# R173T98 — Toolbar dropdown surface ownership

Status: COMPLETE
Date: 2026-09-11

## Symptom

The top toolbar menu `#ai-assistant-dropdown` could be fully interactive and
correctly stacked while its computed `background-color` was transparent.  The
live DevTools capture also showed a malformed declaration with a `var(...)`
value followed by a second `#29313d` color.

## Root cause

Two generations of the same bug existed:

1. the deployed/live CSS contained two colors in one `background-color`
   declaration, so the browser discarded the property;
2. R173T96 accidentally changed the runtime dropdown to
   `var(--ai-speak-toggle-surface)`.  That custom property is owned by the
   in-panel speak toggle and is not defined on the toolbar dropdown or its
   ancestors.  At computed-value time the declaration therefore became invalid
   and the initial transparent background could leak through.

The T96 change was an ownership regression: mobile speak-control hardening
reused a component-local token outside its component boundary.

## Repair

The toolbar dropdown now owns `--ai-assistant-dropdown-surface` locally.

Light fallback chain:

`--pst-color-surface -> --color-background-primary -> #fff`

Dark fallback chain:

`--pst-color-surface -> #29313d`

The dark rule recognizes PyData `data-theme` / `data-mode`, Bootstrap
`data-bs-theme`, and `.dark` hosts.  `background-color` receives exactly one
value: `var(--ai-assistant-dropdown-surface, #fff)`.

## Regression controls

- focused dropdown surface harness: 9/9 GREEN;
- targeted mutations: 6/6 GREEN (three mutants × anchor/execution);
- registered Node/UI harness plane: 170/170 GREEN;
- mutation catalogue/anchors: 275/275 GREEN;
- deliberate mutants: 272/272 caught;
- maintenance core: 35/35 GREEN;
- family maintenance: 2/2 GREEN;
- AI maintenance checker: GREEN;
- independent review: PR_READY / release ELIGIBLE.

## Prevention rule

Component-local custom properties are capabilities, not global theme tokens.
Never reuse one outside the subtree that defines it.  A menu surface must own
its own semantic token/fallback, and `background-color` must contain exactly
one color value.  Dark-mode fallbacks must match the host framework's actual
DOM theme attributes.
