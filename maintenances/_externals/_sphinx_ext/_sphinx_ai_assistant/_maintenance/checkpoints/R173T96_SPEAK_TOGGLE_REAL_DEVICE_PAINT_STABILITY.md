# R173T96 — Speak-toggle real-device paint stability

Status: COMPLETE

Base: R173T95 Presented-file responsive segment continuity.

## Reported UX issue

On some real mobile devices the speak-hint disclosure button remained clickable,
but the chevron could become effectively invisible. Desktop and responsive
emulation were not reliable reproducers. The captured production pages show the
failing surface in the collapsed state (`aria-expanded="false"`) under PyData's
real dark-theme contract (`data-theme="dark"` / `data-mode="dark"`).

## Root cause

The contrast repairs from R173T89/R173T91 were necessary but not sufficient.
The collapsed/expanded hint still used a fragile paint architecture:

- `.ai-assistant-panel-speak-row` had `height:0; min-height:0`;
- every child was lifted with `transform:translateY(-100%)`;
- the chevron SVG itself was also transformed for direction;
- the overflowing transformed children were expected to paint outside a
  zero-height flex container.

That is a valid hit-test tree but an unnecessarily fragile compositing tree for
mobile/WebKit/GPU rendering: the button can keep receiving taps while a painted
SVG layer outside the zero-height box is lost or rasterized inconsistently.

The colour contract was also more indirect than necessary. The button is drawn
on `--pst-color-surface`, but the icon used a generic text token/currentColor
chain instead of PyData's purpose-built `--pst-color-on-surface` foreground.
Explicit dark fallbacks also recognized Bootstrap/.dark conventions but not the
`data-theme` / `data-mode` attributes present in the captured production page.

## Repair

- Give the speak row a real positive paint box (`min-height:2rem; height:auto`).
- Cancel its flow cost with an equal negative block-start margin so the
  transcript still owns the reclaimed space without transformed children.
- Remove the `translateY(-100%)` lift from base and collapsed children.
- Add `isolation:isolate` to keep the floating row's stacking/paint local.
- Define a local semantic pair:
  - `--ai-speak-toggle-surface` from `--pst-color-surface`;
  - `--ai-speak-toggle-ink` from `--pst-color-on-surface`.
- Paint the SVG stroke directly from the owned ink, with explicit
  opacity/visibility, a 2.25 stroke width and non-scaling stroke behavior.
- On touch/coarse pointers keep a 2rem minimum target, `touch-action:manipulation`
  and the same explicit ink without relying on hover.
- Recognize PyData `data-theme="dark"` / `data-mode="dark"` as well as
  Bootstrap `data-bs-theme="dark"` and legacy `.dark` fallbacks.
- Keep `aria-expanded` responsible for chevron direction only, never paint.
- Keep forced-colors system-authoritative for both surface and SVG stroke.

## Verification

- T96 hardware-paint stability contract: **21/21**;
- T89 mobile visibility neighbor: **16/16**;
- T91 sticky-hover/state neighbor: **20/20**;
- speak collapse/expand behavior: **41/41**;
- registered Node/UI harness plane: **166/166**;
- mutation catalogue metadata + unique anchors: **267/267**;
- deliberate mutation execution: **264/264 mutants caught**;
- four new T96 hardware/mobile mutation controls: **4/4 caught**;
- JavaScript syntax: **GREEN**.

## Prevention

A clickable target is not proof that its pixels are safely composited. For
floating controls on mobile, avoid transformed descendants overflowing a
zero-height layout box when a normal positive paint box plus flow-cancelling
margin can express the same geometry. Pair surfaces with their semantic
`on-surface` foreground rather than depending on inherited/general text color,
and test theme selectors against the host framework's actual DOM contract, not
only compatibility conventions.

## Maintenance closure

- maintenance core: **35/35**;
- `_sphinx_ext` family gate: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.
