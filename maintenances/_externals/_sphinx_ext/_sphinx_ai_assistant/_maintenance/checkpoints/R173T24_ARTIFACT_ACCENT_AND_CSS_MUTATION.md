# R173T24 — One artifact accent, and mutation reaches the stylesheet

Status: COMPLETE

Base: R173T23 shared artifact card markup, same package.

## Four controls, one idea

`Download` on a snippet, `Download` on a presented file, `Download all N
files`, and the `Presented N files` heading all express one thing — *take this
file* — and had drifted into three different colours: the snippet label carried
the theme accent, the two others inherited body text, and the heading was plain
bold.

They now read one token:

```css
:root { --ai-artifact-accent: var(--pst-color-primary, var(--color-brand-primary, #2980b9)); }
[data-bs-theme="dark"] { --ai-artifact-accent: #93c5fd; }
```

A token rather than a repeated fallback chain, so a theme that sets
`--pst-color-primary` moves all of them together and a fifth control opts in by
adding its selector rather than by copying three levels of `var()`. The dark
theme carries its own value: the light accent fails contrast on a dark surface,
and a colour that is merely readable is not the same as one that was chosen.

Hover and focus keep the accent — a control that loses its colour at the moment
the reader commits to it reads as disabling itself. Forced-colours modes get
`LinkText`; every one of these is a real button or heading, so the meaning
survives in the element when the hue is discarded.

## The regression that caused it

R173T20 added `color: inherit` to `button.ai-md-artifact-download-label`. That
rule sits later in the stylesheet than the bubble-scoped accent it overrode, so
the one control that already had the right colour quietly lost it. Removed, and
asserted against: no later rule resets that label to inherited text.

## Mutation now reaches the stylesheet

The CSS mutant written for this checkpoint could not work: the runner mutated
`ai-assistant.js` and passed the stylesheet through untouched, so a `target:
"css"` key was accepted and silently ignored.

That gap mattered more than the mutant. Presentation contracts *are* contracts
here — the shared accent, the segmented-control chrome, the line-number gutter's
alignment with the code, the `forced-colors` fallbacks — and every stylesheet
assertion accumulated across this run was unguarded: deletable with no mutant
noticing.

`_mutant_target()` now selects the file, only the mutated one is written to the
temporary directory, and the other is passed through unchanged, so a CSS mutant
is still judged against the real script and vice versa. `target` is validated as
`js` or `css` alongside the existing catalogue checks.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **452/452**, one new mutant caught — `artifact-accent-not-shared`,
  the first CSS-targeted mutant, which drops one selector from the shared rule.
