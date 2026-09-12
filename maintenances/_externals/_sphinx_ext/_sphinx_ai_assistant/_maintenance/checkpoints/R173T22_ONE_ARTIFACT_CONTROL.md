# R173T22 — One artifact control, both surfaces

Status: COMPLETE

Base: R173T21 artifact segmented control, same package.

## What changed

`section.ai-assistant-panel-changed-files` ("Presented files") now renders each
file with the same segmented control the snippet cards in the answer body use:

```
| icon + name + type (preview) │ Download | ⋮ |
```

Both surfaces call one `_buildArtifactSegmentGroup(ariaLabel, primary,
secondary)`. Only the outer grid differs — the group takes the room the
overflow menu leaves, and the preview segment takes the room inside it.

## Why one builder rather than two matching stylesheets

They are the same control on the same kind of object, and a reader meets them
minutes apart. Two implementations drift in exactly the details that are hard
to see and easy to get wrong — the `role="group"`, the accessible name, the
`aria-hidden` on the divider, the `:focus-within` outline. Copying the CSS and
duplicating the DOM construction would have produced two controls that looked
identical the day they shipped.

The harness asserts there is exactly **one** `ai-md-artifact-group`
construction in the file, so a third surface has to reuse it rather than copy
it.

## Verification

- browser wrapper gate: **147/147**;
- architecture gates: **448/448**;
- two mutants retargeted from the snippet call site to the shared builder and
  renamed accordingly — `artifact-download-nested-inside-the-preview` and
  `artifact-group-loses-its-accessible-name`. Both now guard **every** artifact
  surface rather than one of them.

## An assertion that could not see the new failure

`the download button is never nested inside the card button` was written as
`!src.includes('card.appendChild(dlBtn)')` — the variable names of one call
site. Once the construction moved into a shared builder, a mutation nesting the
segments for *every* surface at once did not touch that string, and the
assertion passed while the mutant survived.

It now states the contract on the builder: the builder never appends one
segment into the other. The call-site form is kept as a second, narrower check
rather than the only one.

This is the third time this run that an assertion written against a call site
survived the code moving. The general form: **assert the contract where the
contract lives, not where it happened to be implemented on the day.**
