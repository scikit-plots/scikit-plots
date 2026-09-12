# R173T13 — Save-as and the three-action card

Status: COMPLETE

Base: R173T12 presented-file cards, same package.

## Card actions

The primary row is now `preview | download | save as…`, with the preview
claiming whatever width the two actions leave. The card body *is* the preview
target and opens the same full attachment overlay an uploaded file does, so the
whole block is the affordance rather than a small word at its end.

`Patch` and `Continue` stay on the secondary line. A single file and a
multi-file answer render identically; the only difference is the full-width
`Download all N files` in the footer, which appears once there is more than one.

On a narrow panel the three actions wrap beneath the filename rather than
clipping it — the filename is the part that identifies the card.

## Save-as is a download alias, never a rename

This is the decision worth recording. The ledger is keyed by path, and the
revision chain, diff base, patch `diff --git` headers and any binding an
in-flight request holds all hang off that key. Letting a save dialog rewrite it
would orphan every one of them.

So the tracked file keeps its identity and the reader gets the current bytes
under whatever filename suits their filesystem. `Download` remains the
zero-friction path for readers who want the file under its own name.

A reader-typed filename is still a path the browser will act on, so it passes
the same slug rules as a derived one — with the extension preserved, because
that is usually the only part they cared about typing. A typed directory is
stripped rather than honoured: this writes to their download folder, and
pretending otherwise would be a claim the panel cannot make.

## Verification

- browser wrapper gate: **144/144**;
- architecture gates: **419/419**, one new mutant caught:
  `save-as-accepts-a-typed-directory`;
- the sanitizer is asserted against hostile input in the naming harness:
  traversal paths, Windows paths, bidi controls, reserved device stems,
  punctuation-only stems, multi-dot names, and empty input;
- the card contract is asserted against the paired stylesheet: three columns
  with the preview taking the remainder, and the narrow-panel wrap.

## Stale expectation corrected

`preview and download own the card primary row` pinned a two-append literal.
The row now holds three controls by design; the assertion names all three and
the secondary-line placement of Patch and Continue is still asserted separately.
