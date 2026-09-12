# R173T11 — In-place file preview and the presented-files flow

Status: COMPLETE

Base: R173T10 persistence honesty, same package.

## The problem the rendered page showed

When an answer returns a complete file, the panel printed the entire file into
the chat body. For a 600-line `index.rst` that buries the answer's reasoning
under text the reader has already seen, pushes the file controls off-screen,
and makes a three-file answer unreadable. The transcript becomes a mirror of
the files rather than a conversation about them.

## Collapse, don't relocate

A complete, path-bearing file now collapses to a single row naming it, sizing
it, and carrying its `+N/−M` — and it opens **in place**. Nothing moves to
another surface and nothing is fetched again: the disclosure body holds the
very same `<pre>` the markdown renderer produced, so copy, download and
artifact-path behaviour are untouched by construction rather than by
re-implementation.

Two deliberate exclusions:

- **Short files stay open.** Below 24 lines, collapsing costs a click and
  saves nothing.
- **Anonymous snippets are never collapsed.** A snippet with no `file=` path is
  usually the point of the answer, not a byproduct of it. Only files that
  claimed an identity are treated as attachments to the reasoning.

Collapsing is idempotent (`data-ai-file-disclosure`) and runs **once at
finalization**. The per-chunk sync path would wrap a fence that is still
growing and re-wrap it on every chunk.

## Presented, not changed

`Changed files` became `Presented N files`. Nothing outside the browser
changed, and the old heading claimed it had — the same overstatement class as
"resend this question as-is", "Latest revision r5" and the Remember switch,
all corrected earlier in this run. The draft status now rides with the count
(`drafts, not applied · links open the latest revision`) rather than sitting in
a tooltip nobody opens. The section itself collapses, taking its patch-series
control with it, so a long answer can end in one line.

## Verification

- browser wrapper gate: **144/144**;
- architecture gates: **415/415**, two new mutants, both caught:
  `file-preview-collapse-runs-per-chunk` and
  `presented-files-claims-files-changed`;
- 85 assertions in the owning harness, including the presentation contract
  asserted against the paired stylesheet: the caret rotates from `aria-expanded`
  rather than a parallel class, hidden bodies are actually `display: none`,
  and caret motion honours `prefers-reduced-motion`.

## A weak assertion caught by its own mutant

The first `file-preview-collapse-runs-per-chunk` mutant duplicated the
finalization call and was **not** caught — correctly, because the
`data-ai-file-disclosure` guard makes a second call a genuine no-op. The mutant
described a harm the code had already made impossible, and the paired assertion
(`not present in _registerGeneratedArtifact`) guarded a function that was never
at risk.

Both were retargeted at the real hazard: the call appearing inside
`_syncExplicitCodeArtifacts`, the per-chunk streaming path. The assertion now
names that function and the mutant inserts there. A mutant that cannot fail is
as useless as a test that cannot fail.

## Still open

Unchanged from R173T10: §18 branching and §20 virtualised transcripts remain
deferred, with rationale recorded. This checkpoint reduces the pressure behind
§20 — a transcript of collapsed rows is far cheaper to render than one of
inlined files — without claiming to have measured it.
