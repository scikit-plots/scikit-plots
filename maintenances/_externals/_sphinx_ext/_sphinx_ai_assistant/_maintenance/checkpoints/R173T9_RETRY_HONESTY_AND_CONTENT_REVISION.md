# R173T9 — Retry honesty and the content/event revision split

Status: COMPLETE

Base: R173T8 working-file binding, same package.

## Two claims the panel could no longer support

**"Retry — resend this question as-is."** True when a request was the question
and nothing else. From T7 and T8 onward, history and working files travel with
it, so identical text produces a materially different request depending on how
many turns have accumulated and which revision each file sits at. The label was
promising a replay this panel does not retain.

**"Latest revision r5."** `_generatedArtifactPublishState` advanced `revision`
for `unavailable` and `removed` — neither of which changes a byte. A reader
seeing r5 was being told about four content changes that did not happen, and
staleness binding compared against that number, so an unrelated preview
eviction firing mid-request would reject a perfectly current answer.

## Retry: state the smaller true capability

Two honest options existed for retry. Retain a byte-for-byte snapshot of every
turn's assembled context so exact replay is real, or stop claiming it.

Snapshots would multiply session storage by the transcript length to support
one button, and would keep page text and file contents alive long after the
reader finished with them — a retention decision taken on their behalf for a
convenience they never asked for. The panel already has a persistence budget it
must not silently blow (see the sessionStorage failure path).

So a **context receipt** is recorded per request: counts and identifiers, never
bytes. `question`, contract, history turn count, turns dropped, transcript
length, resource count, and each working file's key/path/content-revision. The
log is bounded at 32 and evicts oldest-first.

Both retry controls now say what they do — "Ask again with the current context"
— and on click, `_turnContextDrift` names anything that has moved since that
question was last asked: later turns that now exist, a file that changed
revision (with both numbers), a file no longer tracked. Matched on question
text, so the sentence always describes a request that really happened; no
receipt means no notice rather than a guess.

A button that tells the truth about a smaller capability is worth more than one
that overstates a larger one.

## Content revision

`_artifactContentRevision(entry)` is now the number readers see and the number
staleness compares. `contentRevision` advances only on a genuine content
change; `revision` remains the ledger event counter that drives refresh and
eviction bookkeeping. Records written before the split carry no
`contentRevision` and fall back to `revision`, so history stays readable rather
than being renumbered.

Applied to: preview status lines, patch filenames, patch base headers,
working-file wire revisions, staleness comparison, and every notification that
quotes a revision to the reader.

## Verification

- browser wrapper gate: **143/143** (one new harness, 18 assertions);
- architecture gates: **406/406**, including two new mutants, both caught:
  `content-revision-advanced-by-preview-eviction` and `retry-claims-exact-replay`;
- the receipt harness asserts the stored records contain no file or page bytes,
  that the log is bounded, and that eviction drops oldest first.

## Tests corrected, not deleted

Three sandboxed harnesses began failing because they build production functions
with `new Function` and now needed `_artifactContentRevision`. Supplied the real
helper rather than a stub — a stub would have tested a function the product
does not ship.

One assertion pinned `baseRevision: old ? old.revision : 0`. The contract it
guards is unchanged; only the expression moved. Updated, and four assertions
added so the split itself is now pinned.

The first draft of the retry assertion searched the whole source for "as-is"
and failed against correct code: the explanatory comments legitimately quote the
old label while explaining why it was wrong. Rescoped to user-facing strings.

## Still open

- Conversation branching and version navigation (original design note §18).
- Virtualised long transcripts (§20).
- Server-side git remains deferred; the push-only courier design is in R173T5.
