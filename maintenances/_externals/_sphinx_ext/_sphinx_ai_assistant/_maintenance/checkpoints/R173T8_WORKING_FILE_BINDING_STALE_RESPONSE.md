# R173T8 — Working-file binding and stale-response protection

Status: COMPLETE

Base: R173T7 browser conversation-context v2 sender, same package.

## The gap this closes

T6 let a reader continue editing a file across turns; the file travelled as an
ordinary attachment. That works, and it is not enough. An attachment carries
bytes but no claim about *which revision* those bytes were, so:

    index.rst r3
        ├── request A starts from r3
        └── reader edits to r4 meanwhile
    request A returns, generated from r3
        → silently commits as r5

The file the reader keeps is then one neither party authored. This checkpoint
adds the binding that makes that detectable, and the guard that acts on it.

## Contract: `working_files`

`working_files` joins `history` on the v2 envelope. Each entry carries `path`,
`revision`, `sha256` and `content`, bounded at 4 files / 48 000 chars each /
96 000 total. Rejections are explicit and were verified individually: v1
carrying working files, traversal paths, absolute paths, drive letters, control
characters in a path, malformed digests, negative and boolean revisions, unknown
keys, duplicate paths, too many files, oversized content.

Paths are validated as strictly here as anywhere else that accepts one. This
server never opens, writes, or names anything from a working-file path — but it
echoes the path into the prompt, and a path that can carry traversal or control
characters into a prompt can carry them into whatever consumes the answer.

The server does not resolve `revision` or `sha256` against anything. It holds no
copy of the reader's file and must not pretend otherwise. They are carried so
the **client** can detect that its own file moved underneath the request, and
they live inside the validated envelope rather than an unvalidated side channel.

Working files are fenced behind their own per-request nonce, like history and
page text, and the fence instruction tells the model to return complete files
annotated with `file=` and never to claim a file was written.

## Browser: binding and protection

- `_sha256Hex` uses `crypto.subtle`. A file that cannot be digested is **not
  sent**: an unbound working file cannot be checked for staleness later, so
  sending it would produce the appearance of protection without the substance.
- Only files the reader explicitly pressed `Continue` on are eligible. Sending
  every tracked file would turn an unrelated question into a multi-file request
  and spend the reader's context budget unasked.
- `/health` bounds are clamped downward, never expanded. An endpoint
  advertising a digest other than `sha256` gets no working files rather than a
  digest it did not ask for under a field name that says otherwise.
- At registration, `_syncExplicitCodeArtifacts` consults the binding this turn
  declared. If the ledger has moved past it, the answer is surfaced as a
  **stale candidate** — readable, downloadable, with a `role="status"` notice
  naming both revisions — and is never committed as the next revision.

The bytes are not discarded. They are a real answer to a real question; what
they must not be is silently promoted over a newer revision.

## Verification

- browser wrapper gate: **142/142** (one new harness, 24 assertions);
- architecture gates: **401/401**, including two new mutants, both caught:
  `working-file-stale-response-committed` and `working-file-sent-without-digest`;
- server contract, parity and app gates: **80/80**;
- **end-to-end**: a `/health` document built from the real server constants was
  parsed by the browser negotiator, which selected v2 with working-file bounds;
  a bound request built from those bounds was accepted by the real server
  parser, fenced behind a nonce, echoed the path, emitted `['system','user']`
  roles only, and carried the `file=` instruction.

## A test corrected during development

The first staleness assertion was `!/(_markStaleAnswerCandidate[\s\S]{0,200}
_registerGeneratedArtifact)/`. It failed against correct code: the statement
immediately after the guarded block *is* the registration, which is exactly
right. Asserting "no registration nearby" was never the contract. Rewritten to
assert that control returns before registration, and that the guard condition
itself is present — the second half added only after a mutant proved the first
version passed with the branch made unreachable.

## Still open

- Retry still resends text without a retained context snapshot, so
  "resend this question as-is" can mean a different effective request.
- `revision` remains an event counter, bumped on `unavailable`/`removed`,
  rather than a pure content version.
- Conversation branching and virtualised long transcripts (T7 of the original
  design note) are untouched.
