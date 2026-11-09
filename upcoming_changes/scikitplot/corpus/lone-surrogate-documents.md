---
title: "A document holding a lone surrogate is accepted, then refused by SQLite and JSONL storage"
status: open
kind: "reliability"
area: "scikitplot/corpus"
discovered_during: "PR 919: research self-test of corpus and mcp (finding CX-06)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A document holding a lone surrogate is accepted, then refused by SQLite and JSONL storage

## Summary

`CorpusDocument.create` accepts text holding a lone surrogate (U+D800 to
U+DFFF). `InMemoryStorage` stores it, but `SQLiteStorage.save` /
`save_batch` and `JSONLStorage.save` raise `UnicodeEncodeError: 'utf-8'
codec can't encode character '\udfff' in position 156: surrogates not
allowed`. The position is inside the serialised JSON, not the text, and the
document is not named.

## Why it matters

- The same document is valid or invalid depending on the backend.
- One such document makes `SQLiteStorage.save_batch` roll back the whole
  batch, so every good document in it is refused too.
- The error surfaces at save time, far from where the text entered: a JSON
  request (`json.loads('"\\udfff"')`) or bytes decoded with
  `errors="surrogateescape"` can produce such text.

## Current evidence

Probe against `main` at `6734def` (CPython 3.13, sqlite 3.53.1):

- `CorpusDocument.create("p.md", 1, "roc\udfffauc here")`: accepted.
- `InMemoryStorage().save(doc)`: stored, text round-trips.
- `JSONLStorage.save(doc)`: `UnicodeEncodeError`; the file keeps only the
  complete lines written before (no partial record).
- `SQLiteStorage.save_batch([good1, bad, good2])`: `UnicodeEncodeError`;
  `count()` is 0 afterwards (the batch is atomic).
- A NUL in document text is stored by all three backends and found by
  `search_text` (FTS5 splits words at it): NUL is not part of this finding.
- On CPython 3.11 to 3.15.0b4 (not 3.8 to 3.10), when an earlier statement
  on the same connection failed, the encoding failure is reported as that
  statement's error (observed: `OperationalError: unterminated string`
  after a failed NUL query), so the message can name the wrong cause.

## Root cause / current understanding

The schema hashes text with `encode("utf-8", errors="surrogatepass")`
(`_schema.py`, `_types.py`, `_generation.py`), so a surrogate never fails
there; the SQLite and JSONL backends encode strictly. No single place
decides whether a document's text must be valid Unicode.

## Expected behavior

One rule for every backend, decided where a document is made: either a lone
surrogate is refused by `CorpusDocument` with a `ValueError` naming the
field and the character position, or every backend stores and returns it
unchanged. A refusal must not depend on which backend is used.

## Affected paths and ownership

- `scikitplot/corpus/_schema.py` (`CorpusDocument`)
- `scikitplot/corpus/_storage/_storage.py` (all three backends)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

- Pre-1.0: no compatibility path for documents already stored.
- Do not change the content hash of valid text.
- Query text is not part of this note: CX-05 (PR 919) made a lone
  surrogate in search text separate words.

## Edge cases to cover

High and low surrogates at both ends of each range; a surrogate pair written
as two code points; a surrogate in `raw_text`, `metadata` values and
`input_path`; `save` and `save_batch` on every backend; a batch with the bad
document first, in the middle and last.

## Proposed direction

Validate in `CorpusDocument` (the one entry point for every reader and
chunker) and refuse a lone surrogate there; then the backends need no rule
of their own. Readers that decode bytes keep their current error policy.

## Verification / acceptance criteria

- The same document gives the same outcome on all three backends.
- The refusal names the field and the position.
- A batch with one refused document stores nothing and says which document.
- Tests fail on `main` and pass after the change; the corpus suite is
  otherwise unchanged.

## Documentation impact

`CorpusDocument` docstring (text rules); the storage section of the corpus
user guide if it states what text is accepted.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
