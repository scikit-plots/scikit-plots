---
title: "Readers report success when they skip or cannot read most of a source (CX-49)"
status: open
kind: "reliability"
area: "scikitplot/corpus"
discovered_during: "Real-data snapshot of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Readers report success when they skip or cannot read most of a source (CX-49)

## Summary

Three places lose content and still report a plain success:

1. `ZipReader` skips members with an unsupported extension at `DEBUG` level
   and then logs `finished X.zip — processed N members` at `INFO`, counting
   the skipped members as processed. An archive whose members are all
   unsupported yields no chunks and no warning.
2. `CorpusBuilder._expand_sources` skips files with an unsupported
   extension during directory expansion at `DEBUG` level only.
3. `PDFReader` warns "probably holds scanned images" only when no page has
   text. A PDF whose pages are images plus a page number gets no warning.

## Why it matters

A user indexing a folder learns that files were left out only by reading
debug logs. The answer engine then cannot say "I could not read X"; it just
lacks the content. Coverage cannot be measured or compared between runs.

## Current evidence

`main` at 13b457d, CPython 3.13:

- An archive holding `page.html`, `data.csv` and `script.py`:
  `list(DocumentReader.create("only_unsupported.zip").get_raw_chunks())`
  is `[]`; the only INFO line is `processed 3 members`.
- Real data: an archive of 6 saved web pages gave 0 chunks with no warning;
  an archive of 78 members skipped 47 (`.py` 30, `.csv` 8, `.json` 7,
  `.yaml` 2) silently, recall 0.45 against reading every text member.
- Real data: two image-only PDFs of about 20 MB (7 and 8 pages) gave 49 and
  48 characters (page numbers) and no warning; four PDFs with no text at
  all were warned about correctly.

## Root cause / current understanding

Verified: `_zip.py` `_iter_members` logs skips with `logger.debug` and its
summary counts members iterated, not members read; `_corpus_builder.py`
logs `Skipping unsupported extension` with `logger.debug`; `_pdf.py`'s
`_report` checks `pages_with_text == 0` only. There is no per-source record
of what was read, skipped or empty.

## Expected behavior

Every read can say what it covered: for each source, members or pages read,
skipped (with the reason: unsupported type, budget, no text) and empty. A
source that yields nothing, or skips more than a configurable share, logs a
`WARNING` that names the counts by reason, and the counts are available to
the caller (for example on the reader after iteration, and summed by
`CorpusBuilder`).

## Affected paths and ownership

`scikitplot/corpus/_readers/_zip.py`, `_pdf.py`, `_image.py` (frames left
unread already warn), `scikitplot/corpus/_corpus_builder.py`, possibly a
small shared report type in `_base.py`. Owner: `maintenances/corpus`.

## Constraints and non-goals

- Counts and types only in messages; never member text.
- No change to which members are read (that is CX-50).
- One warning per source, not one per member.

## Edge cases to cover

Empty archive, archive of only directories, nested archives (counts roll
up), all members refused by the expansion budget, PDF with 1 text page of
100, PDF with alternating text and image pages, a source read twice
(stable counts).

## Proposed direction

Add a coverage record filled during iteration (`read`, `skipped` by reason,
`empty`), expose it after iteration, and log one summary warning when
`read == 0` or the skipped or empty share passes a threshold (PDF: share of
pages with fewer than `ocr_min_chars` characters). `reader_snapshot.py`
records the counts so a snapshot comparison shows coverage changes.

## Verification / acceptance criteria

The ZIP reproducer logs one WARNING naming 3 unsupported members; the
image-only PDF case warns; tests fail on the current code; the snapshot of
the archive shows the new warnings and no other change.

## Documentation impact

Corpus user guide (what a skipped file looks like), `ZipReader` and
`PDFReader` Notes, `CorpusBuilder.build` docstring.

## Release-note promotion

Required: `scikitplot.corpus` / `enhancement`.
