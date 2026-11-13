---
title: "No reader for office documents (Word, Excel, PowerPoint, OpenDocument)"
status: in-progress
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 935: reader snapshot of the research files (math-gnostics.pptx)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot.corpus/935.feature.rst"
---

# No reader for office documents (Word, Excel, PowerPoint, OpenDocument)

## Summary

`DocumentReader.create` raises `ValueError: No DocumentReader registered for
extension '.pptx'` for the research file `pptx/math-gnostics.pptx`, and the
same for `.docx`, `.xlsx`, `.odt` and every other office format. A corpus
built from a folder of office files gets nothing from them.

## Why it matters

Office files are the most common documents after PDF in the folders people
index. Today they must be converted by hand before ingestion.

## Current evidence

- `reader_snapshot.py` on the research folder (images excluded): 139 files,
  1 error, the `.pptx`.
- The `.pptx` is a ZIP of 85 parts; 18 slides in `ppt/slides/slideN.xml`.
  Name order is not slide order (`slide1`, `slide11`, `slide12`, ...): the
  order is `ppt/presentation.xml` `p:sldIdLst` through
  `ppt/_rels/presentation.xml.rels`.

## Root cause / current understanding

No reader is registered for these extensions. The formats fall in four
groups, with different costs:

| Group | Extensions | Container | Needs |
|---|---|---|---|
| OOXML | `.docx .docm .dotx .dotm`, `.xlsx .xlsm .xltx .xltm`, `.pptx .pptm .potx .potm .ppsx .ppsm .sldx .sldm` | ZIP + XML | standard library only |
| ODF | `.odt .ods .odp .odg .odf`, templates `.ott .ots .otp .otg` (ZIP), `.fodt .fods .fodp .fodg` (one XML file) | ZIP + XML or XML | standard library only |
| Legacy binary | `.doc .dot .wbk`, `.xls .xlt`, `.ppt .pot .pps`, `.xlsb` | OLE2 compound file (BIFF12 for `.xlsb`) | an optional backend |
| Code, not documents | `.wll .wwl`, `.xla .xlam .xll .xll_ .xla_ .xla5 .xla8 .xlm .xlw`, `.ppa .ppam` | add-ins, DLLs, macro sheets, workspaces | refused |

## Expected behavior

`DocumentReader.create(path)` reads the text of every document format above,
in reading order, with the same provenance, limits and failure reporting as
the other readers; add-ins are refused with a message that says why.

## Affected paths and ownership

- New: `scikitplot/corpus/_readers/_office.py` (OOXML, ODF), registered in
  `_readers/__init__.py`; tests in `_readers/tests/test__office.py`.
- Reused: `_xml_safety.py` (hardened parsing of every XML part),
  `_archive_handler.stream_copy_bounded` (member size limits), the
  `ZipReader` limits (`max_files`, `max_total_bytes`) as defaults.
- Owner: `maintenances/corpus/`.

## Constraints and non-goals

- No new required dependency: OOXML and ODF are read with `zipfile` and the
  hardened XML parser. Legacy binary formats use an optional backend, chosen
  like `prefer_backend` / `<backend>_kwargs` (MAINTAINING.md rule).
- Never run or load macros, OLE objects, external links or embedded
  programs. Macro-enabled files are read for their text only, and the chunk
  says the file holds macros (`has_macros`).
- Untrusted input: every XML part through `_xml_safety`; parts are read
  with a byte budget; part names are never written to disk (no extraction,
  so no ZipSlip); `../` targets in relationships are refused.
- Not a layout engine: no rendering, fonts or page breaks.
- Embedded images are not OCR'd in the first version (a later `ocr=` option
  can reuse the image reader).

## Edge cases to cover

- Slide, sheet and section order from the manifest, not part names.
- Hidden slides and sheets (`show="0"`, `state="hidden"`): a policy, default
  read and mark them.
- Speaker notes, comments, footnotes, endnotes, headers and footers: each a
  `section_type`, each switchable.
- Tables: one chunk per table, cells tab-separated, rows by line; merged
  cells; empty rows trimmed.
- Spreadsheets: shared strings, inline strings, numbers as stored text,
  formulas as their cached value (never evaluated), dates (number formats),
  very large sheets (row limit, streamed parse), empty sheets.
- Text runs split across many `w:r` / `a:r` / `text:span` elements; tabs,
  line breaks, soft hyphens; right-to-left text.
- Fields (page numbers, dates): cached result text.
- Tracked changes: insertions kept, deletions dropped, as a policy.
- Encrypted OOXML (an OLE2 file named `.docx`): a clear error.
- A truncated or not-a-ZIP file, a ZIP bomb, a part missing from the
  manifest, an empty document.
- Flat ODF (`.fodt`): one XML file, same mapping as the ZIP form.
- A legacy file without its backend installed: an install hint.

## Proposed direction

Phased, one pull request each:

1. **OOXML Word, PowerPoint** (`.docx` family, `.pptx` family): done in
   PR 935 (`_readers/_office.py`, `OfficeReader`). Body paragraphs and
   tables in order, footnotes and endnotes, slides in deck order, notes,
   hidden slides. Left for later: heading levels from styles, headers and
   footers, comments, `.sldx`/`.sldm` (a single-slide package).
2. **OOXML Excel** (`.xlsx` family): done in PR 935. Sheets in workbook
   order, `rows_per_chunk` rows a chunk (`row_start`, `row_end`), shared
   and inline strings (phonetic runs left out), cached formula results,
   booleans, errors; chartsheets skipped; hidden sheets under `hidden=`.
   Left for later: number formats (dates read as serial numbers), merged
   cells, a streamed parse (memory bounded by `max_part_bytes`), `.xlsb`.
3. **ODF**: done in PR 935. `.odt .ott .ods .ots .odp .otp .odg .otg
   .odf` and flat `.fodt .fods .fodp .fodg` from `content.xml` (or the one
   XML file): paragraphs, headings, lists, tables, notes; sheets with
   repeated cells and rows expanded within the sheet size; a chunk per
   page with its notes; formula annotation. Encrypted packages refused.
   Also for OOXML: Strict-conformance namespaces read like Transitional; a
   package holding a part twice is refused; the content decides, not the
   extension. Left for later: hidden pages and sheets (the flag is in a
   style), headers and footers from `styles.xml`, `.odm` master documents.
4. **Legacy binary**: optional backends (for example an OLE2 parser, or a
   conversion through a LibreOffice process), off unless installed.
5. **Add-ins**: registered so they are refused with a reason instead of
   "no reader".

Common to all: one class per family, a shared part loader (manifest order,
byte budget, hardened XML), chunk keys `text`, `section_type`,
`page_number` (slide or sheet index), `part`, and the `custom_extractor`
hook every reader has.

## Later, beyond the five phases

- Format detection for every reader from the file's first bytes and its
  package, so a wrong extension is caught before dispatch.
- Provenance on every chunk: a hash of the source file and the reader
  version, so an index entry can be traced to exact bytes.
- A decompression-ratio limit next to the size limits.
- Formula text next to the cached value for spreadsheets, as an option.
- Heading levels for Word and OpenDocument text, so chunks keep their
  section.

## Verification / acceptance criteria

- The research `.pptx` reads its 18 slides in deck order: 13 chunks, the 5
  other slides hold only a picture. Done (PR 935).
- Generated fixtures for each extension (built in the tests with
  `zipfile`, no binary files committed) cover the edge cases above.
- Hostile inputs: XXE in a part, an entity bomb, a member over the byte
  budget, `../` relationship targets: refused, no file written.
- `reader_snapshot.py` before and after: only the new extensions change.
- Read twice: stable.

## Documentation impact

Reader table in the corpus docs and the corpus usage skill (generated).

## Release-note promotion

Required for each phase: `scikitplot.corpus`, type `feature`.
