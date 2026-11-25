---
title: "Common text formats have no reader: local HTML, saved pages, CSV, JSON, notebooks, code, config, tar/gz (CX-50)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Real-data snapshot of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Common text formats have no reader: local HTML, saved pages, CSV, JSON, notebooks, code, config, tar/gz (CX-50)

## Summary

`DocumentReader.create` raises `ValueError: No DocumentReader registered
for extension ...` for text formats that research folders commonly hold.
The registry (84 entries on `main`) has `.xlsx` but not `.csv`, and HTML is
parsed only for URLs (`WebReader`); a local `.html` or a `file://` URL is
refused.

## Why it matters

A corpus built from a research folder, a cloned repository or a browser
export silently lacks its web pages, data tables, notebooks, code and
configuration. These are plain text and cheap to read.

## Current evidence

`main` at 13b457d. On the archive, 692 document files: every one of the 122
errors was "No DocumentReader registered", by type: `.py` 39, git sample
hooks 28, git internals 22, `.yml`/`.yaml` 8, no extension (`LICENSE`,
`.gitignore`, ...) 8, `.html` 2, `.mhtml` 2, `.xq` 2, `.toml`, `.lock`,
`.sh`, `.gz`, `.whl` 1 each; `.svg` in the media sample. Inside archives,
`.py`, `.csv`, `.json` and `.yaml` members were skipped (CX-49).

Probe: `DocumentReader.create("page.html")` and
`DocumentReader.create("file:///.../page.html")` both raise the error above.

## Progress

- HTML (`.html`, `.htm`, `.xhtml`): `HTMLReader`, PR 975. The shared
  parse path also reads nested elements once (a quadratic read that
  affected `WebReader` too).
- Compiled HTML Help (`.chm`), added on request: `CHMReader`, PR 976,
  pure-Python container and LZX decoder.
- Saved pages (`.mhtml`, `.mht`): `MHTMLReader`, PR 977.
- Tables (`.csv`, `.tsv`): `CSVReader`, PR 978.
- Structured data (`.json`, `.jsonl`, `.ndjson`, `.yaml`, `.yml`,
  `.toml`): `JSONReader`, `YAMLReader`, `TOMLReader`, PR 979.
- Code and plain configuration (about 90 extensions): `CodeReader`,
  PR 980. Extensionless files (`LICENSE`, `Makefile`) wait for the
  content sniff.

## Root cause / current understanding

Verified: no reader registers these extensions; `TextReader` registers
`.txt`, `.md`, `.rst` only; `WebReader` registers `:url` and fetches over
HTTP; `ZipReader` registers `.zip` only (a wheel is a zip, a `.tar.gz` is
not read).

## Expected behavior

Each family below reads into chunks with the right `source_type` and
structure, within the same size and expansion limits as today:

- HTML (`.html`, `.htm`, `.xhtml`): the `WebReader` parsing path, no
  network, scripts and styles dropped.
- Saved pages (`.mhtml`, `.mht`): the HTML part of the MIME message.
- Tables (`.csv`, `.tsv`): rows as records, header kept, dialect sniffed;
  same row and byte limits as the spreadsheet reader.
- Structured data (`.json`, `.jsonl`, `.ndjson`, `.yaml`, `.yml`, `.toml`):
  text values with their key path; JSON Lines one record per line.
- Notebooks (`.ipynb`): markdown and code cells, outputs optional.
- Code and plain configuration (`.py`, `.sh`, `.js`, `.ts`, `.c`, `.cpp`,
  `.h`, `.rs`, `.go`, `.java`, `.sql`, `.ini`, `.cfg`, ...): text with the
  language recorded.
- Extensionless or unknown files: an opt-in content sniff (valid UTF-8 or a
  detected text encoding, no NUL bytes) that reads them as text.
- Archives: `.tar`, `.tar.gz`/`.tgz`, `.gz` (single file), `.whl`/`.jar`
  (zip), under the existing expansion budgets.
- `.svg`: its `<text>` and `<title>`/`<desc>` content (it is XML).

## Affected paths and ownership

`scikitplot/corpus/_readers/` (new readers or registrations, `_text.py`,
`_web.py` parse function reuse, `_zip.py` or a shared archive layer),
`scikitplot/corpus/_base.py` registry. Owner: `maintenances/corpus`.

## Constraints and non-goals

- No new required dependency: parse with the standard library where it
  can (`csv`, `json`, `email`, `tarfile`, `gzip`, `tomllib` on 3.11+);
  optional extras otherwise, with a clear `ImportError` message.
- Git internals and VCS folders stay excluded (`CorpusBuilder` already
  skips hidden paths); the sniff must not read binary blobs.
- Path traversal, link and decompression limits apply to every new
  archive type, as for ZIP (PR 938, PR 941).
- One theme per pull request: split by family.

## Edge cases to cover

HTML with a `<meta charset>` that disagrees with the bytes; MHTML with
quoted-printable and base64 parts; CSV with a BOM, quoted newlines,
semicolons; JSON 1 GB on one line (limits); YAML with anchors and tags
(safe load only); a `.tar.gz` bomb and symlink members; a notebook with
large base64 outputs; an extensionless binary file; UTF-16 text.

## Proposed direction

Phase by family, most common first: HTML (reuse the `WebReader` parser),
CSV/TSV, JSON/JSONL, code and config as text with a language tag,
notebooks, tar/gz, MHTML, SVG, and last the content sniff (a decision:
opt-in flag or default).

## Verification / acceptance criteria

For each family: a fixture per edge case, the hostile-sample gate extended
with one sample per new archive type, and a reader snapshot of the archive
where the matching "No DocumentReader registered" errors become `ok`.

## Documentation impact

Corpus user guide (supported types table), `DocumentReader.create`
docstring, the usage skill's supported-types list.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one fragment per family PR.
