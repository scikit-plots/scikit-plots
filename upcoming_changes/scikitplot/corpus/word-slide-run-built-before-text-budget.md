---
title: "A Word body or slide is joined whole before the text budget sees it (CX-39)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 938: hostile-sample gate (test__hostile_samples.py)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A Word body or slide is joined whole before the text budget sees it (CX-39)

## Summary

A `.docx` or `.pptx` of about 50 KB whose body is one long run of paragraphs is refused by `max_text_chars_per_byte`, but only after the whole run is built as one chunk.

## Current evidence

- 49,930-byte `.docx` (150,000 paragraphs, no heading): refused after 6.8 s; a 66 KB one (200,000 paragraphs) peaks at 212 MiB traced.
- 50,541-byte `.pptx` (one slide, 150,000 paragraphs): refused after 4.7 s; 171 MiB traced at 200,000.
- A bounded read of the same class (the `.ods` sample) peaks at 1.6 MiB in 0.02 s.
- Profile: all time is inside `OfficeReader._sections` / `_word_paragraph` before the first chunk reaches `_within_text_limit`.

## Root cause / current understanding

`_sections` appends every paragraph between two headings to one list and joins it in `flush()`; nothing ends a chunk early. CX-38 ended a spreadsheet line, chunk and table once past the limit in force; paragraph runs (Word body, notes, a slide's text) were not covered. Memory and time grow with the part size (up to `max_part_bytes`, 256 Mi), not with the limit.

## Expected behavior

A paragraph run stops being collected once strictly past the limit in force, as rows do since CX-38, so the refusal (or truncation) costs in proportion to the limit. Files within the limit chunk exactly as before.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_office.py`: `_sections`, the slide and note text paths; reuse `_join_capped`.

## Acceptance

The fix removes the sample's entry from `_OPEN` in
`scikitplot/corpus/_readers/tests/test__hostile_samples.py` (a strict
`xfail`: the gate fails if the entry stays after the fix).
