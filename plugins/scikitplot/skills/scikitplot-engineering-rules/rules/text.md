# Text, Unicode and markup

Categories: Unicode, Spans, Markup parsers

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 58: Unicode - a normalisation rule is tried on real text in many scripts before it ships
- **When:** a validator refuses input that is not in some normal form (NFC, NFKC) or restricts characters by category.
- **Then:** run it on ordinary names or words from at least a dozen scripts (Latin with diacritics, Turkish, Vietnamese, Greek, Cyrillic, Arabic with harakat, Hebrew with niqqud, Devanagari, Thai, Hangul, Japanese, Chinese) and on canonically equivalent spellings; prefer naming the forms to refuse over requiring a normal form, and normalise what is equivalent instead of refusing it.
- **Verified by:** `test__surrogate_sets.TestEntryFloor.test_names_pass` lists names from those scripts.
- **Added:** 2026-10-10. Root cause: requiring NFKC to stop full-width look-alikes also refused Thai `\u0e19\u0e49\u0e33\u0e1d\u0e19` (SARA AM has a compatibility decomposition) and Arabic typed shadda-then-fatha; found by re-running the review brief by hand.

### Rule 117: Spans - an unknown or foreign region is reported as unknown, never as the whole parent
- **When:** a component maps a sub-unit (chunk, segment, page) back to its parent's text by offsets, and the offsets may be absent, disabled, or computed over a different text (preprocessed, markup stripped).
- **Then:** record which text the offsets index (`span_basis`), slice only a parent the offsets index, and return `None` when the region is unknown; the whole parent is right only when the sub-unit is the only one; test each case (offsets off, preprocessed input, a reader text that is not the chunked text).
- **Verified by:** `TestRawText`, `TestPreprocessedSpans` and `TestDocuments.test_without_offsets_the_span_is_unknown` in `corpus/tests/test__chunk_spans.py`.
- **Added:** 2026-10-16. Root cause: `_slice_raw_text` fell back to the full parent when offsets were out of range, so with offsets disabled every chunk of a split parent got the whole source as `raw_text`; and multilang preprocessing (NFC, BOM stripping) shifted the spans while the "raw" slices were still taken from the unprocessed input with them.

### Rule 122: Markup parsers - a parser that claims a reference syntax is tested against the reference implementation
- **When:** code parses a markup whose meaning another implementation defines (reStructuredText and docutils, CommonMark and its reference parser, YAML front matter and Pandoc).
- **Then:** before review, run a differential against the reference on a table of cases (the constructs claimed, their neighbours, and the real documents of the repository); keep the table as a test that `importorskip`s the reference; document each remaining difference by name (error recovery, extensions the reference lacks, rendering).
- **Verified by:** `test_rst_literals_and_titles_match_docutils` in `corpus/_chunkers/tests/test__structure.py` (139 cases, docutils 0.22.4); the whole-docs comparison `r34/run4b/oracle_docs.py` (320 pages; the 13 that differ are Sphinx-only directives).
- **Added:** 2026-10-16. Root cause: `StructureChunker` was written from memory of the reST rules. Three review rounds each found rule gaps (a directive body as a literal block, the literal indent of a definition list, option lines, list markers before explicit markup, tabs); docutils answered each in seconds once it was asked.
