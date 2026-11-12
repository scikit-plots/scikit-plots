# Corpus maintenance history

## 2026-11-12 — second research self-test, PR 928

A second, larger research set (95 text files, 8 PDFs, a slide deck, 322
images and 3 videos) was run through corpus and MCP. Text: every file
processed, known-item recall@5 1.00 (MRR 0.988) on the FTS5 leg, no
non-deterministic result.

- **D12 (a): the outline in the derived graph.** `StructureChunker` records
  `heading_path` on every chunk, but `derive_graph` built nothing from it.
  It now derives one `heading` node per source and heading path
  (`NodeKind.HEADING`), `contains` / `parent_of` between a heading and its
  sub-headings and between the innermost heading and the document, and
  `same_source` from each top heading to its source; evidence
  `metadata.heading_path`; nothing stored (ADR-R08-002). A document without
  a heading path adds nothing, and anything but a list or tuple of non-empty
  titles is not a path. Limit: two sections of one source with the same
  titles all the way down share a node. On the set's 84 markdown files:
  13,395 documents, 8,163 headings, 42,399 outline edges, derived in 0.23 s.
  Tests: 16 new in `TestOutline` (15 fail on `main`); 8 mutations caught,
  one only after the separator test was corrected; suites unchanged on
  CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- **CX-15 (a): `ImageReader` reads frames on demand, up to `max_frames`.**
  It copied every frame of the file into a list before the first OCR call
  and then ran OCR on all of them, without a bound (a 280-frame GIF did not
  finish in 60 s). A private `_LazyFrames` now counts the frames by seeking
  and copies one frame at a time; the new field `max_frames` (default 64,
  `None` for no limit; a positive int, `bool` refused) stops the read, and a
  warning names the file, its frame count, the frames read and the frames
  left unread. On the 280-frame GIF (800 x 1111) with OCR stubbed: peak RSS
  1006 MiB before, 62 MiB with the default, 64 MiB with `max_frames=None`.
  Tests: 15 new in `TestFramesOnDemand` (all fail on `main`); 9 mutations
  caught.
- **Outside corpus, in the same PR (its CI was red):** Sphinx Design 0.8
  gave `card` a `header` option and deprecated the `^^^` separator
  (`design.card_legacy`). The gallery grid's test that `header` is not a
  card option failed, and every card with a header warned. The grid now
  writes a `card-header` directive where Sphinx Design has it and the
  separator where it does not (`_header_block`); a MyST header gets a fence
  longer than any it holds. Tests pass with Sphinx Design 0.7.0 and 0.8.0
  on CPython 3.12; 5 mutations caught.
- **Recorded, not changed** (ledger note): CX-14 `PDFReader` reopens the
  file and walks the page tree for every page (a 277-page PDF: 75 s against
  7.9 s in one pass).
- Measured, not changed: MCP's `_tokens` (131 ms for 22,034 chunks; two exact
  rewrites reached 105-122 ms, not worth a change). `.pptx` has no reader
  (a clear error); pages or frames whose text the default filter omits are
  counted in `n_omitted` and logged, a video without subtitles warns.
- Evidence: `evidence/round-2026-11-12-outline.log`.

## 2026-11-10 — research self-test round, PR 926

- **CX-12, part 2 (decision D14 (a)): one document per run of text-only
  steps.** Every normalizer step built its own frozen document. The five
  built-in steps that only map the text now implement `normalize_text` on a
  private base, `_TextStep`; `NormalizationPipeline` passes the text through
  consecutive ones and builds one document for the run. A step is composed
  only when its `normalize_doc` is `_TextStep`'s own, so a custom step, a
  step that sets another field and a subclass overriding `normalize_doc` run
  as before. Identical documents: one SHA-256 digest over 104,220 results
  (400 ordered chains of one to three of eight steps, fixed texts, research
  paragraphs, a document with `normalized_text` already set) on CPython 3.9
  to 3.15.0b4, 3.13t and 3.14t included. `replace` calls in one build of the
  research corpus: 44,264 to 22,132; `CorpusBuilder.build` 3.01-3.12 s to
  2.70 s (best of three). Tests: 5 new in
  `TestPipelineBuildsOneDocumentPerRun` (4 fail on `main`; the equivalence
  one passes there by design); 6 mutations caught. `_TextStep` stays private
  while the usage skills cannot be regenerated here (a public name makes
  them stale).
- The round's ledger is empty: CX-05 to CX-12 are closed. D11 to D13 stay
  open decisions.
- Evidence: `evidence/round-2026-11-10-normalizer-chain.log`,
  `evidence/probe_normalizer_chain.py`.

## 2026-11-09 — research self-test round, PR 925

- **CX-10: the atomic-publication race test forked a threaded process,
  and could not fail.** It asked for the `fork` start method; on CPython
  3.15.0b4 under `pytest -n 4` the parent already ran threads, CPython warned
  on the fork and the suite's filters made that a failure (3 of 3 runs; 3.12
  to 3.14t passed). Checking the fix showed a second, older gap: with a pool,
  publishers started one by one and never overlapped, so the test passed
  against a deliberately non-atomic writer on `main` (0 of 5 caught). The test
  now uses `forkserver` (else `spawn`), and eight processes wait on a barrier
  and publish together, four races per test: the non-atomic writer is caught
  in 40 of 40 runs (CPython 3.9, 3.13, 3.14t, 3.15.0b4) while the real one
  passes. Cost: about 5 s instead of 0.3 s (four publishers took 2.4 s but
  missed 1 in 20). Tests only: no release note. Suites otherwise unchanged
  on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- With PR 925 every finding of the round is closed except CX-12 part 2
  (decision D14); D11 to D13 remain open.
- Evidence: `evidence/round-2026-11-09-race-test.log`.

## 2026-11-09 — research self-test round, PR 924

- **CX-07: image reader tests failed without Pillow.** 25 tests in
  `_readers/tests/test__image.py` patch `PIL.Image.open` (which imports
  `PIL`) or reach the reader's own Pillow import, which comes before its
  size check; the partial distribution `scikit-plots-corpus` does not depend
  on Pillow. They now carry `_needs_pillow`, a `skipif` with a stated reason:
  without Pillow 29 passed and 25 skipped (25 failed on `main`); with Pillow
  all 54 pass as before. `test_raises_import_error_when_pillow_absent`
  already covers the reader without Pillow. Tests only: no release note.
  Suites otherwise unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP
  on 3.8.
- Evidence: `evidence/round-2026-11-09-pillow-gate.log`.

## 2026-11-09 — research self-test round, PR 923

- **CX-06: a document holding a lone surrogate failed inside SQLite and
  JSONL storage.** `CorpusDocument` accepts such text, and the export layer
  already refuses it per format with the document and path named (JSON,
  JSON Lines, TOML) while YAML keeps it; storage raised a bare
  `UnicodeEncodeError` (on CPython 3.11+ SQLite could report the
  connection's previous error). Decision (c), taken over (a) because (a)
  would break the export contract and its tests: `_encodable_dict` refuses
  before any write, naming the document and the path (dictionary keys
  included), in `JSONLStorage.save` / `save_batch` and
  `SQLiteStorage._doc_to_row`; a batch holding one such document writes
  nothing. `InMemoryStorage` keeps it. Tests: 15 new in
  `TestLoneSurrogateDocuments` (12 fail on `main`; the in-memory and NUL
  ones pass there by design); 6 mutations caught; suites unchanged on
  CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- Evidence: `evidence/round-2026-11-09-surrogate-storage.log`.

## 2026-11-09 — research self-test round, PR 922

- **CX-12, part 1: `CorpusDocument.replace` recomputed its field names.**
  Every call built `{f.name for f in fields(self)}` to check the changed
  names. `_field_names(cls)` now computes them once per class (a subclass
  with more fields gets its own), so the same names are accepted and the
  same error lists them. `replace` went from 15.6-23.6 us to 12.3-18.4 us
  per call on CPython 3.9, 3.13, 3.14t and 3.15.0b4; `CorpusBuilder.build`
  over the 64 research files from 3.15-3.17 s to 2.93-3.00 s (best of
  three). Tests: 4 new in `TestReplaceFieldNames` (3 fail on `main`; the
  error-text one passes there by design); 4 mutations caught; suites
  unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8. Part 2
  (one document per normalizer step) stays a note, waiting on D14.
- Evidence: `evidence/round-2026-11-09-replace-names.log`.

## 2026-11-09 — research self-test round, PR 921

- **CX-11: the chunkers' input guard scanned for surrogates in Python.**
  `_validate_text_input` (`_chunkers/_sentence.py`) walked every character
  through `enumerate` and `ord`; it now runs one compiled search
  (`_LONE_SURROGATE`), which reports the same first surrogate and position,
  so the same texts are refused with the same message. Over the 64 research
  files (3.35 M characters): 120-197 ms to 14-16 ms on CPython 3.9, 3.13,
  3.14t and 3.15.0b4; `CorpusBuilder.build` unchanged within noise (about
  0.1 s of 3.2 s). Tests: 2 new in `TestValidateTextInput` (every part and
  ordered pair of parts against the old definition, which passes on `main`
  by design; clean text not walked, which fails there); 5 mutations caught;
  suites unchanged on CPython 3.9 to 3.15.0b4, 3.13t, 3.14t, MCP on 3.8.
- **CX-09 (decision (a), same PR): the README `CorpusBuilder` example
  raised.** Its `search` used the hybrid default on a build without
  embeddings. The example now passes `match_mode="keyword"` and says how to
  enable hybrid search; `tests/test__corpus_builder.py` runs the README's
  section 2 code on a one-file folder (fails on `main` with the
  `RuntimeError`).
- Evidence: `evidence/round-2026-11-09-input-guard.log`.

## 2026-11-09 — research self-test round, PR 920

- **CX-08: `detect_script` tested Latin last.** Each character of a chunk's
  sample went through 23 other scripts' branches (48 range comparisons)
  before the Latin one, so English text paid for every other script first;
  the sentence, word, paragraph, fixed-window and semantic chunkers run it on
  every chunk. An ASCII sample is now counted without the range tests (in
  ASCII only `A-Z` and `a-z` count, and only as Latin; `str.isalpha` is true
  for exactly those), and both paths decide through one helper,
  `_dominant_script`. Identical results: one SHA-256 digest over 239,275
  cases (edges at every sample size and threshold, 20,000 generated texts,
  the research paragraphs) on CPython 3.9 to 3.15.0b4, free-threaded 3.13t
  and 3.14t included. ASCII samples went from 103-232 us to 4.5-7.8 us per
  call; other samples are unchanged within noise (copies compared at one
  location); `CorpusBuilder.build` over the 64 research files went from
  6.2-6.4 s to 3.2 s (best of three). Tests: 13 new in `TestDetectScript`;
  results cannot differ, so 12 pass on `main` by design, and the one that
  pins the ASCII path skipping the range tests fails there; 8 mutations
  caught.
- **Recorded, not changed** (ledger notes under
  `upcoming_changes/scikitplot/corpus/`): CX-11 the chunkers' input guard
  looks for lone surrogates one character at a time (0.84 s of a profiled
  8.4 s build); CX-12 `CorpusDocument.replace` recomputes its field names
  (0.62 s) and each normalizer step rebuilds the document (1.6 s), the
  second part waiting on decision D14.
- `todo/todo.md` is at 399 of its 400 lines: Runs 4 and 5b are complete and
  can be archived, each with its manifest entry.
- Evidence: `evidence/round-2026-11-09-detect-script.log`,
  `evidence/probe_detect_script.py`.

## 2026-11-09 — research self-test round (PR 919)

64 text files of collected research (3.35 M characters, 22,034 paragraph
chunks) were run through corpus and MCP: every file through
`CorpusPipeline`, `CorpusBuilder`, `SQLiteStorage` and both MCP lexical
retrievers, with timing, peak memory, repeat-run determinism and edge-case
queries. Every file processed; known-item recall@5 1.00 on the FTS5 leg;
no non-deterministic result.

- **CX-05: NUL and lone surrogates broke FTS5 free text.** CX-01 quoted
  every query as FTS5 strings, but no string can carry a NUL (SQLite:
  `OperationalError: unterminated string`) or a lone surrogate, U+D800 to
  U+DFFF (not encodable: `UnicodeEncodeError`, which CPython 3.11 and newer
  report as the connection's previous error when an earlier statement
  failed). `search_text` and `StorageQuery.full_text` raised for 7 of the 13
  probe queries, and the MCP lexical leg reported `failed`, so hybrid search
  lost its keyword leg; a JSON request can carry either character. Both now
  separate words like a space (`_FTS5_UNREPRESENTABLE`, applied in
  `_fts5_phrase` and before `_fts5_any` splits); FTS5 already splits stored
  text at a NUL. Every one of the 1,114,112 code points was checked on the
  three SQLite builds below: only these are refused. Tests: 81 in
  `TestFreeTextSearch` (66 fail on `main`; the other 15 pin the class and
  the characters kept, so a wider or shifted class fails); 10 mutations each
  caught. Verified on CPython 3.8 to 3.15.0b4, free-threaded 3.13t and
  3.14t, SQLite 3.46.0, 3.50.4 and 3.53.1 (corpus suites from its 3.9 floor;
  MCP from 3.8): identical failure sets before and after.
- **Recorded, not changed** (each a ledger note under
  `upcoming_changes/scikitplot/corpus/`): CX-06 a document holding a lone
  surrogate is accepted, then refused by SQLite and JSONL storage; CX-07
  image reader tests fail instead of skipping without Pillow; CX-08
  `detect_script` tests Latin last, 30% of `CorpusBuilder.build` on English
  text; CX-09 the README `CorpusBuilder` example raises (hybrid search,
  no embeddings); CX-10 on CPython 3.15 the atomic-publication test forks a
  threaded process, and the fork warning fails it.
- Known, not new: on Python 3.8 `scikitplot.corpus` does not import
  (`dict[str, Any]` at run time in `_types.py`); the partial distribution
  declares 3.9 and the root `>=3.8` was recorded in
  `maintenances/_repo/_maintenance/history/partial_distributions.md`.
- Evidence: `evidence/round-2026-11-09-fts5-text.log`,
  `evidence/probe_fts5_text.py`, `evidence/probe_fts5_sqlite.py`.

## 2026-09-28 — lexical retrieval review (cross-subsystem round)

Reviewed with MCP after a reading of current hybrid-retrieval practice (BM25 +
dense fused by rank, rerank, context budget, caching).

- **CX-01 — free text reached FTS5 as syntax.** `SQLiteStorage.query` passed
  `StorageQuery.full_text` to `MATCH` verbatim, so FTS5 read it as an
  expression: `roc_auc_score()`, `sklearn.metrics:auc`, `C++`, `random-state`,
  `area under the curve?` and any pasted error message were syntax errors (9 of
  13 realistic queries), and a plain question required every word. `full_text`
  is now one quoted phrase — its documented meaning, and what the in-memory and
  JSONL backends emulate — and never an error. New
  `SQLiteStorage.search_text(text, limit, collection_id=None)` is ranked BM25
  retrieval over free text: any word, each compound word also as its parts,
  score `-bm25()` (higher is better). Tests: `TestFreeTextSearch` (34 fail
  with the query passed verbatim).
- **CX-03 — `_BM25Index` scanned every document for every query term.** It now
  walks a postings list per term. Scores and order are bit-identical to the
  scan (1 500 randomised comparisons; `TestBM25PostingsEqualTheDefinition`
  checks against the definition). 50 000 documents: query 62 ms -> 5 ms; build
  1.17 s -> 1.85 s, paid once.
- **Recorded, not changed:** four BM25 implementations exist (`_BM25Index`, the
  MCP demo backend, `scikitplot.rank_bm25`, SQLite FTS5 `bm25()`), with two
  tokenisations (`\w+` keeps `_`; FTS5 `unicode61` splits on it). Consolidating
  them is a design decision with compatibility cost, not a fix.
- Evidence: `evidence/round-2026-09-28-lexical.log`, `evidence/probe_lexical.py`.
- Consumer note (CX-04, owned by MCP): the MCP `--corpus-mode hybrid|keyword`
  profile now calls `RetrievalIndex.search` directly instead of re-fusing legs
  itself. That makes `search`'s keyword-only `config` and the `LegOutcome`
  fields (`leg`, `status`, `hit_count`, `error.message`) a cross-package
  contract: MCP's `test_corpus_index_retriever.py` fails if they change. No
  corpus code changed.
- Gate state on arrival: `check_trackers.py` stops at `missing required path:
  plugins/scikitplot-dev/skills/corpus` (absent from the supplied archive). Unchanged by this round.
  Suite: 3 490 passed, 10 skipped; the 61 failures are the same set as before
  the round (NLTK data packages absent from the review environment).

The earlier Corpus campaign records describe R00–R16 review and IMPL-01–18 implementation work, including a historical `3206 passed / 27 skipped / 4 xfailed` suite. Those records are retained under `history/` and `checkpoints/` for rationale only.

On 2026-09-12 the supplied archive was found to contain no Corpus runtime files. The maintenance model was therefore rebased to fail closed: historical completion is not current evidence, and the next action is source restoration/retrieval before semantic continuation.
