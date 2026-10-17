<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 33 / Run 3 - ingestion contract (chunker registry, config strictness, spans, cache, logs, resources)

### Context
- Goal: the ingestion side gets the contract Run 2 gave retrieval: one name
  table, configs that refuse what they cannot honour, spans that mean what
  they say, a cache that cannot serve stale model output, quiet logs,
  closed resources.
- Plan items R3-1..R3-8 (`01_verdict_and_run_plan.md` section 3). R3-6 waits
  for decision D4 ("later"). Base: 2g (`6e87d56`). PR 865 fragments.
- Gate (plan): registry round-trip tests for every chunker; the span
  invariant over all chunkers and scripts; no unknown option is ever
  silently ignored.

### Increments
- [x] 3a. Names and configs (R3-1, R3-2, R3-4)
  - Evidence (re-run on 2g, `r34/run3/`): `chunker_kwargs` raises
    `TypeError` for all four named chunkers; `"semantic"` and a chunker
    registered in `ComponentRegistry` are refused by name; `"word"` and
    `"semantic"` are not registered; `normalizer_steps=["unicode",
    "lowercsae"]` drops `lowercase` with a warning. Found while reading:
    `ComponentRegistry.load_from_snapshot(registry.snapshot())` fails for the
    built-in chunkers (nominal `ChunkerBase` check; they conform to
    `ChunkerProtocol`); a subclass of a built-in chunker has no bridge
    (lookup by exact class name) and a protocol-conforming third-party
    chunker has none either; an invalid chunker config fails once per source
    inside the per-source error handler, so `build()` returns no documents
    instead of raising.
  - [x] registry: `register_chunker(name, cls, *, config=None, replace=False)`;
        `get_chunker_config(name)`; `build_chunker(name, **kwargs)` builds the
        config (unknown keys -> `TypeError` naming the valid fields); `word`
        and `semantic` registered with their configs; snapshot carries the
        config classes (optional section) and round-trips every built-in
        (structural check)
  - [x] bridges: lookup through the class MRO; a `ChunkerProtocol` chunker
        without a bridge gets a generic one (`CUSTOM`)
  - [x] builder: names through the registry; `chunker_kwargs` with an
        instance is refused; no silent `None` fallback on import failure;
        normalizer step names through the registry (unknown -> `ValueError`
        naming the valid ones); chunker and normalizer resolved before
        ingestion (a config error raises from `build`)
  - [x] R3-2: `ChunkerConfig.__post_init__` coerces enum fields from their
        values (field type = type of its enum default); an unknown value
        raises at construction
  - [x] also: `register_normalizer` checks `NormalizerBase` (the pipeline's
        type; the text-level `NormalizerProtocol` refused every built-in);
        every built-in chunker exposes `config` and refuses a config of another
        type; `RuntimeCorpus` resolves a chunker config through the registry
        pairs (the caller's registry first, then the built-ins); two stale
        doctests fixed (`build_chunker`, `bridge_chunker`)
  - [x] tests (`corpus/tests/test__chunker_names_and_configs.py`, 57),
        registry tests for the `NormalizerBase` check, guide section
        "Chunkers and normalisers by name" (sync-tested), API reference
        (`ProtocolChunkerBridge`), corpus `865.fix` / `865.api` bullets
  - [x] suites: only the baseline 15 environmental failures
  - [x] independent review: 14 findings (4 medium, 10 low), 13 fixed, 1
        documented (F9b: the old check varied by Python version; now static)
        -- see "Results review - 3a"
  - [x] negative proofs (`r34/neg3a/`): final tests on the reviewed version:
        22 fail; on 2g the new module does not import; final: 148 pass
  - [x] harness on `snap3a` (skinny + corpus + mcp, Python 3.9 and 3.13):
        203 passed, 4 failed (the corpus legs, 4647 passed, the same 14 DNS
        ids as 2g), 2 skipped (mcp on 3.9). Delivered as
        `scikitplot_r33_inc3a_dropin.zip` + `.diff` +
        `12_increment_3a_results.md`.

### Results review - 3a (2026-10-13)
- Review findings (each reproduced, `r34/review3a/`), dispositions:
  - F1 `register_builtins` overwrote user registrations, and the builder now
    calls it on the shared registry: insert-if-absent, `_builtin_keys`
    filled (Rule 116).
  - F2 plans used a fresh registry, so a chunker registered on the shared one
    was unknown to plans: `_component_registry(None)` is the shared registry
    (Rule 116).
  - F3 constructor-only options (`gensim_dictionary`, `adapter`, `config=` +
    `multilang_config`) became unreachable: config fields build the config,
    other options go to the constructor when it takes them, the rest raise.
  - F4 `normalizer_kwargs` was read nowhere: `{step: {options}}`, strict.
  - F5 a chunker class, or a `chunk(text, metadata=)` chunker, bridged and
    failed per source: classes refused; registration and the protocol bridge
    require `chunk(text, extra_metadata=...)` (static signature check).
  - F6 a pre-3a snapshot lost the configs: built-ins declare
    `config_class`, used when no config is given.
  - F7 a registered plain-dataclass config did not resolve in plans: any
    instance of a registered config class resolves.
  - F8 a mismatched config loaded and failed at first use: refused at
    registration (and so at snapshot load) against `config_class`.
  - F9 the protocol bridge overwrote a declared strategy; the runtime
    protocol check accepted `MagicMock` on 3.9-3.11: the declared strategy is
    kept; the check is static.
  - F10 unknown `adapter_unit` accepted: `ADAPTER_UNITS` checked by
    `WritingSystemAdapterConfig` and `SemanticChunkerConfig`.
  - F11 stale registry package doctest: fixed.
  - F12 `chunker_kwargs` beside a `chunker_factory` ignored: refused.
  - F13 enum coercion moves config fingerprints: stated in the api fragment.
  - F14 `config=None` raised: means the default config.
- Suites (b312): corpus + mcp + cleanprompt + _cli + top-level + guide 8875
  passed; failures are the baseline 15 (14 DNS, `test_public_api`).

- [x] 3b. Span model (R3-3)
  - Evidence (`r34/run3/p_span_all.py` on 3a): `text == source[start:end]`
    fails for sentence and paragraph overlap (text = context + anchor, span =
    anchor; sentence context joined with " ", inserting a space into CJK),
    for fixed windows with `strip_whitespace` (text stripped, span not) and
    token windows (tokens re-joined with " "), for the semantic merges
    (`" ".join`; an oversized split gives every piece the whole span), and
    for paragraph `merge_short` / `max_length` (re-joined text, then a
    search that falls back silently to the cursor). Downstream,
    `DocumentReader.get_documents` sets `char_end = char_start + len(text)`,
    so an overlap chunk's document span (and its `raw_text` slice) is the
    wrong region of the source.
  - Design: span-first. Chunkers compute spans in the source and take the
    text as a slice; `Chunk` gains `context_start` / `context_end` (default:
    the anchor), with `anchor_span` and `context_span`; invariant
    `text == source[context_span]` and context covers the anchor.
    `WordChunker` produces derived text (tokens, lowercase, stems) and
    declares it (`text_is_source_slice = False`): its context span is the
    source region the text came from. Locating a splitter's segment that is
    not a substring raises (no silent cursor fallback). The reader maps a
    document to the context span (its text) and records the anchor in
    `metadata["anchor_span"]` when they differ.
  - [x] `Chunk` fields/properties/validation; shared span helpers
        (`_chunkers/_spans.py`: `locate`, `strip_span`, `cover`)
  - [x] sentence, paragraph (span-first split, merge, long split), fixed
        window (chars, tokens), semantic (model merge, undersized merge,
        oversized split), `WordChunker` sentence-mode spans
  - [x] reader mapping + `raw_text`; `anchor_span` metadata; unknown span
        when offsets are disabled
  - [x] invariant test over 15 configs x 5 texts (`test__chunk_spans.py`,
        86); mocked-splitter tests given sources that contain their pieces;
        guide section "Chunk spans" (sync-tested); corpus `865.fix` bullet
  - [x] suites: only the baseline 15
  - [x] independent review (`r34/review3b/`): 9 findings, all fixed; a
        re-review of the fixes (`r34/review3b2/`): 5 findings, all fixed; see
        "Results review - 3b"
  - [x] negative proofs (`r34/neg3b/`): final tests on 3a: the new modules do
        not import (`_spans`, `_merge_short_spans`), and the Word probe gives
        `(0, n)` for every sentence; on the reviewed version 10 fail; on the
        first fix 5 fail (one per re-review finding); final: 169 pass
  - [x] harness on `snap3b` (skinny + corpus + mcp, Python 3.9 and 3.13):
        203 passed, 4 failed (the corpus legs, 4744 passed = 4647 + the 97
        new span tests, the same 14 DNS ids as 3a), 2 skipped (mcp on 3.9).
        Delivered as `scikitplot_r33_inc3b_dropin.zip` + `.diff` +
        `13_increment_3b_results.md`.

### Results review - 3b (2026-10-16)
- Review findings (each reproduced), dispositions:
  - F1 High: offsets disabled gave every document of a split parent the whole
    parent as `raw_text` -> `_slice_raw_text` returns `None` for an unknown
    region (Rule 117)
  - F2 Med: multilang preprocessing (NFC, BOM, controls) made spans index the
    preprocessed text while raw slices came from the input -> `span_basis`
    in the chunk result and the document metadata; no raw slice then
  - F3: sentence spans located only with offsets on -> always located
  - F4: Word sentence-span test too weak -> asserts each sentence's span
  - F5: overlap context includes `min_length`-dropped units -> documented
  - F6: the abbreviation placeholder leaked into multi-script splits ->
    fixed
  - F7: Word and Semantic ignored `include_offsets` -> honoured
  - F8: `_to_tuples` used the anchor start -> context start
  - F9: stale `_slice_raw_text` docstring, duplicate sync-test name, todo
    lost 3a evidence -> fixed, todo rebased on 4744fa4
- Re-review of the fixes:
  - 1 High (new regression): the only chunk got the whole parent even when
    `min_length` left part of it -> the whole parent only when spans do not
    index it and the chunk covers the whole chunked text
  - 2 Med: WordChunker recorded the preprocessed segment as multilang
    `raw_text` -> gated on `span_basis`
  - 3 Med: overlap multilang `raw_text` covered the anchor, the document the
    context -> context
  - 4 Low/Med: Word's `_raw_segment` won over the span checks -> used only when
    spans index the parent and no span is known
  - 5 Low: `ml["raw_text"]` used for a markup parent; `span_basis` recorded
    only with a span -> both fixed
- Suites (3.12, editable): corpus, mcp, cleanprompt, `_cli`, top-level,
  guide: only the baseline 15 (14 DNS, `test_public_api`)
- Gates: towncrier exit 0; names unique; path lengths 0 failing; generated
  files up to date; ruff no new product finding; black no new diff;
  invisible 0; Python 3.9 grammar 0 failures
- [x] 3c. Cache policy (R3-5)
  - Evidence (`r34/run3c/p_cache_base.py` on 3b): an unpinned engine is
    served the old model's vectors after an upstream change (hit, 1.0, fresh
    2.0); 50 distinct calls leave 50 entries, and no quota or eviction field
    exists; `MultimodalEmbeddingEngine.embed_documents_with_cache` keys on
    path + mtime + count: changed text, another custom function, or another
    `normalize` are served the first vectors, and a document left unembedded
    (image without `raw_tensor`) makes it raise from `np.stack`.
  - Design (one policy, one store, both engines):
    - `CachePolicy(unpinned="bypass"|"allow"|"resolve", revision_resolver=
      None, max_bytes=None, max_entries=None)`, frozen, validated.
    - Identity: a pinned `model_revision` (any backend) or a `custom_fn_id`
      makes a hit provable; unpinned follows `unpinned`: `bypass` (default)
      computes and does not cache; `allow` caches under the unpinned name
      (the old behaviour, by explicit choice); `resolve` calls
      `revision_resolver(model_name)` once per engine, then *loads* and keys
      that revision, and bypasses when it fails. `resolve_hub_revision`
      (huggingface_hub `model_info(repo_id).sha`, full repo ids only, no
      short-name guessing) is the shipped resolver.
    - Status: `CacheStatus` (hit, stored, store_failed, disabled,
      bypass_unidentified, bypass_unpinned) and `embed_cached` returning
      `CachedEmbeddings(embeddings, status, key, revision)`;
      `embed_with_cache` keeps its `(array, bool)` contract.
    - Store: `.npy` per 64-hex key; a hit refreshes the entry's mtime (LRU);
      after a store, entries beyond `max_entries` / `max_bytes` are evicted
      oldest first, ties by name; only `<64 hex>.npy` files are ever touched;
      a file removed by a concurrent evictor is a miss.
    - Multimodal: content-addressed key (schema, every backend and model,
      fusion, projection, normalisation, and each document's modality, text
      and `raw_tensor` dtype/shape/bytes); a custom function or projection
      bypasses (unidentified); its unpinned image/audio models bypass unless
      `unpinned="allow"`; stored only when every document received an
      embedding of one shape, in its own dtype.
  - No TTL: staleness is a question of identity (revision), not age.
  - Steps:
    - [x] `_embeddings/_cache_policy.py` + engine wiring + manifest revision
          (keys byte-identical to 3b over 300 random inputs,
          `r34/run3c/keys_3b.json` = `keys_3c.json`)
    - [x] multimodal key and store
    - [x] tests (`_embeddings/tests/test__cache_policy.py`, 60), guide
          section "Embedding cache" (sync-tested), API reference, corpus
          `865.api` / `865.fix` bullets
    - [x] independent review (`r34/review3c/`): 7 findings; re-review of the
          fixes (`r34/review3c2/`): 7 findings; all fixed, see "Results
          review - 3c"
    - [x] suites: only the baseline 15; gates clean
    - [x] negative proofs (`r34/neg3c/`): final tests on 3b: the module does
          not import; on the reviewed version 12 fail; on the first fix 3
          fail; final: 234 pass. Evidence probe on the final version:
          `r34/run3c/p_cache_final.out` (every stale hit gone)
    - [x] harness on `snap3c`: the encoding check failed (two
          `write_text("x")` in the new tests used the locale encoding) ->
          `encoding="utf-8"`; rerun on `snap3c2` (skinny + corpus + mcp,
          Python 3.9 and 3.13): 203 passed, 4 failed (the corpus legs, 4804
          passed = 4744 + the 60 new tests, the same 14 DNS ids as 3b), 2
          skipped (mcp on 3.9). Delivered as `scikitplot_r33_inc3c_dropin.zip`
          + `.diff` + `14_increment_3c_results.md`.

### Results review - 3c (2026-10-16)
- Review findings (each reproduced), dispositions:
  - 1 High: a hit raced with a concurrent eviction (`exists()` then
    `stat()`), `FileNotFoundError` from `embed_cached` -> one stat in a
    try; a vanished entry is a miss (Rule 118)
  - 2 Med: one result larger than `max_bytes` evicted every other entry,
    then itself -> it alone is dropped, nothing else is touched
  - 3 Med/Low: an old entry that cannot be removed made the new one be
    deleted and reported `TOO_LARGE` -> the stored entry is never evicted to
    make room; the bound is met at a later store
  - 4 Low: a directory that cannot be listed raised after the store -> best
    effort, logged at debug level; the docstring's `OSError` was untrue
  - 5 Low/Med: a branch name (`"main"`) counted as pinned -> for
    sentence-transformers only a 40/64-hex commit pins
  - 6 Low: stale docstrings (`embed_documents`, multimodal `input_path`)
  - 7 Low: `resolve_hub_revision` accepted a local directory, and the
    resolver was asked for the openai backend -> both refused
- Re-review of the fixes:
  - 1 Low: `model_revision=None` reached the key on openai -> `TypeError`
    at construction (also for a non-`CachePolicy` `cache_policy`)
  - 2, 3 Low: two NumPyDoc sections broken by the edits -> fixed
  - 4 Low: a stored entry that vanished before the scan cost one extra
    eviction -> counted only when seen
  - 5 Low: a branch with `unpinned="resolve"` bypassed with a misleading
    notice, and the guide did not say "lowercase" -> its own notice; docs
  - 6 Low: `CachedEmbeddings.revision` docstring -> corrected
  - 7 Low: eviction between the stat and the load logged a warning -> miss
- Ruff (too many returns/branches) -> `_read_entry`, `_scan_entries`,
  multimodal `_store`
- Suites (3.12, editable): only the baseline 15. Gates: towncrier exit 0;
  names unique; path lengths 0 failing; generated files up to date; ruff
  and black: nothing new; invisible 0; Python 3.9 grammar 0 failures
- [x] 3d. Log hygiene (R3-7); resource hygiene (R3-8) moves to 3e
  - Evidence (`r34/run3d/p_logs.py` on 3c): two `SemanticChunker`s and
    three `chunk()` calls logged 7 warnings: the Japanese-strategy warning 4
    times, the regex warning twice, jieba once. With `regex` installed
    (`r34/v3d_extra`, Python 3.12 wheel), every text, "東京 Hello, 123!"
    included, came back as one `ScriptType.UNKNOWN` span with the warning
    "unrecognised script property '_unknown'": `ScriptSegmenter` calls
    `regex.get_script`, which the `regex` library does not have (`dir(regex)`
    holds no script function; the `AttributeError` was swallowed). Its spans
    were documented as grapheme-cluster indices while every strategy used
    them as character offsets.
  - Design:
    - `corpus/_notices.py`: `notice_once(logger, key, message, *args)`, one
      notice per process per key, and `reset_notices` for tests; every
      missing-segmenter warning (camel-tools, jieba x2, Japanese, kss,
      PyThaiNLP, the dictionary fallback per script, regex) goes through it
      and names `component_capabilities(role="segmentation")`.
    - Capability specs for the nine optional segmenters (the on-demand
      report instead of runtime noise).
    - `ScriptSegmenter`: one alternation of `\p{Script=X}` named groups
      (documented `regex` syntax, every map key compiles); character offsets
      (`text[start:end] == span.text`); script-free text is UNKNOWN without
      a notice; a script the map does not name is its own UNKNOWN span,
      noticed once; `min_span_chars` merges count grapheme clusters and leave
      one span per run.
  - Steps:
    - [x] notices, capability specs, warning sites
    - [x] `ScriptSegmenter` script lookup, offsets, merge
    - [x] tests (`_chunkers/tests/test__notices_and_scripts.py`; the script
          tests need `regex` and are skipped without it; run here with
          `PYTHONPATH=r34/v3d_extra`), guide section in
          `troubleshooting.rst` (sync-tested), `865.fix` / `865.enhancement`
    - [x] suites without `regex`: only the baseline 15; corpus with `regex`:
          only the 14 DNS tests
    - [x] independent review (`r34/review3d/`): 7 findings; re-review of
          the fixes (`r34/review3d2/`): 6 findings; all fixed, see "Results
          review - 3d"
    - [x] gates clean; suites: baseline 15 (corpus with `regex`: the 14 DNS)
    - [x] negative proofs (`r34/neg3d/`, final tests, with / without
          `regex`): 3c: `_notices` does not import; reviewed version 10 / 1
          fail; first fix 6 / 0 fail; final 128 / 104 pass (24 skipped
          without `regex`)
    - [x] harness on `snap3d` (skinny + corpus + mcp, Python 3.9 and 3.13):
          203 passed, 4 failed (the corpus legs, 4811 passed: 4804 + the new
          tests that need no `regex`; the same 14 DNS ids as 3b), 2 skipped
          (mcp on 3.9). Delivered as `scikitplot_r33_inc3d_dropin.zip` +
          `.diff` + `15_increment_3d_results.md`.

### Results review - 3d (2026-10-16)
- Review findings (each reproduced), dispositions:
  - 1 High: with `regex`, Japanese and Korean were cut at every kanji/kana
    (Hanja/Hangul) boundary, and kanji went to the Chinese strategy ->
    `_group_writing_systems`: Han that meets kana (Hangul) letter to letter
    is one Japanese (Korean) span
  - 2 Med: chunk output changes for `regex` users, undocumented -> fragment
  - 3 Med: Tamil, Telugu, Kannada, Malayalam, Sinhala and Lao (documented
    members of `SOUTH_ASIAN` / `SOUTHEAST_ASIAN`) were unmapped -> mapped;
    Bengali, Gurmukhi, Gujarati, Oriya, Bopomofo stay unmapped (no
    documented `ScriptType`)
  - 4 Low/Med: a MeCab that cannot start warned per construction -> notice
    per backend; the report says "importable"
  - 5 Low: a notice suppressed by the log level was used up -> gated on
    `isEnabledFor`
  - 6 Low: one notice key for every unmapped script -> per script
  - 7 Low: four stale doc claims (ScriptSpan, `_schema.script_spans`,
    fragment, troubleshooting) -> fixed
- Re-review of the fixes:
  - 1 Med: grouping crossed sentence boundaries (a Chinese sentence went to
    the Korean strategy) -> only letter-to-letter joins
  - 2 Med: unnamed characters (private use, surrogates) got one notice each,
    and "OLD ..." scripts shared one -> key = the script part of the Unicode
    name, or the general category for unnamed characters
  - 3 Low: a `min_span_chars` merge labelled kana as Han -> Han absorbing
    kana / Hangul takes that script; neighbours re-coalesced
  - 4, 5 Low: doc wording (`logging.disable` vs handler filters; "Indic")
  - 6 Low: script-free text could be two UNKNOWN spans -> one key
  - Probe found adjacent same-script spans after joins -> one span per run,
    held by a fixed-seed fuzz (500 texts x 3 `min_span_chars`)
- Suites (3.12, editable): baseline 15; corpus with `regex`: the 14 DNS.
  Gates: towncrier exit 0; names unique; path lengths 0 failing; generated
  files up to date; ruff and black: nothing new; invisible 0 (the test file
  first held a literal ZWJ, Rule 52; replaced by escapes); Python 3.9
  grammar 0 failures
- [x] 3e. Resource hygiene (R3-8)
  - Evidence (`r34/run3e/`, the corpus wheel installed on Python 3.13, a
    tracker plugin `r34/plug/conn_tracker.py`): 182 SQLite connections
    opened, 80 still open at exit, all from tests
    (`_storage/tests/test__storage.py` setup methods and `_store()`,
    `tests/test__filter_capability.py` fixture); 17 surfaced as
    `ResourceWarning: unclosed database` in unrelated tests. `StorageBase`
    has no `close()` or context manager; `SQLiteStorage.close()` exists.
    `mcp` `Bm25Retriever.from_corpus_sqlite` holds a connection nothing can
    close.
  - Design: `StorageBase.close()` (no-op default), `__enter__`/`__exit__`;
    tests close what they open; a package autouse guard
    (`corpus/conftest.py`, `SQLiteLeakGuard`) fails the test that leaves a
    connection open and names where; `Bm25Retriever(close=...)`, `close()`,
    context manager.
  - [x] code, tests (`_storage/tests/test__resource_hygiene.py`, mcp
        `test_bm25_from_corpus_closes_its_store`), troubleshooting section
        (sync-tested), corpus `865.enhancement`, mcp `865.api`
  - [x] Python 3.13 rerun with `-W error::ResourceWarning`: 192 opened, 0
        open at exit (`r34/run3e/conn_final.txt`); suites: baseline 15
  - [x] independent review (`r34/review3e/`): 5 findings, all fixed (see
        "Results review - 3e")
  - [x] gates clean; suites: baseline 15; Python 3.13 (wheels, `-X dev`,
        `-W error::ResourceWarning`): 195 opened, 0 open at exit, only the
        14 DNS failures
  - [x] negative proofs (`r34/neg3e/`): final tests on 3d: 7 failed, 44
        errors; the final guard over 3d's own tests: 80 errors, one per
        connection the evidence counted; reviewed version: 1 failed, 2
        errors; final: 148 pass
  - [x] harness on `snap3e` (skinny + corpus + mcp, Python 3.9 and 3.13):
        203 passed, 4 failed (the corpus legs, 4820 passed, the same 14 DNS
        ids as 3b), 2 skipped (mcp on 3.9); mcp on 3.13: 324 / 329 passed.
        Delivered as `scikitplot_r33_inc3e_dropin.zip` + `.diff` +
        `16_increment_3e_results.md`.

### Results review - 3e (2026-10-16)
- Review findings (each reproduced), dispositions:
  - 1 Med: a materialisation refused after the store was opened leaked the
    owned SQLite store -> the store is opened last and closed on any error
  - 2 Med: `SQLiteStorage` on a file that is not a database leaked its
    connection (the PRAGMAs ran outside the closing `try`) -> closed
  - 3 Low: docs claimed `RuntimeCorpus` closes its store without saying
    when, and that the guard sees every connection -> reworded
  - 4 Low: the guard raised on a connection of another thread and stopped
    -> reports it and continues
  - 5 Low: `HybridRetriever` could not close its BM25 leg -> `close()` and
    context manager over every leg, first error re-raised after all closed
- Suites, gates as above
- [ ] R3-6 waits for D4
