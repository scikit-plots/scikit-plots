<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 33 / Run 2 - retrieval contract (one fusion, one filter, durable identities, one fallback rule)

### Context
- User request (2026-10-13, PR 865): start Run 2 from
  `01_verdict_and_run_plan.md` §3: R2-1 one fusion, R2-2 one filter, R2-3
  durable identities, R2-5 fallback/capability vocabulary (F-32, F-40, F-50);
  R2-4 is done (2a). Decisions: D2 = recommended (tie-break by `doc_id`, a
  document counts once per leg at its best rank, weight-0 legs add nothing,
  `hybrid_alpha` is shorthand for two leg weights); D3 = (a) 64-hex ids
  everywhere, stores rebuilt, **plus** "officially supported" RFC 9562 UUIDv7
  document ids. D1, D4, D5: later.
- Reading of the UUIDv7 request (stated to the user at the start): the
  default stays content-addressed (idempotent rebuilds, qrels keyed by
  `doc_id`); UUIDv7 is an opt-in scheme, because a minted id is new on every
  build. Under it, `content_hash` (64 hex) is the dedup key.
- Grounding for UUIDv7: RFC 9562 §5.7 (layout: 48-bit `unix_ts_ms`, ver 0b0111,
  12-bit `rand_a`, var 0b10, 62-bit `rand_b`), §6.2 (Method 1 counter 12-42
  bits, seeded randomly per tick; rollover: wait or advance the timestamp;
  monotonic error checking: reuse the previous timestamp and increment), §6.9
  (CSPRNG, reseed on fork), §6.11 (text form sorts), §8 ("MUST NOT be used as
  security capabilities"; the timestamp signals creation order); appendix A.6
  test vector `017f22e2-79b0-7cc3-98c4-dc0c0c07398f` = 1645557742000 ms.
  CPython 3.14 documents `uuid.uuid7()` with a 48-bit timestamp and a 42-bit
  counter; not used, so 3.9-3.13 and 3.14+ share one implementation.

### Increments
- [x] 2d. Durable identities (R2-3, D3 = a + UUIDv7)
  - [x] `corpus/_identity.py` (new): `IDENTITY_HEX_WIDTH`, `DocIdScheme`,
        `UUID7Generator` (lock, fork reset, clock regression, rollover,
        injectable clock/random), `uuid7`, `uuid7_unix_ms`,
        `doc_id_scheme_of`, `assign_doc_ids` (parent links kept; a parent
        outside the batch or ambiguous is refused), `require_full_width`
  - [x] `make_doc_id` full digest (id schema v3), `make_content_hash` 64 hex,
        document `schema_version` 3.0 (2.x refused)
  - [x] `EmbeddingManifest` (`em2`) and `IndexGeneration` (`gen3`) full
        digest; 16-hex stored fingerprints refused with "rebuild"
  - [x] multimodal embedding cache key, preprocessing-trace hashes (MD5 ->
        SHA-256)
  - [x] `BuilderConfig.doc_id_scheme` (appended last), `CorpusPipeline(
        doc_id_scheme=)`; applied once per build/source after ingestion
        (`_collect_documents` serves `run`, `run_batch`, `run_url`)
  - [x] cleanprompt `redact_documents` keeps each document's scheme
  - [x] tests (`test__identity.py`, width tests, cleanprompt), guide section,
        API reference (+ template), fragments (bullets folded into the
        existing 865 corpus bullets; cleanprompt `865.enhancement.rst`)
  - [x] independent review (12 findings, all fixed), negative proofs, suites,
        gates, cleanprompt EVIDENCE fingerprint refreshed after the last edit
  - [x] CI harness on the final snapshot (`snap2d_v2`): 206 passed, 4 failed
        (the corpus legs, 4420 passed, the same 14 DNS ids each). Delivered as
        `scikitplot_r33_inc2d_dropin.zip` + `.diff`.
- [x] 2e. One fusion (R2-1, D2)
  - [x] `_fusion.py` contract (`FusionPolicy`, `FusedItem`,
        `FusionContribution`), byte-identical in corpus and mcp, registered in
        `DUPLICATED_CONTRACTS`; weighted RRF, `k >= 1`, weights finite >= 0,
        weight-0 legs add nothing, a key once per leg at its best rank, exact
        order (float first, `Fraction` inside the rounding window), ties by key,
        `fsum` scores, contributions returned; `from_alpha`
  - [x] `RetrievalIndex._fuse_hybrid`, `fuse_by_rank`,
        `reciprocal_rank_fusion` / `HybridRetriever` call it;
        `RetrievalConfig` validates `rrf_k`; weight-0 hybrid legs are not run
        (SKIPPED)
  - [x] tests (contract in both submodules, randomised differentials against a
        reference and across all paths), guide section, API reference,
        fragments (corpus `865.api` bullet, `scikitplot.mcp/865.api.rst`)
  - [x] independent review: 10 findings (2 medium, 8 low), all fixed; 16
        tests fail on the reviewed version, 12 integration tests on 2d
  - [x] suites (b312): corpus + mcp + top-level 5653 passed; failures are the
        14 DNS ids and `test_public_api` (httpx absent, also on the baseline)
  - [x] CI harness on a snapshot (`snap2e`), skinny + corpus + mcp, Python
        3.9 and 3.13: 203 passed, 4 failed (the corpus legs, 4471 passed, the
        same 14 DNS ids), 2 skipped (mcp on 3.9: its extra needs >= 3.10).
        Gates PASS. Delivered as `scikitplot_r33_inc2e_dropin.zip` + `.diff`
        + `09_increment_2e_results.md`.
- [x] 2f. One filter (R2-2)
  - [x] `search(..., filter=)` on `RetrievalIndex`, `CustomRetrievalIndex`,
        `CorpusBuilder`, `RuntimeCorpus`; every leg filters before ranking
        (strict, keyword, semantic, hybrid); dense: doubling candidate
        expansion, exact cosine over the admitted rows when the backend
        returns short; `LegOutcome.filter_capability` (EMULATED)
  - [x] `SecurityFilter(expression, label)`: frozen to the JSON-able form
        when bound, AND-ed into every query on `search` and on the vector
        seam `query`; BM25 statistics over the scope; responses report the
        label; `BuilderConfig.security_filter` binds every index a builder
        builds
  - [x] JSON form (`filter_to_dict`, `filter_from_dict`, `validate_filter`):
        fields validated (`filterable_fields()`, `metadata.<key>`), limits on
        depth (32), nodes (256), values (4096), strings (4096), ints (2**63);
        `In`/`NotIn` precompute their lookup; uncomparable values do not
        match; `True` is not `1`
  - [x] tests `test__filter_search.py` (52), guide section "Filtering a
        search" (sync-tested), API reference + template, corpus `865.feature`
        bullet extended
  - [x] independent review: 10 findings (1 high, 4 medium, 5 low) + the
        label design point, all fixed; negative proof: 18 tests fail on the
        reviewed version (`r34/neg2f/result.txt`), 52 pass on the fixed one;
        on 2e the module does not import (`SecurityFilter` absent)
  - [x] suites (b312): corpus + mcp + top-level + guide 5706 passed; failures
        are the 14 DNS ids and `test_public_api` (httpx absent), as on 2e;
        maintenance planes: cleanprompt PASS, corpus FAIL before and after
        (F-53)
  - [x] gates (ruff: no new product finding after the `filter_from_dict`
        split; codespell only the `NotIn` class name)
  - [x] CI harness on `snap2f`, skinny + corpus + mcp, Python 3.9 and
        3.13: 203 passed, 4 failed (the corpus legs, 4523 passed, the same 14
        DNS ids as 2e), 2 skipped (mcp on 3.9). Delivered as
        `scikitplot_r33_inc2f_dropin.zip` + `.diff` +
        `10_increment_2f_results.md` (drop-in over 2e == the 2f commit)
  - Follow-ups (not in 2f): `RuntimeCorpus` and the MCP corpus factory take
    no scope yet (R5-3); storage-layer filter vocabulary still separate.
- [x] 2g. Fallback/capability vocabulary (R2-5, F-32, F-40, F-50)
  - Goal: every corpus component that can run an implementation other than
    the one configured reports the same names (`requested_backend`,
    `backend`, `fallback_reason`, `allow_fallback`); an option the effective
    implementation cannot honour is a fallback (or, with
    `allow_fallback=False`, an error), never a silent discard.
  - Scope (components that substitute an implementation):
    SemanticChunker (done, Run 1); `select_backend` / `RetrievalIndex` dense
    backend; `AnnoyBackend` binding (`impl`); `run_backend_chain` outcomes
    (readers). Out of scope, recorded: levenshtein and mcp spell the switch
    `strict` (an API rename, later); a dense index disabled by invalid
    embeddings is a data error, not a substitution.
  - [x] `_similarity/_backends.py`: declared option table per Annoy binding;
        `impl="auto"` picks a binding that accepts every requested option
        (F-40); explicit binding + unsupported option -> fallback
        (`option_unsupported`) or `ValueError` when `allow_fallback=False`;
        `is_installed()` (no import) next to `is_available()`;
        `select_backend(..., allow_fallback=True)`: under `"auto"` an
        installed but unloadable backend is skipped with
        `fallback_reason="<name>_broken"`, or raises when strict; every backend
        instance reports `provenance()`
  - [x] `RetrievalConfig.allow_fallback` (default True) reaches
        `select_backend` and Annoy; dense legs carry `requested_backend` and
        `fallback_reason`; `RetrievalIndex.backend_provenance`
  - [x] `BackendOutcome.requested_backend`, `allow_fallback`,
        `fallback_reason` (`failed` / `no_result` / `skipped`)
  - [x] F-50: `probe_backend(cls, load=True)` classifies an installed backend
        that fails to load as BROKEN; `capability_snapshot(load=False)` loads
        no backend (annoy reports `unknown: not_loaded` unless already loaded);
        the hygiene test watches scikit-learn and SciPy for the snapshot too
  - [x] F-32: `CommandSpec.capabilities` (read nowhere) deleted; the delegate
        import already reports a missing part with its install hint
  - [x] tests (`corpus/tests/test__fallback_vocabulary.py`, the S-43 discard
        tests rewritten for the declared table, the catalog absent-vs-broken
        test, the snapshot hygiene test now watching scikit-learn and SciPy),
        guide section "What ran, what was asked, and why they differ"
        (sync-tested), fragments (corpus `865.api` bullet,
        `scikitplot._cli/865.api.rst`)
  - [x] `tests/test_submodule_independence.py`: the scanner counted a
        function's direct statements as module level (it flagged the new
        `_import_annoy_impl`) and missed nested module-level blocks; fixed,
        with a scanner test
  - [x] suites (b312): corpus + mcp + _cli + top-level + guide 5924 passed;
        failures are the 14 DNS ids and `test_public_api` (as on 2f)
  - [x] independent review: 13 findings (1 high, 6 medium, 6 low), dispositions
        in "Results review - 2g"; negative proofs; gates
  - [x] harness on `snap2g` (skinny + corpus + mcp, Python 3.9 and 3.13):
        203 passed, 4 failed (the corpus legs, 4568 passed, the same 14 DNS
        ids as 2f), 2 skipped (mcp on 3.9). Delivered as
        `scikitplot_r33_inc2g_dropin.zip` + `.diff` +
        `11_increment_2g_results.md`.
  - Follow-ups (not in 2g): PDFReader per-page and NLPEnricher stopword
    fallbacks do not report the four names; levenshtein and mcp spell the
    switch `strict`.

### Results review - 2g (2026-10-13)
- Independent review (Rule 92) of the first version: 13 findings, each
  reproduced (`r34/review2g/`):
  - F1 (high) `allow_fallback=False` did not raise through `RetrievalIndex`:
    the Annoy refusal was a `ValueError`, caught by the "invalid embeddings"
    handler, so dense search turned off silently (a typo'd `annoy_dtype`
    too): embeddings are validated first and alone; a backend error
    propagates; refusals raise the new `FallbackRefusedError(RuntimeError)`
    (Rule 115).
  - F2 a subclass of `AnnoyBackend` lost the switch (`cls is` vs
    `isinstance`): `issubclass`.
  - F3 the new `RetrievalConfig` field moves plan fingerprints: kept (the
    switch decides between an index and an error, so it belongs to the
    plan); stated in the fragment.
  - F4 `DenseRetriever` legs lacked `requested_backend`/`fallback_reason`:
    one helper, `RetrievalIndex._backend_fields`, feeds every dense leg.
  - F5 a strict chain reported a fallback after a policy skip:
    `requested_backend` is the first *eligible* candidate (a skip is like an
    absent backend under "auto"); reasons `failed` / `no_result`.
  - F6 an absent custom backend became BROKEN: BROKEN needs evidence
    (installed and a failed `load_error()`) (Rule 115).
  - F7 a FAISS/Voyager that is installed but fails to load read as available:
    the base `load_error()` imports the dependency; FAISS and Voyager declare
    `loads_on_probe`.
  - F8 a misspelt `dtype` dropped by a fallback is not caught: documented
    (the binding that honours the option owns its vocabulary; corpus does
    not copy annoy's); the value is reported verbatim; types are checked.
  - F9 `capabilities()["dtype"]` named the class default: the built
    precision.
  - F10 `provenance()` could hold non-JSON values: `dtype`/`index_dtype` must
    be non-empty `str`.
  - F11 snapshot `available` False for UNKNOWN: kept (`available` is derived
    from `status`, a bool by contract); documented in guide and fragment.
  - F12 scanner ignored `import x.y`, `from scikitplot import y`, and counted
    `if TYPE_CHECKING:` bodies: handled, scanner test extended.
  - F13 `impl=None` meant "auto" before: accepted again.
  - Scope note (code reading): PDFReader's per-page fallback and NLPEnricher's
    stopword fallback report none of the four names; the guide and fragment
    now name the covered components (reader chains, vector backends and
    bindings, SemanticChunker). Follow-up.
- Negative proofs (`r34/neg2g/`): final tests on the reviewed version: 16
  fail, one or more per fixed finding; on 2f: 27 fail or do not collect; on
  the final version all pass except 4 that need packages absent from the
  partial environment (annoy, rank_bm25; same on every version). Scanner:
  `scanner_neg.py`, the 2f scanner gets 6 of 10 probe lines wrong.

  - Acceptance: `AnnoyBackend(impl="auto", dtype="float64")` builds on the
    Cython binding with nothing discarded; the same with `impl="highlevel"`
    reports `option_unsupported` (or raises when strict); a monkeypatched
    broken annoy makes `select_backend("auto")` report `annoy_broken` (or
    raise when strict) and the snapshot report `broken`; a fresh process
    running `capability_snapshot()` imports neither scikit-learn nor SciPy.

### Not widened, on purpose (2d)
- `_url_handler._make_temp_filename`: a 16-hex name inside a per-download
  temporary directory, not a stored identity.
- `CorpusPlan.short_fingerprint`: display only, documented "never persist".
- `_embeddings/_hashing.py` `blake2b(digest_size=8)`: the hashing trick maps a
  token to a vector column; not an identity.
- cleanprompt audit lines (`plan=fingerprint[:16]`, `output_sha256[:16]`): log
  abbreviations outside the retrieval contract.

### Results review - 2d (2026-10-13)
- Independent review (Rule 92) of the first version: 12 findings (1 high,
  4 medium, 7 low); all fixed:
  - F1 (high) an old JSONL/SQLite store read as empty (every 2.0 row skipped
    as malformed): `IncompatibleSchemaError(ValueError)`; JSONL refuses at
    open, SQLite checks at open and every read path re-raises it.
  - F2 old pickle/joblib exports loaded silently: exports now hold
    `{"format", "schema_version", "documents"}`; a bare list is refused.
  - F3 old `PipelineGuard` checkpoints resumed nothing: lines record
    `schema_version`; another generation is refused.
  - F4 an external parent made "uuid7" fail; builder (per build) and pipeline
    (per source) had different scopes: external parents kept verbatim; both
    assign per source.
  - F5 only the module generator was fork-safe; a lock held at fork hung the
    child: every instance compares `os.getpid()` and resets.
  - F6 cleanprompt re-minted unchanged documents and linked to an ambiguous
    parent silently: same id when the text is unchanged; ambiguity refused.
  - F7 `HookableCorpusPipeline`, `create_corpus`, `FluentCorpus` could not
    choose the scheme: forwarded; plan domain `identity` (ingest scope).
  - F8 a scheme set after construction failed after ingestion: validated on
    assignment (`__setattr__` / property).
  - F9 a repeating or decreasing generator was accepted: refused.
  - F10 `errors="replace"` made distinct texts share ids (and a
    surrogate-escaped file name crashed `make_doc_id`): `surrogatepass` in
    every identity hash.
  - F11 `StrEnum` backport printed `Class.MEMBER` on 3.9/3.10: `__str__`,
    `__format__`.
  - F12 stale or misleading wording (cache key `[:24]`, "migrate", scope).
- Negative proofs: on the reviewed version 23 of the fixed tests fail
  (`neg2d_review/result.txt`); on the uploaded tree 15 identity tests and 7
  width tests fail; F11 by probe on Python 3.9 (`E.A` vs `alpha`).
- Suites (b312, editable): corpus + cleanprompt + mcp + top-level 8239
  passed; failures are the 14 DNS ids and environment-only modules
  (pydantic/httpx absent; identical on the baseline install); mcp with its
  deps installed 1040 passed. Maintenance planes: cleanprompt 90 passed after
  the refresh; corpus 4 failed before and after (F-53); levenshtein 8.
- Gates: towncrier PASS (fragments folded into the existing bullets: one bullet
  per fragment is required); names unique; path lengths 0 failing;
  `libs._tools check` up to date; ruff no new finding in product files; black
  applied to files that were clean; codespell only pre-existing hits;
  invisible characters 0; 3.9 grammar 0 failures.
- Process: a `pkill -f` on the same command line as the pattern killed the
  shell (Rule 8, repeat); runs now go through `run_suite.sh`, which never
  kills, and a pid file.

### Acceptance criteria (2d)
- Every durable SHA-256 identity in corpus is 64 hex; a v3 `doc_id` shares no
  prefix with the v2 id of the same document.
- A stored 16-hex manifest/generation and a 2.x document payload are refused
  with the remedy; a full-width tampered fingerprint is still "altered".
- UUIDv7: RFC 9562 layout and the A.6 vector; strictly increasing within a
  generator through same-ms bursts, clock steps back and counter rollover;
  unique across 8 threads; a forked child starts a new sequence.
- Both producer paths apply the scheme; parent links survive; an unknown
  scheme fails at construction.
