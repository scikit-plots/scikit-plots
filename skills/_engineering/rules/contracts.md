# API and data contracts

Categories: Validation, Inputs, Options, Contracts, Interfaces, Errors, Fallbacks, Responses, Loaders, Exporters, Identities, Untrusted structured output, Name tables, Design, Protection, Guards, Scope, Caller-supplied resources

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 26: Errors - report the error of the API that failed
- **When:** a Win32 call fails (`MoveFileEx`, `SetEndOfFile`).
- **Then:** read `GetLastError()` immediately and put that code in the message; `errno` is not set by these calls.
- **Verified by:** the message of a failed replace on Windows contains "Windows error <n>", not "No error (0)".
- **Added:** 2026-10-07. Root cause: one error helper (`set_error_from_errno`) used after every call, whichever API the call belongs to.

### Rule 36: Design - what a binary can do and what a run does are two switches
- **When:** a capability costs something at build time (a thread library, a platform that lacks it) and its use is a per-user choice.
- **Then:** one build switch that says whether the capability is compiled in, one run-time switch that says whether it is used, a function that reports both, and a default for the run-time switch that changes nothing for people who did not ask.
- **Verified by:** `threads_info()`; the table in `resolve_n_jobs`; both annoy suites on a wheel with threads in `auto` and in `multi`.
- **Added:** 2026-10-07. Root cause of the question: one switch had to answer both "may releases contain threads" and "do my builds use them".

### Rule 50: Guards - a false positive is fixed in the classifier, never by emptying its input
- **When:** a safety check refuses a configuration that is legitimate.
- **Then:** find the rule that misclassified it, narrow that rule to the contract (here: declared members, not a shared prefix), and add the legitimate configuration as a regression test next to the refusal tests.
- **Verified by:** the refusal tests still pass, and a test named after the legitimate configuration passes.
- **Added:** 2026-10-09. Root cause: `check_namespace` took any `_sphinx_ext.` name as a stack root; the docs' unrelated `_sphinx_ext.mpl_ext` helpers tripped it, and the workaround filtered every name away, which switched the guard off and failed 13 tests.

### Rule 57: Protection - check the forms your own generator produces, not only the form you were given
- **When:** a guard decides whether generated text could reveal an input (stand-ins, slugs, file names, IDs built from names).
- **Then:** list every transformation the generator applies (case, separators, joining, truncation) and make the guard match the input through each of them; test with a multi-word input, not only a one-word one.
- **Verified by:** a regression test with a two-word held value against e-mail, URL and slug-shaped candidates (`TestCP105...`).
- **Added:** 2026-10-10. Root cause: `CP-071` matched held values joined by spaces; the generator builds e-mail local parts by joining words with `.`, so `Marion Holt` passed as `marion.holt@example.invalid`. Found by reading the gallery output of a new feature, not by a test.

### Rule 68: Validation - check the type before any operation that only some types support
- **When:** validating a value that may be one of several types (`str | int | float | bool`).
- **Then:** branch on the type first and apply length/range checks inside the branch; test one value of every accepted type and one of each refused shape.
- **Verified by:** `test_custom_options_accept_every_scalar` and `test_custom_options_refuse_non_scalars_and_oversize` (7 of their cases failed on the regressed code).
- **Added:** 2026-10-11. Root cause: a lint fix hoisted `len(item) > 1024` above `isinstance(item, str)`, so an integer option raised `TypeError` instead of a configuration error; the tests had only string options.

### Rule 82: Guards - a guard over provenance tags is tested from the producer, end to end
- **When:** a consumer refuses mixed provenance (manifest ids, schema versions, model fingerprints).
- **Then:** the test builds both inputs through the public producer API, never by hand-setting the tag, and asserts that the guard fires. Every producer path, including convenience wrappers, delegates to the one that stamps the tag.
- **Verified by:** `TestEmbeddingManifestProvenance::test_an_index_refuses_two_providers_vectors` and `test__builder_customization_seams.py::test_builder_embeddings_are_cached_and_carry_their_manifest`.
- **Added:** 2026-10-12. Root cause: `_reject_mixed_manifests` compared `embedding_manifest_id`, which no producer ever set. The guard's tests set the tag by hand, and `CorpusBuilder` re-implemented embedding without the engine's cache or provenance. Rule 41 in another form: a fixture must hold what the producer actually stores.

### Rule 84: Validation - every public entry point validates, not only the first one written
- **When:** several public methods accept the same argument (a query, a path, a config).
- **Then:** route all of them through one validator, and keep one parametrised test over every entry point and every refused shape.
- **Verified by:** `test__rank_bm25_input_contract.py`: 3 scorers, 5 entry points and 3 refused queries.
- **Added:** 2026-10-12. Root cause: `get_batch_scores`, `get_top_ids` and `candidate_count` skipped `_require_query_tokens`, so a `str` query was scored one character at a time. Rule 68 tested refused shapes for one entry point only.

### Rule 85: Fallbacks - a result names the backend that ran, the one requested, and why they differ
- **When:** a component can fall back to another implementation (a model, a native backend, an optional package).
- **Then:**
  - The result reports `requested_backend`, the effective `backend` and a `fallback_reason`.
  - Provenance is decided after the run, never at construction.
  - Model fields are set only when a model ran: in the result and in every record it contains.
  - Every fallback path is covered: a missing package, a model that fails to load, and a model that fails at run time.
  - The missing-package fallback is warned once per instance.
  - `allow_fallback=False` raises on every path.
- **Verified by:** `corpus/_chunkers/tests/test__semantic.py`: 16 tests with offline fake models (working, failing to encode, failing to load, absent).
  - On the uploaded code all 16 fail.
  - On this round's first fix 8 fail: it covered the result level and the missing package only.
  - On its second fix 2 fail: it missed the load failure.
- **Added:** 2026-10-12.
- **Root cause:**
  - After falling back, `SemanticChunker` reported `backend="hybrid"`, `chunking_strategy="semantic_hybrid"` and a model name with no model run.
  - The constructor wrote the model name into every chunk's metadata, MORPHOLOGICAL runs included.
  - A model that failed while encoding was swallowed inside the refinement step, so even `allow_fallback=False` returned "hybrid".
  - A model that failed to load escaped on the first call; later calls fell back under "sentence_transformers_unavailable", with the package installed.

### Rule 89: Interfaces - a new path to existing state inherits every precondition and convention of the old paths
- **When:** adding an access path (a buffer export, an iterator, an overload) or a field to an existing class.
- **Then:**
  - List the preconditions the existing paths check (here: closed, readable, writable) and enforce each on the new path.
  - Add a dataclass field last, so positional construction keeps its meaning.
  - A remembered exception keeps its traceback but drops the frames' locals (`traceback.clear_frames`).
- **Verified by:**
  - `test_an_unreadable_mapping_exports_nothing_and_cannot_be_read`, a child process.
  - `test_positional_construction_keeps_its_meaning`.
  - `test_a_failed_load_does_not_keep_the_half_loaded_model`, which uses a weakref.
- **Added:** 2026-10-12.
- **Root cause:**
  - The buffer export checked "closed" and "writable" but not "readable": `bytes(m)` on a `PROT_NONE` mapping raised SIGSEGV.
  - `custom_fn_id` was inserted before `cache_dir`.
  - A stored load error held the half-loaded model alive.

### Rule 93: Inputs - an object with `keys()` and `items()` is a mapping or is refused, never iterated as a collection
- **When:** a public function accepts "a mapping or a sequence" for one row (judgments, a ranking, labels per query).
- **Then:** test `collections.abc.Mapping` first; an object that is not one but has callable `keys` and `items` (a pandas `Series`, a `dict`-like store) is refused with advice (`.to_dict()`, `.tolist()`), or converted explicitly. Top-level containers keyed by id align by key; two keyed containers whose keys differ are refused.
- **Verified by:** `TestMappingLikes` in `corpus/_metrics/tests/test__inputs.py`, with a real pandas `Series` where pandas is installed.
- **Added:** 2026-10-12. Root cause: the `Mapping` check failed for a `Series`, and the fallback branch iterated it, which yields the values. A perfect MS MARCO-shaped run scored MRR 0.1667 with no warning.

### Rule 103: Responses - read the status before the payload
- **When:** a response has a status (finish reason, HTTP status, stop reason) and a payload (text).
- **Then:** decide complete/incomplete from the status first; an abnormal end with an empty payload is still "incomplete" (with its token counts), never a type error about the payload.
- **Verified by:** `test_cut_off_without_text_is_incomplete_not_a_type_error` (six provider shapes), `test_callable_completion_is_checked_too`.
- **Added:** 2026-10-12. Root cause: the text check ran inside each adapter, before `complete` looked at the finish reason; a thinking model that spent its budget looked like a provider bug and its tokens vanished from `usage`.

### Rule 104: Untrusted structured output - store only the validated schema
- **When:** parsing model output (or any untrusted JSON) that is stored or exported.
- **Then:** validate the required fields with exact types, then keep only those fields; refuse what no export format can hold (non-finite numbers, lone surrogates, nesting too deep) as a parse failure of that one item.
- **Verified by:** `test_unstorable_verdicts_are_refused`, `test_only_the_schema_is_kept`, `test_deep_nesting_is_a_parse_failure`.
- **Added:** 2026-10-12. Root cause: extra fields were kept verbatim; one reply with `1e400`, a lone surrogate or `[null]` made a whole run's report unwritable in JSON or TOML.

### Rule 105: Caller-supplied resources - classify their failures apart from the remote's, and count from the source
- **When:** a caller passes a resource used on worker threads (a cache mapping), or a report summarises something an option can drop (verdicts).
- **Then:** wrap the resource's operations so their failure raises a distinct error that stops the run (a configuration problem), never a recorded remote failure; compute summary counters (errors, tokens) when the work happens, and do not count cached work as spent.
- **Verified by:** `test_a_failing_cache_is_a_cache_error`, `test_a_broken_cache_stops_the_run`, `test_errors_are_counted_without_verdicts`, `test_cached_responses_cost_no_tokens`.
- **Added:** 2026-10-12. Root cause: on Python 3.13 a `shelve` cache is SQLite-backed and thread-bound; with `n_jobs > 1` every judgement was recorded as a provider error. Error counts were derived from verdicts that `keep_verdicts=False` drops.

### Rule 106: Loaders - a record is checked against an explicit per-field contract before an object is built from it
- **When:** rebuilding objects from text (JSON, YAML, TOML) that may be malformed or hostile.
- **Then:** check every field's type (bool is not an int; a YAML/TOML date is not a string), refuse unknown keys, and check numeric ranges (a value that overflows its dtype) before calling the constructor; keep the contract table equal to the dataclass by a test. A permissive constructor (`from_dict` coercing `"7"` to 7) is not a validator.
- **Verified by:** `test_off_contract_records_are_refused`, `test_yaml_and_toml_dates_are_refused`, `test_record_contract_lists_every_field`.
- **Added:** 2026-10-12. Root cause: 2c's first loader trusted `from_dict`; `chunk_index: "7"` loaded as 7, `tokens: "abc"` as three tokens, a list `text` crashed with `AttributeError`.

### Rule 107: Exporters - refuse at write time what the loader cannot read back
- **When:** an exporter and a loader share a format.
- **Then:** every value the exporter accepts must load back equal (or with a documented, tested change such as JSON metadata `NaN` -> `None`); anything else is refused at export, naming the document and field. Test with a round-trip fuzz over every dtype and value class, not one happy path.
- **Verified by:** `TestRoundTrip` (four formats), `test_embeddings_that_cannot_round_trip_are_refused`, `test_json_refuses_nan_outside_metadata`, `probes/r4c/p_roundtrip.py`.
- **Added:** 2026-10-12. Root cause: an int8 embedding exported with `embedding_dtype: int8` that the loader refused; a 2-D embedding was dropped with a log line; a `NaN` bbox exported to JSON and could not be loaded.

### Rule 108: Identities - a reader of persisted data refuses another generation by its own error, never by the "malformed row" path
- **When:** an identity or payload format changes generation (widths, schema major), and data written before may be read: stores, binary exports, checkpoints, caches.
- **Then:** list every persisted reader (grep `from_dict`, `json.loads`, `pickle.load`, checkpoint loaders); give each a recorded generation to check; raise a dedicated subclass (`IncompatibleSchemaError(ValueError)`) and re-raise it in every loop that skips malformed rows; a format without a version field gets one (a binary export becomes a mapping).
- **Verified by:** `TestOtherGenerations` in `corpus/tests/test__identity.py` and `TestGeneration` in `_export/tests/test__export_security.py` build the old data and expect the error.
- **Added:** 2026-10-13. Root cause: 2d bumped the document schema to 3.0; `from_dict` refused 2.0 payloads, but stores caught every exception per row, so an old store read as empty; pickles and checkpoints recorded no generation and loaded old ids silently.

### Rule 110: Options - a new option is validated on assignment and reaches every producer, wrappers and plans included
- **When:** an option is added to one constructor of a family (builder config, pipeline, hookable pipeline, convenience function, fluent plan).
- **Then:** grep every producer of the same output and add it there (Rule 82); validate in `__setattr__` or a property, so a later assignment fails at once; put it in the identity scope it changes (here the plan's ingest fingerprint).
- **Verified by:** `test_every_producer_takes_the_scheme`, `test_assignment_after_construction_is_validated`.
- **Added:** 2026-10-13. Root cause: `doc_id_scheme` reached `BuilderConfig` and `CorpusPipeline` only; three wrappers rejected or ignored it, and a value set after construction failed after a whole ingestion.

### Rule 111: Identities - text is hashed with `surrogatepass`, never `replace`
- **When:** a `str` is encoded to bytes for a digest that names it (doc id, content hash, trace hash).
- **Then:** `encode("utf-8", errors="surrogatepass")`: valid text gives the same bytes as strict UTF-8, and lone surrogates (an undecodable file name, a broken source) stay distinct instead of collapsing to `?` or raising.
- **Verified by:** `TestTextIdentity` in `corpus/tests/test__identity.py`.
- **Added:** 2026-10-13. Root cause: `errors="replace"` gave `"key \ud800"`, `"key \udfff"` and `"key ?"` one `doc_id`; the strict encode in `make_doc_id` raised on a surrogate-escaped file name.

### Rule 112: Contracts - a shared value object is hashable, picklable and copyable, and a contract runs inside each caller's resilience boundary
- **When:** a frozen dataclass is shipped as a contract (a policy, a config) and called from code that isolates failures per leg or per item.
- **Then:** store containers as tuples (not `MappingProxyType` or `dict`), test `hash`, `pickle`, `deepcopy`, `dataclasses.replace` and `asdict`; normalise or validate each leg's inputs inside the leg's own `try`, so one bad leg cannot fail the whole call.
- **Verified by:** `test_a_policy_is_hashable_picklable_and_copyable`, `test_an_int_doc_id_leg_still_fuses`.
- **Added:** 2026-10-13. Root cause: 2e's `FusionPolicy` held weights in a `MappingProxyType` (unpicklable, unhashable), and the MCP retriever passed identities to the strict contract outside its per-leg `try`, so one leg with an `int` id failed every search.

### Rule 114: Scope - a scope applies at every read seam and to every statistic that ranks
- **When:** a filter, tenant or permission scope is bound to an index, store or service.
- **Then:** list every public read path (`search`, `query`, `get`, iterators, custom scorers, factories that rebuild the index) and route each through the scope; compute ranking statistics (IDF, document counts, average length) over the scope only; freeze the scope's value to plain data when it is bound; report it by label, not by expression; when an approximate backend returns fewer candidates than asked, fall back to exact ranking over the admitted rows.
- **Verified by:** `TestReviewFindings` in `corpus/tests/test__filter_search.py` (`test_the_vector_seam_honours_the_scope`, `test_out_of_scope_documents_do_not_change_in_scope_scores`, `test_a_short_approximate_backend_falls_back_to_exact_ranking`, `test_the_scope_is_frozen_when_bound`, `test_the_builder_binds_its_scope_to_every_index`).
- **Added:** 2026-10-13. Root cause: 2f applied `SecurityFilter` inside `search()` only; `RetrievalIndex.query()` (the seam the MCP layer uses) bypassed it, BM25 IDF over all rows let hidden documents reorder visible ones, and Annoy with a small `search_k` returned short filtered results reported as success.

### Rule 115: Fallbacks - a refusal has its own error type, and "broken" needs evidence
- **When:** a component can fall back (`allow_fallback`) and a caller wraps it in an error handler meant for something else (bad data, optional absence), or a probe classifies a component as broken versus absent.
- **Then:** raise a dedicated error for a refused fallback (`FallbackRefusedError(RuntimeError)`), never the type a caller's data handler catches; validate data first and alone, and let every other error from the component propagate; report "broken" only when the component is installed *and* an attempted load failed, otherwise "absent".
- **Verified by:** `TestRetrievalIndex.test_a_refused_annoy_option_fails_the_build`, `test_a_backend_error_propagates_but_bad_data_disables`, `TestProbes.test_an_absent_custom_backend_is_not_broken` in `corpus/tests/test__fallback_vocabulary.py`.
- **Added:** 2026-10-13. Root cause: 2g's strict Annoy refusal was a `ValueError`; `RetrievalIndex` caught `ValueError` as "invalid embeddings" and silently disabled dense search, so strict mode was quieter than lenient mode. A probe that called every installed-but-unavailable backend "broken" mislabelled custom backends that had simply reported themselves absent.

### Rule 116: Name tables - one shared instance is the default everywhere, and built-ins never displace a user's entry
- **When:** several entry points (a builder, a plan runtime, a CLI) resolve names to components, and a registry is introduced or extended to unify them.
- **Then:** every resolver's default is the same shared registry instance (never a fresh one built per call); built-in registration inserts only absent names, so a registration made first wins; test a user registration made before the first use through every entry point.
- **Verified by:** `TestReviewFindings.test_builtins_never_replace_an_earlier_registration`, `test_a_user_chunker_registered_before_the_first_build_is_kept`, `test_plans_and_the_builder_share_one_table` in `corpus/tests/test__chunker_names_and_configs.py`.
- **Added:** 2026-10-13. Root cause: 3a routed the builder through the shared registry while `RuntimeCorpus` built a fresh one per call, so "one name table" was two; and `register_builtins` overwrote entries, so the builder's first call replaced a user's `word` chunker with the built-in.

### Rule 127: Contracts - a limit is accepted only where the code can enforce it
- **When:** an API takes a limit (a budget, timeout, size or count) for a component it calls.
- **Then:** list where the limit is checked; accept it only for the parts that check point can stop (a check before each call bounds a stage of several calls, never a stage of one call) and refuse it elsewhere at construction, naming what does bound that part (a deadline before the stage, the component's own timeout). Report an overrun the check could not stop.
- **Verified by:** `test_a_single_call_stage_refuses_a_budget` and `test_an_overrun_is_reported` in `scikitplot/corpus/tests/test__execution.py`.
- **Added:** 2026-10-17. Root cause: `RetrievalExecutionPlan` took a `StageBudget` for every stage, but checked it before each call, with zero calls spent and zero seconds elapsed; for `transform`, `rerank` and `compress` (one call each) it could never stop anything, so `budgets={"rerank": StageBudget(max_seconds=2.0)}`, the guide's own example, promised a limit that did not exist (4c review, finding 1). Recurred in 5b in a second form: `CorrectiveRetrieval`'s deadline was checked before each round but not before the rewrite between rounds, a call of its own (a model rewrite is paid); then, moved before the rewrite, it no longer guarded the round after a slow rewrite. A limit on calls is checked before *every* call it bounds, the component's helpers included (`test_no_rewrite_is_asked_for_past_the_deadline`, `test_no_round_starts_past_the_deadline_after_a_slow_rewrite`).

### Rule 129: Contracts - a value that authorises an action is bound to the history it was given in
- **When:** a digest, token or approval authorises one action (a human approval, an idempotency key, a capability).
- **Then:** bind it to everything that makes the action this one: the run, the step, the exact canonical arguments, the effect and scopes as registered, and the history before it (the audit head, which starts from a random nonce); refuse an id that is already in use; refuse to resume a copy that the durable record (store or trail) shows is stale; recompute the binding at use, never trust a stored copy of it.
- **Verified by:** `TestApprovalIsUsedOnce` and `TestReReview` in `scikitplot/agent/tests/test_agent_hardening.py` (the same snapshot resumed twice, a run id reused, an equal goal and clock, a forged idempotent flag).
- **Added:** 2026-10-17. Root cause: the first approval digest covered the run id, step and action only; resuming one paused snapshot twice ran the approved write twice, and a new run reusing the id accepted the old run's approval (5a review, finding 1; re-review 1a, 1b).
