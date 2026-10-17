<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 33 / Run 4-A (pulled forward) - corpus evaluation: metrics, sklearn scorers, LLM judges, exportable reports

### Context
- User request (round 33, second message): move the corpus scorer out of
  `levenshtein/_core.py` (L540, `make_corpus_scorer`) into corpus; add
  `corpus/_metrics/_scorer.py` inheriting scikit-learn's `_BaseScorer` with a
  vendored fallback; add `_ranking.py` (Recall@K, MRR, ...) and LLM-judged
  metrics (faithfulness, answer relevance) for the user's model, GPT, Claude
  and Gemini; JSON/JSONL plus optional TOML/YAML export. No backward
  compatibility; deliver incrementally.
- Review mapping: `01_verdict_and_run_plan.md` Run 4 / R4-1 and
  `31_rag_agent_roadmap.md` section 4 (formulas fixed there). Pulled before
  Run 2 on purpose: Run 2 changes fusion and filters, and the harness is what
  measures that change ("each new strategy reports a delta").
- Facts established before design (probes in `/home/claude/r33/probes/r4`):
  - `levenshtein.make_corpus_scorer` is a *relevance* function
    (`(query, docs, config) -> hits`) that imports corpus lazily: an upward
    dependency from a low-level algorithm to the pipeline.
  - scikit-learn's own `make_scorer` cannot score a corpus index: on 1.9.1,
    `_Scorer._score` calls `is_regressor(estimator)`, which raises
    `AttributeError` for any object without `__sklearn_tags__`.
  - A `_BaseScorer` subclass that calls `super().__init__` with the arguments
    the installed base accepts works on 1.3.2, 1.4.2, 1.5.2, 1.6.1, 1.7.2,
    1.8.0 and 1.9.1: direct call, `sample_weight`, `cross_validate` and
    `GridSearchCV` with dict scoring, `check_scoring`, and metadata routing
    (1.4+; 1.3's `cross_validate` has no `params`). The shared predict cache
    applies from 1.4 (2 predict calls instead of 4 in the probe).
  - `scikit-plots-corpus` does not depend on scikit-learn, and importing
    corpus loads neither sklearn nor scipy today: the scorer must not import
    sklearn at module import (import-hygiene gate).
  - `docs/source/apis_reference.py` must equal its `.in` template byte for
    byte; edit both.

### Design (big picture)
- Two meanings of "scorer" are separated by name:
  - *relevance* (query x document -> score, to rank): `make_pairwise_ranker`
    (any pairwise function, sklearn's `greater_is_better` sign convention)
    and `make_levenshtein_ranker` (the moved adapter, same semantics and
    per-hit backend provenance), both in `corpus/_similarity/_pairwise.py`;
  - *evaluation* (ranking x judgments -> quality, sklearn's meaning):
    `make_corpus_scorer` in `corpus/_metrics/_scorer.py`.
- Inputs: qrels (`{doc_id: grade}` or a collection of relevant ids) and
  rankings (id sequences, best first), aligned per query. Ids are `str` or
  `int`, one type per evaluation. A `str` is never a collection of ids.
- Invariants: every metric is in [0, 1]; one per-query formula per metric,
  called by every path (functions, multi-metric pass, scorer, harness);
  undefined values follow `zero_division` exactly as scikit-learn does
  (`"warn"` -> 0 with a warning, `0.0`, `1.0`, `np.nan` -> excluded from
  the mean); duplicates in a ranking raise unless `duplicates="first"`.
- Failure modes: malformed input raises with the offending query; a degraded
  retrieval (a requested leg did not run) raises in the harness unless
  `on_degraded="warn"`; a judge's unparsable output is a recorded failure,
  never a coerced score.

### Increments
- [x] 2a. Deterministic evaluation (delivered 2026-10-12; results below):
  - [x] `corpus/_metrics/_inputs.py` (validation and normalisation)
  - [x] `corpus/_metrics/_ranking.py` (precision, recall, hit rate, MRR, MAP,
        R-precision, nDCG; `ranking_metrics` single pass)
  - [x] `corpus/_metrics/_text.py` (exact match, token F1, edit similarity,
        ANLS; pure-Python edit distance held equal to `scikitplot.levenshtein`)
  - [x] `corpus/_metrics/_scorer.py` + `_sklearn_scorer.py` +
        `_sklearn_compat.py` (vendored BSD-3 base, used without sklearn or
        when the installed base fails the contract probe)
  - [x] `corpus/_metrics/_evaluate.py` (harness, report, paired comparison)
  - [x] `corpus/_serialize.py` (strict JSON, JSONL, optional YAML/TOML)
  - [x] `corpus/_similarity/_pairwise.py`; remove
        `levenshtein.make_corpus_scorer` and every reference to it
  - [x] import-hygiene gate watches sklearn, yaml, tomli_w and the LLM SDKs
  - [x] docs (API reference + template, user guide), fragments
  - [x] R2-4 pulled in: `levenshtein` `_ExecutionPlan` (backend once per
        call; runtime fallback internal -> rapidfuzz -> python), LV-001/LV-002
        closed in `maintenances/levenshtein`
  - [x] `_atomic.py` staging permissions (review #6); `corpus/conftest.py`
        registers `thread_unsafe` (found by the CI harness)
- [x] 2b. LLM-judged metrics (delivered 2026-10-12; plan and results below): judge protocol, provider adapters (callable,
      OpenAI, Anthropic, Gemini; SDKs lazy, clients injectable), strict
      verdict parsing, faithfulness, answer relevance, context precision and
      recall, answer correctness, pairwise preference with order swap, judge
      panels and inter-judge agreement.
- [x] 2c. Document export (delivered 2026-10-12; plan and results below):
      strict JSON (no `NaN` tokens), consistent JSONL, optional TOML/YAML for
      `export_documents` / `load_documents`.

### Acceptance criteria (2a)
- Each metric matches a hand-computed table (binary and graded) and a
  brute-force reference over random inputs.
- The scorer passes the probe matrix (sklearn 1.3.2 ... 1.9.1) and the
  no-sklearn environment, and pickles.
- `import scikitplot.corpus` still loads no sklearn (gate extended).
- Independence: levenshtein no longer imports corpus; corpus imports
  levenshtein only inside a function.
- Report round-trips JSON, YAML and TOML; JSON output has no `NaN` token.

### Risks and mitigations
- Private sklearn API drifts -> contract probe at first use, vendored base,
  provenance in `scorer_backend_info()`.
- Two edit-distance paths drift -> randomised differential test (Rule 83).


### Results review - 2a (2026-10-12)
- Files: 60 since the Run 1 drop-in (24 added, 36 modified, 0 removed or
  renamed). Delivered as `scikitplot_r33_inc2a_dropin.zip`; cumulative with
  Run 1 as `scikitplot_r33_cumulative_dropin.zip`.
- Acceptance criteria:
  - [x] Hand-computed tables (binary and graded) and a brute-force reference
        over random inputs (`test__ranking.py`); the independent review's
        4,000 Hypothesis cases against a reference written from the docstring
        agreed.
  - [x] Probe matrix: scikit-learn 1.3.2, 1.4.2, 1.5.2, 1.6.1, 1.7.2, 1.8.0,
        1.9.1 and no scikit-learn; pickle and deepcopy (`after4/matrix.txt`).
  - [x] `import scikitplot.corpus` loads no sklearn, scipy, yaml, tomli_w or
        LLM SDK (`test__import_hygiene.py`, `EVALUATION_LIBRARIES`).
  - [x] levenshtein never imports corpus (`test_levenshtein_never_imports_corpus`,
        contract checker `_corpus_boundary_errors`); corpus imports
        levenshtein only inside `make_levenshtein_ranker`.
  - [x] Reports round-trip JSON, JSON Lines, YAML and TOML; JSON has no
        `NaN`/`Infinity` token (`test__serialize.py`, `test__evaluate.py`).
- Evidence:
  - corpus suite 4083 passed, 16 failed (ids identical to Run 1: 14 DNS,
    2 NLTK data), 12 skipped; small group 684 passed.
  - CI harness on a snapshot (`verify-tree3`), skinny + corpus, Python 3.9
    and 3.13, normal and lowest: 174 passed, 4 failed; the 4 are the corpus
    legs, 4029 passed and 14 failed each, the 14 ids identical to Run 1.
  - Negative proofs: 48 new tests fail on the reviewed version.
  - Gates: towncrier PASS (104); test module names unique; path lengths 0
    failing; `libs._tools check` up to date; ruff 0 in new modules, no new
    finding in modified files; Rule 52 scan 0; 3.8/3.9 grammar parse.
- Deviations from plan:
  - `on_degraded`: the harness takes `raise` (default), `warn` or `ignore`;
    the scorer takes `raise` or `ignore` (a scorer returns one number, so a
    warning has nowhere to record the excluded queries).
  - R2-4 and the `_atomic.py` permission change were not in the 2a plan; both
    were the root of review findings #5 and #6.
  - Independent review: 21 findings, all fixed (dispositions in the review
    bundle, `05_increment_2a_results.md`).
- Correction (user, 2026-10-12): this round's PR is 865, not 864. The nine
  fragments of Run 1 and 2a were renamed to `865.<type>.rst` before any
  local apply; round 32's `864.*` fragments are unchanged (Rule 100).
- Technical debt: none new. F-52 (corpus does not import on Python 3.8 while
  the root distribution declares `>=3.8`) predates this round; F-50
  (`capability_snapshot()` imports annoy, which imports scikit-learn and
  SciPy) is assigned to Run 2 (R2-5).

### Plan - 2b: LLM-judged answer evaluation (2026-10-12, PR 865)

#### Context
- The user applied the cumulative drop-in (Run 1 + 2a, PR 865) and asked to
  continue. 2b is the second part of R4-1 ("Faithfulness → Answer relevance",
  judging "my model or GPT or Claude or Gemini").
- Grounding: the reference text lists "Faithfulness • Answer Relevancy •
  Context Recall/Precision • latency • token usage • tracing" and warns that
  option order changes a model's decision ("option-order and label bias").
  It gives no formulas; the formulas below are this design's definitions and
  are stated in the docstrings. `31_rag_agent_roadmap.md` §4: judges are
  optional hooks with recorded judge identity, behind the deterministic metrics.
- SDK surfaces checked by introspection (openai 3.29.0, anthropic 1.13.0,
  google-genai 2.29.0): `chat.completions.create(model, messages, ...)` with
  `choices[0].message.content` and `usage.prompt_tokens/completion_tokens`;
  `messages.create(max_tokens, messages, model, system, ...)` (max_tokens
  required; no `temperature` parameter in this version) with `content[].text`
  and `usage.input_tokens/output_tokens`; `models.generate_content(model,
  contents, config)` with `GenerateContentConfig.system_instruction`, `.text`
  and `usage_metadata.prompt_token_count/candidates_token_count`.
  Consequence: no sampling parameter is sent unless the caller passes it.

#### Design
- `_metrics/_llm.py`: `make_llm(provider, model, *, client, max_tokens,
  options, cache)` -> `LLM` with `complete(prompt, *, system) -> Completion`
  (text, provider, model, input/output tokens, provider usage, finish reason,
  latency, cached). Providers `openai`, `anthropic`, `gemini`, or any callable
  `fn(prompt, system) -> str`. SDKs imported at construction, never at
  package import; `client=` injects any client with the same surface (an
  OpenAI-compatible server, a test fake). Optional `cache` mapping keyed by
  sha256 of the canonical request. `identity()` never holds a secret.
- `_metrics/_judge.py`: `LLMJudge(llm, *, on_error)` with versioned
  `string.Template` prompts and strict verdict parsing (one JSON object,
  optionally in one ```json fence; required keys with exact JSON types;
  booleans are `true`/`false`; ratings are integers in range). Methods return
  `JudgeResult(metric, score, verdict, completions, error)`:
  - faithfulness = supported claims / claims of the answer (no claims: NaN);
  - answer_relevance = (rating - 1) / 4 on a 1-5 rating;
  - context_precision = average precision of the per-context relevance
    verdicts in rank order (no relevant context: 0; no context: NaN);
  - context_recall = reference statements attributable to the contexts /
    statements (needs a reference);
  - answer_correctness = TP / (TP + (FP + FN) / 2) over statements;
  - criteria_score = (rating - 1) / (scale - 1) for a caller's rubric;
  - preference(question, a, b): both orders; a consistent winner wins,
    disagreement between orders is a tie flagged position-inconsistent.
- `_metrics/_agreement.py` (pure numerics): `cohen_kappa` (none, linear,
  quadratic weights), `fleiss_kappa`, `krippendorff_alpha` (nominal, ordinal,
  interval, ratio; missing values).
- `_metrics/_answers.py`: `evaluate_answers(system, dataset, *, judges,
  metrics, text_metrics, n_jobs, on_error, metadata)` -> `AnswerEvaluation`
  (per item and mean per metric, per judge and panel mean; agreement per
  metric; tokens and latency; JSON/JSONL/YAML/TOML; `compare`), and
  `evaluate_preference(dataset, answers_a, answers_b, *, judges)` ->
  `PreferenceEvaluation` (win rate, ties, position consistency).
- `_evaluate.py`: the paired comparison becomes one function used by both
  report classes (no copy).

#### Invariants and failure modes
- Every score is in [0, 1] or NaN; NaN means undefined or failed, and the
  item records which (`error`).
- A judge's unparsable or out-of-schema output is a recorded failure under
  `on_error="record"`, a `JudgeParseError` under `"raise"`; never a coerced
  score. Provider exceptions follow the same policy.
- A truncated completion (OpenAI `finish_reason="length"`, Anthropic
  `stop_reason="max_tokens"`) is a failure, not a partial verdict.
- Results are in dataset order whatever `n_jobs` is.
- Untrusted answers and contexts are delimited in the prompt and the judge
  is told to treat them as data (prompt-injection awareness, reference text);
  this lowers, not removes, the risk, and the docs say so.
- `import scikitplot.corpus` loads no SDK (import-hygiene gate).

#### Steps
- [x] `_llm.py` + tests with fake clients for each provider shape, and the
      real SDKs over HTTP against a local server (old and new SDKs)
- [x] `_judge.py` + tests (parsing matrix, each metric's arithmetic)
- [x] `_agreement.py` + parity tests (scikit-learn, statsmodels,
      krippendorff where installed) and textbook values
- [x] `_answers.py` + comparison refactor + tests
- [x] exports, API reference (+ template), guide `llm_evaluation.rst`,
      fragment `865.feature.rst` (one bullet, appended to the corpus one)
- [x] gates, version matrix, CI harness on a snapshot, independent review,
      negative proofs, review bundle 06, lessons, delivery (2b delta only)

#### Acceptance criteria
- Each provider adapter sends exactly the documented arguments (fake client
  records them) and reads text, tokens and finish reason.
- Each metric's arithmetic matches a hand-computed table; every malformed
  verdict in the parsing matrix is refused with a reason.
- Agreement statistics equal the reference implementations on random data.
- An `AnswerEvaluation` round-trips JSON, YAML and TOML; `compare` is the
  shared function.
- No network call in any test.

### Results review - 2b (2026-10-12)
- Files: 25 since the applied cumulative drop-in (9 added, 16 modified, none
  removed or renamed); see the review bundle, `06_increment_2b_results.md`.
- Acceptance criteria:
  - [x] Each adapter sends exactly the documented arguments: fake clients
        record them, and the real SDKs are driven over HTTP against a local
        server (openai 1.40.0 / anthropic 0.34.0 / google-genai 1.0.0 and
        openai 3.29.0 / anthropic 1.13.0 / google-genai 2.29.0).
  - [x] Metric arithmetic against hand tables; the parsing matrix refuses
        every malformed verdict with a reason.
  - [x] Agreement equals scikit-learn, statsmodels and krippendorff on 3,000
        random trials; Krippendorff's published 0.743/0.815/0.849/0.797.
  - [x] `AnswerEvaluation` and `PreferenceEvaluation` round-trip JSON, YAML
        and TOML; `compare` is the shared `paired_comparison`.
  - [x] No network call in any test (a 127.0.0.1 server only).
- Evidence:
  - `_metrics` tests: b312 516 passed, 9 skipped (optional parity and SDK
    tests); llm venv (3.13, SDKs, references) 515 passed; llmold (3.11, old
    SDKs) 497 passed, 18 skipped; scikit-learn 1.3.2-1.8.0 and none on
    Python 3.9/3.11: 491-506 passed (`after2b/matrix.txt`).
  - corpus suite 4313 passed, the same 16 failed (14 DNS, 2 NLTK), 21
    skipped; top-level package, guide and levenshtein lanes green.
  - CI harness (skinny + corpus, 3.9 and 3.13, normal and lowest) on a
    snapshot: 174 passed, 4 failed; the 4 are the corpus legs with the same
    14 DNS ids as Run 1 (4255 passed each); no marker warning.
  - Negative proofs: 26 review-fix tests fail on the reviewed version (every
    finding covered), plus 3 assertions on renamed usage fields and the new
    `errors` field.
  - Gates: towncrier PASS; test names unique; path lengths 0 failing;
    `libs._tools check` up to date; ruff 0 in the product modules; no
    invisible characters; every changed `.py` parses under 3.9.
- Independent review: 10 findings (3 medium, 7 low), all fixed.
- Found while verifying, and fixed: the 2a delivery shipped
  `maintenances/levenshtein` with a stale fingerprint (recorded before the
  LV-001/LV-002 review fix changed `_core.py`), so
  `test_structural_contract_is_current` failed; refreshed and recorded.
- Deviations from plan:
  - OpenAI `max_tokens` is `max_completion_tokens`, which openai < 1.45 does
    not accept: arguments are now checked against the client's signature at
    construction (the old-SDK run found it).
  - `usage` counts responses, cached responses and uncached tokens;
    `errors` is a stored field (review #4, #8).
- Pre-existing, recorded, not fixed: the corpus maintenance plane fails on
  the upload (F-53).

### Plan and results - 2c: document export and load-back (2026-10-12, PR 865)

#### Context and findings (probe `probes/r4c/p_export_baseline.py`, on the applied 2b tree)
- F-55: JSON and JSON Lines exports wrote the non-standard token `NaN`.
- F-56: a NumPy scalar in metadata made JSON/JSONL export raise `TypeError`.
- F-57: `load_documents` returned `[]` (log warning) for JSON, JSON Lines,
  CSV and every format but pickle/joblib, while its own security message
  recommended JSON as the safe alternative.
- F-58: `ExportFormat` docstrings: JSON described as JSON Lines, NUMPY as
  `savez_compressed` (it is `np.save`), PANDAS/POLARS as in-memory (they
  write files).

#### Design
- `ExportFormat.YAML`, `ExportFormat.TOML`; JSON/JSONL/YAML/TOML written by
  one `_export_text` through `_serialize` (the report contract); TOML keeps
  records under `documents`.
- `_export/_records.py`: the record contract (`FIELD_KINDS`, held equal to the
  dataclass annotations by a test); `export_problem` names the document and
  field a format cannot hold; `record_problem` refuses off-contract records on
  load; embeddings carry `embedding_dtype` (float16/32/64, 1-D only).
- `load_documents`: the four text formats without `trusted`; suffix
  inference; lossy formats refused with the reason; string format values.

#### Results
- [x] Round trips: 200-trial fuzz over all four formats (awkward strings,
      NumPy dtypes, NaN embeddings): 0 mismatches; new tests 109.
- [x] Independent review: 9 findings (2 medium, 7 low), all fixed; 41 tests
      fail on the reviewed version, 106 of 109 on the pre-2c tree (the 3 that
      pass: an empty corpus loaded as `[]` coincidentally, JSONL line safety).
- [x] corpus suite 4422 passed, the same 16 failed, 21 skipped.
- [x] CI harness on an untouched snapshot: 174 passed, 4 failed (the corpus
      legs, 4333 passed and the same 14 DNS ids each). A first run failed two
      wheel-equality checks because a test file was copied into the snapshot
      mid-run (Rule 87 repeat); rerun on a fresh snapshot.
- [x] Version matrix: Python 3.9 (scikit-learn 1.3.2-1.6.1, none) export
      145 passed 31 skipped; 3.11 176; 3.13 176; old-SDK venv 159 passed.
- [x] Gates: towncrier PASS; names unique; path lengths 0 failing;
      `libs._tools check` up to date; ruff clean in product modules;
      invisible-character scan widened to Zl/Zp/Cc: 0; 3.9 grammar.
- Deviation: the record contract module was not in the plan; it is the root
  fix of review #2/#3 (a loose `from_dict` is not a validator).
- Pre-existing, not changed: `_schema.py` doctests `make_doc_id` (expects
  `'...'` without ELLIPSIS) and two polars examples fail when run (F-38 class).
