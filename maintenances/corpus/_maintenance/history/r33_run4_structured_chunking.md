<!-- Archived verbatim from maintenances/corpus/todo/todo.md at d54c683+ (PR 934),
     when the file passed its 400-line budget. Do not edit. -->

## Task: round 33 / Run 4 - structured chunking measured by the harness (R4-2, R4-3)

### Context
- Goal: chunking strategies beyond fixed units, each producing the span
  model (R3-3) and a measured delta in the evaluation harness (R4-1), none a
  default until it wins. R4-1 is complete (2a-2c).
- Plan items (`01_verdict_and_run_plan.md` Run 4): R4-2 ParentChild (no
  producer sets `parent_doc_id`; five consumers), Recursive (separator
  hierarchy), StructureAware (headings, code blocks); R4-3
  `RetrievalExecutionPlan`.
- Base: 3e (`faa7665`). PR 865 fragments.

### Increments
- [x] 4a. Recursive and parent-child chunking (R4-2, part 1)
  - Design:
    - `RecursiveChunker(RecursiveChunkerConfig(chunk_size, chunk_overlap,
      separators=("\n\n", "\n", ". ", " ", ""), strip_whitespace,
      min_length))`: spans first; a span longer than `chunk_size` is cut
      after each occurrence of the first separator it contains (the
      separator stays with the left piece, so pieces partition the span),
      adjacent pieces are merged greedily up to `chunk_size`, and a piece
      still too long recurses with the next separator; `""` cuts characters.
      Every chunk is `source[context_span]`; `chunk_overlap` extends the
      context back to a word start within that many characters, never past
      the previous chunk's start. Strategy `RECURSIVE` (exists).
    - `ParentChildChunker(ParentChildChunkerConfig(parent_size, child_size,
      child_overlap, separators, emit_parents=True))`: parents by the
      recursive rule, children by the same rule inside each parent; emits
      each parent, then its children; a child's metadata names its parent
      by position (`parent_chunk`) and every chunk its `chunk_role`.
    - `DocumentReader.get_documents`: a chunk whose metadata has
      `parent_chunk` gets `parent_doc_id` = that chunk's document id
      (refused if it names no earlier chunk of the same result);
      `chunk_role` goes to document metadata. This is the first producer of
      `parent_doc_id`.
    - Both registered (`recursive`, `parent_child`) with their configs; the
      span invariant test covers them.
  - Gate: a deterministic benchmark on a fixed in-repo corpus (the corpus
    user guide), `r34/run4a/bench.py`: queries are every 4th sentence;
    relevant = a chunk whose context covers the sentence's span; BM25
    (`RetrievalIndex`, keyword) ranks chunks; Recall@1/5, MRR, mean context
    length, and context coverage of the sentence's paragraph, per chunker
    (sentence, paragraph, fixed_window, recursive, parent_child children
    with parent context). Reported, not used to change any default.
  - [x] `_chunkers/_recursive.py` (both chunkers, `chunk_batch`), registry,
        exports, reader linkage (`parent_chunk` -> `parent_doc_id`,
        `chunk_role` metadata)
  - [x] tests (`_chunkers/tests/test__recursive.py`: config, a partition
        fuzz over 6 sizes x 300 texts, overlap rules, parent-child structure,
        documents + `validate_hierarchy`, a refused `parent_chunk`); span
        invariant configs extended; builtin list test updated
  - [x] benchmark (`r34/run4a/bench.out`, `bench_table.txt`; 82 queries):
        R@1 sentence 0.96, paragraph 1.00, fixed_window(500/400) 0.95,
        recursive(500) 0.95, recursive(200) 0.59, parent_child(1500/200)
        1.00; paragraph coverage of the top context 0.49 / 1.00 / 0.76 /
        0.84 / 0.46 / 0.98; mean top-context chars 384 / 504 / 485 / 422 /
        150 / 1263. Verbatim-sentence queries: this measures whether a
        sentence and its surroundings stay together, not semantics
  - [x] guide section (customization.rst, with the table; sync-tested), API
        reference, corpus `865.feature`
  - [x] suites: baseline 15; gates clean
  - [x] independent review (`r34/review4a/`): 6 findings + minor items;
        re-review of the fixes (`r34/review4a2/`): 3 findings; all fixed, see
        "Results review - 4a"
  - [x] negative proofs (`r34/neg4a/`): final tests on 3e: the module does
        not import; on the reviewed version 5 fail (the two scale tests
        excluded: the quadratic version does not finish them); on the first
        fix 1 fails, and its memory test fails; final: all pass
  - [x] harness on `snap4a` (skinny + corpus + mcp, Python 3.9 and 3.13):
        203 passed, 4 failed (the corpus legs, 4891 passed, the same 14 DNS
        ids as 3b), 2 skipped (mcp on 3.9). Delivered as
        `scikitplot_r33_inc4a_dropin.zip` + `.diff` +
        `17_increment_4a_results.md`.

### Results review - 4a (2026-10-16)
- Review findings (each reproduced), dispositions:
  - 1 Med: whitespace runs made `recursive_spans` quadratic (40k spaces:
    38 s) -> per-piece stripped bounds, O(1) merge tests
  - 2 Med: text without separators held one tuple per character (5 MB:
    +667 MB) -> `_character_windows`; re-review: `_cut` still held one
    tuple per piece (5M spaces: +513 MB) -> a generator
  - 3 Low/Med: a filter dropping a parent left children naming a document
    that was not yielded -> only yielded documents are linked; the child
    gets `parent_omitted=True` and no `parent_doc_id` (re-review: nor the
    source-level parent); `parent_chunk=True` is refused consistently
  - 4 Low/Med: `emit_parents=False` lost the link at document level ->
    `parent_span` in document metadata
  - 5 Low/Med: the docs table changed with every edit of the page it is in
    -> the benchmark reads the guide pinned at a commit (`bench.py <repo>
    faa7665`), reproduced twice byte for byte; the text says the pages were
    measured before the section was added and why `recursive (200)` is low
  - 6 Low: `strip_whitespace=False` could emit whitespace-only chunks and
    contexts -> `min_length` on stripped length; contexts start on a
    non-space
  - Minor: list separators accepted (JSON/YAML); CRLF, merge order,
    duplicate single-child parents and the shared `RECURSIVE` strategy
    documented; Raises sections complete
- The rewrite of `recursive_spans` is byte-identical to the first version on
  a 4000-case differential (`r34/run4a/spans_old.json`), and the re-review's
  own differential (180,000 cases, Unicode whitespace, CRLF) found 0
  differences
- Suites: baseline 15; gates clean
- [x] 4b. Structure-aware chunking (R4-2, part 2)
  - Design: `StructureChunker(StructureChunkerConfig(markup="markdown" |
    "rst", chunk_size, include_heading, separators, min_length,
    front_matter))`; markup is given, never guessed. Code blocks are their
    own chunks (split at blank lines, then lines, keeping the first line's
    indentation); the prose of a section is one region split by the
    recursive rule; a heading directly followed by prose begins that
    prose's first context (`include_heading`), any other heading is a chunk
    of its own (`block="heading"`); every chunk and its document carry
    `heading_path` (tuple; a list in documents), `section_level`, `block`.
    Markdown: CommonMark subset (ATX with the closing-sequence rule, setext,
    fences, thematic breaks; lines end at LF/CRLF/CR) plus Pandoc front
    matter. reST: docutils 0.22 rules (title refusal by the Body-state
    patterns, literal blocks with docutils' indentation rules, quoted
    literals, tabs as 8 columns). Strategy `BLOCK`; registered as
    `structure`.
  - [x] `_chunkers/_structure.py`, registry, exports; `_base.py` copies
        `heading_path`, `section_level`, `block` into document metadata
  - [x] tests (`_chunkers/tests/test__structure.py`, 222 incl. 139 docutils
        differential cases), span invariant configs, builtin list
  - [x] benchmark row (`r34/run4b/bench.out`, pinned `faa7665`):
        structure(rst, 500): 301 units, R@1 0.95, paragraph coverage 0.84,
        top context 380 chars (recursive(500): 0.95 / 0.84 / 422); two runs
        byte-identical
  - [x] guide paragraph and table row, API reference, corpus `865.feature`
  - [x] independent review (7 findings), re-review of the fixes (6), second
        re-review (6); all fixed, each with a test
  - [x] negative proofs, harness, delivery

### Results Review - 4b (2026-10-16)
- Reviews:
  - Review 1: ATX regex quadratic (40k chars: 7.4 s) [High]; a directive
    body (`.. note::`) taken as a literal block; a literal block ended only
    at column 0; setext over list items, block quotes and indented code,
    front matter lost, and heading text dropped when no prose followed;
    structure metadata not in documents; mutable `heading_path`; doc
    claims; CommonMark corners (`### Title#`, U+2028).
  - Re-review 1: `::` at the end of directive content refused (regression);
    the literal indent of definition lists and numpydoc parameters;
    front matter after `---` + blank line; thematic breaks; option lines as
    titles; "no heading is lost" vs `min_length`.
  - Re-review 2: field and option bodies; an option without a description
    is a title (regression); explicit markup after a list marker; line
    block and doctest `::`; superlinear time on nested explicit markup
    (2 MB: 12.4 s, now 1.1 s, linear in bytes); a comment without a blank
    line.
  - Self-found: directive option lines (`:alt: Look::`); tabs; quoted
    literal blocks; docutils' own title-refusal patterns.
- Reference checks: 139 reST cases equal docutils 0.22.4 (the test);
  whole-docs comparison over 320 pages: 13 differ, all Sphinx-only
  directives docutils dumps as error text (`r34/run4b/oracle_docs.py`).
  Fuzz (reviewers' `p_fuzz.py`, 3 seeds, ~95k runs each, both markups,
  `include_heading`/`front_matter` on and off): 0 span violations, 0
  non-whitespace characters lost. Adversarial lines up to 200k characters
  linear for every regex.
- Negative proofs (`r34/neg4b/`, the final tests on each version):
  4a: the module does not import (2 errors); reviewed: 155 failed;
  first fixes: 87 failed; second fixes: 54 failed; final: 438 passed.
- Suites: 9222 passed + the baseline 15 (14 DNS, `test_public_api`).
- Gates against `r34/base4a` (rebuilt from `92d5e91`, Rule 123): towncrier
  exit 0; names unique; path lengths 0 failing; ruff and black nothing new;
  non-ASCII 0 (Rule 121); Python 3.9 grammar 0 failures.
- Harness on `snap4b` (untouched; differs from the commit only in
  `tasks/lessons.md`, the Rule 120 note added after the snapshot):
  203 passed, 4 failed, 2 skipped; the 4 failures are the corpus legs,
  each 4986 passed (277 skipped, incl. the 139 docutils cases: docutils
  is not installed there) and the same 14 DNS failures as 3b; mcp on
  3.13: 324 passed.
- Lessons: Rules 121, 122, 123; Rule 120 extended (a check in the
  launching command line matches the launcher).
- [x] 4R. Records by plane: `tasks/` split into `maintenances/`, `plugins/scikitplot-dev/skills/`
  and `maintenances/_docs/` (planned and tracked in [`_repo`](../../_repo/todo/todo.md))
- [x] 4c. `RetrievalExecutionPlan` (R4-3)
  - Goal: query transformation, rerank and compression as typed, bounded
    stages around the existing retrieval contract, each with an outcome and
    a budget; nothing in the plan exists today (roadmap section 2).
  - Design (`corpus/_execution.py`, no new index class):
    - stages, in fixed order: `transform` (query -> queries), `retrieve`
      (`index.search` per query, unchanged), `fuse` (across queries, by the
      one `FusionPolicy`, R2-1), `rerank` (scores for the first
      `rerank_depth` hits), `compress` (character spans of each hit's text,
      so a passage stays a citation, R3-3);
    - each stage reports a `StageOutcome`: status (`success`, `empty`,
      `degraded`, `failed`, `skipped`, `cancelled`; a non-success carries an
      `ErrorRecord`), component, items in and out, calls, seconds, budget;
    - budgets are deterministic (`StageBudget(max_calls, max_seconds)`, a
      plan deadline), checked before each call, never proposed by a
      component; an exhausted budget stops further calls and is reported.
      After review: only `retrieve` (one call per query) takes a
      `StageBudget`; the deadline stops any stage from starting; `fuse`
      always runs (Rule 127);
    - `on_error="degrade"` (default) keeps the previous stage's output and
      reports the failure; `"raise"` propagates;
    - `ExecutionResult` iterates its hits and carries a derived status, so
      `evaluate_retrieval(plan.bind(index), ...)` measures a plan as is.
  - Gate: an empty plan returns exactly what `index.search` returns; the
    harness reports the per-stage cost and the recall delta of a stage;
    nothing becomes a default.
  - [x] module, exports, tests (`corpus/tests/test__execution.py`, 87:
        stage order, each status, budgets with a controlled clock, on_error,
        validation of every component's output, equivalence with search in
        every match mode, the harness)
  - [x] found on the way: `LegOutcome.to_dict()` was not JSON (its
        `generation` is an `IndexGeneration`); fixed, with a test
  - [x] benchmark (`r34/run4c/bench.py`, pinned `faa7665`, deterministic):
        empty plan identical to search; on half-sentence queries
        (recursive 500) a longest-run reranker R@1 0.91 -> 0.94, a halves
        transform 0.91 -> 0.90; on whole sentences no stage changes recall
        (every miss is a sentence no chunk contains); 1500-char compression
        keeps the sentence for 0.94 against recall@10 0.95
  - [x] guide section and table (sync-tested), API reference, fragments
  - [x] review, negative proofs, harness, delivery (see "Results review -
        4c")

### Results review - 4c (2026-10-17)
- Review 1 (`r34/review4c/`), 12 findings, each fixed with a test:
  [High] budgets on single-call stages never enforced (refused now; the
  deadline cancels stages not started; retrieve reports
  `seconds_over_budget`); [Med] an index-cancelled response made the plan
  `failed`; a non-real `hit.score` crashed `fuse` under degrade; empty
  compress entries required `doc.text`; a policy weighing every query 0
  gave `empty` (now `degraded`, `FUSION_DROPPED_ALL`; `top_k=0` refused);
  the hit contract undocumented; [Low] `CharBudgetCompressor` cut only at
  space and newline (now any `isspace`); set and dict scores accepted;
  numpy ints refused in spans; `to_dict` of list responses; docstrings
  (`duplicates` counts proposals not run); guide framing (prefix queries
  make the longest-run reranker near an oracle).
- Re-review (`r34/review4c2/`): 10 fixed, 2 partly; one regression (an
  index's malformed error record crashed a degrading plan) and a string
  status (`"cancelled"`) read inconsistently; both fixed through one
  `_response_status`, an unknown status derives `degraded`; three
  docstrings and a message corrected. Probes rerun: all as expected.
- Negative proofs (`r34/neg4c/`, final tests on the reviewed version): 28
  failed, 92 passed; final (`r34/fin4cf/`): 120 passed.
- Benchmark rerun on the final code: output byte-identical to the guide's
  numbers (`r34/run4c/bench2.out`).
- Suites: 9483 passed + the baseline 15 (same ids as 3b).
- Gates against `r34/base4r` (from `d7a44d9`): towncrier: the fragments
  first held two bullets each, refused by the section generator; merged
  into one (Rule 126); ruff and black nothing new; non-ASCII 0; Python 3.9
  grammar 0 failures; codespell `NotIn` pre-existing.
- Harness on `snap4c` (untouched; identical to the commit before this
  line): 203 passed, 4 failed, 2 skipped; the 4 failures are the corpus
  legs, each 5107 passed and the same 14 DNS failures as 3b.
- Lessons: Rules 126, 127.
