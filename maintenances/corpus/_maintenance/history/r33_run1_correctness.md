<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: round 33 / AI-plane Run 1 - correctness floor (silent wrong answers, a crash, a leak)

### Context
- Ground truth: `scikit_plots_1.zip` (sha256 `ab749f2e...`), byte-identical to
  the tree the external v3 architecture review read. Only one of the five
  reference texts exists (`LLM, RAG, MCP, and AI Agents (here unorde.txt`).
- v3 is a review, not evidence: each of its P0/P1 claims was re-run on a full
  meson build (Python 3.12, `--no-build-isolation`) before being accepted.
- Baseline suites on that build (before any change): small group (levenshtein,
  editdistance, rank_bm25, memmap, random, cexternals/_annoy, _cli,
  independence) 578 passed; mcp 287 passed 1 skipped; annoy 793 passed 2
  skipped 1 xfailed. Every defect below passes the existing tests.

### Root causes (each reproduced by an executable probe)
1. `cexternals/_editdistance` threshold early-exit reads `d[i%1][j]` (always
   row 0): `eval_criterion('a','a',0) -> False`; 91/20000 random cases wrong,
   all exact-match false negatives. Existing tests never use threshold 0.
2. The same wrapper compares `hash(element)`, not elements:
   `eval([-1],[-2]) == 0`, `eval([0],[2**61-1]) == 0`. `levenshtein` auto
   resolves to this backend, so `levenshtein.distance([-1],[-2])` is 0 by
   default and 1 with `backend="python"`. RapidFuzz shares the flaw for
   larger values (`[0]` vs `[2**61-1]` -> 0).
3. The wrapper `malloc`s, then calls `hash()`; a raising `hash` leaks the
   buffer (measured: exactly `len(a)*8` bytes per call, 46,922 KiB over 3,000
   calls; also reached through `levenshtein.distance` on unhashable
   elements). `malloc` results are never checked for NULL.
4. `rank_bm25` sparse path (`get_top_ids`, "the binding to prefer") assumes a
   document lacking a query term scores 0. False for BM25L/BM25+ (delta
   lower bound): ranking differs from `get_scores` in 34/300 and 35/300
   random queries, scores in 188/300 and 201/300. The agreement test covers
   one Okapi single-term query.
5. `rank_bm25.get_batch_scores` never calls `_require_query_tokens`:
   `get_batch_scores("windy", ...)` scores one character at a time.
6. `EmbeddingEngine` cache key omits the custom provider: provider B received
   provider A's vector `[1,1]` and was never called.
7. `EmbeddingEngine` never builds an `EmbeddingManifest`; every embedded
   document is untagged, so `_reject_mixed_manifests` (the R05 guard) never
   fires: two providers' vectors were indexed and ranked together.
   `CorpusBuilder._embed_documents` re-implements embedding with `engine.embed`
   and so bypasses cache and provenance as well.
8. `EmbeddingEngine(normalize=True)` normalises only for sentence-transformers;
   a custom provider's vector came back with norm 4.24.
9. Cache load forces float32: a float64 engine's hit returns `1.0` where the
   miss returned `1 + 1e-12` (a hit is not "equivalent to recomputation").
10. `embed_documents(input_path=None)` disables a cache that is
    content-addressed; URL sources never cache; two docstrings still describe
    a path/mtime key.
11. `SemanticChunker` reports the requested backend after falling back:
    `backend="hybrid"`, `chunking_strategy="semantic_hybrid"`, a model name,
    with no model loaded; the fallback warning repeats on every `chunk()`.
12. `memmap.MemoryMap.close()` unmaps while a NumPy view exists: returning
    `as_numpy_array()` from a `with` block and reading it kills the process
    (exit 139, SIGSEGV).

### Implementation Steps
- [x] 1. `_editdistance`: `d[i&1][j]`; exact element coding (str via code
      points, everything else via one dict = container equality); RAII
      `vector` buffers; drop the wrong `ctypedef int int64_t`; tests
- [x] 2. `levenshtein`: canonicalise non-str/bytes inputs to exact integer
      codes before any backend; tests across all available backends
- [x] 3. `rank_bm25`: one formula per recipe shared by dense, batch and
      sparse paths; sparse adds every query term's zero-frequency
      contribution; `get_batch_scores` validates tokens; tests
- [x] 4. `EmbeddingEngine`: `custom_fn_id` in the key (no id: bypass disk
      cache, said once); uniform `normalize`; dtype-preserving cache load;
      cache independent of `input_path`; resolved manifest stamped on every
      embedded document; builder delegates to `embed_documents`; tests
- [x] 5. `SemanticChunker`: requested vs effective backend, fallback reason,
      model only when used, warn once, `allow_fallback=False` raises; tests
- [x] 6. `memmap`: buffer protocol with an export count; `close()` defers
      the unmap until the last view is released; tests (subprocess crash
      test, `/proc/self/maps` unmap check on Linux)
- [x] 7. Towncrier fragments, lessons, results
- [x] 8. (re-plan, probe-driven) `SemanticChunker`, same root cause on two
      more paths: each chunk's `multilang` metadata named the model
      (written at construction); a model failing at encode time was
      swallowed (even with `allow_fallback=False`); a model failing to load
      escaped once and was then reported as a missing package; a per-chunk
      `encode` ran once per chunk and its vectors were dropped. Fixed; 6 more
      tests
- [x] 9. (re-plan, probe-driven) `memmap`, same root cause on two more
      paths: `close()` from another thread during a GIL-free `read()`
      (SIGSEGV), and `mprotect()` removing a right a live view holds
      (SIGSEGV on the next write). GIL-free methods pin the region;
      `close()` takes the address under the GIL; `mprotect` refuses with
      `BufferError`; 3 more tests
- [x] 10. Independent review (Rule 92): a reviewer that had not seen the
      work probed the diff read-only and reported 12 findings.
      - Fixed, each with a test that fails on the reviewed version:
        - `mprotect` race (it now keeps the GIL);
        - `PROT_NONE` export and `read` (refused);
        - the chunker's load state machine (lock; outcome published last;
          interrupts retried);
        - the retained traceback;
        - a single-unit text naming a model;
        - `float16` normalisation;
        - `custom_fn_id` breaking positional construction;
        - `get_batch_scores` with a tuple.
      - Corrected documentation: the NumPy `frombuffer` claim (true only
        before 1.23), the declared-model precedence, and the free-threading
        caveat.
      - Documented user misuse: `arr.base.release()`.
      - Found while fixing: `model_revision` was recorded but never loaded,
        and is now passed to `SentenceTransformer`.

### Acceptance Criteria
- Every probe above flips: the 12 reproductions fail on the old code and pass
  on the new, inside the module's own test suite.
- Full suites of every touched module green on the meson build; corpus and
  rank-bm25 partial distributions green under `libs._tools verify`.
- Four maintenance gates pass; no new ruff finding in changed files.

### Results Review - round 33 Run 1 - 2026-10-12

#### Verification evidence (meson editable build, CPython 3.12)
| Suite | Uploaded code | After Run 1 |
|---|---|---|
| small group (levenshtein, _editdistance, rank_bm25, memmap, random, cexternals/_annoy, _cli, independence) | 578 passed | 681 passed |
| corpus | 3670 passed, 16 failed, 12 skipped | 3721 passed, the same 16 failed, 12 skipped |
| mcp | 287 passed, 1 skipped | 287 passed, 1 skipped |
| cleanprompt | 2867 passed, 109 skipped | 2867 passed, 109 skipped |
| annoy | 793 passed, 2 skipped, 1 xfailed | not rerun: imports none of the changed modules |

- The 16 corpus failures are environmental, identical by test id before and
  after: 14 need DNS (the SSRF guard fails closed, as designed) and 2 need
  NLTK's stopwords data. NOT PASS; unavailable lanes here.
- Cross-version, CI's own harness, run on a snapshot of the final tree (Rule 87).
  The snapshot differs from the delivered tree only in this file and in one
  docstring example in `mem_map.pyx` (`_ = m.write(...)`), which belongs to none
  of the three distributions:
  `python -m libs._tools verify scikit-plots-skinny scikit-plots-rank-bm25
  scikit-plots-corpus --python 3.9 --python 3.13`: 232 passed, 4 failed.
  - The 4 are `part's own tests pass [corpus]` on 3.9/3.13 x normal/lowest:
    3701 passed, 14 failed, 34 skipped each. The 14 are the DNS tests above.
  - rank-bm25: 233 passed on all four legs.
  - skinny: `_cli` 151 (3.9) and 156 (3.13) passed; `logging` 171 passed.
- Negative proofs (new tests against the uploaded module):
  - `_editdistance` 15/25 fail.
  - levenshtein facade 3 fail with the native fix kept.
  - rank_bm25 33 fail (both changed test files).
  - embeddings 71/144 fail, including construction failures: the shared helper passes `custom_fn_id`, which the uploaded engine does not accept.
  - `SemanticChunker` 22/23 fail. Of the first 16 tests, this round's first fix failed 8 and its second 2.
  - memmap 9/11 new tests fail. Two pass: a guard that holds on both, and the cross-thread export test, which needs the buffer protocol to exist.
  - Review fixes (step 10), against the reviewed version: memmap 2/2, semantic 6/7 (the seventh pins the declared-model precedence), embeddings 4/5 (the fifth is the unpinned guard), rank_bm25 3/3.
- Probes (`/home/claude/r33/probes`) flipped:
  - `bycython`: 91/20000 wrong results, now 0.
  - Custom-provider cache cross-talk: gone.
  - Manifests: `None` before, now stamped.
  - BM25L/BM25+ sparse vs dense: 34/300 and 35/300 rankings differed, now 0 (bit-identical).
  - Three memmap SIGSEGVs: each now exits 0 or raises `BufferError`.
  - One HYBRID `chunk()` made 135 `encode` calls, now 1.
- Docstring examples changed this round were executed (`doctest`, in isolation); they pass.
  `MemoryMap.read`'s example, in a docstring this round edited, failed before (it printed `5` from
  `write`) and now passes.
- Gates:
  - towncrier PASS (104 sections);
  - test module names ok;
  - path lengths 0 failing;
  - `libs._tools check` up to date;
  - ruff: no new finding in any changed file (5 fewer).
- `labeler check` fails identically on the uploaded tree. The archive is not a git checkout, so the generator falls back from `git ls-files` to walking the filesystem (`generate_pr_labeler.py:601-630`) and sees directories your repository does not track (`maintenances/mlflow`, `skills/mlflow`). Not a defect, and not caused by this round.
- `py_compile`/`ast` on 3.8 and 3.9: every changed Python file parses.
- The native modules (`_editdistance`, `memmap`) were built and tested on 3.12 only here. The CI full shard covers the other versions.

#### Deviations from plan
- Steps 8 and 9: two first fixes covered only the reported path. Probes then found the same root cause on more paths, and the fixes were completed in this run (Rules 85 and 86 now name the whole surface).
- Process:
  - `mem_map.pyx` was edited while a verify run was alive. Its builds had finished, and the tested distributions do not contain memmap, so the result stands.
  - A `pgrep` self-match reported that run as alive for two hours.
  - Both are captured in Rule 87.
- Not changed, found on the way (reported in the review bundle with evidence):
  - Three RRF implementations disagree on ties, on a document repeated within a leg (a different winner), and at weight 0, where zero-evidence documents are returned (Run 2).
  - Unconditional `-march=native`/`-mcpu=native` for five native modules. AVX-512 instructions are in the built `annoylib` on this host, so a wheel built so can raise SIGILL elsewhere (separate build PR).
  - `meson.build` computes the NumPy include path relative to `tools/` and uses it from the root.
  - `import scikitplot` raises `IndexError` without a `@software` entry in `CITATION.bib`.
  - Annoy's tests write 12 files into their own package directory, and two modules share `blah.ann` (Rule 30).
  - `SentenceChunkerConfig(backend="regex")` fails only at `chunk()`.
  - `chunker_kwargs` is refused by all four named chunkers.
  - `SemanticBackend.EMBEDDING` runs the HYBRID path.
  - `_cli` `CommandSpec.capabilities` is read nowhere.
  - `KissSeedSequence.spawn` read-modify-writes its counter without a lock. Atomic under the GIL by construction; a race on free-threaded builds, which the meson flags declare supported.

#### Files changed
- `cexternals/_editdistance/{_editdistance.cpp, bycython.pyx, __init__.py}`, `test/test_bycython.py` (new)
- `levenshtein/_core.py`, `tests/test_levenshtein.py`
- `rank_bm25/_rank_bm25.py`, `tests/test_persistence_and_sparse.py`, `tests/test__rank_bm25_input_contract.py`
- `corpus/_embeddings/_embedding.py`, `_embeddings/tests/test__embedding.py`, `_corpus_builder.py`, `tests/test__builder_customization_seams.py`, `_chunkers/_semantic.py`, `_chunkers/tests/test__semantic.py` (new)
- `memmap/_memmap/{mem_map.pyx, mem_map.pyi}`, `tests/test_mman.py`
- five fragments `docs/.../upcoming_changes/<module>/865.fix.rst`
- `tasks/todo.md`, `tasks/lessons.md` (Rules 78-87)
