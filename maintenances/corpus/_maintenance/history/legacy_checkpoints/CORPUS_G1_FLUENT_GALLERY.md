# Corpus G1 FluentCorpus Beginner Gallery Checkpoint

Date: 2026-08-19
Status: VERIFIED — FOCUSED G1 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

G1 is a gallery/user-education increment on top of the verified G0 documentation
alignment. It does not change Corpus runtime, retrieval, reader, storage,
security, NLP, or backend implementation semantics.

## Gallery file

```text
plot_corpus_fluent_corpus_script.py
```

Reviewed input attachment:

```text
plot_corpus_fluent_corpus_script(1).py
```

Delivered canonical revision:

```text
plot_corpus_fluent_corpus_script_v2.py
```

## Teaching goal

Teach the FluentCorpus configuration model first, then demonstrate the smallest
real operational transition:

```text
configure
→ validate / inspect
→ build CorpusPlan
→ materialize RuntimeCorpus
→ run local source
→ context-manager close
```

This file intentionally does not duplicate the larger Hamlet retrieval example.

## Changes applied

- preserved order independence, duplicate-domain conflict, explicit replacement,
  stages, validation, fingerprints, serialization, and CorpusPlan equivalence;
- replaced stale wording that said an execution integration was not yet available;
- clarified that `build()` remains `CorpusPlan` validation while
  `materialize()` constructs runtime state;
- clearly labelled abstract strings as non-materialized placeholders rather
  than implied registry/runtime names;
- added one real portable materialization scenario using:
  - a temporary local text file,
  - `reader="auto"`,
  - `ParagraphChunkerConfig`,
  - `storage="memory"`,
  - `RuntimePolicy(allow_network=False)`;
- demonstrated that materialization does not process the source;
- demonstrated context-manager lifecycle;
- demonstrated that an invalid plan is rejected before runtime execution;
- updated the newbie pattern to branch configuration then materialize only the
  selected executable branch;
- documented the gallery-wide optional-capability classification.

## Optional-capability rule

This G1 executed path deliberately has no optional dependency.

For later examples:

```text
missing optional dependency/resource/native capability/network opt-in
→ visible, specific SKIP

wrong API / invalid contract / security-policy violation / installed-backend defect
→ visible FAIL
```

A skip must not fabricate documents/results and must not be implemented with a
broad `except Exception` that can hide a real regression.

## Verification

```text
python -m py_compile plot_corpus_fluent_corpus_script_v2.py
PASS

full direct script execution against applied Corpus source
PASS

portable runtime smoke
materialize documents before run: 0
materialize storage before run:   0
run PipelineResult documents:     2
runtime documents after run:      2
stored documents after run:       2
context-manager close:            PASS
invalid-plan materialize reject:  PASS

optional-dependency blocked smoke
blocked: nltk, spacy, sentence_transformers, faster_whisper, whisper,
         pytesseract, easyocr, faiss, voyager
result: PASS

maintenance tracker
PASS
```

The source-checkout harness emits its pre-existing generated-version warning
when imported directly from an unpacked source tree; it does not alter the G1
result and is not changed by this gallery increment.

## Compatibility / security / resource impact

```text
runtime API behavior       unchanged
retrieval behavior         unchanged
security boundaries        unchanged
network use                none in executed G1 path
filesystem writes          temporary directory only
optional model downloads   none
native vector backend      none
NLTK/spaCy resources       none
```

## Next exact action

Start G2 on the real-data Hamlet RuntimeCorpus showcase:

1. keep v2 as the only active user-facing Hamlet example;
2. retire v1 from active gallery navigation;
3. preserve the verified `materialize/run/search/export` path;
4. add one bounded `RuntimeCorpus.add()` generation scenario;
5. make context-manager lifecycle the primary form;
6. keep Annoy configuration-only and generic via `index_kwargs`;
7. preserve the optional-capability SKIP/FAIL rule.
