# Corpus G2 Hamlet Runtime Gallery Checkpoint

Date: 2026-08-19
Status: VERIFIED — FOCUSED G2 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

G2 is a gallery/user-education increment on top of G0/G1. It does not change
Corpus runtime, retrieval, reader, storage, security, NLP, or backend
implementation semantics.

## Gallery file

Canonical active file:

```text
plot_corpus_fluent_hamlet_retrieval_script.py
```

Historical input files:

```text
plot_corpus_fluent_hamlet_retrieval_script_v1.py
plot_corpus_fluent_hamlet_retrieval_script_v2.py
```

G2 decision: v2 behavior becomes the canonical active example. v1 is historical
maintenance evidence only and should not remain in active gallery navigation.
The applied Corpus source snapshot does not contain the gallery index/tree, so
physical navigation removal is deferred to the gallery integration step rather
than fabricated here.

## Teaching goal

Teach the complete real-data Fluent/Runtime lifecycle on an offline Hamlet
corpus:

```text
FluentCorpus
→ materialize RuntimeCorpus
→ run first source
→ query storage / hybrid search
→ add second source
→ observe new coherent index generation
→ per-query retrieval override
→ export current generation
→ automatic context-manager close
```

## Changes applied

- made the context manager the primary runtime lifecycle form;
- preserved the portable deterministic hash embedder and local frequency
  enricher so no model/NLTK resource is required;
- preserved the real local Hamlet source and brute-force dense backend;
- added a second bounded Hamlet source for `RuntimeCorpus.add()`;
- verified `add()` grows runtime/storage state and advances `index_generation`;
- added a retrieval query that finds the newly added Polonius passage;
- exported only after `add()` so the exported JSONL represents the current
  complete generation;
- preserved keyword/semantic per-query retrieval overrides;
- preserved Annoy as configuration-only and uses generic `index_kwargs`;
- removed the manual `close()`-as-primary teaching path; context-manager close
  is primary while idempotent `close()` remains a documented runtime contract;
- kept all executed work local/offline.

## Optional-capability rule

The executed G2 path deliberately requires none of these optional capabilities:

```text
NLTK
spaCy
sentence-transformers
Whisper
OCR backends
FAISS
Voyager
Annoy native runtime
```

Annoy appears only as immutable configuration and is not materialized.

For later examples:

```text
missing optional dependency/resource/native capability/network opt-in
→ visible, specific SKIP

wrong API / invalid contract / security-policy violation / installed-backend defect
→ visible FAIL
```

Do not use broad `except Exception` to turn real regressions into skips.

## Verification

```text
python -m py_compile plot_corpus_fluent_hamlet_retrieval_script.py
PASS

full direct script execution against applied Corpus source
PASS

initial run
pipeline documents:       12
runtime documents:        12
stored documents:         12
dense backend:            bruteforce
hybrid top hit:            To be, or not to be...

incremental add
new pipeline documents:   2
runtime documents:        12 -> 14
stored documents:         14
index generation changed: True
new-passage search:        SUCCESS
new top hit:               Polonius / brevity is the soul of with

export
exported documents:       14
JSONL write:              PASS

context manager close:    PASS
Annoy generic config:     PASS

optional-dependency blocked smoke
blocked: nltk, spacy, sentence_transformers, faster_whisper, whisper,
         pytesseract, easyocr, faiss, voyager
result: PASS

maintenance tracker
PASS
```

The unpacked source-tree import emits the pre-existing generated-version warning
in this harness; execution still completes and G2 does not modify that unrelated
behavior.

## Compatibility / security / resource impact

```text
runtime API behavior       unchanged
retrieval behavior         unchanged
security boundaries        unchanged
network use                none
filesystem writes          temporary directory only
optional model downloads   none
native vector backend      none executed
NLTK/spaCy resources       none
```

## Next exact action

Start G3 on `plot_corpus_knowledge_script.py`:

1. preserve its chunking-comparison teaching goal;
2. inspect PipelineResult documents directly instead of exporting/re-reading CSV
   for every strategy;
3. add one compact comparison summary across strategies;
4. make the default executed path portable where the teaching point permits;
5. keep NLTK/model/OCR-dependent variants explicit and skip them specifically
   when the capability is unavailable;
6. do not hide real API or sidecar-asset defects.
