# CORPUS G7 — WHO Multi-Source Integration Gallery Checkpoint

Date: 2026-08-19
Status: **VERIFIED_FOCUSED**

## Authority

```text
archive   scikit-plots(20260818-204251).zip
sha256    4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Requirement

Preserve the broad WHO multi-source integration role while making normal gallery
execution local/offline-first, keeping partial source success observable, and
clarifying when users should choose explicit components, CorpusBuilder, or
FluentCorpus/RuntimeCorpus.

Missing local assets and optional capabilities must produce specific SKIP
outcomes. Missing sidecars must never silently trigger network access.

## Changed gallery

```text
galleries/examples/corpus/plot_corpus_who_per_file_script.py
```

No Corpus implementation file changed in G7.

## User-visible behavior

```text
normal gallery
  -> local sidecars only
  -> no network fallback
  -> no embeddings/model download
  -> SIMPLE lexical enrichment
  -> keyword/strict retrieval

source-level partial success
  -> each source has OK or SKIP status
  -> successful evidence remains usable

optional PDF/OCR/ASR
  -> explicit preflight
  -> missing capability -> SKIP
  -> unexpected post-preflight defect -> FAIL

adapters
  -> grouped by user goal:
       agent/framework
       protocol/MCP
       data/RAG interchange
```

The gallery intentionally uses explicit lower-level components to reveal stage
boundaries. `CorpusBuilder` is presented as the higher-level heterogeneous
partial-success orchestrator. `FluentCorpus` / `RuntimeCorpus` is presented as
the immutable declarative/runtime path.

## Verification

```text
py_compile                         PASS

realistic local fixture run        PASS
sources represented                5
sources succeeded                  4
sources skipped                    1 (ASR disabled)
retained documents                 9
normalized                         9
enriched                           9
keyword index documents            9
dense embeddings                   False
search                             PASS
adapter groups                      PASS
network requests                    0

forced optional libs unavailable   PASS
web/video sources retained          2
PDF                                 SKIP
OCR                                 SKIP
ASR                                 SKIP
index/search/adapters               PASS

all local assets missing            PASS
all source rows                     SKIP
empty downstream phases             PASS
network requests                    0

ASR opt-in + no Whisper backend    specific SKIP / PASS

adapter/builder focused tests       31 passed, 3 skipped
maintenance tracker                PASS
```

## Compatibility / security

- No URL/SSRF, reader, retrieval, adapter, or builder implementation changed.
- Missing local assets never trigger public-network fallback.
- URL examples remain code-only and use the Corpus URL/reader security boundary.
- No managed NLTK download or model download in the normal path.
- `remove_stopwords=False` keeps lexical enrichment resource-light.
- Invalid configuration and security failures remain observable.

## Next action

Proceed to G8 collection closure: consolidate the canonical gallery file set,
retire historical v1/v2 public pages, add/refresh a Corpus gallery index/readme
when the gallery tree is available, and run the full canonical gallery set under
portable/optional-capability-missing conditions.
