# Corpus G3 OCR Chunking Gallery Checkpoint

Date: 2026-08-19
Status: VERIFIED — FOCUSED G3 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

G3 is a gallery/user-education increment on top of G0-G2. It changes no
Corpus reader, OCR, chunker, multilingual, resource-gate, export, or runtime
implementation semantics.

## Gallery file

Canonical target:

```text
plot_corpus_knowledge_script.py
```

## Teaching goal

Compare how multiple chunking strategies divide the **same OCR-extracted text**.
OCR is performed once; chunkers then operate directly on that shared text.

```text
image
→ DocumentReader OCR once
→ shared OCR text
→ Word / Sentence / Fixed / Semantic chunkers
→ in-memory comparison summary
```

## Changes applied

- removed the repeated `CorpusPipeline -> CSV -> pandas.read_csv` round-trip
  from the primary comparison path;
- performs OCR once instead of once per chunker;
- keeps all chunking results in memory;
- makes the default executed word/sentence paths portable with SIMPLE/REGEX and
  no NLTK data requirement;
- preserves the original NLTK sentence/tokenization/Porter/WordNet teaching
  content as separately labelled optional scenarios;
- preflights NLTK package/data availability without downloading resources;
- preflights Pillow/pytesseract/Tesseract before OCR;
- missing OCR capability produces a visible SKIP and still displays the source
  image;
- unexpected failures after a successful capability preflight are not broadly
  swallowed;
- adds a compact cross-strategy summary with chunk count, average/min/max
  character length, bounded sample, and intended use case;
- retains the morphological semantic/multilingual showcase without a model
  download;
- removes pandas/CSV as unrelated requirements of the main chunking lesson.

## Optional-capability rule

```text
missing Pillow/pytesseract/Tesseract
→ SKIP OCR-dependent comparison

missing NLTK package/resource
→ SKIP only the NLTK variants

missing optional writing-system tokenizer used by an internal documented fallback
→ fallback remains observable through Corpus warnings

unexpected API/type/security/backend defect
→ FAIL
```

No broad `except Exception` converts real defects into skips.

## Verification

```text
python -m py_compile plot_corpus_knowledge_script.py
PASS

full direct script execution with OCR capability
PASS

OCR documents:              1
OCR characters:             1,296
mean OCR confidence:        0.772

portable strategies
Word / document             1 chunk
Word / sentence             7 chunks
Sentence / REGEX            7 chunks
Fixed / characters          5 chunks
Fixed / tokens              7 chunks
Semantic / morphological    236 chunks

NLTK resources unavailable in harness
Sentence / NLTK             SKIP
Word / NLTK + WordNet       SKIP
script result               PASS

forced Tesseract-unavailable smoke
OCR extraction              SKIP
all OCR-dependent strategies SKIP
source image display path   retained
script result               PASS

forced NLTK-import-unavailable smoke
portable strategies         PASS
NLTK variants               SKIP
script result               PASS
```

The semantic morphological path emitted existing writing-system fallback
warnings for optional CJK/Japanese tokenizers; those are Corpus's documented
fallback behavior and were not converted into gallery errors.

## Compatibility / security / resource impact

```text
Corpus API behavior         unchanged
chunker behavior            unchanged
OCR behavior                unchanged
security boundaries         unchanged
network use                 none
NLTK downloads              none
model downloads             none
CSV/pandas dependency       removed from primary gallery path
OCR repetitions             N -> 1
```

## Next exact action

Start G4 on `plot_corpus_a_tale_of_two_cities_mp3_script.py`:

1. preserve the MP3/transcription teaching goal;
2. preflight the local audio sidecar and ASR capability;
3. make missing Whisper/native/model capability a specific visible SKIP;
4. separate optional NLTK enrichment from the core audio path;
5. remove CSV export from the primary path unless export itself is taught;
6. show bounded audio-specific evidence such as source type/timecode/confidence
   when available;
7. keep the remote MP3 path code-only/offline-safe.
