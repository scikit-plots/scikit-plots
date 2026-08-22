# CORPUS G8 — Gallery Collection Closure Checkpoint

Date: 2026-08-19
Status: **VERIFIED_PORTABLE_COLLECTION**

## Authority

```text
archive   scikit-plots(20260818-204251).zip
sha256    4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Requirement

Close the file-by-file Corpus gallery campaign as one coherent collection:

- keep one canonical active file per example;
- retire historical Hamlet v1/v2 variants from the active set;
- replace the old dependency-only gallery README with a learning-order/API-choice/capability index;
- preserve only real existing gallery assets;
- do not fabricate WHO/YouTube sidecars to make the multi-source example appear more complete;
- run all seven canonical examples together;
- rerun the collection with major optional Python dependencies simulated unavailable;
- missing optional capabilities must produce bounded SKIPs, not hard gallery failures;
- real API/security/installed-backend defects remain failures.

## Canonical active gallery set

```text
README.txt
plot_corpus_fluent_corpus_script.py
plot_corpus_fluent_hamlet_retrieval_script.py
plot_corpus_knowledge_script.py
plot_corpus_a_tale_of_two_cities_mp3_script.py
plot_corpus_who_zip_script.py
plot_corpus_who_youtube_script.py
plot_corpus_who_per_file_script.py
```

Historical public variants excluded from the active set:

```text
plot_corpus_fluent_hamlet_retrieval_script_v1.py
plot_corpus_fluent_hamlet_retrieval_script_v2.py
```

Git/history remains the source for historical versions.

## Gallery index UX

The refreshed `README.txt` teaches:

```text
learning order
API choice:
  CorpusPipeline
  CorpusBuilder
  FluentCorpus
  RuntimeCorpus
  RetrievalIndex / VectorIndexBackend
capability matrix
optional-capability SKIP policy
no implicit network fallback
install-only-what-you-need guidance
conservative browser/WASM expectations
```

## Assets

The collection uses only real assets already present in the supplied gallery
archive. For the WHO multi-source example, the PDF and image are extracted from
the supplied WHO ZIP member set. No synthetic web-article or YouTube sidecar is
added to the canonical bundle; those source rows therefore remain observable
SKIPs when the corresponding local sidecars are absent.

## Portable collection verification

```text
compileall                          PASS
canonical scripts                  7 / 7 PASS

Fluent basics                      PASS
Hamlet RuntimeCorpus               PASS
OCR chunking                       PASS
MP3 companion-transcript path      PASS
mixed-media ZIP                    PASS
YouTube local-proxy path           PASS
WHO multi-source partial success   PASS

normal opt-ins
  network                           disabled
  Whisper ASR                      disabled
```

Observed optional SKIPs are expected and visible (for example local NLTK data
resources and model-backed/live sections).

## Simulated optional-dependency-missing collection

The seven scripts were rerun with these Python packages hidden from import
resolution:

```text
nltk
faster_whisper
whisper
youtube_transcript_api
pytesseract
pypdf
pdfminer
langchain_core
datasets
```

Result:

```text
canonical scripts                  7 / 7 PASS
OCR example                        OCR + NLTK paths SKIP
MP3 example                        companion path PASS; NLTK/ASR optional paths SKIP
ZIP example                        zero optional member docs allowed; script PASS
YouTube example                    local proxy PASS; NLTK/live path SKIP
WHO multi-source                   all optional/local-missing sources SKIP; empty downstream path PASS
```

This verifies the gallery contract that absent optional capabilities do not
break unrelated portable examples.

## Environment-blocked documentation gate

This harness does not provide:

```text
sphinx
docutils
sphinx-gallery
```

Therefore the real Sphinx-Gallery generation/build gate is recorded as:

```text
SKIP — documentation dependencies unavailable in this verification environment
```

It is not reported as PASS.

## Corpus implementation impact

G8 itself changes no Corpus implementation file. Source fixes discovered by the
preceding gallery review remain the G4/G5 changes already regression-tested:

```text
G4  NLPEnricher remove_stopwords=False resource boundary
G5  ZIP member input_path/source_type provenance preservation
```

## Next action

Run the canonical collection through the project's declared Sphinx/Sphinx-Gallery
documentation environment when those dependencies are available. If that gate
is green, the Corpus gallery UX campaign can be marked closed and attention can
return to normal maintenance/cross-module work.
