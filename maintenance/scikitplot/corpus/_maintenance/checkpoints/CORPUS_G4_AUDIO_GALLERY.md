# Corpus G4 Audio Gallery Checkpoint

Date: 2026-08-19
Status: VERIFIED — SOURCE REGRESSION FIX + FOCUSED G4 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Source defect found before gallery finalization

`NLPEnricher.enrich_documents()` resolved stopword resources even when
`remove_stopwords=False`. If NLTK was installed but its stopword corpus was not,
a SIMPLE/frequency enrichment could raise `ResourceUnavailableError` despite
stopword filtering being disabled.

Smallest fix:

```text
remove_stopwords=True  -> resolve language + stopword resources
remove_stopwords=False -> use empty stopword set; do not touch optional data
```

Focused regression:

```text
3 passed, 160 deselected
```

A direct SIMPLE enrichment with managed downloads disabled also passed.

## Gallery file

Canonical target:

```text
plot_corpus_a_tale_of_two_cities_mp3_script.py
```

## Teaching goal

```text
real MP3
-> AudioReader companion precedence
-> REGEX sentence chunks
-> audio timecode/provenance
-> dependency-free SIMPLE enrichment
-> optional NLTK enrichment
-> optional Whisper ASR
```

The gallery uses a temporary SRT companion beside a copy of the bundled MP3.
`transcribe=True` remains configured, but the companion path wins before Whisper,
so the normal docs build does not import or download an ASR model. The MP3 is
source/provenance in this path; text comes from the sidecar and audio decoding is
not required.

## Optional-capability policy

```text
missing MP3 asset        -> explicit SKIP
missing NLTK/resource    -> skip NLTK subsection only
ASR not explicitly opted -> explicit SKIP
ASR opted + no backend   -> explicit SKIP
ASR opted + backend      -> execute; unexpected backend/model error remains FAIL
```

No broad exception handler converts real defects into skips.

## Verification

```text
py_compile                         PASS
core gallery execution             PASS
documents produced                 5
companion format                   srt
timecodes preserved                PASS
portable SIMPLE enrichment         PASS
NLTK local data absent             specific SKIP / PASS
Whisper default                    opt-in SKIP / PASS
ASR opt-in + no backend            specific SKIP / PASS
missing MP3 asset                  specific SKIP / PASS
network requests                   none
CSV/export primary path            removed
managed downloads                  none
```

## Physical maintenance gate

Source regression adds 6 source LOC and 14 test LOC in `_enrichers`; tracker is
refreshed to 57,106 source LOC / 31,928 test LOC. No tripwire is crossed.

## Next exact action

Start G5 on `plot_corpus_who_zip_script.py`:

1. preserve ZIP member routing and nested per-extension configuration;
2. keep the executed archive path lightweight and deterministic;
3. move Whisper/OCR member policies to optional preflighted scenarios;
4. remove incidental CSV/pandas from the primary archive lesson;
5. summarize member/source provenance and specific optional skips.
