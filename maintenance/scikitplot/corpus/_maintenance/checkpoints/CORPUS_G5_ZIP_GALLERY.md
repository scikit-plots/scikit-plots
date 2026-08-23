# Corpus G5 ZIP Gallery Checkpoint

Date: 2026-08-19
Status: VERIFIED — SOURCE PROVENANCE FIX + FOCUSED G5 GATE GREEN

## Source authority

```text
archive: scikit-plots(20260818-204251).zip
sha256: 4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Source defect found before gallery finalization

`ZipReader` documented member provenance as `"<archive>/<member>"`, but the
aggregate `DocumentReader.get_documents()` layer collapsed final
`CorpusDocument.input_path` to the outer archive name. Per-member inferred
`source_type` was also lost unless a member reader emitted it directly.

Smallest fix:

```text
ZipReader child raw chunk
  -> attach logical archive/member input_path
  -> bridge child source_provenance

DocumentReader.get_documents
  -> honor raw-chunk input_path as the logical CorpusDocument label
  -> keep outer reader filename as fallback
```

Focused reader regression:

```text
70 passed
```

New tests pin raw member provenance, final `CorpusDocument` provenance, and compositional nested paths such as `outer.zip/inner.zip/document.txt`.

## Gallery file

Canonical target:

```text
plot_corpus_who_zip_script.py
```

## Teaching goal

```text
inspect ZIP manifest
-> secure ZipReader routing
-> one shared REGEX sentence pipeline
-> summarize archive/member provenance
-> show per-extension reader kwargs
-> optional ASR only after explicit opt-in
```

The primary path does not configure CSV export and does not enable Whisper.
The real WHO archive produces PDF/image evidence when those member readers are
available; the MP3 yields zero text when it has no companion and ASR is off.

## Optional-capability policy

```text
missing ZIP asset              -> explicit SKIP
optional member reader failure -> ZipReader skips that member per current contract
ASR not explicitly opted       -> explicit SKIP
ASR opted + no backend         -> explicit SKIP
ASR opted + backend            -> execute; unexpected model/backend failure remains visible
archive security-limit failure -> FAIL / never converted to SKIP
```

## Verification

```text
py_compile                         PASS
ZipReader focused test file        70 passed
real archive members               3
portable gallery execution         PASS
sentence documents                 112
PDF/research documents             110
image documents                    2
MP3 documents                      0 (ASR disabled)
member input_path provenance       PASS
member source_type provenance      PASS
per-extension normalization        PASS
ASR default                        opt-in SKIP / PASS
ASR opt-in + no backend            specific SKIP / PASS
missing ZIP asset                  specific SKIP / PASS
CSV/pandas primary path            removed
network requests                   none
```

## Physical maintenance gate

The provenance bridge updates `_base.py`, `_readers/_zip.py`, and focused ZIP
reader tests. Tracker is refreshed to 57,132 source LOC / 31,971 test LOC. No
structural tripwire is crossed; `_base.py` is 3,178 LOC, still below the 3,500
tripwire.

## Next exact action

Start G6 on `plot_corpus_who_youtube_script.py`:

1. preserve the live YouTube URL as code-only/network-opt-in;
2. add a deterministic executed transcript-proxy path so the page produces real Corpus output;
3. use portable sentence processing in the primary path;
4. make optional NLTK enrichment a specific preflighted subsection;
5. avoid implying the proxy is a successful live YouTube request.
