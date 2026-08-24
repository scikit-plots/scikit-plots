# CORPUS G6 — YouTube Transcript Gallery Checkpoint

Date: 2026-08-19
Status: **VERIFIED_FOCUSED**

## Authority

```text
archive   scikit-plots(20260818-204251).zip
sha256    4f62b638647a1645ad6174b646c2aabb4b30bee4a28ac9afb12d25ee71ff4b27
```

## Requirement

Make the YouTube gallery produce useful deterministic Corpus output without
making Sphinx-Gallery depend on YouTube/network access, while preserving the
real YouTubeReader configuration and optional live path. Missing optional
packages/resources must skip specifically rather than fail the whole example.

## Changed gallery

```text
galleries/examples/corpus/plot_corpus_who_youtube_script.py
```

No Corpus implementation file changed in G6.

## User-visible behavior

```text
local synthetic transcript-shaped proxy
  -> SourceType.VIDEO
  -> REGEX sentence chunking
  -> SIMPLE enrichment

DocumentReader.create(real YouTube URL)
  -> YouTubeReader object
  -> no request yet

normal gallery
  -> live request disabled

SCIKITPLOT_GALLERY_RUN_NETWORK=1
  -> API missing: SKIP
  -> external transcript RuntimeError: SKIP
  -> successful service: show bounded cue/timecode/provenance preview
```

The proxy is explicitly not a verbatim transcript and does not claim live cue
metadata.

## Verification

```text
py_compile                         PASS
full offline execution             PASS
proxy documents                    6
video source label                 PASS
portable SIMPLE enrichment         PASS
NLTK missing                       explicit SKIP / PASS
forced NLTK unavailable            explicit SKIP / PASS
network disabled                   explicit SKIP / PASS
network opt-in + API absent        explicit SKIP / PASS
YouTubeReader focused tests        31 passed
maintenance tracker               PASS
```

The broader `_readers/tests/test__web.py` run has 11 unrelated DNS failures in
this harness because `example.com` cannot be resolved; the SSRF validator
correctly fails closed. No security behavior was weakened.

## Compatibility / security

- No reader or URL-security implementation change.
- No live network request in the default gallery path.
- No dependency installation or managed NLTK download.
- Invalid YouTube URL/configuration failures remain observable.

## Next action

Proceed to G7: review `plot_corpus_who_per_file_script.py`, preserve its broad
multi-source integration role, remove automatic missing-sidecar -> network
fallback from the executed path, clarify manual orchestration vs RuntimeCorpus /
CorpusBuilder, and group adapters by user scenario.
