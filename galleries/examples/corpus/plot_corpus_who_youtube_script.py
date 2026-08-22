"""
Process a YouTube Transcript with Corpus
========================================

.. currentmodule:: scikitplot.corpus

YouTube transcript ingestion is a network capability: the reader uses
``youtube-transcript-api`` to request caption tracks from YouTube and does not
download the video or run a local speech model.

A normal documentation build should not depend on that external service.
This example therefore separates three concerns:

1. **Executed offline path** — a small local transcript-shaped proxy exercises
   the downstream Corpus pipeline deterministically.
2. **Real YouTube configuration** — :class:`DocumentReader` is configured for
   the actual URL without fetching the transcript.
3. **Optional live fetch** — enabled only when the user explicitly opts into
   network execution and ``youtube-transcript-api`` is installed.

The local proxy is intentionally synthetic.  It is **not** presented as a
verbatim transcript of the linked video and does not carry YouTube cue
timecodes.  Live :class:`YouTubeReader` documents do carry transcript
timecodes, video ID, transcript type, and transcript language.

YouTube
-------

* Richard Feynman — *The Character of Physical Law*
  `"https://www.youtube.com/watch?v=kEx-gRfuhhk"
  <https://www.youtube.com/watch?v=kEx-gRfuhhk>`_

  .. youtube:: kEx-gRfuhhk

Optional capability rule
------------------------

Missing ``youtube-transcript-api`` or local NLTK resources are reported as
``SKIP``.  Live transcript unavailability is also a bounded external-service
skip after explicit network opt-in.  Invalid Corpus configuration is not
silently swallowed.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import os
import importlib.util
import tempfile
from pathlib import Path

from scikitplot.corpus import (
    CorpusPipeline,
    DocumentReader,
    EnricherConfig,
    NLPEnricher,
    SentenceBackend,
    SentenceChunker,
    SentenceChunkerConfig,
    SourceType,
)

# os.environ["SCIKITPLOT_GALLERY_RUN_ASR"] = "1"
# os.environ["SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS"] = "1"
_RUN_ASR = os.environ.get("SCIKITPLOT_GALLERY_RUN_ASR", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS = os.getenv("SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# %%
# Configuration
# -------------
# The live URL is stable configuration; no request is made merely by creating
# the reader object.

YOUTUBE_URL = "https://www.youtube.com/watch?v=kEx-gRfuhhk"

_RUN_NETWORK = os.environ.get(
    "SCIKITPLOT_GALLERY_RUN_NETWORK", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# %%
# Optional-capability preflight helpers
# -------------------------------------


def _probe_youtube_api() -> tuple[bool, str]:
    """Check whether youtube-transcript-api is importable."""
    if importlib.util.find_spec("youtube_transcript_api") is None:
        return False, "youtube-transcript-api is not installed"
    return True, "youtube-transcript-api is installed"


def _probe_nltk(*resource_paths: str) -> tuple[bool, str]:
    """Check NLTK resources without downloading them."""
    if importlib.util.find_spec("nltk") is None:
        return False, "NLTK is not installed"

    try:
        import nltk
    except ImportError:
        return False, "NLTK is not importable"

    missing: list[str] = []
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(resource_path)

    if missing and not _SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS:
        return False, "missing local NLTK resources: " + ", ".join(missing)

    return True, "NLTK resources available"


# %%
# Build a deterministic local transcript-shaped proxy
# ---------------------------------------------------
# This text is written specifically for the gallery.  It is not copied from the
# linked video's transcript.  Its purpose is to let the page execute a real
# Corpus path even when YouTube is offline or unavailable.

LOCAL_PROXY_TEXT = """\
This local transcript proxy exists only to demonstrate Corpus processing.
It is not a verbatim transcript of the linked YouTube video.

A lecture transcript often contains short spoken sentences and repeated ideas.
Corpus can split that text into sentence-sized documents before later
normalization, enrichment, storage, or retrieval.

For a real YouTube source, YouTubeReader obtains caption cues from the external
transcript service. Those live cues can include start and end timecodes,
transcript language, video identifier, and whether captions were manual or
auto-generated.
"""

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-youtube-gallery-")
_WORK_DIR = Path(_WORKSPACE.name)
_PROXY_PATH = _WORK_DIR / "feynman_youtube_transcript_proxy.txt"
_PROXY_PATH.write_text(LOCAL_PROXY_TEXT, encoding="utf-8")

print("Local proxy:", _PROXY_PATH)
print("Characters:", len(LOCAL_PROXY_TEXT))


# %%
# Execute the offline Corpus path
# -------------------------------
# The proxy is explicitly labelled ``SourceType.VIDEO`` so downstream code can
# exercise video-source semantics without claiming a live YouTube fetch.
#
# REGEX sentence chunking keeps this primary path independent of NLTK data.

pipeline = CorpusPipeline(
    chunker=SentenceChunker(
        SentenceChunkerConfig(
            backend=SentenceBackend.REGEX,
            strip_whitespace=True,
            include_offsets=True,
        )
    )
)

proxy_reader = DocumentReader.create(
    _PROXY_PATH,
    source_type=SourceType.VIDEO,
    source_title="Local proxy for The Character of Physical Law",
    source_author="Gallery-generated teaching text",
    collection_id="youtube-transcript-gallery-proxy",
)

# ``CorpusPipeline.run`` performs its own reader dispatch, so the same metadata
# is passed through ``reader_kwargs`` for the executed pipeline path.
proxy_pipeline = CorpusPipeline(
    chunker=SentenceChunker(
        SentenceChunkerConfig(
            backend=SentenceBackend.REGEX,
            strip_whitespace=True,
            include_offsets=True,
        )
    ),
    reader_kwargs={
        "source_type": SourceType.VIDEO,
        "source_title": "Local proxy for The Character of Physical Law",
        "source_author": "Gallery-generated teaching text",
        "collection_id": "youtube-transcript-gallery-proxy",
    },
)

proxy_result = proxy_pipeline.run(_PROXY_PATH)

print(f"Proxy documents: {proxy_result.n_documents}")

for i, doc in enumerate(proxy_result.documents[:5]):
    print(
        f"[{i}] source_type={doc.source_type} "
        f"title={doc.source_title!r}"
    )
    print(f"    {doc.text[:200]!r}")


# %%
# Portable enrichment
# -------------------
# SIMPLE tokenization plus frequency keywords does not require NLTK when
# stopword removal is disabled.

portable_rich = []

if not proxy_result.documents:
    print("[SKIP] Portable enrichment: proxy produced no documents.")
else:
    portable_rich = NLPEnricher(
        EnricherConfig(
            tokenizer="simple",
            keyword_extractor="frequency",
            max_keywords=8,
            remove_stopwords=False,
            min_token_length=2,
        )
    ).enrich_documents(proxy_result.documents[:1])

    doc = portable_rich[0]
    print("Portable tokens:", doc.tokens)
    print("Portable keywords:", doc.keywords)


# %%
# Optional NLTK enrichment
# ------------------------
# Richer linguistic processing is useful when its local resources are already
# provisioned.  The gallery never downloads them automatically.

nltk_ready, nltk_reason = _probe_nltk(
    "tokenizers/punkt_tab",
    "corpora/wordnet",
    "corpora/omw-1.4",
)

if not proxy_result.documents:
    print("[SKIP] NLTK enrichment: proxy produced no documents.")
elif not nltk_ready:
    print(f"[SKIP] NLTK enrichment: {nltk_reason}")
else:
    nltk_rich = NLPEnricher(
        EnricherConfig(
            tokenizer="nltk",
            lemmatizer="nltk",
            stemmer="snowball",
            stemmer_language="english",
            keyword_extractor="frequency",
            max_keywords=8,
            remove_stopwords=False,
            min_token_length=2,
        )
    ).enrich_documents(proxy_result.documents[:1])

    doc = nltk_rich[0]
    print("NLTK keywords:", doc.keywords)
    print("NLTK lemmas:", doc.lemmas)
    print("NLTK stems:", doc.stems)


# %%
# Configure the real YouTube reader without fetching
# --------------------------------------------------
# URL dispatch happens without a network request for recognised YouTube URLs.
# The object below is the real :class:`YouTubeReader`; calling
# ``get_documents()`` is the point at which transcript retrieval begins.

youtube_reader = DocumentReader.create(
    YOUTUBE_URL,
    preferred_language="en",
    include_auto_generated=True,
    merge_short_cues=True,
    min_cue_chars=20,
)

print("Live reader type:", type(youtube_reader).__name__)
print("Configured URL:", youtube_reader.source_uri)
print("Preferred language:", youtube_reader.preferred_language)
print("Auto-generated fallback:", youtube_reader.include_auto_generated)


# %%
# Optional live YouTube transcript fetch
# --------------------------------------
# A normal gallery build never performs this request.
#
# To opt in manually:
#
# .. code-block:: bash
#
#    SCIKITPLOT_GALLERY_RUN_NETWORK=1 \
#        python plot_corpus_who_youtube_script.py
#
# When enabled, the request still depends on caption availability, YouTube
# policy, network reachability, and the external ``youtube-transcript-api``.
#
# ``RuntimeError`` from YouTubeReader represents the documented external
# transcript-fetch boundary and is reported as a live-service SKIP.  Invalid
# URL/configuration errors are not caught.

youtube_api_ready, youtube_api_reason = _probe_youtube_api()
live_documents = ()

if not _RUN_NETWORK:
    print(
        "[SKIP] Live YouTube transcript: network execution is disabled. "
        "Set SCIKITPLOT_GALLERY_RUN_NETWORK=1 to opt in."
    )
elif not youtube_api_ready:
    print(f"[SKIP] Live YouTube transcript: {youtube_api_reason}")
else:
    try:
        live_documents = tuple(youtube_reader.get_documents())
    except RuntimeError as exc:
        print(
            "[SKIP] Live YouTube transcript: external transcript service "
            f"did not return a usable transcript ({exc})."
        )
    else:
        print(f"Live transcript documents: {len(live_documents)}")

        for i, doc in enumerate(live_documents[:5]):
            transcript_type = doc.metadata.get("transcript_type")
            language = doc.metadata.get("transcript_language")
            video_id = doc.metadata.get("video_id")
            start = (
                "-"
                if doc.timecode_start is None
                else f"{doc.timecode_start:.3f}s"
            )
            end = (
                "-"
                if doc.timecode_end is None
                else f"{doc.timecode_end:.3f}s"
            )

            print(
                f"[{i}] {start}→{end} "
                f"type={transcript_type} language={language} "
                f"video_id={video_id}"
            )
            print(f"    {doc.text[:220]!r}")


# %%
# Pipeline URL forms
# ------------------
# When live network execution is intentionally enabled, both forms below are
# supported:
#
# .. code-block:: python
#
#    result = pipeline.run(YOUTUBE_URL)
#
#    # Equivalent explicit form:
#    result = pipeline.run_url(YOUTUBE_URL)
#
# The first form is convenient; the second makes URL intent explicit.


# %%
# What the local proxy does and does not prove
# --------------------------------------------
# The executed proxy verifies:
#
# * local reader dispatch,
# * video source labelling,
# * sentence chunking,
# * downstream enrichment.
#
# It deliberately does **not** claim to verify:
#
# * current transcript availability for the linked video,
# * YouTube cue timecodes,
# * manual-vs-auto track selection,
# * YouTube network policy.
#
# Those properties belong to the optional live YouTubeReader path.


# %%
# Takeaway
# --------
# For deterministic docs and CI, keep network transcript retrieval optional.
# For interactive use, create the same YouTube reader and opt into the live
# request when network access and the transcript dependency are intentionally
# available.

_WORKSPACE.cleanup()
print("Temporary transcript-proxy workspace cleaned.")


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
