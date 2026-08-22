"""
Process an MP3 with Corpus
==========================

.. currentmodule:: scikitplot.corpus

This example shows the audio ingestion path without making a documentation
build depend on a speech model download.

The default executed scenario uses a real bundled MP3 plus a small companion
SRT transcript.  :class:`AudioReader` checks companion transcript/lyrics files
before Whisper, so the audio can be routed through the normal Corpus media
reader while remaining deterministic and offline-safe.

The example demonstrates:

* audio reader dispatch from an MP3 path,
* companion-transcript precedence over Whisper ASR,
* portable REGEX sentence chunking,
* audio provenance such as source type, section type, and timecodes,
* dependency-free SIMPLE NLP enrichment,
* optional NLTK enrichment when local NLTK resources are present,
* optional real Whisper ASR only when the gallery is explicitly opted in.

Optional capability rule
------------------------

Missing optional packages or local data resources are reported as ``SKIP``.
Unexpected failures after a capability has been explicitly enabled are not
silently swallowed.

The live remote-MP3 path is shown as code only; this gallery performs no
network access.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import os
import importlib.util
import shutil
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
# Resolve the bundled audio asset
# -------------------------------
# The local asset is the reproducible execution path.  If a documentation
# packaging step omitted it, the example reports a bounded skip rather than
# falling through to a live URL automatically.


def _resolve_example_dir() -> Path:
    """Resolve the example directory across script and gallery runtimes."""
    file = globals().get("__file__")
    if file:
        return Path(file).resolve().parent
    return Path.cwd().resolve()


_EXAMPLE_DIR = _resolve_example_dir()
_DATA_DIR = _EXAMPLE_DIR / "data"
_AUDIO_PATH = _DATA_DIR / "tale_of_two_cities_01_dickens_64kb.mp3"


# %%
# Optional-capability preflight helpers
# -------------------------------------
# These checks do not install packages, download NLTK resources, or fetch
# Whisper model weights.


def _probe_nltk(*resource_paths: str) -> tuple[bool, str]:
    """Check NLTK and local NLTK data without downloading anything."""
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


def _probe_whisper_backend() -> tuple[bool, str]:
    """Return whether at least one supported Whisper Python backend exists."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return True, "faster-whisper is installed"
    if importlib.util.find_spec("whisper") is not None:
        return True, "openai-whisper is installed"
    return False, "neither faster-whisper nor openai-whisper is installed"


# %%
# Prepare a deterministic companion transcript
# --------------------------------------------
# ``AudioReader`` searches for a same-stem companion in this order:
#
# ``.lrc → .srt → .vtt → .txt``.
#
# We copy the gallery MP3 into a temporary workspace and place a short SRT next
# to it.  The transcript contains the opening LibriVox announcement already
# represented by this gallery asset.  Timecodes make the resulting
# ``CorpusDocument`` objects visibly different from ordinary plain-text input.

_WORKSPACE: tempfile.TemporaryDirectory[str] | None = None
_WORK_DIR: Path | None = None
_DEMO_AUDIO: Path | None = None

if not _AUDIO_PATH.exists():
    print(f"[SKIP] Bundled MP3 is unavailable: {_AUDIO_PATH}")
else:
    _WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-audio-gallery-")
    _WORK_DIR = Path(_WORKSPACE.name)
    _DEMO_AUDIO = _WORK_DIR / _AUDIO_PATH.name
    shutil.copy2(_AUDIO_PATH, _DEMO_AUDIO)

    _DEMO_AUDIO.with_suffix(".srt").write_text(
        """1
00:00:00,000 --> 00:00:08,000
This is a Librivox recording. All Librivox recordings are in the public domain.

2
00:00:08,000 --> 00:00:13,000
For more information or to volunteer, please visit Librivox.org.

3
00:00:14,000 --> 00:00:21,000
Recording by Michael Sirois.

4
00:00:22,000 --> 00:00:25,000
A Tale of Two Cities by Charles Dickens.
""",
        encoding="utf-8",
    )

    print("Bundled MP3:", _AUDIO_PATH)
    print("Temporary audio:", _DEMO_AUDIO)
    print("Companion:", _DEMO_AUDIO.with_suffix(".srt"))


# %%
# Process the MP3 through CorpusPipeline
# --------------------------------------
# ``transcribe=True`` is intentional here.  Because the SRT companion exists,
# :class:`AudioReader` consumes it first and never initializes Whisper.  In this
# path the MP3 remains the source/provenance object; the text comes from the
# sidecar, so the reader does not need to decode the audio signal.
#
# REGEX sentence chunking is used instead of NLTK so the primary media path
# remains dependency-light.  No export is configured because serialization is
# not the teaching goal of this example.

core_result = None
core_documents = ()

if _DEMO_AUDIO is None:
    print("[SKIP] Core audio pipeline: local MP3 asset is unavailable.")
else:
    pipeline = CorpusPipeline(
        chunker=SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                strip_whitespace=True,
                include_offsets=True,
            )
        ),
        reader_kwargs={
            "transcribe": True,
            "whisper_model": "base",
        },
    )

    core_result = pipeline.run(_DEMO_AUDIO)
    core_documents = core_result.documents

    print(f"Documents produced: {core_result.n_documents}")


# %%
# Inspect audio-specific evidence
# -------------------------------
# Companion SRT cues carry start/end timecodes.  The sentence chunker preserves
# those source timecodes when one cue produces multiple sentence chunks.
#
# ``companion_format`` is stored in metadata because it is reader-specific
# provenance rather than a universal CorpusDocument field.

for i, doc in enumerate(core_documents[:5]):
    companion_format = doc.metadata.get("companion_format")
    start = "-" if doc.timecode_start is None else f"{doc.timecode_start:.1f}s"
    end = "-" if doc.timecode_end is None else f"{doc.timecode_end:.1f}s"
    confidence = (
        "-"
        if doc.confidence is None
        else f"{doc.confidence:.3f}"
    )

    print(
        f"[{i}] source={doc.source_type} "
        f"section={doc.section_type} "
        f"time={start}→{end} "
        f"confidence={confidence} "
        f"companion={companion_format}"
    )
    print(f"    {doc.text[:180]!r}")


# %%
# Dependency-free NLP enrichment
# ------------------------------
# SIMPLE tokenization and frequency keywords do not need NLTK when stopword
# removal is disabled.  This gives the beginner example a real enrichment
# result even on a minimal installation.

portable_rich = []

if not core_documents:
    print("[SKIP] Portable enrichment: no audio documents are available.")
else:
    portable_rich = NLPEnricher(
        EnricherConfig(
            tokenizer="simple",
            keyword_extractor="frequency",
            max_keywords=8,
            remove_stopwords=False,
            min_token_length=2,
        )
    ).enrich_documents(core_documents[:1])

    doc = portable_rich[0]
    print("Portable tokens:", doc.tokens)
    print("Portable keywords:", doc.keywords)


# %%
# Optional richer NLTK enrichment
# -------------------------------
# The original showcase used NLTK tokenization, WordNet lemmatization, and a
# Snowball stemmer.  That scenario is preserved, but it runs only when NLTK and
# the required local resources are already available.
#
# No managed NLTK download is enabled by this example.

nltk_ready, nltk_reason = _probe_nltk(
    "tokenizers/punkt_tab",
    "corpora/wordnet",
    "corpora/omw-1.4",
)

nltk_rich = []

if not core_documents:
    print("[SKIP] NLTK enrichment: no audio documents are available.")
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
    ).enrich_documents(core_documents[:1])

    doc = nltk_rich[0]
    print("NLTK keywords:", doc.keywords)
    print("NLTK lemmas:", doc.lemmas)
    print("NLTK stems:", doc.stems)


# %%
# Optional real Whisper ASR
# -------------------------
# A companion transcript is usually the best reproducible docs/CI path.
# Real ASR is intentionally opt-in because Whisper may need large model files.
#
# To execute this section manually:
#
# .. code-block:: bash
#
#    SCIKITPLOT_GALLERY_RUN_ASR=1 python plot_corpus_a_tale_of_two_cities_mp3_script.py
#
# When opted in, the example requires an installed Whisper backend.  Model
# availability is then the responsibility of that explicitly enabled runtime;
# an unexpected ASR failure is allowed to surface rather than being mislabeled
# as a missing-package skip.

whisper_ready, whisper_reason = _probe_whisper_backend()
asr_documents = ()

if _DEMO_AUDIO is None:
    print("[SKIP] Whisper ASR: local MP3 asset is unavailable.")
elif not _RUN_ASR:
    print(
        "[SKIP] Whisper ASR: optional execution is disabled. "
        "Set SCIKITPLOT_GALLERY_RUN_ASR=1 to opt in."
    )
elif not whisper_ready:
    print(f"[SKIP] Whisper ASR: {whisper_reason}")
else:
    assert _WORK_DIR is not None

    # Use a different stem so the demo SRT companion cannot satisfy this read.
    asr_audio = _WORK_DIR / "tale_of_two_cities_asr_only.mp3"
    shutil.copy2(_DEMO_AUDIO, asr_audio)

    asr_reader = DocumentReader.create(
        asr_audio,
        transcribe=True,
        whisper_model="base",
        default_language="en",
    )
    asr_documents = tuple(asr_reader.get_documents())

    print(f"Whisper documents: {len(asr_documents)}")
    for i, doc in enumerate(asr_documents[:3]):
        start = "-" if doc.timecode_start is None else f"{doc.timecode_start:.1f}s"
        end = "-" if doc.timecode_end is None else f"{doc.timecode_end:.1f}s"
        confidence = (
            "-"
            if doc.confidence is None
            else f"{doc.confidence:.3f}"
        )
        print(
            f"[ASR {i}] time={start}→{end} confidence={confidence} "
            f"text={doc.text[:160]!r}"
        )


# %%
# Remote MP3 ingestion — code-only
# --------------------------------
# ``CorpusPipeline.run`` and ``run_url`` can also accept a remote media URL.
# The network path is not executed during a normal gallery build.
#
# .. code-block:: python
#
#    result = pipeline.run_url(
#        "https://archive.org/details/tale_two_cities_librivox/"
#        "tale_of_two_cities_01_dickens.mp3"
#    )
#
# For reproducible documentation or batch processing, prefer local audio plus a
# pre-generated companion transcript when practical.


# %%
# Takeaway
# --------
# The most reproducible audio workflow is:
#
# ``MP3 + companion transcript → Corpus documents with temporal provenance``.
#
# Whisper remains available as an opt-in fallback when no companion is present.
# Optional NLP resources can enrich the resulting documents, but they should
# not be prerequisites for basic media ingestion.


# %%
# Cleanup
# -------
# Only temporary copies of the MP3 and generated companion transcript were
# created by this example.

if _WORKSPACE is not None:
    _WORKSPACE.cleanup()
    print("Temporary audio workspace cleaned.")


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: beginner
#    purpose: showcase
