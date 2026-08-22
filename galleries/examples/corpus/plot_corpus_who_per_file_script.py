"""
Build a Multi-Source WHO Corpus
===============================

.. currentmodule:: scikitplot.corpus

This is the broad, explicit integration showcase for :mod:`scikitplot.corpus`.

It follows one collection through:

``ingest → normalize → enrich → index → search → adapt``.

Five source shapes are represented independently:

* a saved web-article proxy,
* a saved YouTube-transcript proxy,
* a PDF report,
* an image scan,
* an audio file.

The normal gallery path is **local-first and offline**.  If a local sidecar is
missing, that source is reported as ``SKIP``.  The example does not silently
turn a missing gallery asset into a public-network request.

This file intentionally uses the lower-level public components so the stage
boundaries and per-source partial-success behavior are visible.  For higher
level orchestration, use :class:`CorpusBuilder`; for reusable immutable
configuration and a stateful runtime lifecycle, use
:class:`FluentCorpus` / :class:`RuntimeCorpus`.

Optional capability rule
------------------------

A missing local asset or optional dependency is reported as a specific
``SKIP``.  Successfully ingested evidence from other sources remains usable.

Unexpected failures after a capability preflight are **not** swallowed.  That
keeps real API, reader, security, and installed-backend defects visible.

The live WHO/YouTube/media URLs are shown later as explicit code-only examples.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from scikitplot.corpus import (
    BuilderConfig,
    CorpusBuilder,
    CorpusDocument,
    DocumentReader,
    EnricherConfig,
    MCPCorpusServer,
    NLPEnricher,
    NormalizationPipeline,
    RetrievalConfig,
    RetrievalIndex,
    SourceType,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    to_huggingface_dataset,
    to_jsonl,
    to_langchain_documents,
    to_langgraph_state,
    to_mcp_resources,
    to_mcp_tool_result,
    to_rag_tuples,
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
# Resolve local gallery assets
# ----------------------------
# Sphinx-Gallery can execute generated copies of examples, so paths are anchored
# to the example directory rather than the caller's working directory.


def _resolve_example_dir() -> Path:
    """Resolve the example directory across scripts and Sphinx-Gallery."""
    file = globals().get("__file__")
    if file:
        return Path(file).resolve().parent
    return Path.cwd().resolve()


_EXAMPLE_DIR = _resolve_example_dir()
_DATA_DIR = _EXAMPLE_DIR / "data"

_WEB_TEXT_PATH = _DATA_DIR / "who_health_care_article_raw_html.txt"
_VIDEO_TEXT_PATH = _DATA_DIR / "who_video_transcript.txt"
_PDF_PATH = _DATA_DIR / "WHO-EURO-2025-12555-52329-80560-eng.pdf"
_IMAGE_PATH = _DATA_DIR / "WHO-EURO-2025-12555-52329-80560-eng.pdf.jpg"
_AUDIO_PATH = _DATA_DIR / "can-people-afford-to-pay-for-health-care.mp3"

_COLLECTION_ID = "who-greece-financial-protection"


# %%
# External/live source equivalents — not executed
# -----------------------------------------------
# These URLs document the corresponding live sources.  They are never used as
# an implicit fallback merely because a local sidecar is missing.

WHO_ARTICLE_URL = (
    "https://www.who.int/europe/news/item/"
    "12-12-2023-out-of-pocket-payments-for-primary-health-care-"
    "unaffordable-for-millions-in-europe-new-who-report-shows"
)
WHO_VIDEO_URL = "https://youtu.be/rwPISgZcYIk"
WHO_PDF_URL = (
    "https://iris.who.int/server/api/core/bitstreams/"
    "7ad66865-7f23-4485-8cf5-7b3d78bdf4f9/content"
)
WHO_IMAGE_URL = (
    "https://iris.who.int/server/api/core/bitstreams/"
    "d57241c0-512d-4cfc-9ead-91a83eea83f0/content"
)

print("Live-source equivalents are documented but not fetched:")
for label, url in [
    ("web", WHO_ARTICLE_URL),
    ("video", WHO_VIDEO_URL),
    ("pdf", WHO_PDF_URL),
    ("image", WHO_IMAGE_URL),
]:
    print(f"  {label:5s} {url}")


# %%
# Small display/status helpers
# ----------------------------


def banner(title: str) -> None:
    """Print a visible phase separator."""
    line = "=" * 76
    print(f"\n{line}\n{title}\n{line}")


def show_doc(doc: CorpusDocument, index: int = 0) -> None:
    """Print a bounded evidence preview."""
    preview = doc.text[:120].replace("\n", " ")
    print(
        f"  [{index}] source_type={doc.source_type} "
        f"input={doc.input_path!r}"
    )
    if doc.source_title:
        print(f"      title: {doc.source_title}")
    if doc.page_number is not None:
        print(f"      page: {doc.page_number}")
    if doc.timecode_start is not None:
        end = (
            "?"
            if doc.timecode_end is None
            else f"{doc.timecode_end:.1f}s"
        )
        print(f"      time: {doc.timecode_start:.1f}s → {end}")
    if doc.confidence is not None:
        print(f"      confidence: {doc.confidence:.3f}")
    if doc.ocr_engine:
        print(f"      OCR: {doc.ocr_engine}")
    print(f"      text: {preview!r}")


source_log: list[dict[str, object]] = []
all_documents: list[CorpusDocument] = []


def _record(
    label: str,
    *,
    status: str,
    documents: list[CorpusDocument] | None = None,
    detail: str = "",
) -> None:
    """Record one source outcome and retain successful documents."""
    docs = documents or []
    if docs:
        all_documents.extend(docs)

    source_log.append(
        {
            "source": label,
            "status": status,
            "documents": len(docs),
            "detail": detail,
        }
    )

    marker = "✓" if status == "OK" else "↷"
    suffix = f" — {detail}" if detail else ""
    print(f"{marker} {label}: {len(docs)} documents [{status}]{suffix}")

    if docs:
        show_doc(docs[0])


# %%
# Capability preflights
# ---------------------
# These checks do not install dependencies or download model/resource data.


def _pdf_ready() -> tuple[bool, str]:
    """Check whether at least one built-in PDF extraction backend exists."""
    if importlib.util.find_spec("pdfminer") is not None:
        return True, "pdfminer.six available"
    if importlib.util.find_spec("pypdf") is not None:
        return True, "pypdf available"
    return False, "neither pdfminer.six nor pypdf is installed"


def _ocr_ready() -> tuple[bool, str]:
    """Check the default Tesseract image-reader capability."""
    if importlib.util.find_spec("PIL") is None:
        return False, "Pillow is not installed"
    if importlib.util.find_spec("pytesseract") is None:
        return False, "pytesseract is not installed"
    if shutil.which("tesseract") is None:
        return False, "Tesseract executable is not available on PATH"
    return True, "pytesseract + Tesseract available"


def _whisper_ready() -> tuple[bool, str]:
    """Check only for an installed supported Whisper Python backend."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return True, "faster-whisper installed"
    if importlib.util.find_spec("whisper") is not None:
        return True, "openai-whisper installed"
    return False, "neither faster-whisper nor openai-whisper is installed"


# %%
# Phase 1 — ingest heterogeneous local sources
# --------------------------------------------
# Every source is independent.  A missing optional source does not erase
# evidence already read from another source.

banner("PHASE 1 — INGEST LOCAL SOURCES")


# %%
# Source 1 — saved web article
# ----------------------------
# The local text/HTML proxy is labelled as web evidence.  Missing local data is
# a gallery packaging skip, not a reason to contact the public URL.

if not _WEB_TEXT_PATH.exists():
    _record(
        "web article",
        status="SKIP",
        detail=f"local sidecar missing: {_WEB_TEXT_PATH.name}",
    )
else:
    web_reader = DocumentReader.create(
        _WEB_TEXT_PATH,
        source_type=SourceType.WEB,
        source_title=(
            "Out-of-pocket payments for health care "
            "unaffordable for millions in Europe"
        ),
        source_author="WHO Regional Office for Europe",
        source_date="2023-12-12",
        collection_id=_COLLECTION_ID,
    )
    _record(
        "web article",
        status="OK",
        documents=list(web_reader.get_documents()),
    )


# %%
# Source 2 — saved YouTube transcript proxy
# -----------------------------------------
# The dedicated YouTube gallery covers live caption retrieval.  Here a saved
# transcript sidecar keeps this broad integration example deterministic.

if not _VIDEO_TEXT_PATH.exists():
    _record(
        "YouTube transcript proxy",
        status="SKIP",
        detail=f"local sidecar missing: {_VIDEO_TEXT_PATH.name}",
    )
else:
    video_reader = DocumentReader.create(
        _VIDEO_TEXT_PATH,
        source_type=SourceType.VIDEO,
        source_title="Can people afford to pay for health care? WHO Europe",
        collection_id=_COLLECTION_ID,
    )
    _record(
        "YouTube transcript proxy",
        status="OK",
        documents=list(video_reader.get_documents()),
    )


# %%
# Source 3 — PDF report
# ---------------------
# The local PDF is executed only when a built-in PDF backend is present.

pdf_ready, pdf_reason = _pdf_ready()

if not _PDF_PATH.exists():
    _record(
        "PDF report",
        status="SKIP",
        detail=f"local sidecar missing: {_PDF_PATH.name}",
    )
elif not pdf_ready:
    _record("PDF report", status="SKIP", detail=pdf_reason)
else:
    pdf_reader = DocumentReader.create(
        _PDF_PATH,
        source_type=SourceType.RESEARCH,
        source_title="Financial Protection Review: Greece Summary",
        source_author="WHO Barcelona Office",
        source_date="2023-01-01",
        collection_id=_COLLECTION_ID,
    )
    _record(
        "PDF report",
        status="OK",
        documents=list(pdf_reader.get_documents()),
    )


# %%
# Source 4 — image OCR
# --------------------
# OCR is optional.  Missing Pillow/pytesseract/Tesseract is a bounded skip.
# After a successful preflight, unexpected OCR errors are allowed to surface.

ocr_ready, ocr_reason = _ocr_ready()

if not _IMAGE_PATH.exists():
    _record(
        "image OCR",
        status="SKIP",
        detail=f"local sidecar missing: {_IMAGE_PATH.name}",
    )
elif not ocr_ready:
    _record("image OCR", status="SKIP", detail=ocr_reason)
else:
    image_reader = DocumentReader.create(
        _IMAGE_PATH,
        source_type=SourceType.IMAGE,
        source_title="WHO Greece Report — Page 1 Scan",
        collection_id=_COLLECTION_ID,
        backend="tesseract",
        preprocess_grayscale=True,
    )
    _record(
        "image OCR",
        status="OK",
        documents=list(image_reader.get_documents()),
    )


# %%
# Source 5 — optional audio ASR
# -----------------------------
# Audio transcription is intentionally opt-in in this broad example.  The
# dedicated audio gallery demonstrates the offline companion-transcript path.
#
# To execute Whisper manually:
#
# .. code-block:: bash
#
#    SCIKITPLOT_GALLERY_RUN_ASR=1 python plot_corpus_who_per_file_script.py

whisper_ready, whisper_reason = _whisper_ready()

if not _AUDIO_PATH.exists():
    _record(
        "audio ASR",
        status="SKIP",
        detail=f"local sidecar missing: {_AUDIO_PATH.name}",
    )
elif not _RUN_ASR:
    _record(
        "audio ASR",
        status="SKIP",
        detail="optional ASR disabled; set SCIKITPLOT_GALLERY_RUN_ASR=1",
    )
elif not whisper_ready:
    _record("audio ASR", status="SKIP", detail=whisper_reason)
else:
    audio_reader = DocumentReader.create(
        _AUDIO_PATH,
        source_type=SourceType.AUDIO,
        source_title="Can people afford to pay for health care? (podcast)",
        collection_id=_COLLECTION_ID,
        transcribe=True,
        whisper_model="base",
    )
    _record(
        "audio ASR",
        status="OK",
        documents=list(audio_reader.get_documents()),
    )


# %%
# Ingestion summary
# -----------------
# This is the key partial-success view: every source keeps its own status.

print("\nSource outcome summary")
print("-" * 92)
print(f"{'source':28s} {'status':8s} {'docs':>6s}  detail")
print("-" * 92)

for entry in source_log:
    print(
        f"{str(entry['source']):28.28s} "
        f"{str(entry['status']):8s} "
        f"{int(entry['documents']):>6d}  "
        f"{entry['detail']}"
    )

print(f"\nTotal retained documents: {len(all_documents)}")

source_types = Counter(str(doc.source_type) for doc in all_documents)
if source_types:
    print("Documents by source type:")
    for source_type, count in sorted(source_types.items()):
        print(f"  {source_type:12s} {count:4d}")


# %%
# Phase 2 — normalize
# -------------------
# Normalization creates a stable text representation while retaining the
# original evidence in ``CorpusDocument.text``.

banner("PHASE 2 — NORMALIZE")

normalizer = NormalizationPipeline(
    [
        UnicodeNormalizer(),
        WhitespaceNormalizer(),
    ]
)

all_documents = normalizer.normalize_batch(all_documents)
n_normalised = sum(
    1 for doc in all_documents if doc.normalized_text is not None
)

print(f"Normalized: {n_normalised}/{len(all_documents)}")
if all_documents:
    show_doc(all_documents[0])


# %%
# Phase 3 — dependency-free lexical enrichment
# ---------------------------------------------
# SIMPLE tokenization + frequency keywords are enough for the BM25 demonstration.
# Stopword removal is disabled so this phase does not require NLTK data.

banner("PHASE 3 — ENRICH FOR LEXICAL RETRIEVAL")

enricher = NLPEnricher(
    EnricherConfig(
        tokenizer="simple",
        keyword_extractor="frequency",
        max_keywords=15,
        remove_stopwords=False,
        min_token_length=3,
    )
)

all_documents = enricher.enrich_documents(all_documents)
n_enriched = sum(1 for doc in all_documents if doc.tokens)

print(f"Enriched: {n_enriched}/{len(all_documents)}")
if all_documents:
    show_doc(all_documents[0])


# %%
# Phase 4 — build a keyword index
# -------------------------------
# Dense embeddings are intentionally not required in this broad integration
# example.  Keyword retrieval keeps the executed path portable.

banner("PHASE 4 — BUILD KEYWORD INDEX")

index: RetrievalIndex | None = None

if not all_documents:
    print("[SKIP] Index build: no source produced documents.")
else:
    index = RetrievalIndex(
        config=RetrievalConfig(
            match_mode="keyword",
            top_k=5,
        )
    )
    index.build(all_documents)

    print(f"Indexed documents: {index.n_documents}")
    print(f"Dense embeddings present: {index.has_embeddings}")


# %%
# Phase 5 — search one index under two policies
# ---------------------------------------------
# ``keyword`` ranks lexical relevance.  ``strict`` performs exact-text matching.
# Search policy can change without rebuilding this lexical corpus.

banner("PHASE 5 — SEARCH")

queries = [
    ("catastrophic health spending Greece", "keyword"),
    ("out-of-pocket payments medicines", "keyword"),
    ("dental care", "strict"),
]

if index is None:
    print("[SKIP] Search: no index was built.")
else:
    for query, mode in queries:
        print(f"\n{mode.upper()}: {query!r}")

        response = index.search(
            query,
            config=RetrievalConfig(
                match_mode=mode,
                top_k=3,
            ),
        )

        print(f"status={response.status} hits={len(response)}")

        for rank, hit in enumerate(response, start=1):
            source = hit.doc.source_title or hit.doc.input_path
            preview = hit.doc.text[:100].replace("\n", " ")
            print(
                f"  [{rank}] score={hit.score:.4f} "
                f"source={source!r}"
            )
            print(f"      {preview!r}")


# %%
# Phase 6 — adapt by user goal
# ----------------------------
# The adapters all consume the same canonical CorpusDocument collection.  They
# are grouped here by *why* a user would choose them rather than presented as a
# flat list of unrelated conversions.

banner("PHASE 6 — ADAPT FOR DOWNSTREAM CONSUMERS")


# %%
# 6A. Agent/framework handoff — LangChain and LangGraph
# -----------------------------------------------------
# These adapters are dependency-soft by design: when LangChain is unavailable,
# equivalent plain-Python structures are returned.

print("\nAgent/framework handoff")

lc_docs = to_langchain_documents(all_documents[:3])
print(f"LangChain-compatible documents: {len(lc_docs)}")
if lc_docs:
    first = lc_docs[0]
    if isinstance(first, dict):
        print("  representation: plain dict fallback")
        print(f"  keys: {sorted(first)}")
    else:
        print(f"  representation: {type(first).__name__}")

langgraph_state = to_langgraph_state(
    all_documents[:3],
    query="catastrophic health spending",
    match_mode="keyword",
)

print(f"LangGraph state keys: {sorted(langgraph_state)}")
print(f"LangGraph results: {langgraph_state['n_results']}")


# %%
# 6B. Protocol handoff — MCP
# --------------------------
# MCP resources expose addressable evidence.  The server adapter reuses the
# same RetrievalIndex when one is available.

print("\nProtocol handoff — MCP")

mcp_resources = to_mcp_resources(all_documents[:3])
print(f"MCP resources: {len(mcp_resources)}")
for resource in mcp_resources[:2]:
    print(f"  {resource['uri']} → {resource['text'][:70]!r}")

mcp_tool = to_mcp_tool_result(all_documents[:3])
print(f"MCP tool content items: {len(mcp_tool['content'])}")
print(f"MCP tool isError: {mcp_tool['isError']}")

if index is None:
    print("[SKIP] MCP search server: no index was built.")
else:
    mcp_server = MCPCorpusServer(
        index=index,
        server_name="who-corpus",
    )
    tool_names = [tool["name"] for tool in mcp_server.list_tools()]
    print(f"MCP server tools: {tool_names}")


# %%
# 6C. Data/RAG interchange — HuggingFace, RAG tuples, JSONL
# ----------------------------------------------------------

print("\nData and RAG interchange")

hf_data = to_huggingface_dataset(all_documents[:3])
if isinstance(hf_data, dict):
    print(f"HuggingFace fallback columns: {sorted(hf_data)}")
    print(f"HuggingFace fallback rows: {len(hf_data.get('text', []))}")
else:
    print(f"HuggingFace representation: {type(hf_data).__name__}")
    print(f"HuggingFace rows: {len(hf_data)}")

rag_rows = to_rag_tuples(all_documents[:3])
print(f"RAG tuples: {len(rag_rows)}")

jsonl_rows = list(to_jsonl(all_documents[:3]))
print(f"JSONL rows: {len(jsonl_rows)}")
if jsonl_rows:
    first_json = json.loads(jsonl_rows[0])
    print(f"JSONL first text: {first_json['text'][:80]!r}")


# %%
# Where CorpusBuilder fits
# ------------------------
# This file deliberately exposes each stage.  Most applications do not need to
# keep those variables synchronized manually.
#
# ``CorpusBuilder`` is the higher-level choice when the goal is:
#
# ``many sources → partial success → normalize/enrich/index → adapters``.
#
# Creating the configuration below performs no source I/O and needs no model.

builder_config = BuilderConfig(
    chunker=None,
    normalize=True,
    normalizer_steps=["unicode", "whitespace"],
    enrich=True,
    enricher_kwargs={
        "tokenizer": "simple",
        "keyword_extractor": "frequency",
        "remove_stopwords": False,
        "max_keywords": 15,
    },
    embed=False,
    build_index=True,
    retrieval_config=RetrievalConfig(
        match_mode="keyword",
        top_k=5,
    ),
    collection_id=_COLLECTION_ID,
)

builder = CorpusBuilder(builder_config)

print("\nHigher-level alternative:")
print("  CorpusBuilder configured:", type(builder).__name__)
print("  embeddings enabled:", builder.config.embed)
print("  index enabled:", builder.config.build_index)

# A real application can then call:
#
# .. code-block:: python
#
#    result = builder.build(local_sources)
#    print(result.summary())
#    print(result.errors)      # structured per-source diagnostics
#    hits = builder.search("catastrophic health spending")
#
# ``BuildResult.errors`` is the structured partial-success surface.


# %%
# Where FluentCorpus / RuntimeCorpus fits
# ---------------------------------------
# Use ``FluentCorpus`` when configuration itself needs to be immutable,
# inspectable, branchable, and fingerprinted; materialize it into
# ``RuntimeCorpus`` when that plan becomes operational.
#
# The current RuntimeCorpus sequence path is intentionally fail-fast for a
# coherent runtime generation.  Therefore this broad “some sources may fail,
# keep the rest” scenario is a particularly natural fit for ``CorpusBuilder``
# or explicit source-by-source orchestration.


# %%
# Live source usage — explicit, code-only
# ---------------------------------------
# Network ingestion is intentionally separated from missing-sidecar handling.
# Enable it in an application only when the network and remote-service
# dependency are part of the intended runtime.
#
# .. code-block:: python
#
#    web_docs = list(DocumentReader.from_url(WHO_ARTICLE_URL).get_documents())
#    video_docs = list(DocumentReader.from_url(WHO_VIDEO_URL).get_documents())
#    pdf_docs = list(
#        DocumentReader.from_url(
#            WHO_PDF_URL,
#            source_type=SourceType.RESEARCH,
#        ).get_documents()
#    )
#
# URL validation, redirects, download limits, and SSRF policy remain owned by
# the Corpus URL/reader layer; this gallery does not bypass them.


# %%
# Final summary
# -------------

banner("SUMMARY")

n_ok = sum(1 for entry in source_log if entry["status"] == "OK")
n_skipped = sum(1 for entry in source_log if entry["status"] == "SKIP")

print(f"Sources represented:  {len(source_log)}")
print(f"Sources succeeded:    {n_ok}")
print(f"Sources skipped:      {n_skipped}")
print(f"Documents retained:   {len(all_documents)}")
print(f"Normalized:           {n_normalised}")
print(f"Enriched:             {n_enriched}")
print(f"Index documents:      {index.n_documents if index is not None else 0}")
print(
    "Dense embeddings:     "
    f"{index.has_embeddings if index is not None else False}"
)
print("Network requests:     0 in the normal gallery path")


# %%
# Takeaway
# --------
# Keep source failures observable and keep successful evidence.
#
# Use:
#
# * explicit components when teaching/debugging stage boundaries,
# * ``CorpusBuilder`` for broad heterogeneous partial-success workflows,
# * ``FluentCorpus`` + ``RuntimeCorpus`` for reusable declarative runtime plans.


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: intermediate
#    purpose: showcase
