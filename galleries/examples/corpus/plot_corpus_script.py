"""
corpus with examples
====================

.. currentmodule:: scikitplot.corpus

Examples related to the :py:mod:`~scikitplot.corpus` submodule.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Download the related packages
# #############################
#
# First we download the media preproccess libraries (text, image, audio or video).
# pip install nltk gensim langdetect faster-whisper openai-whisper pytesseract youtube-transcript-api
# sudo apt-get install tesseract-ocr
# pip install scikit-plots[corpus]

# .. seealso::
#    * galleries/examples/00-jupyter_notebooks/corpus/plot_corpus_from_any_media_notebook.ipynb

# import faster_whisper
# import whisper  # openai-whisper
# import youtube_transcript_api

# %%

import os
import json
import sys
import textwrap
from pathlib import Path

import scikitplot as sp
from scikitplot import corpus

from scikitplot.corpus._schema import (
    CorpusDocument,
    SourceType,
    MatchMode,
    SectionType,
    ChunkingStrategy,
    _PROMOTED_RAW_KEYS,
)
from scikitplot.corpus._base import DocumentReader, DefaultFilter
from scikitplot.corpus._normalizers import (
    NormalizationPipeline,
    UnicodeNormalizer,
    WhitespaceNormalizer,
)
from scikitplot.corpus._enrichers._nlp_enricher import NLPEnricher, EnricherConfig
from scikitplot.corpus._similarity._similarity import (
    SimilarityIndex,
    SearchConfig,
    SearchResult,
)
from scikitplot.corpus._adapters import (
    to_langchain_documents,
    to_langgraph_state,
    to_mcp_resources,
    to_mcp_tool_result,
    to_huggingface_dataset,
    to_rag_tuples,
    to_jsonl,
    MCPCorpusServer,
)

# %%

# ===========================================================================
# HELPER: print section banners
# ===========================================================================

def banner(title: str, char: str = "=") -> None:
    line = char * 72
    print(f"\n{line}\n  {title}\n{line}\n")

def mini_banner(title: str) -> None:
    print(f"\n  --- {title} ---\n")

def show_doc(doc: CorpusDocument, index: int = 0) -> None:
    """Pretty-print a single CorpusDocument."""
    text_preview = doc.text[:100].replace("\n", " ")
    norm_preview = (doc.normalized_text or "")[:80].replace("\n", " ")
    print(f"  [{index:3d}] doc_id={doc.doc_id[:12]}…  source_type={doc.source_type}")
    print(f"        text: {text_preview!r}…")
    if norm_preview:
        print(f"        norm: {norm_preview!r}…")
    if doc.tokens:
        print(f"        tokens({len(doc.tokens)}): {doc.tokens[:8]}…")
    if doc.keywords:
        print(f"        keywords: {doc.keywords[:6]}")
    if doc.timecode_start is not None:
        print(f"        timecode: {doc.timecode_start:.1f}s – {doc.timecode_end:.1f}s")
    if doc.confidence is not None:
        print(f"        confidence: {doc.confidence:.3f}")
    if doc.page_number is not None:
        print(f"        page: {doc.page_number}")

# %%

# ===========================================================================
# PHASE 1: INGEST — Process all 5 source types via DocumentReader
# ===========================================================================

banner("PHASE 1: INGEST — 5 Source Types via DocumentReader")

all_documents: list[CorpusDocument] = []
source_log: list[dict] = []

# --- Source ①: Web Article (HTML) ---
# In production: DocumentReader.from_url("https://www.who.int/europe/news/item/...")
# Here we use the local text proxy (same content)
mini_banner("Source ①: Web Article (text proxy for HTML URL)")
try:
    if Path("who_health_care_article.txt").exists():
        # from local raw html page
        reader = DocumentReader.create(
            Path("who_health_care_article.txt"),
            source_type=SourceType.WEB,
            source_title="Out-of-pocket payments for health care unaffordable for millions in Europe",
            source_author="WHO Regional Office for Europe",
            source_date="2023-12-12",
            collection_id="who-greece-financial-protection",
        )
    else:
        # from url raw html page DocumentReader.from_url for raw html
        reader = DocumentReader.from_url(
            "https://www.who.int/europe/news/item/12-12-2023-out-of-pocket-payments-for-primary-health-care-unaffordable-for-millions-in-europe-new-who-report-shows",
        )
    docs = list(reader.get_documents())
    all_documents.extend(docs)
    source_log.append({"type": "web_article", "n_docs": len(docs), "status": "OK"})
    print(f"  ✓ Web article: {len(docs)} chunks ingested")
    if docs:
        show_doc(docs[0], 0)
except Exception as e:
    source_log.append({"type": "web_article", "n_docs": 0, "status": f"ERROR: {e}"})
    print(f"  ✗ Web article: {e}")

# %%

docs[60].text

# %%

all_documents[60].text

# %%

# --- Ingestion Summary ---
mini_banner("Ingestion Summary")
for entry in source_log:
    status = "✓" if entry["status"] == "OK" else "⚠"
    print(f"  {status} {entry['type']:25s} → {entry['n_docs']:3d} docs  [{entry['status']}]")
print(f"\n  Total documents in corpus: {len(all_documents)}")

# %%

# ===========================================================================
# PHASE 4: INDEX — Build SimilarityIndex (KEYWORD mode, no embeddings)
# ===========================================================================

banner("PHASE 4: INDEX — Build SimilarityIndex (BM25 keyword mode)")

index = SimilarityIndex(config=SearchConfig(match_mode="keyword", top_k=5))
index.build(all_documents)
print(f"  ✓ Index built: {index.n_documents} documents, dense={index.has_embeddings}")

# %%

# ===========================================================================
# PHASE 6: ADAPTERS — Export to every downstream format
# ===========================================================================

banner("PHASE 6: ADAPTERS — Export to LangChain / MCP / RAG / LangGraph / HF")

# --- 6a: LangChain Documents ---
mini_banner("6a: LangChain Documents")
lc_docs = to_langchain_documents(all_documents)
first = lc_docs[0]
if isinstance(first, dict):
    print(f"  ✓ {len(lc_docs)} LangChain docs (dict fallback — langchain not installed)")
    print(f"    keys: {list(first.keys())}")
    print(f"    page_content[:80]: {first['page_content'][:80]!r}")
    print(f"    metadata keys: {sorted(first['metadata'].keys())[:10]}")
else:
    print(f"  ✓ {len(lc_docs)} LangChain Document objects")

# --- 6b: LangGraph State ---
mini_banner("6b: LangGraph State")
state = to_langgraph_state(
    all_documents,
    query="catastrophic health spending",
    match_mode="keyword",
)
print(f"  ✓ LangGraph state dict:")
print(f"    keys: {sorted(state.keys())}")
print(f"    n_results: {state['n_results']}")
print(f"    query: {state['query']!r}")

# --- 6c: MCP Resources ---
mini_banner("6c: MCP Resources (Model Context Protocol)")
resources = to_mcp_resources(all_documents[:3])
for r in resources[:2]:
    print(f"  resource:")
    print(f"    uri:      {r['uri']}")
    print(f"    name:     {r['name']}")
    print(f"    mimeType: {r['mimeType']}")
    print(f"    text[:60]: {r['text'][:60]!r}…")

# --- 6d: MCP Tool Result ---
mini_banner("6d: MCP Tool Result (tools/call response)")
tool_result = to_mcp_tool_result(all_documents[:3])
print(f"  ✓ MCP tool result:")
print(f"    isError: {tool_result['isError']}")
print(f"    content items: {len(tool_result['content'])}")
for item in tool_result["content"][:2]:
    print(f"    [{item['type']}] text[:60]: {item['text'][:60]!r}…")
    print(f"         annotations: {item['annotations']}")

# --- 6e: MCP Server (adapter class) ---
mini_banner("6e: MCP Server Adapter")
mcp_server = MCPCorpusServer(index=index, server_name="who-corpus")
tools = mcp_server.list_tools()
print(f"  ✓ MCPCorpusServer: {mcp_server}")
print(f"    tools: {[t['name'] for t in tools]}")
print(f"    tool schema: {json.dumps(tools[0]['inputSchema'], indent=6)[:200]}…")

# --- 6f: HuggingFace Dataset ---
mini_banner("6f: HuggingFace Dataset")
hf = to_huggingface_dataset(all_documents)
if isinstance(hf, dict):
    print(f"  ✓ HuggingFace column dict (datasets lib not installed)")
    print(f"    columns: {sorted(hf.keys())}")
    print(f"    rows: {len(hf['text'])}")
else:
    print(f"  ✓ HuggingFace Dataset: {hf}")

# --- 6g: RAG Tuples ---
mini_banner("6g: RAG Tuples (text, metadata, embedding)")
tuples = to_rag_tuples(all_documents[:3])
for i, (text, meta, emb) in enumerate(tuples):
    print(f"  [{i}] text[:50]: {text[:50]!r}")
    print(f"      meta keys: {sorted(meta.keys())[:8]}")
    print(f"      embedding: {type(emb).__name__}")

# --- 6h: JSONL ---
mini_banner("6h: JSONL Streaming")
lines = list(to_jsonl(all_documents[:3]))
print(f"  ✓ {len(lines)} JSONL lines")
for i, line in enumerate(lines[:2]):
    obj = json.loads(line)
    print(f"  [{i}] keys: {sorted(obj.keys())[:8]}…  text[:50]: {obj['text'][:50]!r}")

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: corpus
#    plot-type: bar
#    level: beginner
#    purpose: showcase
