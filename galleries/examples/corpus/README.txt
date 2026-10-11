.. _corpus_examples:

Corpus
===============

Examples for :py:mod:`~scikitplot.corpus` are ordered as a learning path rather
than by implementation detail.

.. prompt:: bash $

    # 💡 corpus Need additionals packages
    curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/corpus.txt
    pip install -r requirements/corpus.txt
    pip install scikit-plots[corpus]

    # (Recommended)
    # !pip install datasets transformers
    # !pip install nltk gensim langdetect faster-whisper openai-whisper pytesseract youtube-transcript-api
    # sudo apt-get install tesseract-ocr

.. seealso::
  * https://github.com/modelcontextprotocol/python-sdk
  * https://github.com/semantica-agi/semantica
  * https://docs.getsemantica.ai/guides/distance-intelligence/#common-pitfalls

Start here
----------

1. **Configure Corpus declaratively** — learn :class:`FluentCorpus`, immutable
   plans, validation, branching, fingerprints, bounded lazy configuration
   variants, and the ``materialize()`` boundary.
2. **Compose policy families explicitly** — use :class:`CorpusPolicyBundle`
   for reusable local/strict/networked/docs presets without collapsing runtime,
   backend, downloader, and per-document error semantics.
3. **Customize optional backends safely** — preflight the exact reader chain,
   inspect readiness, order backends with :class:`BackendPolicy`, and plug in a
   user-side :class:`ASRBackend` without model downloads.
4. **Use lexical edit distance** — use :mod:`scikitplot.levenshtein` directly
   and as a deterministic Corpus retrieval scorer.
5. **Build and search a real Hamlet corpus** — use
   :class:`RuntimeCorpus` end to end: ``run()``, ``add()``, storage, retrieval,
   export, and lifecycle.
6. **Compare chunking strategies** — compare sentence, word, fixed-window, and
   morphological semantic chunking on the same OCR text.
7. **Process an MP3** — learn audio provenance and companion-transcript
   precedence without requiring Whisper in the normal gallery path.
8. **Process a mixed-media ZIP** — inspect archive-member routing,
   ``archive.zip/member.ext`` provenance, and per-extension reader settings.
9. **Process a YouTube transcript** — execute a deterministic local proxy,
   configure the real YouTube reader, and keep the live transcript request
   explicit and optional.
10. **Build a multi-source WHO corpus** — see the explicit stage-by-stage
   integration path, partial source success, keyword retrieval, adapters, and
   where :class:`CorpusBuilder` fits.

Which API should I use?
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Goal
     - Start with
   * - Process one source with direct stage control
     - :class:`CorpusPipeline`
   * - Build/search heterogeneous sources with partial-success reporting
     - :class:`CorpusBuilder`
   * - Create immutable, reusable, branchable configuration
     - :class:`FluentCorpus`
   * - Execute a Fluent plan and manage runtime state/lifecycle
     - :class:`RuntimeCorpus`
   * - Extend vector indexing/retrieval directly
     - :class:`RetrievalIndex` / :class:`VectorIndexBackend`

Capability matrix
-----------------

The normal gallery path prefers deterministic local execution. Optional
capabilities are either preflighted and skipped when unavailable, or shown as
configuration-only examples.

.. list-table::
   :header-rows: 1
   :widths: 35 18 18 29

   * - Example
     - Normal path
     - Optional capability
     - Behavior when unavailable
   * - FluentCorpus basics
     - local/core
     - none
     - not applicable
   * - Policy bundle
     - local/core
     - none
     - composes existing policies; performs no I/O by itself
   * - Backend policy and custom ASR
     - local/core
     - none; capability probes only
     - preflight and execution use the same candidate chain; no model is loaded
       or downloaded by preflight
   * - Levenshtein retrieval
     - bundled/pure Python
     - RapidFuzz acceleration
     - facade remains importable through safe fallback
   * - Hamlet RuntimeCorpus
     - local/core + NumPy
     - native Annoy branch
     - configuration only; not built
   * - OCR chunking comparison
     - local image
     - Tesseract, NLTK
     - explicit ``SKIP`` for unavailable capability
   * - MP3 ingestion
     - MP3 + local SRT companion
     - NLTK, Whisper
     - optional sections ``SKIP``
   * - Mixed-media ZIP
     - local archive
     - PDF/OCR/Whisper readers
     - individual optional member capability may produce no documents;
       archive-security failures still fail
   * - YouTube transcript
     - local synthetic proxy
     - youtube-transcript-api + network, NLTK
     - live/optional sections ``SKIP``
   * - WHO multi-source integration
     - local sidecars only
     - PDF/OCR/Whisper
     - each unavailable source reports ``SKIP``; successful evidence remains

Gallery reliability rule
------------------------

The examples distinguish optional capability absence from real defects:

``missing optional package/resource/native capability/network opt-in``
    Report a visible, specific ``SKIP`` and continue when the example can
    remain truthful.

``invalid public API / security-policy failure``
    Fail visibly. The gallery must not convert these into a skip.

``optional ASR backend defect``
    Remain observable. Audio/video readers warn and try the next Whisper backend;
    their default ``strict=False`` policy yields no ASR chunks only after the
    fallback chain is exhausted. ``reader.backend_reports`` keeps a structured,
    JSON-compatible ``degraded``/``failed`` record for programmatic inspection.
    Use ``strict=True`` for fail-fast validation.

A missing local sidecar never silently enables public-network access.

Install only what you need
--------------------------

The core text/runtime examples use the normal Corpus installation. Media and
NLP examples may additionally use packages such as NLTK, an OCR backend,
Whisper, or ``youtube-transcript-api``. System tools such as Tesseract may also
be required for the corresponding optional path.

Do not install every optional dependency merely to read the gallery. The
portable path is designed to remain useful when those capabilities are absent.

Browser / WASM note
-------------------

Declarative configuration, local text processing, and portable brute-force
retrieval are the strongest browser/WASM candidates. OCR, Whisper, native ANN
backends, and live external services depend on the actual JupyterLite/xeus
runtime and should not be assumed available until verified in that target
environment.
