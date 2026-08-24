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

Start here
----------

1. **Configure Corpus declaratively** — learn :class:`FluentCorpus`, immutable
   plans, validation, branching, fingerprints, and the ``materialize()``
   boundary.
2. **Build and search a real Hamlet corpus** — use
   :class:`RuntimeCorpus` end to end: ``run()``, ``add()``, storage, retrieval,
   export, and lifecycle.
3. **Compare chunking strategies** — compare sentence, word, fixed-window, and
   morphological semantic chunking on the same OCR text.
4. **Process an MP3** — learn audio provenance and companion-transcript
   precedence without requiring Whisper in the normal gallery path.
5. **Process a mixed-media ZIP** — inspect archive-member routing,
   ``archive.zip/member.ext`` provenance, and per-extension reader settings.
6. **Process a YouTube transcript** — execute a deterministic local proxy,
   configure the real YouTube reader, and keep the live transcript request
   explicit and optional.
7. **Build a multi-source WHO corpus** — see the explicit stage-by-stage
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

``invalid public API / security-policy failure / installed-backend defect``
    Fail visibly. The gallery must not convert a real regression into a skip.

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
