..
  docs/source/user_guide/corpus/index.rst

.. currentmodule:: scikitplot.corpus

.. _corpus-index:

======================================================================
Corpus User Guide
======================================================================

:mod:`scikitplot.corpus` turns local files, URLs, archives and media into
canonical :class:`CorpusDocument` evidence, then lets you normalize, enrich,
embed, store, retrieve and export that evidence without requiring one monolithic
runtime.

The important boundary is that configuration, capability discovery and
execution are separate.  A package can be installed without its model or system
binary being ready; a backend can be selected without becoming active; and a
fail-soft operation can continue while still recording structured degradation.

.. important::

   Corpus treats optional capability failure as data, not as success.  Use
   :func:`component_capabilities` before a run when readiness matters and inspect
   ``reader.backend_reports`` after a reader run when a fallback was possible.
   ``strict=False`` may keep ingestion alive; it does not mean failures are
   invisible.

The whole idea in one picture
-----------------------------

.. code-block:: text

   source
      |
      v
   downloader / local path
      |
      v
   DocumentReader ---- capability/readiness ---- optional backends
      |                                      \
      |                                       +-- fallback policy + reports
      v
   filter -> normalize -> chunk -> enrich -> embed -> store -> index
                                                     |
                                                     v
                                          retrieve / adapt / export

There are three complementary configuration levels:

``CorpusPipeline``
    Direct control of one execution pipeline.

``CorpusBuilder``
    High-level heterogeneous ingestion, downloading, search and adaptation.

``FluentCorpus``
    Immutable declarative plans that can be generated, branched, compared and
    materialized explicitly.

Thirty seconds
--------------

.. prompt:: python >>>

   from scikitplot.corpus import FluentCorpus

   plan = FluentCorpus.from_config({
       "chunker": "paragraph",
       "storage": "memory",
   })
   plan.explain()["configured"]
   # ['chunker', 'storage']

Inspect optional component readiness without importing heavy models:

.. prompt:: python >>>

   from scikitplot.corpus import component_capabilities

   status = component_capabilities(["asr:faster-whisper", "ocr:pytesseract"])
   status["asr:faster-whisper"]["installed"]
   # True or False depending on the environment

Which surface should I use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Goal
     - Start with
   * - Process one source with explicit stages
     - :class:`CorpusPipeline`
   * - Ingest heterogeneous local/remote sources
     - :class:`CorpusBuilder`
   * - Generate reusable configuration variants
     - :class:`FluentCorpus` (:ref:`corpus-fluent-policies`)
   * - Supply a custom reader/filter/downloader/component
     - :class:`CorpusBuilder` with :class:`BuilderFactories`
       (:ref:`corpus-customization`)
   * - Inspect optional package/model/binary readiness
     - :func:`component_capabilities` (:ref:`corpus-readers-backends`)
   * - Preview the exact ASR chain without running models
     - :meth:`AudioReader.plan_asr_backends` / :class:`BackendPlan`
   * - Control backend order/fallback/offline behavior
     - :class:`BackendPolicy`
   * - Reuse one typed policy family across runtime/reader/builder seams
     - :class:`CorpusPolicyBundle` (:ref:`corpus-fluent-policies`)
   * - Control transfer TLS/SSRF/size/retry budgets
     - :class:`DownloadPolicy` (:ref:`corpus-downloads-network`)
   * - Add a private/local ASR implementation
     - :class:`ASRBackend`
   * - Download HTTP/GDrive/GitHub/YouTube inputs
     - :class:`AnyDownloader` (:ref:`corpus-downloads-network`)
   * - Do fuzzy lexical ranking without requiring Corpus
     - :mod:`scikitplot.levenshtein` (:ref:`corpus-retrieval-similarity`)
   * - Search vectors/lexical/hybrid indexes
     - :class:`RetrievalIndex`
   * - Export/adapt documents
     - the export and adapter APIs (:ref:`corpus-formats-export`)

How this guide is organised
---------------------------

Read the first two pages first. After that, choose the page for the task in
front of you.

..
  .. toctree::
    :maxdepth: 2

    getting_started
    architecture
    readers_and_backends
    fluent_and_policies
    customization
    downloads_and_network
    retrieval_and_similarity
    formats_and_export
    security_and_limits
    troubleshooting

.. grid:: 1 1 1 1

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **getting-started**
      ^^^
      .. toctree::
         :maxdepth: 2

        getting_started

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **architect**
      ^^^
      .. toctree::
         :maxdepth: 2

        architecture

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **reader**
      ^^^
      .. toctree::
         :maxdepth: 2

        readers_and_backends

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **fluent**
      ^^^
      .. toctree::
         :maxdepth: 2

        fluent_and_policies

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **customize**
      ^^^
      .. toctree::
         :maxdepth: 2

        customization

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **download**
      ^^^
      .. toctree::
         :maxdepth: 2

        downloads_and_network

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **retrieval**
      ^^^
      .. toctree::
         :maxdepth: 2

        retrieval_and_similarity

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **format**
      ^^^
      .. toctree::
         :maxdepth: 2

        formats_and_export

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **security**
      ^^^
      .. toctree::
         :maxdepth: 2

        security_and_limits

   .. grid-item-card::
      :columns: 12 12 6 6
      :padding: 2

      **troubleshoot**
      ^^^
      .. toctree::
         :maxdepth: 2

        troubleshooting

The executable gallery, :ref:`corpus_examples`, follows the same architecture
with deterministic local examples first and optional live/model paths clearly
marked.

.. seealso::

   * :ref:`corpus_examples`
   * :ref:`cleanprompt-index`
   * :mod:`scikitplot.corpus`
