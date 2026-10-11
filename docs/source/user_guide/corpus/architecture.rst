.. currentmodule:: scikitplot.corpus

.. _corpus-architecture:

Architecture
============

Corpus separates mechanism from policy.

.. code-block:: text

   input
     |
     +-- local file -----------------------------+
     |                                           |
     +-- URL -> downloader -> DownloadResult ----+
                                                 v
                                          DocumentReader
                                                 |
                             +-------------------+--------------------+
                             |                                        |
                      built-in backend                         custom backend
                             |                                        |
                             +---- BackendPolicy / readiness ---------+
                                                 |
                                           raw chunks
                                                 |
                                  filter / normalize / chunk
                                                 |
                                  enrich / embed / store / index

The central seams
-----------------

``DocumentReader``
    Owns format dispatch, common document construction, custom extractor
    execution and per-run backend reports.

``CorpusBuilder._make_reader``
    One construction seam for local files, downloads and archive members.
    :class:`FactoryCorpusBuilder` can replace it through ``reader_factory``.

``CorpusBuilder._make_downloader``
    One construction seam for downloadable URLs. It delegates specialist
    routing to :class:`AnyDownloader`, consumes :class:`DownloadPolicy`, and can
    be replaced through ``downloader_factory``.

``plan_backend_chain`` / ``run_backend_chain``
    Share one selection algorithm. :class:`BackendPlan` exposes side-effect-free
    preflight ordering/readiness; runtime adds attempt/failure/active evidence.
    Readers still decide what a valid result means and what failure should
    ultimately do.

``CapabilityRegistry``
    Owns lightweight readiness facts. It does not load models or contact the
    network.

``CorpusPlan`` / ``FluentCorpus``
    Own immutable configuration identity, conflict detection and deterministic
    plan generation.

Why there is no mega fallback manager
-------------------------------------

Different operations have different semantics. A missing optional XML parser
can fall back to the standard library, but malformed XML should stay an error.
A Whisper backend can fail and another ASR implementation can be tried. An OCR
backend may download a model, so switching automatically can change network and
resource behavior. PDF extraction may have provisional/empty-page semantics
that are not ordinary failure.

The shared layer therefore centralizes *how* attempts are observed, while each
reader owns *when* fallback is correct.

Policy scopes stay orthogonal
-----------------------------

Corpus deliberately does not create one mega ``strict`` switch:

* :class:`RuntimePolicy` decides whether URL sources may execute;
* :class:`DownloadPolicy` owns transfer security/resource budgets;
* :class:`BackendPolicy` owns optional implementation selection/fallback;
* :class:`ErrorPolicy` owns per-document pipeline failure behavior.

Each layer has presets/config validation, but changing one does not silently
change the others. This is easier to reason about than a convenience mode whose
meaning varies by reader or transport.
