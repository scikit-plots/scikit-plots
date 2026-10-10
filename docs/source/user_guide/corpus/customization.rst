.. currentmodule:: scikitplot.corpus

.. _corpus-customization:

Customization
=============

Corpus has multiple extension layers. Choose the narrowest one that solves the
problem; do not replace the whole builder when a single stage is custom.

Reader-level extractor
----------------------

``DocumentReader.custom_extractor`` is the lowest-friction escape hatch for a
single reader instance. Common invocation, kwargs forwarding, output
normalization and exception wrapping are centralized by :class:`DocumentReader`.

Builder factories
-----------------

:class:`BuilderFactories` is the construction policy used directly by
:class:`CorpusBuilder`. :class:`FactoryCorpusBuilder` remains a compatibility
facade for code that already uses it. The native builder seam is preferred for
new code because every ingestion branch consults the same factory object.

Factories are available for:

* ``reader_factory``
* ``chunker_factory``
* ``filter_factory``
* ``downloader_factory``
* ``normalizer_factory``
* ``enricher_factory``
* ``embedding_engine_factory``

Reader/filter/downloader factories apply through the builder's central
``_make_*``/``_get_*`` seams, so archive members and downloaded files do not
bypass them.

.. code-block:: python

   from scikitplot.corpus import BuilderFactories, CorpusBuilder

   def reader_factory(source, *, chunker=None, **kwargs):
       from scikitplot.corpus import DocumentReader
       return DocumentReader.create(source, chunker=chunker, **kwargs)

   builder = CorpusBuilder(
       factories=BuilderFactories(reader_factory=reader_factory)
   )

The factory container is validated at construction time, while the factories
themselves stay lazy. This keeps ordinary :class:`CorpusBuilder` construction
free from optional model imports. Reader, filter and downloader factories are
used for local files, downloaded files and archive members rather than being
patched onto one call path after construction.

Custom downloader
-----------------

Use :class:`CustomDownloader` when the transfer itself is special, or provide a
``downloader_factory`` when the builder should own a custom download policy.
The downloader result is still a :class:`DownloadResult`, keeping download and
reader dispatch separate.

Custom retrieval
----------------

:class:`CustomRetrievalIndex` accepts a scorer callback. The independent
:mod:`scikitplot.levenshtein` facade provides :func:`scikitplot.levenshtein.make_corpus_scorer`
for edit-distance ranking without coupling the Levenshtein module back to
Corpus.

Custom registries
-----------------

:class:`CapabilityRegistry` is an explicit object. Copy the default registry or
build a private one and register :class:`CapabilitySpec` objects. Registration
refuses accidental replacement unless ``replace=True`` is explicit.
