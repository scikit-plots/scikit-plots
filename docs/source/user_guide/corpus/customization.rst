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

Chunkers and normalisers by name
--------------------------------

``BuilderConfig(chunker="...")`` and ``normalizer_steps`` resolve names through
one table, the component ``registry`` (:class:`ComponentRegistry`). The
built-in chunkers are ``sentence``, ``paragraph``, ``fixed_window``, ``word``,
``semantic``, ``recursive``, ``parent_child`` and ``structure``, each
registered with its configuration dataclass;
``chunker_kwargs`` are that config's fields, plus any option the chunker's
constructor takes itself (``gensim_dictionary`` for ``word``), and an unknown
key raises ``TypeError`` naming the valid ones. ``normalizer_kwargs`` maps a
step name to that step's options. A chunker you register becomes reachable by
name (from the builder and from plans alike) and is never replaced by a
built-in of the same name; any class whose ``chunk`` takes ``extra_metadata``
runs in the pipeline (:class:`ProtocolChunkerBridge`). An unknown chunker name
or normaliser step raises ``ValueError`` when the build starts, before any
source is read.

.. code-block:: python

   from scikitplot.corpus import BuilderConfig, registry

   registry.register_builtins()
   registry.build_chunker("sentence", overlap=1).config.overlap  # 1
   # registry.register_chunker("mine", MyChunker, config=MyChunkerConfig)
   config = BuilderConfig(chunker="sentence", chunker_kwargs={"overlap": 1},
                          normalizer_steps=["unicode", "lowercase"],
                          normalizer_kwargs={"unicode": {"form": "NFKC"}})

An enum option of a chunker config accepts the member or its value
(``SentenceChunkerConfig(backend="regex")``); anything else raises when the
config is constructed. ``registry.snapshot()`` records each chunker's config
class, and ``ComponentRegistry.load_from_snapshot`` restores every built-in.

Chunk spans
-----------

A chunk's ``start_char``/``end_char`` are its *anchor*: the region it stands
for. Its ``context_start``/``context_end`` (``context_span``) are the region its
text is taken from, which covers the anchor and includes any overlap: for every
built-in chunker whose ``text_is_source_slice`` is ``True``,
``chunk.text == source[context_start:context_end]``. Overlap, merged and split
chunks are slices of the source, not parts re-joined with a space.
``WordChunker`` derives its text (tokens, case, stems), so its
``text_is_source_slice`` is ``False`` and its context span is the region the
text came from. A sentence splitter that returns text the source does not
contain raises when offsets are enabled. A document built from a chunk spans
the chunk's context (``char_start``/``char_end``, with ``raw_text`` that slice)
and records the anchor in ``metadata["anchor_span"]`` when the two differ.
An overlap chunk's context runs from its first overlapped unit to its anchor,
so it includes a unit ``min_length`` dropped in between.

Spans index the text the chunker split. When multilang preprocessing (NFC,
BOM and control-character stripping) changed it, the chunk result's
``metadata["span_basis"]`` is ``"preprocessed"`` instead of ``"input"``, every
document built from it records the same in ``metadata["span_basis"]``, and no
``raw_text`` is sliced with those spans. A document's ``raw_text`` is its
region of the reader's raw text, and ``None`` whenever that region is unknown:
text that preprocessing changed, or a reader ``raw_text`` that is not the
chunked text (HTML or XML markup) unless one chunk covers the whole of it. It
is never the whole parent for a chunk that covers only part of it.
``WordChunker`` reports offsets only with ``include_offsets=True``; without
them its ``raw_text`` is the segment it took its tokens from.

.. code-block:: python

   from scikitplot.corpus import SentenceChunker, SentenceChunkerConfig

   source = "Alpha one. Beta two."
   chunk = SentenceChunker(SentenceChunkerConfig(overlap=1, min_length=1)).chunk(source).chunks[1]
   chunk.text == source[slice(*chunk.context_span)]  # True: "Alpha one. Beta two."
   source[slice(*chunk.anchor_span)]                 # "Beta two."

Recursive and parent-child chunking
-----------------------------------

:class:`RecursiveChunker` cuts a text by a hierarchy of separators (blank
line, line, sentence end, space, character) only as far as needed for each
piece to fit ``chunk_size`` characters (not counting the whitespace around
it). :class:`ParentChildChunker` applies the same rule at two sizes: parents
that give context, and children inside each parent that are retrieved. Each
parent is followed by its children; a document built from a child carries
``parent_doc_id``, its parent's ``doc_id``, so a hit on a child can return the
parent (:func:`children` and :func:`ancestors` traverse them). Neither is a
default.

.. code-block:: python

   from scikitplot.corpus import BuilderConfig, CorpusBuilder

   config = BuilderConfig(chunker="parent_child",
                          chunker_kwargs={"parent_size": 1500, "child_size": 200})
   docs = CorpusBuilder(config).build("guide.txt").documents
   children = [d for d in docs if d.metadata["chunk_role"] == "child"]

Measured on the pages of this user guide as they were before this section was
added (82 one-paragraph sentences as queries, BM25 over the retrievable units;
for ``parent_child`` the children are indexed and a hit returns its parent),
a hit counts when the returned context covers the whole sentence:

==========================  =====  ========  ===================  =========================
Chunker                     Units  Recall@1  Context chars (top)  Paragraph coverage (top)
==========================  =====  ========  ===================  =========================
sentence                    303    0.96      384                  0.49
paragraph                   540    1.00      504                  1.00
fixed_window (500 / 400)    228    0.95      485                  0.76
recursive (500)             235    0.95      422                  0.84
recursive (200)             610    0.59      150                  0.46
parent_child (1500 / 200)   614    1.00      1263                 0.98
structure (rst, 500)        301    0.95      380                  0.84
==========================  =====  ========  ===================  =========================

The queries are verbatim sentences, so this measures whether a chunker keeps
a sentence and its surroundings together, not semantic retrieval. The guide's
lines are hard-wrapped, so ``recursive (200)`` cuts at line ends before
sentence ends, which is why it splits so many sentences.

:class:`StructureChunker` chunks Markdown or reStructuredText by its headings
and code blocks (``markup`` is given, never guessed): code blocks are chunks
of their own, the prose of each section is split by the recursive rule, and
every chunk's metadata holds ``heading_path`` (a tuple of the titles above
it, outermost first), ``section_level`` and ``block`` (``"prose"``,
``"code"``, ``"heading"`` or ``"front_matter"``); documents built from the
chunks carry all three. With ``include_heading`` (the default) a heading
directly followed by prose begins the context of that prose's first chunk;
any other heading is a chunk of its own, so no heading is lost (unless
``min_length`` drops it). Markdown: ATX and setext headings, fenced code,
thematic breaks and YAML front matter (``front_matter``); indented code is
prose. reStructuredText: section titles,
``code-block``/``code``/``sourcecode`` directives and ``::`` literal blocks
(a directive such as ``.. note::`` is not one; its body is prose).

.. code-block:: python

   from scikitplot.corpus import StructureChunker, StructureChunkerConfig

   chunker = StructureChunker(StructureChunkerConfig(markup="rst", chunk_size=500))
   chunk = chunker.chunk(text).chunks[0]
   chunk.metadata["heading_path"], chunk.metadata["block"]

Custom downloader
-----------------

Use :class:`CustomDownloader` when the transfer itself is special, or provide a
``downloader_factory`` when the builder should own a custom download policy.
The downloader result is still a :class:`DownloadResult`, keeping download and
reader dispatch separate.

Custom retrieval
----------------

:class:`CustomRetrievalIndex` accepts a ranking callback
(``custom_scorer_fn``). :func:`make_pairwise_ranker` builds one from any
pairwise function, such as :func:`edit_similarity` or a cross-encoder, with
scikit-learn's ``greater_is_better`` sign convention, and
:func:`make_levenshtein_ranker` ranks by :mod:`scikitplot.levenshtein` similarity
with the backend that ran recorded on each hit. The Levenshtein module itself
does not import Corpus.

To *evaluate* any retriever, see :ref:`corpus-evaluation`.

Custom registries
-----------------

:class:`CapabilityRegistry` is an explicit object. Copy the default registry or
build a private one and register :class:`CapabilitySpec` objects. Registration
refuses accidental replacement unless ``replace=True`` is explicit.
