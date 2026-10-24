..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

..
  # https://rsted.info.ucl.ac.be/
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
  # https://documatt.com/restructuredtext-reference/element/admonition.html
  # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
  # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.annoy

.. _annoy-index:

Annoy vector search
===================

``scikitplot.annoy`` wraps Spotify [1]_ Annoy-style approximate nearest-neighbor
search with a higher-level ``Index`` facade. It is designed for immutable or
build-once vector indexes where memory-mapped persistence, predictable query
semantics, and process sharing matter more than online mutation.

.. important::

   Approximate nearest-neighbor search is a retrieval strategy, not a proof of
   semantic relevance. Vector dimensionality and metric are index invariants; add
   items before building, and treat saved indexes as data artifacts with their own
   lifecycle.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   architecture
   persistence_and_io
   performance_and_threads
   python_api
   limits_and_troubleshooting
   annoy_index_vector_database
   annoy_index_inheritance_diagrams
   _annoy

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: annoy-getting-started
      :link-type: ref

      Build a small index, add vectors, build trees, and query neighbors.

   .. grid-item-card:: Architecture
      :link: annoy-architecture
      :link-type: ref

      Understand the compiled backend, high-level ``Index`` facade, and mixins.

   .. grid-item-card:: Persistence and I/O
      :link: annoy-persistence-and-io
      :link-type: ref

      Choose native files, bundles, bytes, metadata, or pickle deliberately.

   .. grid-item-card:: Performance and threads
      :link: annoy-performance-and-threads
      :link-type: ref

      Tune tree count, search effort, metrics, and build concurrency.

   .. grid-item-card:: Python API
      :link: annoy-python-api
      :link-type: ref

      Map the high-level public surface and advanced mixins.

   .. grid-item-card:: Limits and troubleshooting
      :link: annoy-limits-and-troubleshooting
      :link-type: ref

      Compiled-backend requirements, mmap lifetime, vector rules, and build caveats.

Deep dives
----------

* :doc:`annoy_index_vector_database` -- vector-database concepts and similarity search.
* :doc:`annoy_index_inheritance_diagrams` -- wrapper/backend inheritance diagrams.
* :doc:`_annoy` -- private Cython-backend notes for advanced maintainers.

Examples
--------

See :ref:`annoy_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`cexternals-annoy-index`
* :ref:`corpus-index`

Further reading
---------------

.. seealso::
  * :ref:`ANNoy <annoy-index>`
  * :ref:`cexternals/ANNoy (experimental) <cexternals-annoy-index>`
  * https://github.com/spotify/annoy
  * https://pypi.org/project/annoy

.. seealso::
  * Python pickling: https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
  * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
  * `github: ANNoy based on random projection (hyperplane) trees <https://github.com/spotify/annoy>`__
  * `pypi: ANNoy based on random projection (hyperplane) method <https://pypi.org/project/annoy>`__
  * `github: Voyager based on HNSW algorithm (hnswlib) <https://github.com/spotify/voyager>`__
  * `pypi: Voyager based on HNSW algorithm (hnswlib) <https://pypi.org/project/voyager>`__
  * `github: HNSW implementation Header-only C++/python <https://github.com/nmslib/hnswlib>`__
  * `pypi: HNSW implementation Header-only C++/python <https://pypi.org/project/hnswlib>`__
  * `github: Non-Metric Space Library (NMSLIB) <https://github.com/nmslib/nmslib>`__

.. * Annoy overview: https://www.sandgarden.com/learn/annoy-approximate-nearest-neighbors-oh-yeah
.. * FAISS overview: https://www.sandgarden.com/learn/faiss
.. seealso::
  * Nearest neighbor search (background): https://en.wikipedia.org/wiki/Nearest_neighbor_search
  * https://www.researchgate.net/publication/386374637_Optimizing_Domain-Specific_Image_Retrieval_A_Benchmark_of_FAISS_and_Annoy_with_Fine-Tuned_Features
  * https://www.researchgate.net/publication/363234433_Analysis_of_Image_Similarity_Using_CNN_and_ANNOY
  * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
  * https://link.springer.com/chapter/10.1007/978-981-97-7831-7_2

References
----------
.. [1] `Spotify AB. (2013, Feb 20). "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy <https://pypi.org/project/annoy>`_
