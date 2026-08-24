.. _annoy_examples:

Annoy
=====

.. currentmodule:: scikitplot.annoy

Examples for :py:mod:`~scikitplot.annoy` cover approximate nearest-neighbor
index construction, querying, persistence, memory mapping, precision trade-offs,
and lower-level native/Cython interfaces.

The gallery intentionally contains several API layers.  Most users should
start with the public Python :class:`Index` interface and move to the lower
layers only when they need dtype/native compatibility or implementation-level
benchmarking.

.. seealso::
    * `Houses Prices Tree Based Models (pickling ANNImputer) <https://www.kaggle.com/code/clkmuhammed/houses-prices-tree-based-models>`_


Start here: ANNoy Vector Index DB
---------------------------------

**1. Simple nearest-neighbor search**

Start with ``plot_simple_script.py``.

It demonstrates the shortest useful workflow:

.. code-block:: python

    from scikitplot.annoy import Index

    index = Index(
        f=3,
        metric="angular",
    )

    index.add_item(0, [1, 0, 0])
    index.add_item(1, [0, 1, 0])
    index.add_item(2, [0, 0, 1])

    index.build(-1)

    print(index.get_nns_by_item(0, 10))
    print(index.get_nns_by_vector([1.0, 0.5, 0.5], 10))

Use this page to understand the core lifecycle before looking at persistence,
Cython, legacy, or benchmark examples.


Recommended learning order
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 19 51

   * - Example
     - Level
     - What to learn
   * - ``plot_simple_script.py``
     - Beginner
     - construct, add vectors, build, and query an :class:`Index`
   * - ``plot_Annoy_python_api.py``
     - Beginner / intermediate
     - broader public Python API, parameters, inspection, compatibility
   * - ``plot_mmap_script.py``
     - Intermediate
     - save, load, and memory-map an index
   * - ``plot_annoy_to_NPY_CSV.py``
     - Intermediate
     - inspect/export index-oriented data and plotting utilities
   * - ``plot_precision_script.py``
     - Intermediate / advanced
     - query/build trade-offs on a larger generated vector set
   * - ``plot_annoy_cython_hamlet_example.py``
     - Advanced
     - real-text vectorization plus dtype/metric/native index comparison
   * - ``plot_annoy_cython_api.py``
     - Advanced
     - direct Cython-layer API and concrete native type combinations
   * - ``plot_Annoy_legacy_c_api.py``
     - Advanced / compatibility
     - low-level/legacy C-extension compatibility behavior
   * - ``plot_annoy_cython_0benchmark.py``
     - Maintainer / benchmark
     - subprocess-driven native dtype benchmark coverage


Choose the right API layer
--------------------------

The gallery contains three distinct layers.

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Layer
     - Typical import
     - Use when
   * - Public Python API
     - ``from scikitplot.annoy import Index``
     - normal application code and new examples
   * - Cython/native API
     - ``from scikitplot.annoy._annoy import Index``
     - dtype/native experiments, implementation-level testing
   * - Legacy C-extension API
     - ``scikitplot.cexternals._annoy``
     - compatibility testing and migration work

Prefer the public Python API unless the example is specifically teaching a
lower-level contract.

Private/internal imports in the advanced examples are deliberate teaching or
maintenance surfaces; they should not be copied into ordinary application code
without understanding their compatibility implications.


Core index lifecycle
--------------------

The common Annoy lifecycle is:

.. code-block:: text

    vectors
       ↓
    Index(f, metric)
       ↓
    add_item(...)
       ↓
    build(n_trees)
       ↓
    query
      ├── get_nns_by_item(...)
      └── get_nns_by_vector(...)
       ↓
    optional save(...)
       ↓
    optional load(...) / mmap-backed querying

Build parameters affect index construction, while query parameters affect the
search work performed against an already-built index.


Metrics
-------

Examples use several supported metric families, including:

``angular``
    cosine-like angular distance for directional similarity.

``euclidean``
    L2 distance.

``manhattan``
    L1 distance.

``dot``
    dot-product-oriented similarity.

``hamming``
    Hamming-distance-oriented indexing where supported by the selected native
    type combination.

Not every dtype/metric/native combination should be assumed equivalent.
Use the public API defaults for ordinary applications and the Cython examples
when explicitly investigating concrete type combinations.


Persistence and memory mapping
------------------------------

``plot_mmap_script.py`` demonstrates the persistent workflow:

.. code-block:: python

    index.save("example.annoy")

    restored = Index(
        f=3,
        metric="angular",
    )
    restored.load("example.annoy")

Annoy indexes are designed to support file-backed querying.  Keep the vector
dimension and metric consistent with the index that was written.

Treat bundled ``.annoy`` / ``.tree`` files in this gallery as example/test
artifacts, not as a stable cross-version interchange format unless the relevant
compatibility contract has been explicitly verified.


Precision and benchmark examples
--------------------------------

The precision and benchmark pages are not beginner tutorials.

They may:

* generate many random vectors,
* build many trees,
* exercise multiple dtype/metric combinations,
* launch subprocesses or pytest benchmarks,
* write index artifacts,
* consume substantially more CPU, memory, or disk than the simple example.

Use them for engineering comparison, regression testing, or capacity planning.
Do not infer production sizing from a single gallery benchmark.


Hamlet Cython showcase
----------------------

``plot_annoy_cython_hamlet_example.py`` is the broadest Annoy demonstration.

It uses a real text corpus, converts passages to vectors, and compares native
Annoy behavior across advanced configurations.

Conceptually:

.. code-block:: text

    Hamlet passages
        ↓
    text vectorization
        ↓
    native Annoy Index
        ↓
    dtype / metric variants
        ↓
    nearest-neighbor search
        ↓
    build-time / size / retrieval comparison

Use it after the public Python examples.  It is a showcase of the lower-level
native implementation surface, not the minimum API required to use Annoy.


Native capability
-----------------

Annoy in scikit-plots includes compiled/native components.  A gallery or CI
environment may therefore have the Python package available while a required
native implementation is unavailable for that platform/build.

The desired gallery rule is:

``native Annoy capability unavailable``
    Report a specific ``SKIP`` when the example cannot truthfully continue.

``native Annoy imports successfully but build/query fails``
    Fail visibly.  Do not convert an installed-backend regression into a skip.

The same distinction applies to compiler-dependent benchmark examples.


Files and working directories
-----------------------------

Several historical/advanced examples read or write artifacts such as:

.. code-block:: text

    *.annoy
    *.tree
    *.npy
    *.csv
    *.joblib

New examples should prefer a temporary directory or a path derived from the
example location rather than relying on the caller's current working directory.

Gallery examples should not overwrite repository test fixtures merely to
demonstrate persistence.


Reproducibility
---------------

For deterministic examples:

* seed random-number generators where random vectors are generated,
* keep dimensions and item counts bounded,
* print bounded result summaries,
* separate correctness checks from performance measurements,
* record the metric, number of trees, and query parameters used for benchmark
  comparisons.

Approximate nearest-neighbor results can depend on build/search configuration;
benchmark pages should make those parameters visible.


Relationship to Corpus and MCP
------------------------------

Annoy can also be used as a retrieval backend by higher-level scikit-plots
components.

For document retrieval:

.. code-block:: text

    scikitplot.corpus
        ↓
    document embeddings
        ↓
    Annoy vector backend
        ↓
    retrieval

For a protocol/server workflow:

.. code-block:: text

    Corpus
      ↓
    Annoy
      ↓
    CorpusAnnoyRetriever
      ↓
    scikitplot.mcp

Use the Corpus/MCP galleries when the goal is document ingestion or MCP
serving.  Use this Annoy gallery when the goal is understanding and validating
the vector index itself.


CI guidance
-----------

A practical CI split is:

**Portable/public API gate**
    Run the small public Python examples and focused Annoy tests.

**Native capability gate**
    Run Cython/native examples only on environments that intentionally provide
    the required compiled extension.

**Benchmark gate**
    Keep expensive dtype/precision/compiler benchmarks separate from the normal
    documentation build when they materially increase build time or resource
    consumption.

An unavailable optional native/compiler capability should be reported
explicitly.  Security, serialization, index-corruption, or installed-backend
failures should remain visible.


Browser / WASM note
-------------------

Do not assume that the native Annoy implementation, memory mapping, filesystem
semantics, subprocesses, or C/C++ compiler examples are available in
JupyterLite/Pyodide/other browser-WASM runtimes.

Use the simple/public API examples there only when the relevant native package
has been explicitly built and verified for that runtime.

The Cython, mmap, compiler, and benchmark pages should otherwise be treated as
reference material in browser environments.


Gallery reliability rule
------------------------

Keep Annoy examples small, explicit, and honest:

``missing optional native/compiler capability``
    visible ``SKIP`` when continuation is safe.

``wrong public API / invalid vector dimension / corrupt required artifact /
installed-backend failure``
    visible failure.

Do not fabricate fallback nearest-neighbor results merely to keep a gallery
page green.
