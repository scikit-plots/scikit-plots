.. _annoy-limits-and-troubleshooting:

Limits and troubleshooting
==========================

Compiled backend unavailable
----------------------------

This submodule requires compiled Annoy support. An import/build problem in
``scikitplot.annoy`` is different from a problem in the independent
``scikitplot.cexternals._annoy`` owner. Keep those build domains separate when
diagnosing failures.

The repository maintenance snapshot also records a build-structure issue in the
private Cython Annoy path involving a missing Tempita generator. That is a build
and release concern; user code should not work around it by copying headers or
private generator files.

Common mistakes
---------------

* adding a vector with the wrong dimensionality;
* querying before a useful index has been built;
* loading an index with a different metric/dimension expectation;
* expecting approximate neighbors to equal an exact search for every query;
* deleting or replacing a file while another process still memory-maps it; and
* using pickle with untrusted input.

When debugging retrieval quality, first verify vector generation and metric
choice, then measure Annoy recall. Do not compensate for a broken embedding
pipeline by increasing ``search_k`` indefinitely.
