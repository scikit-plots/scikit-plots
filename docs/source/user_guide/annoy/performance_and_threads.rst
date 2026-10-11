.. _annoy-performance-and-threads:

Performance and threads
=======================

Annoy exposes an explicit build/query trade-off.

``n_trees``
   More trees generally increase build time and index size and can improve
   retrieval recall.

``search_k``
   Controls query effort. Larger search effort can improve recall at additional
   latency cost.

``n_jobs`` / thread policy
   Controls build concurrency where supported. It is a resource setting, not a
   similarity parameter.

Benchmark with your vectors
---------------------------

Do not infer production recall from a toy corpus. Measure at least:

* representative vector dimensions and distribution;
* the metric you will actually use;
* index size and build time;
* query latency percentiles; and
* recall against a trusted exact baseline for a representative query set.

The gallery includes benchmark-style examples, but those examples demonstrate
measurement mechanics rather than promising a universal configuration.
