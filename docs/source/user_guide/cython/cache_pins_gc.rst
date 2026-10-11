.. _cython-cache-pins-gc:

Cache, pins, and garbage collection
===================================

The cache stores executable native artifacts and metadata. It is an operational
asset, not merely disposable bytecode.

``pin`` / ``unpin``
   Give selected cache keys stable aliases so application code can resolve a
   reviewed artifact deliberately.

``list_cached`` / ``list_cached_packages``
   Inspect what is present before deleting anything.

``cache_stats`` / ``gc_cache``
   Measure and collect stale cache entries under the module's coordination and
   lock rules.

Do not manually delete individual files from a cache entry while another
process may be importing it. Use the public cache/GC operations so publication
and lock invariants remain intact.
