.. _cython-build-cache-import:

Build, cache, and import
========================

The runtime path is intentionally transactional:

.. code-block:: text

   validate -> fingerprint/key -> per-key lock -> staging build
            -> validate artifact -> atomic cache publish -> import

Cache keys include source and runtime/toolchain information required to decide
whether an existing artifact is reusable. Cache metadata is schema-versioned;
unknown future schemas are rebuilt rather than trusted optimistically.

Use ``compile_and_load_result`` or package ``*_result`` APIs when automation
needs structured evidence. Use ``force_rebuild`` to invalidate reuse for a
particular operation, not as a routine substitute for understanding cache
identity.

Artifact import APIs
--------------------

``import_artifact_path`` and ``import_artifact_bytes`` load native artifacts.
Treat those artifacts exactly like executable code. Registering an artifact in
the cache does not establish provenance or trust.
