.. currentmodule:: scikitplot.corpus

.. _corpus-security-limits:

Security and limits
===================

Fail-soft is not silent success
-------------------------------

An optional backend may be allowed to fail without aborting a heterogeneous
build, but the failure remains visible through logging and structured
``backend_reports``. Use strict policy when a missing result must invalidate the
run.

Network boundaries
------------------

URL downloading is subject to scheme/redirect/SSRF/size controls. Network source
permission in :class:`RuntimePolicy` is explicit. Model/resource downloads are a
separate capability decision.

Model-backed readiness
----------------------

``installed=True`` is not enough for systems such as Whisper/EasyOCR. A model
may be absent and first use may download it. The readiness registry reports
``ready=None`` when it cannot prove local asset presence without side effects.

User code
---------

Custom extractors, factories, ASR backends, downloader handlers and retrieval
scorers are executable Python supplied by the caller. Treat configuration that
contains callables as code, not as untrusted data to deserialize automatically.

Resource exhaustion
-------------------

Keep existing archive/file/download limits enabled. A custom backend also needs
its own model/input/time/memory bounds; the generic backend runner cannot infer
safe limits for arbitrary third-party engines.

Levenshtein limits
------------------

The pure-Python Levenshtein fallback uses memory-bounded Wagner-Fischer dynamic
programming but still has quadratic time in the input lengths. Use bounded input
sizes for untrusted very-long strings. The optional native backends improve
speed; they do not turn edit distance into constant-cost validation.
