.. currentmodule:: scikitplot.corpus

.. _corpus-readers-backends:

Readers, backends, and readiness
================================

Reader dispatch
---------------

:meth:`DocumentReader.create` selects a file reader from the registered suffix.
:meth:`DocumentReader.from_url` selects a URL-aware reader. The higher-level
:class:`CorpusBuilder` routes every local/downloaded file through its single
``_make_reader`` seam so filters and user factories are applied consistently.

Backend outcome is not a boolean
--------------------------------

Optional backend operations distinguish:

``success``
    A backend produced an accepted non-empty result.

``empty``
    A backend completed normally and produced a valid empty result. This is not
    a reason to try another backend unless the reader explicitly says so.

``degraded``
    An earlier preferred backend failed and a later fallback succeeded.

``unavailable``
    Readiness/policy left no backend eligible to execute. This is **not** a
    successful empty result.

``exhausted``
    One or more backends ran normally but every returned value was rejected as
    unusable. This is also distinct from a valid empty result.

``failed``
    At least one attempted backend raised and no fallback recovered.

Reader instances expose these records through ``reader.backend_reports``. The
records contain safe error metadata, not live exception/traceback objects.
``skip_details`` explains policy/readiness exclusions (backend, reason, capability,
capability status, readiness) so an ``unavailable`` result is diagnosable without
replaying the operation or scraping debug logs.

What ran, what was asked, and why they differ
---------------------------------------------

The components that choose between implementations -- reader backend chains
(:class:`BackendOutcome`), vector backends and their Annoy bindings, and
``SemanticChunker`` -- report the same four names:

``requested_backend``
    What the configuration asked for (``"auto"``, a backend name, a binding).

``backend``
    What actually ran.

``fallback_reason``
    Why they differ, from a closed vocabulary per component; ``None`` when they
    do not. Reader chains (:class:`BackendOutcome`) use ``"failed"`` and
    ``"no_result"`` (a candidate the policy skips was never eligible, as an
    absent backend is not a fallback under ``"auto"``); vector backends use
    ``"<name>_broken"`` (installed but failing to load, skipped under
    ``"auto"``), ``"option_unsupported"`` and ``"<binding>_unavailable"``;
    ``SemanticChunker`` uses ``"sentence_transformers_unavailable"``,
    ``"model_load_failed"`` and ``"embedding_failed"``.

``allow_fallback``
    The switch: ``True`` (default) runs the fallback and reports it, ``False``
    raises instead (:class:`FallbackRefusedError` for vector backends; a
    strict chain attempts only its first eligible candidate). An unknown
    option name raises in both modes, and an explicitly named vector backend
    is never replaced. An Annoy ``dtype`` value is checked by the binding that
    honours it; one dropped by a fallback is reported verbatim.

For the dense index, :attr:`RetrievalIndex.backend_provenance` reports the four
names (and, for Annoy, the binding and any ``discarded_options``), and every
dense leg (``RetrievalIndex`` and ``DenseRetriever``) carries
``requested_backend`` and ``fallback_reason``. Invalid embeddings (non-finite,
not two-dimensional) still disable the dense index with a warning; a backend
error, a refused fallback included, propagates from ``build``.
``AnnoyBackend(impl="auto")`` picks a binding that accepts every requested
option, so ``annoy_dtype="float64"`` runs on the Cython class rather than being
dropped by the high-level one:

.. code-block:: python

    from scikitplot.corpus import RetrievalConfig, RetrievalIndex

    config = RetrievalConfig(backend="annoy", annoy_dtype="float64")
    strict = RetrievalConfig(backend="annoy", annoy_impl="highlevel",
                             annoy_dtype="float64", allow_fallback=False)
    # index = RetrievalIndex(config); index.build(documents)
    # index.backend_provenance
    #   -> {"requested_backend": "annoy", "backend": "annoy", "impl": "cython",
    #       "fallback_reason": None, ...}
    # RetrievalIndex(strict).build(documents) raises FallbackRefusedError: the
    #   high-level binding cannot honour dtype.

:func:`capability_snapshot` loads no backend by default, so it does not import
Annoy (or, through it, scikit-learn and SciPy): Annoy reports ``unknown``
(``not_loaded``, and so ``available`` is ``False``) until this process has
loaded it; an installed FAISS or Voyager likewise. ``capability_snapshot(load=True)`` loads
them to tell ``available`` from ``broken``. :func:`probe_backend` reports a
backend that is installed but fails to load as ``broken``, never as
``absent``; one with no evidence of a failed load stays ``absent``.

Readiness vocabulary
--------------------

:class:`CapabilityReport` separates five facts that are commonly conflated:

``installed``
    The Python package/module can be located.

``assets_ready``
    A required model, corpus, executable or other asset is known ready. ``None``
    means the lightweight probe cannot know without doing real work.

``ready``
    Combined preflight answer: ``True``, ``False`` or ``None``.

``selected``
    Policy selected the capability for an operation.

``active``
    Runtime evidence says the capability actually completed work.

Do not turn ``UNKNOWN`` into ``AVAILABLE`` for presentation. Unknown readiness
is useful evidence.


Role-based discovery
--------------------

Callers do not need to memorize every capability identifier. Discovery can be
scoped by semantic role while keeping probes side-effect free:

.. code-block:: python

   from scikitplot.corpus import component_capabilities

   asr = component_capabilities(role="asr")
   ocr = component_capabilities(role="ocr")
   pdf = component_capabilities(role="pdf")
   distance = component_capabilities(role="edit-distance")

Plan the exact reader backend chain before running
--------------------------------------------------

Audio and video readers expose :meth:`AudioReader.plan_asr_backends` /
:meth:`VideoReader.plan_asr_backends`. The plan is built by the **same**
candidate factory and policy resolver used during transcription, but does not
load a model, decode media, or call a user backend.

.. code-block:: python

   from pathlib import Path
   from scikitplot.corpus import AudioReader

   reader = AudioReader(
       Path("meeting.wav"),
       transcribe=True,
       backend_policy="offline",
   )
   plan = reader.plan_asr_backends()
   print(plan.to_dict())
   print(plan.capability_view())

:class:`BackendPlan` separates ``declared``, ``ordered``, ``eligible``,
``selected`` and ``skipped`` candidates. A skipped backend includes its reason
and readiness evidence. After execution, :class:`BackendOutcome` exposes the
same capability view with ``active=True`` only for the backend that actually
completed the operation.

For lower-level/custom operations, :func:`plan_backend_chain` provides the same
side-effect-free preflight over explicit :class:`BackendCandidate` objects.

Custom readiness registries
---------------------------

A private backend can publish readiness without mutating process-global
discovery. Pass the same :class:`CapabilityRegistry` to the reader and give the
custom :class:`ASRBackend` its capability identifier. This keeps preflight
(``selected``) and runtime evidence (``active``) connected end to end.

.. code-block:: python

   from scikitplot.corpus import (
       ASRBackend,
       AudioReader,
       CapabilityRegistry,
       CapabilitySpec,
   )

   registry = CapabilityRegistry([
       CapabilitySpec("asr:company", "asr"),
   ])
   backend = ASRBackend(
       "company-asr",
       local_asr,
       capability="asr:company",
       offline_capable=True,
   )
   reader = AudioReader(
       "meeting.wav",
       transcribe=True,
       asr_backends=(backend,),
       capability_registry=registry,
   )

The registry is evidence, not a sandbox: arbitrary user backends remain trusted
Python code and must honour any offline/network promises they declare.

Backend policies
----------------

:class:`BackendPolicy` provides reusable orchestration presets:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Policy
     - Meaning
   * - ``resilient``
     - Declared order, fallback allowed, structured failure collection.
   * - ``strict``
     - Same fallback opportunities, then caller raises on exhaustion.
   * - ``offline``
     - Forbid network/download side effects. A backend that can enforce a
       local-only mode may still be attempted when cache readiness is
       ``UNKNOWN``; a backend that cannot guarantee local-only execution is
       skipped.
   * - ``first``
     - Attempt only the first selected candidate.

A fully custom policy is a normal dataclass:

.. code-block:: python

   from scikitplot.corpus import BackendPolicy, ErrorPolicy

   policy = BackendPolicy(
       name="local-first",
       order=("company-asr", "faster-whisper"),
       include_unlisted=False,
       allow_network=False,
       allow_download=False,
       require_ready=False,
       on_exhausted=ErrorPolicy.COLLECT,
   )

Custom ASR
----------

A user backend receives one immutable :class:`ASRRequest`; this avoids an
unstable expanding keyword callback signature.

.. code-block:: python

   from scikitplot.corpus import ASRBackend, AudioReader, BackendPolicy

   def local_asr(request):
       return [{
           "text": "locally generated transcript",
           "timecode_start": 0.0,
           "timecode_end": 1.0,
       }]

   backend = ASRBackend("company-asr", local_asr)
   policy = BackendPolicy().with_order("company-asr", include_unlisted=False)

   reader = AudioReader(
       input_path="meeting.wav",
       transcribe=True,
       backend_policy=policy,
       asr_backends=(backend,),
   )

Built-in backend names are reserved; custom code cannot silently replace
``faster-whisper`` or ``openai-whisper``. A custom backend that declares
``offline_capable=True`` promises to honour the request's
``allow_network=False`` and ``allow_download=False`` fields. The orchestrator
cannot prove that promise for arbitrary user code, so only trusted backends
should opt in. The built-in faster-whisper adapter enforces local-only loading
when downloads are forbidden; OpenAI Whisper is conservatively skipped by the
offline preset because its normal loader may fetch a missing model.

OCR is explicit, not an automatic cascade
-----------------------------------------

:class:`ImageReader` defaults to ``backend="tesseract"``. It does **not**
automatically switch to EasyOCR when Tesseract is missing or broken, because
EasyOCR may download model weights and has different resource/accuracy
semantics. Choose ``backend="easyocr"`` explicitly when that behavior is
acceptable. This is an example of why shared backend orchestration does not
imply one universal fallback policy.
