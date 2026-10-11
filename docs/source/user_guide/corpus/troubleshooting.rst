.. currentmodule:: scikitplot.corpus

.. _corpus-troubleshooting:

Troubleshooting
===============

Package installed, backend still unavailable
--------------------------------------------

Inspect :func:`component_capabilities`. For model/system-backed components, look
at ``assets_ready`` and ``reason_code`` rather than only ``installed``.

Whisper fails but the gallery should continue
---------------------------------------------

The default media-reader policy is fail-soft and observable. A runtime failure
in ``faster-whisper`` is logged and the OpenAI Whisper backend is attempted. If
all backends fail, no ASR chunks are produced and ``backend_reports`` records the
failure. Set ``strict=True`` or ``backend_policy="strict"`` when exhaustion
should raise.

Need a fully local ASR
----------------------

Provide :class:`ASRBackend` and select it first with an explicit
:class:`BackendPolicy`. Use ``include_unlisted=False`` if built-in Whisper
backends must not be attempted.

A builder factory seems ignored
-------------------------------

Pass :class:`BuilderFactories` directly to :class:`CorpusBuilder` for new code.
:class:`FactoryCorpusBuilder` is a compatibility facade over the same native
seams. Reader, filter and downloader factories are wired through central
construction paths; if behavior bypasses one of these seams, treat that as a
regression rather than adding another special-case factory.

A URL works in a browser but Corpus refuses it
----------------------------------------------

Do not disable SSRF/fail-closed resolution globally just to make one environment
pass. Check DNS/proxy/container network configuration first. Private-network
access should be an explicit, narrow policy decision.

A Levenshtein backend is missing
--------------------------------

``backend="auto"`` remains usable because it ends in the dependency-free Python
implementation. An explicitly requested unavailable backend warns and falls back
unless ``strict=True``. Use ``backend_info()`` to inspect the selected backend.


Offline policy says unavailable, not empty
------------------------------------------

``BackendStatus.UNAVAILABLE`` means policy/readiness left no runnable backend;
it is intentionally different from ``EMPTY``, where a backend really ran and
produced a valid empty result. For faster-whisper, the offline policy forces a
local-files-only load so a cached model may still work even when preflight
readiness is ``UNKNOWN``.
