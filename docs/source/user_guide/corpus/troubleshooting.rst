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

A segmenter warning appears once, then never again
--------------------------------------------------

A missing optional segmenter (``jieba``, MeCab/fugashi/SudachiPy, ``kss``,
PyThaiNLP, ``khmer-nltk``, ``camel-tools``, or ``regex`` for script detection)
is a fact about the environment, so its warning is logged once per process,
not once per text or per chunker; so is an installed analyser that fails to
start (MeCab without a dictionary). A warning suppressed by the logger's level
or by ``logging.disable`` when it first occurs is not used up (one dropped by
a handler's level or a filter is). To see every optional segmenter and whether it is
importable, ask for the report instead of reading the log:

.. code-block:: python

   from scikitplot.corpus import component_capabilities

   for name, record in component_capabilities(role="segmentation").items():
       print(name, record["status"], record["remedy"])

With ``regex`` installed, ``ScriptSegmenter`` detects the script of each span
that ``ScriptSegmenter.SCRIPT_PROPERTY_MAP`` names (Latin, Han, Arabic, ...),
and its ``start``/``end`` are character offsets into the text. Han next to
kana stays in one Japanese span, and Han next to Hangul in one Korean span,
when they meet letter to letter (a space or punctuation separates them).
Text with no script at all (digits, punctuation) is ``ScriptType.UNKNOWN``
without a warning; a script the map does not name (Bengali, for example) is
its own ``UNKNOWN`` span, noted once per process for each such script (as
its Unicode character names call it; characters without a name, such as
private-use icons, share one notice per general category).

``ResourceWarning: unclosed database``
--------------------------------------

Python 3.13 and later report a SQLite connection that is garbage-collected
while open. Close every store you open, preferably with ``with``:

.. code-block:: python

   from scikitplot.corpus import SQLiteStorage

   with SQLiteStorage("corpus.db") as store:
       store.save_batch(documents)

Every storage backend supports ``with`` and an idempotent ``close()``. A
``RuntimeCorpus`` closes the store it created itself when it is closed
(``close()`` or ``with``), and a materialisation that fails closes it at once;
a ``HybridRetriever`` closes its legs the same way.
