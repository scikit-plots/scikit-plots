.. currentmodule:: scikitplot.corpus

.. _corpus-getting-started:

Getting started
===============

Installation philosophy
-----------------------

The core Corpus import does not need every media, NLP, OCR, ASR or vector
backend. Install only the capabilities needed by your workload. Optional
packages are loaded lazily by the operation that uses them.

A local text build can start with :class:`CorpusBuilder`:

.. code-block:: python

   from scikitplot.corpus import BuilderConfig, CorpusBuilder

   builder = CorpusBuilder(
       BuilderConfig(
           chunker="paragraph",
           filter_kwargs={"min_words": 2, "min_chars": 8},
       )
   )
   result = builder.build("notes.txt")

``filter_kwargs`` is applied by the builder's central filter factory and is
shared by the readers it creates.

Immutable configuration
-----------------------

Use :class:`FluentCorpus` when configuration should be reusable before any
source is opened:

.. code-block:: python

   from scikitplot.corpus import FluentCorpus

   base = FluentCorpus.from_config(
       chunker="paragraph",
       storage="memory",
   )

   strict = base.with_overrides(retrieval="strict")
   print(base.diff(strict))

``FluentCorpus`` does not make call order into execution order. The plan remains
data until it is materialized and run.

Capability preflight
--------------------

Package presence is only the first question for optional backends. A wrapper can
be installed while a model, NLTK corpus, executable or other asset is absent.
Use the readiness API when that distinction matters:

.. code-block:: python

   from scikitplot.corpus import component_capabilities

   report = component_capabilities([
       "asr:faster-whisper",
       "asr:openai-whisper",
       "ocr:pytesseract",
       "xml:lxml",
   ])

   for name, state in report.items():
       print(name, state["installed"], state["assets_ready"], state["ready"])

``ready=None`` means readiness was deliberately not guessed. For model-backed
systems, proving the model is cached can itself require backend-specific work;
Corpus does not silently download a model merely to make a diagnostic green.
