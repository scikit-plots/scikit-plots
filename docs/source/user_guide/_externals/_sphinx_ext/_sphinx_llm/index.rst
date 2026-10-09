.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_llm

.. _externals-sphinx-ext-sphinx-llm-index:

======================================================================
Sphinx LLM (experimental, maintenance-gated)
======================================================================

``_sphinx_llm`` is the Scikit-Plots subsystem for static machine-consumable
Sphinx artifacts.  It combines a pinned NVIDIA-derived baseline with
Scikit-Plots compatibility, semantic-adapter, curation and artifact layers.

.. warning::

   This subsystem is under an explicit maintenance campaign and is not the
   default documentation path in the current Scikit-Plots ``conf.py``.  Treat
   it as experimental until its maintained compatibility gates are complete.
   Do not infer release readiness merely because ``setup(app)`` exists.

Intended output model
----------------------------------------------------------------------

The current design favors targeted retrieval::

   HTML page
      |-- discover -> canonical Markdown for that page
      `-- discover -> applicable llms.txt

rather than requiring every consumer to load one giant full-corpus file.
``llms-full.txt`` remains optional and is controlled by an explicit size/policy
surface.

Basic registration
----------------------------------------------------------------------

For an experimental build that has passed the subsystem's local maintenance
gates::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_llm",
   ]

   llms_txt_enabled = True
   llms_txt_discovery_links = True
   llms_txt_unknown_node_policy = "warn"
   llms_txt_full_build = False
   llms_txt_html_fallback = False

The extension injects page-relative discovery links only for HTML/dirhtml
builders.

Semantic directives
----------------------------------------------------------------------

Two author-facing directives are currently registered:

``llms-ignore``
   Suppresses a subtree from canonical machine Markdown while leaving normal
   human output transparent.

``docref``
   Participates in the vendored/compatibility document-reference and summary
   workflow.

Configuration groups
----------------------------------------------------------------------

The configuration surface is intentionally broader than a single boolean.  It
includes:

* core enablement, description and suffix routing;
* include/exclude/ordering and section-curation policies;
* unknown-node behavior and optional HTML fallback;
* ``llms-full`` byte/character/line/document size limits and size policy;
* discovery-link publication;
* optional page-summary provider/model/endpoint/cache settings.

When remote summary generation is used, API credentials are read from the
configured environment-variable name.  The implementation includes safeguards
against sending an API key to a non-loopback insecure endpoint unless the
insecure-auth policy is explicitly enabled.

Coexistence with AI Assistant
----------------------------------------------------------------------

:doc:`../_sphinx_ai_assistant/index` also has an existing Markdown/``llms.txt``
artifact pipeline.  Until the repository defines one explicit coexistence or
migration contract, do not enable both default writers and rely on event order
to decide which ``llms.txt`` survives.  Select one artifact owner or disable
the overlapping output in one extension.
