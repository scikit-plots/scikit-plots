.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_ai_learn

.. _externals-sphinx-ext-sphinx-ai-learn-index:

======================================================================
Sphinx AI Learn
======================================================================

``_sphinx_ai_learn`` is the JSON-first materializer used by the Scikit-Plots
Learn documentation.  Canonical repository content is validated JSON; the
extension deterministically derives sibling RST during ``config-inited`` so the
same Sphinx build can discover and render the generated documents.

The central ownership rule is::

   canonical JSON  ->  generated RST  ->  Sphinx output

Edit the JSON authority, not generated RST.

Enable it
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_ai_learn",
   ]

   ai_learn_content_root = "learn-ai"
   ai_learn_site_id = "scikit-plots-learn"
   ai_learn_runtime = "none"       # or "assistant"
   ai_learn_media = False

``sphinx_design`` is loaded by the extension itself.  When media support is
enabled, the extension also loads the local gallery/player dependencies.  When
``ai_learn_runtime = "assistant"``, it loads the sibling AI Assistant extension.

Build-time guarantees
----------------------------------------------------------------------

Materialization is deliberately local and deterministic.  The build-time
materializer performs no model calls, network fetches, Git operations,
telemetry, or publication writes.

The materializer validates the canonical tree, writes only changed
extension-owned RST, and prunes stale RST only when that file is recognized as
extension-owned.  Publication is a separate lifecycle and must not be confused
with documentation compilation.

Important configuration
----------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Setting
     - Default
     - Meaning
   * - ``ai_learn_content_root``
     - ``"learn-ai"``
     - Canonical Learn content root below the Sphinx source tree.
   * - ``ai_learn_site_id``
     - ``"scikit-plots-learn"``
     - Logical site identifier used by the rendered runtime contracts.
   * - ``ai_learn_runtime``
     - ``"none"``
     - ``"none"`` or ``"assistant"``; invalid values fail configuration.
   * - ``ai_learn_explorer_search_variant``
     - ``"pill-overflow"``
     - Shared explorer search presentation.
   * - ``ai_learn_media``
     - ``False``
     - Enables gallery/player dependencies and media presentation.
   * - ``ai_learn_youtube_subscribe_url``
     - ``""``
     - Optional YouTube subscription destination used by Learn pages.
   * - ``ai_learn_buttons_ratings``
     - left/right balanced mapping
     - Controls reviewed-count placement in the compact Learn feedback buttons.

Canonical content contracts
----------------------------------------------------------------------

The current materializer recognizes typed JSON contracts for structural pages,
records, record-owned sections, reusable topic prompts and reusable skills.
The normalized catalog is an in-memory graph derived from those files; it is
not a second repository ``catalog.json`` authority.

Record/section bodies remain data.  Where structural page layout is needed, a
bounded typed design-grid contract is used instead of accepting arbitrary raw
RST directives from JSON.

Generated directives
----------------------------------------------------------------------

The extension registers ``ai-learn`` plus the page/record directives used by
materialized RST, including topic, media, prompt, skill, explorer and generation
surfaces.  Normal content authors should not treat those directives as the
canonical authoring format: they are an implementation target of the JSON
materializer.

Include versus toctree composition
----------------------------------------------------------------------

Record JSON can choose between two derived-document layouts:

* default include composition keeps one navigable parent and treats child RST
  as orphan/no-search fragments;
* optional toctree composition makes children normal navigable documents.

Both modes derive from the same JSON contracts.  Changing navigation policy
should not require duplicating content authority.

Validation failures
----------------------------------------------------------------------

Unknown fields, unsafe document targets, invalid runtime values and unsupported
layout values fail validation instead of being silently ignored.  This is
intentional: a documentation build should expose schema drift before publishing
partially interpreted Learn content.
