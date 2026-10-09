.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_gallery_grid

.. _externals-sphinx-ext-sphinx-gallery-grid-index:

======================================================================
Sphinx Gallery Grid
======================================================================

``_sphinx_gallery_grid`` provides the theme-independent ``gallery-grid``
directive for structured card collections.  Records can be written inline as
YAML or loaded from a UTF-8 YAML file.  The directive delegates card rendering
to Sphinx Design and delegates generic selection/browser behavior to
:doc:`../_sphinx_collection/index`.

Enable it
----------------------------------------------------------------------

::

   extensions += [
       "scikitplot._externals._sphinx_ext._sphinx_gallery_grid",
   ]

The extension loads ``sphinx_design`` itself.

Minimal gallery
----------------------------------------------------------------------

::

   .. gallery-grid::
      :grid-columns: 1 1 2 3

      - title: Getting started
        content: Install the package and make a first plot.
        link: ../../../index
        link-alt: Open the guide
      - title: API reference
        content: Browse the documented public API.
        link: ../../../../apis/index
        link-alt: Open the API reference

Card records may carry presentation fields such as ``title``, ``header``,
``image``, ``content`` and ``link`` plus additional data-only fields used by
filtering, sorting or grouping.

File-backed gallery
----------------------------------------------------------------------

::

   .. gallery-grid:: ./_data/resources.yaml
      :sort: title
      :group-by: category
      :show-count:

File paths are resolved relative to the authoring document and are confined to
the Sphinx source/configuration tree after symlink resolution.  A gallery cannot
use a relative path to read arbitrary files elsewhere on the build host.

Build-time selection
----------------------------------------------------------------------

The directive supports build-time operations including:

* ``:filter:`` for field predicates;
* ``:sort:`` for deterministic ordering;
* ``:group-by:`` for top-level sections;
* ``:offset:`` and ``:limit:`` for build-time slicing;
* ``:show-count:`` for an honest selected/total count.

Build-time ``limit`` is distinct from the browser's bounded display window.
Once records are omitted by the build-time selection, browser controls cannot
recover them.

Interactive reader controls
----------------------------------------------------------------------

Use ``:searchable:`` or ``:interactive:`` to activate the shared collection
browser.  Both accept the configured search presentation, and a directive can
override it with ``:search-variant:``.  The current variants are
``pill-overflow`` and ``classic``.

Useful controls include ``:filter-fields:``, ``:sort-fields:``,
``:search-fields:``, ``:search-label:`` and a stable ``:collection-id:``.
Interactive views use the shared bounded-card display controller rather than
rendering a second gallery implementation.

Presentation customization
----------------------------------------------------------------------

``grid-*`` options are validated using the installed Sphinx Design grid option
specification, and ``card-*`` options use Sphinx Design's card specification.
The directive also accepts ``grid-columns``, ``class-container`` and
``class-card`` conveniences.

Option values that are re-emitted as generated directive source must stay on
one line.  URL-like presentation options reject active schemes; card links
accept HTTP(S), ``mailto`` and relative/fragment destinations.

Input safety
----------------------------------------------------------------------

Inline and file-backed YAML use the shared bounded loader documented in
:doc:`../_sphinx_collection/index`.  Python-object YAML constructors are not
accepted, recursive aliases are rejected, and collection/resource limits are
applied before rendering.

MyST documents
----------------------------------------------------------------------

The implementation detects the source parser and emits matching nested Sphinx
Design syntax, so the same extension can be used from reStructuredText and MyST
sources.  Authors do not need a separate gallery implementation for Markdown.
