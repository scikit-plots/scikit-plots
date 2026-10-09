.. currentmodule:: scikitplot._externals._sphinx_ext._sphinx_collection

.. _externals-sphinx-ext-sphinx-collection-index:

======================================================================
Sphinx collection engine
======================================================================

``_sphinx_collection`` is the shared support library behind collection-style
Sphinx directives.  It owns domain-agnostic filtering, sorting, grouping,
bounded YAML loading, browser metadata, result/status controls and generated
collection assets.

It is **not** a standalone Sphinx extension: there is no package-level
``setup(app)`` entry point to add to ``extensions``.  Use a consuming extension
such as :doc:`../_sphinx_gallery_grid/index` or
:doc:`../_sphinx_youtube_gallery/index`.

Why it exists
----------------------------------------------------------------------

Keeping collection behavior in one package prevents ``gallery-grid`` and typed
adapters from drifting on search/filter/sort behavior.  The intended ownership
is:

* a domain adapter normalizes records;
* ``gallery-grid`` owns card/layout rendering;
* ``_sphinx_collection`` owns selection and browser-control behavior.

This means a YouTube-specific module should not implement a second search UI,
and the leaf video player should never paginate gallery cards.

Programmatic use
----------------------------------------------------------------------

The package lazily exposes selection and grouping helpers such as
``parse_filter``, ``parse_sort``, ``apply_selection`` and ``group_records``.
It also exposes the shared asset/integrity hooks used by consuming Sphinx
extensions.

The package initializer intentionally avoids eager imports so data-only helpers
remain usable in code paths that do not have Sphinx/docutils installed.

Bounded YAML contract
----------------------------------------------------------------------

The shared YAML loader uses ``yaml.safe_load`` **and** deterministic resource
limits.  Current limits are:

.. list-table::
   :header-rows: 1

   * - Resource
     - Limit
   * - Input bytes
     - 8 MiB
   * - YAML aliases
     - 100
   * - Nesting depth
     - 32
   * - Expanded values
     - 100,000
   * - Scalar characters
     - 1,048,576
   * - Collection items
     - 5,000

Recursive aliases and inputs that exceed a bound raise ``BoundedYAMLError``.
These limits exist because safe YAML object construction alone does not prevent
resource-exhaustion input.

Generated browser assets
----------------------------------------------------------------------

Consuming extensions register a digest of the shared CSS/JavaScript as an HTML
rebuild dependency.  The assets are written atomically and verified again at
``build-finished``.  Searchable gallery output is also checked for the expected
document-owned status marker so a mixed old/new incremental build fails instead
of publishing a structurally inconsistent UI.

Progressive enhancement
----------------------------------------------------------------------

The static collection remains readable without JavaScript.  Browser controls
may hide, reorder or add local cards after load, but they do not own the
canonical build-time record set.
