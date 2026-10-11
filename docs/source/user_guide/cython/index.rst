..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

..
  # https://rsted.info.ucl.ac.be/
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
  # https://documatt.com/restructuredtext-reference/element/admonition.html
  # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
  # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.cython

.. _cython-index:

Runtime Cython compilation
==========================

``scikitplot.cython`` is a runtime native-code build service: it validates
caller-supplied Cython source, resolves a toolchain, builds in a bounded staging
area, publishes deterministic cache entries, and imports the resulting
extension.

.. important::

   Compiling caller-supplied Cython means compiling executable native code. The
   security policy reduces accidental or disallowed build inputs; it is not a
   sandbox for hostile source code. Only compile code you are willing to execute
   with the current process privileges.

.. youtube:: rxy_9I8967Q

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   trust_and_security
   build_cache_import
   profiles_and_toolchains
   packages_templates_workflows
   cache_pins_gc
   python_api
   troubleshooting
   _templates/templates_index

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: cython-getting-started
      :link-type: ref

      Compile and load one small Cython module through the public runtime service.

   .. grid-item-card:: Trust and security
      :link: cython-trust-and-security
      :link-type: ref

      Understand executable-code trust, validation policy, and build budgets.

   .. grid-item-card:: Build, cache, and import
      :link: cython-build-cache-import
      :link-type: ref

      Follow keys, locks, staging, atomic publication, and transactional import.

   .. grid-item-card:: Profiles and toolchains
      :link: cython-profiles-and-toolchains
      :link-type: ref

      Inspect compiler/runtime capability before selecting a profile.

   .. grid-item-card:: Packages and templates
      :link: cython-packages-templates-workflows
      :link-type: ref

      Build multi-module packages and use validated template/workflow assets.

   .. grid-item-card:: Cache, pins, and GC
      :link: cython-cache-pins-gc
      :link-type: ref

      Operate executable cache artifacts without violating coordination rules.

   .. grid-item-card:: Python API
      :link: cython-python-api
      :link-type: ref

      Use stability tiers and structured result objects.

   .. grid-item-card:: Troubleshooting
      :link: cython-troubleshooting
      :link-type: ref

      Diagnose compiler, security, timeout, cache, and platform failures.

Template reference
------------------

See :doc:`_templates/templates_index` for the existing shipped template reference.

Examples
--------

See :ref:`cython_examples` for executable examples that accompany this guide.
