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

Cython User Guide
======================================================================

Instant PKG/MOD Generation
----------------------------------------------------------------------

The :mod:`~scikitplot.cython` submodule provides runtime Cython compilation,
artifact caching, template-based examples, controlled loading, and cache
maintenance utilities.

This guide separates the main concepts into reusable diagrams so that each
workflow can be reviewed and maintained independently.

Examples relevant to the :py:mod:`~.cython` module.

A lightweight runtime Cython development kit with caching, pinning,
garbage collection, and templating support. ``"Simple Foundation. Truly Sovereign."``

:mod:`~scikitplot.cython` enables real-time, in-place (in-situ) live, on demand
generation of low-level Cython packages and modules for immediate use and testing, at runtime.

* :ref:`sphx_glr_auto_examples_cython_plot_cython_template.py`: Example usage of
  :func:`~.compile_and_load` using template.

.. seealso::
  * https://doc.sagemath.org/html/en/reference/misc/sage/misc/cython.html
  * https://github.com/cython/cython
  * https://cython.readthedocs.io/en/latest/index.html

.. rubric:: Examples

..
  .. jupyter-execute
  .. code-block:: python

.. prompt:: python >>>

  from scikitplot.cython import compile_and_load

  m = compile_and_load("def f(int n):\n    return n*n")
  m.f(10)

..
  # https://mermaid.js.org/community/contributing.html
  # https://www.mermaideditor.io/
  # https://pypi.org/project/sphinxcontrib-mermaid

Architecture
------------

.. include:: _diagrams/architecture_overview.rst
  :start-after: :orphan:

Public API flow
---------------

.. include:: _diagrams/public_api_flow.rst
  :start-after: :orphan:

Build workflows
---------------

Single-module build
~~~~~~~~~~~~~~~~~~~

.. include:: _diagrams/single_module_build_flow.rst
  :start-after: :orphan:

Package build
~~~~~~~~~~~~~

.. include:: _diagrams/package_build_flow.rst
  :start-after: :orphan:

Lifecycle and state
-------------------

Cache lifecycle
~~~~~~~~~~~~~~~

.. include:: _diagrams/cache_lifecycle_state.rst
  :start-after: :orphan:

Lock lifecycle
~~~~~~~~~~~~~~

.. include:: _diagrams/lock_lifecycle_state.rst
  :start-after: :orphan:

Security validation
-------------------

.. include:: _diagrams/security_validation_flow.rst
  :start-after: :orphan:

Templates
---------

.. include:: _diagrams/templates_flow.rst
  :start-after: :orphan:

Garbage collection and pins
---------------------------

.. include:: _diagrams/gc_and_pins_flow.rst
  :start-after: :orphan:

Failure and recovery
--------------------

.. include:: _diagrams/failure_recovery_state.rst
  :start-after: :orphan:

Further reading
---------------

.. grid:: 1 1 1 1

  .. grid-item-card::
    :padding: 2
    :shadow: none

    **architecture**
    ^^^
    .. toctree::
      :maxdepth: 2

      architecture

  .. grid-item-card::
    :padding: 2
    :shadow: none

    **workflows**
    ^^^
    .. toctree::
      :maxdepth: 2

      workflows

  .. grid-item-card::
    :padding: 2
    :shadow: none

    **lifecycle**
    ^^^
    .. toctree::
      :maxdepth: 2

      lifecycle


.. grid:: 1 1 1 1

  .. grid-item-card::
    :padding: 2
    :shadow: none

    .. python -c 'from scikitplot import cython;cython.generate_sphinx_template_docs("./")'

    **cython templates**
    ^^^
    .. toctree::
      :maxdepth: 2

      _templates/templates_index.rst
