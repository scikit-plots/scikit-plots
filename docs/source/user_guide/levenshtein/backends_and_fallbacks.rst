.. _levenshtein-backends:

Backends and fallback
=====================

.. currentmodule:: scikitplot.levenshtein

Inspect capabilities
--------------------

.. code-block:: python

   from scikitplot.levenshtein import available_backends

   for info in available_backends():
       print(info.name, info.available, info.version, info.license)

The capability record is informational and does not import every optional
accelerator eagerly when the facade itself is imported.

Automatic policy
----------------

``backend="auto"`` uses this preference order:

.. list-table::
   :header-rows: 1
   :widths: 20 22 20 38

   * - Backend
     - Implementation
     - License posture
     - Automatic?
   * - ``internal``
     - bundled ``cexternals._editdistance``
     - MIT component in Scikit-Plots
     - yes
   * - ``rapidfuzz``
     - ``rapidfuzz.distance.Levenshtein``
     - MIT
     - yes
   * - ``python``
     - local dynamic-programming fallback
     - BSD-3-Clause
     - yes, terminal fallback
   * - ``levenshtein``
     - external ``Levenshtein`` package
     - GPL-2.0-or-later
     - **no; explicit only**

Explicit selection
------------------

.. code-block:: python

   from scikitplot.levenshtein import distance

   distance("kitten", "sitting", backend="rapidfuzz")

If that backend is unavailable:

``strict=False`` (default)
   Log a warning and fall back to the safe automatic path.

``strict=True``
   Raise :class:`ImportError`.

.. code-block:: python

   distance(
       "kitten",
       "sitting",
       backend="rapidfuzz",
       strict=True,
   )

Runtime failure
---------------

An accelerator can import correctly and still fail for a particular input or
environment.

With ``strict=False`` the facade warns and uses pure Python. With
``strict=True`` the original runtime exception is propagated.

The pure-Python implementation is therefore the portability floor, not merely
a testing backend.

Why the GPL backend is explicit-only
------------------------------------

Scikit-Plots can interoperate with an installed external package without
making it part of the default automatic selection policy. This keeps automatic
behavior license-conservative and predictable.

Use the external backend only when your environment and distribution policy
have made that choice deliberately.
