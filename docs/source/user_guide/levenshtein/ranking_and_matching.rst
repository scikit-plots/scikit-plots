.. _levenshtein-ranking:

Ranking and matching
====================

.. currentmodule:: scikitplot.levenshtein

Stable ranking
--------------

:func:`rank` orders results by:

.. code-block:: text

   highest normalized similarity
   then lowest raw distance
   then original input position

The last rule makes ties deterministic.

.. code-block:: python

   from scikitplot.levenshtein import rank

   result = rank("mat", ["cat", "bat", "hat"], backend="python")
   assert [item.choice for item in result] == ["cat", "bat", "hat"]

Limit
-----

``limit`` bounds the number of returned matches.

With a positive limit the implementation keeps only ``O(limit)`` candidate
matches rather than materializing a complete match list before truncation.

``limit=0`` is special: it returns immediately and does not consume a generator.

Score cutoff
------------

``score_cutoff`` is normalized similarity in ``[0, 1]``:

.. code-block:: python

   rank(
       "kitten",
       ["kitten", "bitten", "completely different"],
       score_cutoff=0.7,
   )

Only matches at or above the threshold remain.

.. important::

   The cutoff is currently a **semantic post-computation filter**. It does not
   promise backend-native early termination. That distinction keeps behavior
   identical across internal, RapidFuzz and pure-Python implementations.

Custom extraction with ``key``
------------------------------

Use ``key`` when choices are objects or when preprocessing is application
policy:

.. code-block:: python

   records = [
       {"name": "scikit plots"},
       {"name": "scikit learn"},
   ]

   matches = rank(
       "Scikit Plots",
       records,
       key=lambda row: row["name"].casefold(),
   )

This is intentionally explicit; the distance functions themselves do no hidden
preprocessing.

Closest
-------

:func:`closest` is equivalent to ranking with ``limit=1``.

It returns ``None`` when:

- the iterable is empty; or
- ``score_cutoff`` filters every choice.
