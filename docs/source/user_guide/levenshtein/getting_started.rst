.. _levenshtein-getting-started:

Getting started
===============

.. currentmodule:: scikitplot.levenshtein

Classic edit distance
---------------------

Levenshtein distance is the minimum number of single-element insertions,
deletions and substitutions needed to transform one sequence into another.

.. code-block:: python

   from scikitplot.levenshtein import distance

   distance("kitten", "sitting")
   # 3

The three edits are one substitution, another substitution, and one insertion.

Normalized similarity
---------------------

For easier comparison across different lengths:

.. code-block:: python

   from scikitplot.levenshtein import normalized_similarity

   score = normalized_similarity("kitten", "sitting")
   assert 0.0 <= score <= 1.0

``1.0`` means identical under the metric. ``0.0`` is the minimum normalized
similarity for unit-cost Levenshtein under this facade.

Find the closest choice
-----------------------

.. code-block:: python

   from scikitplot.levenshtein import closest

   match = closest("levnshtein", ["Levenshtein", "Hamming", "Jaccard"])
   print(match.choice, match.distance, match.similarity)

The returned :class:`Match` also records the backend that actually produced
the score.

Rank several choices
--------------------

.. code-block:: python

   from scikitplot.levenshtein import rank

   matches = rank(
       "kitten",
       ["sitting", "kitten", "bitten", "written"],
       limit=3,
   )

   assert [item.choice for item in matches] == [
       "kitten",
       "bitten",
       "sitting",
   ]

Next
----

Read :ref:`levenshtein-backends` before selecting an accelerator explicitly.
