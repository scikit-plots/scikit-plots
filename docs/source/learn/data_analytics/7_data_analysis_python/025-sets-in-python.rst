:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-025:
.. _data-analytics-python-025:
.. _da-foundations-python-025:
.. _da-decisions-python-025:
.. _da-prep-python-025:
.. _da-cleaning-python-025:
.. _da-analyze-python-025:
.. _da-viz-python-025:
.. _da-python-python-025:
.. _da-jobsearch-python-025:

========================================================================
Sets in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 025`

◀ :doc:`Previous <024-advanced-dictionary-usage-in-python>` · :doc:`Next <026-libraries-packages-and-modules-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Unordered collections of unique values
----------------------------------------

The last core data structure is the **set** — an unordered collection of *unique* values.
Sets excel at membership testing, removing duplicates, and set operations (union,
intersection), and understanding them completes Python's built-in structures. This lesson
covers sets, closing the structures stage.

What a set is
-------------

A **set** is a collection with two defining properties: its values are **unique** (no
duplicates) and **unordered** (no positional index). Written with curly braces (like a
dictionary, but values not pairs):

.. code-block:: python

   regions = {"North", "South", "East", "North"}   # duplicate ignored
   print(regions)            # {"North", "South", "East"} — 3 unique values

Adding a duplicate has no effect (the value is already present), and there is no
``set[0]`` — sets are not indexed. Their purpose is *membership* and *uniqueness*, not
order.

Set operations
--------------

Sets support fast membership testing and mathematical set operations:

.. code-block:: python

   "North" in regions       # True — fast membership test

   a = {1, 2, 3}
   b = {2, 3, 4}
   a | b                     # {1, 2, 3, 4} — union (in either)
   a & b                     # {2, 3} — intersection (in both)
   a - b                     # {1} — difference (in a but not b)

Membership testing (``in``) is *very fast* on a set — faster than searching a list — and the
operations (union ``|``, intersection ``&``, difference ``-``) answer "in either / both /
one but not the other" directly. These are exactly the set operations of mathematics, and of
SQL's ``UNION``/``INTERSECT``/``EXCEPT``.

Why sets matter
---------------

Sets serve specific, common needs:

- **Removing duplicates** — converting a list to a set drops duplicates instantly
  (``set(my_list)``), the fastest deduplication in Python.
- **Fast membership testing** — checking whether a value is in a large collection is far
  faster with a set than a list, which matters at scale.
- **Comparing collections** — the set operations answer "what is common / different between
  these two collections?" directly (which customers are in both lists? which are only in
  one?).

For these tasks — uniqueness, membership, comparison — the set is the right tool, cleaner and
faster than working around a list.

The caveat
------------

Sets' properties are also their limitations. Being *unordered*, a set cannot be indexed or
sliced, and does not preserve insertion order — if order matters, a set is the wrong
structure (use a list). Being *unique*, a set cannot hold duplicates — which is the point
for deduplication, but means a set cannot represent data where repetition is meaningful (a
set of sales figures would collapse identical amounts into one, losing information). And set
elements must be *immutable* (like dictionary keys), so a set cannot contain lists. Use a set
precisely when uniqueness and membership are what you want, and a list when order or
repetition matters. This completes Python's core data structures; the next lessons open the
final stage — the libraries that make Python a data-analysis powerhouse.

.. hint::

   - :doc:`Dictionaries in Python <023-dictionaries-in-python>`
   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
   - :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   - :doc:`Libraries, Packages, and Modules in Python <026-libraries-packages-and-modules-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/sets-in-python/ <https://insightful-data-lab.com/2023/12/06/sets-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures
