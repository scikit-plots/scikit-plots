:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-022:
.. _data-analytics-python-022:
.. _da-foundations-python-022:
.. _da-decisions-python-022:
.. _da-prep-python-022:
.. _da-cleaning-python-022:
.. _da-analyze-python-022:
.. _da-viz-python-022:
.. _da-python-python-022:
.. _da-jobsearch-python-022:

========================================================================
Advanced Use of Loops, Lists, Tuples & List Comprehension
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 022`

◀ :doc:`Previous <021-tuples-in-python>` · :doc:`Next <023-dictionaries-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Combining structures and loops, concisely
-------------------------------------------

With lists, tuples, and loops in hand, this lesson covers using them *together* more
powerfully — iterating structures in richer ways — and introduces **list comprehension**, a
concise Python idiom for building lists that experienced Python programmers use constantly.
It marks the transition from basic structure use to fluent, idiomatic Python.

Richer iteration
----------------

Python offers cleaner ways to iterate structures than a bare index loop:

.. code-block:: python

   sales = [100, 250, 175]

   for i, amount in enumerate(sales):     # index AND value together
       print(i, amount)

   regions = ["N", "S", "E"]
   for region, amount in zip(regions, sales):   # iterate two lists in parallel
       print(region, amount)

``enumerate`` gives both the index and the item (cleaner than tracking an index manually);
``zip`` iterates several collections in lockstep (pairing regions with sales). These make
common iteration patterns readable — and ``zip`` pairs naturally with tuple unpacking
(``for region, amount in ...`` unpacks each pair).

List comprehension
------------------

**List comprehension** builds a list concisely in a single expression, replacing a
build-with-a-loop pattern:

.. code-block:: python

   # the loop way:
   doubled = []
   for x in sales:
       doubled.append(x * 2)

   # the comprehension way — same result, one line:
   doubled = [x * 2 for x in sales]           # [200, 500, 350]

The comprehension ``[expression for item in collection]`` reads as "the expression, for
each item" — building a new list by transforming each element. It can include a *condition*
to filter:

.. code-block:: python

   large = [x for x in sales if x > 150]      # [250, 175] — only items over 150

``[x for x in sales if x > 150]`` keeps only items meeting the condition — transformation
and filtering in one concise expression.

Why comprehensions matter
---------------------------

List comprehensions are idiomatic Python — the natural, readable way to build a list by
transforming or filtering another, replacing the more verbose loop-and-append. They express
"make a new list from this one" in a single clear line, and recognising and using them is a
mark of Python fluency. The pattern also connects forward: it is conceptually the same
element-wise transformation and boolean filtering that ``numpy`` and ``pandas`` do
*vectorised* (the libraries stage), so comprehensions bridge explicit loops and the
vectorised idioms ahead.

The caveat
------------

List comprehensions are powerful and can be *overused*. A simple transformation or filter
is clearer as a comprehension than a loop; but a comprehension with multiple conditions,
nested loops, or complex logic crammed into one line becomes *harder* to read than the
equivalent loop — the clarity-over-cleverness principle warns against the dense,
show-off comprehension. The guidance: use a comprehension when it is *more* readable (a
single clear transformation or filter), and fall back to an explicit loop when the logic is
complex enough that a comprehension would obscure it. Concise is good only when it is also
clear. The next lesson turns to a different structure: the dictionary.

.. hint::

   - :doc:`Tuples in Python <021-tuples-in-python>`
   - :doc:`Modifying Lists in Python <020-modifying-lists-in-python>`
   - :doc:`For Loops in Python <014-for-loops-in-python>`
   - :doc:`Dictionaries in Python <023-dictionaries-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/advanced-use-of-loops-lists-tuples-list-comprehension/ <https://insightful-data-lab.com/2023/12/06/advanced-use-of-loops-lists-tuples-list-comprehension/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures
