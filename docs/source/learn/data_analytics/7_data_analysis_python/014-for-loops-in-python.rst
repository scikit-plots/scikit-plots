:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-014:
.. _da-7-python-python-014:

========================================================================
For Loops in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🔀 Control Flow` :bdg-info:`Lesson 014`

◀ :doc:`Previous <013-while-loops-and-iteration-in-python>` · :doc:`Next <015-range-function-and-loop-control-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Repeating over a collection
-----------------------------

When you need to do something *for each item* in a collection, the **for loop** is the
natural tool — it iterates over the items of a sequence, running its block once per item.
This is the loop analysts use most, because data work is largely "do this to every row,
value, or record." This lesson covers the for loop, the workhorse of iteration.

The for loop
------------

A ``for`` loop iterates over the items of a collection, binding each to a variable in
turn:

.. code-block:: python

   regions = ["North", "South", "East", "West"]
   for region in regions:
       print(region)

Python takes each item of ``regions`` in order, assigns it to ``region``, and runs the
block — printing all four region names. The loop variable (``region``) holds the current
item on each pass. Unlike the while loop, the for loop *automatically* stops when the
collection is exhausted — no manual progress-tracking, and no risk of an infinite loop
over a finite collection.

Iterating and accumulating
----------------------------

A common pattern combines a for loop with a variable that *accumulates* a result across
iterations:

.. code-block:: python

   sales = [100, 250, 175, 300]
   total = 0
   for amount in sales:
       total += amount        # accumulate
   print(total)               # 825

The ``total`` starts at zero and grows by each amount as the loop visits it — computing a
sum by iteration. This accumulate-across-a-loop pattern (summing, counting, collecting,
building) is one of the most useful in programming, and it is how manual aggregation is
expressed in code (though pandas, later, does it far more concisely).

For loops over different collections
--------------------------------------

For loops iterate any *iterable* — lists, strings (character by character), dictionaries
(the structures stage), and more:

.. code-block:: python

   for char in "data":        # iterates characters: d, a, t, a
       print(char)

This generality makes the for loop the standard way to process collections of any kind —
whatever the data, "for each item, do something" is a for loop.

The caveat
------------

For loops are safer than while loops (they cannot loop infinitely over a finite
collection), but they have their own pitfalls. Modifying a collection *while* iterating
over it causes subtle bugs (the collection changes underfoot) and should be avoided —
build a new collection instead. And a deeper point looms for data work: explicit Python
for loops over large datasets are *slow* compared to the vectorised operations of
``numpy`` and ``pandas`` (the libraries stage), which do the same work far faster without
an explicit loop. Loops are essential to understand and correct for general programming,
but for large-scale data the idiom shifts to vectorised operations — a for loop over a
million-row dataset is usually the wrong tool. Learn loops thoroughly, and later learn
when *not* to loop. The next lesson covers generating sequences and controlling loops.

.. hint::

   - :doc:`While Loops and Iteration in Python <013-while-loops-and-iteration-in-python>`
   - :doc:`range() Function and Loop Control in Python <015-range-function-and-loop-control-in-python>`
   - :doc:`Data Types vs Data Structures & Introduction to Lists <019-data-types-vs-data-structures-and-introduction-to-lists>`
   - :doc:`Branching and Conditional Statements in Python <012-branching-and-conditional-statements-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/for-loops-in-python/ <https://insightful-data-lab.com/2023/12/06/for-loops-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: control
