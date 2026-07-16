:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-019:
.. _da-7-python-python-019:

========================================================================
Data Types vs Data Structures & Introduction to Lists
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 019`

◀ :doc:`Previous <018-string-formatting-with-format-in-python>` · :doc:`Next <020-modifying-lists-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


From single values to collections
-----------------------------------

So far, most values have been *single* — one number, one string. But data is usually
*many* values together, and holding collections requires **data structures**. This lesson
draws the distinction between a **data type** (the kind of a single value) and a **data
structure** (an organised collection of values), and introduces the most fundamental
structure: the **list**.

Data types versus data structures
------------------------------------

The distinction is foundational:

- A **data type** describes a *single* value's kind — ``int``, ``float``, ``str``,
  ``bool`` (the types lesson). It answers "what kind of value is this?"
- A **data structure** *organises multiple* values into a collection with a particular
  arrangement and behaviour — lists, tuples, dictionaries, sets (this stage). It answers
  "how are these many values held together?"

A single sales figure is a value of type ``float``; a *collection* of sales figures is a
data structure (a list). Types classify individual values; structures organise many. Both
matter: you choose a type for each value and a structure for how values are grouped.

The list
---------

A **list** is an ordered, changeable collection of values, written in square brackets:

.. code-block:: python

   sales = [100, 250, 175, 300]
   regions = ["North", "South", "East", "West"]
   mixed = [1, "two", 3.0, True]        # lists can hold mixed types

A list holds items *in order*, each accessible by index (the same zero-based indexing and
slicing as strings):

.. code-block:: python

   sales[0]          # 100 — first item
   sales[-1]         # 300 — last item
   sales[1:3]        # [250, 175] — a slice (a sub-list)
   len(sales)        # 4 — number of items

Lists are the workhorse structure for holding sequences of data — a column of values, a
series of records, any ordered collection.

Why lists matter
----------------

Lists are the most-used Python data structure, and the foundation for much data work. They
hold the collections that loops iterate over (the for-loop lesson), the sequences that get
transformed and aggregated, and conceptually they underlie the columns and series of
``pandas`` (the libraries stage — a DataFrame column is list-like). Understanding lists —
ordered, indexed, changeable collections — is understanding the basic shape of "many
values together" that all of data analysis works with.

The caveat
------------

Lists' flexibility invites a few missteps. They can hold *mixed types*, which is
occasionally useful but often a sign of disorganised data — a list meant to hold sales
figures should hold numbers, not a stray string, or later operations break. The zero-based
indexing and exclusive-stop slicing carry over from strings, with the same off-by-one
hazards. And a subtle point the next lessons develop: lists are **mutable** (changeable in
place), which is powerful but means a list can be modified unexpectedly if shared — the
opposite of strings' immutability, and a source of surprising bugs. Keep lists
type-consistent, mind the indexing, and be aware that lists change in place. The next lesson
covers modifying them.

.. hint::

   - :doc:`Modifying Lists in Python <020-modifying-lists-in-python>`
   - :doc:`Tuples in Python <021-tuples-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`For Loops in Python <014-for-loops-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/data-types-vs-data-structures-introduction-to-lists/ <https://insightful-data-lab.com/2023/12/06/data-types-vs-data-structures-introduction-to-lists/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures
