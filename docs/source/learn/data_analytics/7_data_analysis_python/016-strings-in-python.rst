:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-016:
.. _data-analytics-python-016:
.. _da-foundations-python-016:
.. _da-decisions-python-016:
.. _da-prep-python-016:
.. _da-cleaning-python-016:
.. _da-analyze-python-016:
.. _da-viz-python-016:
.. _da-python-python-016:
.. _da-jobsearch-python-016:

========================================================================
Strings in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`📚 Strings & Data Structures` :bdg-info:`Lesson 016`

◀ :doc:`Previous <015-range-function-and-loop-control-in-python>` · :doc:`Next <017-string-indexing-and-slicing-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Working with text
------------------

Data is full of text — names, categories, codes, addresses — and Python's **string** type
is how text is represented and manipulated. Opening the data-structures stage, this lesson
covers strings in Python: creating them, their operations, and the crucial property of
immutability. It extends the string work from the spreadsheet and SQL sections into
Python.

Creating and combining strings
--------------------------------

A **string** is text, written in single or double quotes:

.. code-block:: python

   name = "North Region"
   code = 'NR-001'

Strings combine and repeat with operators:

.. code-block:: python

   greeting = "Hello, " + name       # concatenation: "Hello, North Region"
   line = "-" * 20                    # repetition: 20 dashes

The ``+`` concatenates strings (as in the spreadsheet's ``&`` and SQL's ``CONCAT``), and
``*`` repeats a string — the basic ways to build text.

String methods
--------------

Strings are objects (the OOP lesson) with many useful methods, mirroring the string
functions from earlier sections:

.. code-block:: python

   text = "  North Region  "
   text.strip()              # "North Region" — remove surrounding whitespace (like TRIM)
   text.upper()              # "  NORTH REGION  " — uppercase
   text.lower()              # lowercase
   text.replace("North", "South")   # substitute (like SUBSTITUTE / REPLACE)
   "NR-001".split("-")       # ["NR", "001"] — split on a delimiter
   len("North")              # 5 — length (like LEN)

These are the same cleaning and manipulation operations from the spreadsheet (``TRIM``,
``UPPER``, ``SUBSTITUTE``) and SQL (``TRIM``, ``UPPER``, ``REPLACE``, ``SUBSTR``) — now as
Python string methods, called with dot notation on the string object.

String immutability
--------------------

A crucial property: strings in Python are **immutable** — once created, a string cannot be
changed in place. String methods do not modify the original; they *return a new string*:

.. code-block:: python

   text = "north"
   text.upper()              # returns "NORTH", but...
   print(text)               # still "north" — unchanged!
   text = text.upper()       # to keep the result, reassign
   print(text)               # now "NORTH"

This catches many beginners: calling ``text.upper()`` does not change ``text``; you must
*assign* the result back. Immutability means string operations produce new strings, and
using the result requires capturing it — a fundamental and frequently-forgotten point.

Why strings matter
------------------

Text manipulation is constant in data work — cleaning categories, parsing codes, extracting
parts, formatting output — and Python's string methods are the tools for all of it, more
flexible than their spreadsheet and SQL counterparts. Because so much real data is text
(or arrives as text needing conversion, the type lesson), fluency with strings is
essential to Python data analysis. The following lessons go deeper into indexing,
slicing, and formatting strings.

The caveat
------------

String immutability is the pitfall to internalise: the single commonest string mistake is
calling a method and expecting the original to change — ``text.strip()`` on its own does
nothing lasting; you must write ``text = text.strip()``. Every string "modification" is
really "create a new string and (usually) reassign." Beyond that, strings carry the
encoding and special-character subtleties of all text (the Unicode considerations), and
splitting or extracting assumes a structure that real text may not consistently have (the
defensive-extraction point from the spreadsheet strings lesson applies). Capture method
results, and handle text's irregularity. The next lesson covers reaching into strings by
position: indexing and slicing.

.. hint::

   - :doc:`String Indexing and Slicing in Python <017-string-indexing-and-slicing-in-python>`
   - :doc:`String Formatting with .format() in Python <018-string-formatting-with-format-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <../5_analyze_data/010-working-with-strings-in-spreadsheets-len-left-right-find>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/strings-in-python/ <https://insightful-data-lab.com/2023/12/06/strings-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: structures
