:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-015:
.. _data-analytics-analyze-015:
.. _da-foundations-analyze-015:
.. _da-decisions-analyze-015:
.. _da-prep-analyze-015:
.. _da-cleaning-analyze-015:
.. _da-analyze-analyze-015:
.. _da-viz-analyze-015:
.. _da-python-analyze-015:
.. _da-jobsearch-analyze-015:

========================================================================
Using VLOOKUP to Combine Data Across Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🔗 Problem-Solving & Combining Data` :bdg-info:`Lesson 015`

◀ :doc:`Previous <014-preparing-data-for-vlookup-in-spreadsheets>` · :doc:`Next <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Performing the lookup
-----------------------

With the data prepared — clean keys, a leftmost lookup column, an adequate range —
VLOOKUP performs the actual combination: pulling matching values from one table
into another. This lesson covers the formula itself, its four arguments, and how it
brings separate tables together by a shared key.

The VLOOKUP formula
---------------------

VLOOKUP takes four arguments:

.. code-block:: text

   =VLOOKUP(lookup_value, table_range, column_index, FALSE)

- **lookup_value** — the value to search for (the key), e.g. this row's product
  code.
- **table_range** — the table to search, whose *first column* holds the keys and
  whose other columns hold the values to return.
- **column_index** — *which column* of that range to return, counted from the left
  (1 is the key column itself, 2 the next, and so on).
- **the match type** — ``FALSE`` (or ``0``) for an **exact match**, ``TRUE`` for an
  approximate match. **Almost always use ``FALSE``**: exact match is what combining
  data by key requires; ``TRUE`` (approximate) is for a different, rarer purpose and
  is a common source of silent wrong results.

.. code-block:: text

   =VLOOKUP(A2, products!A:C, 2, FALSE)

This reads: take the key in ``A2``, search the first column of ``products!A:C``,
and return the value from the *second* column of that range (an exact match). Fill
it down the column, and every row pulls its matching product detail from the
products sheet.

Combining across sheets
-------------------------

The power is combining *separate* tables. An orders sheet and a products sheet,
linked by product code, become one enriched view: each order row gains its
product's name and price via VLOOKUP, without manually copying anything. This is
exactly the relational combine — matching on a key to bring related data together —
performed in a spreadsheet, and it is the everyday tool for the "combine data from
two sources" task the cleaning section described.

VLOOKUP and the always-FALSE rule
-----------------------------------

The single most important habit: **use ``FALSE`` (exact match) unless you have a
specific reason not to.** Approximate match (``TRUE``) assumes the lookup column is
sorted and returns the *closest* value at or below the lookup value — which for
combining data by key is almost never what you want, and produces plausible *wrong*
matches with no error. New users who omit the argument or use ``TRUE`` get
mysterious wrong results; ``FALSE`` is the safe default for combining data, and
should be your reflex.

The caveat
------------

VLOOKUP returns the *first* match and only pulls *one* column per formula, which has
consequences: if the key is not unique in the lookup table, you silently get only
the first matching row's value (a data problem the preparation lesson flagged); and
returning several columns needs several VLOOKUPs (or a different function). VLOOKUP
also breaks if columns in the lookup range are inserted or deleted, since the
``column_index`` is a fixed number — a fragility that INDEX/MATCH and XLOOKUP avoid.
And a returned value that *looks* right can still be wrong if the keys did not truly
match as intended, so verify a sample of lookups against the source. The next
lesson turns to diagnosing exactly these VLOOKUP failures.

.. hint::

   - :doc:`Preparing Data for VLOOKUP in Spreadsheets <014-preparing-data-for-vlookup-in-spreadsheets>`
   - :doc:`Troubleshooting VLOOKUP and Building a Problem-Solving Framework <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>`
   - :doc:`Using JOIN in SQL to Combine Tables <017-using-join-in-sql-to-combine-tables>`
   - :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <010-working-with-strings-in-spreadsheets-len-left-right-find>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-vlookup-to-combine-data-across-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/using-vlookup-to-combine-data-across-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: combine
