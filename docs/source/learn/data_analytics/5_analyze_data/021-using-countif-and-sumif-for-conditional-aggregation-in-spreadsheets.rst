:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-021:
.. _da-5-analyze-analyze-021:

========================================================================
Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 021`

◀ :doc:`Previous <020-using-spreadsheet-formulas-for-sales-trend-analysis>` · :doc:`Next <022-using-sumproduct-for-advanced-spreadsheet-calculations>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Aggregating with a condition
------------------------------

Plain ``SUM`` and ``COUNT`` aggregate *everything*; analysis usually needs to
aggregate only the rows meeting a *condition* — sales in one region, orders above a
threshold, customers of one type. **COUNTIF** and **SUMIF** are the spreadsheet's
conditional-aggregation functions, and they are among the most-used analytical
tools, the spreadsheet counterparts of SQL's ``COUNT``/``SUM`` with ``WHERE``.

COUNTIF and SUMIF
-------------------

- ``COUNTIF(range, condition)`` — counts the cells in a range that meet a
  condition:

  .. code-block:: text

     =COUNTIF(region, "North")        how many northern orders
     =COUNTIF(amount, ">100")         how many orders over 100

- ``SUMIF(range, condition, sum_range)`` — sums the values in ``sum_range`` for the
  rows where ``range`` meets the condition:

  .. code-block:: text

     =SUMIF(region, "North", amount)  total revenue from northern orders
     =SUMIF(amount, ">100", amount)   total of all orders over 100

``COUNTIF`` answers "how many meeting X"; ``SUMIF`` answers "the total of Y for rows
meeting X." Together they compute the conditional counts and totals that most
analytical questions reduce to.

Multiple conditions: COUNTIFS and SUMIFS
------------------------------------------

For *several* conditions at once, the plural forms ``COUNTIFS`` and ``SUMIFS`` take
multiple range-condition pairs:

.. code-block:: text

   =COUNTIFS(region, "North", amount, ">100")
   =SUMIFS(amount, region, "North", month, "January")

``COUNTIFS`` counts rows meeting *all* the conditions (northern *and* over 100);
``SUMIFS`` sums for rows meeting all conditions. These handle the segmented
questions — "January revenue in the northern region" — that a single condition
cannot express, and they are the workhorses of spreadsheet analysis.

Why conditional aggregation matters
-------------------------------------

Most analytical questions are conditional: not "total sales" but "sales *in this
segment*", not "how many orders" but "how many *of this type*". ``COUNTIF`` and
``SUMIF`` (and their plural forms) are how a spreadsheet answers these directly,
without first filtering the data by hand. They are also the conceptual bridge to
SQL's ``WHERE`` plus aggregate and to pivot tables — the same "aggregate a subset"
idea in three forms, which is why recognising the pattern here pays off repeatedly.

The caveat
------------

Conditional-aggregation functions are precise about their conditions, and small
mistakes mislead: a condition written as text must match exactly (``"North"`` will
not catch ``"north"`` or ``"North "`` with a space — the cleaning issues resurface),
and the condition syntax for comparisons (``">100"`` in quotes) trips up beginners.
The ranges must also align — ``SUMIF``'s condition range and sum range must be the
same size and correspond row-for-row, or the result is silently wrong. As always,
verify a conditional total against a hand-check or an order-of-magnitude estimate:
a ``SUMIF`` that returns an implausible number usually has a condition or range
error. The next lesson covers a more advanced calculation function: SUMPRODUCT.

.. hint::

   - :doc:`Using Spreadsheet Formulas for Sales Trend Analysis <020-using-spreadsheet-formulas-for-sales-trend-analysis>`
   - :doc:`Using SUMPRODUCT for Advanced Spreadsheet Calculations <022-using-sumproduct-for-advanced-spreadsheet-calculations>`
   - :doc:`Spreadsheet Functions <../2_data_driven_decisions/016-spreadsheet-functions>`
   - :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL <019-aggregating-data-with-subqueries-having-and-case-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
