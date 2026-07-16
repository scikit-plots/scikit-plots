:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-025:
.. _da-5-analyze-analyze-025:

========================================================================
Comparing Calculations in Spreadsheets and SQL
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 025`

◀ :doc:`Previous <024-using-pivot-table-filters-and-calculated-fields-for-deeper-analysis>` · :doc:`Next <026-embedding-calculations-in-sql-queries>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The same calculation, two tools
---------------------------------

Having computed aggregates in both spreadsheets and SQL, it is worth comparing them
directly — because the *same* calculation is often expressible in either, and
knowing how they correspond (and where each suits) makes you fluent across both.
This lesson maps the correspondence, consolidating the calculation stage.

The direct correspondences
----------------------------

Most calculations translate cleanly between the two:

- **Aggregate a column** — spreadsheet ``=SUM(amount)`` ↔ SQL
  ``SELECT SUM(amount) FROM orders``.
- **Conditional aggregate** — spreadsheet ``=SUMIF(region,"North",amount)`` ↔ SQL
  ``SELECT SUM(amount) FROM orders WHERE region='North'``.
- **Group and aggregate** — spreadsheet *pivot table* (region → rows, sum of
  amount → values) ↔ SQL ``SELECT region, SUM(amount) FROM orders GROUP BY
  region``.
- **Filter then aggregate** — spreadsheet filter + ``SUM`` ↔ SQL ``WHERE`` +
  ``SUM``.

The pattern is consistent: the spreadsheet's ``SUMIF``/``COUNTIF`` and pivot tables
are the *same operations* as SQL's ``WHERE`` plus aggregates and ``GROUP BY``. The
concepts are identical; only the expression differs — a formula and a menu in one,
a query in the other.

Where each suits the calculation
----------------------------------

The correspondence does not make them interchangeable in practice:

- **Spreadsheets suit** small data, interactive exploration, calculations you want
  to *see* and adjust cell by cell, and results a stakeholder will open. A quick
  pivot to explore is often faster than writing a query.
- **SQL suits** large data, calculations that must be *reproducible* and rerun on
  new data, complex multi-table aggregation, and computation at the source. A
  calculation over millions of rows, or one that runs every week, belongs in SQL.

The deciding factors are the familiar ones — size, repetition, complexity, audience
— applied to the specific calculation.

Why the comparison matters
----------------------------

Seeing the calculations as *the same operations in different tools* is what makes an
analyst tool-fluent rather than tool-bound. It means you can prototype a calculation
in a spreadsheet where it is quick to see, then translate it to SQL when it needs to
scale or recur — and recognise that a ``GROUP BY`` query and a pivot table are the
same idea, so learning one deepens the other. The tools are different expressions of
one analytical vocabulary.

The caveat
------------

The correspondence is close but not perfect, and the gaps cause errors. The tools
can **handle edge cases differently** — nulls, blanks, text-versus-number, rounding
— so the "same" calculation can give subtly different results in each (a spreadsheet
average that skips blank cells versus a SQL ``AVG`` that ignores nulls may or may not
match, depending on the data). Translating a calculation between tools therefore
requires *verifying the results match*, not assuming they do. Use the comparison to
move fluently between tools, but check that a translated calculation reproduces the
original, especially around missing values. The next lessons go deeper into SQL
calculation.

.. hint::

   - :doc:`Using Pivot Tables for Calculations and Trend Analysis <023-using-pivot-tables-for-calculations-and-trend-analysis>`
   - :doc:`Embedding Calculations in SQL Queries <026-embedding-calculations-in-sql-queries>`
   - :doc:`Spreadsheets vs. SQL <../4_data_cleaning_preparation/019-spreadsheets-vs-sql>`
   - :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/comparing-calculations-in-spreadsheets-and-sql/ <https://insightful-data-lab.com/2023/11/26/comparing-calculations-in-spreadsheets-and-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
