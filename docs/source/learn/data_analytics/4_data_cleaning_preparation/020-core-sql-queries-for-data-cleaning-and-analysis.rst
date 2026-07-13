:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-020:
.. _data-analytics-cleaning-020:
.. _da-foundations-cleaning-020:
.. _da-decisions-cleaning-020:
.. _da-prep-cleaning-020:
.. _da-cleaning-cleaning-020:
.. _da-analyze-cleaning-020:
.. _da-viz-cleaning-020:
.. _da-python-cleaning-020:
.. _da-jobsearch-cleaning-020:

========================================================================
Core SQL Queries for Data Cleaning and Analysis
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 020`

◀ :doc:`Previous <019-spreadsheets-vs-sql>` · :doc:`Next <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The query patterns that clean
-------------------------------

SQL cleaning rests on a small set of **core query patterns** — combinations of
clauses that inspect, filter, and summarise data. These patterns do the bulk of
practical cleaning and analysis, and they extend the basic ``SELECT``/``FROM``/
``WHERE`` from the prep section with the tools that make queries genuinely
powerful. This lesson assembles the working toolkit.

Inspecting and filtering
--------------------------

The starting patterns find and isolate what needs attention:

.. code-block:: sql

   SELECT DISTINCT region          -- what distinct values exist? (consistency check)
   FROM   customers;

   SELECT *                        -- which rows have a problem?
   FROM   customers
   WHERE  spend < 0;               -- invalid negative values

   SELECT *
   FROM   customers
   WHERE  city IS NULL;            -- missing values (note: IS NULL, not = NULL)

``SELECT DISTINCT`` lists the unique values in a column — the SQL version of the
consistency-audit pivot, revealing whether "NY", "New York", and "new york" all
lurk in the data. ``WHERE`` isolates rows failing a validity rule. Note
``IS NULL`` (not ``= NULL``) is how SQL tests for missing values — a common
beginner trap, since ``= NULL`` never matches.

Summarising with GROUP BY
---------------------------

The pattern that turns rows into insight is ``GROUP BY``, which collapses rows
into groups and computes an aggregate per group:

.. code-block:: sql

   SELECT   region,
            COUNT(*)     AS n,
            AVG(spend)   AS avg_spend
   FROM     customers
   GROUP BY region
   ORDER BY n DESC;

This counts customers and averages spend *per region* — the SQL equivalent of a
pivot table, and one of the most-used analytical patterns. Grouping by a column
and counting is also a cleaning tool: ``GROUP BY email HAVING COUNT(*) > 1`` finds
duplicate emails (``HAVING`` filters *groups*, as ``WHERE`` filters rows).

The aggregate functions
-------------------------

``GROUP BY`` pairs with aggregate functions — the same operations as the
spreadsheet's: ``COUNT`` (how many), ``SUM`` (total), ``AVG`` (mean), ``MIN`` /
``MAX`` (extremes). These collapse many rows to one summary value, per group or
over the whole table. Recognising them as the familiar spreadsheet aggregates,
now in query form, is what makes SQL analysis feel like an extension of what you
already know rather than a new discipline.

Putting the patterns together
-------------------------------

Real cleaning and analysis chains these: filter to valid rows with ``WHERE``,
group with ``GROUP BY``, aggregate with ``COUNT``/``AVG``, order with
``ORDER BY``, and inspect distinct values with ``DISTINCT`` to check consistency
along the way. A handful of clauses, combined, answer a large share of real
questions — which is why SQL stays learnable despite its power, and why these
core patterns are the foundation the analysis section's advanced queries build
on.

The caveat
------------

The core patterns are precise about what they compute, which is not always what
you intend: ``COUNT(*)`` counts rows including nulls while ``COUNT(column)`` skips
nulls, ``AVG`` silently ignores nulls (changing the denominator), and a
``GROUP BY`` on an uncleaned column groups "NY" and "New York" *separately* —
producing a summary that looks authoritative but is fragmented by the very
dirtiness you were checking for. The lesson underneath: **clean before you
aggregate**, because summarising dirty data produces confident, wrong totals. The
next lessons apply these patterns to specific cleaning tasks — removing
duplicates, cleaning strings, and converting types.

.. hint::

   - :doc:`Introduction to SQL <018-introduction-to-sql>`
   - :doc:`Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   - :doc:`Using CAST to Clean and Format Data in SQL <022-using-cast-to-clean-and-format-data-in-sql>`
   - :doc:`Advanced SQL Functions for Data Cleaning <023-advanced-sql-functions-for-data-cleaning>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/core-sql-queries-for-data-cleaning-and-analysis/ <https://insightful-data-lab.com/2023/11/01/core-sql-queries-for-data-cleaning-and-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
