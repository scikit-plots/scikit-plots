:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-027:
.. _da-5-analyze-analyze-027:

========================================================================
Using GROUP BY and ORDER BY for Aggregated Calculations in SQL
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 027`

◀ :doc:`Previous <026-embedding-calculations-in-sql-queries>` · :doc:`Next <028-data-validation-as-an-ongoing-analytical-process>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The SQL pivot, complete
-------------------------

The calculation stage culminates where SQL aggregation began: **GROUP BY** to
compute aggregates per group, and **ORDER BY** to order the summary. Together they
are the SQL equivalent of the pivot table — grouping data into categories,
computing an aggregate for each, and presenting the result in a meaningful order.
This lesson assembles them into the complete aggregated-calculation pattern, closing
the calculation stage.

The core aggregated-calculation query
---------------------------------------

The pattern combines grouping, aggregating, and ordering:

.. code-block:: sql

   SELECT   region,
            COUNT(*)      AS orders,
            SUM(amount)   AS revenue,
            AVG(amount)   AS avg_order
   FROM     orders
   GROUP BY region
   ORDER BY revenue DESC;

This groups orders by region, computes each region's order count, total revenue,
and average order value, and orders the result by revenue, highest first — a
complete regional summary in one query. It is exactly a pivot table (region as the
grouping, the aggregates as values, sorted) expressed as SQL, and it answers the
"summarise X by Y, ranked" question directly.

Ordering the summary
----------------------

``ORDER BY`` on an aggregated query orders the *groups*, and can order by an
aggregate:

- ``ORDER BY revenue DESC`` — regions from highest revenue to lowest, surfacing the
  top performers.
- ``ORDER BY COUNT(*) DESC`` — groups by how many rows each contains.
- ``ORDER BY region ASC`` — the groups in category order.

Ordering by an aggregate is what turns a summary into a *ranking* — "regions by
revenue", "products by units sold" — one of the most common analytical outputs.

The full analytical query
---------------------------

Combined with the earlier clauses, the complete pattern layers filtering,
grouping, group-filtering, and ordering:

.. code-block:: sql

   SELECT   region, SUM(amount) AS revenue
   FROM     orders
   WHERE    order_date >= '2024-01-01'    -- filter rows first
   GROUP BY region                        -- group
   HAVING   SUM(amount) > 10000           -- filter groups
   ORDER BY revenue DESC;                 -- order the result

This reads as a complete analytical sentence: *from* the orders, *where* they are
recent, *grouped by* region, *keeping* high-revenue regions, *ordered by* revenue.
The clause order (``WHERE`` → ``GROUP BY`` → ``HAVING`` → ``ORDER BY``) is both the
required SQL syntax and the logical sequence of the analysis, and mastering it is
mastering SQL aggregation.

The caveat
------------

The full pattern concentrates the section's precision traps in one place: the
``WHERE``/``HAVING`` distinction (rows before grouping, aggregates after), nulls
interacting with aggregates (``COUNT(*)`` versus ``COUNT(column)``, ``AVG`` ignoring
nulls), and the requirement that every non-aggregated column in ``SELECT`` appear in
``GROUP BY`` (or the query errors or, in some databases, returns arbitrary values).
A grouped-calculation query that looks right can be subtly wrong, so the
build-incrementally-and-verify discipline is essential — get the grouping right,
add each clause, check the result against expectation. This closes the calculation
stage; the final stage of the section covers advanced analytical techniques,
including temporary tables.

.. hint::

   - :doc:`Embedding Calculations in SQL Queries <026-embedding-calculations-in-sql-queries>`
   - :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL <019-aggregating-data-with-subqueries-having-and-case-in-sql>`
   - :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   - :doc:`Comparing Calculations in Spreadsheets and SQL <025-comparing-calculations-in-spreadsheets-and-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-group-by-and-order-by-for-aggregated-calculations-in-sql/ <https://insightful-data-lab.com/2023/11/26/using-group-by-and-order-by-for-aggregated-calculations-in-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
