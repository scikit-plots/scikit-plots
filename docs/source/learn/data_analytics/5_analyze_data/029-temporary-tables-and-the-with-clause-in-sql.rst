:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-029:
.. _data-analytics-analyze-029:
.. _da-foundations-analyze-029:
.. _da-decisions-analyze-029:
.. _da-prep-analyze-029:
.. _da-cleaning-analyze-029:
.. _da-analyze-analyze-029:
.. _da-viz-analyze-029:
.. _da-python-analyze-029:
.. _da-jobsearch-analyze-029:

========================================================================
Temporary Tables and the WITH Clause in SQL
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🚀 Validation & Temporary Tables` :bdg-info:`Lesson 029`

◀ :doc:`Previous <028-data-validation-as-an-ongoing-analytical-process>` · :doc:`Next <030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Holding intermediate results
------------------------------

Complex analysis often needs to compute an intermediate result and then work with
it — and writing everything as one deeply nested query becomes unreadable. SQL
offers ways to hold intermediate results: **temporary tables** and, most usefully,
the **WITH clause** (common table expressions). They let you break a complex
analysis into named, readable steps, which is essential for the advanced queries
this stage covers.

The WITH clause (common table expressions)
--------------------------------------------

The ``WITH`` clause defines a named, temporary result set that exists for the
duration of a single query — a **common table expression (CTE)**:

.. code-block:: sql

   WITH regional_revenue AS (
       SELECT region, SUM(amount) AS revenue
       FROM   orders
       GROUP BY region
   )
   SELECT region, revenue
   FROM   regional_revenue
   WHERE  revenue > 10000
   ORDER BY revenue DESC;

The ``WITH`` block computes ``regional_revenue`` (revenue per region); the main
query then treats it as if it were a table — filtering and ordering it. This does
the same work as a subquery in the ``FROM`` clause, but *named and readable*: the
intermediate step has a name and stands on its own, rather than being buried inside
the outer query.

Why CTEs improve complex queries
----------------------------------

CTEs make complex analysis manageable in ways that matter:

- **Readability** — each step is named and separate, so a multi-step analysis reads
  top to bottom as a sequence rather than as nested parentheses read inside-out.
- **Reuse within the query** — a CTE can be referenced multiple times in the main
  query, computing it once.
- **Chaining** — multiple CTEs can build on each other (``WITH a AS (...), b AS
  (SELECT ... FROM a)``), expressing a genuine multi-step pipeline as a readable
  sequence.
- **Debugging** — each CTE can be tested on its own by selecting from it, isolating
  where a problem is (the isolate step of the problem-solving framework).

For analytical queries of any complexity, CTEs are the standard way to keep them
comprehensible.

Temporary tables
------------------

A **temporary table** is a table that exists only for a session, holding
intermediate results that persist across *multiple* queries (unlike a CTE, which
lives only within one query). When an analysis runs several queries against the same
intermediate result, computing it once into a temporary table and querying that
repeatedly is more efficient than recomputing it each time. The next lesson covers
creating them in detail.

The caveat
------------

CTEs and temporary tables serve overlapping but distinct needs, and choosing wrongly
costs clarity or performance. A **CTE** lives for one query and is ideal for
structuring a single complex query readably; a **temporary table** persists across a
session and suits reusing an intermediate result across several queries. Overusing
either — wrapping trivial queries in CTEs, or creating temp tables for one-time use —
adds complexity without benefit. And in some databases a CTE is recomputed each time
it is referenced, which can be slow if referenced repeatedly on large data (where a
temp table would be better). Match the tool to whether the intermediate result is
used once or many times. The next lesson details creating temporary tables.

.. hint::

   - :doc:`Creating Temporary Tables in SQL — Methods, Trade-offs, and Best Practices <030-creating-temporary-tables-in-sql-methods-trade-offs-and-best-practices>`
   - :doc:`Subqueries in SQL <018-subqueries-in-sql>`
   - :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL <019-aggregating-data-with-subqueries-having-and-case-in-sql>`
   - :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/temporary-tables-and-the-with-clause-in-sql/ <https://insightful-data-lab.com/2023/11/26/temporary-tables-and-the-with-clause-in-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: advanced
