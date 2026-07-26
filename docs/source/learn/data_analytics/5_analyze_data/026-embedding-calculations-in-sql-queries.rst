:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-026:
.. _da-5-analyze-analyze-026:

========================================================================
Embedding Calculations in SQL Queries
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 026`

◀ :doc:`Previous <025-comparing-calculations-in-spreadsheets-and-sql>` · :doc:`Next <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Computing inside the query
----------------------------

SQL does not only *retrieve* data — it *computes* on it, producing derived values as
part of the query result. **Embedding calculations in SQL queries** means writing
arithmetic and expressions directly in the ``SELECT`` clause, so the database
returns computed columns alongside stored ones. This is where SQL becomes a
calculation engine, not just a retrieval tool.

Calculated columns in SELECT
------------------------------

Any expression in the ``SELECT`` clause produces a computed column:

.. code-block:: sql

   SELECT product,
          quantity,
          price,
          quantity * price               AS line_total,
          price * 0.9                    AS discounted_price,
          (revenue - cost) / revenue     AS margin
   FROM   orders;

Each expression — ``quantity * price``, ``price * 0.9``, the margin formula — is
computed per row and returned as a new column (named with ``AS``). The arithmetic
operators (``+``, ``-``, ``*``, ``/``) work as expected, and expressions can combine
columns, constants, and functions. This is the SQL equivalent of a spreadsheet
formula column, computed by the database.

Calculations with functions
------------------------------

Embedded calculations extend beyond arithmetic to SQL's functions:

- **String calculations** — ``CONCAT``, ``SUBSTR``, and the string functions from
  earlier, producing derived text.
- **Date calculations** — extracting parts of dates, computing differences between
  dates (days between order and ship), which are central to trend and duration
  analysis.
- **Conditional calculations** — ``CASE`` expressions producing values that depend
  on conditions (the conditional logic from the combine stage).
- **Type conversions** — ``CAST`` within a calculation, ensuring arithmetic happens
  on the right types.

Combining these, a single ``SELECT`` can compute a rich set of derived values from
the stored data — the analytical columns a question needs, produced at query time.

Why compute in the query
--------------------------

Embedding calculations in the query, rather than retrieving raw data and computing
elsewhere, has real advantages: the computation happens *at the source* on the full
data (no exporting), the query is *reproducible* (the calculation is recorded in the
query text and reruns on new data), and only the *results* travel, not millions of
raw rows. It keeps the calculation logic with the data and the query, which is
cleaner and more scalable than pulling data out to compute on it.

The caveat
------------

Embedded calculations inherit SQL's precision demands. **Integer division** is a
classic trap — in many databases, ``5 / 2`` yields ``2`` (integer division) rather
than ``2.5``, so a ratio computed on integer columns can be silently wrong unless
you cast to a decimal type first. **Nulls propagate** — any arithmetic involving a
null yields null (``price * quantity`` is null if either is null), so calculated
columns can be unexpectedly empty. And **division by zero** errors or yields null
depending on the database. As always, the calculation returns exactly what the
expression specifies — verify that derived columns hold the values you intend,
checking especially division, nulls, and types. The next lesson combines embedded
calculation with grouping — the SQL pivot.

.. hint::

   - :doc:`Comparing Calculations in Spreadsheets and SQL <025-comparing-calculations-in-spreadsheets-and-sql>`
   - :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <../4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Aggregating Data with Subqueries, HAVING, and CASE in SQL <019-aggregating-data-with-subqueries-having-and-case-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/embedding-calculations-in-sql-queries/ <https://insightful-data-lab.com/2023/11/26/embedding-calculations-in-sql-queries/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
