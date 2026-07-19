:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-017:
.. _da-5-analyze-analyze-017:

========================================================================
Using JOIN in SQL to Combine Tables
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🔗 Problem-Solving & Combining Data` :bdg-info:`Lesson 017`

◀ :doc:`Previous <016-troubleshooting-vlookup-and-building-a-problem-solving-framework>` · :doc:`Next <018-subqueries-in-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The SQL way to combine tables
-------------------------------

VLOOKUP combines tables in a spreadsheet; **JOIN** does it in SQL — and far more
powerfully. A JOIN combines rows from two (or more) tables based on a related
column between them, producing a single result set that draws columns from each.
It is the operation the relational model from the prep section was built for, and
one of the most important skills in all of SQL.

How a JOIN works
------------------

A JOIN matches rows across tables on a **key** — the primary/foreign key
relationship from the prep section made operational:

.. code-block:: sql

   SELECT orders.order_id,
          orders.amount,
          customers.name,
          customers.region
   FROM   orders
   JOIN   customers
     ON   orders.customer_id = customers.customer_id;

The ``ON`` clause states the match condition — where ``orders.customer_id`` equals
``customers.customer_id`` — and the result pairs each order with its customer's
details, drawing columns from both tables. This is the relational combine: data
stored separately (orders in one table, customers in another) brought together by
their key relationship.

The types of JOIN
-------------------

Which rows appear depends on the *join type*:

- **INNER JOIN** (the default ``JOIN``) — returns only rows with a match in *both*
  tables. Orders whose customer is missing, and customers with no orders, are
  excluded. The most common join.
- **LEFT JOIN** — returns *all* rows from the left table, matched where possible,
  with nulls where the right table has no match. "All orders, with customer
  details where available" — keeps unmatched left rows.
- **RIGHT JOIN** — the mirror: all rows from the right table, nulls where the left
  has no match.
- **FULL (OUTER) JOIN** — all rows from *both* tables, matched where possible,
  nulls elsewhere.

The choice matters: an INNER JOIN silently *drops* unmatched rows (which may be
exactly what you want, or a silent loss of data), while a LEFT JOIN *keeps* them —
so choosing the wrong type changes which rows your analysis sees.

JOIN versus VLOOKUP
--------------------

JOIN is VLOOKUP's more capable relative. Where VLOOKUP pulls *one column* from
*one* lookup table by a leftmost key and returns the *first* match, a JOIN combines
*all needed columns* from *multiple* tables, on *any* matching columns (not just a
leftmost one), handling multiple matches explicitly. For combining data at any
scale or complexity, JOIN is the tool — which is why SQL is preferred over
spreadsheets for serious multi-table work.

The caveat
------------

JOINs have a signature danger: joining on a **non-unique key** multiplies rows. If a
customer has three orders and you join in a way that matches each order to each of
two addresses, you get *six* rows where you expected three — a "fan-out" that
silently inflates counts and sums. This is the SQL version of the merge row-count
explosion from the cleaning section, and the defence is the same: **check the row
count** before and after the join against expectation, and understand the *
cardinality* of the relationship (one-to-one, one-to-many, many-to-many) before
joining. A JOIN that returns more rows than expected has usually matched on a key
that was not as unique as assumed. The next lessons add nesting queries within
queries — subqueries.

.. hint::

   - :doc:`Using VLOOKUP to Combine Data Across Spreadsheets <015-using-vlookup-to-combine-data-across-spreadsheets>`
   - :doc:`Databases and Relational Database Concepts <../3_data_preparation/016-databases-and-relational-database-concepts>`
   - :doc:`Subqueries in SQL <018-subqueries-in-sql>`
   - :doc:`Cleaning and Merging Multiple Datasets <../4_data_cleaning_preparation/013-cleaning-and-merging-multiple-datasets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-join-in-sql-to-combine-tables/ <https://insightful-data-lab.com/2023/11/26/using-join-in-sql-to-combine-tables/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: combine
