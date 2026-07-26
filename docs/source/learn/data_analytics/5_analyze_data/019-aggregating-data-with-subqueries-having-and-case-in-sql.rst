:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-019:
.. _da-5-analyze-analyze-019:

========================================================================
Aggregating Data with Subqueries, HAVING, and CASE in SQL
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🔗 Problem-Solving & Combining Data` :bdg-info:`Lesson 019`

◀ :doc:`Previous <018-subqueries-in-sql>` · :doc:`Next <020-using-spreadsheet-formulas-for-sales-trend-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Bringing the pieces together
------------------------------

Real analytical queries combine several SQL capabilities at once: aggregating with
``GROUP BY``, filtering groups with ``HAVING``, applying conditional logic with
``CASE``, and layering with subqueries. This lesson assembles them — the
culmination of the combine stage — into the kind of query that answers a genuine
business question, closing the loop on SQL as an analytical tool.

Filtering groups with HAVING
------------------------------

``WHERE`` filters *rows*; ``HAVING`` filters *groups* after aggregation — the
distinction that trips up beginners:

.. code-block:: sql

   SELECT   region, COUNT(*) AS orders, SUM(amount) AS revenue
   FROM     orders
   GROUP BY region
   HAVING   SUM(amount) > 10000;

This groups orders by region, computes each region's count and revenue, then keeps
only regions whose *total revenue* exceeds 10,000. ``WHERE`` cannot do this — the
condition is on an aggregate (``SUM``) that does not exist until after grouping, so
it must be ``HAVING``. The rule: **``WHERE`` filters before grouping (on individual
rows), ``HAVING`` filters after grouping (on aggregated values).**

Conditional aggregation with CASE
-----------------------------------

``CASE`` inside an aggregate enables powerful conditional counting and summing —
computing different aggregates for different conditions in one query:

.. code-block:: sql

   SELECT region,
          COUNT(*)                                          AS total_orders,
          SUM(CASE WHEN amount > 100 THEN 1 ELSE 0 END)     AS large_orders,
          SUM(CASE WHEN amount > 100 THEN amount ELSE 0 END) AS large_revenue
   FROM     orders
   GROUP BY region;

Each ``CASE`` inside a ``SUM`` counts or totals only the rows meeting its condition
— "how many large orders, and their revenue, per region" — in a single grouped
query. This conditional-aggregation pattern answers segmented questions (this
category versus that, above threshold versus below) without separate queries, and
it is one of the most useful analytical techniques in SQL.

Layering with subqueries
--------------------------

Subqueries add another layer — aggregating, then querying the aggregate:

.. code-block:: sql

   SELECT region, revenue
   FROM   (SELECT region, SUM(amount) AS revenue
           FROM orders GROUP BY region) AS regional
   WHERE  revenue > (SELECT AVG(amount) * 100 FROM orders);

The inner query aggregates revenue by region; the outer query filters those
regional totals against a data-derived threshold. Composing aggregation inside a
subquery lets you analyse *summaries* — asking questions about grouped results, a
genuinely multi-step analysis in one statement.

The analytical payoff
-----------------------

Together, ``GROUP BY``, ``HAVING``, ``CASE``, and subqueries turn SQL from a
retrieval language into an *analytical* one: segment the data, aggregate it, filter
the aggregates, apply conditional logic, and layer the whole thing — answering
questions that would take many manual spreadsheet steps in one repeatable query.
This is the analytical core the section has been building toward in SQL, and it is
why SQL is the backbone of serious data analysis.

The caveat
------------

Combining these features multiplies both power and the room for error. The
``WHERE``/``HAVING`` confusion produces wrong results silently (filtering rows when
you meant groups, or vice versa); ``CASE`` conditions that do not cover all cases
leave gaps; nulls interact with aggregates subtly (``COUNT(*)`` versus
``COUNT(column)``, ``AVG`` ignoring nulls); and a complex query can be *confidently
wrong* in ways only careful checking reveals. The discipline is to **build complex
queries incrementally** — get the ``GROUP BY`` right, add ``HAVING``, add ``CASE``,
verifying at each step — rather than writing the whole thing at once, and to check
the results against expectation and simpler queries. Complexity earns power only if
it stays correct. This closes the combine stage; the next turns to calculations and
trend analysis, in spreadsheets and SQL.

.. hint::

   - :doc:`Subqueries in SQL <018-subqueries-in-sql>`
   - :doc:`Using JOIN in SQL to Combine Tables <017-using-join-in-sql-to-combine-tables>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <../4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/aggregating-data-with-subqueries-having-and-case-in-sql/ <https://insightful-data-lab.com/2023/11/26/aggregating-data-with-subqueries-having-and-case-in-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: combine
