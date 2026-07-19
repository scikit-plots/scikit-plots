:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-005:
.. _da-5-analyze-analyze-005:

========================================================================
Sorting and Filtering Data in SQL Using ORDER BY and WHERE
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 005`

◀ :doc:`Previous <004-sorting-data-in-spreadsheets>` · :doc:`Next <006-data-formatting-and-unit-conversion-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Sort and filter, in query form
--------------------------------

The two foundational moves — sorting and filtering — have direct SQL equivalents:
**ORDER BY** sorts, and **WHERE** filters. Everything you did with a spreadsheet's
sort and filter, SQL does with these two clauses, at database scale and in
repeatable text. This lesson makes the correspondence concrete, extending the
basic queries from the prep and cleaning sections into their analytical use.

Filtering with WHERE
----------------------

``WHERE`` restricts a query to rows meeting a condition — the SQL filter:

.. code-block:: sql

   SELECT product, region, amount
   FROM   orders
   WHERE  region = 'North'
     AND  amount > 100;

Only rows where the region is North *and* the amount exceeds 100 are returned. The
``WHERE`` toolkit from earlier applies fully: comparisons (``=``, ``<>``, ``>``,
``<``), combinations (``AND``, ``OR``, ``NOT``), ranges (``BETWEEN``), sets
(``IN``), and pattern matching (``LIKE``). ``WHERE`` is how you point analysis at
exactly the subset a question concerns.

Ordering with ORDER BY
------------------------

``ORDER BY`` sorts the result — the SQL sort:

.. code-block:: sql

   SELECT product, amount
   FROM   orders
   WHERE  region = 'North'
   ORDER  BY amount DESC;

``ORDER BY amount DESC`` returns rows largest-first (``ASC``, the default, is
smallest-first). Multi-column sorting works exactly as in a spreadsheet — list
columns in priority order:

.. code-block:: sql

   ORDER BY region ASC, amount DESC   -- by region, then by amount within region

This orders by region first, then by amount within each region — the SQL version
of the multi-column spreadsheet sort.

The two together: the analytical query
----------------------------------------

Combining ``WHERE`` and ``ORDER BY`` is the SQL filter-then-sort — the same
reflexive analytical move, now as a query. "The top ten northern orders this
quarter" becomes:

.. code-block:: sql

   SELECT   product, amount
   FROM     orders
   WHERE    region = 'North'
     AND    order_date >= '2024-01-01'
   ORDER BY amount DESC
   LIMIT    10;

``WHERE`` filters to the subset, ``ORDER BY`` ranks it, and ``LIMIT`` (a handy
companion) caps the output to the top ten. One query answers what would take
several spreadsheet steps — and reruns identically on new data.

The safety advantage
----------------------

SQL sorting sidesteps the spreadsheet's most dangerous sort mistake entirely:
``ORDER BY`` reorders the *query result*, always keeping each row's values
together, so the isolated-column corruption that plagues spreadsheets simply
cannot happen. And ``WHERE`` filters without altering the stored data — the result
is a view, the table untouched. SQL's structure makes these operations inherently
safer than their manual spreadsheet equivalents.

The caveat
------------

``WHERE`` and ``ORDER BY`` are precise about what they include and how they order,
which is not always what you intend. ``WHERE amount > 100`` silently excludes rows
where amount is *null* (the ``IS NULL`` trap from earlier), so a filter can drop
rows you meant to keep; and ``ORDER BY`` on a column with mixed or wrong types
sorts unexpectedly (text-numbers sort alphabetically, so "100" sorts before "99").
The result reflects exactly what you asked — verify it matches what you meant, the
same check-your-results habit as everywhere. The next lessons turn from ordering
and subsetting to getting the data's *format* analysis-ready.

.. hint::

   - :doc:`Sorting Data in Spreadsheets <004-sorting-data-in-spreadsheets>`
   - :doc:`Querying Data with SQL <../3_data_preparation/023-querying-data-with-sql>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <../4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Sorting and Filtering in Data Analysis <003-sorting-and-filtering-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/sorting-and-filtering-data-in-sql-using-order-by-and-where/ <https://insightful-data-lab.com/2023/11/26/sorting-and-filtering-data-in-sql-using-order-by-and-where/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
