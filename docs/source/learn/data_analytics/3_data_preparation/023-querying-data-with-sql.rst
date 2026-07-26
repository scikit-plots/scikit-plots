:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-023:
.. _da-3-prep-prep-023:

========================================================================
Querying Data with SQL
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🔢 Spreadsheets, SQL & Organization` :bdg-info:`Lesson 023`

◀ :doc:`Previous <022-bigquery-account-types>` · :doc:`Next <024-organizing-data-for-personal-and-work-projects>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Asking a database a question
------------------------------

With a database environment in hand, the core skill is the **query** — an SQL
statement that retrieves exactly the data you want. The foundations introduced
SQL's shape; this lesson is the hands-on core, the three clauses that answer the
majority of real questions and the foundation everything later in the SQL thread
builds on.

The essential query
---------------------

Every basic retrieval combines three clauses:

.. code-block:: sql

   SELECT product, region, amount
   FROM   orders
   WHERE  amount > 100;

- ``SELECT`` names the **columns** to return. ``SELECT *`` returns all columns;
  naming specific columns is tidier and, on large cloud warehouses, cheaper
  (you are billed for the columns scanned).
- ``FROM`` names the **table** the data comes from.
- ``WHERE`` sets the **condition** rows must meet to be included — the filter,
  in query form.

Read top to bottom, it is a sentence: *select* these columns *from* this table
*where* this condition holds.

Building up the WHERE clause
------------------------------

Most of a query's power lives in ``WHERE``, which supports:

- **Comparisons** — ``=``, ``<>`` (not equal), ``>``, ``<``, ``>=``, ``<=``:
  ``WHERE region = 'North'``, ``WHERE amount >= 50``.
- **Combinations** — ``AND`` (both must hold), ``OR`` (either), ``NOT``:
  ``WHERE region = 'North' AND amount > 100``.
- **Ranges and sets** — ``BETWEEN`` for ranges, ``IN`` for lists:
  ``WHERE amount BETWEEN 50 AND 200``, ``WHERE region IN ('North', 'South')``.
- **Pattern matching** — ``LIKE`` with wildcards for text:
  ``WHERE product LIKE 'Pro%'`` (products starting with "Pro").

Ordering the results
----------------------

A query can also sort its output with ``ORDER BY`` — the SQL counterpart of the
spreadsheet sort:

.. code-block:: sql

   SELECT product, amount
   FROM   orders
   WHERE  region = 'North'
   ORDER  BY amount DESC;

``ORDER BY amount DESC`` returns the rows largest-first; ``ASC`` (the default) is
smallest-first. Filtering with ``WHERE`` and ordering with ``ORDER BY`` together
are the SQL version of the spreadsheet's filter-then-sort — the same analytical
move, now at database scale.

Why queries beat exporting
----------------------------

A query goes to the data where it lives, returns only what you asked for, and is
**text** — repeatable, shareable, reviewable. Instead of exporting a million rows
to a spreadsheet and filtering there, you ask the database for the thousand that
matter. This is why SQL scales where spreadsheets strain, and why it is the most
consistently demanded analyst skill.

The caveat
------------

A query returns exactly what you specify — which is not always what you meant.
``WHERE amount > 100`` silently excludes rows where amount is *null* (missing),
and no error warns you; a ``LIKE`` pattern that is slightly off quietly returns
the wrong rows. SQL's precision demands precision from you, and the habit of
**checking results** — does the row count look right, do sampled rows match
expectations — applies to every query, exactly as the sanity-check habit applied
to spreadsheet formulas. The deeper query techniques come in the analysis
section; this is the foundation they extend.

.. hint::

   - :doc:`Databases and Relational Database Concepts <016-databases-and-relational-database-concepts>`
   - :doc:`The Concept and Basic Use of SQL (Query Language) <../1_foundations/022-the-concept-and-basic-use-of-sql-query-language>`
   - :doc:`BigQuery Account Types <022-bigquery-account-types>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <021-sorting-and-filtering-data-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/15229/ <https://insightful-data-lab.com/2023/09/04/15229/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: spreadsheets_sql
