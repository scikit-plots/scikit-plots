:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-022:
.. _da-1-foundations-foundations-022:

========================================================================
The Concept and Basic Use of SQL (Query Language)
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🧰 Tools, Applications & Ethics` :bdg-info:`Lesson 022`

◀ :doc:`Previous <021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>` · :doc:`Next <023-the-role-and-importance-of-data-visualization>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Asking questions of databases
-------------------------------

Organisational data mostly does not live in spreadsheets; it lives in
**databases** — structured collections of tables holding millions of records.
**SQL** (Structured Query Language) is how you talk to them: a language for
*describing the data you want* and letting the database fetch it. Where a
spreadsheet shows you everything and lets you point, SQL lets you **ask** — and
the asking scales to sizes no grid could display.

The core idea: declare, don't fetch-and-loop
----------------------------------------------

An SQL **query** states *what* you want, not *how* to find it:

.. code-block:: sql

   SELECT customer_id, order_date, amount
   FROM   orders
   WHERE  amount > 100;

Read it as a sentence: *select* these columns, *from* this table, *where* this
condition holds. The three clauses are the irreducible core:

- ``SELECT`` — which columns (attributes) to return; ``SELECT *`` means all.
- ``FROM`` — which table the data lives in.
- ``WHERE`` — which rows qualify, via conditions (``=``, ``>``, ``<``,
  combined with ``AND`` / ``OR``).

The database engine — not you — figures out the efficient way to satisfy the
request across millions of rows, which is precisely why the language stays this
simple at this scale.

The spreadsheet translation
-----------------------------

Every clause has a spreadsheet counterpart you already know: ``SELECT`` is
choosing columns to look at, ``FROM`` is opening the right sheet, ``WHERE`` is
a filter. The difference is not conceptual but operational: the query is
**text** — repeatable, shareable, reviewable — and it runs against the live,
shared source of record rather than a private copy. When the analysis section
later adds grouping and joining, those too will be filters and pivots you have
already met, in query form.

Why every analyst learns it
-----------------------------

Three durable reasons. **Location:** the data you need is already in a
database; SQL goes to it rather than exporting fragile copies. **Scale:** a
``WHERE`` over ten million rows returns in seconds. **Ubiquity:** SQL has been
the standard interface to relational data for decades, is asked for in more
analyst job postings than any other technical skill, and transfers almost
unchanged across database products.

The caveat
------------

A query answers exactly what it says, which is not always what you meant —
``WHERE amount > 100`` silently excludes the row where amount is missing, and
no error will tell you so. SQL's precision is a feature that demands the same
precision of your question; the deeper query techniques and their pitfalls get
full treatment in the cleaning and analysis sections.

.. hint::

   - :doc:`Overview of Core Tools Used by Data Analysts <020-overview-of-core-tools-used-by-data-analysts>`
   - :doc:`The Role of Spreadsheets in Data Analysis and Basic Concepts <021-the-role-of-spreadsheets-in-data-analysis-and-basic-concepts>`
   - :doc:`Introduction to SQL <../4_data_cleaning_preparation/018-introduction-to-sql>`
   - :doc:`Querying Data with SQL <../3_data_preparation/023-querying-data-with-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/the-concept-and-basic-use-of-sql-query-language/ <https://insightful-data-lab.com/2023/07/30/the-concept-and-basic-use-of-sql-query-language/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: tools
