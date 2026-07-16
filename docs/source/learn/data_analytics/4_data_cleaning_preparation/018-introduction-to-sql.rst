:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-018:
.. _da-4-cleaning-cleaning-018:

========================================================================
Introduction to SQL
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 018`

◀ :doc:`Previous <017-data-mapping-and-the-big-picture-of-clean-data>` · :doc:`Next <019-spreadsheets-vs-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Cleaning at database scale
----------------------------

Spreadsheet cleaning works beautifully until the data outgrows the sheet — and
organisational data usually has. **SQL** (Structured Query Language) is the tool
that scales cleaning and analysis to data of any size, running the same
operations you learned in the spreadsheet against databases holding millions of
rows. The foundations introduced SQL's shape; this lesson reintroduces it
specifically as a *cleaning* tool, opening the SQL stage of this section.

Why SQL for cleaning
----------------------

SQL brings three advantages that matter especially for cleaning:

- **Scale.** A ``TRIM`` or a duplicate-removal that a spreadsheet performs on
  ten thousand rows, SQL performs on ten million in seconds — because the
  database engine does the work, not your laptop.
- **Reproducibility.** A cleaning query is *text*. Save it, and re-cleaning next
  month's data is re-running the query — the reproducibility that manual
  spreadsheet cleaning lacked, achieved by default. The query *is* the
  documentation of what cleaning was applied.
- **Working at the source.** SQL cleans data *where it lives*, in the database,
  rather than exporting fragile copies to a spreadsheet and back — reducing the
  transfer errors that threaten integrity.

The same operations, in query form
-------------------------------------

Everything you cleaned in a spreadsheet has a SQL equivalent, and recognising the
correspondence makes SQL cleaning feel familiar rather than foreign:

- Filtering to inspect (spreadsheet filter) → ``WHERE``.
- Finding distinct values to check consistency → ``SELECT DISTINCT``.
- Counting per category to audit (pivot of counts) → ``GROUP BY`` with
  ``COUNT``.
- Trimming, changing case, substituting → SQL string functions (``TRIM``,
  ``UPPER``, ``LOWER``, ``REPLACE``).
- Converting types → ``CAST`` (a dedicated lesson ahead).
- Removing duplicates → ``DISTINCT`` or grouping (a dedicated lesson ahead).

The syntax differs, but the *operations* are the ones you already know — SQL is a
new language for familiar cleaning ideas.

What makes SQL cleaning powerful
----------------------------------

Because a cleaning query is repeatable text that runs at the source at scale, SQL
turns cleaning from a manual chore into an *engineered pipeline*. A well-written
cleaning query documents the transformation, runs identically every time, and
handles data volumes no spreadsheet could open. This is why, past a certain
scale, professional cleaning moves to SQL (and Python) — not because spreadsheets
are wrong, but because rerunnable queries are what recurring, large-scale
cleaning requires.

The caveat
------------

SQL's power raises the stakes of a mistake: a cleaning query applied to a
production database can transform millions of rows at once, and an error scales
just as fast as a fix. The disciplines from spreadsheet cleaning apply with more
force — work against a copy or in a transaction you can undo, test the query's
``SELECT`` (what it *would* affect) before running any modification, and verify
row counts before and after. The precision SQL demands is the same precision the
whole section has stressed, now operating at scale. The next lesson weighs
directly when to reach for SQL versus a spreadsheet.

.. hint::

   - :doc:`The Concept and Basic Use of SQL (Query Language) <../1_foundations/022-the-concept-and-basic-use-of-sql-query-language>`
   - :doc:`Spreadsheets vs. SQL <019-spreadsheets-vs-sql>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Databases and Relational Database Concepts <../3_data_preparation/016-databases-and-relational-database-concepts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/introduction-to-sql/ <https://insightful-data-lab.com/2023/11/01/introduction-to-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
