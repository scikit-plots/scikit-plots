:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-021:
.. _da-4-cleaning-cleaning-021:

===========================================================================
Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables
===========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 021`

◀ :doc:`Previous <020-core-sql-queries-for-data-cleaning-and-analysis>` · :doc:`Next <022-using-cast-to-clean-and-format-data-in-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The two most common SQL cleans
--------------------------------

Two cleaning tasks dominate SQL work: removing **duplicate** records and
standardising **string** (text) values. They are the SQL versions of the
spreadsheet's Remove Duplicates and Find-and-Replace, now at database scale, and
learning them is learning the everyday core of SQL cleaning.

Removing duplicates
---------------------

The simplest deduplication is ``SELECT DISTINCT``, which returns only unique
rows:

.. code-block:: sql

   SELECT DISTINCT customer_id, name, email
   FROM   customers;

``DISTINCT`` collapses exact duplicate rows across the selected columns. To
*find* duplicates rather than just remove them — to see which records repeat and
how often — group and count:

.. code-block:: sql

   SELECT   email, COUNT(*) AS copies
   FROM     customers
   GROUP BY email
   HAVING   COUNT(*) > 1;

This lists every email appearing more than once (``HAVING`` filters the *groups*,
keeping only those with a count above one) — the diagnostic step before deciding
how to resolve them. Which duplicate to *keep* when they disagree is a judgement,
often resolved by keeping the most recent or most complete record.

Cleaning string variables
----------------------------

Text fields accumulate the inconsistencies from earlier — stray spaces, mixed
case, unwanted characters — and SQL's string functions fix them:

.. code-block:: sql

   SELECT TRIM(name)                    AS name_trimmed,   -- remove edge spaces
          UPPER(region)                 AS region_upper,   -- uniform case
          REPLACE(phone, '-', '')       AS phone_digits    -- strip separators
   FROM   customers;

- ``TRIM`` removes leading and trailing whitespace (``LTRIM`` / ``RTRIM`` for one
  side).
- ``UPPER`` / ``LOWER`` normalise case, so ``"NY"`` and ``"ny"`` become uniform.
- ``REPLACE(text, from, to)`` substitutes one substring for another, stripping or
  swapping characters.

These are the same operations as the spreadsheet's ``TRIM``, ``UPPER``, and
``SUBSTITUTE`` — the SQL syntax differs slightly, the ideas are identical.

Cleaning into a result, not in place
--------------------------------------

Notice these queries *select* cleaned values — they produce a cleaned result set
without altering the stored table. This is the safe default: you build and verify
the cleaned output with a ``SELECT`` first, confirming it does what you intend,
before ever writing changes back. Producing a cleaned view or table from a query
is both safer and more reproducible than editing data in place.

The caveat
------------

SQL deduplication has a subtlety that catches beginners: ``SELECT DISTINCT``
removes rows that are *identical across the selected columns*, so two records for
the same customer that differ in even one field (a different timestamp, a typo'd
name) are **not** duplicates to ``DISTINCT`` and both survive. True
deduplication of near-duplicates requires deciding *which columns define
sameness* and grouping on those — not a blind ``DISTINCT *``. And string
functions are precise: ``REPLACE`` replaces *every* occurrence, so an over-broad
replacement corrupts as readily in SQL as in a spreadsheet. Verify the
``SELECT`` output before trusting it. The next lesson tackles the type-conversion
cleaning that ``CAST`` handles.

.. hint::

   - :doc:`Core SQL Queries for Data Cleaning and Analysis <020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Using CAST to Clean and Format Data in SQL <022-using-cast-to-clean-and-format-data-in-sql>`
   - :doc:`Using Spreadsheet Functions for Data Cleaning <015-using-spreadsheet-functions-for-data-cleaning>`
   - :doc:`Advanced SQL Functions for Data Cleaning <023-advanced-sql-functions-for-data-cleaning>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables/ <https://insightful-data-lab.com/2023/11/01/cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
