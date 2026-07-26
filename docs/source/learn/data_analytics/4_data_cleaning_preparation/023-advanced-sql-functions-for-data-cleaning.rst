:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-023:
.. _da-4-cleaning-cleaning-023:

========================================================================
Advanced SQL Functions for Data Cleaning
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 023`

◀ :doc:`Previous <022-using-cast-to-clean-and-format-data-in-sql>` · :doc:`Next <024-coalesce>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Beyond the basics
-------------------

The core string and conversion functions handle most cleaning, but harder cases —
conditional fixes, extracting parts of a field, pattern-based standardisation —
call for more capable tools. This lesson covers the **advanced SQL functions**
that handle the cleaning jobs the basics cannot, extending the toolkit toward the
messy realities of production data.

Conditional cleaning with CASE
--------------------------------

``CASE`` is SQL's if-then-else, applying different fixes depending on a value's
condition — the SQL counterpart of the spreadsheet's nested ``IF``:

.. code-block:: sql

   SELECT name,
          CASE
            WHEN spend < 0        THEN 0          -- invalid negatives -> 0
            WHEN spend IS NULL    THEN 0          -- missing -> 0
            ELSE spend
          END                     AS spend_clean
   FROM   customers;

``CASE`` evaluates each ``WHEN`` in order and returns the first matching
``THEN``, falling through to ``ELSE`` if none match. It handles the "different
fix for different conditions" cleaning that a single function cannot —
standardising categories, bucketing values, or applying rules that vary by row.

Extracting parts of a field
-----------------------------

When a column crams several values together, substring functions pull them
apart — the SQL version of the spreadsheet's ``LEFT``/``RIGHT``/``MID``:

.. code-block:: sql

   SELECT SUBSTR(product_code, 1, 3)   AS category_code,  -- first 3 chars
          LENGTH(name)                 AS name_length,
          POSITION('@' IN email)       AS at_position     -- find a character
   FROM   products;

``SUBSTR`` (or ``SUBSTRING``) extracts characters by position and length;
``LENGTH`` gives a string's length; ``POSITION`` (or ``INSTR``) locates a
substring. Together they split composite fields into their components — restoring
the one-value-per-column structure.

Pattern-based cleaning
------------------------

For standardising by *pattern* rather than exact match, ``LIKE`` with wildcards
finds values fitting a shape (``WHERE phone LIKE '___-___-____'`` for a phone
format), and combined with ``CASE`` it standardises variants that a simple
``REPLACE`` cannot. More advanced engines offer regular-expression functions for
complex pattern matching, though these vary by database.

Composing the advanced tools
------------------------------

The power comes from combining them: a ``CASE`` that applies different
``SUBSTR`` extractions depending on a field's pattern, or a ``TRIM`` wrapped
around a ``REPLACE`` inside a ``CAST``. Real cleaning queries nest these functions
into a pipeline that inspects, extracts, conditions, and converts in one pass —
the same composition principle as spreadsheet formula nesting, with SQL's richer
function set.

The caveat
------------

Advanced functions bring two cautions. First, **portability**: the basics
(``TRIM``, ``UPPER``, ``CASE``, ``CAST``) work almost everywhere, but the exact
names and behaviours of substring, position, and especially regular-expression
functions **differ across database systems** — ``SUBSTR`` versus ``SUBSTRING``,
``INSTR`` versus ``POSITION`` — so a query written for one database may need
adjustment for another. Second, **readability**: deeply nested advanced functions
grow hard to read and debug, so build them in stages and comment the intent.
Power without clarity is its own maintenance problem — the clarity-over-cleverness
principle, applied to SQL. The final SQL cleaning lesson covers the dedicated tool
for missing values: ``COALESCE``.

.. hint::

   - :doc:`Using CAST to Clean and Format Data in SQL <022-using-cast-to-clean-and-format-data-in-sql>`
   - :doc:`COALESCE <024-coalesce>`
   - :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <../5_analyze_data/010-working-with-strings-in-spreadsheets-len-left-right-find>`
   - :doc:`Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/advanced-sql-functions-for-data-cleaning/ <https://insightful-data-lab.com/2023/11/01/advanced-sql-functions-for-data-cleaning/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
