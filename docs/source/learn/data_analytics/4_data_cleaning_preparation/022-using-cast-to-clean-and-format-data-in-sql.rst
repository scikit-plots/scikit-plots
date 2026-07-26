:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-022:
.. _da-4-cleaning-cleaning-022:

========================================================================
Using CAST to Clean and Format Data in SQL
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 022`

◀ :doc:`Previous <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>` · :doc:`Next <023-advanced-sql-functions-for-data-cleaning>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Fixing the type
-----------------

A recurring dirty-data defect is a value stored as the *wrong type* — a number
kept as text, a date kept as a string — which breaks the calculations and
comparisons that expect the right type. In SQL, the tool that fixes this is
**CAST**, which converts a value from one data type to another. It is the SQL
counterpart of the spreadsheet's ``VALUE`` and ``DATEVALUE``, and a core cleaning
operation.

How CAST works
----------------

``CAST`` takes a value and a target type and returns the value converted:

.. code-block:: sql

   SELECT CAST(price_text AS DECIMAL)      AS price,      -- text -> number
          CAST(order_date_text AS DATE)    AS order_date, -- text -> date
          CAST(quantity AS INTEGER)        AS quantity    -- decimal -> whole
   FROM   orders;

The syntax reads plainly: ``CAST(value AS type)``. Common target types include
``INTEGER`` and ``DECIMAL`` (numbers), ``DATE`` and ``TIMESTAMP`` (dates and
times), and ``VARCHAR`` / ``STRING`` (text). Once a text-number is cast to
``DECIMAL``, it can be summed and averaged; once a text-date is cast to ``DATE``,
it can be sorted chronologically and used in date arithmetic — the defect is
fixed.

Why type conversion matters for cleaning
------------------------------------------

Type problems are among the most common consequences of the import traps from the
prep section: data loaded from CSVs and external systems frequently arrives with
everything as text, so numeric and date columns *look* right but refuse to
compute. ``CAST`` is how you correct this at scale — one query converts a whole
column of text-numbers into real numbers, ready for the aggregation and
arithmetic the analysis will need. Without it, ``SUM`` and ``AVG`` on a
text-typed column either error or silently misbehave.

CAST in a cleaning pipeline
-----------------------------

``CAST`` composes with the string functions from the previous lesson, because raw
values often need *cleaning before conversion*. A price stored as ``"$1,000"``
cannot be cast to a number directly — the ``$`` and ``,`` must be removed first:

.. code-block:: sql

   SELECT CAST(REPLACE(REPLACE(price_text, '$', ''), ',', '') AS DECIMAL) AS price
   FROM   orders;

Strip the currency symbol and thousands separator with ``REPLACE``, *then*
``CAST`` the clean numeric string to ``DECIMAL``. This clean-then-convert pattern
is exactly the nested-function pipeline from the spreadsheet lesson, in SQL.

The caveat
------------

``CAST`` fails — or behaves unexpectedly — when a value cannot be converted to
the target type: casting ``"N/A"`` or ``"twelve"`` to a number raises an error or,
in some databases, produces a null, and a single unconvertible value can break a
query over millions of rows. This is a feature, not a bug: it forces you to
confront the values that do not fit the type, which are often *themselves*
dirty data needing attention. The safe practice is to inspect a column's distinct
values (``SELECT DISTINCT``) before casting, handle the non-conforming ones
first, and be aware that different databases treat cast failures differently
(error versus null). Type conversion reveals dirt as much as it fixes it. The
next lesson covers the more advanced functions for harder cleaning cases.

.. hint::

   - :doc:`Cleaning Data with SQL: Removing Duplicates and Cleaning String Variables <021-cleaning-data-with-sql-removing-duplicates-and-cleaning-string-variables>`
   - :doc:`Understanding Data Types and Data Formats <../3_data_preparation/003-understanding-data-types-and-data-formats>`
   - :doc:`Data Types in Spreadsheets <../3_data_preparation/005-data-types-in-spreadsheets>`
   - :doc:`Advanced SQL Functions for Data Cleaning <023-advanced-sql-functions-for-data-cleaning>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/using-cast-to-clean-and-format-data-in-sql/ <https://insightful-data-lab.com/2023/11/01/using-cast-to-clean-and-format-data-in-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
