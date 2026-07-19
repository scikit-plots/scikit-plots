:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-009:
.. _da-5-analyze-analyze-009:

========================================================================
Using CONCAT in SQL to Combine Text from Multiple Columns
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>` · :doc:`Next <010-working-with-strings-in-spreadsheets-len-left-right-find>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Joining text columns
----------------------

Analysis often needs to *combine* text from several columns into one — a full name
from first and last, an address from its parts, a label from a code and a
description. In SQL, this is **concatenation**, done with the ``CONCAT`` function
(or the concatenation operator), and it is the SQL counterpart of the
spreadsheet's ``CONCATENATE`` and ``&``.

How CONCAT works
------------------

``CONCAT`` takes several values and joins them into one string:

.. code-block:: sql

   SELECT CONCAT(first_name, ' ', last_name)        AS full_name,
          CONCAT(city, ', ', state, ' ', zip)        AS address
   FROM   customers;

Each argument is joined in order; literal text (like the space ``' '`` or the
comma-space ``', '``) is included as a separator so the result reads correctly.
Many databases also support the ``||`` operator for the same purpose
(``first_name || ' ' || last_name``), though the exact operator varies by system —
``CONCAT`` is the more portable choice.

Why concatenation matters in analysis
---------------------------------------

Combining columns serves several analytical needs:

- **Building keys** — concatenating fields to form a combined identifier for
  matching or grouping (a key from region and product code).
- **Creating labels** — assembling readable labels for output and visualisation
  from separate data columns.
- **Reassembling split data** — joining back fields that were separated (the
  inverse of the Text-to-Columns / ``SUBSTR`` splitting), when analysis needs them
  together.
- **Formatting for presentation** — producing human-readable combined strings for
  a report.

Concatenation is the "combine" counterpart to the "split" operations from earlier
— together they let you restructure text data into whatever shape the analysis or
output needs.

Concatenation with clean inputs
---------------------------------

Concatenation composes with the cleaning functions, because combining *dirty* text
produces dirty results — joining an untrimmed first name to a last name yields
``"Jane  Smith"`` with a stray space. The clean-then-combine pattern applies:
``CONCAT(TRIM(first_name), ' ', TRIM(last_name))`` trims the parts before joining,
so the combined result is clean. As with all such pipelines, the inputs' quality
determines the output's quality.

The caveat
------------

Concatenation has a null trap that catches beginners: in many databases, if *any*
argument to the older concatenation *operator* is null, the entire result becomes
null — so one missing middle name can null out an entire assembled address.
``CONCAT`` the *function* often handles nulls more gracefully (treating them as
empty strings), but behaviour **varies by database**, so combining columns that
may contain nulls requires care — often wrapping arguments in ``COALESCE`` to
substitute an empty string for nulls first. Verify how your database treats nulls
in concatenation before relying on it. The next lesson covers the parallel string
toolkit in spreadsheets.

.. hint::

   - :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <010-working-with-strings-in-spreadsheets-len-left-right-find>`
   - :doc:`Advanced SQL Functions for Data Cleaning <../4_data_cleaning_preparation/023-advanced-sql-functions-for-data-cleaning>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <../4_data_cleaning_preparation/020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Using Spreadsheet Functions for Data Cleaning <../4_data_cleaning_preparation/015-using-spreadsheet-functions-for-data-cleaning>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-concat-in-sql-to-combine-text-from-multiple-columns/ <https://insightful-data-lab.com/2023/11/26/using-concat-in-sql-to-combine-text-from-multiple-columns/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
