:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-024:
.. _da-4-cleaning-cleaning-024:

========================================================================
COALESCE
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 024`

◀ :doc:`Previous <023-advanced-sql-functions-for-data-cleaning>` · :doc:`Next <025-verifying-and-reporting-data-integrity>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Handling missing values directly
----------------------------------

Missing values — nulls — are one of the most common dirty-data defects, and SQL
has a dedicated function for handling them: **COALESCE**, which returns the first
non-null value from a list of arguments. It is SQL's standard, clean way to
substitute a fallback for missing data, and it earns its own lesson because
missing-value handling is so pervasive in real cleaning.

How COALESCE works
--------------------

``COALESCE`` takes any number of arguments and returns the first that is not
null:

.. code-block:: sql

   SELECT name,
          COALESCE(phone, 'no phone on file')  AS phone,
          COALESCE(nickname, first_name, 'Unknown') AS display_name
   FROM   customers;

For each row, ``COALESCE(phone, 'no phone on file')`` returns the phone if it
exists, otherwise the fallback text. The multi-argument form
``COALESCE(nickname, first_name, 'Unknown')`` tries each in turn: the nickname if
present, else the first name, else a final default — returning the first
available value down the list. This replaces scattered nulls with meaningful
values in a single, readable expression.

Why COALESCE matters
----------------------

Nulls cause trouble throughout analysis: they break calculations
(``price + tax`` is null if either is null), skew aggregates (``AVG`` ignores
them, changing the denominator), and display as blanks that confuse readers.
``COALESCE`` addresses this at the point of query, substituting a sensible value
so downstream calculations and displays behave. A common pattern fills missing
numerics with zero for summation — ``COALESCE(amount, 0)`` — so a ``SUM`` counts
the missing rows as zero rather than being thrown off by nulls.

COALESCE versus other approaches
----------------------------------

``COALESCE`` is the *standard* (cross-database) way to handle nulls; some
databases also offer ``IFNULL`` or ``NVL`` for the two-argument case, but
``COALESCE`` works everywhere and handles any number of fallbacks, so it is the
portable choice. It relates to the ``CASE`` from the previous lesson —
``COALESCE(x, y)`` is shorthand for ``CASE WHEN x IS NOT NULL THEN x ELSE y
END`` — but is far more concise for the common "use this, or a fallback if
missing" pattern.

The caveat
------------

``COALESCE`` is powerful, and that is exactly its danger: **substituting a value
for missing data is a decision with analytical consequences**, not a neutral
tidy-up. Filling missing prices with zero changes the average; filling missing
categories with "Unknown" creates a category that did not exist in reality;
replacing nulls hides the fact that data *was* missing, which may itself be a
finding (why is this field so often empty?). The honest practice is to choose the
fallback *deliberately* — is zero, the mean, or a flag the right substitute for
*this* analysis? — and to consider whether the missingness should be preserved
and reported rather than filled. ``COALESCE`` makes filling easy; whether to fill
at all is the judgement. This completes the SQL cleaning stage; the final stage
of the section turns to verifying, documenting, and reporting the cleaning work.

.. hint::

   - :doc:`Advanced SQL Functions for Data Cleaning <023-advanced-sql-functions-for-data-cleaning>`
   - :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`
   - :doc:`Verifying Data-Cleaning Efforts <026-verifying-data-cleaning-efforts>`
   - :doc:`Using CAST to Clean and Format Data in SQL <022-using-cast-to-clean-and-format-data-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/coalesce/ <https://insightful-data-lab.com/2023/11/01/coalesce/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
