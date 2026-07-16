:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-008:
.. _da-5-analyze-analyze-008:

========================================================================
Combining Data Validation and Conditional Formatting in Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 008`

◀ :doc:`Previous <007-data-validation-in-spreadsheets>` · :doc:`Next <009-using-concat-in-sql-to-combine-text-from-multiple-columns>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Two features, one quality workflow
------------------------------------

Data validation and conditional formatting are individually useful, but together
they form a stronger data-quality workflow: **validation enforces rules going
forward, conditional formatting reveals rule violations visually**. Combining them
gives both prevention and visibility — the two capabilities that keep data clean
and make remaining problems obvious.

The complementary roles
-------------------------

The two features address quality from different angles:

- **Data validation** — *constrains and prevents*. It stops bad values from being
  entered (a dropdown that only allows valid categories, a range that rejects
  negatives) and can reject or warn on violations. Its focus is the *future*:
  keeping new data clean.
- **Conditional formatting** — *reveals and highlights*. It colours cells by rule
  to make patterns and problems visible (highlighting duplicates, flagging
  out-of-range values, shading blanks). Its focus is *visibility*: making the
  state of existing data obvious at a glance.

Validation acts at the gate; conditional formatting acts on the whole field.
Together they cover both preventing and seeing.

Combining them in practice
----------------------------

A robust quality setup uses both on the same data:

- **Validation** on the input columns constrains what can be entered — dropdowns
  for categories, ranges for numbers, type rules for dates — preventing whole
  classes of dirty data.
- **Conditional formatting** on the same columns highlights anything noteworthy
  that validation does not outright prevent — values near a threshold, blanks that
  need filling, duplicates that validation alone does not catch — making them
  visible for review.

For example, a data-entry sheet might *validate* that ``status`` comes only from a
dropdown (preventing invalid statuses) while *conditionally formatting* rows where
``status`` is "overdue" in red (making the important ones visible). One feature
controls what goes in; the other draws the eye to what matters.

Why the combination is powerful
---------------------------------

Neither feature alone is complete. Validation prevents bad data but does not help
you *see* the state of what is there; conditional formatting reveals problems but
does not *prevent* them. Together they close the gap: bad data is largely stopped
at entry, and whatever slips through or needs attention is made visible. This is
the same defence-in-depth idea as the verification section — multiple checks
catching different problems — applied to spreadsheet data quality, combining a
preventive control with a detective one.

The caveat
------------

Combining features adds a maintenance burden: validation rules and formatting
rules both need to stay correct as the data and its requirements evolve, and rules
that drift out of date (a dropdown missing a newly-valid category, a formatting
threshold no longer meaningful) mislead rather than help. And neither feature, nor
both together, substitutes for actually *understanding* the data — they enforce
and reveal the rules you specified, but the judgement about what is genuinely right
or wrong in the data remains yours. This closes the spreadsheet-formatting and
validation portion of the organise stage; the next lessons turn to combining and
transforming text and data with functions and SQL.

.. hint::

   - :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`
   - :doc:`Data Formatting and Unit Conversion in Spreadsheets <006-data-formatting-and-unit-conversion-in-spreadsheets>`
   - :doc:`Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND) <010-working-with-strings-in-spreadsheets-len-left-right-find>`
   - :doc:`Using CONCAT in SQL to Combine Text from Multiple Columns <009-using-concat-in-sql-to-combine-text-from-multiple-columns>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/combining-data-validation-and-conditional-formatting-in-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/combining-data-validation-and-conditional-formatting-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
