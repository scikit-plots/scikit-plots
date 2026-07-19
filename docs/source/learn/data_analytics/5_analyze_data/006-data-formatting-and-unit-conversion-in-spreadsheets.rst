:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-006:
.. _da-5-analyze-analyze-006:

========================================================================
Data Formatting and Unit Conversion in Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 006`

◀ :doc:`Previous <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>` · :doc:`Next <007-data-validation-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Making numbers comparable
---------------------------

Organised data is not yet analysis-ready if its values are inconsistently
formatted or expressed in different units — you cannot meaningfully compare or
combine what is not on a common footing. **Data formatting and unit conversion**
is the step that puts values into consistent, comparable form, and it belongs in
the organise-and-prepare phase of analysis because analysis on mixed formats or
mixed units produces nonsense.

Consistent formatting
-----------------------

Formatting makes values *display and behave* consistently:

- **Number formats** — a consistent number of decimal places, consistent use of
  thousands separators, currency shown uniformly. Formatting affects display; the
  underlying value is unchanged, but consistent display prevents misreading.
- **Date formats** — one date format throughout (the wide/regional variation from
  the prep section), so dates sort and compare correctly and are not misread
  (is ``03/04`` March 4th or April 3rd?).
- **Text formats** — consistent capitalisation and spacing (the cleaning
  functions), so categories match.

Consistent formatting is partly cosmetic (readability) and partly functional
(correct sorting, comparison, and matching depend on it).

Unit conversion
-----------------

Unit conversion is more than cosmetic — it changes values so they are *genuinely
comparable*:

- **Currency** — amounts in different currencies must be converted to one before
  summing or comparing; adding dollars and euros as if identical is a real error.
- **Measurement units** — mixing metric and imperial (kilometres and miles,
  kilograms and pounds) produces meaningless aggregates until converted to one.
- **Time units** — durations in a mix of minutes and hours must be unified before
  arithmetic.
- **Scale** — values recorded in different scales (thousands vs. actual, per-day
  vs. per-month) must be brought to one scale to compare.

The rule: **before combining or comparing values, confirm they share a unit**, and
convert if they do not. This is one of the most common sources of the confident-
wrong result — an aggregate that summed incompatible units.

Why this belongs in analysis
------------------------------

Formatting and unit consistency is the "format and adjust" step of the analysis
process — the bridge between organised data and correct computation. Skipping it
produces the classic errors: a total that added mixed currencies, a comparison
between differently-scaled numbers, a date sort scrambled by inconsistent formats.
Getting values onto a common footing *before* calculating is what makes the
calculation meaningful.

The caveat
------------

Two cautions. First, **formatting versus value**: changing a cell's *display
format* (showing two decimals, adding a currency symbol) does not change the
underlying number, while *unit conversion* genuinely changes the value — confusing
the two leads to error, such as thinking you have converted currency when you have
only added a "$" to euro amounts. Second, **conversion introduces its own risk**:
an exchange rate applied wrongly, a conversion factor mistyped, or rounding
accumulated across many conversions can corrupt data — so conversions deserve the
same verification as any transformation. Confirm units are truly unified, not just
relabelled. The next lessons turn to *validating* data against rules.

.. hint::

   - :doc:`Data Types in Spreadsheets <../3_data_preparation/005-data-types-in-spreadsheets>`
   - :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`
   - :doc:`Common Spreadsheet Errors and How to Fix Them <../2_data_driven_decisions/015-common-spreadsheet-errors-and-how-to-fix-them>`
   - :doc:`The Importance of Clean Data <../4_data_cleaning_preparation/001-the-importance-of-clean-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/data-formatting-and-unit-conversion-in-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/data-formatting-and-unit-conversion-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
