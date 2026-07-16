:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-007:
.. _da-5-analyze-analyze-007:

========================================================================
Data Validation in Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 007`

◀ :doc:`Previous <006-data-formatting-and-unit-conversion-in-spreadsheets>` · :doc:`Next <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Rules that constrain a cell
-----------------------------

Much dirty data can be prevented at the point of entry by *constraining what a
cell may contain*. **Data validation** is a spreadsheet feature that sets rules
for the values allowed in cells — rejecting or flagging entries that violate them.
It is both a *cleaning* tool (catching existing bad data) and, more powerfully, a
*prevention* tool (stopping bad data from being entered in the first place),
which makes it a key part of preparing data for reliable analysis.

What validation rules do
--------------------------

A validation rule specifies what is acceptable in a cell:

- **Value ranges** — a number between 0 and 100, a date within a valid period,
  an amount that must be positive. Entries outside the range are rejected.
- **Lists (dropdowns)** — the cell may contain only a value from a fixed list
  (a set of valid regions, statuses, categories). This is the single most
  effective validation for preventing the inconsistency defect — if "region" can
  only be chosen from a dropdown, "NY"/"New York"/"new york" variants never arise.
- **Data types** — the cell must contain a number, a date, or text of a certain
  form.
- **Text length or format** — a code of exactly five characters, an entry
  matching a required pattern.
- **Custom rules** — formula-based conditions for more complex requirements.

When an entry violates the rule, the spreadsheet can reject it outright or warn
the user — either way, the bad value is caught at the moment it would enter.

Validation as prevention
--------------------------

The deepest value of validation is *preventing* dirty data rather than cleaning it
afterward — the source-improvement principle from the cleaning section, applied at
the cell level. A dropdown that constrains categories eliminates the inconsistency
defect before it exists; a range rule that rejects negative amounts prevents the
invalid values you would otherwise have to find and fix. Validation moves quality
control *upstream*, to the point of entry, where it is cheapest and most
effective. Every bad value prevented is one that never has to be detected,
diagnosed, and corrected.

Validation as detection
-------------------------

Applied to *existing* data, validation also *detects* violations: setting a
validation rule on a populated column flags the cells that fail it, surfacing the
invalid values already present. This makes validation a checking tool as well —
a way to audit a column against its rules and see what does not conform.

The caveat
------------

Validation is only as good as its rules, and rules have two failure modes.
**Too strict**, and legitimate values are rejected — a name-length rule that
blocks unusually long real names, a range that excludes valid extremes — which
frustrates entry and can lose real data. **Too loose**, and bad values still slip
through — a range rule that catches negatives but not implausibly large typos.
Well-designed validation reflects the data's genuine rules, neither more nor less,
and even good validation is a *filter*, not a guarantee: it enforces the
constraints you thought to specify, and cannot catch the errors you did not
anticipate. The next lesson combines validation with conditional formatting for a
stronger quality workflow.

.. hint::

   - :doc:`Data Formatting and Unit Conversion in Spreadsheets <006-data-formatting-and-unit-conversion-in-spreadsheets>`
   - :doc:`Combining Data Validation and Conditional Formatting in Spreadsheets <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <../3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`
   - :doc:`The Importance of Clean Data <../4_data_cleaning_preparation/001-the-importance-of-clean-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/data-validation-in-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/data-validation-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
