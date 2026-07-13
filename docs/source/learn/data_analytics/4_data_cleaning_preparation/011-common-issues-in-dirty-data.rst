:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-011:
.. _data-analytics-cleaning-011:
.. _da-foundations-cleaning-011:
.. _da-decisions-cleaning-011:
.. _da-prep-cleaning-011:
.. _da-cleaning-cleaning-011:
.. _da-analyze-cleaning-011:
.. _da-viz-cleaning-011:
.. _da-python-cleaning-011:
.. _da-jobsearch-cleaning-011:

========================================================================
Common Issues in Dirty Data
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-the-importance-of-clean-data-revisited>` · :doc:`Next <012-data-cleaning-with-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


A catalogue of defects
------------------------

Dirty data is dirty in recurring, recognisable ways, and knowing the catalogue is
what lets an analyst *hunt* defects systematically rather than stumble on them.
A handful of issue types account for the large majority of real data problems;
each has a signature and a standard remedy, previewed here and developed in the
hands-on lessons that follow.

The common issues
-------------------

- **Duplicate data** — the same record appearing more than once, whether
  identical copies or near-duplicates (the same customer with slight spelling
  differences). Duplicates inflate counts and over-weight the repeated records.
  *Remedy:* identify and remove duplicates, carefully — deciding which copy to
  keep.
- **Missing data** — absent values, from scattered blank cells to whole missing
  fields or records. Missing data breaks completeness and can bias results if the
  gaps are not random. *Remedy:* remove affected records, or fill (impute) with a
  reasonable estimate — each with trade-offs.
- **Inconsistent data** — the same thing recorded different ways: "NY"/"New
  York", "St."/"Street", mixed date formats, inconsistent capitalisation.
  Inconsistency fragments what should be one group. *Remedy:* standardise to a
  single consistent form.
- **Incorrect / invalid data** — values that violate their rules or reality: a
  negative age, a date in the future, a price of zero where zero is impossible.
  *Remedy:* identify against validity rules and correct or remove.
- **Wrong data type** — values stored as the wrong type: numbers as text, dates
  as text (the import problems from the prep section), which break calculations.
  *Remedy:* convert to the correct type.
- **Outliers** — values far outside the normal range. These are the *subtle* case:
  an outlier may be an **error** to fix or a **genuine extreme** to keep, and
  telling which requires context. *Remedy:* investigate — do not delete blindly.
- **Extra whitespace and formatting** — leading/trailing spaces and stray
  characters that make "New York " differ from "New York" to a computer.
  *Remedy:* trim and standardise.

The systematic hunt
---------------------

The value of the catalogue is that it becomes a **checklist**. Faced with a new
dataset, an analyst runs down the list — check for duplicates, scan for missing
values, look for inconsistent categories, validate ranges, confirm types, examine
outliers, trim whitespace — rather than hoping problems reveal themselves.
Systematic checking finds defects that ad-hoc glancing misses, especially the
quiet ones (a few duplicates in a million rows, one inconsistent spelling) that
never announce themselves but still distort results.

The caveat
------------

The remedies involve judgement, and each carries its own risk — removing
duplicates can delete legitimate records, imputing missing values can invent
data, "correcting" an outlier can erase a real finding. The catalogue tells you
*what to look for*; deciding *what to do* about each defect requires
understanding what the value means and what the analysis needs. Blind application
of cleaning rules is itself a source of error, which is why the hands-on lessons
that follow stress verification and documentation at every step. The next lesson
begins that hands-on work in the spreadsheet.

.. hint::

   - :doc:`Dirty Data vs. Clean Data <009-dirty-data-vs-clean-data>`
   - :doc:`Data Cleaning with Spreadsheets <012-data-cleaning-with-spreadsheets>`
   - :doc:`Using Spreadsheet Functions for Data Cleaning <015-using-spreadsheet-functions-for-data-cleaning>`
   - :doc:`Understanding Bias in Data Analysis <../3_data_preparation/008-understanding-bias-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/common-issues-in-dirty-data/ <https://insightful-data-lab.com/2023/11/01/common-issues-in-dirty-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty
