:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-027:
.. _data-analytics-cleaning-027:
.. _da-foundations-cleaning-027:
.. _da-decisions-cleaning-027:
.. _da-prep-cleaning-027:
.. _da-cleaning-cleaning-027:
.. _da-analyze-cleaning-027:
.. _da-viz-cleaning-027:
.. _da-python-cleaning-027:
.. _da-jobsearch-cleaning-027:

==============================================================================
Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors
==============================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`✅ Verification, Documentation & Next Steps` :bdg-info:`Lesson 027`

◀ :doc:`Previous <026-verifying-data-cleaning-efforts>` · :doc:`Next <028-documenting-data-cleaning-changes>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Concrete checks in both tools
-------------------------------

Verification is only as good as the specific checks you run, and both the
spreadsheet and SQL offer concrete techniques for confirming data integrity —
especially for catching **repeated** errors, the systematic defects that recur
across many rows and matter most. This lesson assembles the practical
verification toolkit in both tools.

Spreadsheet verification techniques
-------------------------------------

- **COUNTIF / COUNTA for counts.** ``COUNTA`` counts non-empty cells (confirming
  how many records remain); ``COUNTIF`` counts cells meeting a condition
  (``=COUNTIF(spend, "<0")`` should return 0 after cleaning negatives). A
  conditional count that should be zero is a powerful integrity check.
- **Conditional formatting to re-flag.** Re-apply the formatting that highlighted
  defects; if nothing highlights now, the defect is gone — a visual re-detection.
- **Pivot tables to re-audit consistency.** Re-build the counts-per-value pivot;
  if "NY" and "New York" no longer appear as separate rows, standardisation
  worked.
- **Sorting to re-inspect extremes.** Re-sort the cleaned column; the impossible
  values that sat at the top should be gone.
- **Spot-check formulas.** Compare cleaned values to raw with a formula
  (``=A2=raw!A2``) to flag where they differ, confirming transformations were
  correct.

SQL verification techniques
-----------------------------

- **COUNT to confirm row totals.** ``SELECT COUNT(*)`` before and after; the
  difference should exactly match the intended change.
- **COUNT with WHERE to confirm fixes.** ``SELECT COUNT(*) FROM t WHERE spend <
  0`` should return 0 after cleaning negatives — a conditional count that must be
  zero.
- **SELECT DISTINCT to confirm consistency.** Re-run the distinct-values query; a
  clean column shows only the standardised forms, no variants.
- **GROUP BY ... HAVING to confirm deduplication.** Re-run
  ``GROUP BY key HAVING COUNT(*) > 1``; it should return no rows if duplicates are
  gone.
- **Aggregate reconciliation.** Compare a ``SUM`` or ``COUNT`` against a known
  expected total to confirm nothing was lost or duplicated.

Catching *repeated* errors specifically
-----------------------------------------

The emphasis on *repeated* errors is deliberate: a one-off typo affects one row,
but a systematic error — a whole column imported as text, a category
misspelled everywhere, a unit wrong throughout — affects thousands and distorts
every aggregate. These are both the most damaging defects and the most *checkable*,
because a single query or formula tests the whole column at once:
``SELECT COUNT(*) WHERE <the systematic condition>`` catches a repeated error
across a million rows instantly. Verification's greatest leverage is on exactly
these repeated, systematic defects.

The caveat
------------

These techniques verify what they *test for*, and a check returning the expected
result confirms only that specific property — a row count matching expectation
does not guarantee the *right* rows survived, and a distinct-values check
confirming consistency says nothing about whether the values are *correct*.
Layer multiple checks (count *and* spot-check *and* reconcile) rather than
trusting any single one, because each catches a different class of error and none
catches all. Verification is a net woven from several checks, not a single test.
The final lesson of the stage covers documenting all of this.

.. hint::

   - :doc:`Verifying Data-Cleaning Efforts <026-verifying-data-cleaning-efforts>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`COALESCE <024-coalesce>`
   - :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors/ <https://insightful-data-lab.com/2023/11/01/verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: verify
