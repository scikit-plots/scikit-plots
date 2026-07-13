:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-025:
.. _data-analytics-cleaning-025:
.. _da-foundations-cleaning-025:
.. _da-decisions-cleaning-025:
.. _da-prep-cleaning-025:
.. _da-cleaning-cleaning-025:
.. _da-analyze-cleaning-025:
.. _da-viz-cleaning-025:
.. _da-python-cleaning-025:
.. _da-jobsearch-cleaning-025:

========================================================================
Verifying and Reporting Data Integrity
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`✅ Verification, Documentation & Next Steps` :bdg-info:`Lesson 025`

◀ :doc:`Previous <024-coalesce>` · :doc:`Next <026-verifying-data-cleaning-efforts>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The step that makes cleaning trustworthy
------------------------------------------

Cleaning data is only half the job; **confirming it is now sound** is the other
half, and the one beginners most often skip. This final stage of the section is
about **verification** — checking that cleaning actually worked — and
**reporting** — communicating the data's integrity to the people who will rely on
it. Without verification, cleaning is an act of faith; with it, cleaning becomes a
trustworthy, defensible process.

Why verification is non-negotiable
------------------------------------

Every cleaning operation can go wrong in ways that produce plausible results,
exactly like the defects it was meant to fix. A duplicate-removal might delete
legitimate records; a find-and-replace might over-match; a type conversion might
silently null out unconvertible values; a merge might drop or multiply rows.
Because these failures produce data that *looks* fine, the only way to know
cleaning succeeded is to *check* — verification is what catches the errors
cleaning itself introduced. The integrity-in-motion principle reaches its
conclusion here: after every transformation, confirm the data is still sound.

What verifying integrity involves
-----------------------------------

Verifying data integrity after cleaning means confirming the data is now
complete, accurate, consistent, and valid — that the defects are gone and no new
ones were introduced:

- **Confirm the fix worked** — the duplicates are actually gone, the categories
  actually consistent, the types actually converted.
- **Confirm nothing broke** — the row count is what it should be (not
  mysteriously smaller from over-deletion or larger from a bad merge), the totals
  still reconcile, no legitimate data was lost.
- **Confirm against the source** — spot-check cleaned values against the original
  raw data to ensure cleaning transformed them correctly, not wrongly.
- **Confirm fitness for purpose** — the data now meets the standard the analysis
  requires (the alignment and sufficiency checks from earlier, revisited after
  cleaning).

Reporting integrity
---------------------

Verification produces something worth communicating: **confidence in the data**.
Reporting data integrity means telling stakeholders what state the data is in —
what was cleaned, what issues were found and fixed, what limitations remain, and
therefore how much the conclusions can be trusted. This is not bureaucratic
overhead; it is what lets a stakeholder weigh the analysis appropriately, and it
is the honest-communication obligation from Section 2 applied to data quality. A
brief, clear statement of the data's integrity — "duplicates removed, categories
standardised, 2% of records had missing values that were excluded, results
reliable for the top four regions" — is what turns cleaned data into *trusted*
cleaned data.

The caveat
------------

Verification can never be exhaustive — you cannot check every value in a large
dataset by hand, and some errors will inevitably escape any practical check. The
professional standard is *proportionate* verification: check thoroughly the
things most likely to be wrong and most consequential if they are (the key
fields, the counts, the transformations you are least sure of), and be honest in
reporting about what was and was not verified. Verification reduces risk to an
acceptable level and documents that it did; it does not achieve certainty, and
claiming it does is its own dishonesty. The next lessons make verification
concrete.

.. hint::

   - :doc:`Data Integrity and Its Risks in Data Analysis <002-data-integrity-and-its-risks-in-data-analysis>`
   - :doc:`Verifying Data-Cleaning Efforts <026-verifying-data-cleaning-efforts>`
   - :doc:`Reporting Data-Cleaning Results <029-reporting-data-cleaning-results>`
   - :doc:`Documenting Data-Cleaning Changes <028-documenting-data-cleaning-changes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/verifying-and-reporting-data-integrity/ <https://insightful-data-lab.com/2023/11/01/verifying-and-reporting-data-integrity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: verify
