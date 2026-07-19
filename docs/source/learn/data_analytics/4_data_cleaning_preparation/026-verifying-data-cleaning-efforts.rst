:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-026:
.. _da-4-cleaning-cleaning-026:

========================================================================
Verifying Data-Cleaning Efforts
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`✅ Verification, Documentation & Next Steps` :bdg-info:`Lesson 026`

◀ :doc:`Previous <025-verifying-and-reporting-data-integrity>` · :doc:`Next <027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Checking your own work
------------------------

Verification begins with the cleaning you just did: confirming each step achieved
its goal *and* had no unintended side effects. **Verifying data-cleaning efforts**
is the disciplined review of your own transformations — the habit that separates
professional cleaning from hopeful cleaning, and the practical core of this
stage.

The two questions every cleaning step must answer
---------------------------------------------------

For each cleaning operation, verification asks two things:

1. **Did it do what I intended?** The duplicates I removed are gone; the values I
   standardised are now consistent; the column I converted is now the right type.
   Confirm the *intended* effect actually happened.
2. **Did it do anything I did not intend?** No legitimate records were deleted
   alongside the duplicates; the standardisation did not over-match and merge
   distinct values; the conversion did not silently null out data it could not
   convert. Confirm the *absence* of side effects.

The second question is the one beginners forget and professionals never do —
because the side effects are precisely what produce plausible-wrong data.

The verification techniques
-----------------------------

Concrete checks answer these questions:

- **Row counts before and after.** The single most valuable check. Know how many
  rows you started with and expect to end with; a mismatch (deduplication removed
  more than the known duplicate count, a merge changed the count unexpectedly)
  signals a problem immediately.
- **Spot-checking.** Examine specific records in detail — pick some you know, and
  confirm they cleaned correctly. Spot-checks catch errors that aggregate numbers
  hide.
- **Re-running the detection.** Apply the defect-finding checks again after
  cleaning: if you removed duplicates, re-run the duplicate query and confirm it
  now returns none; if you standardised categories, re-run the distinct-values
  check and confirm they are now uniform.
- **Comparing to the source.** Check cleaned values against the untouched raw data
  to confirm the transformation was correct, not just that *a* transformation
  happened.
- **Sanity-checking totals.** Do the aggregate numbers still make sense against
  the order-of-magnitude expectations from the mathematical-thinking lesson? A
  total that shifted implausibly after cleaning signals lost or duplicated data.

The re-run principle
----------------------

The most reliable verification is **re-running the detection that found the
defect**: the check that revealed a problem should reveal *none* after the fix.
This closes the loop — detection, correction, re-detection — and it is why the
viewing-data-differently lenses and the SQL detection queries are verification
tools as much as discovery tools. If the duplicate query still finds duplicates,
the cleaning did not work, and you know it immediately rather than discovering it
in the analysis.

The caveat
------------

Verifying your own cleaning has a blind spot: you tend to check for the problems
you *thought about*, and miss the ones you did not — the same cognitive limit as
confirmation bias, applied to your own work. Re-running your own detection
confirms the defects you looked for are fixed, but cannot reveal defects you
never checked for. This is why fresh eyes help (a colleague spotting what you
assumed), and why the systematic defect *checklist* from earlier matters — it
prompts checks you might not think to make. Verify against a list, not just
against memory. The next lesson gives the concrete spreadsheet and SQL techniques.

.. hint::

   - :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`
   - :doc:`Verification Techniques: Using Spreadsheets and SQL to Catch Repeated Errors <027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>`
   - :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`
   - :doc:`Documenting Data-Cleaning Changes <028-documenting-data-cleaning-changes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/verifying-data-cleaning-efforts/ <https://insightful-data-lab.com/2023/11/01/verifying-data-cleaning-efforts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: verify
