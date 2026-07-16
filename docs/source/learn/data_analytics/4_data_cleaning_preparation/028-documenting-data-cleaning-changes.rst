:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-028:
.. _da-4-cleaning-cleaning-028:

========================================================================
Documenting Data-Cleaning Changes
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`✅ Verification, Documentation & Next Steps` :bdg-info:`Lesson 028`

◀ :doc:`Previous <027-verification-techniques-using-spreadsheets-and-sql-to-catch-repeated-errors>` · :doc:`Next <029-reporting-data-cleaning-results>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The record that makes cleaning real
-------------------------------------

Cleaning that is not documented is cleaning that cannot be trusted, reproduced,
or defended. **Documenting data-cleaning changes** — keeping a clear record of
what was changed, why, and how — is what turns a series of ad-hoc fixes into a
transparent, reproducible process. It is the discipline that has run beneath this
entire section, stated now as its own practice, and it is what separates
professional data work from irreproducible manual editing.

What to document
------------------

A cleaning record captures, for each change or the cleaning as a whole:

- **What was changed** — which columns, which operations: "removed duplicate
  customer records", "standardised state abbreviations to full names", "converted
  ``price`` from text to decimal".
- **Why** — the defect that motivated it: "312 exact-duplicate rows inflated
  counts", "state recorded inconsistently across three formats".
- **How** — the method or query used, ideally the actual code: the SQL query, the
  formula, the tool applied. The *how* is what makes it reproducible.
- **What was affected** — how many records changed, and any that could not be
  cleaned and were excluded or flagged.
- **What remains** — known limitations, unresolved issues, decisions deferred
  (the missing values filled with a default, the outliers left in pending
  review).

Why documentation matters
---------------------------

Documentation serves several masters at once. **Reproducibility** — next month's
data can be cleaned the same way, because the steps are recorded (and if they are
recorded *as code*, re-running is trivial). **Reviewability** — a colleague, or an
auditor, can see exactly what was done and judge whether it was right.
**Trust** — a stakeholder who can see the cleaning log trusts the data more than
one asked to take it on faith. **Your future self** — six months on, you will not
remember why you excluded those records; the documentation will tell you. And
**error-tracing** — when a problem surfaces downstream, the cleaning log is where
you look to find whether cleaning caused it.

Documentation as reproducibility
----------------------------------

The deepest reason to document connects to the whole section's arc: the
difference between manual spreadsheet cleaning and scripted SQL/Python cleaning is
largely that the latter *documents itself* — the query or script **is** the record
of what was done. When cleaning is manual, documentation must be added
deliberately (a notes tab, a change log); when it is code, the code is the
documentation, which is a further reason professional cleaning gravitates toward
rerunnable scripts. Either way, the principle is the same: the cleaning must leave
a trail.

The caveat
------------

Documentation has a cost and a failure mode: over-documenting trivial changes
buries the important ones, and documentation that drifts out of sync with what was
actually done is worse than none — a change log that says one thing while the data
reflects another misleads confidently. The goal is documentation that is
*accurate, sufficient, and maintained* — enough to reproduce and review the
cleaning, kept truthful to what actually happened, without ceremony that no one
will sustain. This completes the verification stage; the final lessons of the
section turn to reporting cleaning results and using cleaning feedback to improve
data quality at the source.

.. hint::

   - :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`
   - :doc:`Reporting Data-Cleaning Results <029-reporting-data-cleaning-results>`
   - :doc:`The Importance of Clean Data <001-the-importance-of-clean-data>`
   - :doc:`Verifying Data-Cleaning Efforts <026-verifying-data-cleaning-efforts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/documenting-data-cleaning-changes/ <https://insightful-data-lab.com/2023/11/01/documenting-data-cleaning-changes/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: verify
