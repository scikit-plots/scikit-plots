:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-028:
.. _da-5-analyze-analyze-028:

========================================================================
Data Validation as an Ongoing Analytical Process
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🚀 Validation & Temporary Tables` :bdg-info:`Lesson 028`

◀ :doc:`Previous <027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>` · :doc:`Next <029-temporary-tables-and-the-with-clause-in-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Validation never really ends
------------------------------

Data validation appeared earlier as a spreadsheet feature and a cleaning step, but
its deepest form is a *mindset*: **validation as an ongoing analytical process**,
running continuously throughout analysis rather than once at the start. Opening the
advanced stage, this lesson reframes validation from a gate you pass to a habit you
maintain — checking that data and results remain sound at every step.

Why validation must be continuous
-----------------------------------

Data can go wrong at *any* point in analysis, not just at entry. A join can
introduce duplicates; a calculation can produce impossible values; a filter can
silently drop rows; a transformation can corrupt a field. Validating once at the
beginning cannot catch problems that arise *during* the work — so validation must
happen throughout: after each significant operation, confirm the data still makes
sense. This is the integrity-in-motion principle from the cleaning section, elevated
into a continuous analytical discipline.

What ongoing validation looks like
------------------------------------

Validation as a process weaves checks through the whole analysis:

- **After each transformation** — did the row count change as expected? Do the
  values still fall in valid ranges? (The verification techniques from the cleaning
  section, applied continuously.)
- **Against expectations** — does each result pass the order-of-magnitude sanity
  check? A number wildly off from what you expected signals a problem to
  investigate, not a finding to report.
- **Cross-checks** — does the total from one method match the total from another?
  Do the parts sum to the whole? Independent computations that should agree are a
  powerful validation.
- **Against reality** — does the result make sense in the real world? A negative
  count, a percentage over 100, a customer older than 150 — impossibilities that
  signal an error somewhere upstream.

The point is that validation is not a phase but a *reflex*, applied at every step.

Validation and trustworthy analysis
--------------------------------------

Continuous validation is what makes analysis *trustworthy* rather than merely
completed. Analysis that runs end to end without validation produces a result, but
no assurance the result is correct — and dirty data, bad joins, and calculation
errors all produce plausible-looking wrong answers that only validation catches.
The analysts whose work can be relied on are those who validate continuously, so
that by the time they present a finding, it has survived checks at every step. This
is the whole course's check-your-results habit, made into a systematic practice.

The caveat
------------

Continuous validation is essential but can tip into paralysis — validating so
obsessively that the analysis never progresses, or treating every minor anomaly as
a crisis. The judgement is *proportionate* validation: check the things most likely
to be wrong and most consequential if they are, at the points where errors most
often enter (transformations, joins, calculations), without re-checking everything
constantly. And validation confirms data is *plausible and consistent*, not
necessarily *correct* — it catches errors, but passing validation is not proof of
truth. Validate enough to trust the work, proportionate to its stakes. The next
lessons turn to a technique that makes complex, validated analysis manageable:
temporary tables.

.. hint::

   - :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`
   - :doc:`Combining Data Validation and Conditional Formatting in Spreadsheets <008-combining-data-validation-and-conditional-formatting-in-spreadsheets>`
   - :doc:`Verifying and Reporting Data Integrity <../4_data_cleaning_preparation/025-verifying-and-reporting-data-integrity>`
   - :doc:`Temporary Tables and the WITH Clause in SQL <029-temporary-tables-and-the-with-clause-in-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/data-validation-as-an-ongoing-analytical-process/ <https://insightful-data-lab.com/2023/11/26/data-validation-as-an-ongoing-analytical-process/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: advanced
