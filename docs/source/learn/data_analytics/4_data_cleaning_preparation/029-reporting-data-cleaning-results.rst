:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-029:
.. _da-4-cleaning-cleaning-029:

========================================================================
Reporting Data-Cleaning Results
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`✅ Verification, Documentation & Next Steps` :bdg-info:`Lesson 029`

◀ :doc:`Previous <028-documenting-data-cleaning-changes>` · :doc:`Next <030-using-feedback-from-data-cleaning-to-improve-data-quality>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Cleaning is not done until it is communicated
-----------------------------------------------

Documenting cleaning creates the record; **reporting** it delivers that record to
the people who need it. Reporting data-cleaning results means communicating what
the cleaning process found and fixed — clearly, honestly, and to the right
audience — so that everyone relying on the data understands its state. It is the
data-cleaning application of the Section 2 principle that value is created only
when communicated, and it is what converts private cleaning work into shared
trust.

What a cleaning report contains
---------------------------------

A good cleaning report answers the questions a data consumer will have:

- **What was the data's starting state** — the defects found: how many
  duplicates, what inconsistencies, how much missing data.
- **What was done** — the cleaning actions taken, in plain language (the *what*
  and *why* from the documentation, summarised for the audience).
- **What changed** — how many records were affected, removed, or corrected, and
  the net effect on the dataset (e.g. "12,450 rows in, 12,138 after removing
  312 duplicates").
- **What remains** — the limitations: unresolved issues, records that could not
  be cleaned, decisions made about missing values, and how these bound the
  analysis.
- **The bottom line** — whether the data is now fit for its purpose, and with
  what caveats.

Matching the report to the audience
-------------------------------------

The adapting-communication principle applies directly. A fellow analyst wants the
detail — which queries, which counts, which edge cases. A business stakeholder
wants the bottom line — "the data is reliable for the top four regions; two
smaller regions had too much missing data to include." An auditor wants the full
documented trail. Same cleaning, different reports: lead with what each audience
needs to decide how much to trust the analysis built on this data.

Why honest reporting matters most here
----------------------------------------

Reporting cleaning results is where the temptation to overstate is strong and the
cost of doing so is high. It is tempting to present data as "cleaned" full stop,
implying a completeness that no real cleaning achieves. The honest report states
the *remaining* limitations as prominently as the fixes — because a stakeholder
who knows the data excludes two regions will interpret the analysis correctly,
while one who believes it is complete and perfect will over-trust it. This is the
fairness-and-honesty thread at the data-quality level: report what you fixed
*and* what you could not, so decisions rest on an accurate picture of the data's
reliability.

The caveat
------------

A cleaning report can err toward too much as easily as too little: burying the
one limitation that matters under exhaustive detail about trivial fixes serves no
one, and a report so long nobody reads it fails as surely as no report at all.
The goal is a report *proportionate* to the audience and the stakes — enough for
them to correctly gauge the data's trustworthiness, foregrounding what actually
affects the conclusions, without ceremony. The next lesson turns cleaning results
into lasting improvement.

.. hint::

   - :doc:`Documenting Data-Cleaning Changes <028-documenting-data-cleaning-changes>`
   - :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`
   - :doc:`Data Creates Value Only When It Is Communicated <../2_data_driven_decisions/007-data-creates-value-only-when-it-is-communicated>`
   - :doc:`Using Feedback from Data Cleaning to Improve Data Quality <030-using-feedback-from-data-cleaning-to-improve-data-quality>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/reporting-data-cleaning-results/ <https://insightful-data-lab.com/2023/11/01/reporting-data-cleaning-results/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: verify
