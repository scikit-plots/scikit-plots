:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-002:
.. _da-4-cleaning-cleaning-002:

========================================================================
Data Integrity and Its Risks in Data Analysis
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧱 Data Integrity & Sampling` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-the-importance-of-clean-data>` · :doc:`Next <003-aligning-data-with-business-objectives>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Clean, and staying clean
--------------------------

Cleaning data once is not enough; data must *stay* accurate and consistent
through everything that happens to it. **Data integrity** is the accuracy,
completeness, consistency, and trustworthiness of data throughout its life cycle
— the property that data remains sound as it is stored, moved, transformed, and
combined. Where the previous lesson was about *achieving* clean data, integrity
is about *preserving* it against the many things that can quietly corrupt it.

What threatens integrity
--------------------------

Data integrity has recognisable enemies, and analysts meet them constantly:

- **Replication and syncing errors** — when data is copied between systems and
  the copies drift out of agreement, so the warehouse and the spreadsheet
  disagree about the same fact (the silo problem from the data-life-cycle
  lessons).
- **Transfer errors** — data corrupted or truncated while moving between systems
  or formats: a field cut short, an encoding mangled, rows dropped mid-import.
- **Transformation errors** — mistakes introduced *by processing itself*: a bad
  join that duplicates rows, a formula that drifts (the relative-reference bug),
  a conversion that loses precision. Cleaning done carelessly can *reduce*
  integrity.
- **Human error** — manual edits, mistyped corrections, accidental deletions —
  the reason the raw-stays-raw and access-control disciplines exist.
- **Storage and system issues** — corruption, incomplete writes, hardware
  faults.

The unifying theme: integrity is threatened most often not at rest but *in
motion* — every time data is copied, moved, transformed, or edited is a chance
for it to degrade.

Integrity and the cleaning process itself
-------------------------------------------

A subtle and important point: the act of cleaning data is itself a risk to
integrity. Every transformation you apply to fix one problem can introduce
another — dropping rows to remove duplicates might delete legitimate records, a
find-and-replace might over-match, a type conversion might silently fail. This is
precisely why the disciplines of this section exist: work on copies (raw stays
raw), document every change, and *verify* after each step. Cleaning without those
safeguards trades one integrity problem for another.

Why analysts must protect it
------------------------------

Integrity failures produce the same plausible-wrong results as dirty data and
bias, and they are often *harder* to spot because the data looked fine when it
arrived — the corruption happened in transit or in processing. An analyst who
assumes data kept its integrity through every hop is assuming away a major risk.
The habits that protect integrity — checking row counts before and after each
operation, validating that transformations did what was intended, keeping an
untouched original — are the same sanity-check and verification habits the course
has built throughout, applied to the data's whole journey.

The caveat
------------

Perfect integrity through a long pipeline is hard to guarantee, and some loss is
often invisible until something downstream breaks. The professional standard is
not a guarantee but *vigilance and traceability*: minimising the risky
operations, checking integrity at each step, and keeping enough of a trail
(untouched raw data, documented transformations) that when a problem does
surface, you can find where it entered. The next lessons turn to a different
integrity question — whether the data fits the objective, and whether there is
enough of it.

.. hint::

   - :doc:`The Importance of Clean Data <001-the-importance-of-clean-data>`
   - :doc:`Aligning Data with Business Objectives <003-aligning-data-with-business-objectives>`
   - :doc:`Verifying and Reporting Data Integrity <025-verifying-and-reporting-data-integrity>`
   - :doc:`Sample Size and Data Integrity <007-sample-size-and-data-integrity>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/10/31/data-integrity-and-its-risks-in-data-analysis/ <https://insightful-data-lab.com/2023/10/31/data-integrity-and-its-risks-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: integrity
