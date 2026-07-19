:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-009:
.. _da-4-cleaning-cleaning-009:

========================================================================
Dirty Data vs. Clean Data
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 009`

◀ :doc:`Previous <008-margin-of-error>` · :doc:`Next <010-the-importance-of-clean-data-revisited>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Two states of the same table
------------------------------

The integrity stage asked whether there is *enough* sound data; this stage
confronts the data's *quality* directly. **Dirty data** is data that is
incomplete, incorrect, inconsistent, duplicated, or otherwise flawed — data with
errors that make it unreliable to analyse. **Clean data** is its opposite: the
complete, accurate, consistent, unique, valid, and uniform data the first lesson
defined. Putting the two side by side makes the abstract quality criteria
concrete and shows exactly what cleaning must fix.

The contrast, defect by defect
--------------------------------

The same customer table, dirty versus clean:

.. code-block:: text

   DIRTY                              CLEAN
   name        city        spend      name        city        spend
   Jane Smith  New York    100        Jane Smith  New York     100
   jane smith  new york    100        (duplicate removed)
   Bob Jones   NY          -50        Bob Jones   New York       0
   Ann Lee                 250        Ann Lee     Boston       250
   Tom Web     Chicago     1O0        Tom Web     Chicago      100

Reading the defects: rows 1–2 are a **duplicate** (same person, different
casing); "NY"/"New York"/"new york" is an **inconsistency**; the ``-50`` spend is
**invalid** (negative); Ann Lee's blank city is **missing**; and ``1O0`` (letter
O for zero) is a **type error** masquerading as a number. Each is a specific,
nameable defect — and each would distort an analysis in a specific way: the
duplicate inflates counts, the inconsistency fragments the New York group, the
invalid value skews the average, the missing value breaks completeness, and the
type error breaks arithmetic.

Why the distinction is the whole job
--------------------------------------

Cleaning is precisely the work of moving data from the left column to the right —
identifying each defect and correcting it so the data becomes trustworthy. Naming
the state matters because "this data is dirty" is not actionable, but "this column
has duplicates, inconsistent categories, and three negative values" is a work
list. The rest of this stage catalogues the defects and the techniques for each;
this lesson establishes the target — what clean looks like — that all of it aims
at.

The caveat
------------

The dirty/clean line is not always obvious: is a ``-50`` spend an error, or a
legitimate refund? Is a blank a missing value, or a genuine "none"? Is an unusual
value a mistake, or a real outlier worth keeping? Cleaning requires *judgement
and context*, not blind rule-following — the same value can be dirt to remove or
signal to preserve depending on what it means, which is why understanding the
data (metadata, provenance, the business) precedes cleaning it. The next lesson
revisits *why* this painstaking work is worth doing.

.. hint::

   - :doc:`The Importance of Clean Data <001-the-importance-of-clean-data>`
   - :doc:`The Importance of Clean Data (revisited) <010-the-importance-of-clean-data-revisited>`
   - :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`
   - :doc:`Data Integrity and Its Risks in Data Analysis <002-data-integrity-and-its-risks-in-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/dirty-data-vs-clean-data/ <https://insightful-data-lab.com/2023/11/01/dirty-data-vs-clean-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty
