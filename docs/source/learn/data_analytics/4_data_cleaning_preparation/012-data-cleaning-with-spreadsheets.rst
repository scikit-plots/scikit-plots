:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-012:
.. _da-4-cleaning-cleaning-012:

========================================================================
Data Cleaning with Spreadsheets
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 012`

◀ :doc:`Previous <011-common-issues-in-dirty-data>` · :doc:`Next <013-cleaning-and-merging-multiple-datasets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Cleaning, hands on
--------------------

With the defects catalogued, this lesson turns to actually *fixing* them — and
the spreadsheet, the analyst's most accessible tool, is where hands-on cleaning
most often begins. Spreadsheets offer built-in features for each common defect,
and their visibility — you *see* every change — makes them an excellent place to
learn cleaning before the same operations scale up to SQL and Python.

Spreadsheet cleaning, defect by defect
----------------------------------------

- **Duplicates** — the *Remove Duplicates* feature finds and deletes identical
  rows in one operation; conditional formatting can *highlight* duplicates first
  so you see them before removing.
- **Inconsistent text** — *Find and Replace* standardises variants ("NY" →
  "New York") across the sheet; *TRIM* removes stray leading and trailing spaces;
  *UPPER*/*LOWER*/*PROPER* normalise capitalisation.
- **Missing values** — *filtering* to blanks isolates the affected rows so you
  can decide per column whether to remove or fill them; sorting brings blanks
  together for inspection.
- **Wrong types** — the type-conversion tools and format settings turn
  text-numbers into real numbers and text-dates into real dates (the import-trap
  fixes from the prep section).
- **Splitting and combining** — *Text to Columns* splits a crammed column ("New
  York, NY" → city and state); *CONCATENATE* / ``&`` combines columns when
  needed — restoring the one-variable-per-column tabular structure.
- **Outliers and invalid values** — *sorting* a column surfaces the extremes at
  top and bottom for inspection; *conditional formatting* flags values outside a
  valid range for review.

The disciplined workflow
--------------------------

The features are only half the lesson; *how* you apply them is the other half,
and it is where the section's integrity disciplines become concrete:

1. **Work on a copy.** Duplicate the raw data to a working tab and clean *there*
   — raw stays raw, so a mistake is never fatal.
2. **Clean one defect at a time.** Handle duplicates, then inconsistencies, then
   types — in steps, each verifiable, rather than a tangled all-at-once edit.
3. **Verify after each step.** Check the row count after removing duplicates,
   spot-check replaced values, confirm converted types compute — catch a bad fix
   immediately, not three steps later.
4. **Document what you did.** Record each cleaning action in a notes tab, so the
   work is reproducible and a reviewer (or you, later) can see exactly what
   changed and why.

This workflow is what separates trustworthy cleaning from risky manual surgery —
the same fix applied carelessly can corrupt data, while applied with copies,
steps, verification, and documentation it reliably improves it.

The caveat
------------

Spreadsheet cleaning is powerful for modest data but shares the spreadsheet's
limits: it strains on very large datasets, and — most dangerously — manual
cleaning is **not automatically reproducible**. A series of hand-applied
Find-and-Replaces and deletions leaves no record unless you *make* one, so
re-cleaning next month's data means redoing the work from memory, with the risk
of doing it differently. This is precisely why cleaning graduates to SQL and
Python (and documented, rerunnable scripts) as data grows and recurs — a theme
the following lessons develop. For now, the spreadsheet is where the cleaning
intuitions form, one visible fix at a time.

.. hint::

   - :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`
   - :doc:`Spreadsheet Tools for Data Cleaning <014-spreadsheet-tools-for-data-cleaning>`
   - :doc:`Using Spreadsheet Functions for Data Cleaning <015-using-spreadsheet-functions-for-data-cleaning>`
   - :doc:`Cleaning and Merging Multiple Datasets <013-cleaning-and-merging-multiple-datasets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/data-cleaning-with-spreadsheets/ <https://insightful-data-lab.com/2023/11/01/data-cleaning-with-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty
