:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-013:
.. _da-4-cleaning-cleaning-013:

========================================================================
Cleaning and Merging Multiple Datasets
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 013`

◀ :doc:`Previous <012-data-cleaning-with-spreadsheets>` · :doc:`Next <014-spreadsheet-tools-for-data-cleaning>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


When cleaning means combining
-------------------------------

Real analysis rarely draws on a single tidy table; it usually **combines**
several — sales from one system, customers from another, products from a third.
Merging datasets is powerful, but it multiplies cleaning challenges, because now
the data must be consistent not only *within* each source but *across* them.
This lesson covers the distinct problems that arise when data from multiple
sources is brought together.

The merge and its prerequisite
--------------------------------

Merging combines datasets by matching records on a shared field — a **key** that
identifies the same entity across tables (a ``customer_id`` present in both an
orders table and a customers table). The relational-database concepts from the
prep section are exactly this: primary and foreign keys are what make a clean
merge possible. The prerequisite for a correct merge is that the key **means the
same thing and is formatted the same way** in every dataset — and that is where
most merge problems live.

The cross-dataset cleaning problems
-------------------------------------

- **Mismatched keys.** The join field is formatted differently across sources —
  ``"C-001"`` in one, ``"001"`` in another, or one storing IDs as numbers and
  the other as text. Records that *should* match silently do not, and rows go
  missing from the result. *Fix:* standardise the key's format in every source
  before merging.
- **Inconsistent formats.** The same field recorded differently across sources —
  dates as ``MM/DD/YYYY`` here and ``YYYY-MM-DD`` there, categories labelled
  differently, units differing. *Fix:* reconcile to one format before combining.
- **Duplicate and conflicting records.** The same entity appearing in multiple
  sources, sometimes with *disagreeing* values (two addresses for one customer).
  *Fix:* decide a rule for which source wins, and deduplicate after merging.
- **Different granularity.** One dataset is per-transaction, another is
  per-day-summary; merging them naively double-counts or misaligns. *Fix:*
  aggregate to a common grain first.
- **Structural mismatch.** Different column names for the same thing, or wide
  versus long layouts (the prep-section shapes). *Fix:* align structure and
  naming before the merge.

The disciplined merge
-----------------------

A safe merge follows a sequence: **clean each source individually first** (so
you are combining already-tidy data), **standardise the join key** across
sources, **verify the match** (how many records matched, how many did not, and
*why* the unmatched failed), and **check the result's row count** against
expectation — a merge that unexpectedly multiplies rows usually signals a
duplicate key or a many-to-many join gone wrong. Verifying the merge is as
important as performing it, because a bad merge silently drops or duplicates
data.

The caveat
------------

Merging is where the *integrity-in-motion* risk from earlier peaks: combining
datasets is a transformation, and transformations introduce errors — a join on a
non-unique key can explode row counts, a format mismatch can silently drop
records that belong. The row-count check before and after is not optional
housekeeping; it is the primary defence against a merge that quietly corrupted
the data. The deeper mechanics of combining data — VLOOKUP in spreadsheets, JOIN
in SQL — get full treatment in the analysis section; here the point is that
*cleaning must precede and follow every merge*.

.. hint::

   - :doc:`Data Cleaning with Spreadsheets <012-data-cleaning-with-spreadsheets>`
   - :doc:`Using VLOOKUP to Combine Data Across Spreadsheets <../5_analyze_data/015-using-vlookup-to-combine-data-across-spreadsheets>`
   - :doc:`Spreadsheet Tools for Data Cleaning <014-spreadsheet-tools-for-data-cleaning>`
   - :doc:`Data Mapping and the Big Picture of Clean Data <017-data-mapping-and-the-big-picture-of-clean-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/cleaning-and-merging-multiple-datasets/ <https://insightful-data-lab.com/2023/11/01/cleaning-and-merging-multiple-datasets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty
