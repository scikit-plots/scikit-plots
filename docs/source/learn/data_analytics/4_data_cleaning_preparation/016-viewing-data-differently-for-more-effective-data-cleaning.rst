:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-016:
.. _da-4-cleaning-cleaning-016:

========================================================================
Viewing Data Differently for More Effective Data Cleaning
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🧹 Dirty Data & Spreadsheet Cleaning` :bdg-info:`Lesson 016`

◀ :doc:`Previous <015-using-spreadsheet-functions-for-data-cleaning>` · :doc:`Next <017-data-mapping-and-the-big-picture-of-clean-data>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


You cannot fix what you cannot see
------------------------------------

Every cleaning technique so far assumes you have *found* the defect — but the
hardest part of cleaning is often *seeing* the problem in the first place. A few
bad rows hide easily in thousands of good ones. The most effective cleaners are
skilled at **viewing data differently** — using sorting, filtering, pivoting, and
formatting as *lenses* that make hidden defects jump out. This lesson is about
detection, the step before correction.

Lenses that reveal defects
----------------------------

- **Sorting** surfaces the extremes. Sort a numeric column and outliers,
  negatives, and impossible values collect at the top and bottom where you can
  see them; sort a text column and inconsistent spellings ("New York" vs "new
  york") land *next to each other* instead of scattered, and blanks group
  together. Sorting turns "somewhere in ten thousand rows" into "at the top."
- **Filtering** isolates suspects. Filter to blanks to see all missing values at
  once; filter to a category to check whether it is recorded consistently; filter
  to a range to find values outside it. Filtering removes the noise so the subset
  you are checking stands alone.
- **Pivot tables** summarise into patterns. A pivot counting records per category
  instantly reveals inconsistency: if "New York", "NY", and "new york" appear as
  *three separate rows* with separate counts, the fragmentation is visible at a
  glance — and a pivot of counts per value is one of the fastest ways to audit a
  column's consistency.
- **Conditional formatting** flags visually. Colour rules highlight duplicates,
  mark out-of-range values, or shade blanks, so defects are *visible* while you
  scroll rather than hidden in uniform grids.

The detection mindset
-----------------------

Behind the lenses is a habit: approach a new dataset *expecting* defects and
actively hunting them, rather than assuming it is clean until an error surfaces.
Run the column-consistency pivot, sort each key numeric column to check its
extremes, filter to blanks, scan the value counts. This deliberate
multi-angle inspection — looking at the same data sorted, filtered, summarised,
and formatted — finds the quiet defects (a handful of duplicates, one stray
category, a few impossible values) that a single straight-on glance always
misses. It is the same "view it from several angles" principle behind analytical
thinking, applied to finding dirt.

Detection before correction
------------------------------

The sequence matters: **see the full scope of a defect before fixing it**. A
pivot showing that "NY" appears 400 times and "New York" 3,000 times tells you
the standardisation is worth doing *and* which form to standardise toward.
Filtering to blanks shows *how many* values are missing before you decide whether
to remove or fill them. Jumping to correction before understanding the defect's
extent leads to incomplete or wrong fixes — you standardise the two spellings you
noticed and miss the third. Look first, fully, then fix.

The caveat
------------

Viewing lenses reveal defects but can also *mislead* about them: a sort makes a
few extreme values look alarming when they may be legitimate, a filtered view can
be mistaken for the whole dataset, and a pivot's groupings depend on the data
being typed correctly (a pivot cannot group text-numbers sensibly). The lenses
are for *detection and investigation*, not automatic judgement — seeing an
anomaly is the start of asking whether it is an error or a real feature, not a
verdict. This closes the hands-on dirty-data cleaning; the next lessons step back
to the big picture of clean data and the transition to cleaning with SQL.

.. hint::

   - :doc:`Using Spreadsheet Functions for Data Cleaning <015-using-spreadsheet-functions-for-data-cleaning>`
   - :doc:`Spreadsheet Tools for Data Cleaning <014-spreadsheet-tools-for-data-cleaning>`
   - :doc:`Data Mapping and the Big Picture of Clean Data <017-data-mapping-and-the-big-picture-of-clean-data>`
   - :doc:`Common Issues in Dirty Data <011-common-issues-in-dirty-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/viewing-data-differently-for-more-effective-data-cleaning/ <https://insightful-data-lab.com/2023/11/01/viewing-data-differently-for-more-effective-data-cleaning/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: dirty
