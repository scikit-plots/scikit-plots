:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-003:
.. _da-5-analyze-analyze-003:

========================================================================
Sorting and Filtering in Data Analysis
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 003`

◀ :doc:`Previous <002-data-organization-in-analysis>` · :doc:`Next <004-sorting-data-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The two foundational moves
----------------------------

Two operations underlie a huge share of all data analysis: **sorting** (ordering
data) and **filtering** (subsetting data). You met them as *cleaning* tools in the
preparation section; here they return as *analysis* tools, and the shift in
purpose is worth making explicit — the same operations, aimed now at finding
answers rather than fixing defects.

Sorting for analysis
----------------------

**Sorting** arranges rows in order by one or more columns, and in analysis its
purpose is to *reveal structure*:

- **Ranking** — sort descending to find the top (best-selling products, highest
  spenders) or ascending for the bottom.
- **Extremes** — sorting surfaces the largest and smallest values, where outliers
  and notable cases live.
- **Chronology** — sorting by date reveals the sequence and makes trends over time
  visible.
- **Grouping visually** — sorting by a category brings like records together for
  comparison.

Where cleaning used sorting to *find defects*, analysis uses it to *find answers*
— but the operation is identical.

Filtering for analysis
------------------------

**Filtering** shows only the rows meeting a condition, and in analysis its purpose
is *focus*:

- **Isolating a subset** — analyse just one segment (a region, a period, a
  customer type) by filtering to it.
- **Answering conditional questions** — "how many orders over $100?" is a filter
  plus a count.
- **Comparing subsets** — filter to one group, note the result; filter to another,
  compare.

Filtering is how you direct analysis at exactly the part of the data a question
concerns, ignoring the rest.

Sorting and filtering together
--------------------------------

Combined, they answer a vast range of questions with no further machinery: filter
to a subset, then sort it to rank — "the top ten customers in the northern region
this quarter" is one filter and one sort. This filter-then-sort pattern is the
analyst's most reflexive analytical move, and it recurs in every tool: spreadsheet
filter-and-sort, SQL ``WHERE`` plus ``ORDER BY``. Mastering it is mastering the
foundation the more advanced techniques build on.

The caveat
------------

Sorting and filtering shape what you *see*, which makes them subtly capable of
*misleading* an analysis — filtering to only the data that supports a conclusion
(dropping the inconvenient subset), or reading a sorted extreme as typical when it
is exceptional. The discipline is to filter and sort to answer the *question*
fairly, not to manufacture a desired result, and to stay aware of what a filtered
view excludes. And a reminder from the cleaning section: a filter only *hides*
rows while a sort *reorders* them — know which is active, and never mistake a
filtered view for the whole. The next lessons implement sorting in spreadsheets
and SQL.

.. hint::

   - :doc:`Data Organization in Analysis <002-data-organization-in-analysis>`
   - :doc:`Sorting Data in Spreadsheets <004-sorting-data-in-spreadsheets>`
   - :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <../3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/sorting-and-filtering-in-data-analysis/ <https://insightful-data-lab.com/2023/11/26/sorting-and-filtering-in-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
