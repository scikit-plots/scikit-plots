:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-004:
.. _da-5-analyze-analyze-004:

========================================================================
Sorting Data in Spreadsheets
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 004`

◀ :doc:`Previous <003-sorting-and-filtering-in-data-analysis>` · :doc:`Next <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Sorting, in practice
----------------------

The first hands-on analysis technique is **sorting in a spreadsheet** — ordering
rows by the values in one or more columns to surface the structure analysis
depends on. It is simple to do and immediately useful, and doing it *safely* is
the one thing that separates a helpful sort from a data-corrupting one.

Single-column sorting
-----------------------

The basic sort orders the whole table by one column, ascending or descending:

- **Ascending** — A→Z for text, smallest→largest for numbers,
  earliest→latest for dates.
- **Descending** — the reverse: Z→A, largest→smallest, latest→earliest.

Sort a sales column descending and the biggest sales rise to the top; sort a date
column ascending and the timeline orders itself. The single-column sort answers
"what are the extremes?" and "what is the order?" directly.

Multi-column sorting
----------------------

Sorting by *several* columns in priority order handles finer questions. Sort by
region first, then by sales within region: the spreadsheet groups all rows of each
region together, and within each region orders them by sales. This reveals "the
top sellers *within each* region" — a two-level structure a single sort cannot
show. The column order matters: the first sort column is primary, the second
breaks ties within it, and so on.

The critical safety rule
--------------------------

The one rule that makes sorting safe: **always sort the entire table together, so
every row moves as a complete unit.** When you sort, the spreadsheet must reorder
*all* columns' values by row, keeping each record intact. Sorting a *single column
in isolation* — reordering one column while the others stay put — silently
destroys the data by breaking the correspondence between a record's values: now a
customer's name sits beside a different customer's sales. This produces no error,
just quietly corrupted data, and it is one of the most damaging spreadsheet
mistakes. Modern spreadsheets warn you and default to whole-table sorting, but the
danger is real enough that the rule bears repeating: the record is the unit; sort
records, never lone columns.

Sorting as an analytical tool
-------------------------------

Beyond its safety mechanics, sorting is genuinely analytical: it is often the
fastest way to answer a ranking question ("who are our biggest customers?"), spot
the range of a variable (its smallest and largest values sit at the two ends), and
prepare data for a scan. Combined with filtering — filter to a subset, then sort
it — it handles a large share of everyday analytical questions with two clicks.

The caveat
------------

Sorting permanently reorders the data (unlike a filter, which only hides), so the
original row order — if it carried meaning, such as the sequence of entry — is
lost once you sort, unless you preserved it (an index column, or a copy). Before
sorting, consider whether the current order matters and needs saving; the
raw-stays-raw discipline protects you here. And sorting reveals extremes but does
not *explain* them — a value at the top of a sort is worth investigating, not
automatically a finding. The next lesson brings sorting, and filtering, into SQL.

.. hint::

   - :doc:`Sorting and Filtering in Data Analysis <003-sorting-and-filtering-in-data-analysis>`
   - :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <../3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`
   - :doc:`Data Organization in Analysis <002-data-organization-in-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/sorting-data-in-spreadsheets/ <https://insightful-data-lab.com/2023/11/26/sorting-data-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
