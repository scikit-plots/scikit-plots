:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-021:
.. _da-3-prep-prep-021:

========================================================================
Sorting and Filtering Data in Spreadsheets
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🔢 Spreadsheets, SQL & Organization` :bdg-info:`Lesson 021`

◀ :doc:`Previous <020-importing-data-into-spreadsheets>` · :doc:`Next <022-bigquery-account-types>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The two moves you reach for first
-----------------------------------

Once data is imported and tidy, the first things an analyst does to make sense
of it are **sort** and **filter**. They are the most-used operations in any
spreadsheet, the fastest way to turn a wall of rows into something legible, and
the intuition behind their SQL equivalents (``ORDER BY`` and ``WHERE``) that the
analysis section develops. This lesson makes them precise.

Sorting: imposing order
-------------------------

**Sorting** reorders the rows of a table by the values in one or more columns —
ascending (A→Z, smallest→largest, earliest→latest) or descending. Its analytical
value is that order reveals: sort sales descending and the top performers rise to
the top; sort by date and the timeline becomes visible; sort by region then by
sales and you see the best within each group.

- **Single-column sort** orders by one column.
- **Multi-column sort** orders by one column, breaking ties with a second (region
  first, then sales within each region) — the spreadsheet applies them in
  priority order.

The one non-negotiable rule: sort the **whole table together**, so every row
moves as a unit. Sorting a single column in isolation — leaving the others in
place — silently scrambles which value belongs to which record, one of the most
destructive spreadsheet mistakes precisely because it produces no error, just
quietly corrupted data. (The one-row-one-record discipline from the organisation
lesson is what makes whole-table sorting safe.)

Filtering: narrowing the view
-------------------------------

**Filtering** temporarily hides rows that do not meet a condition, showing only
those that do — orders over $100, one region, this month's dates. Crucially,
filtering *hides* rather than *deletes*: the data is all still there, and
clearing the filter restores the full view. Filters can combine conditions
(region = "North" *and* amount > 100) to narrow to exactly the subset a question
concerns.

Filtering's analytical value is focus: most questions concern a *subset*, and
filtering isolates it so you can examine or summarise just that slice without the
rest as noise.

Sorting and filtering together
--------------------------------

The two combine constantly: filter to this quarter's northern orders, then sort
them by value to see the largest. This filter-then-sort move answers a huge range
of everyday questions ("what were our biggest northern deals this quarter?")
with two clicks and no formulas — which is exactly why it is the analyst's
reflexive first pass on new data.

The caveat
------------

Sorting and filtering change what you *see*, and it is easy to forget a filter is
active — drawing conclusions from a filtered view as though it were the whole
dataset, or exporting filtered data thinking it is complete. Always know whether
a filter is on, and remember that a *sort* permanently reorders the data (it
persists after you look away) while a *filter* only hides — different footprints,
both easy to lose track of. The next lessons move from the spreadsheet to
querying data at database scale.

.. hint::

   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Importing Data into Spreadsheets <020-importing-data-into-spreadsheets>`
   - :doc:`Sorting and Filtering in Data Analysis <../5_analyze_data/003-sorting-and-filtering-in-data-analysis>`
   - :doc:`Querying Data with SQL <023-querying-data-with-sql>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/sorting-and-filtering-data-in-spreadsheets/ <https://insightful-data-lab.com/2023/09/04/sorting-and-filtering-data-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: spreadsheets_sql
