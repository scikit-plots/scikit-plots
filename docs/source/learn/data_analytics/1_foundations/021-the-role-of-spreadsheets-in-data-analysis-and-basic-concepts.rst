:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-foundations-021:
.. _data-analytics-foundations-021:
.. _da-foundations-foundations-021:
.. _da-decisions-foundations-021:
.. _da-prep-foundations-021:
.. _da-cleaning-foundations-021:
.. _da-analyze-foundations-021:
.. _da-viz-foundations-021:
.. _da-python-foundations-021:
.. _da-jobsearch-foundations-021:

========================================================================
The Role of Spreadsheets in Data Analysis and Basic Concepts
========================================================================

:bdg-primary:`🌱 Foundations` :bdg-secondary:`🧰 Tools, Applications & Ethics` :bdg-info:`Lesson 021`

◀ :doc:`Previous <020-overview-of-core-tools-used-by-data-analysts>` · :doc:`Next <022-the-concept-and-basic-use-of-sql-query-language>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The visible workbench
-----------------------

A **spreadsheet** is data you can *see*: a grid where every value sits in a
labelled cell, every change is immediate, and every intermediate step is
inspectable. That visibility is why it is the first tool this course teaches
and the tool most analyses still begin in — it makes the abstract operations of
analysis (sort, filter, compute, summarise) concrete and watchable.

The anatomy
-------------

- **Cells** — the atoms, each addressed by column letter and row number
  (``B7``). A cell holds a value *or* a formula that computes one.
- **Rows and columns** — by convention, a **row is one record** (one sale, one
  customer, one day) and a **column is one attribute** (date, region, amount).
  Keeping that convention is half of good data design.
- **Headers** — the first row, naming each column. A header is context made
  explicit: ``ORDER_DATE`` tells you what the values below mean.
- **Formulas** — expressions beginning with ``=`` that compute from other
  cells: ``=C2*D2`` multiplies price by quantity. Change an input and every
  dependent formula updates — the spreadsheet's quiet superpower.
- **Functions** — named, prebuilt operations used inside formulas:
  ``=SUM(E2:E100)``, ``=AVERAGE(...)``, ``=COUNT(...)``, ``=MAX(...)``. They
  are the vocabulary the analysis sections expand enormously.

What analysts actually do with them
-------------------------------------

Across the six phases, the spreadsheet serves at least four roles: **inspect**
(eyeball raw data for obvious problems), **organise** (sort and filter into
meaningful order), **calculate** (derive new columns and summary figures), and
**communicate** (a labelled table or quick chart a stakeholder can open with no
special tools). A first pass on almost any small dataset — scan it, sort it,
total it — is a spreadsheet task done in minutes.

A worked miniature
--------------------

A sheet of orders with ``PRICE`` in column C and ``QUANTITY`` in column D:
add a header ``REVENUE`` in E1, put ``=C2*D2`` in E2, fill it down, and
``=SUM(E2:E101)`` gives total revenue for a hundred orders. Three formulas,
and raw records have become a business number — the entire shape of analysis,
in miniature.

The honest limits
-------------------

Spreadsheets strain as data grows past tens of thousands of rows, and their
flexibility is a double edge: any cell can be quietly overtyped, so errors hide
in plain sight and there is no built-in record of what changed. Later lessons
treat spreadsheet *errors* and *verification* as first-class topics for exactly
this reason. The scaling limit is what SQL, next, exists to remove.

.. hint::

   - :doc:`Overview of Core Tools Used by Data Analysts <020-overview-of-core-tools-used-by-data-analysts>`
   - :doc:`The Concept and Basic Use of SQL (Query Language) <022-the-concept-and-basic-use-of-sql-query-language>`
   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Spreadsheet Calculations with Formulas <../2_data_driven_decisions/014-spreadsheet-calculations-with-formulas>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/07/30/the-role-of-spreadsheets-in-data-analysis-and-basic-concepts/ <https://insightful-data-lab.com/2023/07/30/the-role-of-spreadsheets-in-data-analysis-and-basic-concepts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: foundations, topic: tools
