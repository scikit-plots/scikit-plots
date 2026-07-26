:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-cleaning-019:
.. _da-4-cleaning-cleaning-019:

========================================================================
Spreadsheets vs. SQL
========================================================================

:bdg-primary:`🧽 Data Cleaning & Preparation` :bdg-secondary:`🐬 Cleaning with SQL` :bdg-info:`Lesson 019`

◀ :doc:`Previous <018-introduction-to-sql>` · :doc:`Next <020-core-sql-queries-for-data-cleaning-and-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Two tools, different jobs
---------------------------

You now have two cleaning-and-analysis tools — the spreadsheet and SQL — and a
recurring practical question: which to use when? Neither is universally better;
each excels at different tasks, and knowing the trade-offs lets you reach for the
right one instead of forcing every job into your favourite. This lesson lays out
the comparison directly.

Where spreadsheets win
------------------------

- **Small to medium data** — anything that fits comfortably in a sheet (up to
  tens of thousands of rows).
- **Visibility and directness** — you *see* all the data and every change
  immediately; you can click a cell and edit it. This makes spreadsheets ideal
  for exploration, one-off fixes, and quick manual work.
- **Accessibility** — everyone has one and can open your file; no database
  access or query knowledge required, which matters for sharing with
  non-analysts.
- **Ad-hoc analysis and presentation** — quick charts, pivot tables, formatted
  summaries a stakeholder reads.

Where SQL wins
----------------

- **Large data** — millions of rows SQL handles in seconds and a spreadsheet
  cannot even open.
- **Reproducibility** — a query is repeatable text; the same clean or analysis
  reruns identically on new data, where manual spreadsheet work must be redone.
- **Working at the source** — SQL queries the database directly, avoiding fragile
  export-and-reimport cycles.
- **Complex, multi-table work** — joining several large tables and aggregating is
  SQL's home turf.
- **Automation** — queries embed in pipelines that run on a schedule.

The comparison, summarised
----------------------------

A rough rule: **spreadsheets for small, visual, one-off, shareable work; SQL for
large, reproducible, source-level, automated work.** Data size, whether the work
will *recur*, and who consumes the result usually settle the choice — the same
three questions (size, repetition, audience) the tools-overview lesson posed in
the foundations, now with hands-on experience of both sides.

They work together
--------------------

The choice is rarely exclusive; the tools *chain*. A common professional pattern:
use **SQL** to clean and aggregate large data at the source, export a manageable
result, then use a **spreadsheet** to explore it, chart it, and build the summary
a stakeholder reads. SQL does the heavy lifting on volume; the spreadsheet does
the last-mile presentation. Fluency across both — and knowing which stage each
serves — is what the coming sections build, culminating in Python, which can do
the work of both and automate the whole chain.

The caveat
------------

The spreadsheet-versus-SQL choice is a spectrum, not a wall, and the boundary
moves — modern spreadsheets connect to databases and handle larger data than they
once could, while SQL tools increasingly offer visual interfaces. The durable
guidance is not a fixed row-count threshold but the *reasoning*: match the tool
to the data's scale, the work's repetition, and the audience's needs, and expect
to use both, often on the same project. The next lessons get concrete about the
SQL cleaning operations themselves.

.. hint::

   - :doc:`Introduction to SQL <018-introduction-to-sql>`
   - :doc:`Core SQL Queries for Data Cleaning and Analysis <020-core-sql-queries-for-data-cleaning-and-analysis>`
   - :doc:`Querying Data with SQL <../3_data_preparation/023-querying-data-with-sql>`
   - :doc:`Data Cleaning with Spreadsheets <012-data-cleaning-with-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/01/spreadsheets-vs-sql/ <https://insightful-data-lab.com/2023/11/01/spreadsheets-vs-sql/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: cleaning, topic: sql
