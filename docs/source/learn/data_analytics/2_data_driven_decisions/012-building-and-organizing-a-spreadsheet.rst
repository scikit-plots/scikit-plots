:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-012:

========================================================================
Building and Organizing a Spreadsheet
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📗 Spreadsheets for Analysis` :bdg-info:`Lesson 012`

◀ :doc:`Previous <011-spreadsheets-in-data-analysis>` · :doc:`Next <013-how-data-analysts-use-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Layout is destiny
-------------------

Whether a spreadsheet analysis takes ten minutes or a wrestling match is
usually decided before any formula is written — by the **layout**. A
well-organised sheet makes sorting, filtering, pivoting, and charting
one-step operations; a poorly organised one fights every move. The
organising principles are few and learnable in one sitting.

The rules of a workable sheet
-------------------------------

- **One row, one record.** Each row is one observation — one order, one
  customer, one day. Never let a single record sprawl across several rows, or
  two records share one.
- **One column, one attribute.** Each column holds exactly one kind of value,
  consistently typed: dates in the date column, numbers unmixed with text
  (no ``"12 approx"``).
- **Headers in row one, once.** Short, descriptive, unique names —
  ``order_date``, ``region``, ``amount`` — with no merged cells and no
  repeated header blocks mid-sheet. Headers are the context every later tool
  reads.
- **No layout-as-data.** Colour, bolding, and merged cells communicate to
  eyes but are invisible to formulas, sorts, and pivots. If a distinction
  matters, give it a **column** (``status``), not a highlight.
- **Raw stays raw.** Keep the untouched data on its own tab; do cleaning and
  analysis on copies or in adjacent tabs. When (not if) a step goes wrong,
  the original is still there.
- **A notes tab.** Where the data came from, when, and what was changed — the
  chain of custody, in its cheapest form.

Why these rules are the rules
-------------------------------

They all serve one principle: **the sheet should be readable by software, not
only by people**. Sorting breaks on merged cells; pivots break on repeated
headers; filters break on mixed types. The "one row per record, one column
per attribute" shape is exactly what every downstream tool — pivot tables,
SQL imports, pandas — expects, so a sheet built this way flows into the rest
of the toolkit unchanged. (Data folk call this *tidy* structure; the prep
section's wide-versus-long lesson formalises it.)

A worked contrast
-------------------

The monthly sales workbook, done badly: one tab per month, merged title rows,
totals typed between data rows, regions distinguished by cell colour. Done
well: a single ``sales`` tab — ``date, region, product, units, revenue`` —
one row per sale, plus a ``summary`` tab of formulas and a ``notes`` tab. The
first requires manual surgery for any cross-month question; the second
answers "revenue by region by quarter" with one pivot table.

The caveat
------------

These rules organise *analysis* sheets. Presentation sheets — the formatted
summary a stakeholder reads — legitimately use merged headers and colour,
because their reader is human. The discipline is keeping the two roles on
**separate tabs**: analyse in tidy structure, present in formatted views
built from it, and never let the presentation copy become the working data.

.. hint::

   - :doc:`Spreadsheets in Data Analysis <011-spreadsheets-in-data-analysis>`
   - :doc:`Spreadsheet Calculations with Formulas <014-spreadsheet-calculations-with-formulas>`
   - :doc:`Common Spreadsheet Errors and How to Fix Them <015-common-spreadsheet-errors-and-how-to-fix-them>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <../3_data_preparation/021-sorting-and-filtering-data-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/building-and-organizing-a-spreadsheet/ <https://insightful-data-lab.com/2023/08/31/building-and-organizing-a-spreadsheet/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: spreadsheets
