:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-020:
.. _da-5-analyze-analyze-020:

========================================================================
Using Spreadsheet Formulas for Sales Trend Analysis
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 020`

◀ :doc:`Previous <019-aggregating-data-with-subqueries-having-and-case-in-sql>` · :doc:`Next <021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Calculating change over time
------------------------------

The heart of analysis is *calculation* — turning organised data into the numbers
that answer a question — and one of the most common calculations is **trend
analysis**: how a value changes over time. Sales trend analysis is the classic
case, and the spreadsheet formulas that compute it — growth rates, running totals,
period-over-period comparisons — are foundational analytical tools, opening the
calculation stage.

The core trend calculations
-----------------------------

Trends are built from a few standard computations, each a formula pattern:

- **Period-over-period change** — the difference between consecutive periods:
  ``= this_period - last_period``. The absolute change.
- **Growth rate (percentage change)** — the change *relative* to the base:

  .. code-block:: text

     = (this_period - last_period) / last_period

  This expresses growth as a percentage (12% up, 5% down), which is comparable
  across different scales in a way absolute change is not.
- **Running (cumulative) total** — the sum accumulated up to each period, built
  with a mix of absolute and relative references so it extends down the column:

  .. code-block:: text

     = SUM($B$2:B2)     fill down: running total through each row

  The ``$B$2`` anchors the start (absolute) while ``B2`` moves (relative), so each
  row sums from the beginning through itself.
- **Comparison to a baseline** — each period against a fixed reference (a target,
  the same month last year), using an absolute reference to the baseline cell.

Reading the trend
------------------

The formulas produce numbers; *reading* them is the analysis. A growth rate turns
raw sales into a story — accelerating, steady, declining. A running total shows
progress toward a goal. A year-over-year comparison strips out seasonality that a
month-over-month view would confuse. The analyst's job is not just computing these
but interpreting what they reveal about the direction and health of what is
measured.

The absolute/relative reference connection
--------------------------------------------

Trend formulas lean heavily on the relative-versus-absolute reference distinction
from Section 2: the running total's ``SUM($B$2:B2)`` and the baseline comparison's
anchored reference both depend on getting the ``$`` right so the formula behaves
correctly when filled down. This is the earlier concept put to real analytical
work — a reminder that the foundational skills compound into the analytical ones.

The caveat
------------

Trend calculations can mislead in familiar ways. A percentage change from a *small*
base is volatile and can look dramatic while representing little (100% growth from 2
to 4 sales); a **short time span** can show a "trend" that is really noise (the
sufficiency lessons); and **seasonality** can masquerade as trend if not accounted
for (December sales are not a growth trend). The numbers compute regardless — the
judgement is whether the trend is *real*: enough data, an appropriate base, and
seasonality considered. Compute the trend, then ask whether it is signal or
artefact. The next lessons add conditional aggregation to the calculation toolkit.

.. hint::

   - :doc:`Spreadsheet Calculations with Formulas <../2_data_driven_decisions/014-spreadsheet-calculations-with-formulas>`
   - :doc:`Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets <021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
   - :doc:`Using Pivot Tables for Calculations and Trend Analysis <023-using-pivot-tables-for-calculations-and-trend-analysis>`
   - :doc:`Mathematical Thinking <../2_data_driven_decisions/010-mathematical-thinking>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-spreadsheet-formulas-for-sales-trend-analysis/ <https://insightful-data-lab.com/2023/11/26/using-spreadsheet-formulas-for-sales-trend-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
