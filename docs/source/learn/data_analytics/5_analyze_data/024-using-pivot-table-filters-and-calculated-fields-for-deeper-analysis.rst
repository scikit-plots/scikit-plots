:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-024:
.. _da-5-analyze-analyze-024:

========================================================================
Using Pivot Table Filters and Calculated Fields for Deeper Analysis
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🧮 Calculations & Aggregation` :bdg-info:`Lesson 024`

◀ :doc:`Previous <023-using-pivot-tables-for-calculations-and-trend-analysis>` · :doc:`Next <025-comparing-calculations-in-spreadsheets-and-sql>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Beyond the basic pivot
------------------------

A basic pivot table summarises; two features push it toward *deeper* analysis:
**filters** (restricting what the pivot includes) and **calculated fields** (adding
computed values the source data does not contain). Together they turn the pivot
from a summarising tool into a flexible analytical instrument, extending the
previous lesson's foundation.

Pivot table filters
----------------------

Filters restrict which data a pivot summarises, letting you focus the summary
without changing the underlying data:

- **Report filters** — restrict the whole pivot to a subset (show the summary for
  one region, one year, one product line), with a control to switch the subset.
- **Slicers** — visual filter buttons that make filtering interactive and obvious,
  and can filter multiple pivots at once.
- **Row/column filters** — hide specific categories within the pivot.

Filtering a pivot answers "what does this summary look like *for this segment*" —
and switching the filter re-answers it instantly for another segment, enabling
rapid comparison across subsets.

Calculated fields
-------------------

A **calculated field** adds a new value to the pivot computed from existing fields
— a metric the source data does not store directly:

- **Ratios and rates** — profit margin from revenue and cost, conversion rate from
  conversions and visits.
- **Derived amounts** — a computed total, a per-unit value, a difference.
- **Custom metrics** — any formula combining the pivot's aggregated values.

For example, a pivot summing revenue and cost by region can gain a calculated field
``profit = revenue - cost``, or ``margin = (revenue - cost) / revenue`` — computing
the derived metric *per group* automatically. This lets the pivot present not just
raw aggregates but the *analytical* values a question actually needs.

Why these deepen analysis
---------------------------

Basic pivots answer "sum X by Y"; filters and calculated fields answer "the derived
metric M, for segment S, by Y" — a substantially richer question. They let an
analyst move from *reporting* aggregates to *analysing* them: computing the rates
and ratios that reveal what raw totals hide, and slicing to the segments where the
story lives. This is the pivot table doing genuine analytical work, not just
tabulation.

The caveat
------------

The added power adds ways to mislead. A **filter left active** shows a partial
summary that is easily mistaken for the whole — the same forgotten-filter risk as
elsewhere, now hiding in a pivot that looks complete. **Calculated fields on
aggregates** can compute subtly wrong values: a "margin" calculated as the average
of per-row margins differs from the margin of the summed totals, and the pivot's
field computes one when you may mean the other. And filters plus calculated fields
make a pivot complex enough that its logic is no longer obvious at a glance. Verify
what a filtered, calculated pivot actually computes — especially that ratios are
computed on the totals you intend — before trusting it. The next lesson steps back
to compare spreadsheet and SQL calculation directly.

.. hint::

   - :doc:`Using Pivot Tables for Calculations and Trend Analysis <023-using-pivot-tables-for-calculations-and-trend-analysis>`
   - :doc:`Comparing Calculations in Spreadsheets and SQL <025-comparing-calculations-in-spreadsheets-and-sql>`
   - :doc:`Using COUNTIF and SUMIF for Conditional Aggregation in Spreadsheets <021-using-countif-and-sumif-for-conditional-aggregation-in-spreadsheets>`
   - :doc:`Using SUMPRODUCT for Advanced Spreadsheet Calculations <022-using-sumproduct-for-advanced-spreadsheet-calculations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/using-pivot-table-filters-and-calculated-fields-for-deeper-analysis/ <https://insightful-data-lab.com/2023/11/26/using-pivot-table-filters-and-calculated-fields-for-deeper-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: calc
