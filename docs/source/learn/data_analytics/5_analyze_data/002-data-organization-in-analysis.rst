:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-002:
.. _da-5-analyze-analyze-002:

========================================================================
Data Organization in Analysis
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-understanding-data-analysis>` · :doc:`Next <003-sorting-and-filtering-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Arranging data to be analysed
-------------------------------

Analysis begins not with computation but with **organisation** — arranging the
data so that the patterns you seek can actually surface. Well-organised data makes
analysis fast and reliable; poorly organised data fights every operation. This
first step of analysis extends the tabular-structure and spreadsheet-organisation
disciplines from earlier into the service of finding answers.

What organising for analysis means
------------------------------------

Organising data for analysis includes several arranging activities:

- **Sorting** — ordering rows by a column to reveal structure (the next lessons'
  subject): extremes, rankings, chronology.
- **Filtering** — narrowing to the subset the question concerns, so the relevant
  data stands alone.
- **Grouping** — arranging data into the categories the analysis will summarise
  by (the aggregation lessons ahead build on this).
- **Structuring** — ensuring the data is in the right shape (tidy, correctly
  wide or long) for the analysis and tools you will use.

The goal is to get from "clean data sitting in a table" to "data arranged so the
question's answer is reachable" — the setup that makes the actual computation
straightforward.

Why organisation precedes computation
---------------------------------------

Jumping straight to calculation on unorganised data produces confusion or error:
an aggregate over unfiltered data answers the wrong question, a comparison across
inconsistently grouped data misleads, a trend sought in unsorted data stays
hidden. Organising first — sorting, filtering, grouping to match the question —
is what makes the subsequent computation both *possible* and *correct*. It is the
analysis-phase echo of the whole course's big-picture-first discipline: arrange
deliberately before you compute.

Organisation and the tools
----------------------------

Every tool in this section rewards good organisation. Spreadsheet sorting,
filtering, and pivot tables all assume the tidy structure from the prep section;
SQL's ``ORDER BY``, ``WHERE``, and ``GROUP BY`` are organisation expressed as
query. The organising *concepts* — order, subset, group, shape — are the same
across tools; only the syntax changes. Learning to think in these terms is
learning to see how to arrange any dataset toward any question, whichever tool you
reach for.

The caveat
------------

Organisation is preparation for analysis, not analysis itself — and it is possible
to over-organise, endlessly rearranging data without ever extracting the insight,
or to arrange it in a way that *presupposes* the answer (sorting and filtering
until only the data that supports a hunch remains). The discipline is to organise
in service of the *question*, not a desired conclusion, and to move on to the
actual analysis once the data is reachable. Organising is the means; the insight
is the end. The next lessons make the first organising move — sorting and
filtering — concrete.

.. hint::

   - :doc:`Understanding Data Analysis <001-understanding-data-analysis>`
   - :doc:`Sorting and Filtering in Data Analysis <003-sorting-and-filtering-in-data-analysis>`
   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/02/data-organization-in-analysis/ <https://insightful-data-lab.com/2023/11/02/data-organization-in-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
