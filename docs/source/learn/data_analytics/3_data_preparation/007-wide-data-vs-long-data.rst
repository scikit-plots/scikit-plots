:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-007:
.. _da-3-prep-prep-007:

========================================================================
Wide Data vs. Long Data
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🧬 Data Types & Structure` :bdg-info:`Lesson 007`

◀ :doc:`Previous <006-data-tables-tabular-data>` · :doc:`Next <008-understanding-bias-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Two shapes for the same facts
-------------------------------

Well-formed tabular data still comes in two distinct layouts, and knowing the
difference — and how to move between them — is a genuinely useful preparation
skill. The same information can be arranged as **wide** data or **long** data;
neither is universally correct, and different tools and tasks prefer different
shapes.

Wide data
-----------

In **wide** format, each subject occupies a single row, and each measured
variable (or each time point) gets its own column. Sales by quarter, wide:

.. code-block:: text

   region    Q1     Q2     Q3     Q4
   North    120    135    128    150
   South     98    102    115    120

One row per region; the four quarters spread across columns. Wide data is
**compact and human-readable** — easy to scan, and the natural shape for a
report or a spreadsheet a person will read.

Long data
-----------

In **long** format, each row is a single observation, with columns identifying
*what* is measured and *its value*. The same sales data, long:

.. code-block:: text

   region   quarter   sales
   North    Q1        120
   North    Q2        135
   North    Q3        128
   ...
   South    Q4        120

Every region–quarter combination is its own row. Long data is more repetitive
to read but far more **flexible for analysis**: adding a new measure means
adding rows, not restructuring columns, and most plotting and statistical
tools expect this "one observation per row" shape (it is the *tidy* form the
tabular lesson gestured at).

Choosing and converting
-------------------------

The rule of thumb: **wide for reading, long for analysis**. A stakeholder-facing
summary is usually wide; the data feeding a chart, a pivot, or a statistical
routine is usually long. Analysts convert between them constantly — reshaping
wide to long before analysis, and often pivoting long back to wide for
presentation. Spreadsheets do this with pivot tables; the Python section does it
with pandas' reshape operations. The key realisation is that **the choice is
about convenience, not correctness** — the underlying facts are identical, only
their arrangement differs.

A worked reason it matters
----------------------------

Suppose you want a line chart of sales over quarters, one line per region. From
*long* data, the chart is a direct mapping: quarter on the x-axis, sales on the
y-axis, region as the colour — one instruction. From *wide* data, you must
first tell the tool that four separate columns are really one variable measured
four times. Same data, but the long shape matched the task and the wide shape
fought it. Recognising which shape a task wants saves exactly this friction.

The caveat
------------

Real datasets are often *neither* cleanly wide nor cleanly long — they are
half-reshaped, with some variables in columns and some in rows, or time points
mixed with attributes. Part of preparation is recognising the current shape and
deciding the target shape the analysis needs, then converting deliberately.
This closes the data-types stage; the next turns to the quality and fairness of
data, beginning with the bias that unexamined data carries.

.. hint::

   - :doc:`Data Tables (Tabular Data) <006-data-tables-tabular-data>`
   - :doc:`Structured Data and Data Models <004-structured-data-and-data-models>`
   - :doc:`Understanding Data Types and Data Formats <003-understanding-data-types-and-data-formats>`
   - :doc:`Data Types in Spreadsheets <005-data-types-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/wide-data-vs-long-data/ <https://insightful-data-lab.com/2023/09/04/wide-data-vs-long-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: types
