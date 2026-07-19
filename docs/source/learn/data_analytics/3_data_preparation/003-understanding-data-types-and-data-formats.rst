:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-003:
.. _da-3-prep-prep-003:

========================================================================
Understanding Data Types and Data Formats
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🧬 Data Types & Structure` :bdg-info:`Lesson 003`

◀ :doc:`Previous <002-choosing-the-right-data-to-collect>` · :doc:`Next <004-structured-data-and-data-models>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Two different questions
-------------------------

"What type is this data?" has two distinct meanings, and analysts need both.
One is about the **nature of the values** — are they numbers, categories,
dates? The other is about the **file format** the data travels in — CSV, JSON,
a database table. This lesson separates them, because the first governs what
analysis is *valid* and the second governs how you *load* the data at all.

Value types: the measurement levels
-------------------------------------

The nature of a value determines which operations make sense on it. The
standard classification:

- **Nominal** — named categories with no inherent order: country, product
  colour, payment method. You can count and group them, but "average colour"
  is meaningless.
- **Ordinal** — categories with a meaningful order but no fixed spacing:
  satisfaction ratings (poor/fair/good), t-shirt sizes. You can rank them, but
  the gap between "good" and "fair" is not a defined quantity.
- **Discrete** (quantitative) — countable numbers: orders placed, employees,
  defects. Whole units; you cannot have 2.5 orders.
- **Continuous** (quantitative) — measured numbers on a scale, any value in a
  range: revenue, temperature, duration. Arithmetic and averages are fully
  meaningful.

The practical payoff: the value type dictates the valid summary and chart. You
average continuous data, count nominal data, and never compute a mean of
category labels — the ``#VALUE!``-style mistakes of the spreadsheet lessons
often start as a value-type confusion.

Data formats: the containers
------------------------------

Independently, data arrives in formats — the file structures that hold it:

- **CSV / TSV** — plain text, one row per line, values separated by commas or
  tabs. Simple, universal, the lingua franca of tabular exchange.
- **JSON** — nested key–value structure, good for hierarchical data; the native
  shape of most web APIs.
- **Spreadsheet files** (``.xlsx``) — tabular data plus formatting and formulas.
- **Database tables** — structured, queryable storage (the SQL section's home).
- **XML, Parquet, and others** — further containers for specific needs.

The format determines *how you get the data in* — which tool and which step —
but not what the data *means*; a column of prices is continuous whether it
arrives as CSV or JSON.

Why keep them separate
------------------------

Confusing the two questions causes trouble. Loading a JSON file as if it were
CSV fails at the format level. Averaging a column of postal codes fails at the
value-type level — the load succeeded, the analysis is nonsense. Competent
preparation checks both: *can I read this container*, and *what may I validly
do with these values*.

The caveat
------------

Formats and types blur at the edges. A CSV stores everything as text, so a
column of numbers arrives *looking* like text and must be converted before
arithmetic — a frequent source of the type errors seen earlier. Part of the
Prepare phase is confirming that each column's *actual* type matches its
*intended* type, not merely its appearance. The next lesson goes deeper into
how data is structured overall.

.. hint::

   - :doc:`Choosing the Right Data to Collect <002-choosing-the-right-data-to-collect>`
   - :doc:`Structured Data and Data Models <004-structured-data-and-data-models>`
   - :doc:`Data Types in Spreadsheets <005-data-types-in-spreadsheets>`
   - :doc:`Data Tables (Tabular Data) <006-data-tables-tabular-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/understanding-data-types-and-data-formats/ <https://insightful-data-lab.com/2023/09/04/understanding-data-types-and-data-formats/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: types
