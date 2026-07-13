:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-005:
.. _data-analytics-prep-005:
.. _da-foundations-prep-005:
.. _da-decisions-prep-005:
.. _da-prep-prep-005:
.. _da-cleaning-prep-005:
.. _da-analyze-prep-005:
.. _da-viz-prep-005:
.. _da-python-prep-005:
.. _da-jobsearch-prep-005:

========================================================================
Data Types in Spreadsheets
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🧬 Data Types & Structure` :bdg-info:`Lesson 005`

◀ :doc:`Previous <004-structured-data-and-data-models>` · :doc:`Next <006-data-tables-tabular-data>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The type behind the cell
--------------------------

The spreadsheet lessons of Section 2 treated cells as holding "values"; here we
get specific about what *kind* of value a cell holds, because a spreadsheet
quietly assigns every cell a **data type**, and that type governs how the cell
behaves — what you can compute, how it sorts, and whether a formula works or
throws an error.

The core spreadsheet types
----------------------------

- **Text** (string) — letters, words, or any characters treated as label rather
  than quantity: names, categories, IDs. Text is left-aligned by default and
  cannot be summed. Numbers *stored as text* (a common import problem) look
  numeric but refuse arithmetic — a frequent cause of the ``#VALUE!`` errors
  from earlier.
- **Number** — numeric values you can calculate with: integers and decimals,
  right-aligned by default. Currency and percentage are number *formats* — the
  underlying value is a number, displayed with a symbol or scaled by 100.
- **Date and time** — stored internally as serial numbers so they can be
  subtracted and sorted chronologically, but displayed as calendar dates. This
  dual nature is why a "date" that is really text will sort alphabetically
  (wrongly) and refuse date arithmetic.
- **Boolean** — logical ``TRUE`` / ``FALSE`` values, produced by comparisons
  and consumed by ``IF`` and filters.

Why the type matters
----------------------

The type determines the valid operation, exactly as the measurement levels did:
you sum numbers, sort dates chronologically, and count text. The everyday
failures trace to a mismatch between a cell's *actual* type and its *apparent*
one — the classic being numbers or dates imported as text, which look right and
compute wrong. Recognising and fixing these is core Prepare-phase work, and it
connects directly to the value types (nominal, ordinal, discrete, continuous)
of the previous lessons: the spreadsheet type is how those abstract levels are
physically stored.

Checking and fixing types
---------------------------

Two habits catch most trouble. **Watch the alignment**: numbers and dates that
sit left-aligned are secretly text and will misbehave. **Convert deliberately**:
use the spreadsheet's type-conversion tools (or a helper column) to turn
text-numbers into real numbers and text-dates into real dates *before*
computing, rather than discovering the problem inside a broken formula. The
cleaning section builds these fixes into a systematic workflow.

The caveat
------------

Spreadsheets guess types automatically, and the guess is sometimes wrong —
famously, identifiers that look like numbers (a product code ``00123``) lose
leading zeros, and codes that look like dates get silently converted. Automatic
typing is a convenience that occasionally corrupts data on import, so part of
preparation is verifying that each column's type is the one you *intended*, not
merely the one the software chose. The next lesson steps up from individual
cells to the table they form.

.. hint::

   - :doc:`Understanding Data Types and Data Formats <003-understanding-data-types-and-data-formats>`
   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Data Tables (Tabular Data) <006-data-tables-tabular-data>`
   - :doc:`Data Formatting and Unit Conversion in Spreadsheets <../5_analyze_data/006-data-formatting-and-unit-conversion-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/data-types-in-spreadsheets/ <https://insightful-data-lab.com/2023/09/04/data-types-in-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: types
