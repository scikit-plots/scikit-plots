:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-analyze-010:
.. _data-analytics-analyze-010:
.. _da-foundations-analyze-010:
.. _da-decisions-analyze-010:
.. _da-prep-analyze-010:
.. _da-cleaning-analyze-010:
.. _da-analyze-analyze-010:
.. _da-viz-analyze-010:
.. _da-python-analyze-010:
.. _da-jobsearch-analyze-010:

========================================================================
Working with Strings in Spreadsheets (LEN, LEFT, RIGHT, FIND)
========================================================================

:bdg-primary:`📊 Analyze Data` :bdg-secondary:`🗂️ Organizing & Formatting Data` :bdg-info:`Lesson 010`

◀ :doc:`Previous <009-using-concat-in-sql-to-combine-text-from-multiple-columns>` · :doc:`Next <011-problem-solving-and-seeking-help-in-data-analysis>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The spreadsheet string toolkit
--------------------------------

Text data — names, codes, addresses, categories — needs its own set of operations
to measure, extract, and locate parts of it, and spreadsheets provide a compact
**string function** toolkit for exactly this. Building on the cleaning functions
from earlier, these functions let you take apart, inspect, and manipulate text as
precisely as you manipulate numbers.

The core string functions
----------------------------

- ``LEN(text)`` — returns the **length** (number of characters) of a string.
  Useful for validation (is this code the right length?) and for calculations
  with other string functions.
- ``LEFT(text, n)`` — extracts the **first n characters** from the left
  (``LEFT("Product-123", 7)`` → ``"Product"``).
- ``RIGHT(text, n)`` — extracts the **last n characters** from the right
  (``RIGHT("Product-123", 3)`` → ``"123"``).
- ``MID(text, start, n)`` — extracts **n characters starting at a position**, for
  pulling from the middle.
- ``FIND(substring, text)`` — returns the **position** where a substring first
  appears (``FIND("-", "Product-123")`` → ``8``), or an error if not found.
  (``SEARCH`` is the case-insensitive variant.)

.. code-block:: text

   =LEN(A2)                    length of the text
   =LEFT(A2, 3)                first 3 characters
   =RIGHT(A2, 4)               last 4 characters
   =MID(A2, 5, 2)              2 characters starting at position 5
   =FIND("-", A2)              position of the first hyphen

Combining them to extract
---------------------------

The functions' real power comes from *combining* them to extract variable-length
parts. To pull everything before a hyphen when the hyphen's position varies, nest
``LEFT`` and ``FIND``:

.. code-block:: text

   =LEFT(A2, FIND("-", A2) - 1)    everything before the first hyphen

``FIND`` locates the hyphen, and ``LEFT`` takes everything up to just before it —
so ``"Product-123"`` yields ``"Product"`` and ``"Widget-45"`` yields ``"Widget"``,
each correctly, despite the different lengths. This ``LEFT``/``RIGHT``/``MID`` with
``FIND`` pattern is how you split text on a delimiter by formula, the flexible
counterpart to the fixed-position extractions.

Why string functions matter for analysis
-------------------------------------------

Text manipulation is constant analytical work: extracting a category code from a
product ID, pulling a domain from an email, splitting a combined field into its
parts for grouping, validating that codes have the right format. These functions
are the tools for all of it, and they mirror the SQL string functions (``SUBSTR``,
``LENGTH``, ``POSITION``) exactly — the same operations in a different syntax, so
learning them in the visible spreadsheet builds intuition that transfers to SQL.

The caveat
------------

String functions are precise about positions and counts, which makes them brittle
if the text's structure varies unexpectedly: ``LEFT(A2, 7)`` assumes the part you
want is always seven characters, and breaks when it is not; ``FIND`` returns an
error when the substring is absent, which propagates through a nested formula. The
robust approach uses *position-finding* (``FIND``) rather than fixed counts where
lengths vary, and wraps extractions in ``IFERROR`` to handle the cases where the
expected structure is missing — so a few irregular rows do not break the whole
column. Text is messier than it looks; extract defensively. This closes the
organise stage; the next lessons turn to the analyst's problem-solving craft as the
combine stage opens.

.. hint::

   - :doc:`Using CONCAT in SQL to Combine Text from Multiple Columns <009-using-concat-in-sql-to-combine-text-from-multiple-columns>`
   - :doc:`Spreadsheet Functions <../2_data_driven_decisions/016-spreadsheet-functions>`
   - :doc:`Using Spreadsheet Functions for Data Cleaning <../4_data_cleaning_preparation/015-using-spreadsheet-functions-for-data-cleaning>`
   - :doc:`Data Validation in Spreadsheets <007-data-validation-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/11/26/working-with-strings-in-spreadsheets-len-left-right-find/ <https://insightful-data-lab.com/2023/11/26/working-with-strings-in-spreadsheets-len-left-right-find/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: analyze, topic: organize
