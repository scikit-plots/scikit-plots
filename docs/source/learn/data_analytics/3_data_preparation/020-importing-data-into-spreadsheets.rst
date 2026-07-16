:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-prep-020:
.. _da-3-prep-prep-020:

========================================================================
Importing Data into Spreadsheets
========================================================================

:bdg-primary:`📦 Data Preparation` :bdg-secondary:`🔢 Spreadsheets, SQL & Organization` :bdg-info:`Lesson 020`

◀ :doc:`Previous <019-accessing-data-internal-and-external-sources>` · :doc:`Next <021-sorting-and-filtering-data-in-spreadsheets>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


From source to sheet
----------------------

The most common first step of a hands-on analysis is getting data *into* a
spreadsheet — importing it from a file, a database export, or an external
source. Done carelessly, import is where a surprising share of data problems are
born; done deliberately, it sets up everything after. This lesson is the
practical bridge from "data exists somewhere" to "data is in my sheet, correctly
typed and ready."

Common import routes
----------------------

- **CSV / TSV files** — the universal exchange format from earlier; spreadsheets
  open them directly, parsing rows and delimiter-separated columns.
- **Other spreadsheet files** — opening or importing an existing ``.xlsx``.
- **Database exports** — data pulled from a database (often *as* CSV) and loaded
  in.
- **Copy and paste** — quick for small data, but the most error-prone route for
  anything structured.
- **Connected imports** — some spreadsheets can pull directly from a URL, an
  API, or a database connection, refreshing as the source updates.

The import traps
------------------

Import is where the *type* problems from earlier lessons are created, so it is
where to catch them:

- **Numbers as text** — a numeric column arrives as text (common from CSV,
  where everything is text) and refuses arithmetic until converted. The
  left-alignment tell from the data-types lesson is your early warning.
- **Mangled dates** — dates parse into the wrong format or get silently
  converted, especially across regional day/month conventions.
- **Lost leading zeros** — identifier codes like ``00042`` lose their zeros when
  auto-typed as numbers, corrupting keys.
- **Delimiter and encoding issues** — a comma inside a text field splits a
  column wrongly; non-UTF-8 characters arrive garbled.
- **Header confusion** — the header row imported as data, or missing entirely.

Importing cleanly
-------------------

Three habits prevent most trouble. **Check types immediately** after import —
scan for left-aligned numbers, malformed dates, and dropped leading zeros before
doing anything else. **Control the import** rather than accepting defaults —
spreadsheets' import dialogs let you specify delimiters and column types up
front, which is far easier than fixing corruption afterward. And **keep the raw
import untouched** — paste it to its own tab and work on copies, the
raw-stays-raw rule from the spreadsheet-organisation lesson, so a botched
transformation never destroys the original.

The caveat
------------

A clean-looking import is not a verified one: data can import without error and
still be subtly wrong — a shifted column, a truncated field, an encoding that
corrupted a few characters. Import is the moment to apply the sanity checks from
the mathematical-thinking lesson — does the row count match the source, do
totals look right, do spot-checked values match the origin? Getting data into
the sheet is the *start* of trusting it, not the end. The next lessons, in the
analysis section proper, turn that imported data into answers — beginning with
sorting and filtering.

.. hint::

   - :doc:`Accessing Data: Internal and External Sources <019-accessing-data-internal-and-external-sources>`
   - :doc:`Data Types in Spreadsheets <005-data-types-in-spreadsheets>`
   - :doc:`Building and Organizing a Spreadsheet <../2_data_driven_decisions/012-building-and-organizing-a-spreadsheet>`
   - :doc:`Sorting and Filtering Data in Spreadsheets <021-sorting-and-filtering-data-in-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/09/04/importing-data-into-spreadsheets/ <https://insightful-data-lab.com/2023/09/04/importing-data-into-spreadsheets/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: prep, topic: spreadsheets_sql
