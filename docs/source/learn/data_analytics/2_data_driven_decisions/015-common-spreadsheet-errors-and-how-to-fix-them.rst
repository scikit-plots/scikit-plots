:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-015:
.. _data-analytics-ddd-015:
.. _da-foundations-ddd-015:
.. _da-decisions-ddd-015:
.. _da-prep-ddd-015:
.. _da-cleaning-ddd-015:
.. _da-analyze-ddd-015:
.. _da-viz-ddd-015:
.. _da-python-ddd-015:
.. _da-jobsearch-ddd-015:

========================================================================
Common Spreadsheet Errors and How to Fix Them
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📗 Spreadsheets for Analysis` :bdg-info:`Lesson 015`

◀ :doc:`Previous <014-spreadsheet-calculations-with-formulas>` · :doc:`Next <016-spreadsheet-functions>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Errors are messages, not failures
-----------------------------------

When a spreadsheet shows ``#DIV/0!`` or ``#REF!``, it is not broken — it is
**telling you something specific**. Each error code names a distinct cause, and
reading the code is the fastest route to the fix. This lesson decodes the ones
you will actually meet, then covers the errors that show *no* code at all —
which are the dangerous kind.

The coded errors, decoded
---------------------------

- ``#DIV/0!`` — **division by zero**. A formula divided by an empty or
  zero-valued cell (common in averages when the count is zero). Fix: ensure the
  denominator is non-zero, or handle the empty case explicitly.
- ``#VALUE!`` — **wrong data type**. A formula expected a number but got text —
  ``="ten"+5``, or a numeric column contaminated with ``"12 approx"``. Fix:
  correct the offending value's type.
- ``#REF!`` — **invalid reference**. The formula points at a cell that no
  longer exists, usually because a row or column it referenced was deleted, or
  a relative formula was copied off the edge of the data. Fix: restore the
  reference or repair the formula (absolute references prevent the copy
  variant).
- ``#NAME?`` — **unrecognised name**. A misspelled function or range —
  ``=SUMM(...)`` instead of ``=SUM(...)``. Fix: correct the spelling; the
  editor's autocomplete prevents most of these.
- ``#N/A`` — **not available**. A lookup (``VLOOKUP`` and friends) could not
  find its target. Often legitimate — the value genuinely is not there — and
  often caused by stray spaces or a mismatched type. Fix: confirm the lookup
  value exists and matches exactly.

Suppress with care
--------------------

Functions like ``IFERROR`` can replace an error with a blank or a message —
``=IFERROR(B2/C2, 0)``. This is right for errors you *expect and understand*
(an average over a genuinely empty group). It is dangerous as a reflex:
wrapping everything in ``IFERROR`` **hides** structural problems — a ``#REF!``
that signals broken logic gets silently turned into a zero, and the wrong
number propagates. Diagnose first; suppress only what you have understood.

The errors with no code
-------------------------

The coded errors are the *safe* ones — the spreadsheet flagged them. The
genuinely dangerous errors show a perfectly normal-looking number:

- a **relative reference that drifted** on fill-down (previous lesson),
  computing against the wrong cells;
- a **wrong range** — ``=SUM(B2:B99)`` when data runs to B100, silently
  dropping the last row;
- a **duplicate or off-by-one** in a manual selection;
- **mixed units** summed together (dollars and euros, hours and minutes).

None raises an error. All produce confident, wrong output — which is why the
order-of-magnitude sanity check and, later, systematic **verification** exist:
the spreadsheet cannot warn you about the mistakes it cannot see.

The caveat
------------

A visible error code is a gift — it points at its own cause. Train yourself to
find silent errors *before* trusting a number: does the total match a rough
estimate, does the row count look right, do spot-checked cells recompute by
hand? The coded errors teach the vocabulary; the discipline is hunting the
uncoded ones.

.. hint::

   - :doc:`Spreadsheet Calculations with Formulas <014-spreadsheet-calculations-with-formulas>`
   - :doc:`Spreadsheet Functions <016-spreadsheet-functions>`
   - :doc:`Data Cleaning with Spreadsheets <../4_data_cleaning_preparation/012-data-cleaning-with-spreadsheets>`
   - :doc:`How Data Analysts Use Spreadsheets <013-how-data-analysts-use-spreadsheets>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/common-spreadsheet-errors-and-how-to-fix-them/ <https://insightful-data-lab.com/2023/08/31/common-spreadsheet-errors-and-how-to-fix-them/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: spreadsheets
