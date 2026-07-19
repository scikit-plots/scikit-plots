:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-ddd-014:

========================================================================
Spreadsheet Calculations with Formulas
========================================================================

:bdg-primary:`🎯 Data-Driven Decisions` :bdg-secondary:`📗 Spreadsheets for Analysis` :bdg-info:`Lesson 014`

◀ :doc:`Previous <013-how-data-analysts-use-spreadsheets>` · :doc:`Next <015-common-spreadsheet-errors-and-how-to-fix-them>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Formulas: computation you can read
------------------------------------

A **formula** is an expression, beginning with ``=``, that computes a value
from other cells. Formulas are what make a spreadsheet a calculator rather than
a table: change an input and every dependent formula recomputes instantly. This
lesson covers the mechanics that trip up beginners — chiefly how cell
references behave when a formula is copied.

The anatomy of a formula
--------------------------

``=B2*C2`` multiplies two cells. ``=(B2-C2)/C2`` computes a percentage change.
Formulas combine **cell references** (``B2``), **operators**
(``+ - * /``), **numbers**, and **functions** (next lesson). The power is the
reference: ``=B2*C2`` does not mean "6" — it means "whatever is in B2 times
whatever is in C2", so the sheet stays live.

The distinction that matters most: relative vs absolute
---------------------------------------------------------

When you copy a formula down a column, its references **move** by default —
this is a **relative reference**. Copy ``=B2*C2`` from row 2 to row 3 and it
becomes ``=B3*C3``, which is usually exactly what you want: revenue per row,
computed for every row by writing the formula once and filling down.

Sometimes a reference must **not** move — a single tax rate in ``E1`` that
every row multiplies by. Freeze it with **dollar signs**: ``$E$1`` is an
**absolute reference** that stays fixed no matter where the formula is copied.
The ``$`` before the column letter locks the column; the ``$`` before the row
number locks the row; you can lock one and not the other (``$E1`` or ``E$1``)
for row- or column-only anchoring.

.. code-block:: text

   A2:  =B2*$E$1     -> fill down -> B3*$E$1, B4*$E$1, ...   (rate stays E1)
   A2:  =B2*E1       -> fill down -> B3*E2,   B4*E3,  ...    (rate drifts -- a bug)

Getting this wrong is one of the most common spreadsheet bugs: a fill-down that
silently walks the "fixed" reference down the sheet, producing numbers that
look plausible and are wrong.

Building calculations that hold up
------------------------------------

Two habits prevent trouble. **Reference, don't retype**: put the tax rate in a
cell and reference ``$E$1``, so changing it updates everything — a value typed
into fifty formulas is fifty places to miss. And **build in steps**: a column
for subtotal, another for tax, another for total, rather than one monster
formula — each step is inspectable, and the order-of-magnitude sanity check
from the mathematical-thinking lesson has somewhere to land.

The caveat
------------

Formulas are invisible once entered — the cell shows the result, not the logic
— so errors hide behind reasonable-looking numbers. That invisibility is why
the next lesson is entirely about *errors*: recognising when a formula has gone
wrong, and why.

.. hint::

   - :doc:`How Data Analysts Use Spreadsheets <013-how-data-analysts-use-spreadsheets>`
   - :doc:`Spreadsheet Functions <016-spreadsheet-functions>`
   - :doc:`Common Spreadsheet Errors and How to Fix Them <015-common-spreadsheet-errors-and-how-to-fix-them>`
   - :doc:`Building and Organizing a Spreadsheet <012-building-and-organizing-a-spreadsheet>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/08/31/spreadsheet-calculations-with-formulas/ <https://insightful-data-lab.com/2023/08/31/spreadsheet-calculations-with-formulas/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: ddd, topic: spreadsheets
