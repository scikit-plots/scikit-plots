:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-011:
.. _da-7-python-python-011:

========================================================================
Boolean Data, Comparators, and Logical Operators in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🔀 Control Flow` :bdg-info:`Lesson 011`

◀ :doc:`Previous <010-comments-algorithms-and-docstrings-in-python>` · :doc:`Next <012-branching-and-conditional-statements-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


True, false, and decisions
----------------------------

Programs make *decisions*, and every decision rests on a question with a true-or-false
answer. **Boolean** values (``True`` and ``False``), the **comparators** that produce
them, and the **logical operators** that combine them are the foundation of all control
flow. Opening the control stage, this lesson covers the boolean logic that the branching
and looping lessons build on.

Boolean values
---------------

A **boolean** is one of exactly two values, ``True`` or ``False`` — the ``bool`` type
from the types lesson. Booleans represent the answer to a yes/no question, and they are
what decisions are made from:

.. code-block:: python

   is_valid = True
   has_errors = False

Comparators
-----------

**Comparison operators** compare two values and produce a boolean:

.. code-block:: python

   5 > 3           # True   (greater than)
   5 < 3           # False  (less than)
   5 == 5          # True   (equal to — note double equals)
   5 != 3          # True   (not equal to)
   5 >= 5          # True   (greater than or equal)
   3 <= 5          # True   (less than or equal)

The critical one to note is ``==`` (equality comparison), *two* equals signs — distinct
from ``=`` (assignment), *one* equals sign. Confusing them is a classic error: ``=``
assigns a value, ``==`` asks whether two values are equal. Comparators are how a program
turns data into the true/false answers decisions need.

Logical operators
------------------

**Logical operators** combine booleans into compound conditions:

.. code-block:: python

   (age >= 18) and (age < 65)      # True only if BOTH are true
   (region == "N") or (region == "S")   # True if EITHER is true
   not is_valid                    # inverts: True becomes False

- ``and`` — true only if *both* operands are true.
- ``or`` — true if *at least one* operand is true.
- ``not`` — inverts a boolean.

These are the same logical combinations as SQL's ``AND``/``OR``/``NOT`` in ``WHERE``
clauses (the analysis section) — the identical logic, now in Python. Compound conditions
let a program ask complex questions ("is the customer an adult *and* in an eligible
region?") as a single boolean.

Why boolean logic matters
---------------------------

Boolean logic is the foundation of *control flow* — every branch and loop the next
lessons cover is directed by a boolean condition. It is also the basis of *filtering*
data (the pandas lessons will filter rows by boolean conditions, exactly as SQL's
``WHERE`` and spreadsheet filters did). Mastering comparators and logical operators is
therefore mastering the mechanism behind decisions, loops, and data filtering alike — a
small piece of logic that underlies a large share of programming.

The caveat
------------

Boolean logic has precise rules that produce surprises when misread. The ``=`` versus
``==`` confusion is the commonest (and Python catches many but not all such mistakes);
operator precedence means compound conditions sometimes need *parentheses* to group them
as intended (``a and b or c`` may not mean what you expect — parenthesise for clarity);
and comparisons involving different types or ``None`` can behave unexpectedly. Writing
compound conditions with explicit parentheses, and testing that a condition is true
exactly when it should be, guards against logic that looks right but is not — the
check-your-logic discipline applied to booleans. The next lesson uses these conditions to
make code branch.

.. hint::

   - :doc:`Branching and Conditional Statements in Python <012-branching-and-conditional-statements-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Data Validation in Spreadsheets <../5_analyze_data/007-data-validation-in-spreadsheets>`
   - :doc:`While Loops and Iteration in Python <013-while-loops-and-iteration-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/boolean-data-comparators-and-logical-operators-in-python/ <https://insightful-data-lab.com/2023/12/06/boolean-data-comparators-and-logical-operators-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: control
