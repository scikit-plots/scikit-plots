:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-012:
.. _da-7-python-python-012:

========================================================================
Branching and Conditional Statements in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🔀 Control Flow` :bdg-info:`Lesson 012`

◀ :doc:`Previous <011-boolean-data-comparators-and-logical-operators-in-python>` · :doc:`Next <013-while-loops-and-iteration-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Making code decide
--------------------

With boolean conditions in hand, a program can **branch** — run different code depending
on whether a condition is true. **Conditional statements** (``if``, ``elif``, ``else``)
are how Python decides, directing which instructions execute. This lesson covers
branching, the first form of control flow.

The if statement
-----------------

An ``if`` statement runs a block of code *only if* a condition is true:

.. code-block:: python

   if sales > 1000:
       print("High sales")

The condition (``sales > 1000``) is a boolean; if it is ``True``, the indented block
runs; if ``False``, it is skipped. The indentation defines the block — the code that
belongs to the ``if`` — which is why Python's meaningful indentation matters here
especially.

else and elif
--------------

``else`` provides an alternative when the condition is false, and ``elif`` (else-if)
checks further conditions:

.. code-block:: python

   if sales > 1000:
       category = "High"
   elif sales > 500:
       category = "Medium"
   else:
       category = "Low"

Python checks each condition in order: if ``sales > 1000``, category is "High" and the
rest is skipped; otherwise if ``sales > 500``, "Medium"; otherwise "Low". Only the *first*
matching branch runs. This ``if``/``elif``/``else`` chain is the direct counterpart of
SQL's ``CASE`` expression (the analysis section) — the same "different result for
different conditions" logic, now controlling which code executes.

Branching on compound conditions
----------------------------------

Branches use the full boolean logic from the previous lesson:

.. code-block:: python

   if (age >= 18) and (region in ("N", "S")):
       status = "eligible"
   else:
       status = "ineligible"

Compound conditions let a branch depend on several factors at once — the logical operators
combining into a single decision. This is how programs express real decision rules, which
usually involve multiple criteria.

Why branching matters
-----------------------

Branching is what makes a program *responsive* rather than fixed — it does different
things in different situations, which is essential to any non-trivial logic. In data work,
branching drives categorisation (bucketing values, exactly the ``CASE`` work), validation
(handling valid versus invalid data differently), and conditional processing throughout.
It is one of the two pillars of control flow (with loops, next), and the foundation of
code that adapts to its data.

The caveat
------------

Branching logic is precise and can be subtly wrong. The order of ``elif`` conditions
*matters* — since only the first match runs, mis-ordered conditions can make a branch
unreachable (checking ``sales > 500`` before ``sales > 1000`` would catch high sales in
the wrong branch); conditions must be *exhaustive* where every case should be handled (an
``else`` catching what the explicit conditions miss); and deeply *nested* branches (ifs
within ifs within ifs) grow hard to follow and are often better restructured. Test that
each branch runs exactly when intended, especially the boundaries between conditions — the
same edge-case discipline from the foundations, applied to decisions. The next lessons
cover the other pillar of control flow: loops.

.. hint::

   - :doc:`Boolean Data, Comparators, and Logical Operators in Python <011-boolean-data-comparators-and-logical-operators-in-python>`
   - :doc:`While Loops and Iteration in Python <013-while-loops-and-iteration-in-python>`
   - :doc:`For Loops in Python <014-for-loops-in-python>`
   - :doc:`Functions in Python <008-functions-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/branching-and-conditional-statements-in-python/ <https://insightful-data-lab.com/2023/12/06/branching-and-conditional-statements-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: control
