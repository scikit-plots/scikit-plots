:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-015:
.. _da-7-python-python-015:

========================================================================
range() Function and Loop Control in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🔀 Control Flow` :bdg-info:`Lesson 015`

◀ :doc:`Previous <014-for-loops-in-python>` · :doc:`Next <016-strings-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Sequences and finer loop control
----------------------------------

Two tools refine looping: the **range()** function, which generates sequences of numbers
to loop over, and **loop control** statements (``break`` and ``continue``), which give
finer control over how a loop proceeds. This lesson covers both, completing the control
stage's treatment of iteration.

The range() function
---------------------

``range()`` generates a sequence of numbers, most often used to repeat something a set
number of times or to loop over numeric indices:

.. code-block:: python

   for i in range(5):         # 0, 1, 2, 3, 4
       print(i)

   for i in range(1, 6):      # 1, 2, 3, 4, 5 (start, stop)
       print(i)

   for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (start, stop, step)
       print(i)

``range(n)`` produces 0 up to (but *not including*) ``n``; ``range(start, stop)`` starts
elsewhere; ``range(start, stop, step)`` steps by an interval. The *exclusive* upper bound
— ``range(5)`` stops at 4, not 5 — is the detail to remember, echoing zero-based indexing.
``range()`` is how a for loop repeats a fixed number of times or walks a numeric sequence.

Loop control: break and continue
----------------------------------

Two statements alter a loop's normal flow:

.. code-block:: python

   for amount in sales:
       if amount < 0:
           continue           # skip this iteration, go to the next
       if amount > 10000:
           break              # exit the loop entirely
       process(amount)

- ``break`` — *exits* the loop immediately, skipping any remaining iterations. Useful for
  stopping once a condition is met (found what you sought, hit a limit).
- ``continue`` — *skips the rest of the current iteration* and moves to the next. Useful
  for skipping items that should not be processed (invalid values, ones to ignore).

These give a loop finer control than "process every item to the end" — stop early, or skip
selectively.

Why these matter
-----------------

``range()`` and loop control complete the looping toolkit. ``range()`` handles
count-based and index-based repetition (repeat *n* times, iterate positions), complementing
the for loop's collection iteration. ``break`` and ``continue`` let a loop respond to
conditions — stopping when done, skipping what should be skipped — making loops precise
rather than all-or-nothing. Together with the while and for loops, they cover the iteration
patterns programming requires.

The caveat
------------

Each tool has a trap. ``range()``'s *exclusive upper bound* is a persistent
off-by-one source — ``range(1, 5)`` gives 1,2,3,4, not 1–5 — so getting the bounds right
requires care (the edge-case discipline). ``break`` and ``continue`` used heavily can make
a loop's flow *hard to follow* — a loop with several breaks and continues scattered through
it can be as tangled as deeply nested branches, so use them where they clarify (a clean
early exit) rather than as a substitute for well-structured loop logic. And the earlier
caution stands: for large data, ``range()``-driven index loops are usually slower and less
clear than the vectorised operations of pandas/numpy. Clear, correct loops first; know
their limits for scale. The next stage turns to Python's data structures, starting with
strings.

.. hint::

   - :doc:`For Loops in Python <014-for-loops-in-python>`
   - :doc:`While Loops and Iteration in Python <013-while-loops-and-iteration-in-python>`
   - :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`
   - :doc:`Strings in Python <016-strings-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/range-function-and-loop-control-in-python/ <https://insightful-data-lab.com/2023/12/06/range-function-and-loop-control-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: control
