:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-013:
.. _da-7-python-python-013:

========================================================================
While Loops and Iteration in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🔀 Control Flow` :bdg-info:`Lesson 013`

◀ :doc:`Previous <012-branching-and-conditional-statements-in-python>` · :doc:`Next <014-for-loops-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Repeating while a condition holds
-----------------------------------

The second pillar of control flow (after branching) is **iteration** — repeating code —
and the **while loop** is its most basic form: run a block *repeatedly as long as a
condition remains true*. Opening the discussion of loops, this lesson covers the while
loop and the idea of iteration that the for-loop and data-processing lessons build on.

The while loop
--------------

A ``while`` loop repeats its block while its condition is true:

.. code-block:: python

   count = 1
   while count <= 5:
       print(count)
       count += 1          # crucial: move toward ending the loop

Python checks the condition (``count <= 5``); if true, it runs the block, then checks
again — repeating until the condition becomes false. Here it prints 1 through 5, then
stops when ``count`` reaches 6. The condition uses the boolean logic from the earlier
lesson, and the loop continues until that condition fails.

The crucial detail: making progress
--------------------------------------

The single most important thing about a while loop is that *something in the loop must
eventually make the condition false* — otherwise it runs forever. In the example,
``count += 1`` is what moves toward termination; without it, ``count`` stays 1, the
condition stays true, and the loop never ends (an **infinite loop**). Every while loop
must contain something that progresses toward its exit condition, and forgetting this is
the classic while-loop bug.

Iteration as a concept
------------------------

**Iteration** — doing something repeatedly — is fundamental to programming and
especially to data work, where the same operation applies to many items (every row, every
value, every file). The while loop expresses iteration in its most general form: repeat
until a condition says stop. This general form suits situations where you do not know in
advance how many repetitions are needed — keep going until something changes (until the
data runs out, until a target is reached, until input stops). For iterating a *known*
collection, the for loop (next lesson) is usually cleaner, but the while loop handles the
open-ended cases.

The caveat
------------

The while loop's power to repeat is exactly its danger: the **infinite loop**. A
condition that never becomes false — because the loop forgets to make progress, or the
progress never reaches the exit — hangs the program indefinitely. This is the signature
while-loop failure, and guarding against it means ensuring every while loop has a clear
path to its exit condition and that something in the body advances toward it. When you do
know how many times to repeat, or you are iterating a collection, the for loop is safer
because it *cannot* loop infinitely over a finite collection. Reach for while only when
the open-ended condition genuinely calls for it, and always verify the loop can end. The
next lesson covers the for loop.

.. hint::

   - :doc:`Branching and Conditional Statements in Python <012-branching-and-conditional-statements-in-python>`
   - :doc:`For Loops in Python <014-for-loops-in-python>`
   - :doc:`range() Function and Loop Control in Python <015-range-function-and-loop-control-in-python>`
   - :doc:`Boolean Data, Comparators, and Logical Operators in Python <011-boolean-data-comparators-and-logical-operators-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/while-loops-and-iteration-in-pytho/ <https://insightful-data-lab.com/2023/12/06/while-loops-and-iteration-in-pytho/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: control
