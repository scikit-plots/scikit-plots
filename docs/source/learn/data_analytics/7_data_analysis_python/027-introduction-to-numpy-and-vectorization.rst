:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-027:
.. _da-7-python-python-027:

========================================================================
Introduction to NumPy and Vectorization
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 027`

◀ :doc:`Previous <026-libraries-packages-and-modules-in-python>` · :doc:`Next <028-numpy-arrays-ndarray-and-core-concepts>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Fast numbers without loops
----------------------------

The first essential data library is **NumPy** — the foundation of numerical computing in
Python, and the base that pandas itself is built on. Its central idea is **vectorization**:
performing operations on entire arrays at once, without explicit loops, far faster than
looping. This lesson introduces NumPy and vectorization, the shift from loop-based to
array-based thinking.

What NumPy provides
--------------------

NumPy provides the **array** — a fast, homogeneous collection of numbers — and a vast set of
operations on arrays. Imported conventionally as ``np``:

.. code-block:: python

   import numpy as np

   arr = np.array([1, 2, 3, 4, 5])     # a NumPy array
   arr * 2                             # array([2, 4, 6, 8, 10]) — whole array at once
   arr + 10                            # array([11, 12, 13, 14, 15])
   arr.sum()                           # 15
   arr.mean()                          # 3.0

The striking thing: ``arr * 2`` multiplies *every* element by 2 in one operation — no loop.
This is vectorization.

Vectorization
-------------

**Vectorization** means applying an operation to a whole array at once, rather than looping
over elements. Compare:

.. code-block:: python

   # loop way (slow, verbose):
   result = []
   for x in [1, 2, 3, 4, 5]:
       result.append(x * 2)

   # vectorized way (fast, concise):
   arr = np.array([1, 2, 3, 4, 5])
   result = arr * 2

Both double each element, but the vectorized version is *dramatically faster* on large data
(NumPy runs the operation in optimised low-level code, not a Python loop) and *more concise*
(one expression, no loop). This is the payoff the for-loop lesson foreshadowed — for numeric
data at scale, vectorized array operations replace explicit loops.

Why vectorization matters
--------------------------

Vectorization is the core idiom of numerical data work in Python, for two reasons. **Speed**
— vectorized NumPy operations are orders of magnitude faster than Python loops on large
arrays, which is what makes Python viable for real data sizes. **Clarity** — ``arr * 2`` or
``arr1 + arr2`` expresses an operation on all the data in one readable line, closer to the
mathematical intent than a loop. This array-and-vectorization thinking underlies pandas too
(a DataFrame column is array-like and vectorized), so learning it here is learning the idiom
of all Python data analysis.

The caveat
------------

Vectorization requires a *mental shift* that trips up those coming from loop-based
programming: you stop thinking "for each element, do X" and start thinking "do X to the whole
array." Reaching for a Python loop over a NumPy array *defeats the purpose* — it is both
slower and less clear than the vectorized operation, yet it is the instinctive move for
loop-trained programmers. The discipline is to express operations on whole arrays, using
NumPy's vectorized operations and functions rather than looping. There are genuine cases
where a loop is unavoidable, but for element-wise arithmetic and aggregation, vectorization
is the right and expected approach. Think in arrays, not elements. The next lesson covers the
array structure itself in depth.

.. hint::

   - :doc:`Libraries, Packages, and Modules in Python <026-libraries-packages-and-modules-in-python>`
   - :doc:`NumPy Arrays (ndarray) and Core Concepts <028-numpy-arrays-ndarray-and-core-concepts>`
   - :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`
   - :doc:`Advanced Use of Loops, Lists, Tuples & List Comprehension <022-advanced-use-of-loops-lists-tuples-and-list-comprehension>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/introduction-to-numpy-and-vectorization/ <https://insightful-data-lab.com/2023/12/06/introduction-to-numpy-and-vectorization/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
