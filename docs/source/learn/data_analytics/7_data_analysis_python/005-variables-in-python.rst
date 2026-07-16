:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-005:
.. _da-7-python-python-005:

========================================================================
Variables in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 005`

◀ :doc:`Previous <004-object-oriented-programming-oop-in-python>` · :doc:`Next <006-naming-conventions-and-restrictions-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Naming values for reuse
-------------------------

A program that could not *remember* values would be useless; **variables** are how
Python remembers. A variable is a named container for a value, letting you store data,
refer to it by name, and change it as the program runs. This lesson covers variables in
Python — assigning, reassigning, and using them — the foundation of holding data in code.

Assignment
-----------

A variable is created by *assigning* a value to a name with ``=``:

.. code-block:: python

   sales = 1000
   region = "North"
   tax_rate = 0.08

The name on the left is bound to the value on the right; afterwards, the name stands for
the value:

.. code-block:: python

   total = sales + sales * tax_rate
   print(total)              # 1080.0
   print(region)             # North

The ``=`` here is *assignment*, not mathematical equality — it means "let this name
refer to this value," a distinction worth keeping clear.

Reassignment
-------------

A variable's value can *change* — reassigning binds the name to a new value:

.. code-block:: python

   count = 5
   count = count + 1         # count is now 6
   count += 1                # shorthand for count = count + 1; now 7

The last form, ``+=``, is an *augmented assignment* — a common shorthand for updating a
variable based on its current value (``-=``, ``*=``, ``/=`` work similarly). Reassignment
is what lets a variable track a changing value as a program runs.

Why variables matter
----------------------

Variables serve the same purposes as naming anything: *reuse* (compute a value once,
use it many times), *clarity* (a well-named variable documents what a value means —
``tax_rate`` is clearer than ``0.08`` scattered through code), and *changeability*
(update a value in one place and everything using it updates). This is the
abstraction-and-naming principle from the foundations, in code: meaningful names for
values make programs readable and maintainable, exactly as meaningful column names make
data readable.

The caveat
------------

Variables have subtleties that catch beginners. A variable must be *assigned before it
is used* — referencing a name Python has not seen raises an error. Reassignment means a
variable's value depends on *when* you look (its value is whatever was last assigned),
which matters especially in notebooks where cells run out of order — a variable can hold
a surprising value if cells ran in an unexpected sequence. And Python variables are
*case-sensitive* (``Sales`` ≠ ``sales``), a frequent source of "undefined name" errors.
These are learned by writing code and reading the errors, which say precisely what went
wrong. The next lesson covers naming variables well.

.. hint::

   - :doc:`Python Fundamentals <002-python-fundamentals>`
   - :doc:`Naming Conventions and Restrictions in Python <006-naming-conventions-and-restrictions-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Object-Oriented Programming (OOP) in Python <004-object-oriented-programming-oop-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/variables-in-python/ <https://insightful-data-lab.com/2023/12/06/variables-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics
