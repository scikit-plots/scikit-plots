:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-002:
.. _da-7-python-python-002:

========================================================================
Python Fundamentals
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 002`

◀ :doc:`Previous <001-introduction-to-python-and-programming-fundamentals>` · :doc:`Next <003-jupyter-notebook-and-coding-environments>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


The building blocks
--------------------

With the motivation established, this lesson covers the **fundamentals of Python** — the
core building blocks from which every program is made. These basics (values,
expressions, statements, and how code runs) are the vocabulary the rest of the section
builds on, introduced here in Python's clean syntax.

Values and expressions
------------------------

Python works with *values* — the data a program manipulates:

.. code-block:: python

   42              # an integer
   3.14            # a float (decimal number)
   "hello"         # a string (text)
   True            # a boolean (True or False)

Values combine into *expressions* that Python evaluates to a result:

.. code-block:: python

   2 + 3           # evaluates to 5
   "data" + "!"    # evaluates to "data!" (string concatenation)
   10 > 5          # evaluates to True

An expression is anything Python can compute a value from — the basic unit of doing
something with data.

Statements and printing
-------------------------

A *statement* is an instruction Python executes. The ``print`` function displays a
value, the standard way to see a program's output:

.. code-block:: python

   print("Hello, data analysis!")
   print(2 + 3)                       # displays 5

Programs are sequences of statements executed top to bottom, in order — the "sequence of
instructions" from the previous lesson made concrete.

Variables: naming values
--------------------------

A *variable* stores a value under a name, so it can be reused and changed (a dedicated
lesson follows):

.. code-block:: python

   sales = 1000
   tax_rate = 0.08
   total = sales + sales * tax_rate
   print(total)                       # displays 1080.0

Assigning with ``=`` binds a name to a value; the name then stands for the value
wherever used. Variables are what let a program hold and manipulate data through
meaningful names rather than raw values.

Python's readability
----------------------

A defining feature of Python is *indentation as structure* — Python uses indentation
(spaces at the start of a line) to group code into blocks, where other languages use
braces. This enforces the visual clarity that makes Python readable, and it means
consistent indentation is not optional style but *required syntax*. This readability —
clean syntax, English-like keywords, meaningful indentation — is much of why Python
suits analysts, and why code written in it is comparatively easy to read and maintain.

The caveat
------------

Python's fundamentals are simple individually but combine into subtle behaviour, and a
few basics trip up beginners. Python is *case-sensitive* (``Sales`` and ``sales`` are
different names); indentation errors (inconsistent spaces) are a common frustration
precisely because indentation is meaningful; and the distinction between *values* and
the *variables* naming them matters as programs grow. These are learned by writing and
running code, not by reading about it — which is why the environment for doing so, the
next lesson's subject, matters. The fundamentals reward practice: type the examples, run
them, and change them to see what happens. The next lesson covers where to do that.

.. hint::

   - :doc:`Introduction to Python and Programming Fundamentals <001-introduction-to-python-and-programming-fundamentals>`
   - :doc:`Variables in Python <005-variables-in-python>`
   - :doc:`Data Types and Type Conversion in Python <007-data-types-and-type-conversion-in-python>`
   - :doc:`Jupyter Notebook and Coding Environments <003-jupyter-notebook-and-coding-environments>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/python-fundamentals/ <https://insightful-data-lab.com/2023/12/06/python-fundamentals/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics
