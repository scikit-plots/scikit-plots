:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-007:
.. _da-7-python-python-007:

========================================================================
Data Types and Type Conversion in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 007`

◀ :doc:`Previous <006-naming-conventions-and-restrictions-in-python>` · :doc:`Next <008-functions-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The kinds of values
--------------------

Every value in Python has a **type** — the kind of data it is — and the type determines
what can be done with the value. Understanding Python's core **data types** and how to
**convert** between them is essential, because type mismatches are among the most common
sources of errors, and type conversion is a routine part of preparing data (echoing the
``CAST`` work from SQL).

The core data types
---------------------

Python's fundamental types for analysis:

- ``int`` — whole numbers: ``42``, ``-7``, ``1000``.
- ``float`` — decimal numbers: ``3.14``, ``0.08``, ``-2.5``.
- ``str`` — text (strings): ``"North"``, ``"data"``.
- ``bool`` — boolean truth values: ``True``, ``False``.
- ``list`` — an ordered collection: ``[1, 2, 3]`` (the structures stage covers these).
- ``dict`` — key-value pairs: ``{"region": "North"}`` (also the structures stage).

The ``type()`` function reports a value's type:

.. code-block:: python

   type(42)          # <class 'int'>
   type(3.14)        # <class 'float'>
   type("North")     # <class 'str'>

Knowing a value's type explains what operations it supports — you can add ``int`` s, and
you can concatenate ``str`` s, but adding an ``int`` to a ``str`` is an error.

Type conversion
----------------

Converting a value from one type to another uses the type's conversion function — the
Python counterpart of SQL's ``CAST``:

.. code-block:: python

   int("42")         # 42     — string to integer
   float("3.14")     # 3.14   — string to float
   str(1000)         # "1000" — number to string
   int(3.9)          # 3      — float to int (truncates, does not round)

This matters constantly in data work: data read from files often arrives as *strings*
(the import type-trap, familiar from earlier sections), and must be converted to numbers
before arithmetic — exactly the text-to-number cleaning done with ``VALUE`` in
spreadsheets and ``CAST`` in SQL, now in Python.

Why types matter
-----------------

Type errors are pervasive and often puzzling until understood. ``"5" + "3"`` gives
``"53"`` (string concatenation), not ``8``, because the values are strings, not numbers;
``"5" + 3`` raises an error (cannot add string and int). Understanding types explains
these behaviours and the fixes (convert the strings to numbers first). Since data
frequently arrives with the wrong types, type awareness and conversion are a routine,
essential part of Python data analysis — the same "get the types right before computing"
discipline that ran through cleaning and analysis.

The caveat
------------

Type conversion can *fail* or behave surprisingly, exactly as ``CAST`` did. Converting a
non-numeric string to a number (``int("hello")``) raises an error, and a single bad
value can break a conversion over a whole column — which, as in SQL, usefully *forces*
you to confront the non-conforming data. ``int()`` on a float *truncates* rather than
rounds (``int(3.9)`` is ``3``, not ``4``), a common surprise. And converting can lose
information (float to int drops the decimal). As always, understand what a conversion
does to the data, verify the result, and handle the values that will not convert. The
next lesson turns to functions — reusable blocks of code.

.. hint::

   - :doc:`Variables in Python <005-variables-in-python>`
   - :doc:`Understanding Data Types and Data Formats <../3_data_preparation/003-understanding-data-types-and-data-formats>`
   - :doc:`Using CAST to Clean and Format Data in SQL <../4_data_cleaning_preparation/022-using-cast-to-clean-and-format-data-in-sql>`
   - :doc:`Functions in Python <008-functions-in-python>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/data-types-and-type-conversion-in-python/ <https://insightful-data-lab.com/2023/12/06/data-types-and-type-conversion-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics
