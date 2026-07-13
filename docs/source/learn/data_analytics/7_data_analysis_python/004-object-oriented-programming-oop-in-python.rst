:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-004:
.. _data-analytics-python-004:
.. _da-foundations-python-004:
.. _da-decisions-python-004:
.. _da-prep-python-004:
.. _da-cleaning-python-004:
.. _da-analyze-python-004:
.. _da-viz-python-004:
.. _da-python-python-004:
.. _da-jobsearch-python-004:

========================================================================
Object-Oriented Programming (OOP) in Python
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐍 Python Fundamentals` :bdg-info:`Lesson 004`

◀ :doc:`Previous <003-jupyter-notebook-and-coding-environments>` · :doc:`Next <005-variables-in-python>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Everything is an object
-------------------------

Python is built on a model called **object-oriented programming**, and although an
analyst need not write elaborate object-oriented code, understanding the model *matters*
— because everything in Python is an object, including the data structures and library
objects analysis relies on. This lesson introduces OOP as the model that shapes how
Python (and pandas, and numpy) work.

Objects and methods
---------------------

In Python, a value is not just data — it is an **object** that bundles data together
with *methods* (functions that belong to it and act on it). You call a method with dot
notation:

.. code-block:: python

   name = "data analysis"
   name.upper()            # "DATA ANALYSIS" — upper() is a string method
   name.title()            # "Data Analysis"

   numbers = [3, 1, 2]
   numbers.sort()          # sorts the list in place — sort() is a list method
   numbers.append(4)       # adds to the list

``name.upper()`` calls the ``upper`` method *on* the string object ``name``. This
dot-notation — object, then method — is pervasive in Python, and recognising it as "call
this object's method" is key to reading Python code.

Classes: the blueprint
------------------------

Objects are created from **classes** — blueprints defining what data an object holds and
what methods it has. A string is an object of the ``str`` class; a list, of the ``list``
class; and the ``DataFrame`` you will use for data is an object of the ``DataFrame``
class from pandas. The class defines the type; each object is an *instance* of its class.
Analysts mostly *use* objects of existing classes (strings, lists, DataFrames) rather
than writing their own, but knowing that an object's available methods come from its
class explains why a DataFrame has different methods than a string.

Why OOP matters for analysis
------------------------------

The payoff is practical: the data tools of this section *are* objects, and using them is
calling their methods. A pandas ``DataFrame`` is an object with methods like
``.head()``, ``.groupby()``, and ``.mean()``:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({"region": ["N", "S"], "sales": [100, 200]})
   df.head()               # a DataFrame method — shows the first rows
   df["sales"].mean()      # method on the column — computes the mean

Understanding OOP means this syntax reads naturally: ``df.groupby(...)`` is "call the
DataFrame's groupby method," exactly the object-dot-method pattern. The whole of pandas
and numpy usage is calling methods on objects, so the OOP model is the grammar of Python
data analysis.

The caveat
------------

For analysts, the risk is the *opposite* of neglect — over-investing in OOP theory that
data analysis rarely requires. Writing custom classes, inheritance hierarchies, and
elaborate object-oriented designs is software-engineering work most analysts seldom
need; the analyst's use of OOP is mostly *understanding* it well enough to use library
objects fluently, not *building* object-oriented systems. Learn the model to the depth
that makes ``df.groupby(...).mean()`` legible and the library methods sensible — the
objects-and-methods grammar — and leave the deeper OOP design to when a genuine need
arises. The next lessons return to the hands-on basics: variables, naming, and types.

.. hint::

   - :doc:`Python Fundamentals <002-python-fundamentals>`
   - :doc:`Functions in Python <008-functions-in-python>`
   - :doc:`Variables in Python <005-variables-in-python>`
   - :doc:`Introduction to Python and Programming Fundamentals <001-introduction-to-python-and-programming-fundamentals>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/object-oriented-programming-oop-in-python/ <https://insightful-data-lab.com/2023/12/06/object-oriented-programming-oop-in-python/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: basics
