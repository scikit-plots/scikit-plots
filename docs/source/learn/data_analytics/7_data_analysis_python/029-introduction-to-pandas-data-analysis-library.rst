:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-029:
.. _da-7-python-python-029:

========================================================================
Introduction to Pandas (Data Analysis Library)
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 029`

◀ :doc:`Previous <028-numpy-arrays-ndarray-and-core-concepts>` · :doc:`Next <030-pandas-dataframe-and-series>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Tables in Python
-----------------

If NumPy is the foundation, **pandas** is the tool an analyst uses most — the library that
brings *tabular data* to Python, with the row-and-column tables familiar from spreadsheets
and SQL. Built on NumPy, pandas is the primary Python tool for data analysis, and this lesson
introduces it: what it is, and why it ties the whole course together.

What pandas provides
--------------------

Pandas provides labelled, tabular data structures and a vast toolkit for working with them.
Imported conventionally as ``pd``:

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({
       "region": ["North", "South", "East"],
       "sales":  [1000, 800, 1200],
   })

This creates a **DataFrame** — a table with named columns and indexed rows, exactly the
tabular structure the whole course has worked with. Pandas can read data from files
(``pd.read_csv("data.csv")``), databases, and more, turning external data into a DataFrame to
analyse.

Pandas as the culmination
---------------------------

Pandas is where the course's threads converge, because it does *in Python* what the earlier
tools did separately:

- **The tabular structure** from the data-preparation section — rows and columns, tidy data —
  is the DataFrame.
- **The cleaning** from Section 4 — handling duplicates, missing values, types — pandas does
  with methods (``drop_duplicates()``, ``fillna()``, ``astype()``).
- **The analysis** from Section 5 — sorting, filtering, grouping, aggregating, joining —
  pandas does with methods (``sort_values()``, boolean masks, ``groupby()``, ``merge()``).
- **The visualization** from Section 6 — pandas integrates with plotting libraries to chart
  directly.

Everything done in spreadsheets and SQL, pandas can do in Python — automated, reproducible,
and at scale. It is the single tool that spans the whole analytical workflow.

Why pandas matters
------------------

Pandas matters because it unifies the analyst's work in one powerful, programmable tool. The
spreadsheet's visibility and the SQL query's power, combined with Python's automation and
scale, come together in the DataFrame — clean it, transform it, analyse it, and visualize it,
all in reproducible code. For serious Python data analysis, pandas is *the* tool, which is why
this section builds toward it, and why the concepts from every prior section reappear here in
DataFrame form.

The caveat
------------

Pandas is powerful and correspondingly large, with a steep learning curve and many ways to do
each thing — some efficient, some not. Two cautions matter early. First, pandas rewards the
*vectorized* thinking from NumPy: iterating over a DataFrame's rows with a Python loop is slow
and un-idiomatic, where a vectorized operation or built-in method does the same far faster —
the "don't loop" lesson applies doubly to DataFrames. Second, pandas' size means there is
usually a clean built-in method for what you want, so reaching for a convoluted manual
approach often means missing the tool that does it in one call. Learn the idiomatic,
vectorized pandas — the methods and masks — rather than writing loops against it. The next
lessons cover its structures and operations. The whole course's analytical vocabulary is
about to reappear, in Python.

.. hint::

   - :doc:`NumPy Arrays (ndarray) and Core Concepts <028-numpy-arrays-ndarray-and-core-concepts>`
   - :doc:`Pandas DataFrame & Series <030-pandas-dataframe-and-series>`
   - :doc:`Libraries, Packages, and Modules in Python <026-libraries-packages-and-modules-in-python>`
   - :doc:`Introduction to NumPy and Vectorization <027-introduction-to-numpy-and-vectorization>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/introduction-to-pandas-data-analysis-library/ <https://insightful-data-lab.com/2023/12/06/introduction-to-pandas-data-analysis-library/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
