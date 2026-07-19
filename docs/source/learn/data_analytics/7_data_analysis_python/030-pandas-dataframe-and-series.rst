:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-030:
.. _da-7-python-python-030:

========================================================================
Pandas DataFrame & Series
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 030`

◀ :doc:`Previous <029-introduction-to-pandas-data-analysis-library>` · :doc:`Next <031-boolean-masking-in-pandas>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


The two core structures
------------------------

Pandas is built on two structures: the **Series** (a single column) and the **DataFrame** (a
table of columns). Understanding them — how they relate, and how to access their data — is the
foundation of all pandas work. This lesson covers the DataFrame and Series in depth.

The Series: a column
---------------------

A **Series** is a one-dimensional labelled array — essentially a single column of data with an
index:

.. code-block:: python

   import pandas as pd
   sales = pd.Series([1000, 800, 1200], index=["North", "South", "East"])
   sales["North"]            # 1000 — access by label
   sales.mean()              # 1000.0 — vectorized aggregate

A Series is like a NumPy array (vectorized, homogeneous-ish) but with *labels* (an index) and
pandas' richer methods. Each column of a DataFrame is a Series.

The DataFrame: a table
-----------------------

A **DataFrame** is a two-dimensional labelled table — rows and named columns — the central
pandas structure:

.. code-block:: python

   df = pd.DataFrame({
       "region": ["North", "South", "East"],
       "sales":  [1000, 800, 1200],
       "customers": [50, 40, 60],
   })

   df["sales"]               # a column (a Series)
   df[["region", "sales"]]   # multiple columns (a DataFrame)
   df.head()                 # first rows
   df.shape                  # (3, 3) — rows, columns
   df.info()                 # summary of columns and types
   df.describe()             # summary statistics of numeric columns

A DataFrame is a collection of Series (columns) sharing an index (rows) — exactly the tabular,
one-column-per-variable structure from the data-preparation section, now a Python object with
methods.

Accessing rows and columns
----------------------------

Pandas accesses data by label or position:

.. code-block:: python

   df["sales"]               # a column by name
   df.loc[0]                 # a row by label (index)
   df.iloc[0]                # a row by position
   df.loc[0, "sales"]        # a specific cell by label
   df.iloc[0, 1]             # a specific cell by position

``loc`` accesses by *label*, ``iloc`` by *integer position* — the distinction to keep clear.
Columns are accessed by name (``df["sales"]``), rows by ``loc``/``iloc``. These are how you
reach any part of the table.

Why these structures matter
----------------------------

The DataFrame and Series are the objects *all* pandas analysis operates on, and they directly
embody the course's data concepts: the DataFrame is the tidy table, each column a Series (a
variable), each row an observation. Every operation ahead — filtering, grouping, aggregating,
joining — is a method on these structures, and the aggregates (``mean``, ``sum``) are
vectorized over the Series. Understanding that a DataFrame is a labelled table of Series
columns is the mental model that makes all of pandas coherent.

The caveat
------------

The DataFrame's flexibility hides subtleties that cause classic pandas confusion. The
``loc`` versus ``iloc`` distinction (label versus position) trips up beginners constantly —
they look similar but differ, and using the wrong one selects the wrong data. Pandas also has a
famous **SettingWithCopyWarning** arising from the difference between a *view* and a *copy* of
data — modifying what you think is the DataFrame but is actually a temporary slice, so the
change does not stick (or warns). And the index — pandas' row labels — behaves in ways that
surprise those expecting simple row numbers. These are learned through use and by reading
pandas' (generally helpful) warnings; the key early discipline is knowing ``loc`` from
``iloc`` and being deliberate about whether you are viewing or copying. The next lesson covers
filtering rows: boolean masking.

.. hint::

   - :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`
   - :doc:`Boolean Masking in Pandas <031-boolean-masking-in-pandas>`
   - :doc:`Grouping and Aggregation in Pandas (groupby, agg) <032-grouping-and-aggregation-in-pandas-groupby-agg>`
   - :doc:`NumPy Arrays (ndarray) and Core Concepts <028-numpy-arrays-ndarray-and-core-concepts>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/pandas-dataframe-series/ <https://insightful-data-lab.com/2023/12/06/pandas-dataframe-series/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
