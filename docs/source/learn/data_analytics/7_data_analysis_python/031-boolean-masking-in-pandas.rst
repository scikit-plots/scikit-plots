:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-031:
.. _da-7-python-python-031:

========================================================================
Boolean Masking in Pandas
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 031`

◀ :doc:`Previous <030-pandas-dataframe-and-series>` · :doc:`Next <032-grouping-and-aggregation-in-pandas-groupby-agg>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Filtering rows by condition
-----------------------------

The most common data operation — selecting the rows that meet a condition — pandas does with
**boolean masking**: a condition produces an array of True/False, and the DataFrame keeps the
True rows. This is the pandas equivalent of SQL's ``WHERE`` and the spreadsheet filter, and it
is fundamental. This lesson covers boolean masking.

How masking works
-----------------

A comparison on a DataFrame column produces a **boolean Series** (a mask), which then selects
rows:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({"region": ["N","S","E"], "sales": [1000, 800, 1200]})

   df["sales"] > 900         # a boolean Series: [True, False, True]
   df[df["sales"] > 900]     # keeps rows where the mask is True

``df["sales"] > 900`` computes the condition for *every* row at once (vectorized, the NumPy
boolean-indexing idea), producing True/False per row; ``df[ ... ]`` with that mask returns only
the rows where it is True. This is filtering — "the rows where sales exceed 900" — in one
readable expression.

Combining conditions
--------------------

Multiple conditions combine with ``&`` (and), ``|`` (or), and ``~`` (not), each condition
parenthesised:

.. code-block:: python

   df[(df["sales"] > 900) & (df["region"] == "N")]   # both conditions
   df[(df["region"] == "N") | (df["region"] == "S")] # either
   df[~(df["sales"] > 900)]                            # not

The parentheses around each condition are *required* (pandas' operator precedence demands
them), and ``&``/``|`` are used, not the words ``and``/``or`` — a pandas-specific rule.
Compound masks express multi-condition filters, exactly the ``WHERE ... AND ...`` of SQL.

Masking versus the earlier filters
------------------------------------

Boolean masking is precisely the *filtering* concept from every prior section, in pandas form.
``df[df["sales"] > 900]`` is SQL's ``SELECT * FROM df WHERE sales > 900`` and the spreadsheet's
filter-to-sales-over-900 — the identical operation, vectorized in Python. Recognising this
makes masking intuitive: it is the familiar "show only the rows meeting this condition,"
expressed as a boolean Series selecting from a DataFrame. It is also used for *modifying*
subsets (``df.loc[df["sales"] > 900, "flag"] = True`` sets a value on matching rows).

Why masking matters
-------------------

Boolean masking is one of the most-used pandas operations, because filtering to relevant rows
is constant in analysis — isolating a segment, removing outliers, selecting valid data,
finding records meeting criteria. It is vectorized (fast on large data) and expressive (complex
conditions in one line), and it underlies much of data cleaning and analysis in pandas. Master
masking and a large fraction of practical pandas filtering follows.

The caveat
------------

Masking has pandas-specific traps that catch beginners predictably. The operators are ``&``,
``|``, ``~`` — *not* ``and``, ``or``, ``not`` (using the words raises an error on Series); and
each condition *must* be parenthesised (``df[df["a"]>1 & df["b"]<2]`` is wrong — precedence
mis-groups it — while ``df[(df["a"]>1) & (df["b"]<2)]`` is right). Masks involving *missing
values* (NaN) behave carefully — a comparison with NaN is False, so masking silently excludes
missing-value rows (the null-handling theme, in pandas). And modifying a masked subset invites
the view-versus-copy ``SettingWithCopyWarning``, so use ``df.loc[mask, col] = ...`` for
assignment. Parenthesise conditions, use the symbol operators, and mind NaN and assignment. The
next lesson covers grouping and aggregation.

.. hint::

   - :doc:`Pandas DataFrame & Series <030-pandas-dataframe-and-series>`
   - :doc:`Sorting and Filtering Data in SQL Using ORDER BY and WHERE <../5_analyze_data/005-sorting-and-filtering-data-in-sql-using-order-by-and-where>`
   - :doc:`Grouping and Aggregation in Pandas (groupby, agg) <032-grouping-and-aggregation-in-pandas-groupby-agg>`
   - :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/boolean-masking-in-pandas/ <https://insightful-data-lab.com/2023/12/06/boolean-masking-in-pandas/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
