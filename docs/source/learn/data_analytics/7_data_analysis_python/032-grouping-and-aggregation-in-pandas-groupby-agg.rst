:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-032:
.. _da-7-python-python-032:

========================================================================
Grouping and Aggregation in Pandas (groupby, agg)
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 032`

◀ :doc:`Previous <031-boolean-masking-in-pandas>` · :doc:`Next <033-combining-data-in-pandas-concat-and-merge>` ▶ · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.


Grouping and summarising
--------------------------

The analytical workhorse — grouping rows by a category and computing an aggregate per group —
pandas does with **groupby** and aggregation. This is the direct equivalent of SQL's
``GROUP BY`` and the spreadsheet's pivot table, and it is where pandas becomes a serious
analytical tool. This lesson covers grouping and aggregation.

The groupby operation
---------------------

``groupby`` splits a DataFrame into groups by a column, then an aggregate is computed for each
group:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame({
       "region": ["N", "S", "N", "S", "E"],
       "sales":  [100, 200, 150, 250, 300],
   })

   df.groupby("region")["sales"].sum()
   # region
   # E    300
   # N    250
   # S    450

``df.groupby("region")["sales"].sum()`` groups rows by region, then sums sales within each
group — producing the per-region total. This is exactly SQL's
``SELECT region, SUM(sales) FROM df GROUP BY region`` and a pivot table of sales by region, in
pandas.

Aggregation functions and agg
-------------------------------

Any aggregate can be applied per group, and ``agg`` computes several at once:

.. code-block:: python

   df.groupby("region")["sales"].mean()     # average per region
   df.groupby("region")["sales"].count()    # count per region

   df.groupby("region")["sales"].agg(["sum", "mean", "count"])
   # multiple aggregates per group, as a table

   df.groupby("region").agg(
       total_sales=("sales", "sum"),
       avg_sales=("sales", "mean"),
   )                                          # named aggregates

The aggregate functions are the familiar ones (``sum``, ``mean``, ``count``, ``min``, ``max``,
``std``); ``agg`` applies several, optionally with names — the pandas way to build a
multi-metric summary per group, like a richer pivot table or a ``GROUP BY`` with several
aggregates.

Grouping versus the earlier tools
-----------------------------------

Groupby is the *grouped aggregation* from every prior section, in pandas. It is SQL's
``GROUP BY`` with aggregate functions (Section 5), the spreadsheet's pivot table (Section 5),
and the conditional-aggregation ideas — all expressed as ``df.groupby(col).agg(...)``. The
"split-apply-combine" it performs (split into groups, apply an aggregate, combine into a
result) is the same operation, and recognising it as the familiar pivot/GROUP BY makes it
immediately meaningful. It is the analytical core of pandas, as GROUP BY was of SQL.

The caveat
------------

Groupby is powerful and carries the aggregation subtleties seen throughout, plus pandas
specifics. Aggregates handle *missing values* by skipping them (``mean`` ignores NaN, changing
the denominator — the null-in-aggregate theme); grouping on a column with inconsistent values
fragments groups (the "NY" versus "New York" problem, so *clean before grouping*, exactly as in
SQL); and the *result* of a groupby has the grouping column as its index, which sometimes
surprises (``reset_index()`` turns it back into a column). As always, the operation computes
precisely what you specify — verify the groups are what you intend and the aggregates handle
missing data as you want. This nearly completes the section; the final lesson covers combining
DataFrames — the pandas equivalent of the JOIN.

.. hint::

   - :doc:`Boolean Masking in Pandas <031-boolean-masking-in-pandas>`
   - :doc:`Using GROUP BY and ORDER BY for Aggregated Calculations in SQL <../5_analyze_data/027-using-group-by-and-order-by-for-aggregated-calculations-in-sql>`
   - :doc:`Combining Data in Pandas (concat and merge) <033-combining-data-in-pandas-concat-and-merge>`
   - :doc:`Pandas DataFrame & Series <030-pandas-dataframe-and-series>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/grouping-and-aggregation-in-pandas-groupby-agg/ <https://insightful-data-lab.com/2023/12/06/grouping-and-aggregation-in-pandas-groupby-agg/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
