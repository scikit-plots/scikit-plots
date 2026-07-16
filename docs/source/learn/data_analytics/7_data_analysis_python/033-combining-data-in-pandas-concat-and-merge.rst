:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _da-python-033:
.. _da-7-python-python-033:

========================================================================
Combining Data in Pandas (concat and merge)
========================================================================

:bdg-primary:`🐍 Data Analysis Using Python` :bdg-secondary:`🐼 NumPy & pandas` :bdg-info:`Lesson 033`

◀ :doc:`Previous <032-grouping-and-aggregation-in-pandas-groupby-agg>` · :doc:`↑ Section <index>` · :doc:`↑ Hub <../index>`


Bringing DataFrames together
------------------------------

Data rarely lives in one table, and the final pandas skill — and the last lesson of the
Python section — is **combining DataFrames**: stacking them with ``concat`` and joining them
on keys with ``merge``. These are the pandas equivalents of the SQL ``JOIN`` and ``UNION``,
completing the analyst's ability to do in Python everything the earlier tools did. This lesson
covers combining data in pandas.

Stacking with concat
---------------------

``pd.concat`` stacks DataFrames together — vertically (more rows) or horizontally (more
columns):

.. code-block:: python

   import pandas as pd
   jan = pd.DataFrame({"region": ["N", "S"], "sales": [100, 200]})
   feb = pd.DataFrame({"region": ["N", "S"], "sales": [150, 250]})

   pd.concat([jan, feb])                 # stack vertically: 4 rows
   pd.concat([jan, feb], ignore_index=True)   # renumber the index

Vertical ``concat`` appends rows — combining data of the *same columns* from different sources
(January and February sales, files from different periods), exactly SQL's ``UNION``. This is
how you assemble one dataset from multiple like-structured pieces.

Joining with merge
------------------

``pd.merge`` combines DataFrames by *matching keys* — the pandas ``JOIN``:

.. code-block:: python

   sales = pd.DataFrame({"region": ["N", "S", "E"], "sales": [100, 200, 300]})
   info  = pd.DataFrame({"region": ["N", "S", "E"], "manager": ["Amy", "Bo", "Cy"]})

   pd.merge(sales, info, on="region")    # join on the shared 'region' key

   # join types mirror SQL exactly:
   pd.merge(sales, info, on="region", how="inner")   # only matching keys (default)
   pd.merge(sales, info, on="region", how="left")    # all left rows, matched where possible
   pd.merge(sales, info, on="region", how="outer")   # all rows from both

``merge`` matches rows on a key column (``on="region"``) and combines their columns — and its
``how`` parameter (``inner``, ``left``, ``right``, ``outer``) is *exactly* the SQL join types
from the analysis section. Everything learned about JOINs — matching keys, the join types, what
each keeps — applies directly; ``pd.merge`` is the SQL JOIN in pandas.

Combining versus the earlier tools
------------------------------------

These operations complete the mapping of the whole analytical workflow into pandas.
``pd.concat`` (vertical) is ``UNION`` — stacking like-structured data; ``pd.merge`` is
``JOIN`` — combining related tables on keys, with the same inner/left/right/outer semantics.
The relational-combination concepts from Section 5 (the SQL section) carry over one-to-one, so
an analyst who understands JOINs already understands ``merge`` — only the syntax differs. With
combining in hand, pandas covers the entire span: import, clean, transform, filter, group,
aggregate, and join — the whole workflow, in reproducible Python.

The caveat
------------

Combining data concentrates the hazards flagged throughout the JOIN and integration lessons.
``merge`` on a key with *duplicates* can multiply rows unexpectedly (a many-to-many match
produces the cross-product, the row-explosion warned about with JOINs); a ``merge`` on keys
that do not cleanly correspond (inconsistent formatting, "N" versus "North") silently *fails to
match*, dropping rows in an inner join or producing NaN in an outer — the key-integrity
discipline from the cleaning section, now in pandas. ``concat`` assumes the pieces share
structure; stacking DataFrames with mismatched columns produces misaligned data or unexpected
NaN. As always: know your keys, verify the row count after a merge is what you expect (a sharp
change signals a join problem), and ensure concatenated pieces truly align. Combine
deliberately, checking the result.

This completes the Data Analysis with Python section — and with it, the analyst's core
technical toolkit. Across this section you have moved from Python's fundamentals through its
data structures to the libraries that make it a data powerhouse: NumPy's fast vectorized
arrays, and pandas' DataFrames, on which you can clean, transform, filter, group, aggregate,
and join data — everything the spreadsheet and SQL sections taught, now automated and
reproducible in code. Python does not replace the earlier tools so much as unify and scale
them: the concepts are the same, expressed in one programmable environment. With data skills
from foundations through Python now in place, the final section turns from *doing* the work to
*getting the role* — the job search: presenting your skills, finding opportunities, and
succeeding in interviews as a data analyst.

.. hint::

   - :doc:`Grouping and Aggregation in Pandas (groupby, agg) <032-grouping-and-aggregation-in-pandas-groupby-agg>`
   - :doc:`Using JOIN in SQL to Combine Tables <../5_analyze_data/017-using-join-in-sql-to-combine-tables>`
   - :doc:`Pandas DataFrame & Series <030-pandas-dataframe-and-series>`
   - :doc:`Introduction to Pandas (Data Analysis Library) <029-introduction-to-pandas-data-analysis-library>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2023/12/06/combining-data-in-pandas-concat-and-merge/ <https://insightful-data-lab.com/2023/12/06/combining-data-in-pandas-concat-and-merge/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analytics, topic: python, topic: libraries
