.. _dpa-the-first-step-in-knowing-your-data:

========================================================================
The First Step in Knowing Your Data
========================================================================

**Stage 1 · 📋 Foundations**  ·  Lesson 05 of 56  ·  *beginner*

:doc:`◀ Previous · Big Data: Definition, Characteristics, Evolution, and Business Impact <04-big-data-definition-characteristics-evolution-and-business-impact>`   ·   :doc:`Next · IEEE 754 Floating-Point Standard ▶ <06-ieee-754-floating-point-standard>`   ·   :doc:`↑ Section <index>`


Look before you model
-----------------------

Before fitting anything, **look at the data**. This is CRISP-DM's **Data Understanding** phase, and
skipping it is how projects quietly go wrong — a mis-typed column, a hidden pile of missing values, or
an outlier that wrecks a model. The first step is always **profiling** what you actually have.

What to check
---------------

A good first pass answers a handful of questions:

* **Shape** — how many rows and columns?
* **Types** — which columns are numeric, categorical, or dates?
* **Ranges and distributions** — the min, max, centre and spread of each numeric field;
* **Missingness** — how many values are absent, and where?
* **Cardinality** — how many distinct values do categorical fields take?
* **Duplicates and oddities** — repeated rows, impossible values, surprising codes.

A first pass in pandas
------------------------

In practice a few ``pandas`` calls cover most of it:

.. code-block:: python

   df.shape            # (rows, columns)
   df.info()           # column names, dtypes, non-null counts
   df.describe()       # count, mean, std, min, quartiles, max (numeric)
   df.isnull().sum()   # missing values per column
   df.nunique()        # distinct values per column

Understanding, not just numbers
---------------------------------

Numbers alone are not understanding. The real goal is to know **what each variable means** — its
units, how it was collected, what a missing value signifies — so that later choices (which features to
use, how to handle gaps) are informed rather than mechanical. Every stage that follows, from
association measures to model evaluation, rests on this first honest look.

.. hint::

   **Related lessons:** :doc:`The Process of Data Analysis <02-the-process-of-data-analysis>`  ·  :doc:`CRISP-DM for Data Science <03-crisp-dm-for-data-science>`  ·  :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`IEEE 754 Floating-Point Standard <06-ieee-754-floating-point-standard>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/the-first-step-in-knowing-your-data/ <https://insightful-data-lab.com/2026/01/14/the-first-step-in-knowing-your-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner
