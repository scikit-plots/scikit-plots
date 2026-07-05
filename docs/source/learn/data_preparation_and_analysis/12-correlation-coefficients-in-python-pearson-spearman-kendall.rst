.. _dpa-correlation-coefficients-in-python-pearson-spearman-kendall:

========================================================================
Correlation Coefficients in Python (Pearson, Spearman, Kendall)
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 12 of 56  ·  *beginner*

:doc:`◀ Previous · Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`   ·   :doc:`Next · Karl Pearson <13-karl-pearson> ▶`


Three coefficients
--------------------

Pearson is one of **three** standard correlation coefficients, and the other two fix its blind spots.
All three run from :math:`-1` to :math:`+1` and share the same sign convention, but they measure
**different notions** of "moving together".

Pearson, Spearman, Kendall
----------------------------

The trio:

* **Pearson** :math:`r` — the **linear** coefficient from the last lesson; assumes a straight-line relationship and is sensitive to outliers.
* **Spearman** :math:`\rho` — Pearson applied to the **ranks** of the data. It captures any **monotonic** relationship (always rising or always falling, even if curved), and, working on ranks, it is **robust to outliers** and fine for **ordinal** data:

  .. math::

     \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}, \qquad d_i = \operatorname{rank}(x_i) - \operatorname{rank}(y_i).

* **Kendall** :math:`\tau` — based on counting **concordant** versus **discordant** pairs:

  .. math::

     \tau = \frac{n_c - n_d}{\tfrac{1}{2}\,n(n-1)},

  also a monotonic measure, often preferred for **small samples** and data with many **ties**.

Which to use
--------------

A rule of thumb: reach for **Pearson** when the relationship looks **linear** and the data are
well-behaved; switch to **Spearman** or **Kendall** when it is **monotonic but curved**, when there
are **outliers**, or when a variable is **ordinal** rather than numeric. Spearman and Kendall usually
agree; Kendall is the more conservative on small or tie-heavy data.

In Python
-----------

All three are one call away. A whole correlation matrix comes from ``df.corr(method="pearson")`` (or
``"spearman"`` / ``"kendall"``); for a single pair with a **p-value**, use ``scipy.stats.pearsonr``,
``spearmanr`` or ``kendalltau``. On the taxi data, fare against distance shows a strong positive
correlation by any of the three.

.. seealso::

   **Related lessons:** :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`  ·  :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Karl Pearson <13-karl-pearson>`  ·  :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/14/correlation-coefficients-in-python-pearson-spearman-kendall/ <https://insightful-data-lab.com/2026/01/14/correlation-coefficients-in-python-pearson-spearman-kendall/>`__

.. tags:: purpose: reference, topic: data preparation, level: beginner
