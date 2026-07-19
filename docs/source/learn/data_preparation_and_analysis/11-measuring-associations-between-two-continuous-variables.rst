.. _dpa-measuring-associations-between-two-continuous-variables:

========================================================================
Measuring Associations Between Two Continuous Variables
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 11 of 56  ·  *beginner*

:doc:`◀ Previous · Measuring Associations in Data <10-measuring-associations-in-data>`   ·   :doc:`Next · Correlation Coefficients in Python (Pearson, Spearman, Kendall) ▶ <12-correlation-coefficients-in-python-pearson-spearman-kendall>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Covariance: direction
-----------------------

The starting point for two continuous variables is **covariance**, which measures whether they vary
in the **same direction**:

.. math::

   \operatorname{cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}).

When above-average :math:`x` tends to pair with above-average :math:`y`, the products are positive and
covariance is **positive**; when high :math:`x` pairs with low :math:`y`, it is **negative**; near
zero means no linear tendency.

The problem with covariance
-----------------------------

Covariance has a flaw as a **strength** measure: its size depends on the variables' **units**.
Covariance of fare and distance changes if you switch miles to kilometres, so its magnitude is **not
comparable** across variable pairs — it ranges without bound. You can read its **sign**, but not judge
"how strong" from its value.

Pearson correlation
---------------------

The fix is to **standardise** covariance by the two standard deviations, giving the **Pearson
correlation coefficient**:

.. math::

   r = \frac{\operatorname{cov}(X, Y)}{\sigma_X \, \sigma_Y}
     = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
            {\sqrt{\sum (x_i - \bar{x})^2}\,\sqrt{\sum (y_i - \bar{y})^2}}.

Dividing out the units confines :math:`r` to the range :math:`[-1, 1]`, making it **comparable**
everywhere.

Reading r
-----------

On that scale, :math:`r = +1` is a **perfect positive** linear relationship, :math:`r = -1` a
**perfect negative** one, and :math:`r = 0` **no linear** relationship. The crucial caveat: Pearson
measures **linear** association only. A strong curved relationship can still give :math:`r \approx 0`,
and :math:`r` is **sensitive to outliers** — reasons the next lesson reaches for rank-based
alternatives.

.. hint::

   **Related lessons:** :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`  ·  :doc:`Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>`  ·  :doc:`Karl Pearson <13-karl-pearson>`  ·  :doc:`Least Squares Regression <31-least-squares-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/measuring-associations-between-two-continuous-variables/ <https://insightful-data-lab.com/2026/01/14/measuring-associations-between-two-continuous-variables/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner
