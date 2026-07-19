.. _dpa-identifying-outliers-using-residuals-and-studentized-residuals:

========================================================================
Identifying Outliers Using Residuals and Studentized Residuals
========================================================================

**Stage 8 · 📊 Model Evaluation**  ·  Lesson 54 of 56  ·  *advanced*

:doc:`◀ Previous · Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>`   ·   :doc:`Next · AUC–ROC Curve: Evaluating Classification Model Performance ▶ <55-auc-roc-curve-evaluating-classification-model-performance>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Evaluating regression fits
----------------------------

For regression models, evaluation is not only an average error — it is also asking whether **individual
points** are being fit sensibly. The tool is the **residual**, :math:`e_i = y_i - \hat{y}_i`, familiar
from least squares. A point whose residual is **far larger** than the rest is an **outlier** in the
response — a case the model badly mispredicts — and finding these is a core diagnostic, one that
scikit-plots visualises with residual plots.

Raw residuals mislead
-----------------------

Raw residuals are an awkward yardstick, for two reasons. Their size depends on the **units** of
:math:`y`, so "large" has no absolute meaning. Worse, they do **not share a common variance**: the
variance of :math:`e_i` is :math:`\sigma^2 (1 - h_{ii})`, where :math:`h_{ii}` is the point's
**leverage** — how unusual its feature values are. A **high-leverage** point *pulls the fitted line
toward itself*, artificially **shrinking** its own residual. The very points most able to distort the
fit are the ones whose raw residuals look most innocent.

Standardising
---------------

The first fix is the **standardized** (internally studentized) residual — the raw residual divided by
its own estimated standard deviation:

.. math::

   r_i = \frac{e_i}{\hat{\sigma}\,\sqrt{1 - h_{ii}}}.

Now every point is on a **common scale** of standard-deviation units, comparable across observations
and datasets, with :math:`|r_i| > 3` a common flag for an outlier.

Studentized residuals
-----------------------

One subtlety remains: a truly extreme point **inflates** :math:`\hat{\sigma}` itself, partially
**masking** its own residual. The **studentized** (externally studentized, or *deleted*) residual
removes the circularity by estimating the error scale **without** observation :math:`i` — refit the
model leaving the point out, and scale by that :math:`\hat{\sigma}_{(i)}`:

.. math::

   t_i = \frac{e_i}{\hat{\sigma}_{(i)}\,\sqrt{1 - h_{ii}}}.

Under the usual assumptions :math:`t_i` follows a **t-distribution**, so the flag becomes a genuine
**statistical test**. In practice, :math:`|t_i| > 2` marks a point worth examining and
:math:`|t_i| > 3` a strong outlier — first check for a **data error**; if the value is real, consider a
robust fit or report it as a notable exception. And when **many** points flag at once, the message is
usually not "bad data" but a **misspecified model** — a missing curve or interaction. Residual
diagnostics evaluate the *model* as much as the points.

.. hint::

   **Related lessons:** :doc:`Least Squares Regression <31-least-squares-regression>`  ·  :doc:`Multiple Linear Regression <32-multiple-linear-regression>`  ·  :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`  ·  :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/16/identifying-outliers-using-residuals-and-studentized-residuals/ <https://insightful-data-lab.com/2026/01/16/identifying-outliers-using-residuals-and-studentized-residuals/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: advanced
