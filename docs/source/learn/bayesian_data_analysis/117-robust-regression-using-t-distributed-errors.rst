.. _bda-robust-regression-using-t-distributed-errors:

========================================================================
Robust regression using t-distributed errors
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 117 of 144  ·  *advanced*

:doc:`◀ Previous · Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`   ·   :doc:`Next · Notation ▶ <118-notation>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Regression that ignores outliers
----------------------------------

Ordinary regression puts a **normal** on the residuals, so a single aberrant point drags the whole line
toward it — least squares is exquisitely sensitive to outliers in :math:`y`. **Robust regression**
replaces the normal error with a **Student-**:math:`t`, and the fix is a one-line change with an
outsized effect on reliability.

The model
-----------

Keep the linear predictor; change only the error distribution:

.. math::

   y_i \sim t_\nu\bigl(X_i \beta, \; \sigma\bigr).

The degrees of freedom :math:`\nu` set the tail weight, and can be **estimated** so the data report their
own outlier-proneness. Everything else — priors on :math:`\beta`, hierarchical extensions, the sampler —
is unchanged from the normal model; only the likelihood's tails have thickened.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       sigma = pm.HalfNormal("sigma", 1)
       nu = pm.Gamma("nu", 2, 0.1)                          # estimate tail weight; Gamma keeps nu > 0
       pm.StudentT("y", nu=nu, mu=X @ beta, sigma=sigma, observed=y)
       idata = pm.sample()

Why it works, and what to watch
---------------------------------

The mechanism is the scale-mixture of the previous lesson: a :math:`t` residual is a normal whose
variance can inflate for a single point, so an outlier is granted a **large private variance** and pulls
on the coefficients only weakly. The line is fitted to the **bulk** of the data. Because :math:`\nu` is
estimated, the model interpolates: heavy tails when outliers are present, near-normal when they are not.

Two cautions carry the honest limits. Robustness is against outliers in :math:`y`, **not in the
predictors** — a high-leverage point at an extreme :math:`x` can still distort the fit, because it sits
where the line has little data to argue with it. And down-weighting is not deletion: if an "outlier" is a
real signal (a regime change, a rare but valid case), the :math:`t` will quietly discount evidence you
might have wanted. Robust regression is the right default for messy continuous outcomes — reach for it
before least squares on real data — but it guards against one failure mode, not all of them.

.. hint::

   **Related lessons:** :doc:`Posterior inference and computation <115-posterior-inference-and-computation>`  ·  :doc:`Overdispersed versions of standard models <114-overdispersed-versions-of-standard-models>`  ·  :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`  ·  :doc:`Aspects of robustness <113-aspects-of-robustness>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/robust-regression-using-t-distributed-errors/ <https://insightful-data-lab.com/2025/12/07/robust-regression-using-t-distributed-errors/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
