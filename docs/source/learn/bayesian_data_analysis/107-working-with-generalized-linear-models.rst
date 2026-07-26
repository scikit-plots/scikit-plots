.. _bda-working-with-generalized-linear-models:

========================================================================
Working with generalized linear models
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 107 of 144  ·  *advanced*

:doc:`◀ Previous · Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`   ·   :doc:`Next · Weakly informative priors for logistic regression ▶ <108-weakly-informative-priors-for-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Fitting is easy; interpreting is the work
--------------------------------------------

A GLM is a one-line change from linear regression to fit. The effort moves to **interpretation** — the
link function makes coefficients nonlinear on the outcome scale — and to the **checks** that catch a
misspecified likelihood.

Interpreting on the right scale
---------------------------------

A coefficient lives on the **link** scale, not the outcome scale, and must be translated.

* **Logistic.** :math:`\beta_j` is a change in **log odds** per unit of :math:`x_j`;
  :math:`e^{\beta_j}` is an **odds ratio**. On the probability scale the effect is *nonlinear* — the same
  :math:`\beta_j` moves the probability a lot near :math:`0.5` and little near the extremes. The
  **divide-by-4 rule** gives a quick upper bound: :math:`\beta_j / 4` is the maximum change in
  probability per unit.
* **Poisson.** :math:`e^{\beta_j}` is a **rate ratio** — a multiplicative effect on the count.

Because effects are nonlinear, a single number rarely captures them; **average predictive comparisons**
and predicted-probability plots communicate far better than a coefficient table.

.. code-block:: python

   import numpy as np
   beta = idata.posterior["beta"].values.reshape(-1, k)
   odds_ratio = np.exp(beta[:, j])                         # logistic: multiplicative on odds
   np.percentile(odds_ratio, [2.5, 97.5])

   # honest effect: predicted-probability difference at representative x, averaged over posterior
   from scipy.special import expit
   p_hi = expit(X_hi @ beta.T); p_lo = expit(X_lo @ beta.T)
   (p_hi - p_lo).mean()                                    # average change in probability

Checking the fit
------------------

The posterior predictive check adapts to the outcome type. For **counts**, compare the observed and
predicted frequency of each value, and especially the number of **zeros** — excess zeros are the classic
sign of a wrong likelihood. For **binary** data, check **calibration**: among cases with predicted
probability near :math:`p`, is the observed rate near :math:`p`?

.. code-block:: python

   import arviz as az
   az.plot_ppc(idata)                     # observed vs predicted outcome distribution
   # counts: does the model reproduce the spike at zero?  binary: is it calibrated?

The recurring failure
-----------------------

Most GLM trouble is the **variance assumption**. The Poisson forces variance to equal the mean; real
counts are usually **overdispersed**, with variance far larger, so Poisson intervals come out much too
narrow. The binomial makes an analogous assumption. Detecting this — variance exceeding what the
likelihood permits — and fixing it with a richer likelihood is the subject two lessons on. First,
though, the priors that make even the basic logistic model behave.

.. hint::

   **Related lessons:** :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`  ·  :doc:`Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>`  ·  :doc:`Do the Inferences from the Model Make Sense? <041-do-the-inferences-from-the-model-make-sense>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/working-with-generalized-linear-models/ <https://insightful-data-lab.com/2025/12/06/working-with-generalized-linear-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
