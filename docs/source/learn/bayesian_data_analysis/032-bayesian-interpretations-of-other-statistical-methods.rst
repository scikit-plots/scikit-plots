.. _bda-bayesian-interpretations-of-other-statistical-methods:

========================================================================
Bayesian interpretations of other statistical methods
========================================================================

**Part 1 · Stage 4 · 📏 Asymptotics & Frequentist Ties**  ·  Lesson 032 of 144  ·  *beginner*

:doc:`◀ Previous · Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>`   ·   :doc:`Next · Constructing a Parameterized Prior Distribution ▶ <033-constructing-a-parameterized-prior-distribution>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Priors in disguise
--------------------

Many familiar non-Bayesian procedures turn out to be Bayesian estimates under some prior. Recognising
this is not point-scoring: it clarifies **what a method assumes**, and it converts tuning parameters
chosen by cross-validation into statements about prior beliefs — which can then be examined, criticised
and improved.

Penalised likelihood is MAP estimation
----------------------------------------

Maximising a **penalised** log-likelihood is exactly finding a **posterior mode**:

.. math::

   \underbrace{\arg\max_{\theta} \; \bigl[\log p(y \mid \theta) + \log p(\theta)\bigr]}_{\text{MAP}}
   \;\;=\;\;
   \underbrace{\arg\min_{\theta} \; \bigl[-\log p(y \mid \theta)
              + \lambda\, \mathrm{pen}(\theta)\bigr]}_{\text{penalised likelihood}} ,

with the penalty as the negative log prior. Two headline cases:

* **Ridge regression** :math:`= ` MAP under a **Gaussian** prior :math:`\beta_j \sim \mathrm{N}(0,
  \tau^2)`, with :math:`\lambda = \sigma^2 / \tau^2`. The :math:`\ell_2` penalty *is*
  :math:`-\log p(\beta)` up to a constant.
* **Lasso** :math:`= ` MAP under a **Laplace** (double-exponential) prior, whose peak at zero produces
  exact zeros at the mode.

Maximum likelihood itself is MAP under a flat prior — which is why, per Bernstein–von Mises, the two
agree asymptotically.

Where the analogy stops
-------------------------

The correspondence is between **point estimates**, not distributions, and that is the catch. The lasso's
sparse solution is a property of the **mode**; the full posterior under a Laplace prior puts **zero**
probability on any coefficient being exactly zero. Reporting the mode alone hides this. The Bayesian
version supplies uncertainty — and reveals that the sparsity was an artifact of the summary. (Modern
Bayesian sparsity uses **horseshoe** or spike-and-slab priors instead.)

.. code-block:: python

   from sklearn.linear_model import Ridge
   import numpy as np
   # Ridge with alpha = sigma^2 / tau^2 reproduces the Gaussian-prior posterior MEAN
   # (for linear-Gaussian models the mode and mean coincide)
   Ridge(alpha=1.0).fit(X, y).coef_

Others in the family
----------------------

The list extends: **shrinkage/empirical-Bayes** estimators (James–Stein) are hierarchical posteriors
with hyperparameters fitted from data; **smoothing splines** are Gaussian-process posteriors with a
roughness prior (Stage 15); **regularised logistic regression** is the weakly-informative prior that
cures separation (Part IV). The Bayesian reading gives each a language for **why** it works: not "the
penalty stabilises the fit", but "here is what the analysis assumes about the world" — a claim that can
be stated, checked, and defended.

.. hint::

   **Related lessons:** :doc:`Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>`  ·  :doc:`Large-Sample Theory <029-large-sample-theory>`  ·  :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`  ·  :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/bayesian-interpretations-of-other-statistical-methods/ <https://insightful-data-lab.com/2025/11/09/bayesian-interpretations-of-other-statistical-methods/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
