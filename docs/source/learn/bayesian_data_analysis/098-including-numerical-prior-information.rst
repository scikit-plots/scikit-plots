.. _bda-including-numerical-prior-information:

========================================================================
Including numerical prior information
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 098 of 144  ·  *advanced*

:doc:`◀ Previous · Unequal variances and correlations <097-unequal-variances-and-correlations>`   ·   :doc:`Next · Regression coeﬃcients exchangeable in batches ▶ <099-regression-coefficients-exchangeable-in-batches>`   ·   :doc:`↑ Section <index>`


Quantitative prior knowledge
------------------------------

Regression rarely starts from ignorance. A previous study estimated a similar coefficient; theory bounds
an elasticity; physics fixes a sign or a scale. Bayesian regression **incorporates** such knowledge
directly, through informative priors on :math:`\beta` and :math:`\sigma` — turning "we already know
roughly this" from a footnote into part of the model.

Where the numbers come from
-----------------------------

Three honest sources, in decreasing order of comfort:

* **A previous analysis.** If an earlier study reported :math:`\hat{\beta}` with standard error
  :math:`s`, a prior :math:`\beta \sim \mathrm{N}(\hat{\beta}, s^2)` carries that result forward — the
  Bayesian version of accumulating evidence across studies, and the honest alternative to ignoring prior
  work or double-counting it.
* **Substantive constraints.** A rate must be positive; a probability lies in :math:`[0, 1]`; a dose
  response is monotone. Encode these as bounded or ordered priors so the posterior cannot violate what
  is known a priori.
* **Elicited ranges.** An expert's plausible interval becomes a prior with roughly that spread —
  useful, but the softest source, and the one to report most carefully.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # coefficient centred on a previous study's estimate, with its uncertainty
       beta_prev = pm.Normal("beta_prev", mu=0.32, sigma=0.08)   # from prior work
       # a coefficient known to be non-negative on theoretical grounds
       beta_pos = pm.HalfNormal("beta_pos", sigma=1.0)
       mu = alpha + beta_prev * x1 + beta_pos * x2
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

Calibrating strength
----------------------

An informative prior is a claim about how much you know, so its **width** must be defensible. Too tight
and it overrides data it should merely inform; too wide and it contributes nothing. Standardising the
predictors first makes the width interpretable — a :math:`\mathrm{N}(0, 1)` prior on a standardised
coefficient says the effect of a one-SD change is probably within a couple of SDs of the outcome. The
prior predictive check is the test: simulate data from the prior alone and see whether it spans the
plausible and excludes the absurd.

The discipline, and the next step
-----------------------------------

Report the prior and its **source**, always — a prior encoding a citable study is science; a prior tuned
until the answer looks right is not. When the prior is genuinely informative, the posterior blends it
with the data in proportion to their precisions, exactly as combining two datasets would. That blending
has an exact and illuminating algebraic form — a normal prior behaves like a set of extra data points —
which the next lesson makes precise.

.. hint::

   **Related lessons:** :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`  ·  :doc:`Interpreting a normal prior distribution as extra data <101-interpreting-a-normal-prior-distribution-as-extra-data>`  ·  :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`  ·  :doc:`Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/24/including-numerical-prior-information/ <https://insightful-data-lab.com/2025/11/24/including-numerical-prior-information/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
