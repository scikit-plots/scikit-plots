.. _bda-interpreting-a-normal-prior-distribution-as-extra-data:

========================================================================
Interpreting a normal prior distribution as extra data
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 101 of 144  ·  *advanced*

:doc:`◀ Previous · Example: forecasting U.S. presidential elections <100-example-forecasting-u-s-presidential-elections>`   ·   :doc:`Next · Varying intercepts and slopes ▶ <102-varying-intercepts-and-slopes>`   ·   :doc:`↑ Section <index>`


A prior is imaginary data
---------------------------

The previous lesson added quantitative prior knowledge through a prior on :math:`\beta`. There is an
exact algebraic identity underneath it, and it is one of the most clarifying facts in Bayesian
regression: **a normal prior on the coefficients is equivalent to a set of extra data points** appended
to the regression.

The augmented-data identity
-----------------------------

Take the linear model with a normal prior :math:`\beta \sim \mathrm{N}(\beta_0, \Sigma_0)`. Its
posterior mode is **identical** to the least-squares fit of an augmented dataset — the real observations,
plus one pseudo-observation per prior constraint:

.. math::

   \bar{X} = \begin{bmatrix} X \\ \Sigma_0^{-1/2} \end{bmatrix}, \qquad
   \bar{y} = \begin{bmatrix} y \\ \Sigma_0^{-1/2}\beta_0 \end{bmatrix},
   \qquad
   \bar{X}^{\top}\bar{X} = \underbrace{X^{\top}X}_{\text{data}} + \underbrace{\Sigma_0^{-1}}_{\text{prior}} .

The precision decomposes into a data term and a prior term that simply **add**. Each augmented row is one
imaginary observation stating "at this design point the response was :math:`\beta_0`", carrying a
precision set by the prior. The special case :math:`\beta_0 = 0`, :math:`\Sigma_0 = (\sigma^2/\lambda) I`
recovers **ridge regression** exactly — appending :math:`\sqrt{\lambda}\, e_j` rows with response zero.

.. code-block:: python

   import numpy as np
   # prior beta ~ N(beta0, diag(tau^2)) as pseudo-observations, then plain least squares
   P = np.diag(1.0 / tau)                            # Sigma0^{-1/2}
   X_aug = np.vstack([X, P])
   y_aug = np.concatenate([y, P @ beta0])
   beta_post_mode = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]   # == posterior mode

Why the picture helps
-----------------------

It makes the **strength** of a prior concrete. A tight prior is *many* pseudo-observations; a vague one
is *few*; their precisions add to the data's exactly as a second dataset would. It demystifies
regularisation — ridge, and its relatives, are priors and nothing more. And it clarifies the balance of
evidence: where the real data are informative about a coefficient, they swamp the handful of pseudo-rows;
where they are silent (collinearity, few observations), the prior carries the estimate, which is
precisely when you want it to.

The counting interpretation
-----------------------------

The identity even assigns the prior an **effective sample size**. A prior with precision
:math:`\Sigma_0^{-1}` contributes as much information as that many real observations at the corresponding
design points, so "how strong is this prior?" has a literal answer in units of data. That is the honest
way to feel the weight of a prior — and the honest warning against a prior so tight it silently adds
hundreds of observations you never collected.

.. hint::

   **Related lessons:** :doc:`Including numerical prior information <098-including-numerical-prior-information>`  ·  :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`  ·  :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`  ·  :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/24/interpreting-a-normal-prior-distribution-as-extra-data/ <https://insightful-data-lab.com/2025/11/24/interpreting-a-normal-prior-distribution-as-extra-data/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
