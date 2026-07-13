.. _bda-analysis-of-variance-and-the-batching-of-coefficients:

========================================================================
Analysis of variance and the batching of coeﬃcients
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 104 of 144  ·  *advanced*

:doc:`◀ Previous · Computation: batching and transformation <103-computation-batching-and-transformation>`   ·   :doc:`Next · Hierarchical models for batches of variance components ▶ <105-hierarchical-models-for-batches-of-variance-components>`   ·   :doc:`↑ Section <index>`


ANOVA, rebuilt as a hierarchy
-------------------------------

Classical analysis of variance partitions variation among **sources** — main effects, interactions,
residual — and tests each with an :math:`F` statistic. The Bayesian view keeps the partition but changes
its meaning: each source of variation is a **batch of exchangeable coefficients** with its own variance
component, and ANOVA becomes the estimation of those variances.

The reframing
---------------

A two-way layout with factors :math:`A` and :math:`B` and their interaction becomes

.. math::

   y_i = \mu + a_{j[i]} + b_{k[i]} + (ab)_{jk[i]} + \epsilon_i,

with each batch drawn from its own distribution,

.. math::

   a_j \sim \mathrm{N}(0, \sigma_A^2), \quad
   b_k \sim \mathrm{N}(0, \sigma_B^2), \quad
   (ab)_{jk} \sim \mathrm{N}(0, \sigma_{AB}^2), \quad
   \epsilon_i \sim \mathrm{N}(0, \sigma_y^2).

The classical **mean squares** are essentially estimates of these variance components. But where ANOVA
asks "is :math:`\sigma_A^2` exactly zero?" — a point null — the hierarchical model **estimates each
:math:`\sigma`**, which is almost always the more useful quantity. How large the batch of effects is
matters more than whether it is precisely null.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5)
       sA = pm.HalfNormal("sA", 1); sB = pm.HalfNormal("sB", 1); sAB = pm.HalfNormal("sAB", 1)
       a  = pm.Normal("a", 0, sA, shape=nA)                 # batch A
       b  = pm.Normal("b", 0, sB, shape=nB)                 # batch B
       ab = pm.Normal("ab", 0, sAB, shape=(nA, nB))         # interaction batch
       mu_i = mu + a[jA] + b[jB] + ab[jA, jB]
       pm.Normal("y", mu_i, pm.HalfNormal("sy", 1), observed=y)
       # posteriors for sA, sB, sAB ARE the analysis of variance

Finite and superpopulation variance
--------------------------------------

Gelman's framing sharpens what a variance component *means*. The **superpopulation** standard deviation
describes the distribution from which the batch of effects is drawn — relevant to a *new*, unobserved
level. The **finite-population** standard deviation is the actual spread of the effects you *have* — the
:math:`nA` levels in this study. They differ when the batch is small, and the distinction answers a real
question: are you generalising to new levels, or summarising these?

Why the batched view wins
---------------------------

Three advantages over the :math:`F`-test tradition. It handles **unbalanced** and **complex** designs
uniformly, where classical ANOVA formulas fragment. It gives **partial pooling** of the effects within
each batch, stabilising estimates that classical ANOVA leaves noisy. And it reports **magnitudes with
uncertainty** rather than a reject/retain verdict on a null nobody believes exactly. ANOVA was never
really about testing; it was about **decomposing variation**, and the hierarchical model does that
directly — the same batching that ran through this entire stage, applied to designed factors.

.. hint::

   **Related lessons:** :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`  ·  :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`  ·  :doc:`Hierarchical models for batches of variance components <105-hierarchical-models-for-batches-of-variance-components>`  ·  :doc:`Computation: batching and transformation <103-computation-batching-and-transformation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/analysis-of-variance-and-the-batching-of-coe%ef%ac%83cients/ <https://insightful-data-lab.com/2025/12/06/analysis-of-variance-and-the-batching-of-coe%ef%ac%83cients/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
