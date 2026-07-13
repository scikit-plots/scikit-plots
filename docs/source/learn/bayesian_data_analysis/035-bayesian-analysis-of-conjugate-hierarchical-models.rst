.. _bda-bayesian-analysis-of-conjugate-hierarchical-models:

========================================================================
Bayesian analysis of conjugate hierarchical models
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 035 of 144  ·  *beginner*

:doc:`◀ Previous · Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`   ·   :doc:`Next · Normal model with exchangeable parameters ▶ <036-normal-model-with-exchangeable-parameters>`   ·   :doc:`↑ Section <index>`


The rat tumours
-----------------

The canonical worked example: :math:`J = 71` historical laboratory experiments, each reporting
:math:`y_j` rats developing tumours out of :math:`n_j`. Some experiments are tiny. Estimating each rate
separately gives wild answers (an experiment with 0 of 5 suggests a rate of exactly zero); pooling them
all into one rate denies that experiments differ. Exchangeability says: model the rates as drawn from a
**common population distribution**.

The three-level model
-----------------------

Binomial likelihood, Beta population, hyperprior on the Beta's parameters:

.. math::

   y_j \mid \theta_j \sim \mathrm{Binomial}(n_j, \theta_j), \qquad
   \theta_j \mid \alpha, \beta \sim \mathrm{Beta}(\alpha, \beta), \qquad
   (\alpha, \beta) \sim p(\alpha, \beta) .

Because the Beta is conjugate to the binomial, the **conditional** posterior of each group is immediate:

.. math::

   \theta_j \mid \alpha, \beta, y \;\sim\; \mathrm{Beta}(\alpha + y_j,\; \beta + n_j - y_j),

which makes :math:`\mathrm{E}[\theta_j \mid \alpha, \beta, y]` the familiar weighted average of the
group's own rate :math:`y_j/n_j` and the population mean :math:`\alpha/(\alpha+\beta)` — with the
population now acting as a **prior estimated from all 71 experiments**. Small experiments are pulled
hard toward the population; large ones barely move.

Two levels, two computations
------------------------------

The hyperparameters are handled by marginalising the group parameters analytically (conjugacy again),
leaving a two-dimensional marginal posterior :math:`p(\alpha, \beta \mid y)` that can be evaluated on a
**grid** — exactly the bioassay trick. Then draw :math:`(\alpha, \beta)`, and draw each
:math:`\theta_j` from its conditional Beta. Modern practice simply hands the whole thing to a sampler:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hyperprior on population mean and "prior sample size"
       mu  = pm.Beta("mu", 1, 1)                    # population mean rate
       kap = pm.HalfNormal("kappa", 50)             # concentration = alpha + beta
       theta = pm.Beta("theta", mu * kap, (1 - mu) * kap, shape=J)
       pm.Binomial("y", n=n, p=theta, observed=y)
       idata = pm.sample()

Choosing the hyperprior
-------------------------

One trap deserves naming. A flat prior on :math:`(\alpha, \beta)` is **improper** and yields an
**improper posterior** — the concentration :math:`\alpha + \beta` runs off to infinity. Gelman
reparameterises to the population mean :math:`\alpha/(\alpha+\beta)` and a transformed concentration,
placing a proper prior there. The lesson from Stage 2 returns with teeth: for hierarchical variance and
concentration parameters, **check propriety**, and prefer weakly informative hyperpriors.

.. hint::

   **Related lessons:** :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Constructing a Parameterized Prior Distribution <033-constructing-a-parameterized-prior-distribution>`  ·  :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`  ·  :doc:`Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/bayesian-analysis-of-conjugate-hierarchical-models/ <https://insightful-data-lab.com/2025/11/09/bayesian-analysis-of-conjugate-hierarchical-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
