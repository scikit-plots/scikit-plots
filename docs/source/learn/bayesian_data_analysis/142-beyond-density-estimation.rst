.. _bda-beyond-density-estimation:

========================================================================
Beyond density estimation
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 142 of 144  ·  *advanced*

:doc:`◀ Previous · Dirichlet process mixtures <141-dirichlet-process-mixtures>`   ·   :doc:`Next · Hierarchical dependence ▶ <143-hierarchical-dependence>`   ·   :doc:`↑ Section <index>`


Nonparametrics for other functionals
--------------------------------------

Dirichlet process mixtures were introduced for density estimation, but their reach is far wider. Once you
have a flexible posterior over a **whole distribution**, any **functional** of that distribution inherits
a posterior — so nonparametric Bayes answers questions that are not about the density's shape at all.

Functionals come free
------------------------

A DPM yields posterior draws of the entire distribution :math:`G` (or the implied density). Push each
draw through any functional and you get its posterior automatically — no new model required:

.. math::

   T(G) : \quad \text{mean, variance, quantiles, } \Pr(Y > c), \; \text{entropy, mode count, } \dots

Each posterior draw of :math:`G` gives one draw of :math:`T(G)`, so a **median**, a **tail probability**,
or the **number of modes** arrives with full uncertainty — including uncertainty about the distributional
*shape*, which a parametric model would have suppressed by assuming it away. Estimating a 99th percentile
from a skewed, multimodal distribution is exactly where this pays off.

.. code-block:: python

   import numpy as np
   # each posterior draw is a full mixture -> evaluate any functional per draw
   def functional_posterior(weights_draws, mu_draws, sigma_draws, T):
       return np.array([T(w, m, s)                          # one value per posterior draw
                        for w, m, s in zip(weights_draws, mu_draws, sigma_draws)])
   # e.g. T = tail probability P(Y > c), or a quantile, or the number of modes

Model-based clustering as inference
-------------------------------------

The DPM's **partition** is itself a rich object. Because the model puts a posterior over *how the data
divide into groups*, clustering stops being a point estimate from an algorithm and becomes **inference**:
you get the posterior probability that two points share a cluster, the distribution of the number of
clusters, and a principled way to report clustering **uncertainty** — none of which :math:`k`-means or a
dendrogram provides.

Other nonparametric objects
-----------------------------

The DP is one of a **family**. The **Indian buffet process** gives a nonparametric prior for
**latent-feature** models — objects possessing an unbounded set of overlapping features rather than
belonging to one cluster. **Pólya trees** and **Gaussian processes** are nonparametric priors on
densities and functions. **Survival** and **hazard** functions get nonparametric treatments too. The
unifying theme: put a prior on an **infinite-dimensional** object — a distribution, a function, a feature
matrix — and let the data determine its complexity, with every downstream quantity carrying honest
posterior uncertainty.

.. hint::

   **Related lessons:** :doc:`Dirichlet process mixtures <141-dirichlet-process-mixtures>`  ·  :doc:`Hierarchical dependence <143-hierarchical-dependence>`  ·  :doc:`Bayesian histograms <139-bayesian-histograms>`  ·  :doc:`Mixture models for classification and regression <138-mixture-models-for-classification-and-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/13/beyond-density-estimation/ <https://insightful-data-lab.com/2025/12/13/beyond-density-estimation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
