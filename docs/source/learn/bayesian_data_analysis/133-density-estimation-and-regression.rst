.. _bda-density-estimation-and-regression:

========================================================================
Density estimation and regression
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 133 of 144  ·  *advanced*

:doc:`◀ Previous · Functional data analysis <132-functional-data-analysis>`   ·   :doc:`Next · Setting up and interpreting mixture models ▶ <134-setting-up-and-interpreting-mixture-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Flexibility from components, not curves
-----------------------------------------

Part V has pursued flexibility through smooth functions — splines, Gaussian processes. The final stage
takes a different route: build flexible distributions and regressions by **combining simple components**.
This lesson frames the two problems the stage solves, and the shift in thinking that unites them.

Two nonparametric problems
----------------------------

**Density estimation** asks for the whole distribution of a variable, :math:`p(y)`, when no parametric
family fits — a distribution with several bumps, heavy skew, or unknown shape. **Density regression**
goes further: it lets the *entire* conditional distribution :math:`p(y \mid x)` change with predictors,
not merely its mean. A standard regression models :math:`\mathrm{E}[y \mid x]` and assumes the rest of
the shape is fixed; density regression frees the variance, the skew, even the number of modes to vary
with :math:`x`.

The mixture idea
------------------

Both are solved by **mixtures**. Any sufficiently complex distribution can be approximated by a **weighted
sum of simple components** — usually normals:

.. math::

   p(y) = \sum_{k=1}^{K} \pi_k \, \mathrm{N}(y \mid \mu_k, \sigma_k^2), \qquad
   \sum_k \pi_k = 1 .

With enough components a mixture of normals can approximate **any** continuous density to arbitrary
accuracy — the theoretical guarantee that makes mixtures a general-purpose nonparametric tool. The
components need not correspond to real subpopulations; they are basis elements for building a shape.

.. code-block:: python

   import numpy as np, pymc as pm
   with pm.Model():
       w = pm.Dirichlet("w", a=np.ones(K))                  # mixture weights, sum to 1
       mu = pm.Normal("mu", 0, 5, shape=K)
       sigma = pm.HalfNormal("sigma", 1, shape=K)
       pm.NormalMixture("y", w=w, mu=mu, sigma=sigma, observed=y)   # flexible density

Two readings, and the plan
----------------------------

Mixtures carry a productive ambiguity. **Model-based clustering** reads each component as a real
**subpopulation** — the mixture *discovers groups*, and the membership of each point is the inference of
interest. **Density estimation** reads the components as mere **building blocks** for an arbitrary shape,
with no claim that they are real. The same model, two purposes; which one you intend governs how you
interpret the fit. The stage builds from finite mixtures (this lesson and the next few) to the case of an
**unknown** number of components, and finally to **infinite** mixtures — the Dirichlet process — where the
component count is itself learned. Flexibility, assembled from simple parts.

.. hint::

   **Related lessons:** :doc:`Functional data analysis <132-functional-data-analysis>`  ·  :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Bayesian histograms <139-bayesian-histograms>`  ·  :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/density-estimation-and-regression/ <https://insightful-data-lab.com/2025/12/09/density-estimation-and-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
