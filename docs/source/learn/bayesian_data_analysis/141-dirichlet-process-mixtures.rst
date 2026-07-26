.. _bda-dirichlet-process-mixtures:

========================================================================
Dirichlet process mixtures
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 141 of 144  ·  *advanced*

:doc:`◀ Previous · Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`   ·   :doc:`Next · Beyond density estimation ▶ <142-beyond-density-estimation>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Putting a likelihood on the process
-------------------------------------

A Dirichlet process draw is a **discrete** distribution — a countable set of atoms — so it cannot model
continuous data directly. The **Dirichlet process mixture** (DPM) fixes this by using the DP not for the
data but for the **parameters**: each atom :math:`\theta_k` is the parameter of a smooth kernel, and the
data are drawn from the resulting infinite mixture. This is *the* workhorse of Bayesian nonparametrics.

The model
-----------

Draw a random distribution from the DP; draw each observation's parameter from it; draw the observation
from a kernel at that parameter:

.. math::

   G \sim \mathrm{DP}(\alpha, H), \qquad
   \theta_i \sim G, \qquad
   y_i \sim f(y \mid \theta_i).

Because :math:`G` is discrete, the :math:`\theta_i` **repeat** — several observations share the same
drawn value — and points sharing a :math:`\theta_i` form a **cluster**. With a normal kernel this is an
infinite mixture of Gaussians whose number of occupied components is learned, dissolving the
:math:`K`-selection problem entirely.

The Chinese restaurant process
--------------------------------

Integrating :math:`G` out gives the **Chinese restaurant process**, the DPM's computational heart and its
clearest intuition. Customers (observations) enter a restaurant and choose tables (clusters): a new
customer joins an existing table with probability proportional to how many already sit there, and starts
a **new** table with probability proportional to :math:`\alpha`:

.. math::

   \Pr(\text{join table } k) \propto n_k, \qquad
   \Pr(\text{new table}) \propto \alpha .

This "rich get richer" rule produces a few large clusters and a tail of small ones, and — crucially —
lets the number of clusters **grow with the data**. It is the basis of the collapsed Gibbs samplers
(Neal's algorithms) that fit DPMs by reseating one customer at a time.

.. code-block:: python

   import pymc as pm
   K = 20                                                   # truncation level (generous)
   with pm.Model():
       alpha = pm.Gamma("alpha", 1, 1)                      # concentration, inferred
       beta = pm.Beta("beta", 1, alpha, shape=K)            # stick-breaking fractions
       w = pm.Deterministic("w", beta * pm.math.concatenate(
           [[1.0], pm.math.cumprod(1 - beta)[:-1]]))        # mixture weights
       mu = pm.Normal("mu", 0, 5, shape=K)                  # atom locations ~ H
       pm.NormalMixture("y", w=w, mu=mu, sigma=pm.HalfNormal("s", 1, shape=K), observed=y)

Why it matters
----------------

The DPM delivers what the stage promised: **density estimation and clustering with the number of
components learned, not chosen**. It adapts complexity to the data — more data can reveal more clusters —
and gives a full posterior over partitions, honestly expressing uncertainty about *how many* groups there
are and *which* points belong together. In practice a **truncated** stick-breaking (a generous finite
:math:`K` with most weights near zero) makes it fit with standard HMC. From here the nonparametric idea
extends outward: to functionals beyond the density, to shared clustering across groups, and to
covariate-dependent distributions — the final three lessons.

.. hint::

   **Related lessons:** :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`  ·  :doc:`Beyond density estimation <142-beyond-density-estimation>`  ·  :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Density regression <144-density-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/13/dirichlet-process-mixtures/ <https://insightful-data-lab.com/2025/12/13/dirichlet-process-mixtures/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
