.. _bda-dirichlet-process-prior-distributions:

========================================================================
Dirichlet process prior distributions
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 140 of 144  ·  *advanced*

:doc:`◀ Previous · Bayesian histograms <139-bayesian-histograms>`   ·   :doc:`Next · Dirichlet process mixtures ▶ <141-dirichlet-process-mixtures>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A prior over distributions
----------------------------

The finite mixtures of this stage all fixed, or struggled to infer, the number of components. The
**Dirichlet process** removes the bound entirely: it is a prior over **probability distributions
themselves**, supporting a *countably infinite* number of components, of which any finite dataset uses a
finite but data-determined number. It is the foundation of Bayesian nonparametric modelling.

The definition
----------------

A Dirichlet process :math:`\mathrm{DP}(\alpha, H)` is specified by two things: a **base distribution**
:math:`H`, the "mean" around which the random distribution is centred, and a **concentration parameter**
:math:`\alpha > 0`, governing how much a draw deviates from :math:`H`. A draw :math:`G \sim
\mathrm{DP}(\alpha, H)` is itself a (discrete) probability distribution — the DP is a distribution *over
distributions*.

The stick-breaking construction
---------------------------------

The DP becomes concrete through **stick-breaking**, which builds :math:`G` explicitly. Start with a unit
stick; repeatedly break off a Beta-distributed fraction; the pieces are the mixture weights, each
attached to a location drawn from :math:`H`:

.. math::

   \beta_k \sim \mathrm{Beta}(1, \alpha), \quad
   \pi_k = \beta_k \prod_{j=1}^{k-1}(1 - \beta_j), \quad
   \theta_k \sim H, \qquad
   G = \sum_{k=1}^{\infty} \pi_k \, \delta_{\theta_k}.

The weights :math:`\pi_k` **decay** (each break takes a fraction of what remains), so although there are
infinitely many components, a handful carry most of the mass — the reason a finite dataset engages only
finitely many.

.. code-block:: python

   import numpy as np
   def stick_breaking(alpha, K_trunc):                      # truncated DP weights
       betas = np.random.beta(1, alpha, size=K_trunc)
       remaining = np.concatenate([[1.0], np.cumprod(1 - betas)[:-1]])
       return betas * remaining                             # pi_k, summing to ~1

   # small alpha -> few dominant components; large alpha -> many, closer to H

The concentration parameter
-----------------------------

:math:`\alpha` tunes complexity. **Small** :math:`\alpha` concentrates the mass on a **few** components,
so the data are explained by few clusters; **large** :math:`\alpha` spreads it over **many**, approaching
the base distribution :math:`H`. Because :math:`\alpha` controls the expected number of occupied
components, it can be given its own prior and **inferred** — the model learns not just the components but
how many to use.

Why it matters
----------------

The DP dissolves the model-selection problem this stage opened with. There is **no** :math:`K` to choose
or to jump between: the number of components is unbounded a priori and **determined by the data** a
posteriori, growing gracefully as more data arrive. What remains is to attach a likelihood to each
component — placing a smooth kernel at each atom :math:`\theta_k` — which is the **Dirichlet process
mixture** of the next lesson, the workhorse of nonparametric density estimation and clustering.

.. hint::

   **Related lessons:** :doc:`Unspecified number of mixture components <137-unspecified-number-of-mixture-components>`  ·  :doc:`Dirichlet process mixtures <141-dirichlet-process-mixtures>`  ·  :doc:`Bayesian histograms <139-bayesian-histograms>`  ·  :doc:`Bayesian analysis of conjugate hierarchical models <035-bayesian-analysis-of-conjugate-hierarchical-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/13/dirichlet-process-prior-distributions/ <https://insightful-data-lab.com/2025/12/13/dirichlet-process-prior-distributions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
