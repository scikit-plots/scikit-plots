.. _bda-hierarchical-modeling-applied-to-a-meta-analysis:

========================================================================
Hierarchical modeling applied to a meta-analysis
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 038 of 144  ·  *beginner*

:doc:`◀ Previous · Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`   ·   :doc:`Next · Weakly Informative Priors for Variance Parameters ▶ <039-weakly-informative-priors-for-variance-parameters>`   ·   :doc:`↑ Section <index>`


The same model, a different name
----------------------------------

**Meta-analysis** combines results from separate studies of the same question — a drug trial run in
twelve hospitals, a treatment tested in a dozen papers. Each study reports an effect estimate and a
standard error. That is precisely the eight-schools structure, and the Bayesian hierarchical model is
the **random-effects meta-analysis** familiar to medicine, arrived at from first principles rather than
by convention.

The model
-----------

Study :math:`j` estimates :math:`y_j` of its own true effect :math:`\theta_j`, with within-study error
:math:`\sigma_j` treated as known; the true effects are exchangeable draws around a population effect:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2).

Here :math:`\mu` is the **overall effect** and :math:`\tau` the **between-study heterogeneity**. Classical
meta-analysis calls :math:`\tau = 0` the *fixed-effect* model and :math:`\tau > 0` the *random-effects*
model, then chooses between them by a test. The Bayesian analysis does not choose: it **estimates**
:math:`\tau` and averages over its uncertainty.

Why that matters
------------------

Two payoffs follow. First, the pooled effect's interval **widens honestly** when studies disagree,
because uncertainty in :math:`\tau` propagates into :math:`\mu` — a plug-in :math:`\hat{\tau}` (the
DerSimonian–Laird estimator, say) understates it, and badly so when :math:`J` is small. Second, each
study gets a **shrunken** estimate :math:`\theta_j`, so a small outlying trial is not read literally.
And a genuinely new quantity comes free: the **predictive** effect in a *future* study,
:math:`\tilde{\theta} \sim \mathrm{N}(\mu, \tau^2)`, which is what a clinician deciding for the next
patient actually needs — always wider than the interval for :math:`\mu`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 1)                  # scale: log odds ratio
       tau = pm.HalfNormal("tau", 0.5)              # heterogeneity, weakly informative
       eta = pm.Normal("eta", 0, 1, shape=J)
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=se, observed=y)  # se = within-study standard errors
       # effect in a new study:
       theta_new = pm.Normal("theta_new", mu, tau)
       idata = pm.sample(target_accept=0.9)

The assumption worth interrogating
------------------------------------

Everything rests on **exchangeability of the studies**. If trials differ systematically — dose,
population, decade, industry funding — they are not exchangeable as they stand, and pooling them
averages apples with oranges. The repair is the one from the exchangeability lesson: model the known
differences as **covariates** (a hierarchical *meta-regression*) and assume exchangeability of what
remains. Publication bias is a harder problem still: it corrupts the likelihood itself, and no prior
fixes a biased sample of studies.

.. hint::

   **Related lessons:** :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`  ·  :doc:`Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/hierarchical-modeling-applied-to-a-meta-analysis/ <https://insightful-data-lab.com/2025/11/10/hierarchical-modeling-applied-to-a-meta-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
