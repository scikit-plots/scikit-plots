.. _bda-summarizing-posterior-inference:

========================================================================
Summarizing Posterior Inference
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 013 of 144  ·  *beginner*

:doc:`◀ Previous · Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`   ·   :doc:`Next · Informative Prior Distributions ▶ <014-informative-prior-distributions>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The posterior is the answer
-----------------------------

Formally, the whole answer of a Bayesian analysis is the posterior **distribution**. But a distribution
is hard to report and harder to act on, so we summarise it — carefully, because every summary discards
information, and some discard the wrong information.

Point summaries
-----------------

Three point estimates are standard, and they differ once the posterior is **skewed**:

* the **posterior mean** :math:`\mathrm{E}[\theta \mid y]` — the Bayes estimator under squared-error
  loss, and the usual default;
* the **posterior median** — robust, and optimal under absolute-error loss;
* the **posterior mode** (the **MAP** estimate) — the density's peak, and the point where Bayesian
  estimation coincides with penalised maximum likelihood.

For a symmetric unimodal posterior all three agree. When they disagree, the disagreement itself is
information: the posterior is **asymmetric**, and no single number represents it.

Intervals: two kinds
----------------------

A **credible interval** is any region holding a stated posterior probability — say 95%. Two
constructions dominate:

* the **equal-tailed interval (ETI)**, cutting 2.5% from each tail (the 2.5th and 97.5th percentiles);
* the **highest-density interval (HDI/HPD)**, the **shortest** interval containing 95% of the mass —
  equivalently, every point inside has higher density than every point outside.

They coincide for symmetric posteriors. For **skewed** ones they do not, and the ETI can include
parameter values *less* plausible than some it excludes; the HDI always contains the **mode**.

.. code-block:: python

   import numpy as np, arviz as az
   draws = ...                                  # posterior draws
   draws.mean(), np.median(draws)               # point summaries
   np.percentile(draws, [2.5, 97.5])            # equal-tailed 95% CI
   az.hdi(draws, hdi_prob=0.95)                 # highest-density 95% CI
   (draws > 0.5).mean()                         # P(theta > 0.5 | y)

Say what you mean
-------------------

The interpretive prize is real: a 95% credible interval supports the statement *"there is a 95%
probability :math:`\theta` lies in this range"* — precisely what a frequentist **confidence** interval
does **not** license. But do not let the summary replace the distribution: plot the posterior, and
report the probability of the hypotheses you actually care about, which the draws give directly.

.. hint::

   **Related lessons:** :doc:`Estimating a Probability from Binomial Data <011-estimating-a-probability-from-binomial-data>`  ·  :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`  ·  :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/summarizing-posterior-inference/ <https://insightful-data-lab.com/2025/11/09/summarizing-posterior-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
