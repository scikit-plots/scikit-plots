.. _bda-inference-and-assessing-convergence:

========================================================================
Inference and assessing convergence
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 072 of 144  ·  *intermediate*

:doc:`◀ Previous · Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`   ·   :doc:`Next · Eﬀective number of simulation draws ▶ <073-effective-number-of-simulation-draws>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The two questions
-------------------

MCMC draws are neither independent nor, at first, from the posterior. So two questions must be answered
before any summary is trusted. **Has the chain converged** — is it now sampling the target rather than
still travelling toward it? And **how much information** do its correlated draws actually carry?

Warm-up, and many chains
--------------------------

Discard an initial **warm-up** phase (the sampler is also adapting its step size there). Then run
**several chains from dispersed starting points**: a single chain stuck in one mode looks perfectly
healthy from the inside. Convergence is diagnosed by **disagreement between chains**, which requires
more than one.

R-hat
-------

The classical diagnostic compares **between-chain** to **within-chain** variance: if the chains have
forgotten where they started and are exploring the same distribution, the pooled variance is no larger
than the average within-chain variance, and

.. math::

   \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^{+}(\theta \mid y)}{W}} \;\longrightarrow\; 1 .

Vehtari, Gelman, Simpson, Carpenter and Bürkner showed the original :math:`\hat{R}` has **serious
flaws** — it fails to detect trouble when a chain has a **heavy tail** or when the **variance differs
across chains**. Their replacement **rank-normalises** the draws (robust to heavy tails), **folds** them
about the median to catch differing scales, and **splits** each chain in half to catch drift within a
chain. Modern practice: require :math:`\hat{R} < 1.01`, not the old 1.1.

Effective sample size
-----------------------

Autocorrelation means :math:`S` draws carry the information of :math:`S_{\text{eff}} < S` independent
ones. Report **bulk-ESS** (governing the mean and sd) and **tail-ESS** (governing the 5% and 95%
quantiles) separately, and require **both above about 400** — a chain can mix well in the middle and
crawl in the tails.

.. code-block:: python

   import arviz as az
   az.summary(idata)                 # r_hat, ess_bulk, ess_tail, mcse_mean, mcse_sd
   az.plot_rank(idata)               # rank plots: uniform bars = chains agree
   idata.sample_stats["diverging"].sum()    # HMC: must be zero

Prefer **rank plots** to trace plots: with many chains the "fat hairy caterpillar" is unreadable, whereas
rank histograms should be uniform across chains if all are sampling the same distribution.

What convergence is not
-------------------------

These diagnostics are **necessary, not sufficient**. They can only detect the failures the chains
happened to reveal: a mode never visited by any chain leaves no trace in :math:`\hat{R}`. And a converged
sampler faithfully reproduces the posterior of a model that may be **wrong** — computation and model
adequacy are separate questions, checked by separate tools. Convergence buys you the right posterior for
the model you actually wrote.

.. hint::

   **Related lessons:** :doc:`Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>`  ·  :doc:`Debugging Bayesian computing <068-debugging-bayesian-computing>`  ·  :doc:`How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>`  ·  :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/inference-and-assessing-convergence/ <https://insightful-data-lab.com/2025/11/12/inference-and-assessing-convergence/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
