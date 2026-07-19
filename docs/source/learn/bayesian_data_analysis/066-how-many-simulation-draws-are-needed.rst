.. _bda-how-many-simulation-draws-are-needed:

========================================================================
How many simulation draws are needed?
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 066 of 144  ·  *intermediate*

:doc:`◀ Previous · Importance sampling <065-importance-sampling>`   ·   :doc:`Next · Computing environments ▶ <067-computing-environments>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Monte Carlo error is not posterior uncertainty
------------------------------------------------

Two uncertainties coexist in a simulation-based analysis. The **posterior** standard deviation
:math:`\sigma_{\theta}` expresses what the data leave unknown — it cannot be reduced by computing
harder. The **Monte Carlo standard error** expresses how imprecisely your finite sample of draws
estimates a posterior summary — and it shrinks as you run longer:

.. math::

   \mathrm{MCSE}(\bar{\theta}) = \frac{\sigma_{\theta}}{\sqrt{S_{\text{eff}}}} .

Note :math:`S_{\text{eff}}`, not :math:`S`: MCMC draws are **autocorrelated**, so a thousand draws may
carry the information of a hundred.

Fewer than you think, for the mean
------------------------------------

The classic argument is bracing. Suppose you estimate the posterior mean with :math:`S = 100`
independent draws. The MCSE is :math:`\sigma_{\theta}/10`, so the *total* uncertainty about
:math:`\theta` — posterior plus simulation — inflates from :math:`\sigma_{\theta}` to
:math:`\sigma_{\theta}\sqrt{1 + 1/100} \approx 1.005 \, \sigma_{\theta}`. **A 0.5% increase.** For
reporting a posterior mean and interval, a hundred effective draws is already enough, and the fourth
decimal place of a posterior mean was never meaningful anyway.

More than you think, for tails
--------------------------------

The picture changes for quantities that depend on **rare** draws. A 2.5% quantile is estimated from the
draws in that tail; a probability like :math:`\Pr(\theta > c \mid y) = 0.001` requires enough draws to
see the event repeatedly. As a working rule, modern practice targets :math:`S_{\text{eff}} \gtrsim 400`
per quantity of interest — enough for stable tail quantiles and for the convergence diagnostics of Stage
9 to be trustworthy themselves.

.. code-block:: python

   import arviz as az
   az.summary(idata)     # columns: mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat
   # ess_bulk governs the mean/sd; ess_tail governs the 5%/95% quantiles.
   # Report a number only to the precision its MCSE supports.

The discipline
----------------

Three habits follow. **Report MCSE**, or at least check it: a posterior mean of 2.43 with an MCSE of
0.05 should be written as 2.4. **Check ess_tail separately** from ess_bulk, because a chain that mixes
well in the middle can crawl in the tails. And remember what more draws cannot buy: they shrink Monte
Carlo error toward zero and leave posterior uncertainty exactly where it was. If the interval is too
wide to act on, the remedy is more **data** or a better **model**, never a longer chain.

.. hint::

   **Related lessons:** :doc:`Importance sampling <065-importance-sampling>`  ·  :doc:`Numerical integration <062-numerical-integration>`  ·  :doc:`Eﬀective number of simulation draws <073-effective-number-of-simulation-draws>`  ·  :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/how-many-simulation-draws-are-needed/ <https://insightful-data-lab.com/2025/11/11/how-many-simulation-draws-are-needed/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
