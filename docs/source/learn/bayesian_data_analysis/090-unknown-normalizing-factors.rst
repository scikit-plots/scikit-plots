.. _bda-unknown-normalizing-factors:

========================================================================
Unknown normalizing factors
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 090 of 144  ·  *intermediate*

:doc:`◀ Previous · Other approximations <089-other-approximations>`   ·   :doc:`Next · Conditional modeling ▶ <091-conditional-modeling>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

When the likelihood has a constant you cannot compute
-------------------------------------------------------

MCMC is celebrated for not needing the posterior's normalising constant, since it cancels in the
Metropolis ratio. But some **likelihoods** are themselves unnormalised:

.. math::

   p(y \mid \theta) = \frac{q(y \mid \theta)}{Z(\theta)}, \qquad
   Z(\theta) = \int q(y \mid \theta) \, dy ,

with :math:`Z(\theta)` intractable — a sum over :math:`2^{n}` configurations for an Ising model or an
undirected graphical model, an integral with no closed form for a spatial point process. Now the
constant **depends on** :math:`\theta` and therefore does **not** cancel:

.. math::

   r = \frac{q(y \mid \theta^{*}) \, p(\theta^{*})}{q(y \mid \theta^{(t-1)}) \, p(\theta^{(t-1)})}
       \cdot \underbrace{\frac{Z(\theta^{(t-1)})}{Z(\theta^{*})}}_{\text{unknown}} .

Such posteriors are called **doubly intractable**: the evidence :math:`p(y)` is intractable, as always,
*and* so is :math:`Z(\theta)` at every step.

The exchange algorithm
------------------------

The elegant solution introduces an **auxiliary dataset**. Propose :math:`\theta^{*}`, then simulate
fresh data :math:`y' \sim p(\cdot \mid \theta^{*})` from the model itself. In the acceptance ratio for
the augmented target, the auxiliary term contributes :math:`Z(\theta^{*}) / Z(\theta^{(t-1)})`, which
**cancels** the offending factor exactly:

.. math::

   r_{\text{ex}} = \frac{q(y \mid \theta^{*}) \, p(\theta^{*}) \, q(y' \mid \theta^{(t-1)})}
                        {q(y \mid \theta^{(t-1)}) \, p(\theta^{(t-1)}) \, q(y' \mid \theta^{*})} .

Every term is computable. Building on Møller and colleagues' auxiliary-variable scheme, Murray,
Ghahramani and MacKay's **exchange algorithm** is asymptotically exact — its only demand is the ability
to draw **exact** samples from the likelihood, which perfect-sampling algorithms provide for several
model classes.

Estimating the constant instead
---------------------------------

Where exact sampling is impossible, estimate :math:`Z(\theta)` — or, better, **ratios** of it.
Gelman and Meng's unifying account runs from **importance sampling** through **bridge sampling** to
**path sampling**, the last computing :math:`\log Z(\theta_1) - \log Z(\theta_0)` as an integral of an
expected score along a path connecting the two parameter values. A separate route is **pseudo-marginal**
MCMC: substitute an **unbiased estimate** of the likelihood into the Metropolis ratio, and — the
surprising theorem — the chain still targets the **exact** posterior, at the cost of extra variance.

.. code-block:: python

   import numpy as np
   # exchange step: cancel Z(theta) using auxiliary data drawn from the model
   th_prop = propose(th)
   y_aux = simulate_exactly(th_prop)                       # exact draw from p(. | th_prop)
   log_r = (log_q(y, th_prop) - log_q(y, th)
            + log_q(y_aux, th) - log_q(y_aux, th_prop)     # <- the Z's cancel here
            + log_prior(th_prop) - log_prior(th))
   th = th_prop if np.log(np.random.rand()) < log_r else th

Why it closes Part III
------------------------

This is where the computational story reaches its edge. Parts I and II assumed a likelihood you could
evaluate; Part III has been about integrating one. Models whose likelihood you cannot even *evaluate*
demand a different bargain — auxiliary variables, unbiased estimators, or simulation-based inference —
and each buys correctness with either exact sampling or extra Monte Carlo noise. With the machinery
established, Part IV returns to modelling: regression, and the structure that makes models useful.

.. hint::

   **Related lessons:** :doc:`Other approximations <089-other-approximations>`  ·  :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`  ·  :doc:`Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>`  ·  :doc:`Importance sampling <065-importance-sampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/unknown-normalizing-factors/ <https://insightful-data-lab.com/2025/11/23/unknown-normalizing-factors/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
