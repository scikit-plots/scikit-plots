.. _bda-direct-simulation-and-rejection-sampling:

========================================================================
Direct simulation and rejection sampling
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 064 of 144  ·  *intermediate*

:doc:`◀ Previous · Distributional approximations <063-distributional-approximations>`   ·   :doc:`Next · Importance sampling ▶ <065-importance-sampling>`   ·   :doc:`↑ Section <index>`


When you can draw directly
----------------------------

Some posteriors can be sampled **exactly**, with no iteration. Conjugate models are the obvious case —
`stats.beta(a + y, b + n - y).rvs()` is an exact posterior draw. So are models that **factor**: the
normal model with unknown variance draws :math:`\sigma^2` from its marginal, then :math:`\mu` from its
conditional, giving exact joint draws. Direct simulation is the gold standard: independent draws, no
convergence to diagnose.

Rejection sampling
--------------------

When direct draws are unavailable but the **unnormalised** density :math:`q(\theta) \propto p(\theta
\mid y)` can be evaluated, rejection sampling manufactures exact draws from a proposal
:math:`g(\theta)` you *can* sample. The requirement is an **envelope**: a constant :math:`M` with
:math:`q(\theta) \le M \, g(\theta)` everywhere.

The algorithm is three lines. Draw :math:`\theta^{*} \sim g`; draw :math:`u \sim \mathrm{Uniform}(0,1)`;
**accept** :math:`\theta^{*}` if

.. math::

   u \;\le\; \frac{q(\theta^{*})}{M \, g(\theta^{*})} .

Accepted draws are **exact** samples from the target — no approximation, no burn-in. The acceptance
probability is :math:`1/M` (for normalised :math:`q`), so :math:`M` measures the waste.

.. code-block:: python

   import numpy as np
   from scipy import stats

   g = stats.t(df=4, loc=mode, scale=scale)             # heavy-tailed proposal, must cover target
   M = np.exp((log_q(grid) - g.logpdf(grid)).max())     # envelope constant, found numerically
   out = []
   while len(out) < 1000:
       th = g.rvs()
       if np.random.rand() <= np.exp(log_q(th) - np.log(M) - g.logpdf(th)):
           out.append(th)
   accept_rate = 1000 / n_proposed                       # ≈ 1/M

Two failure modes
-------------------

If the envelope condition is **violated** anywhere — a proposal with lighter tails than the target — the
draws are silently **wrong**, not merely inefficient. Hence the standing advice to use a heavy-tailed
proposal (a :math:`t`, not a normal), and to find :math:`M` by maximising the log-ratio rather than
guessing.

And even a valid envelope becomes useless in high dimensions: the acceptance rate falls **exponentially**
with :math:`d`, because a proposal that is a decent match in each coordinate is a poor match in all of
them at once. Rejection sampling survives as a component — for univariate draws, for truncated
distributions (reject anything outside the support) — but not as a general engine. That role belongs to
**MCMC**, which abandons independent draws in exchange for scaling.

.. hint::

   **Related lessons:** :doc:`Numerical integration <062-numerical-integration>`  ·  :doc:`Importance sampling <065-importance-sampling>`  ·  :doc:`Distributional approximations <063-distributional-approximations>`  ·  :doc:`Gibbs sampler <069-gibbs-sampler>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/direct-simulation-and-rejection-sampling/ <https://insightful-data-lab.com/2025/11/11/direct-simulation-and-rejection-sampling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
