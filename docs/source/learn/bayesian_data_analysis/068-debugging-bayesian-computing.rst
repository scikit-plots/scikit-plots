.. _bda-debugging-bayesian-computing:

========================================================================
Debugging Bayesian computing
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 068 of 144  ·  *intermediate*

:doc:`◀ Previous · Computing environments <067-computing-environments>`   ·   :doc:`Next · Gibbs sampler ▶ <069-gibbs-sampler>`   ·   :doc:`↑ Section <index>`


Silence is the danger
-----------------------

A broken sampler rarely raises an exception. It returns numbers — plausible, well-formatted, confidently
summarised numbers — that are simply wrong. Bayesian debugging therefore cannot wait for a crash; it
must **manufacture** situations in which the right answer is known.

Fake-data simulation
----------------------

The single most valuable technique. **Choose** parameter values, **simulate** data from your own model,
then fit that data and check whether the posterior recovers the parameters you chose:

.. code-block:: python

   import numpy as np, pymc as pm, arviz as az
   rng = np.random.default_rng(0)
   true_mu, true_sigma = 1.5, 0.8                       # 1. pick the truth
   y_fake = rng.normal(true_mu, true_sigma, size=200)   # 2. simulate from the model

   with pm.Model():                                     # 3. fit the same model
       mu = pm.Normal("mu", 0, 5)
       sigma = pm.HalfNormal("sigma", 5)
       pm.Normal("y", mu, sigma, observed=y_fake)
       idata = pm.sample(random_seed=0)
   az.summary(idata)      # 4. do the intervals contain 1.5 and 0.8?

If the posterior misses the truth, the bug is in the model, the code, or the sampler — and you know
before touching real data. Repeat the loop many times and count coverage, and you have the
**simulation-based calibration** of Stage 4: 50% intervals should contain the truth 50% of the time.

Read the diagnostics, always
------------------------------

Then check what the sampler is telling you. :math:`\hat{R}` near 1.00 (chains agree), adequate
``ess_bulk`` and ``ess_tail``, and — with HMC — **zero divergences**. A divergence is not a warning to
be suppressed: it says the sampler encountered geometry it could not follow, and the region it failed to
explore is systematically **excluded** from your posterior. The eight-schools funnel is the archetype,
and the fix was a reparameterisation, not a longer chain.

The folk theorem
------------------

Gelman's rule of thumb: *when your computation fails, it is usually a problem with your model*. Slow
mixing, divergences and wandering chains most often signal a posterior that is weakly identified,
badly scaled, or improper — not merely an unlucky algorithm.

So the debugging ladder runs: **start simple** and add one component at a time; **fit to fake data**
before real; **check the priors** by simulating from them alone; **rescale** so parameters are of order
one; **reparameterise** (non-centred) when a hierarchy funnels; and only then blame the sampler. Each
rung tells you something about the model, which is the point.

.. hint::

   **Related lessons:** :doc:`Computing environments <067-computing-environments>`  ·  :doc:`Do the Inferences from the Model Make Sense? <041-do-the-inferences-from-the-model-make-sense>`  ·  :doc:`Frequency Evaluations of Bayesian Inferences <031-frequency-evaluations-of-bayesian-inferences>`  ·  :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/debugging-bayesian-computing/ <https://insightful-data-lab.com/2025/11/11/debugging-bayesian-computing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
