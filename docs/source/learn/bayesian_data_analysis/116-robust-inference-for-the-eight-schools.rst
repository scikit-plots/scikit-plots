.. _bda-robust-inference-for-the-eight-schools:

========================================================================
Robust inference for the eight schools
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 116 of 144  ·  *advanced*

:doc:`◀ Previous · Posterior inference and computation <115-posterior-inference-and-computation>`   ·   :doc:`Next · Robust regression using t-distributed errors ▶ <117-robust-regression-using-t-distributed-errors>`   ·   :doc:`↑ Section <index>`


Robustness meets the canonical example
----------------------------------------

The eight-schools model has run through this book as the archetype of hierarchical inference. Here it
returns as a test of **robustness**: what happens to the pooling when the population distribution of
school effects is given **heavy tails** instead of the usual normal?

The standard model, recalled
------------------------------

The original assumes the true school effects are drawn from a normal population:

.. math::

   y_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad \theta_j \sim \mathrm{N}(\mu, \tau^2).

The normal population is what drives the shrinkage — every school pulled toward :math:`\mu` by an amount
set by :math:`\tau`. But it also encodes an assumption: that no school's true effect is a genuine
**outlier**. A normal population makes a school far from :math:`\mu` very improbable, so the model shrinks
it hard, whether or not that is warranted.

The robust version
--------------------

Replace the normal population with a :math:`t`:

.. math::

   \theta_j \sim t_\nu(\mu, \tau).

Now the population itself tolerates outliers. A school whose observed effect is genuinely extreme is
**shrunk less** — the heavy-tailed population grants that a true effect can lie far from the centre, so
the model does not force it back as aggressively. Ordinary schools are still pooled as before; only the
apparent outlier is treated differently.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       nu  = pm.Gamma("nu", 2, 0.1)                         # population tail weight
       eta = pm.StudentT("eta", nu=nu, mu=0, sigma=1, shape=8)   # non-centred, heavy-tailed
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma_j, observed=y)

What the comparison shows
---------------------------

Two lessons, both general. First, on the eight-schools data the robust and normal fits are **similar** —
none of the eight effects is a dramatic outlier, so heavy tails change little, which is itself
reassuring: the robust model does not distort an already-well-behaved analysis. Robustness that only
acts when needed is the goal. Second, the value of the exercise is the **sensitivity check** it
constitutes — fitting both and comparing is exactly the prior-robustness discipline of Stage 6, applied
to the population distribution. If the conclusions had diverged, that divergence would itself be the
finding: a signal that one school was driving the result and that the normal assumption was doing more
work than the data justified.

The closing point
-------------------

Robustness completes the hierarchical story. The normal population is a *choice*, and the
:math:`t`-population is the tool for asking how much that choice matters. Fitting the robust version
costs almost nothing and answers a question every hierarchical analysis should ask: are my pooled
estimates a property of the data, or an artefact of assuming no outliers? Stage 14 turns next from
outliers to the other great departure from clean data — **missingness**.

.. hint::

   **Related lessons:** :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Aspects of robustness <113-aspects-of-robustness>`  ·  :doc:`Posterior inference and computation <115-posterior-inference-and-computation>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/robust-inference-for-the-eight-schools/ <https://insightful-data-lab.com/2025/12/07/robust-inference-for-the-eight-schools/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
