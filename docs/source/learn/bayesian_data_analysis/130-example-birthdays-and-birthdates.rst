.. _bda-example-birthdays-and-birthdates:

========================================================================
Example: birthdays and birthdates
========================================================================

**Part 5 · Stage 15 · 🌊 Basis Functions & Gaussian Processes**  ·  Lesson 130 of 144  ·  *advanced*

:doc:`◀ Previous · Gaussian process regression <129-gaussian-process-regression>`   ·   :doc:`Next · Latent Gaussian process models ▶ <131-latent-gaussian-process-models>`   ·   :doc:`↑ Section <index>`


Decomposing a time series with kernels
----------------------------------------

The number of babies born in the United States each day, over two decades, is a deceptively rich time
series — and the showcase, on the cover of *Bayesian Data Analysis*, for what additive Gaussian-process
kernels can do. The daily count carries several overlapping patterns at different timescales, and the GP
recovers them by **adding kernels**, one per component.

The components
----------------

Births vary on at least four rhythms, each captured by a kernel matched to its shape:

* a **long-term trend** over the years — a slow squared-exponential (large length-scale);
* a **seasonal** pattern within the year (more births in late summer) — a **periodic** kernel with a
  one-year period;
* a **day-of-week** effect (far fewer births on weekends, when scheduled deliveries pause) — a periodic
  kernel with a one-week period;
* **special days** — sharp dips on holidays, a spike or dip on dates like Valentine's Day and a notable
  avoidance of the 13th — short-scale deviations.

The additive model
--------------------

Because a sum of independent GPs is a GP whose kernel is the **sum** of the components' kernels, the whole
decomposition is a single Gaussian process:

.. math::

   f(t) = f_{\text{trend}}(t) + f_{\text{year}}(t) + f_{\text{week}}(t) + f_{\text{special}}(t),
   \qquad
   k = k_{\text{trend}} + k_{\text{year}} + k_{\text{week}} + k_{\text{special}} .

Fitting the sum and then examining each term's posterior **separates** the rhythms — the model discovers
the trend, the seasonal curve and the weekly dip individually, each with uncertainty.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # additive kernel: long trend + yearly + weekly periodic components
       k_trend = pm.HalfNormal("a1", 1)**2 * pm.gp.cov.ExpQuad(1, ls=pm.Gamma("l1", 4, 0.1))
       k_year  = pm.HalfNormal("a2", 1)**2 * pm.gp.cov.Periodic(1, period=365.25,
                                                                ls=pm.Gamma("l2", 2, 1))
       k_week  = pm.HalfNormal("a3", 1)**2 * pm.gp.cov.Periodic(1, period=7,
                                                                ls=pm.Gamma("l3", 2, 1))
       gp = pm.gp.Marginal(cov_func=k_trend + k_year + k_week)     # sum of GPs is a GP
       gp.marginal_likelihood("y", X=t[:, None], y=births, sigma=pm.HalfNormal("s", 1))

What it demonstrates
----------------------

Two lessons close out the GP method. **Kernel composition is a modelling language** — adding and
multiplying kernels expresses structural beliefs (a trend *plus* a cycle *times* a slow modulation)
directly in the covariance, and the posterior teases the pieces apart. And a Gaussian process turns
**time-series decomposition** into inference: trend, seasonality and calendar effects that classical
methods extract by separate procedures fall out of one coherent model, with propagated uncertainty on
every component. The birthday series is the vivid proof that flexible, structured, interpretable models
need not be built from a fixed basis.

.. hint::

   **Related lessons:** :doc:`Gaussian process regression <129-gaussian-process-regression>`  ·  :doc:`Latent Gaussian process models <131-latent-gaussian-process-models>`  ·  :doc:`Unequal variances and correlations <097-unequal-variances-and-correlations>`  ·  :doc:`Functional data analysis <132-functional-data-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/example-birthdays-and-birthdates/ <https://insightful-data-lab.com/2025/12/09/example-birthdays-and-birthdates/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
