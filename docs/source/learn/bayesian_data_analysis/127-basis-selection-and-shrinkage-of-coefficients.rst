.. _bda-basis-selection-and-shrinkage-of-coefficients:

========================================================================
Basis selection and shrinkage of coeﬃcients
========================================================================

**Part 5 · Stage 15 · 🌊 Basis Functions & Gaussian Processes**  ·  Lesson 127 of 144  ·  *advanced*

:doc:`◀ Previous · Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>`   ·   :doc:`Next · Non-normal models and regression surfaces ▶ <128-non-normal-models-and-regression-surfaces>`   ·   :doc:`↑ Section <index>`


How smooth should the curve be?
---------------------------------

A basis expansion poses one question: how many functions, or equivalently how wiggly a fit? Choosing the
knot count by hand or by cross-validation is crude. The Bayesian answer inverts the problem — use
**many** basis functions and let a **prior on the coefficients** determine the effective smoothness,
so the data, not a discrete choice, set the flexibility.

The penalized-spline idea
---------------------------

Lay down a rich basis — more knots than you could need — and place a prior that **penalises roughness**.
The standard device is a prior on the **differences** between adjacent coefficients: if neighbours are
tied together, the curve cannot wiggle sharply.

.. math::

   \beta_k - \beta_{k-1} \sim \mathrm{N}(0, \tau^2), \quad\text{equivalently}\quad
   \beta_k \sim \mathrm{N}(\beta_{k-1}, \tau^2),

a **random-walk** prior on the coefficients. The smoothing scale :math:`\tau` is the one knob, and —
this is the elegant part — it is **estimated from the data** as an ordinary variance parameter. Small
:math:`\tau` forces a smooth curve; large :math:`\tau` permits wiggliness; its posterior finds the
balance.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       tau = pm.HalfNormal("tau", 1)                        # smoothing scale, ESTIMATED
       z = pm.Normal("z", 0, 1, shape=B.shape[1])
       beta = pm.Deterministic("beta", pm.math.cumsum(z * tau))   # random-walk (P-spline) prior
       pm.Normal("y", B @ beta, pm.HalfNormal("s", 1), observed=y)

Smoothness as a variance component
------------------------------------

The reframing is the whole point: **the amount of smoothing is a variance parameter**, so choosing it
becomes *estimating* it, with all the hierarchical machinery of this book. Adaptive smoothing —
different roughness in different regions — follows by letting :math:`\tau` vary. And for **selecting**
which basis functions matter, a **sparsity** prior (the horseshoe of the regularisation lesson) drives
irrelevant coefficients to zero, choosing the basis automatically.

Why it matters
----------------

Two payoffs. **No discrete model selection** — instead of fitting many knot counts and comparing, fit
once with a rich basis and let the prior shrink to the right complexity, uncertainty in the smoothness
included. And a **unifying view**: penalised splines are hierarchical models, their smoothing parameter a
variance component, so the batching and regularisation stages apply directly. This is the bridge to
Gaussian processes, where the basis becomes infinite and the smoothing prior becomes a covariance
function.

.. hint::

   **Related lessons:** :doc:`Splines and weighted sums of basis functions <126-splines-and-weighted-sums-of-basis-functions>`  ·  :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`  ·  :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`  ·  :doc:`Non-normal models and regression surfaces <128-non-normal-models-and-regression-surfaces>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/basis-selection-and-shrinkage-of-coe%ef%ac%83cients/ <https://insightful-data-lab.com/2025/12/09/basis-selection-and-shrinkage-of-coe%ef%ac%83cients/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
