.. _ts-understanding-arma-processes:

========================================================================
Understanding ARMA Processes
========================================================================

**Stage 3 · 🔗 Linear & ARMA Processes**  ·  Lesson 06 of 18  ·  *intermediate*

:doc:`◀ Previous · Linear Processes <05-linear-processes>`   ·   :doc:`Next · Computing ACFs of Causal AR(2) Processes Using Difference Equations ▶ <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>`   ·   :doc:`↑ Section <index>`


The model
----------

An **ARMA(p, q)** process blends two mechanisms: an **autoregressive (AR)** part, where the value
depends on its own **p** past values, and a **moving-average (MA)** part, where it depends on the
last **q** white-noise **shocks**:

.. math::

   x_t = \phi_1 x_{t-1} + \dots + \phi_p x_{t-p}
         + w_t + \theta_1 w_{t-1} + \dots + \theta_q w_{t-q}.

Backshift form
----------------

Using the **backshift operator** :math:`B` (with :math:`B^j x_t = x_{t-j}`), the model compresses
to

.. math::

   \phi(B)\, x_t = \theta(B)\, w_t,

where :math:`\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p` is the **AR polynomial** and
:math:`\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q` the **MA polynomial**. This algebra makes
the next two properties easy to state.

Causality and invertibility
----------------------------

Two root conditions govern behaviour. The process is **causal** — expressible as a one-sided MA(∞)
of past shocks — when **all roots of** :math:`\phi(z)` lie **outside** the unit circle
(:math:`|z| > 1`). It is **invertible** — expressible as an AR(∞) in past values — when **all roots
of** :math:`\theta(z)` lie **outside** the unit circle. Causality is what makes an ARMA a proper
linear process; invertibility makes its parameters **identifiable** from the data.

Watch for redundancy
----------------------

If the AR and MA polynomials share a **common factor**, the model is **over-parameterised**: the
factor cancels and a simpler model fits identically. The extreme case :math:`\phi(B) = \theta(B)`
reduces the whole model to :math:`x_t = w_t`, plain white noise. Always **cancel common roots**
before trusting a fit; ``statsmodels`` exposes the fitted ``.arroots`` and ``.maroots`` for exactly
this check.

.. hint::

   **Related lessons:** :doc:`Linear Processes <05-linear-processes>`  ·  :doc:`Computing ACFs of Causal AR(2) Processes Using Difference Equations <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>`  ·  :doc:`Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`  ·  :doc:`Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`  ·  :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/understanding-arma-processes/ <https://insightful-data-lab.com/2026/01/17/understanding-arma-processes/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: intermediate
