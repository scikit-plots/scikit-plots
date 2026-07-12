.. _ts-linear-processes:

========================================================================
Linear Processes
========================================================================

**Stage 3 · 🔗 Linear & ARMA Processes**  ·  Lesson 05 of 18  ·  *intermediate*

:doc:`◀ Previous · Weak and Strong Stationarity <04-weak-and-strong-stationarity>`   ·   :doc:`Next · Understanding ARMA Processes ▶ <06-understanding-arma-processes>`   ·   :doc:`↑ Section <index>`


What it is
------------

A **linear process** writes a series as a **weighted sum of white-noise shocks** — a linear filter
applied to an underlying noise sequence. It is the general form from which AR, MA and ARMA models
all descend.

The white-noise filter
------------------------

With white noise :math:`\{w_t\}` (mean zero, variance :math:`\sigma^2`), a linear process is

.. math::

   x_t = \mu + \sum_{j=-\infty}^{\infty} \psi_j\, w_{t-j},
   \qquad \sum_{j=-\infty}^{\infty} |\psi_j| < \infty.

The **absolute summability** of the weights :math:`\psi_j` guarantees the sum **converges** and the
result is **stationary**; it is a little stronger than square-summability and is what lets the
usual limit theorems apply.

Causal and one-sided
----------------------

A linear process is **causal** when it uses only **present and past** shocks — the future never
appears:

.. math::

   x_t = \mu + \sum_{j=0}^{\infty} \psi_j\, w_{t-j}.

This one-sided form is the **MA(∞)** representation. It is exactly the sense in which a causal ARMA
model can be "unrolled" into an infinite moving average of past noise.

Why it's central
----------------

**Wold's decomposition** says that *every* weakly stationary process has a linear-process (MA(∞))
component — so linear processes are not one model among many but the **canonical template** for
stationary series. Causality and invertibility (next lesson) are precisely the conditions under
which an ARMA model collapses into, or inverts back out of, this form.

.. hint::

   **Related lessons:** :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Weak and Strong Stationarity <04-weak-and-strong-stationarity>`  ·  :doc:`Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`  ·  :doc:`Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/linear-processes/ <https://insightful-data-lab.com/2026/01/17/linear-processes/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: intermediate
