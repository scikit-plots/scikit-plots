.. _ts-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q:

========================================================================
Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q)
========================================================================

**Stage 3 · 🔗 Linear & ARMA Processes**  ·  Lesson 08 of 18  ·  *intermediate*

:doc:`◀ Previous · Computing ACFs of Causal AR(2) Processes Using Difference Equations <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>`   ·   :doc:`Next · Best Linear Predictor of a Stationary Process ▶ <09-best-linear-predictor-of-a-stationary-process>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The general rule
------------------

The AR(2) result generalises. For an **AR(p)** — and for an **ARMA(p, q)** once you are far enough
from the start — the autocorrelations satisfy a **homogeneous difference equation driven by the AR
polynomial**:

.. math::

   \gamma(h) - \phi_1\,\gamma(h-1) - \dots - \phi_p\,\gamma(h-p) = 0, \qquad h > q.

Its general solution is a **linear combination of the terms** :math:`z_i^{-h}` from the **p roots**
of :math:`\phi(z)` — damped exponentials for real roots, damped sinusoids for complex ones.

Where the MA part enters
--------------------------

The moving-average order **q** does not change this decay law; it only sets the **initial
conditions**. For lags :math:`h \le q` the MA terms contribute directly, so the first few
autocorrelations look "irregular"; from :math:`h > q` onward the pure AR recursion takes over and
the smooth decay begins. In short, the **AR roots fix the pattern**, the **MA part fixes the first
few values**.

The shape of the ACF
----------------------

Because the solution is built from roots **outside** the unit circle, every term **decays**, so an
AR or ARMA autocorrelation **tails off** geometrically (possibly oscillating) but **never reaches
exactly zero**. A pure **MA(q)**, by contrast, has an ACF that **cuts off** sharply after lag
:math:`q` — a clean visual distinction.

ACF vs PACF
------------

This is the heart of **Box–Jenkins identification**: the **ACF cutting off** points to an **MA**
order, while the **PACF cutting off** points to an **AR** order. AR and ARMA both leave the ACF
tailing off, so the PACF (next lesson) is what pins down the autoregressive order.

.. hint::

   **Related lessons:** :doc:`Computing ACFs of Causal AR(2) Processes Using Difference Equations <07-computing-acfs-of-causal-ar-2-processes-using-difference-equations>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/understanding-acfs-via-difference-equations-for-arp-and-armap-q/ <https://insightful-data-lab.com/2026/01/17/understanding-acfs-via-difference-equations-for-arp-and-armap-q/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: intermediate
