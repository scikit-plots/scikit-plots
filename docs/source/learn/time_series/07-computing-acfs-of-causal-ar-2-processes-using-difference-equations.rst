.. _ts-computing-acfs-of-causal-ar-2-processes-using-difference-equations:

========================================================================
Computing ACFs of Causal AR(2) Processes Using Difference Equations
========================================================================

**Stage 3 · 🔗 Linear & ARMA Processes**  ·  Lesson 07 of 18  ·  *intermediate*

:doc:`◀ Previous · Understanding ARMA Processes <06-understanding-arma-processes>`   ·   :doc:`Next · Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q> ▶`


The recursion
--------------

For a causal **AR(2)**, :math:`x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t`, the autocorrelations
obey the **same recursion as the process itself** — a homogeneous linear **difference equation**:

.. math::

   \rho(h) = \phi_1\,\rho(h-1) + \phi_2\,\rho(h-2), \qquad h \ge 1.

With :math:`\rho(0) = 1`, the **Yule–Walker** start gives :math:`\rho(1) = \phi_1 / (1 - \phi_2)`,
and every later lag follows by iterating.

Solving it
------------

Rather than iterate forever, solve the difference equation **in closed form** through its
**characteristic equation** — equivalently, the **roots** :math:`z_1, z_2` of the AR polynomial
:math:`1 - \phi_1 z - \phi_2 z^2`. The ACF is then a combination of the terms :math:`z_i^{-h}`,
whose magnitudes are controlled by how far the roots sit **outside** the unit circle (causality
guarantees they do).

Two regimes
------------

The **discriminant** :math:`\phi_1^2 + 4\phi_2` decides the shape. When it is **positive**, the
roots are **real** and the ACF is a sum of two **damped exponentials** (decaying monotonically, with
uniform or alternating sign). When it is **negative**, the roots are **complex conjugates** and the
ACF is a **damped sinusoid** — a decaying oscillation with system frequency

.. math::

   f_0 = \frac{1}{2\pi}\cos^{-1}\!\left( \frac{\phi_1}{2\sqrt{-\phi_2}} \right).

What it tells you
------------------

Either way the ACF **tails off** toward zero but never truly **cuts off** — the signature of an
autoregressive process. (Its partner, the **PACF**, does cut off, after lag 2.) Reading whether the
decay is exponential or oscillatory is a first clue to the underlying dynamics.

.. seealso::

   **Related lessons:** :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Understanding ACFs via Difference Equations for AR(p) and ARMA(p, q) <08-understanding-acfs-via-difference-equations-for-ar-p-and-arma-p-q>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`Weak and Strong Stationarity <04-weak-and-strong-stationarity>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/17/computing-acfs-of-causal-ar2-processes-using-difference-equations/ <https://insightful-data-lab.com/2026/01/17/computing-acfs-of-causal-ar2-processes-using-difference-equations/>`__

.. tags:: purpose: reference, topic: time series, level: intermediate
