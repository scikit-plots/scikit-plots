.. _ts-weak-and-strong-stationarity:

========================================================================
Weak and Strong Stationarity
========================================================================

**Stage 2 · 📐 Stationarity**  ·  Lesson 04 of 18  ·  *beginner*

:doc:`◀ Previous · A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>`   ·   :doc:`Next · Linear Processes ▶ <05-linear-processes>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Two definitions
----------------

"Stationarity" comes in two strengths. **Strong** (strict) stationarity constrains the **entire
distribution**; **weak** (second-order, or covariance) stationarity constrains only the **first
two moments**. Most classical time-series theory — including ARMA — needs only the weaker one.

Strong stationarity
--------------------

A series is **strictly stationary** if shifting time leaves its **whole joint distribution**
unchanged: for any lag :math:`h` and any set of times, the block :math:`(x_{t_1}, \dots, x_{t_k})`
has the same joint distribution as :math:`(x_{t_1+h}, \dots, x_{t_k+h})`,

.. math::

   (x_{t_1}, \dots, x_{t_k}) \;\stackrel{d}{=}\; (x_{t_1+h}, \dots, x_{t_k+h}).

Nothing about the probabilistic structure depends on **when** you look — a strong condition that
is hard to verify from a single finite sample.

Weak stationarity
-----------------

A series is **weakly stationary** if its **mean is constant**, its **variance is finite and
constant**, and its **autocovariance depends only on the lag** — not on absolute time:

.. math::

   \mathbb{E}[x_t] = \mu, \qquad
   \gamma(s, t) = \operatorname{Cov}(x_s, x_t) = \gamma(|s - t|).

This is all that ARMA modelling requires, and — unlike strict stationarity — it is checkable from
data through the sample mean and the sample autocorrelation.

How they relate
----------------

The two coincide under mild conditions. **Strong stationarity plus finite second moments implies
weak** stationarity. The converse fails in general, **except for Gaussian** processes: a Gaussian
process is fully described by its mean and covariance, so **weak + Gaussian implies strong**.
**White noise** — zero mean, constant variance, zero autocorrelation — is the canonical weakly
stationary building block.

.. hint::

   **Related lessons:** :doc:`A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>`  ·  :doc:`Linear Processes <05-linear-processes>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/weak-and-strong-stationarity/ <https://insightful-data-lab.com/2026/01/17/weak-and-strong-stationarity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: beginner
