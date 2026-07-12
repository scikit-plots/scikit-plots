.. _ts-a-gentle-introduction-to-stationarity:

========================================================================
A Gentle Introduction to Stationarity
========================================================================

**Stage 2 · 📐 Stationarity**  ·  Lesson 03 of 18  ·  *beginner*

:doc:`◀ Previous · Getting Started with R <02-getting-started-with-r>`   ·   :doc:`Next · Weak and Strong Stationarity ▶ <04-weak-and-strong-stationarity>`   ·   :doc:`↑ Section <index>`


What it is
------------

A series is **stationary** when its statistical behaviour **does not depend on time**: loosely,
the mean, the variance, and the way points co-vary with their lagged selves all stay **constant**
as the observation window slides. A stationary series has **no predictable long-run pattern** — no
trend and no fixed seasonal swing.

The intuition
--------------

Stationarity is what makes a series **learnable from a single realisation**. If the rules do not
change over time, the past is a fair guide to the future, and one long stretch of data is enough
to estimate them. Trend and seasonality break this because the distribution of the value shifts
with **when** you look. (The subtlety worth remembering: a series with cycles of **no fixed
length** can still be stationary — you cannot say in advance where the peaks will fall.)

Achieving stationarity
------------------------

The classical fix is **differencing** — model the change rather than the level:

.. math::

   \nabla x_t = x_t - x_{t-1},

which removes a linear trend; a **seasonal** difference :math:`x_t - x_{t-s}` removes a
period-:math:`s` pattern. (Detrending by regression is an alternative.) This "integrate" step is
the **I** in ARIMA. Over-differencing injects artefacts, so use the **smallest** order that works.

How to check
--------------

Plot first: a stationary series looks **flat and stable** around a constant level, with no drift.
Then formalise with tests — the **Augmented Dickey–Fuller (ADF)** test (null hypothesis:
*non-stationary*, so a small p-value argues for stationarity) and the **KPSS** test (null:
*stationary*), commonly read together. In ``statsmodels`` these are ``adfuller(x)`` and ``kpss(x)``.

.. hint::

   **Related lessons:** :doc:`Weak and Strong Stationarity <04-weak-and-strong-stationarity>`  ·  :doc:`What Are Time Series, and How Are They Used? <01-what-are-time-series-and-how-are-they-used>`  ·  :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`  ·  :doc:`Linear Processes <05-linear-processes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/a-gentle-introduction-to-stationarity/ <https://insightful-data-lab.com/2026/01/17/a-gentle-introduction-to-stationarity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: beginner
