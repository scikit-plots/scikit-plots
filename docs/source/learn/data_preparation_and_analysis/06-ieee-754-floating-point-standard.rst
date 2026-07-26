.. _dpa-ieee-754-floating-point-standard:

========================================================================
IEEE 754 Floating-Point Standard
========================================================================

**Stage 1 · 📋 Foundations**  ·  Lesson 06 of 56  ·  *beginner*

:doc:`◀ Previous · The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`   ·   :doc:`Next · Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) ▶ <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Storing real numbers
----------------------

Computers store real numbers in a **finite** number of bits, and the near-universal scheme for doing
so is the **IEEE 754** standard. Understanding it explains a whole class of surprises — why sums do
not quite add up, why you should never test two floats for exact equality — that otherwise look like
bugs.

Sign, exponent, mantissa
--------------------------

A floating-point number is stored in three parts, like scientific notation in binary: a **sign** bit,
an **exponent** (which scales the value), and a **mantissa** (the significant digits). The two common
sizes are **single precision** (32 bits: 1 sign, 8 exponent, 23 mantissa) and **double precision**
(64 bits: 1 sign, 11 exponent, 52 mantissa) — the ``float64`` that ``numpy`` and ``pandas`` use by
default. More mantissa bits mean more precision.

Why 0.1 + 0.2 ≠ 0.3
---------------------

With finite mantissa bits, most decimal fractions **cannot be represented exactly** — :math:`0.1` in
binary is a repeating fraction, rounded to fit. The rounding errors accumulate, so the famous result
is

.. math::

   0.1 + 0.2 = 0.30000000000000004 \neq 0.3.

It is not a language bug; it is the unavoidable cost of squeezing infinite decimals into 64 bits.

What it means for data work
-----------------------------

Three habits follow. **Never test floats for exact equality** — compare within a tolerance
(``numpy.isclose``) instead. **Beware accumulated error** when summing many values, and prefer stable
formulations. And know the **special values** the standard defines — positive and negative infinity,
and ``NaN`` (not-a-number) — because ``NaN`` in particular is how missing or undefined numeric results
surface throughout ``pandas``.

.. hint::

   **Related lessons:** :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`  ·  :doc:`Big Data: Definition, Characteristics, Evolution, and Business Impact <04-big-data-definition-characteristics-evolution-and-business-impact>`  ·  :doc:`Least Squares Regression <31-least-squares-regression>`  ·  :doc:`Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/ieee-754-floating-point-standard/ <https://insightful-data-lab.com/2026/01/14/ieee-754-floating-point-standard/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner
