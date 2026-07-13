:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-conversion-rate-uplift:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Conversion Rate Uplift</b></div>`

========================
Conversion Rate Uplift
========================

*The increase in conversion rate attributable to a treatment versus control.*

What it is
----------

**Conversion-rate uplift** measures the improvement (or decline) in conversion rate of a
**treatment** (variant) relative to a **control** (baseline) — the headline success
metric of an A/B test. It answers: *by how much did the new variant move conversions
versus the baseline?*

Absolute vs relative
--------------------

With :math:`CR_A` the control rate and :math:`CR_B` the treatment rate, there are two
distinct quantities:

.. math::

   \text{Absolute uplift} = CR_B - CR_A \ \text{(percentage points)}, \qquad
   \text{Relative uplift} = \frac{CR_B - CR_A}{CR_A} \times 100\%.

Examples
--------

- **Positive** — control 5%, treatment 6%: absolute uplift **+1 point**, relative uplift
  :math:`(6-5)/5 = +20\%`.
- **Negative** — control 10%, treatment 9.5%: absolute uplift **−0.5 point**, relative
  uplift :math:`(9.5-10)/10 = -5\%`.

Why both matter
---------------

The two can tell very different stories: a **1-point** absolute uplift sounds tiny, but
on a **2%** baseline it is a **50% relative** improvement. Quoting only one can mislead —
report both.

How it drives A/B decisions
---------------------------

Uplift is the primary metric, and the **minimum detectable uplift / effect (MDE)** — the
smallest effect you want to catch given sample size, :math:`\alpha` and power — sets how
big the experiment must be. The decision rule: a **significant positive** uplift → ship;
**not significant** → inconclusive; **significant negative** → stop or rethink. Whether
the uplift is real or just noise is settled by a **two-proportion z-test** (frequentist)
or the **posterior probability of uplift** (Bayesian).

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Conversion Rate Uplift <https://insightful-data-lab.com/2025/08/25/conversion-rate-uplift/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
