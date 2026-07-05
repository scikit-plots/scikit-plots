:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-brier-score:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Brier Score</b></div>`

=============
Brier Score
=============

*The mean squared error of probabilistic predictions.*

What it is
----------

The **Brier score** is the **mean squared error** of probabilistic predictions — the average squared gap
between the predicted probability and the **actual** (0/1) outcome:

.. math::

   \text{BS} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2.

**Lower** is better, with **0** perfect. It is a single sample-level number for binary or multiclass problems.

Why it's special
----------------

Unlike ECE, the Brier score is a **strictly proper scoring rule** — it is minimized only by **honest**
probabilities, and by **Murphy's decomposition** it splits into **calibration** plus **refinement** terms. So
a low Brier score means the model is **both** well-calibrated **and** discriminating.

Its limitation
--------------

Because it **blends** calibration and discrimination, the Brier score can't tell you **which** is lacking — a
sharp-but-miscalibrated model and a calibrated-but-vague one can score similarly. That is why it is reported
**alongside** ECE and reliability curves, which isolate the calibration piece.

----

**Mind map — connected ideas**

   :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>`

----

**More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Brier Score <https://insightful-data-lab.com/2025/08/19/brier-score/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
