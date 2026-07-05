:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-overconfident:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Overconfident</b></div>`

===============
Overconfident
===============

*When predicted probabilities are more extreme than the true accuracy warrants.*

What it is
----------

A model is **overconfident** when its predicted probabilities are **too high** for its actual accuracy — it
claims more certainty than it earns. A model that is 99% confident but only 90% accurate is overconfident.

How to spot it
--------------

On a **reliability diagram**, overconfident points fall **below** the diagonal (accuracy < confidence), and
the histogram piles predictions near **1.0**. Modern deep networks are frequently overconfident, having
"memorized" training data and carried that certainty to new inputs; **log loss** flags it by punishing
confident-wrong predictions heavily.

Why it's dangerous
------------------

Overconfidence is a **safety** hazard in high-stakes settings — an overconfident medical or fraud model
triggers costly actions on cases it has wrong. It is corrected **post-hoc** with methods like temperature
scaling, Platt scaling or isotonic regression.

----

**Mind map — connected ideas**

   :doc:`Underconfident <283-underconfident>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Softmax Function <296-softmax-function>`

----

**More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Overconfident <https://insightful-data-lab.com/2025/08/21/overconfident/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
