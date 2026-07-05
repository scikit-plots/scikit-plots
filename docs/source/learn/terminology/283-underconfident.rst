:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-underconfident:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Underconfident</b></div>`

================
Underconfident
================

*When predicted probabilities are less extreme than the true accuracy warrants.*

What it is
----------

A model is **underconfident** when its predicted probabilities are **too low** for its actual accuracy — it
hedges, claiming less certainty than it deserves. A model that is only 80% confident but 90% accurate is
underconfident.

How to spot it
--------------

On a **reliability diagram**, underconfident points fall **above** the diagonal (accuracy > confidence), and
predictions **cluster near 0.5** rather than committing. It is the mirror image of overconfidence, and a
single model can be **overconfident in some ranges and underconfident in others**.

Why it matters
--------------

Though it feels "safe," underconfidence **wastes** the model's discriminative signal — useful, correct
predictions get muted probabilities, so **thresholds** and **risk-based decisions** under-trigger. Like
overconfidence, it is fixed by **recalibration**.

----

**Mind map — connected ideas**

   :doc:`Overconfident <284-overconfident>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Sigmoid Function <297-sigmoid-function>`

----

**More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>`

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Underconfident <https://insightful-data-lab.com/2025/08/21/underconfident/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
