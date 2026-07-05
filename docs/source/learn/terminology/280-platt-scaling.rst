:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-platt-scaling:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Platt Scaling</b></div>`

===============
Platt Scaling
===============

*Fitting a logistic function to scores to produce calibrated probabilities.*

What it is
----------

**Platt scaling** calibrates a classifier by fitting a **logistic (sigmoid)** function on its raw scores,
mapping them to probabilities:

.. math::

   \hat{Q} = \sigma(a\,z + b).

The two parameters **a** and **b** are fit by **negative log-likelihood** on a validation set. It was invented
by John Platt for **SVMs**.

How it relates
--------------

It is a **parametric** method that assumes a **sigmoid**-shaped miscalibration. **Temperature scaling** is
essentially its **one-parameter, multi-class** special case (fixing the slope and dropping the offset), so
the two are close cousins.

When to use it
--------------

Platt scaling is a solid default for **binary** classifiers with **monotonic** score miscalibration and
limited calibration data, since two parameters rarely overfit — but if the true miscalibration isn't
sigmoid-shaped, a more flexible method like **isotonic regression** fits better.

----

**Mind map — connected ideas**

   :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Confidence Level <285-confidence-level>`

----

**More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Platt Scaling <https://insightful-data-lab.com/2025/08/21/platt-scaling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
