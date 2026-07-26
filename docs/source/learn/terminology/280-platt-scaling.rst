:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-platt-scaling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Platt Scaling</b></div>`

===============
Platt Scaling
===============

*Fitting a logistic function to scores to produce calibrated probabilities.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

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

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Confidence Level <285-confidence-level>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Platt Scaling <https://insightful-data-lab.com/2025/08/21/platt-scaling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
