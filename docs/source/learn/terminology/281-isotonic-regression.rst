:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-isotonic-regression:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Isotonic Regression</b></div>`

=====================
Isotonic Regression
=====================

*A nonparametric, monotonic fit used to calibrate predicted probabilities.*

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

**Isotonic regression** is a **non-parametric** calibration method that fits a **monotonic** (non-decreasing)
**step function** mapping raw scores to calibrated probabilities — it assumes only that a higher score should
mean a higher probability, nothing about the shape.

Its strength
------------

Because it is **model-free**, it can correct **arbitrary** monotonic miscalibration that parametric methods
(Platt, temperature) miss — and with **enough** calibration data it typically **outperforms** them. It is
also a general tool for **monotonic regression**, not only calibration.

Its weakness
------------

That flexibility makes it **prone to overfitting** when calibration data is **scarce**, and the
piecewise-constant fit is less smooth. Unlike temperature scaling it does **not** guarantee the model's
accuracy is preserved.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Overconfident <284-overconfident>` · :doc:`Regression Models <309-regression-models>` · :doc:`Underconfident <283-underconfident>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Isotonic Regression <https://insightful-data-lab.com/2025/08/21/isotonic-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
