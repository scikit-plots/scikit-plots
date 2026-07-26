:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-calibration-quality-model-calibration:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Calibration quality (Model Calibration)</b></div>`

=========================================
Calibration quality (Model Calibration)
=========================================

*How well predicted probabilities match real-world frequencies.*

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

**Calibration quality** describes how well a model's predicted **probabilities** match **reality** — a
well-calibrated model that says "80% confident" is right about **80%** of the time. It is a property of the
**probabilities**, separate from whether the model is **accurate**.

Calibration vs accuracy
-----------------------

A model can be **accurate** yet badly calibrated (right often, but its confidence numbers are meaningless) or
**calibrated** yet weakly **discriminating**. The two are distinct axes, which is why **proper scores** like
log loss and Brier — which reward **both** — are read together with pure calibration measures.

Why it matters
--------------

Whenever probabilities feed **decisions** — thresholds, expected-value calculations, downstream systems —
calibration is essential, and modern deep networks are typically **overconfident**. It is measured with
**ECE**, **reliability curves**, and **Brier score**, and repaired **post-hoc** with **temperature**,
**Platt**, or **isotonic** scaling.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Overconfident <284-overconfident>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Brier Score <418-brier-score>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Calibration quality (Model Calibration) <https://insightful-data-lab.com/2025/08/19/calibration-model-calibration/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
