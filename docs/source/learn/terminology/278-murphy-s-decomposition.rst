:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-murphy-s-decomposition:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Murphy's Decomposition</b></div>`

========================
Murphy's Decomposition
========================

*A breakdown of a probabilistic score into calibration and refinement parts.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Murphy's decomposition** (1973) splits a **proper scoring rule** — classically the **Brier score** — into
three interpretable pieces:

.. math::

   \text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}.

It reveals *why* a probabilistic forecast scores as it does.

The three terms
---------------

**Reliability** is the **calibration** error (how far forecast probabilities sit from observed frequencies —
**lower** is better); **resolution** is how much the forecasts **vary** from the base rate to **separate**
outcomes (**higher** is better); **uncertainty** is the **irreducible** variance of the event itself,
independent of the model.

Why it matters
--------------

It shows a good score needs **both** good calibration **and** good resolution — a perfectly calibrated model
that always predicts the base rate has **zero** reliability error but **zero** resolution, and is useless. It
is the theoretical reason calibration metrics like **ECE** tell only **half** the story.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Temperature Scaling <279-temperature-scaling>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Murphy's Decomposition <https://insightful-data-lab.com/2025/08/21/murphys-decomposition/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
