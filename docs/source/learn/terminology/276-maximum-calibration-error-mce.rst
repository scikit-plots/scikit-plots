:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-maximum-calibration-error-mce:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Maximum Calibration Error (MCE)</b></div>`

=================================
Maximum Calibration Error (MCE)
=================================

*The largest gap between confidence and accuracy across calibration bins.*

What it is
----------

**Maximum Calibration Error** reports the **worst** calibration gap rather than the average — the **largest**
difference between accuracy and confidence over all bins:

.. math::

   \text{MCE} = \max_{m}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

Where ECE asks *how miscalibrated on average?*, MCE asks *how bad does it get?*

When it matters
---------------

MCE is the right lens for **safety-critical** systems — medical, autonomous, financial — where a single
**badly** miscalibrated confidence region can cause harm, even if the **average** looks fine. Lower is
better, as with ECE.

Its limits
----------

Like ECE it is **binning-dependent** (the answer shifts with bin count and scheme), and it is **not a proper
scoring rule** — a model can achieve low calibration error with **trivial** predictions, so MCE must be read
**alongside** discrimination metrics, not alone.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Overconfident <284-overconfident>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Maximum Calibration Error (MCE) <https://insightful-data-lab.com/2025/08/22/maximum-calibration-error-mce/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
