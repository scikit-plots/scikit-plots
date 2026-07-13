:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-adaptive-ece-expected-calibration-error-with-adaptive-binning:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Adaptive ECE (Expected Calibration Error with Adaptive Binning)</b></div>`

=================================================================
Adaptive ECE (Expected Calibration Error with Adaptive Binning)
=================================================================

*A calibration error using adaptive bins so each holds a similar count.*

What it is
----------

**Adaptive ECE** measures a classifier's **miscalibration** — the gap between its **confidence** and its
actual **accuracy** — using **equal-count** bins. Like standard **Expected Calibration Error**, it is a
weighted average of |accuracy − confidence| across bins:

.. math::

   \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

The "adaptive" part changes only **how the bins are drawn**.

The problem it fixes
--------------------

Standard ECE uses **fixed equal-width** bins ([0.0–0.1], …), so when predictions **cluster** (modern nets
pile probabilities near 1.0), some bins hold **few or zero** samples and give **noisy** estimates. Adaptive
binning instead makes each bin hold **the same number** of predictions, so bin **widths vary** with the data.

Why it helps
------------

Equal-count bins yield a **more stable, fairer** calibration estimate on **skewed** predictions, where
equal-width ECE is unreliable. It shares ECE's caveat, though: the result still depends on the **number of
bins**, and neither is a **proper scoring rule**.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Overconfident <284-overconfident>` · :doc:`Underconfident <283-underconfident>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Adaptive ECE (Expected Calibration Error with Adaptive Binning) <https://insightful-data-lab.com/2025/08/22/adaptive-ece-expected-calibration-error-with-adaptive-binning/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
