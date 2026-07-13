:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-expected-calibration-error-ece:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Expected Calibration Error (ECE)</b></div>`

==================================
Expected Calibration Error (ECE)
==================================

*The average gap between confidence and accuracy across probability bins.*

What it is
----------

**Expected Calibration Error** summarizes miscalibration in **one number** — the **weighted average** gap
between a model's **confidence** and its **accuracy**, taken over bins of predictions:

.. math::

   \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.

Geometrically, it is the average distance of the **calibration curve** from the diagonal.

How it's computed
-----------------

Predictions are grouped into **bins** by confidence; in each bin you compare the **fraction correct**
(accuracy) to the **average confidence**, and weight each bin's gap by its **size**. The result is
**bounded** in [0, 1] and easy to report — the standard scalar for comparing calibration.

Its caveats
-----------

ECE is **bin-dependent** (the number and placement of bins move the value) and, being an **average**, it can
**hide** a badly miscalibrated region behind well-behaved bins. It is also **not a proper scoring rule** — a
trivial model can score low — so it is read with **reliability curves** and **Brier score**. Its variants are
**MCE** and **Adaptive ECE**.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Brier Score <418-brier-score>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Expected Calibration Error (ECE) <https://insightful-data-lab.com/2025/08/19/expected-calibration-error-ece/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
