:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-temperature-scaling:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Temperature Scaling</b></div>`

=====================
Temperature Scaling
=====================

*A simple post-hoc method that rescales logits to calibrate probabilities.*

What it is
----------

**Temperature scaling** is the simplest **post-hoc calibration** method for neural nets — it divides the
**logits** by a single learned scalar **T** before the **softmax**, softening or sharpening the
probabilities:

.. math::

   \hat{Q} = \mathrm{softmax}(\mathbf{z} / T), \quad T > 0.

It is a one-parameter fix applied **after** training.

What T does
-----------

**T = 1** leaves the model unchanged; **T > 1** makes predictions **less confident** (softer), which corrects
the **overconfidence** typical of modern networks; **T < 1** makes them sharper. T is fit on a **held-out**
validation set by minimizing **negative log-likelihood**.

Why it's popular
----------------

It is **simple**, **effective**, and crucially **accuracy-preserving** — dividing every logit by the same T
never changes the **argmax**, so the decision boundary and accuracy are untouched. Its limit is
**expressiveness**: it can only rescale confidence uniformly, not fix region-specific miscalibration.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Overconfident <284-overconfident>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Softmax Function <296-softmax-function>` · :doc:`Underconfident <283-underconfident>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Temperature Scaling <https://insightful-data-lab.com/2025/08/21/temperature-scaling/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
