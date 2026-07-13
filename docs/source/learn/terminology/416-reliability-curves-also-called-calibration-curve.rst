:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-reliability-curves-also-called-calibration-curves:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎯&nbsp;&nbsp;<b>Reliability Curves (also called Calibration Curves)</b></div>`

=====================================================
Reliability Curves (also called Calibration Curves)
=====================================================

*Plots of predicted probability against observed frequency.*

What it is
----------

A **reliability curve** (or calibration curve / reliability diagram) is the **visual** check for calibration —
it plots **predicted probability** on the x-axis against the **observed frequency** of the outcome on the
y-axis. A perfectly calibrated model traces the **diagonal** :math:`y = x`.

Reading it
----------

Points **below** the diagonal mean the model is **overconfident** (accuracy falls short of its confidence);
points **above** mean it is **underconfident**. A companion **histogram** of confidences shows whether
predictions pile up at the **extremes** — a hallmark of overconfident networks.

Why it complements ECE
----------------------

A single ECE number can't say **where** miscalibration happens, and two models with the **same** ECE can have
very different curves. The reliability curve **localizes** the problem across the confidence range — and the
weighted gap between it and the diagonal **is** the ECE.

----

*Theme:* :ref:`Probability Calibration <term-theme-calibration>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Overconfident <284-overconfident>` · :doc:`Underconfident <283-underconfident>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Brier Score <418-brier-score>` · :doc:`Temperature Scaling <279-temperature-scaling>`

----

.. hint::
   **More in Probability Calibration**

   :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>` · :doc:`Brier Score <418-brier-score>` · :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>` · :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>` · :doc:`Isotonic Regression <281-isotonic-regression>` · :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>` · :doc:`Murphy's Decomposition <278-murphy-s-decomposition>` · :doc:`Overconfident <284-overconfident>` · :doc:`Platt Scaling <280-platt-scaling>` · :doc:`Temperature Scaling <279-temperature-scaling>` · :doc:`Underconfident <283-underconfident>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Reliability Curves (also called Calibration Curves) <https://insightful-data-lab.com/2025/08/19/reliability-curves-also-called-calibration-curves/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
