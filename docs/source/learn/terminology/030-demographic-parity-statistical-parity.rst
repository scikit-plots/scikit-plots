:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-demographic-parity-statistical-parity:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Demographic Parity (Statistical Parity)</b></div>`

=========================================
Demographic Parity (Statistical Parity)
=========================================

*Requires the positive-prediction rate to be equal across groups, independent of the true label.*

What it is
----------

**Demographic parity** (or **statistical parity**) asks that the model's positive
decisions be **independent of the protected attribute** — every group receives a
positive prediction at the same rate:

.. math::

   P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b) \quad \forall\, a, b.

Crucially it looks only at the prediction :math:`\hat{Y}`, never at the true label
:math:`Y`.

Where it sits among fairness criteria
-------------------------------------

This is the **independence** criterion, :math:`\hat{Y} \perp A`. It is the simplest
and most label-blind of the three families — independence (here), **separation**
(equalized odds / equal opportunity) and **sufficiency** (predictive parity).

Measuring it
------------

Two common gap metrics, with :math:`a` the disadvantaged group:

.. math::

   \text{DPD} = P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b), \qquad
   \text{DPR} = \frac{P(\hat{Y}=1 \mid A=a)}{P(\hat{Y}=1 \mid A=b)}.

The ratio connects to the legal **four-fifths (80%) rule**: a selection-rate ratio
below 0.8 is treated as evidence of adverse impact.

Example
-------

A hiring model marks 60% of men but only 40% of women as interview-worthy. The rates
differ, so demographic parity is violated (and the 0.40 / 0.60 ≈ 0.67 ratio fails the
four-fifths rule).

The catch
---------

Because it ignores the label, demographic parity can be satisfied only by
**approving unqualified members** of one group to match rates — which may raise risk
and clash with equal opportunity, equalized odds and predictive parity.

Limitations
-----------

- Ignores genuine differences in qualification (the true label).
- Conflicts with the error-rate and calibration criteria when base rates differ.

In code
-------

.. code-block:: python

   from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

   dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=A)
   dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=A)

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Demographic Parity (Statistical Parity) <https://insightful-data-lab.com/2025/08/29/demographic-parity-statistical-parity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
