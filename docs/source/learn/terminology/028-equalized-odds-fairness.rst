:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-equalized-odds-fairness:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Equalized Odds (Fairness)</b></div>`

===========================
Equalized Odds (Fairness)
===========================

*Requires equal true-positive and false-positive rates across groups.*

What it is
----------

**Equalized odds** requires a classifier to have the **same true-positive rate and
the same false-positive rate** across groups. Formally,

.. math::

   P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b)
   \quad \forall\, y \in \{0,1\},\ \forall\, a, b.

So among those who truly are positive (:math:`Y=1`) every group is recognised at the
same rate, and among those who truly are negative (:math:`Y=0`) every group is
wrongly flagged at the same rate.

Where it sits among fairness criteria
-------------------------------------

This is the **separation** criterion, :math:`\hat{Y} \perp A \mid Y` — the prediction
is independent of group once you condition on the truth. It is the strictest of the
error-rate criteria: equalized odds = **equal opportunity** (equal TPR) *plus* equal
FPR.

Example
-------

A loan model approves qualified men at TPR 80% but qualified women at TPR 60%, and
wrongly approves unqualified men at FPR 20% but unqualified women at FPR 30%. Both
rates differ by group, so equalized odds is violated on both counts.

The catch
---------

Like the other criteria, equalized odds collides with **predictive parity** when base
rates differ (the impossibility theorem), and enforcing it can cost overall accuracy
— so practitioners often target *approximate* equalized odds within a tolerance.

Limitations
-----------

- Hard to satisfy exactly, especially with unequal base rates.
- Trades off against accuracy; usually relaxed rather than enforced exactly.

In code
-------

.. code-block:: python

   from fairlearn.metrics import equalized_odds_difference

   eod = equalized_odds_difference(y_true, y_pred, sensitive_features=A)  # 0 = parity

----

**Mind map — connected ideas**

   :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`

----

**More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Equalized Odds (Fairness) <https://insightful-data-lab.com/2025/08/29/equalized-odds-fairness/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
