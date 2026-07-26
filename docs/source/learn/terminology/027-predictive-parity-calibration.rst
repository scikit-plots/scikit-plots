:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-predictive-parity-calibration:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Predictive Parity (Calibration)</b></div>`

=================================
Predictive Parity (Calibration)
=================================

*A fairness criterion requiring equal positive predictive value (precision) across groups.*

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

**Predictive parity**, also called **calibration by group**, asks that a score mean
the same thing regardless of group: among everyone assigned the same predicted
probability, the actual positive rate is equal across groups. Formally, for a
predicted probability :math:`\hat{p}`,

.. math::

   P(Y=1 \mid \hat{P}=\hat{p}, A=a) = P(Y=1 \mid \hat{P}=\hat{p}, A=b)
   \quad \forall\, a, b,

with :math:`Y` the true outcome, :math:`\hat{P}` the score and :math:`A` the
protected attribute. In plain terms: if two people from different groups are both
scored at 70%, then about 70% of *each* group should actually turn out positive.

Where it sits among fairness criteria
-------------------------------------

This is the **sufficiency** criterion, :math:`Y \perp A \mid \hat{Y}` — the label is
independent of group once you condition on the score. It is the counterpart to the
**independence** criterion (demographic parity) and the **separation** criterion
(equalized odds / equal opportunity).

Example
-------

A loan model scores a set of applicants at 0.7 predicted default probability. If 70%
of group A but only 50% of group B actually default, the score 0.7 does not carry the
same meaning across groups — predictive parity is violated.

The catch: it conflicts with the others
---------------------------------------

When the two groups have **different base rates**, predictive parity and equalized
odds **cannot both hold** (except in degenerate cases). This is the *fairness
impossibility theorem* (Kleinberg–Chouldechova): independence, separation and
sufficiency are mutually incompatible whenever base rates differ, so you must choose
which to prioritise.

Limitations
-----------

- A model can be perfectly calibrated yet still distribute **errors** unevenly — it
  says nothing about false-positive or false-negative rates.
- Small groups look mis-calibrated from variance alone; check with enough data.

In code
-------

.. code-block:: python

   from sklearn.calibration import calibration_curve

   # compare reliability curves per group
   for a in groups:
       frac_pos, mean_pred = calibration_curve(y_true[A == a], y_score[A == a], n_bins=10)

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Predictive Parity (Calibration) <https://insightful-data-lab.com/2025/08/29/predictive-parity-calibration/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
