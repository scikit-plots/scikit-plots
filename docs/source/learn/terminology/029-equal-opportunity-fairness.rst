:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-equal-opportunity-fairness:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">⚖️&nbsp;&nbsp;<b>Equal Opportunity (Fairness)</b></div>`

==============================
Equal Opportunity (Fairness)
==============================

*Requires equal true-positive rates across groups — a relaxation of equalized odds.*

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

**Equal opportunity** is the relaxation of equalized odds that equalises only the
**true-positive rate** across groups:

.. math::

   P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b) \quad \forall\, a, b.

Among the people who *should* receive a positive outcome, every group has the same
chance of being correctly recognised — so it targets **false negatives not falling
disproportionately on a disadvantaged group**.

Where it sits among fairness criteria
-------------------------------------

It is a one-sided **separation** criterion: equalized odds asks for equal TPR *and*
equal FPR; equal opportunity keeps only the TPR condition. It sits between the
label-blind **demographic parity** and the full **equalized odds**.

Example
-------

Among genuinely qualified applicants, a loan model approves 80% of men but only 60%
of women. Qualified women are recognised less often — equal opportunity is violated,
even if overall approval rates happen to match.

The catch
---------

Because it uses ground-truth labels it is more practical than demographic parity, but
it ignores **false positives**, and when base rates differ across groups achieving it
can still reduce accuracy and may clash with predictive parity.

Limitations
-----------

- Says nothing about false-positive fairness (a group could be over-approved).
- Base-rate differences can force an accuracy trade-off.

In code
-------

.. code-block:: python

   from fairlearn.metrics import true_positive_rate, MetricFrame

   tpr_by_group = MetricFrame(metrics=true_positive_rate,
                              y_true=y_true, y_pred=y_pred,
                              sensitive_features=A).by_group

----

*Theme:* :ref:`Fairness & Calibration <term-theme-fairness>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`

----

.. hint::
   **More in Fairness & Calibration**

   :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>` · :doc:`Fairness Guardrails <351-fairness-guardrails>` · :doc:`Fairness parity <372-fairness-parity>` · :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>` · :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>` · :doc:`Selection Rate <390-selection-rate>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Equal Opportunity (Fairness) <https://insightful-data-lab.com/2025/08/29/equal-opportunity-fairness/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
