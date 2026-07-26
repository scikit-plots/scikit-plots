:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-margin-of-error-moe:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Margin of Error (MoE)</b></div>`

=======================
Margin of Error (MoE)
=======================

*The half-width of a confidence interval: the plus/minus range around a point estimate.*

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

The **margin of error (MoE)** is the **maximum expected gap between a sample estimate and
the true population parameter**, at a stated confidence level. It is the **± half-width**
of a confidence interval:

.. math::

   \text{Confidence Interval} = \text{Estimate} \pm \text{MoE}.

A small MoE means a precise estimate; a large one, an imprecise estimate.

How it's built
--------------

.. math::

   \text{MoE} = (\text{critical value}) \times (\text{standard error}),

where the **critical value** comes from the confidence level (1.96 for 95% under a normal
model) and the **standard error** measures sampling variability.

Two levers: confidence and sample size
--------------------------------------

- **Confidence level ↑** → larger critical value → **larger** MoE (more confidence costs
  width).
- **Sample size ↑** → smaller SE (:math:`\text{SE} \propto 1/\sqrt{n}`) → **smaller**
  MoE. Because of the square root, **halving the MoE requires 4× the sample**.

What it is not
--------------

MoE captures **random sampling error only**. It does **not** include bias, measurement
error, bad sampling design or model misspecification — so a tight MoE means **precise,
not necessarily accurate**. Two common traps: it isn't a hard maximum (it's
probabilistic), and it depends on confidence and variability, not sample size alone.

Why it matters
--------------

MoE turns a point estimate into an honest range ("support = 52% ± 3%" → true support
roughly 49–55%). It encourages **interval thinking** over point thinking, and ties
directly to significance: if a CI **excludes** the null value, the MoE is small enough to
declare a difference; if it **includes** the null, uncertainty swamps the effect.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Critical Value <087-critical-value>` · :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Frequentist <059-frequentist>` · :doc:`True Mean (Population Mean) <085-true-mean-population-mean>` · :doc:`A/B Testing <380-a-b-testing>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Margin of Error (MoE) <https://insightful-data-lab.com/2025/08/25/margin-of-error-moe/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
