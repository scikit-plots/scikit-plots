:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-incremental-conversions:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Incremental Conversions</b></div>`

=========================
Incremental Conversions
=========================

*Extra conversions caused by a treatment beyond the baseline.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Incremental conversions** measure the **additional number of conversions caused by a treatment** —
a campaign, promotion or intervention — compared with what would have happened without it (the
control). They are the extra conversions **directly attributable** to the treatment, not the raw
total.

The formula
-----------

The basic form is treatment conversions minus control conversions. To scale the effect to a full
population, use the rate-based version,

.. math::

   \text{Incremental Conversions} = \left( \frac{y^T}{n^T} - \frac{y^C}{n^C} \right) \times N,

where :math:`y^T/n^T` and :math:`y^C/n^C` are the treatment and control conversion rates and
:math:`N` is the total population you would target. This adjusts for the baseline rate and scales the
per-user uplift up to everyone.

A worked example
----------------

Run a campaign on 10,000 users: the treatment group (5,000) converts at 12% (600), the control group
(5,000) at 8% (400). The rate difference is **4%**, so across the full 10,000 the incremental
conversions are :math:`0.04 \times 10{,}000 = 400` — the extra conversions the campaign genuinely
caused.

Why it matters
--------------

**Total** conversions mislead, because some users would have converted anyway. **Incremental**
conversions isolate the **causal effect** of the treatment, which is why they are the foundation of
**uplift modelling** and incrementality testing across marketing, ads and A/B testing.

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Treatment Effect <072-treatment-effect>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Uplift <424-uplift>` · :doc:`Causal Inference <117-causal-inference>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>` · :doc:`Qini Curve <203-qini-curve>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Incremental Conversions <https://insightful-data-lab.com/2025/08/19/incremental-conversions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
