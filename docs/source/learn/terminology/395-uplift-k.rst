:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-uplift-k:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔗&nbsp;&nbsp;<b>Uplift@k</b></div>`

==========
Uplift@k
==========

*The incremental gain captured within the top-k targeted population.*

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

**Uplift@k** is a performance metric in uplift modelling and causal ML. It measures the **incremental
effect** achieved if you target only the **top k%** of customers, ranked by the model's predicted
uplift score. In plain terms: *if I contact only the top k%, how much extra impact do I get compared
to not contacting them?*

The formula
-----------

It is the **difference in average outcome** between the treatment and control groups **within the
top-k% segment**,

.. math::

   \text{Uplift@}k = \bar{y}_{\text{treatment}}^{(k)} - \bar{y}_{\text{control}}^{(k)},

where the averages are taken over the top k% by predicted uplift. Random targeting yields a small or
zero value (treatment and control behave alike); a good model makes treatment clearly outperform
control in that segment.

A worked example
----------------

With 10,000 customers, targeting the **top 20%** (k = 20%) selects 2,000. If, within that segment, the
treatment group's purchase probability exceeds the control group's by **5 percentage points**, then
uplift@k = **+5pp** — the extra impact the model captures by choosing those 2,000.

Uses, and versus uplift
-------------------------

In **marketing** it estimates incremental sales from promoting only the top k%; in **healthcare**, the
incremental recovery from treating the top-k patients; in **recommendation**, the extra engagement
from targeting the top-k users. The distinction from plain uplift is subtle but important: **uplift**
asks *how effective is the treatment overall?*, while **uplift@k** asks *how good is my model at
selecting the best subset to treat?*

----

*Theme:* :ref:`Causal Inference & Uplift <term-theme-causal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Uplift Models <205-uplift-models>` · :doc:`Uplift Score <204-uplift-score>` · :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Qini Curve <203-qini-curve>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Uplift <424-uplift>`

----

.. hint::
   **More in Causal Inference & Uplift**

   :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>` · :doc:`Causal Effect <306-causal-effect>` · :doc:`Causal Impact <112-causal-impact>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>` · :doc:`Causal Trees <301-causal-trees>` · :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>` · :doc:`Cumulative Uplift <198-cumulative-uplift>` · :doc:`Incremental Conversions <394-incremental-conversions>` · :doc:`Incremental Gain <200-incremental-gain>` · :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>` · :doc:`Incremental Revenue <193-incremental-revenue>` · :doc:`Incremental Sales <195-incremental-sales>` · :doc:`Qini Coefficient <397-qini-coefficient>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Uplift@k <https://insightful-data-lab.com/2025/08/19/upliftk/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
