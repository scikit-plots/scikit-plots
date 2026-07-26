:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-risk-of-peeking:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Risk of Peeking</b></div>`

=================
Risk of Peeking
=================

*The inflated false-positive risk from repeatedly checking a fixed-horizon test early.*

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

**Peeking** is looking at an experiment's results **before it officially ends** and acting
on that interim data. The **risk of peeking** is the **inflated false-positive (Type I)
rate** that results from repeatedly checking and **stopping as soon as significance
appears**.

Why it breaks the test
----------------------

A fixed-horizon test assumes a **single look at a predetermined sample size**. Each extra
peek is another **independent chance** for noise to cross :math:`p < 0.05`, so the true
error rate compounds far above the nominal :math:`\alpha`. Peek ten times at a 5% threshold
and the real false-positive rate can reach **20–30%** — this compounding is called **alpha
inflation**.

What it costs
-------------

Concretely: a button test shows A ahead on day 1 (:math:`p = 0.04`), you stop and crown
A — but over the full two weeks **B** would have won. The early stop produced a **false
conclusion**, and at scale that means wrong launches, lost revenue, and eroded trust in
experimentation.

How to avoid it
---------------

Four routes: **predefine** the sample size and duration and only check at the end; use
**sequential testing / alpha-spending** designs built for interim looks (**group
sequential**, **O'Brien–Fleming**, **Pocock**); use **Bayesian** methods designed for
continuous monitoring; or, if peeks are unavoidable, apply **multiplicity corrections**
(Bonferroni, Holm). The peeking problem is precisely why the whole machinery of
fixed-horizon and sequential testing exists.

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Type I Error <080-type-i-error>`

----

.. hint::
   **More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Risk of Peeking <https://insightful-data-lab.com/2025/08/24/risk-of-peeking/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
