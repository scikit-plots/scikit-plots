:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-minimum-detectable-lift-mdl:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Minimum Detectable Lift (MDL)</b></div>`

===============================
Minimum Detectable Lift (MDL)
===============================

*The smallest effect an experiment is powered to detect reliably.*

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

The **minimum detectable lift (MDL)** is the **smallest relative change in a metric** —
conversion, revenue, clicks — that an experiment can **reliably detect**, given its sample
size :math:`n`, significance level :math:`\alpha` and power :math:`1 - \beta`. It is, in
effect, the smallest effect you have decided is worth catching.

Why it matters
--------------

It stops teams **over-optimising for trivial effects** and forces the design question up
front: *what improvement is big enough to justify the test?* Crucially, the relationship
is inverse — **the smaller the MDL you want to detect, the larger the sample you need**.

The formula
-----------

For conversion rates,

.. math::

   \text{MDL} = \frac{p_{\text{treatment}} - p_{\text{control}}}{p_{\text{control}}}.

With a 5% baseline, :math:`\alpha = 0.05`, power 0.80 and 20,000 per variant, the design
can detect a **10% relative lift** (5% → 5.5%) — so the MDL is +10%. A true lift of only
+2% would likely slip past undetected.

MDL vs MDE
----------

The two are easy to confuse. The **minimum detectable effect (MDE)** is the smallest
**absolute** change (e.g. +0.5 percentage points); the **MDL** is the smallest
**relative** change (a percentage lift). For control 5% → treatment 5.5%, the MDE is +0.5
points while the MDL is +10%.

Striking the balance
--------------------

Set the MDL **too high** and you miss small-but-valuable wins; set it **too low** and the
test may need millions of users. The resolution is a negotiation: the **business** names
the smallest improvement worth acting on, and the **statistician** sizes the experiment to
detect at least that.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Minimum Detectable Lift (MDL) <https://insightful-data-lab.com/2025/08/24/minimum-detectable-lift-mdl/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
