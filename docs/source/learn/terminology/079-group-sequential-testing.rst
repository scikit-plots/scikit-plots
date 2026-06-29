:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-group-sequential-testing:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Group Sequential Testing</b></div>`

==========================
Group Sequential Testing
==========================

*Frequentist designs allowing interim analyses at planned points with adjusted error spending.*

What it is
----------

**Group sequential testing** lets you **analyse accumulating data at several pre-planned
interim points** during an experiment — not just once at the end — and at each look
decide to **stop for efficacy** (the effect is clearly there), **stop for futility** (it
clearly isn't), or **continue**. It is standard in clinical trials and increasingly used
in A/B testing where stopping early saves resources.

The peeking problem it solves
-----------------------------

Repeatedly checking a fixed-:math:`\alpha` test and stopping the moment :math:`p < 0.05`
**inflates the Type I error** badly — with many looks, the chance of a false positive can
climb to **20–30%**. Group sequential designs fix this with an **α-spending function**
that divides the error budget across looks so the *overall* Type I rate stays at
:math:`\alpha`.

How it works
------------

Plan the number of interim analyses up front, then use an **α-spending rule** to set a
significance cutoff at each: early looks get **stricter** thresholds, later looks more
**lenient** ones, and you stop as soon as a threshold is crossed.

The α-spending rules
--------------------

- **O'Brien–Fleming** — very strict early, lenient late (final ≈ fixed :math:`\alpha`).
- **Pocock** — the same moderate threshold at every look.
- **Lan–DeMets** — a flexible spending function that allocates :math:`\alpha`
  adaptively, without fixing the look times in advance.

Example
-------

A trial of 1,000 patients, analysed every 250 with total :math:`\alpha = 0.05` under an
O'Brien–Fleming schedule, might spend :math:`\alpha = 0.001, 0.01, 0.02, 0.04` across the
four looks. A p-value of 0.008 at 500 patients crosses the second bound → **stop early
for efficacy**.

Where it sits
-------------

It is the middle ground between extremes: more efficient than the **fixed-horizon** A/B
test (one look, may waste data) and statistically valid unlike **naive peeking** (which
inflates false positives), while **bandit / adaptive** methods go further still by
reallocating traffic continuously.

----

**Mind map — connected ideas**

   :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Type I Error <080-type-i-error>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Group Sequential Testing <https://insightful-data-lab.com/2025/08/25/group-sequential-testing/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
