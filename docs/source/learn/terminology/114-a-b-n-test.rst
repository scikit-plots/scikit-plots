:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-a-b-n-test:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>A/B/n Test</b></div>`

============
A/B/n Test
============

*An experiment comparing more than two variants simultaneously.*

What it is
----------

An **A/B/n test** generalises the two-arm A/B test to **several variants at once** —
A vs B vs C vs … n — of a page, app or feature, splitting traffic across all of them to find
which scores best on a chosen metric (conversion, click-through, engagement).

How it works
------------

Pick a **control** (A, the current design), build variants B, C, D…; **randomly split**
users across them; track the metric; and use a statistical test (two-proportion z-test,
chi-square or a Bayesian model) to pick the winner. Testing button text — "Buy Now" vs
"Shop Now" vs "Get Yours Today" vs "Order Now" — sends each of four equal groups one variant
and compares conversions.

Why and when
------------

It tests **many ideas in one experiment** rather than a sequence of A/B tests, which is
faster when you have several candidate designs and enough traffic for a **winner-takes-all**
verdict.

The costs
---------

Two prices. **Traffic**: a 50/50 split becomes 33/33/33 and beyond, so each arm gets less
data and significance takes longer. And the **multiple-comparisons problem**: every extra
variant is another chance for a false positive, so the family-wise error rate climbs unless
corrected. It also reveals less about **element interactions** than a multivariate test.

Example
-------

Optimising newsletter sign-ups across four calls to action at 25% traffic each, after two
weeks A converts 5%, B 6%, **C 7.5%** and D 5.2%. C is the significant winner and ships.

----

**Mind map — connected ideas**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>` · :doc:`Type I Error <080-type-i-error>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `A/B/n Test <https://insightful-data-lab.com/2025/08/24/a-b-n-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
