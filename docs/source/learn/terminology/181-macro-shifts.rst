:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-shifts:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Macro Shifts</b></div>`

==============
Macro Shifts
==============

*Broad, large-scale changes in data distribution affecting many features.*

What it is
----------

**Macro shifts** are **large-scale, external changes** in the broader environment — economic, social,
political or technological — big enough to move markets and break models. In ML terms they are
**system-wide distribution changes**, structural shifts well beyond ordinary small drift and usually
outside the business's control.

Examples
--------

The pattern recurs across domains. A **global recession** reshapes consumer spending; a **pandemic**
collapses travel and surges e-commerce overnight; **inflation** rewrites buying habits. Each breaks
models trained on the old world — a pre-pandemic **credit-risk** model misreads new borrower
behaviour, **demand forecasts** built on old habits miss, and **supply-chain** lead times jump after
a geopolitical disruption.

Why they matter, and detecting them
-------------------------------------

Models assume **stationarity** — that the future resembles the past — and macro shifts shatter that
assumption, causing prediction failure, strategic risk, and new **fairness** problems. They are
caught with **drift measures** (PSI, KL or Jensen-Shannon divergence, KS tests), **performance
monitoring** (sudden AUC or lift drops), and **external signals** (economic indicators, policy
changes).

Responding to them
------------------

The playbook: **retrain** on post-shift data, prefer **adaptive** models (online or fast time-series
learners), run **scenario planning and stress tests**, keep **humans in the loop** under drastic
change, and **diversify data sources**. A macro shift is the broad external force that often drives
**concept drift** and **data drift** at the same time.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Concept Drift <330-concept-drift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>` · :doc:`Continuous Retraining <161-continuous-retraining>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro Shifts <https://insightful-data-lab.com/2025/08/23/macro-shifts/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
