:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-off-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🌊&nbsp;&nbsp;<b>Off-Distribution</b></div>`

==================
Off-Distribution
==================

*Inputs that fall outside the distribution a model was trained on.*

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

**Off-distribution** data are points that **differ significantly from the distribution the model was
trained on** — inputs outside its "familiar range". Models assume production data is drawn from the
**same distribution** as training (the i.i.d. assumption); when that breaks, the input is
off-distribution, also called **out-of-distribution (OOD)**.

Examples
--------

A cats-versus-dogs classifier shown a **giraffe** is off-distribution. A credit model trained on
2015-2020 applications meets **post-COVID** borrower behaviour it never saw. A diagnostic model
trained on adult MRIs is handed a **child's** scan. In each case the input falls outside the learned
scope.

Why it's a problem, and detecting it
--------------------------------------

Models optimised for in-distribution data make **unreliable or overconfident** predictions on OOD
inputs, with fairness risks for unseen subgroups. Detection draws on **distance metrics** (KL,
Jensen-Shannon, KS), **embedding methods** (Mahalanobis or cosine distance in latent space),
**uncertainty estimation** (Bayesian neural nets, deep ensembles, MC dropout), and dedicated **OOD
classifiers**.

Handling it
-----------

Five responses: **augment** the training data to broaden coverage; **adapt** the model to the new
domain; build **robust** models with regularisation or adversarial training; add a **reject option**
so the model can abstain ("I don't know"); and run **monitoring pipelines** to flag drift in
production. Off-distribution is the abrupt cousin of gradual **data** and **concept drift**.

----

*Theme:* :ref:`Distribution Shift & Drift <term-theme-drift>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Data Drift <331-data-drift>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Representation Shift <174-representation-shift>`

----

.. hint::
   **More in Distribution Shift & Drift**

   :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>` · :doc:`Categorical Drift <179-categorical-drift>` · :doc:`Categorical Explosions <182-categorical-explosions>` · :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>` · :doc:`Concept Drift <330-concept-drift>` · :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>` · :doc:`Data Drift <331-data-drift>` · :doc:`Dataset Shift <353-dataset-shift>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Drift Guardrails <349-drift-guardrails>` · :doc:`Energy Distance <176-energy-distance>` · :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>` · :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>` · :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Off-Distribution <https://insightful-data-lab.com/2025/08/23/off-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
