.. _bda-example-reaction-times-and-schizophrenia:

========================================================================
Example: reaction times and schizophrenia
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 135 of 144  ·  *advanced*

:doc:`◀ Previous · Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`   ·   :doc:`Next · Label switching and posterior computation ▶ <136-label-switching-and-posterior-computation>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A mixture with a scientific meaning
-------------------------------------

Not every mixture is a mere density approximation — sometimes the components are **real and
interpretable**. A study of reaction times, comparing schizophrenic patients with a control group,
provides the archetype: the mixture structure encodes a genuine hypothesis about how the mind produces
the data.

The finding, and the model
----------------------------

Schizophrenic patients show more variable reaction times than controls, but the excess is not a uniform
slowing. The data suggest that on **most** trials a patient responds like a control, while on a
**minority** of trials an attentional lapse produces a much slower response. That is a **mixture within
each patient**: two components, a "normal" reaction-time distribution and a "delayed" one, with a
patient-specific probability of lapsing.

.. math::

   y_{ij} \sim \lambda_i \, \mathrm{N}(\mu_i, \sigma^2)
              + (1 - \lambda_i) \, \mathrm{N}(\mu_i + \tau, \sigma^2),

where :math:`\lambda_i` is patient :math:`i`'s probability of a normal (non-lapse) response and
:math:`\tau > 0` the extra delay when attention lapses. The parameters carry meaning: :math:`\tau` is the
size of a lapse, :math:`\lambda_i` how often patient :math:`i` lapses.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hierarchical: each patient's lapse probability, pooled across patients
       lam = pm.Beta("lam", 2, 2, shape=n_patients)         # P(normal response)
       mu = pm.Normal("mu", 0, 5, shape=n_patients)
       tau = pm.HalfNormal("tau", 5)                         # lapse delay (ordered => identified)
       w = pm.math.stack([lam[pid], 1 - lam[pid]], axis=1)
       comp = [pm.Normal.dist(mu[pid], sigma), pm.Normal.dist(mu[pid] + tau, sigma)]
       pm.Mixture("y", w=w, comp_dists=comp, observed=rt)

Why this example matters
--------------------------

Three points. The components are **substantive**, not basis elements — "normal response" and
"attentional lapse" are the hypothesis, and :math:`\lambda_i`, :math:`\tau` are the quantities of
scientific interest. The **hierarchy** ties it to the rest of the book: lapse probabilities are pooled
across patients, so a patient with few trials borrows strength. And the ordering :math:`\tau > 0`
**identifies** the components — the delayed component is, by construction, the slower one — which
sidesteps the label-switching problem the next lesson tackles in general. A mixture, here, is a **model of
a mechanism**: two cognitive states, their sizes and frequencies estimated from response times.

.. hint::

   **Related lessons:** :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Label switching and posterior computation <136-label-switching-and-posterior-computation>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Mixture models for classification and regression <138-mixture-models-for-classification-and-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/example-reaction-times-and-schizophrenia/ <https://insightful-data-lab.com/2025/12/09/example-reaction-times-and-schizophrenia/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
