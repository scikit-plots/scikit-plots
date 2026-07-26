.. _bda-general-notation-for-statistical-inference:

========================================================================
General Notation for Statistical Inference
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 002 of 144  ·  *beginner*

:doc:`◀ Previous · The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>`   ·   :doc:`Next · Bayesian Inference ▶ <003-bayesian-inference>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The three symbols
-------------------

Bayesian writing is remarkably compact once its notation is fixed. Three symbols carry most of the
weight:

* :math:`\theta` — the **unobservable** quantities: parameters, or anything else you want to learn.
* :math:`y` — the **observed** data.
* :math:`\tilde{y}` — **potentially observable** data: future or replicated observations you have not
  yet seen. (Predictors, when present, are written :math:`x`.)

The joint model
-----------------

A full probability model is a **joint distribution** over the unknowns and the observables, which
always factors into prior times likelihood:

.. math::

   p(\theta, y) = p(\theta)\, p(y \mid \theta).

Everything else follows by conditioning and marginalising. Inference about parameters is
:math:`p(\theta \mid y)`; prediction of new data is the **posterior predictive**
:math:`p(\tilde{y} \mid y)`, obtained by averaging the likelihood over the posterior.

Exchangeability
-----------------

Why is it ever legitimate to model observations as **identically distributed**? The Bayesian answer is
**exchangeability**: if the labels on the observations carry no information — if any reordering of
:math:`y_1, \dots, y_n` is equally plausible — then their joint distribution can be written as
independent draws given some parameter, mixed over a distribution for that parameter. Exchangeability,
not an assumption of "random sampling", is what licenses the usual iid likelihood, and it becomes the
foundation of hierarchical models later in this course.

Read the conditioning bar
---------------------------

One habit repays itself constantly: read every vertical bar as "**given**", and check what is on its
right. :math:`p(y \mid \theta)` and :math:`p(\theta \mid y)` are built from the same joint model but
answer opposite questions. Bayesian and frequentist methods differ mainly in **what they condition on**
— the Bayesian conditions on the data actually observed and treats :math:`\theta` as random; the
frequentist conditions on :math:`\theta` and treats the data as random.

.. hint::

   **Related lessons:** :doc:`The three steps of Bayesian data analysis <001-the-three-steps-of-bayesian-data-analysis>`  ·  :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/general-notation-for-statistical-inference/ <https://insightful-data-lab.com/2025/11/08/general-notation-for-statistical-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
