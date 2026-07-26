.. _bda-exchangeability-and-hierarchical-models:

========================================================================
Exchangeability and hierarchical models
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 034 of 144  ·  *beginner*

:doc:`◀ Previous · Constructing a Parameterized Prior Distribution <033-constructing-a-parameterized-prior-distribution>`   ·   :doc:`Next · Bayesian analysis of conjugate hierarchical models ▶ <035-bayesian-analysis-of-conjugate-hierarchical-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The licence to pool
---------------------

Hierarchical models let groups **borrow strength** from one another. What entitles them to? The answer
is **exchangeability**: the assumption that the group parameters are, before seeing the data,
**interchangeable** — that nothing in the labelling distinguishes them.

The definition
----------------

Parameters :math:`\theta_1, \dots, \theta_J` are exchangeable if their joint distribution is invariant
under any permutation :math:`\pi` of the indices:

.. math::

   p(\theta_1, \dots, \theta_J) = p(\theta_{\pi(1)}, \dots, \theta_{\pi(J)}) .

It is a statement of **ignorance, not of equality**. The schools may differ enormously; you simply have
no prior reason, before the data, to expect school 3 to differ from school 7 in any particular
direction. Independent and identically distributed is a special case; exchangeability is weaker and
more useful.

De Finetti's bridge
---------------------

De Finetti's theorem makes the connection precise: an exchangeable sequence can (in the infinite case)
be represented as **iid draws given some parameter**, mixed over a distribution for that parameter,

.. math::

   p(\theta_1, \dots, \theta_J)
   = \int \left[ \prod_{j=1}^{J} p(\theta_j \mid \phi) \right] p(\phi) \; d\phi .

Read the right-hand side: that **is** the hierarchical model of the last lesson. Exchangeability does
not merely permit hierarchy — it **implies** it. The population distribution :math:`p(\theta \mid \phi)`
is not an extra assumption bolted on; it is what exchangeability means.

Three pooling regimes
-----------------------

The hyperparameter governing spread (call it :math:`\tau`) interpolates between two extremes, and the
data choose where to sit:

* :math:`\tau \to 0` — **complete pooling**: all :math:`\theta_j` collapse to a common value; group
  identity is ignored.
* :math:`\tau \to \infty` — **no pooling**: each group is estimated alone, noisily.
* :math:`0 < \tau < \infty` — **partial pooling**: each estimate is shrunk toward the population mean
  by an amount reflecting its own precision and the between-group spread.

When exchangeability fails
----------------------------

If you **do** know something distinguishing — schools differ by funding, counties by population — then
the parameters are not exchangeable as they stand. The repair is not to abandon the hierarchy but to
make the known structure explicit: model :math:`\theta_j` as depending on covariates :math:`x_j`, and
assume exchangeability of the **residuals**. This is **conditional exchangeability**, and it is the
road from hierarchical models to hierarchical *regression* in Part IV.

.. hint::

   **Related lessons:** :doc:`Constructing a Parameterized Prior Distribution <033-constructing-a-parameterized-prior-distribution>`  ·  :doc:`General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`  ·  :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`  ·  :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/exchangeability-and-hierarchical-models/ <https://insightful-data-lab.com/2025/11/09/exchangeability-and-hierarchical-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
