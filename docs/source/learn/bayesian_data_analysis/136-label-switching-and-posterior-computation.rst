.. _bda-label-switching-and-posterior-computation:

========================================================================
Label switching and posterior computation
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 136 of 144  ·  *advanced*

:doc:`◀ Previous · Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>`   ·   :doc:`Next · Unspecified number of mixture components ▶ <137-unspecified-number-of-mixture-components>`   ·   :doc:`↑ Section <index>`


The symmetry problem
----------------------

Mixture models carry a computational pathology absent from everything before them. If the components
share a prior and come from the same family, then **permuting the component labels leaves the model
unchanged** — calling component 1 "component 2" and vice versa gives an identical likelihood and prior.
The posterior is therefore invariant under all :math:`K!` relabellings, and this is **label switching**.

Why it breaks inference
-------------------------

The posterior has :math:`K!` identical modes, one per permutation. A sampler that mixes well **visits all
of them**, so a chain's draws for "the mean of component 1" jump between what are really different
components — averaging them gives the overall mean of all components, a meaningless number. Any inference
that refers to a component *by its label* is corrupted: the marginal posterior of :math:`\mu_1` is
identical to that of :math:`\mu_2`, both equal to the mixture of all components' means.

.. code-block:: python

   import numpy as np
   # symptom: per-component means are identical across the chain because labels swap
   mu = idata.posterior["mu"].values.reshape(-1, K)
   mu.mean(axis=0)          # all K entries nearly equal  <-  label switching

Solutions
-----------

Three standard remedies, in rough order of robustness:

* **Ordering constraint.** Impose an identifying restriction — :math:`\mu_1 < \mu_2 < \cdots` or ordered
  weights — that picks one labelling. Simple and often enough, as the reaction-time delay
  (:math:`\tau > 0`) showed; but a poor choice of ordering variable can distort a genuinely multimodal
  posterior.
* **Post-hoc relabelling.** Let the sampler run unconstrained, then **permute each draw** to a common
  labelling by solving an assignment problem — matching components across draws by their parameters.
  More flexible than a hard constraint and the standard modern approach.
* **Label-invariant summaries.** Report only quantities that do **not** depend on labels — the density
  :math:`p(y)`, the number of components, the clustering of observations (which points group together) —
  all unchanged by permutation.

The deeper lesson
-------------------

Which fix to use follows from the **goal**. For **density estimation**, label switching is a *non-problem*:
the estimated density is label-invariant, so an unconstrained chain gives exactly the right answer with
nothing to fix. It bites only for **component-specific** interpretation — "what is the mean of the second
group?" — which presupposes the components are real and distinguishable. So the presence of a
label-switching problem is a signal to ask whether component-level inference is even meaningful for the
question, or whether a label-invariant summary answers it directly.

.. hint::

   **Related lessons:** :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Unspecified number of mixture components <137-unspecified-number-of-mixture-components>`  ·  :doc:`Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>`  ·  :doc:`Density estimation and regression <133-density-estimation-and-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/label-switching-and-posterior-computation/ <https://insightful-data-lab.com/2025/12/09/label-switching-and-posterior-computation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
