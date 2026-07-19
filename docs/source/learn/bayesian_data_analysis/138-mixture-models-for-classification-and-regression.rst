.. _bda-mixture-models-for-classification-and-regression:

========================================================================
Mixture models for classification and regression
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 138 of 144  ·  *advanced*

:doc:`◀ Previous · Unspecified number of mixture components <137-unspecified-number-of-mixture-components>`   ·   :doc:`Next · Bayesian histograms ▶ <139-bayesian-histograms>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Mixtures for supervised learning
----------------------------------

Mixtures are usually met in **unsupervised** settings — clustering, density estimation. The same
structure powers **supervised** learning too: classification and regression gain flexibility when the
model is a mixture, either of the classes themselves or of local expert models. The latent-component idea
carries directly across.

Mixture discriminant analysis
-------------------------------

Ordinary discriminant analysis models each class as a single Gaussian — too rigid when a class is itself
**heterogeneous** (handwritten "4"s come in two styles; a disease has subtypes). Model each class as its
**own mixture** of Gaussians, and the decision boundary bends to the real, multimodal structure:

.. math::

   p(x \mid y = c) = \sum_{k=1}^{K_c} \pi_{ck} \, \mathrm{N}(x \mid \mu_{ck}, \Sigma_{ck}), \qquad
   \Pr(y = c \mid x) \propto p(x \mid y = c) \, \Pr(y = c),

with classification by the posterior class probability. Each class density is as flexible as it needs to
be, and the Bayesian fit carries uncertainty into the predicted class probabilities.

Mixture of experts
--------------------

For regression, a **mixture of experts** lets *different regressions hold in different regions* of the
predictor space. Several "expert" models each fit part of the space, and a **gating** function — itself a
function of the inputs — decides which expert governs where:

.. math::

   p(y \mid x) = \sum_{k=1}^{K} g_k(x) \, p_k(y \mid x), \qquad \sum_k g_k(x) = 1,

the gate :math:`g_k(x)` a softmax over the inputs. The result is a flexible regression that can switch
regime with :math:`x` — piecewise-linear where the experts are linear, but with soft, learned boundaries.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # gate: input-dependent component probabilities (softmax)
       Wg = pm.Normal("Wg", 0, 1, shape=(k_experts, X.shape[1]))
       g = pm.math.softmax(X @ Wg.T, axis=1)                # which expert, per input
       beta = pm.Normal("beta", 0, 1, shape=(k_experts, X.shape[1]))   # each expert's slope
       mu = (g * (X @ beta.T)).sum(axis=1)                  # gated combination
       pm.Normal("y", mu, pm.HalfNormal("s", 1), observed=y)

Why mixtures help here
------------------------

Two reasons close the supervised case. Mixtures grant **local flexibility** — different structure in
different regions or classes — without a global nonlinear form, and the components can carry
interpretation (a subtype, a regime). And because the whole apparatus is Bayesian, the **uncertainty in
which component applies** flows into predictions: a point near a class boundary or an expert's edge gets
an honestly wider predictive distribution. Mixtures thus extend supervised learning the same way they
extended density estimation — flexibility assembled from simple, interpretable local pieces.

.. hint::

   **Related lessons:** :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Example: reaction times and schizophrenia <135-example-reaction-times-and-schizophrenia>`  ·  :doc:`Density regression <144-density-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/09/mixture-models-for-classification-and-regression/ <https://insightful-data-lab.com/2025/12/09/mixture-models-for-classification-and-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
