.. _bda-bayesian-histograms:

========================================================================
Bayesian histograms
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 139 of 144  ·  *advanced*

:doc:`◀ Previous · Mixture models for classification and regression <138-mixture-models-for-classification-and-regression>`   ·   :doc:`Next · Dirichlet process prior distributions ▶ <140-dirichlet-process-prior-distributions>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The simplest nonparametric density
-------------------------------------

Before the elaborate machinery, the humblest density estimator deserves a Bayesian treatment: the
**histogram**. Partition the range into bins and estimate the probability in each. A Bayesian histogram
puts a **prior on the bin probabilities** and reports a posterior — turning a familiar picture into a
model with honest uncertainty, and illustrating nonparametric ideas at their most transparent.

The model
-----------

With :math:`B` bins and counts :math:`n_b` of the :math:`N` observations falling in each, the natural
model is multinomial with a **Dirichlet prior** on the bin probabilities:

.. math::

   (n_1, \dots, n_B) \sim \mathrm{Multinomial}(N, p), \qquad
   p \sim \mathrm{Dirichlet}(\alpha_1, \dots, \alpha_B).

Conjugacy makes the posterior immediate — :math:`p \mid n \sim \mathrm{Dirichlet}(\alpha_b + n_b)` — so
each bin's height comes with a full posterior, and sparsely-populated bins are **smoothed** toward the
prior instead of estimated as zero. The Dirichlet concentration controls that smoothing.

.. code-block:: python

   import numpy as np
   from scipy import stats
   counts, edges = np.histogram(y, bins=B)
   alpha = np.ones(B) * 1.0                                  # symmetric Dirichlet prior
   post = stats.dirichlet(alpha + counts)                    # conjugate posterior on bin probs
   heights = post.mean() / np.diff(edges)                    # density estimate
   draws = post.rvs(500) / np.diff(edges)                    # posterior uncertainty bands

Smoothing and the bin-width problem
-------------------------------------

The histogram's perennial weakness is the **bin width** — too wide oversmooths, too narrow is noisy.
Bayesian versions address it by borrowing this stage's tools: a **prior favouring smoothness** across
adjacent bins (neighbouring probabilities tied, as in the P-spline lesson) regularises the jagged
edges; the bin count or width can itself be **inferred** rather than fixed; and finer partitions can be
combined hierarchically. **Pólya trees** carry the idea to its logical end — a *nested*, recursively
refined binning with a prior at every level, giving a genuine nonparametric prior on densities built
from binary splits.

Where it fits
---------------

The Bayesian histogram is the **conceptual bridge** into nonparametrics. It shows the essential moves —
a prior over a flexible, high-dimensional object (the bin probabilities), a posterior that **smooths where
data are thin**, and uncertainty on the whole estimated density — in the simplest possible container. The
Dirichlet prior here is the finite ancestor of the **Dirichlet process** of the next lessons, which
replaces a fixed grid of bins with an adaptive, unbounded partition. From counting in boxes to infinite
mixtures is one continuous idea.

.. hint::

   **Related lessons:** :doc:`Density estimation and regression <133-density-estimation-and-regression>`  ·  :doc:`Dirichlet process prior distributions <140-dirichlet-process-prior-distributions>`  ·  :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/13/bayesian-histograms/ <https://insightful-data-lab.com/2025/12/13/bayesian-histograms/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
