.. _bda-hierarchical-dependence:

========================================================================
Hierarchical dependence
========================================================================

**Part 5 · Stage 16 · ♾️ Mixtures & Nonparametric Bayes**  ·  Lesson 143 of 144  ·  *advanced*

:doc:`◀ Previous · Beyond density estimation <142-beyond-density-estimation>`   ·   :doc:`Next · Density regression ▶ <144-density-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Sharing nonparametric structure across groups
-----------------------------------------------

Data often arrive in **groups** — documents in a corpus, patients across hospitals, measurements per
site — and each group may need its own flexible distribution. Fitting a **separate** DP per group shares
nothing; fitting **one** DP to the pooled data ignores group identity. The **hierarchical Dirichlet
process** (Teh, Jordan, Beal and Blei) resolves the tension, letting groups have distinct distributions
that **share components**.

The construction
------------------

Make the DPs themselves draws from a **higher-level** DP. A global :math:`G_0` is drawn from a top DP;
each group's :math:`G_j` is then drawn from a DP with :math:`G_0` as its base distribution:

.. math::

   G_0 \sim \mathrm{DP}(\gamma, H), \qquad
   G_j \mid G_0 \sim \mathrm{DP}(\alpha, G_0) \;\; \text{for each group } j .

Because :math:`G_0` is itself discrete, every group-level :math:`G_j` draws its atoms **from the same
shared set** — so a cluster discovered in one group is *available* to all, while each group has its own
weights over that common inventory. Groups differ in **how much** they use each component, not in the
components available.

.. code-block:: python

   # Chinese restaurant franchise: the HDP's metaphor
   #   each group is a restaurant; tables within a restaurant share DISHES from a global menu
   #   a popular dish (component) recurs across restaurants -> shared clusters
   #   new dish chosen w.p. ~ gamma (global);  new table w.p. ~ alpha (local)
   # collapsed Gibbs (Teh et al. 2006) reseats customers and re-serves dishes

Why sharing matters
---------------------

The HDP is the nonparametric version of the **partial pooling** that runs through this entire book —
groups borrow strength by sharing a common set of components, and the number of shared components is
**learned**, not fixed. Its most famous use is **topic modelling**: documents are groups, words are
observations, and the shared components are **topics** discovered across the corpus, with the number of
topics inferred rather than set. The same structure serves grouped density estimation, multi-population
clustering, and infinite hidden Markov models — anywhere related groups each need a flexible distribution
but ought to share what they have in common.

.. hint::

   **Related lessons:** :doc:`Dirichlet process mixtures <141-dirichlet-process-mixtures>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Beyond density estimation <142-beyond-density-estimation>`  ·  :doc:`State-level opinons from national polls <110-state-level-opinons-from-national-polls>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/13/hierarchical-dependence/ <https://insightful-data-lab.com/2025/12/13/hierarchical-dependence/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
