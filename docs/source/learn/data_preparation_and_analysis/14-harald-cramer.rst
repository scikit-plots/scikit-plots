.. _dpa-harald-cramer:

========================================================================
Harald Cramér
========================================================================

**Stage 2 · 🔗 Associations & Correlation**  ·  Lesson 14 of 56  ·  *beginner*

:doc:`◀ Previous · Karl Pearson <13-karl-pearson>`   ·   :doc:`Next · What Are Statistical Tests? ▶ <15-what-are-statistical-tests>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A Swedish statistician
------------------------

**Harald Cramér** (1893–1985) was a Swedish mathematician and statistician whose name attaches to
several cornerstones of the field. Where Pearson built the discipline's foundations, Cramér gave much
of it its **rigorous mathematical footing**, bridging probability theory and statistics. Two of his
contributions appear directly in this course.

Cramér's V
------------

The first is **Cramér's V**, the standard measure of association between **two categorical (nominal)
variables** — the categorical counterpart to Pearson's :math:`r`. Built on Pearson's **chi-square
statistic**, it rescales it to a clean range of :math:`0` (no association) to :math:`1` (perfect):

.. math::

   V = \sqrt{\frac{\chi^2}{n \,\min(r - 1,\, c - 1)}},

where :math:`n` is the sample size and :math:`r, c` the numbers of rows and columns in the
contingency table. It fills the "categorical ↔ categorical" slot of the association taxonomy.

The Cramér–Rao bound
----------------------

The second is the **Cramér–Rao bound**, a deep result in **estimation** theory: it sets a hard
**lower limit** on the variance any unbiased estimator can achieve. An estimator that reaches it is as
precise as possible — a benchmark that underlies the **maximum-likelihood** methods appearing later in
this course.

A foundational text
---------------------

Cramér's influence also came through teaching. His 1946 book **"Mathematical Methods of Statistics"**
was among the first to put statistics on a firm measure-theoretic basis, and it trained a generation
of statisticians. Alongside the V and the bound, his broader work spanned probability, actuarial
mathematics and number theory.

.. hint::

   **Related lessons:** :doc:`Karl Pearson <13-karl-pearson>`  ·  :doc:`What Are Statistical Tests? <15-what-are-statistical-tests>`  ·  :doc:`Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`  ·  :doc:`Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/14/harald-cramer/ <https://insightful-data-lab.com/2026/01/14/harald-cramer/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, topic: data preparation, level: beginner
