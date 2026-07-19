.. _bda-model-comparison-using-bayes-factors:

========================================================================
Model comparison using Bayes factors
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 047 of 144  ·  *intermediate*

:doc:`◀ Previous · Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`   ·   :doc:`Next · Continuous model expansion ▶ <048-continuous-model-expansion>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Betting on whole models
-------------------------

If the candidate models are treated as hypotheses with prior probabilities, Bayes' rule applies to
**them** as well. The **Bayes factor** is the ratio of the two **marginal likelihoods** — the evidence
term that inference could ignore, now doing all the work:

.. math::

   \mathrm{BF}_{12} = \frac{p(y \mid M_1)}{p(y \mid M_2)},
   \qquad
   p(y \mid M_k) = \int p(y \mid \theta_k, M_k) \; p(\theta_k \mid M_k) \; d\theta_k .

It multiplies the prior odds into posterior odds. Because the marginal likelihood **averages** the
likelihood over the prior, it automatically penalises models that spread their prior mass over regions
the data reject — an intrinsic Occam's razor, with no explicit parameter count.

Three difficulties
--------------------

That elegance carries a heavy bill.

* **Improper priors make it undefined.** An improper prior has an arbitrary normalising constant, which
  does not cancel; the Bayes factor takes an arbitrary value.
* **Vague proper priors favour the simpler model** — the **Jeffreys–Lindley (Bartlett) paradox**.
  Widening a prior dilutes the alternative's marginal likelihood, so a diffuse prior on the alternative
  drives the Bayes factor toward the null **no matter what the data say**, and the effect does not
  vanish as :math:`n` grows.
* **It is hard to compute.** The marginal likelihood is exactly the normalising constant that MCMC is
  built to avoid, and naive estimators of it are notoriously bad.

The sensitivity is the real objection
---------------------------------------

Notice what makes this different from ordinary prior sensitivity. Changing :math:`\mathrm{Beta}(1,1)` to
:math:`\mathrm{Beta}(30,30)` may barely move the **posterior** for :math:`\theta` — but it can change the
Bayes factor substantially, because the evidence integrates the likelihood against the prior itself. A
quantity that is insensitive where inference is sensitive, and sensitive where inference is not, is a
poor guide.

.. code-block:: python

   # Bayes factors demand priors you would defend as *predictions*, not as regularisers.
   # In practice, prefer:
   import arviz as az
   az.compare({"m1": idata1, "m2": idata2})   # predictive comparison, no marginal likelihood

When to use it
----------------

Bayes factors are appropriate when the model space is genuinely **discrete and exhaustive** — one of
these hypotheses is true — and the priors are honest, informative statements you would stake a
prediction on. That describes some scientific hypothesis tests and few applied models. Gelman's
recommended route is to **bypass the choice**: check models predictively, compare them by elpd, and,
where they disagree, **expand** rather than select — the subject of the next lesson.

.. hint::

   **Related lessons:** :doc:`Measures of predictive accuracy <045-measures-of-predictive-accuracy>`  ·  :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`  ·  :doc:`Noninformative Prior Distributions <018-noninformative-prior-distributions>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/model-comparison-using-bayes-factors/ <https://insightful-data-lab.com/2025/11/10/model-comparison-using-bayes-factors/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
