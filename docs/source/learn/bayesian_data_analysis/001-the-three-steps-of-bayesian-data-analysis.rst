.. _bda-the-three-steps-of-bayesian-data-analysis:

========================================================================
The three steps of Bayesian data analysis
========================================================================

**Part 1 · Stage 1 · 🎲 The Bayesian Idea**  ·  Lesson 001 of 144  ·  *beginner*

:doc:`Next · General Notation for Statistical Inference ▶ <002-general-notation-for-statistical-inference>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

A process, not a formula
--------------------------

Bayesian data analysis is often reduced to a single equation, but in practice it is a **three-step
process** — and the equation is only the middle step. Naming the steps makes the workflow explicit and
shows where the real work lies: not in applying Bayes' rule (mechanical), but in **building** the model
and **checking** it.

The three steps
-----------------

1. **Set up a full probability model** — a joint distribution over everything unknown and everything
   observed. This means choosing a **likelihood** :math:`p(y \mid \theta)` for how the data arise, and
   a **prior** :math:`p(\theta)` for the parameters, consistent with what you know about the problem.
2. **Condition on the observed data** — compute and interpret the **posterior**
   :math:`p(\theta \mid y)`, the conditional distribution of the unknowns given what was actually seen.
3. **Evaluate the fit** — does the model describe the data? Are its implications reasonable? Are the
   conclusions sensitive to the assumptions? If not, refine the model and return to step 1.

The middle step
-----------------

Only the second step is fixed by mathematics. It is Bayes' rule:

.. math::

   p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}
   \;\;\propto\;\; \underbrace{p(y \mid \theta)}_{\text{likelihood}} \;
   \underbrace{p(\theta)}_{\text{prior}} .

The denominator :math:`p(y)` is a normalising constant, so for inference about :math:`\theta` the
proportionality is what matters. In code the whole pipeline is short:

.. code-block:: python

   import pymc as pm

   with pm.Model() as model:            # step 1: the full probability model
       theta = pm.Beta("theta", 1, 1)   #   prior
       pm.Binomial("y", n=10, p=theta, observed=8)   # likelihood
       idata = pm.sample(2000, tune=1000)            # step 2: condition on data
   # step 3: check the fit (posterior predictive, diagnostics) — later lessons

Uncertainty, quantified directly
----------------------------------

What distinguishes the approach is the **direct quantification of uncertainty**: the answer is a whole
distribution, not a point estimate with an asterisk. And step 3 is not a formality — a model is a
*simplification*, and a Bayesian analysis is only as trustworthy as the checks in its final step. The
three steps are also a **loop**, iterated as each check teaches you what the model missed.

.. hint::

   **Related lessons:** :doc:`General Notation for Statistical Inference <002-general-notation-for-statistical-inference>`  ·  :doc:`Bayesian Inference <003-bayesian-inference>`  ·  :doc:`The Place of Model Checking in Applied Bayesian Statistics <040-the-place-of-model-checking-in-applied-bayesian-statistics>`  ·  :doc:`Bayesian Inference in Applied Statistics <010-bayesian-inference-in-applied-statistics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/08/the-three-steps-of-bayesian-data-analysis/ <https://insightful-data-lab.com/2025/11/08/the-three-steps-of-bayesian-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
