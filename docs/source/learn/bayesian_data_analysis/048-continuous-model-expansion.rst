.. _bda-continuous-model-expansion:

========================================================================
Continuous model expansion
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 048 of 144  ·  *intermediate*

:doc:`◀ Previous · Model comparison using Bayes factors <047-model-comparison-using-bayes-factors>`   ·   :doc:`Next · Implicit assumptions and model expansion: an example ▶ <049-implicit-assumptions-and-model-expansion-an-example>`   ·   :doc:`↑ Section <index>`


Don't choose — embed
----------------------

Model comparison presumes a **discrete** menu: :math:`M_1` or :math:`M_2`. But rival models are usually
special cases of a richer one, and the more informative move is to **embed** them in a continuous
family, fit it, and let the posterior for the extra parameter say how much of each the data want. The
choice becomes an **estimate**, and its uncertainty is reported rather than hidden.

The pattern
-------------

Introduce a parameter whose values recover the candidates as limits:

* Normal errors versus heavy tails: fit a :math:`t` with **unknown degrees of freedom** :math:`\nu`.
  :math:`\nu \to \infty` is the normal; a posterior for :math:`\nu` concentrated near 4 says the data
  want heavy tails, and by how much.
* Complete versus no pooling: the hierarchical model with unknown :math:`\tau` already **contains**
  both, at :math:`\tau = 0` and :math:`\tau \to \infty`.
* Fixed versus varying slopes: allow varying slopes with a variance parameter that can shrink to zero.
* Linear versus nonlinear: add a spline or GP term whose amplitude can vanish.

Each replaces a hypothesis test with a **continuous parameter and a posterior**.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # embeds normal (nu -> inf) and heavy-tailed alternatives in one model
       nu = pm.Gamma("nu", alpha=2, beta=0.1)      # weakly informative, mass over 2..50
       sigma = pm.HalfNormal("sigma", 1)
       pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y)
   # posterior for nu answers "how non-normal?" instead of "normal: yes/no?"

Why this is better
--------------------

Three reasons. It **propagates uncertainty**: selecting a model and then analysing as if it were true
ignores the uncertainty in the selection, and understates the final intervals. It avoids the
**discontinuity** of a decision rule that flips on an arbitrary threshold. And it turns a failed
posterior predictive check into a **research direction**: the check told you *which* feature was missed;
the expansion adds a parameter for exactly that feature and asks the data how big it is.

The discipline
----------------

Expansion is not licence to add everything. Each new parameter needs a **weakly informative prior**
(otherwise a barely-identified addition wanders), and the expanded model must itself be **checked** —
the loop from the first lesson of this stage. Nor is expansion free: it can make the geometry harder
(the funnel), so diagnostics matter more. But the direction of travel is right, and it is the workflow
that the remainder of this course follows: fit, check, find the misfit, **expand**, refit.

.. hint::

   **Related lessons:** :doc:`Model comparison based on predictive performance <046-model-comparison-based-on-predictive-performance>`  ·  :doc:`Implicit assumptions and model expansion: an example <049-implicit-assumptions-and-model-expansion-an-example>`  ·  :doc:`The Place of Model Checking in Applied Bayesian Statistics <040-the-place-of-model-checking-in-applied-bayesian-statistics>`  ·  :doc:`Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/continuous-model-expansion/ <https://insightful-data-lab.com/2025/11/10/continuous-model-expansion/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
