.. _bda-varying-intercepts-and-slopes:

========================================================================
Varying intercepts and slopes
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 102 of 144  ·  *advanced*

:doc:`◀ Previous · Interpreting a normal prior distribution as extra data <101-interpreting-a-normal-prior-distribution-as-extra-data>`   ·   :doc:`Next · Computation: batching and transformation ▶ <103-computation-batching-and-transformation>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Let the relationship itself vary
----------------------------------

A varying-intercept model lets each group have its own baseline but forces a **common slope** — every
group responds to the predictor identically. Often that is wrong: the effect of income on voting differs
by state, the effect of a treatment differs by clinic. The **varying-intercept, varying-slope** model
lets both the level *and* the response vary by group, pooling each toward a common value.

The model
-----------

Write the coefficient **vector** for group :math:`j` — intercept and slope together — as a draw from a
common multivariate distribution:

.. math::

   y_i = \alpha_{j[i]} + \beta_{j[i]} \, x_i + \epsilon_i, \qquad
   \begin{pmatrix} \alpha_j \\ \beta_j \end{pmatrix}
   \sim \mathrm{N}\!\left(
     \begin{pmatrix} \mu_\alpha \\ \mu_\beta \end{pmatrix}, \;
     \Sigma \right).

The notation :math:`j[i]` — group :math:`j` containing observation :math:`i` — is the multilevel
convention: the data :math:`y_i` exist at the individual level, and the grouping enters as an index, not
as a reordering. Each group's intercept and slope are **partially pooled** toward the population means,
by amounts the data determine through :math:`\Sigma`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=2)                  # population (intercept, slope)
       sd = pm.HalfNormal("sd", 1, shape=2)                 # scales of each
       L = pm.LKJCholeskyCov("L", n=2, eta=2,               # correlation + scales
                             sd_dist=pm.HalfNormal.dist(1))
       z = pm.Normal("z", 0, 1, shape=(n_groups, 2))
       ab = pm.Deterministic("ab", mu + z @ L.T)            # non-centred group coeffs
       mu_i = ab[group, 0] + ab[group, 1] * x
       pm.Normal("y", mu_i, pm.HalfNormal("s", 1), observed=y)

Fixed effects are a special case
----------------------------------

The multilevel frame **subsumes** the classical distinction. A "fixed effect" is the limit of a random
effect as its group-level variance goes to **infinity** — no pooling, each group estimated on its own
data. A single pooled coefficient is the **zero-variance** limit. The varying-slope model sits between,
and — crucially — **estimates where**, rather than forcing you to choose. There is no separate machinery
for fixed versus random; there is one model with a variance the data inform.

Why pool the slopes
---------------------

The same argument as always, now for effects rather than means. A group with little data gets an
**unstable** slope on its own; pooling toward the population slope stabilises it, trading a little bias
for a large variance reduction. Groups with ample data are barely moved. And the population-level
:math:`\mu_\beta` and :math:`\Sigma` are themselves of interest: they describe **how much** the effect
varies across groups — often the substantive question. The one subtlety, addressed next, is that
intercepts and slopes typically **covary**, and modelling that correlation is what :math:`\Sigma` is for.

.. hint::

   **Related lessons:** :doc:`Regression coeﬃcients exchangeable in batches <099-regression-coefficients-exchangeable-in-batches>`  ·  :doc:`Computation: batching and transformation <103-computation-batching-and-transformation>`  ·  :doc:`Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/24/varying-intercepts-and-slopes/ <https://insightful-data-lab.com/2025/11/24/varying-intercepts-and-slopes/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
