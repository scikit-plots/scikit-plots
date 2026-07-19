.. _bda-constructing-a-parameterized-prior-distribution:

========================================================================
Constructing a Parameterized Prior Distribution
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 033 of 144  ·  *beginner*

:doc:`◀ Previous · Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`   ·   :doc:`Next · Exchangeability and hierarchical models ▶ <034-exchangeability-and-hierarchical-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Where does the prior come from?
---------------------------------

Every analysis so far fixed the prior's parameters by hand: :math:`\mathrm{Beta}(2, 8)`,
:math:`\mathrm{N}(0, 2.5^2)`. But in the cancer-rate example the honest answer was that the "national
rate" :math:`\alpha/\beta` should have been **learned from the counties themselves**. That single move
— treating the prior's parameters as **unknowns with their own distribution** — is the whole of
hierarchical modelling.

Hyperparameters and hyperpriors
---------------------------------

Give the prior parameters a name and a distribution. The parameters :math:`\phi` of the prior
:math:`p(\theta \mid \phi)` are **hyperparameters**; the distribution :math:`p(\phi)` placed over them
is a **hyperprior**. The joint model gains a level:

.. math::

   p(\phi, \theta, y) = \underbrace{p(\phi)}_{\text{hyperprior}}
                        \; \underbrace{p(\theta \mid \phi)}_{\text{population}}
                        \; \underbrace{p(y \mid \theta)}_{\text{likelihood}} .

Nothing new is required to fit it: condition on :math:`y`, and marginalise whatever you do not want.
The population distribution :math:`p(\theta \mid \phi)` is now a **prior estimated from data** — not
circular, because it is estimated from *all* the groups jointly while each group's :math:`\theta_j` is
informed by its own data.

Why this is not cheating
--------------------------

The obvious worry: are we using the data twice? No. The model is a single joint distribution written
down **before** seeing :math:`y`, and Bayes' rule is applied once. What looks like "learning the prior"
is simply the marginal posterior :math:`p(\phi \mid y)` — inference about a parameter that happens to
sit one level up. The distinction from **empirical Bayes**, which plugs in a point estimate
:math:`\hat{\phi}` and thereby understates uncertainty, is exactly the nuisance-parameter lesson: a
full Bayesian analysis **integrates** over :math:`\phi`.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # hyperprior: the population's location and spread are unknowns
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfNormal("tau", 5)
       # population distribution: each group's parameter drawn from it
       theta = pm.Normal("theta", mu, tau, shape=J)
       pm.Normal("y", theta[group], sigma_obs, observed=y)

What it buys
--------------

Three things, all visible in the next lessons. Estimates for data-poor groups are **stabilised** by
borrowing strength from the others. The **amount** of borrowing is not chosen by the analyst but
**inferred** from how similar the groups turn out to be. And the uncertainty in that similarity
propagates into every group's interval. But the construction needs a justification — why should
:math:`\theta_j` share a distribution at all? That is **exchangeability**, and it is the next lesson.

.. hint::

   **Related lessons:** :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Informative Prior Distribution for Cancer Rates <017-informative-prior-distribution-for-cancer-rates>`  ·  :doc:`Bayesian analysis of conjugate hierarchical models <035-bayesian-analysis-of-conjugate-hierarchical-models>`  ·  :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/constructing-a-parameterized-prior-distribution/ <https://insightful-data-lab.com/2025/11/09/constructing-a-parameterized-prior-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
