.. _bda-overdispersed-versions-of-standard-models:

========================================================================
Overdispersed versions of standard models
========================================================================

**Part 4 · Stage 14 · 🛡️ Robustness & Missing Data**  ·  Lesson 114 of 144  ·  *advanced*

:doc:`◀ Previous · Aspects of robustness <113-aspects-of-robustness>`   ·   :doc:`Next · Posterior inference and computation ▶ <115-posterior-inference-and-computation>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Adding a tail to every likelihood
-----------------------------------

The standard likelihoods — normal, Poisson, binomial — each impose a **rigid variance**. Robustness is
bought by relaxing that rigidity: replace each with an **overdispersed** version carrying an extra
parameter that lets the tails or the variance grow. The pattern is uniform across families.

The normal to the t
---------------------

For continuous data, replace the normal with the **Student-**:math:`t`, whose degrees of freedom
:math:`\nu` tune tail weight:

.. math::

   y_i \sim t_\nu(\mu, \sigma), \qquad \nu > 0.

Small :math:`\nu` gives heavy tails that tolerate outliers; large :math:`\nu` recovers the normal (the
:math:`t` converges to it as :math:`\nu \to \infty`). Estimating :math:`\nu` lets the data decide **how
heavy-tailed** they are — a self-diagnosing robustness.

Poisson and binomial, overdispersed
-------------------------------------

The count families gain a variance parameter the same way. The **negative binomial** is a Poisson whose
rate is itself Gamma-distributed, giving variance that **exceeds** the mean by a controllable amount —
the closed-form counterpart of the log-normal error added by hand in the police-stops lesson. The
**beta-binomial** does the same for the binomial, letting the success probability vary across trials.

.. code-block:: python

   import pymc as pm
   # robust continuous: Student-t with estimated tail weight
   with pm.Model():
       nu = pm.Gamma("nu", 2, 0.1)                          # degrees of freedom, >~ 1
       pm.StudentT("y", nu=nu, mu=X @ beta, sigma=sigma, observed=y)

   # overdispersed counts: negative binomial
   with pm.Model():
       mu = pm.math.exp(X @ beta)
       alpha = pm.Exponential("alpha", 1)                   # dispersion; Poisson as alpha -> inf
       pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=counts)

Nested, so testable
---------------------

The elegance is that the standard model is a **special case** of its overdispersed version — Poisson is
negative-binomial with infinite dispersion, normal is :math:`t` with infinite :math:`\nu`. So fitting
the richer model and inspecting the extra parameter's posterior is a **direct test** of the restrictive
assumption: if :math:`\nu` is estimated small, the data are heavy-tailed and the normal was wrong; if the
negative-binomial dispersion is large, Poisson was adequate. You never have to choose in advance —
embed the standard model in its generalisation and let the posterior report which you needed.

The unifying idea
-------------------

Each overdispersed model adds **one parameter** for the variance the standard model fixed by fiat, and
partial pooling of that parameter connects straight back to the hierarchical stage: the police-stops
model *built* overdispersion from a normal error term, while the negative binomial *packages* the same
idea in closed form. Robustness, here, is just refusing to let a convenient distribution dictate how much
your data are allowed to vary.

.. hint::

   **Related lessons:** :doc:`Aspects of robustness <113-aspects-of-robustness>`  ·  :doc:`Posterior inference and computation <115-posterior-inference-and-computation>`  ·  :doc:`Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>`  ·  :doc:`Robust regression using t-distributed errors <117-robust-regression-using-t-distributed-errors>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/07/overdispersed-versions-of-standard-models/ <https://insightful-data-lab.com/2025/12/07/overdispersed-versions-of-standard-models/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
