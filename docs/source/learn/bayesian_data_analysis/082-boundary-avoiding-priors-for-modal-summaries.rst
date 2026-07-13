.. _bda-boundary-avoiding-priors-for-modal-summaries:

========================================================================
Boundary-avoiding priors for modal summaries
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 082 of 144  ·  *intermediate*

:doc:`◀ Previous · Finding posterior modes <081-finding-posterior-modes>`   ·   :doc:`Next · Normal and related mixture approximations ▶ <083-normal-and-related-mixture-approximations>`   ·   :doc:`↑ Section <index>`


Zero variance, implausibly
----------------------------

Fit a hierarchical model by maximum likelihood with only a few groups, and a familiar pathology appears:
the estimated group-level variance is exactly **zero**. Not small — zero. The likelihood's maximum lies
on the boundary of the parameter space, and the software reports complete pooling as a certainty.
Anyone who believes the groups differ at all knows the estimate is wrong.

Why it happens
----------------

With :math:`J` small, the between-group variation observed in the data can easily be **no larger than**
what sampling noise alone would produce. The likelihood for :math:`\tau` is then monotonically
decreasing, so its maximum is at :math:`\tau = 0`. Nothing is broken; the *mode* is simply a bad
summary of a posterior that has plenty of mass at positive :math:`\tau`. Full Bayes avoids the problem
by **averaging** rather than maximising — but modal estimation is popular for speed, so it needs a fix
of its own.

The fix: penalise the boundary
--------------------------------

Chung, Rabe-Hesketh, Dorie, Gelman and Liu proposed a **boundary-avoiding prior**: put a
:math:`\mathrm{Gamma}(\alpha, \lambda)` prior on the group-level **standard deviation** with shape
:math:`\alpha > 1`, so that the density vanishes at zero and the posterior mode cannot land there.

.. math::

   p(\tau) \propto \tau^{\alpha - 1} e^{-\lambda \tau}, \qquad \alpha = 2, \; \lambda \to 0 .

With :math:`\alpha = 2` the prior is proportional to :math:`\tau` near the origin — enough to push the
mode off the boundary, not enough to fight real data. The default :math:`\mathrm{Gamma}(2, \lambda \to
0)` has three properties worth stating precisely: the penalised estimate is **approximately one standard
error from zero** when the ML estimate is zero; it closely approximates the **posterior median** under a
noninformative prior; and as the number of groups grows it **coincides with maximum likelihood**. The
prior does work exactly where the data are silent, and withdraws where they speak.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       # boundary-avoiding: density -> 0 as tau -> 0, so the MODE stays positive
       tau = pm.Gamma("tau", alpha=2, beta=1e-4)
       # (for full-Bayes sampling, HalfNormal/HalfCauchy remain the default;
       #  boundary-avoidance is specifically about modal summaries)
       ...

Covariance matrices too
-------------------------

The same authors extended the idea to group-level **covariance** matrices, where maximum likelihood
routinely returns a **singular** estimate: use a **Wishart** prior — deliberately *not* the
inverse-Wishart of Stage 3 — with degrees of freedom equal to the number of varying coefficients plus
two, which keeps the modal estimate positive definite.

The moral generalises. A prior chosen for **modal** estimation is not the prior you would choose for
**sampling**: it must shape the density's *peak*, not merely its mass. Naming which summary you intend
is part of specifying the model.

.. hint::

   **Related lessons:** :doc:`Finding posterior modes <081-finding-posterior-modes>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`  ·  :doc:`Counterexamples to large-sample (asymptotic) Bayesian theorems <030-counterexamples-to-large-sample-asymptotic-bayesian-theorems>`  ·  :doc:`Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/boundary-avoiding-priors-for-modal-summaries/ <https://insightful-data-lab.com/2025/11/22/boundary-avoiding-priors-for-modal-summaries/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
