.. _bda-finding-marginal-posterior-modes-using-em:

========================================================================
Finding marginal posterior modes using EM
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 084 of 144  ·  *intermediate*

:doc:`◀ Previous · Normal and related mixture approximations <083-normal-and-related-mixture-approximations>`   ·   :doc:`Next · Conditional and marginal posterior approximations ▶ <085-conditional-and-marginal-posterior-approximations>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Modes of what, exactly?
-------------------------

In a hierarchical model the joint mode of :math:`(\theta, \phi)` — group parameters and hyperparameters
together — is often useless: it is the point where all :math:`\theta_j` equal :math:`\mu` and
:math:`\tau = 0`, the degenerate solution of the boundary lesson. What you usually want is the
**marginal** mode of the hyperparameters,

.. math::

   \hat{\phi} = \arg\max_{\phi} \; p(\phi \mid y)
              = \arg\max_{\phi} \int p(\theta, \phi \mid y) \, d\theta ,

with the nuisance :math:`\theta` **integrated out** rather than maximised over. The integral usually has
no closed form, and **EM** computes the maximiser without doing it directly.

The algorithm
---------------

Treat :math:`\theta` as **missing data**. Alternate:

* **E-step.** Given the current :math:`\phi^{(t)}`, form the expected complete-data log posterior,
  averaging over the conditional distribution of the missing parameters:

  .. math::

     Q\bigl(\phi \mid \phi^{(t)}\bigr) =
     \mathrm{E}_{\theta \mid \phi^{(t)}, y}\bigl[\log p(\theta, \phi \mid y)\bigr].

* **M-step.** Maximise :math:`Q` over :math:`\phi` to get :math:`\phi^{(t+1)}`.

Each iteration **cannot decrease** the marginal posterior density — the guarantee that makes EM stable
without a step size. It converges to a **local** mode, monotonically, from wherever it starts.

.. code-block:: python

   import numpy as np
   # hierarchical normal: E-step gives E[theta_j], var[theta_j]; M-step updates mu, tau
   for _ in range(n_iter):
       V = 1 / (1 / sigma**2 + 1 / tau**2)              # E-step: conditional moments
       Etheta = V * (ybar / sigma**2 + mu / tau**2)
       mu = Etheta.mean()                                # M-step: closed-form maximisers
       tau = np.sqrt(np.mean((Etheta - mu) ** 2 + V))    # note: +V, the E-step variance

That ``+ V`` is the whole point: EM adds back the **uncertainty** in :math:`\theta`, which a naive
"plug in the estimates and maximise" would discard, and which is why the resulting :math:`\tau` is not
biased toward zero.

Uses, and the ceiling
-----------------------

EM shines where the complete-data problem is easy: mixture models (the latent component labels are the
missing data), models with censoring, factor models, and hierarchical normal models. Variants extend it
— **ECM** for hard M-steps, **SEM** for standard errors, **MCEM** when the E-step needs simulation.

Its ceiling is inherent. EM returns a **point**, not a distribution: the curvature at the mode gives an
approximate covariance, but the uncertainty in :math:`\phi` is not propagated into inferences about
:math:`\theta`, which is the empirical-Bayes understatement flagged in Stage 5. EM is the right tool when
:math:`n` is large, the posterior is regular, and speed matters — and a **starting point** for full Bayes
otherwise.

.. hint::

   **Related lessons:** :doc:`Finding posterior modes <081-finding-posterior-modes>`  ·  :doc:`Averaging Over Nuisance Parameters <020-averaging-over-nuisance-parameters>`  ·  :doc:`Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`  ·  :doc:`Setting up and interpreting mixture models <134-setting-up-and-interpreting-mixture-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/finding-marginal-posterior-modes-using-em/ <https://insightful-data-lab.com/2025/11/22/finding-marginal-posterior-modes-using-em/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
