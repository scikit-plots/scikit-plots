.. _bda-further-extensions-to-gibbs-and-metropolis:

========================================================================
Further extensions to Gibbs and Metropolis
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 077 of 144  ·  *intermediate*

:doc:`◀ Previous · Eﬃcient Metropolis jumping rules <076-efficient-metropolis-jumping-rules>`   ·   :doc:`Next · Hamiltonian Monte Carlo ▶ <078-hamiltonian-monte-carlo>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Beyond the two basics
-----------------------

Gibbs needs conditionals; Metropolis needs a tuned proposal. A family of extensions removes one
requirement or the other, usually by the same device: **enlarge the space** with auxiliary variables, so
that a hard move in :math:`\theta` becomes an easy move in :math:`(\theta, u)`.

Slice sampling
----------------

Introduce a height :math:`u` beneath the unnormalised density and sample **uniformly from the area under
the curve**. Given :math:`\theta`, draw :math:`u \sim \mathrm{Uniform}(0, q(\theta))`; given :math:`u`,
draw :math:`\theta` uniformly from the **slice** :math:`\{\theta : q(\theta) \ge u\}`. Marginally,
:math:`\theta` has the target distribution.

.. math::

   p(\theta, u) \propto \mathbf{1}\{0 < u < q(\theta)\}
   \quad \Longrightarrow \quad p(\theta) \propto q(\theta).

The appeal is that it is **self-tuning**: no step size, no acceptance rate. The slice is found by
stepping out from the current point and shrinking on rejection. Slice sampling is the workhorse for
one-dimensional conditionals inside a larger Gibbs sweep, and it handles awkward univariate shapes that
would defeat a fixed proposal.

Reversible jump
-----------------

When the **number of parameters is itself unknown** — how many mixture components, which predictors
belong in the regression — the posterior lives on a union of spaces of different dimension. **Reversible
jump MCMC** moves between them, proposing births and deaths of parameters and correcting the acceptance
ratio with a Jacobian for the dimension change. It is powerful, notoriously fiddly to tune, and largely
superseded in applied work by two alternatives: **continuous model expansion** (Stage 6) and the
**nonparametric** priors of Stage 16, which let complexity grow without discrete jumps.

Other auxiliary-variable tricks
---------------------------------

* **Data augmentation** — latent variables that restore conjugacy: a :math:`t` as a scale-mixture of
  normals, probit regression given latent utilities.
* **Simulated tempering / parallel tempering** — run chains at several "temperatures", flattening the
  posterior so a chain can cross **low-probability valleys** between modes, and swap states between
  them.
* **Adaptive MCMC** — learn the proposal covariance online, with the adaptation vanishing over time so
  that ergodicity is preserved.

.. code-block:: python

   # PyMC assigns samplers per variable: NUTS for continuous, specialised steppers otherwise
   import pymc as pm
   with model:
       idata = pm.sample(step=[pm.NUTS([mu, tau]), pm.BinaryGibbsMetropolis([z])])

Where they matter now
-----------------------

Gradient-based sampling handles continuous parameters better than any of these. Their enduring role is
the **complement**: discrete unknowns (which HMC cannot touch, since it needs derivatives), multimodal
posteriors (where tempering still helps), and univariate conditionals inside composite samplers. Modern
practice **marginalises** discrete parameters analytically where possible, and reaches for these tools
where it is not.

.. hint::

   **Related lessons:** :doc:`Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`  ·  :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Unspecified number of mixture components <137-unspecified-number-of-mixture-components>`  ·  :doc:`Continuous model expansion <048-continuous-model-expansion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/21/further-extensions-to-gibbs-and-metropolis/ <https://insightful-data-lab.com/2025/11/21/further-extensions-to-gibbs-and-metropolis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
