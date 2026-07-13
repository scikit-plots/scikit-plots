.. _bda-averaging-over-nuisance-parameters:

========================================================================
Averaging Over Nuisance Parameters
========================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 020 of 144  ·  *beginner*

:doc:`◀ Previous · Weakly Informative Prior Distributions <019-weakly-informative-prior-distributions>`   ·   :doc:`Next · Normal Data with a Noninformative Prior Distribution ▶ <021-normal-data-with-a-noninformative-prior-distribution>`   ·   :doc:`↑ Section <index>`


The parameters you don't want
-------------------------------

Real models have more parameters than questions. Estimating a mean :math:`\mu` usually drags along an
unknown variance :math:`\sigma^2`; a regression coefficient of interest comes with a dozen others. The
unwanted ones are **nuisance parameters** — necessary for the model to be honest, irrelevant to the
conclusion.

Integrate, don't fix
----------------------

The Bayesian treatment is uniform and unremarkable: obtain the **joint** posterior, then **marginalise**
the nuisance away. To learn about :math:`\theta_1` in the presence of nuisance :math:`\theta_2`,

.. math::

   p(\theta_1 \mid y) = \int p(\theta_1, \theta_2 \mid y) \; d\theta_2
   = \int \underbrace{p(\theta_1 \mid \theta_2, y)}_{\text{conditional}}
          \; \underbrace{p(\theta_2 \mid y)}_{\text{weight}} \; d\theta_2 .

That second form is the useful one: the marginal posterior of :math:`\theta_1` is a **mixture** of its
conditional posteriors, weighted by how plausible each value of the nuisance is. Uncertainty about
:math:`\sigma^2` is not discarded — it is **averaged in**.

Why plugging in is wrong
--------------------------

The tempting shortcut is to fix the nuisance at an estimate, :math:`p(\theta_1 \mid \hat{\theta}_2, y)`.
This **understates uncertainty**, and the variance decomposition says exactly by how much:

.. math::

   \mathrm{var}(\theta_1 \mid y) = \mathrm{E}\bigl[\mathrm{var}(\theta_1 \mid \theta_2, y)\bigr]
                                 + \mathrm{var}\bigl(\mathrm{E}[\theta_1 \mid \theta_2, y]\bigr) .

Plugging in keeps only the first term and throws away the second — the variation induced by not
knowing :math:`\theta_2`. This is why a normal mean with **unknown** variance has a heavier-tailed
:math:`t` posterior rather than a normal one: the extra width is the price of honesty about
:math:`\sigma^2`.

Marginalising with draws
--------------------------

In simulation the operation is invisible: draw from the joint posterior, then **ignore the columns you
do not need**.

.. code-block:: python

   # draws[:, 0] = mu, draws[:, 1] = sigma  (joint posterior draws)
   mu = draws[:, 0]                 # already the marginal posterior of mu
   mu.mean(), np.percentile(mu, [2.5, 97.5])

Dropping a column *is* integration over that parameter. It is one of the quiet reasons the
simulation-based workflow scales to models where the integrals could never be done in closed form —
starting with the normal model of the next lesson.

.. hint::

   **Related lessons:** :doc:`Some Useful Results from Probability Theory <008-some-useful-results-from-probability-theory>`  ·  :doc:`Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`  ·  :doc:`Normal Data with a Conjugate Prior Distribution <022-normal-data-with-a-conjugate-prior-distribution>`  ·  :doc:`Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/averaging-over-nuisance-parameters/ <https://insightful-data-lab.com/2025/11/09/averaging-over-nuisance-parameters/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
