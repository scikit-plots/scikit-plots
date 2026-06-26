:html_theme.sidebar_secondary.remove:

..
   ##################################################################
   learn/bayesian_data_analysis/index.rst
   ==================================================================
   Bayesian inference hub, framed for scikit-plots users.
   Source context (framing only, re-expressed here):
   https://insightful-data-lab.com/category/bayesian-data-analysis/ (144 posts)
   The source closely tracks Gelman et al., *Bayesian Data Analysis*.
   ------------------------------------------------------------------
   Extensions: sphinx_design, sphinx_tags (bottom), sphinx_copybutton.
   Underlines: = section (overline)  - subsection  ^ subsubsection
   ##################################################################

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _bayesian-data-analysis-index:

:raw-html:`<div style="text-align:center"><strong>` 🎲 Bayesian Data Analysis
|br| Reasoning about uncertainty with priors, likelihoods and posteriors
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Bayesian Data Analysis
======================================================================

Bayesian analysis treats unknown quantities as **probability
distributions** and updates them with data. Instead of a single "best"
estimate, you get a full **posterior** — a principled account of what the
data do and do not tell you. This hub builds from first principles up to
the **nonparametric** models (mixtures, density estimation, Dirichlet
processes) that the source corpus emphasises.

Three reading levels run through the page:

* **newcomers** — the intuition of prior → likelihood → posterior;
* **practitioners** — how to actually compute and check posteriors;
* **researchers** — hierarchical and nonparametric (infinite-mixture)
  models.

.. note::

   Open a dropdown for detail; follow **See also** links to related
   ideas. Code snippets use real ``scipy.stats`` / ``PyMC`` / ``ArviZ``
   / ``scikit-learn`` calls. This page pairs with the
   :ref:`Terminology reference <terminology-index>` (probability and
   distributions) and the :ref:`Time Series hub <time-series-index>`
   (where Bayesian estimation also appears).

----------------------------------------------------------------------

.. _bayes-discovery:

Discovery at a Glance
----------------------------------------------------------------------

.. tab-set::
   :class: sd-width-100

   .. tab-item:: 🟢 Start Here — The Bayesian Idea
      :sync: level-foundations

      The one equation everything rests on.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🔁 Bayes' Theorem
            :link: bayes-theorem
            :link-type: ref
            :class-card: sd-border-1

            Posterior ∝ likelihood × prior — how belief is updated by
            evidence.

         .. grid-item-card:: 🎯 Prior, Likelihood, Posterior
            :link: bayes-pieces
            :link-type: ref
            :class-card: sd-border-1

            The three ingredients, what each encodes, and where they come
            from.

         .. grid-item-card:: 📐 Credible Intervals
            :link: bayes-credible
            :link-type: ref
            :class-card: sd-border-1

            A 95 % interval you *can* read as "95 % probability" — unlike
            a confidence interval.

   .. tab-item:: 🔵 Core — Computing Posteriors
      :sync: level-core

      From conjugate shortcuts to general-purpose sampling.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: ✨ Conjugacy
            :link: bayes-conjugacy
            :link-type: ref
            :class-card: sd-border-1

            When prior and posterior share a family, the update is exact
            and closed-form.

         .. grid-item-card:: ⛓️ MCMC Sampling
            :link: bayes-mcmc
            :link-type: ref
            :class-card: sd-border-1

            Drawing from any posterior when no formula exists — the
            workhorse of modern Bayes.

         .. grid-item-card:: 🔮 Posterior Predictive
            :link: bayes-ppc
            :link-type: ref
            :class-card: sd-border-1

            Simulating new data to check the model and forecast.

   .. tab-item:: 🔴 Advanced — Hierarchies & Nonparametrics
      :sync: level-advanced

      Sharing strength across groups; letting complexity grow with data.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🏛️ Hierarchical Models
            :link: bayes-hierarchical
            :link-type: ref
            :class-card: sd-border-1

            Partial pooling: groups borrow strength from each other.

         .. grid-item-card:: 🌗 Mixture Models
            :link: bayes-mixtures
            :link-type: ref
            :class-card: sd-border-1

            Sub-populations, label switching, and choosing the number of
            components.

         .. grid-item-card:: ♾️ Dirichlet Processes
            :link: bayes-dp
            :link-type: ref
            :class-card: sd-border-1

            Nonparametric priors that let the number of clusters grow with
            the data.

----------------------------------------------------------------------

.. _bayes-foundations:

Part 1 — The Bayesian Idea
----------------------------------------------------------------------

.. dropdown:: Bayes' Theorem
   :color: primary
   :icon: sync
   :name: bayes-theorem
   :open:

   **What is it?**

   Bayes' theorem inverts conditional probability to turn a model of "how
   data arise given parameters" into "what parameters are plausible given
   data":

   .. math::

      p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}
      \;\;\propto\;\; \underbrace{p(y \mid \theta)}_{\text{likelihood}}\;
      \underbrace{p(\theta)}_{\text{prior}}

   The denominator :math:`p(y)` (the *evidence*) is a normalising
   constant; for inference about :math:`\theta` the proportionality on
   the right is what matters.

   **When to use it** — whenever you want to combine prior knowledge with
   observed data and quantify the remaining uncertainty as a distribution.

   .. seealso::

      :ref:`bayes-pieces` · :ref:`terminology-statistics`

.. dropdown:: Prior, Likelihood & Posterior
   :color: primary
   :icon: stack
   :name: bayes-pieces

   * **Prior** :math:`p(\theta)` — belief about the parameter *before*
     seeing this data (from theory, past studies, or a deliberately weak
     "let the data speak" choice).
   * **Likelihood** :math:`p(y\mid\theta)` — the data-generating model,
     read as a function of :math:`\theta` for the observed :math:`y`.
   * **Posterior** :math:`p(\theta\mid y)` — the updated belief; the
     output of the analysis and the input to every decision.

   As data accumulate, the likelihood dominates and the posterior becomes
   insensitive to a reasonable prior.

   .. seealso::

      :ref:`bayes-conjugacy` · :ref:`bayes-credible`

.. dropdown:: Credible Intervals (and how they differ from CIs)
   :color: primary
   :icon: number
   :name: bayes-credible

   **What is it?**

   A 95 % **credible interval** is any region containing 95 % of the
   posterior probability mass. It supports the natural statement *"there
   is a 95 % probability the parameter lies in this range"* — which a
   frequentist **confidence** interval does **not**.

   .. code-block:: python

      import numpy as np
      # equal-tailed 95% credible interval from posterior samples
      lo, hi = np.percentile(posterior_samples, [2.5, 97.5])

   .. seealso::

      :ref:`terminology-bootstrap` · :ref:`bayes-ppc`

----------------------------------------------------------------------

.. _bayes-computation:

Part 2 — Computing Posteriors
----------------------------------------------------------------------

.. dropdown:: Conjugacy (the exact, closed-form case)
   :color: info
   :icon: sparkle-fill
   :name: bayes-conjugacy

   **What is it?**

   A prior is **conjugate** to a likelihood when the posterior stays in
   the same family. The classic example is Beta–Binomial: a
   :math:`\text{Beta}(\alpha, \beta)` prior on a success probability,
   with :math:`k` successes in :math:`n` trials, yields

   .. math::

      p(\theta \mid y) = \text{Beta}(\alpha + k,\; \beta + n - k)

   **scipy**

   .. code-block:: python

      from scipy import stats
      alpha, beta = 1, 1          # uniform prior
      k, n = 8, 10
      post = stats.beta(alpha + k, beta + n - k)
      print(post.mean(), post.interval(0.95))

   **When to use it** — quick, exact updates for simple models, and as
   building blocks inside larger samplers.

   .. seealso::

      :ref:`bayes-mcmc`

.. dropdown:: MCMC Sampling
   :color: info
   :icon: link
   :name: bayes-mcmc

   **What is it?**

   When the posterior has no closed form, **Markov chain Monte Carlo**
   draws correlated samples whose stationary distribution *is* the
   posterior. Modern tools use Hamiltonian Monte Carlo / NUTS for
   efficient exploration.

   **PyMC + ArviZ**

   .. code-block:: python

      import pymc as pm
      import arviz as az

      with pm.Model() as model:
          theta = pm.Beta("theta", alpha=1, beta=1)
          y = pm.Binomial("y", n=10, p=theta, observed=8)
          idata = pm.sample(2000, tune=1000)

      az.summary(idata)            # means, sd, 94% HDI, r_hat
      az.plot_trace(idata)         # convergence diagnostics

   **Check before trusting** — :math:`\hat{R} \approx 1.0`, healthy
   effective sample size, no divergences.

   .. seealso::

      :ref:`bayes-ppc` · :ref:`bayes-hierarchical`

.. dropdown:: Posterior Predictive Checks
   :color: info
   :icon: beaker
   :name: bayes-ppc

   **What is it?**

   The **posterior predictive** distribution simulates new data by
   integrating the likelihood over the posterior:

   .. math::

      p(\tilde{y} \mid y) = \int p(\tilde{y}\mid\theta)\,p(\theta\mid y)\,d\theta

   Comparing simulated datasets to the real one is the primary Bayesian
   **model-checking** tool: systematic mismatch signals a misspecified
   model.

   .. code-block:: python

      with model:
          pm.sample_posterior_predictive(idata, extend_inferencedata=True)
      az.plot_ppc(idata)

   .. seealso::

      :ref:`bayes-mixtures`

----------------------------------------------------------------------

.. _bayes-advanced:

Part 3 — Hierarchies, Mixtures & Nonparametrics
----------------------------------------------------------------------

.. dropdown:: Hierarchical Models (Partial Pooling)
   :color: secondary
   :icon: organization
   :name: bayes-hierarchical

   **What is it?**

   When data come in **groups** (schools, patients, sites), a hierarchical
   model gives each group its own parameter while tying those parameters
   to a shared population distribution:

   .. math::

      y_{ij} \sim p(\cdot \mid \theta_j), \qquad
      \theta_j \sim \mathcal{N}(\mu, \tau^2)

   This **partial pooling** shrinks noisy small-group estimates toward the
   overall mean — between "one estimate for everyone" (complete pooling)
   and "every group alone" (no pooling). The source's *hierarchical
   dependence* posts develop exactly this structure.

   .. seealso::

      :ref:`bayes-mcmc` · :ref:`bayes-mixtures`

.. dropdown:: Mixture Models & Label Switching
   :color: secondary
   :icon: git-merge
   :name: bayes-mixtures

   **What is it?**

   A finite mixture models a population as a weighted blend of
   sub-populations:

   .. math::

      p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k),
      \qquad \sum_k \pi_k = 1

   **Label switching** is the identifiability quirk that the components
   can be permuted without changing the likelihood — a thing to handle
   when summarising posteriors. Choosing :math:`K` is a model-selection
   problem (AIC / BIC, or let it be infinite — see Dirichlet processes).

   **scikit-learn + scikit-plots**

   .. code-block:: python

      from sklearn.mixture import GaussianMixture
      gmm = GaussianMixture(n_components=3).fit(X)

      # scikit-plots: compare K via AIC / AICc / BIC
      import scikitplot as skplt
      skplt.stats.plot_gaussian_mixture_models(X)

   .. seealso::

      :ref:`bayes-dp` · :ref:`terminology-statistics`

.. dropdown:: Dirichlet Processes (Nonparametric Bayes)
   :color: secondary
   :icon: infinity
   :name: bayes-dp

   **What is it?**

   A **Dirichlet process** :math:`\text{DP}(\alpha, G_0)` is a prior over
   *distributions* — the foundation of infinite mixture / density-
   estimation models where the number of clusters is **not fixed in
   advance** but grows with the data. The stick-breaking view builds the
   mixing weights as

   .. math::

      \pi_k = v_k \prod_{l<k}(1 - v_l), \qquad v_k \sim \text{Beta}(1, \alpha)

   The concentration :math:`\alpha` controls how readily new clusters
   appear. This underpins the source's *Dirichlet process mixtures*,
   *Bayesian histograms*, and *density estimation* posts.

   **When to use it** — clustering / density estimation where you cannot
   commit to a fixed number of components a priori.

   .. seealso::

      :ref:`bayes-mixtures`

----------------------------------------------------------------------

.. _bayes-skplt-map:

Map to scikit-plots & the Bayesian Stack
----------------------------------------------------------------------

scikit-plots' role here is diagnostic and model-selection visual support;
the heavy lifting is done by the probabilistic-programming stack.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Gaussian Mixture Models (AIC / BIC)
      :link: https://scikit-plots.github.io/dev/auto_examples/stats/plot_gaussian_mixture_models.html

      Choose the number of mixture components by information criteria.

   .. grid-item-card:: Residuals distribution
      :link: https://scikit-plots.github.io/dev/auto_examples/stats/plot_residuals_distribution_script.html

      Distributional / Q–Q checks on fitted models.

   .. grid-item-card:: PyMC
      :link: https://www.pymc.io/

      Probabilistic programming for building and sampling models.

   .. grid-item-card:: ArviZ
      :link: https://python.arviz.org/

      Diagnostics, summaries and plots for Bayesian inference.

----------------------------------------------------------------------

.. _bayes-sources:

Sources
----------------------------------------------------------------------

Verified during preparation of this page; resolvable at build date.

**Source context (framing only, re-expressed in our own words)**

* Bayesian Data Analysis category (144 posts):
  https://insightful-data-lab.com/category/bayesian-data-analysis/

**Official documentation (API calls used above)**

* SciPy — ``scipy.stats`` distributions:
  https://docs.scipy.org/doc/scipy/reference/stats.html
* scikit-learn — Gaussian mixture models:
  https://scikit-learn.org/stable/modules/mixture.html
* PyMC — probabilistic programming: https://www.pymc.io/
* ArviZ — exploratory analysis of Bayesian models:
  https://python.arviz.org/

**scikit-plots (this project)**

* Example gallery: https://scikit-plots.github.io/dev/auto_examples/index.html
* Terminology reference: :ref:`terminology-index`

**Standard reference**

* Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data
  Analysis* (3rd ed.): http://www.stat.columbia.edu/~gelman/book/

..
   ##################################################################
   Tags — bottom of page, project controlled vocabulary only.
   (Promote to `domain: bayesian` once multiple Bayesian pages exist.)
   ##################################################################

.. tags::
   purpose: reference,
   domain: statistics,
   level: beginner,
   level: intermediate,
   level: advanced
