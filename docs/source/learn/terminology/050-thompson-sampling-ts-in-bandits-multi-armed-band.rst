:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-thompson-sampling-ts-in-bandits-multi-armed-bandit-problem-mab:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎰&nbsp;&nbsp;<b>Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))</b></div>`

======================================================================
Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))
======================================================================

*A bandit strategy that samples from posterior beliefs to balance exploration and exploitation.*

The multi-armed bandit problem
------------------------------

Picture a row of slot machines (**arms**). Each arm :math:`i` pays out from an unknown
reward distribution with mean :math:`\mu_i`, and the goal is to **maximise total reward
over time**. That forces a trade-off between:

- **Exploration** — trying arms to learn their payoffs, and
- **Exploitation** — playing the arm that looks best so far.

Pull only the current best and you may never discover a better arm; explore too much
and you waste pulls on bad ones. **Thompson Sampling** resolves this elegantly.

The Bayesian view
-----------------

Keep a **posterior distribution** over each arm's reward parameter. For Bernoulli
(success/failure) rewards the natural choice is a Beta posterior:

.. math::

   p_i \sim \text{Beta}(\alpha_i, \beta_i),

updated as successes and failures accumulate.

The algorithm
-------------

Each round:

1. **Sample** a value :math:`\tilde{\mu}_i` from every arm's posterior.
2. **Select** the arm with the highest sampled value.
3. **Play** it and observe reward :math:`r`.
4. **Update** that arm's posterior by Bayes' rule.

The intuition is the whole trick: an *uncertain* arm has a wide posterior, so its
samples are sometimes high — it gets explored; an arm you're confident is bad rarely
produces a winning sample — it's quietly dropped.

Worked example — the Bernoulli bandit
-------------------------------------

Arm :math:`i` returns 1 with probability :math:`p_i`. Start from a uniform prior
:math:`p_i \sim \text{Beta}(1,1)`. After :math:`s` successes and :math:`f` failures,
conjugacy gives

.. math::

   p_i \mid \text{data} \sim \text{Beta}(1+s,\, 1+f).

Each round, draw :math:`\tilde{p}_i \sim \text{Beta}(1+s, 1+f)` for every arm and pull
the one with the largest draw.

.. code-block:: python

   import numpy as np
   # alpha, beta hold Beta posterior params per arm (start at 1, 1)
   samples = np.random.beta(alpha, beta)      # one draw per arm
   arm = int(np.argmax(samples))              # play the best sampled arm
   # observe reward r in {0, 1}, then update:
   alpha[arm] += r
   beta[arm]  += 1 - r

Why it works, and how well
--------------------------

Exploration and exploitation are handled **automatically** by the sampling — no manual
exploration rate to tune. As evidence accumulates the posteriors concentrate, and the
best arm comes to dominate. Its **regret** — the gap between the optimal arm's expected
reward and what you actually earned — grows only **logarithmically**,
:math:`O(\log T)`, which is near the theoretical optimum.

Extensions
----------

- **Contextual Thompson Sampling** — condition on covariates :math:`x` (user features)
  via Bayesian linear or logistic regression, so the chosen arm depends on context.
- **Other reward models** — Gaussian posteriors for continuous rewards, Dirichlet for
  categorical.
- **Parallel Thompson Sampling** — for distributed settings with many simultaneous
  users.

Where it fits
-------------

Thompson Sampling is the Bayesian member of the **bandit-algorithm** family and a
direct alternative to fixed-split **A/B testing**: instead of holding a 50/50 split for
a set horizon, it shifts traffic toward the winning variant as evidence builds, cutting
the cost of showing the worse option. Compared with :math:`\epsilon`-greedy it needs no
hand-tuned exploration rate, and compared with UCB it explores via posterior sampling
rather than confidence bounds.

----

*Theme:* :ref:`Sequential Methods & Bandits <term-theme-bandits>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Posterior <063-posterior>` · :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`

----

.. hint::
   **More in Sequential Methods & Bandits**

   :doc:`Bandit Algorithms <113-bandit-algorithms>` · :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>` · :doc:`Pocock Method <077-pocock-method>` · :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Sequential Settings <058-sequential-settings>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <https://insightful-data-lab.com/2025/08/29/thompson-sampling-ts-in-bandits/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
