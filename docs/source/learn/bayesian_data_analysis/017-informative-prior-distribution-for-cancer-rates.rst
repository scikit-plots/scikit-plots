.. _bda-informative-prior-distribution-for-cancer-rates:

========================================================================
Informative Prior Distribution for Cancer Rates
========================================================================

**Part 1 · Stage 2 · 📍 Single-Parameter Models & Priors**  ·  Lesson 017 of 144  ·  *beginner*

:doc:`◀ Previous · Other Standard Single-Parameter Models <016-other-standard-single-parameter-models>`   ·   :doc:`Next · Noninformative Prior Distributions ▶ <018-noninformative-prior-distributions>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The map that lies
-------------------

Colour a map of US counties by **kidney-cancer rate** and something strange appears: the highest-rate
counties are mostly **small, rural** ones. Colour it by the **lowest** rates and — the same counties
appear again. No environmental story explains both. The pattern is a **statistical artifact**, and it
is the classic argument for informative priors.

Small denominators, wild rates
--------------------------------

Kidney cancer is rare. A county of 60,000 people might record **4** cases (about 6.6 per 100,000); a
county of 17,000 might record **7** (about 41 per 100,000). One case fewer in the small county would
drop its rate by roughly a sixth. The raw rate :math:`y_j / n_j` is an **unbiased but hopelessly noisy**
estimate when :math:`n_j` is tiny, so the extremes of the ranking are populated not by the most
dangerous places, but by the **smallest** ones.

The Poisson–Gamma remedy
--------------------------

Model the counts as :math:`y_j \sim \mathrm{Poisson}(\theta_j n_j)` with a
:math:`\mathrm{Gamma}(\alpha, \beta)` prior on the county rate, and the conjugate posterior mean is a
weighted average of the county's own rate and the prior (national) rate:

.. math::

   \mathrm{E}[\theta_j \mid y_j] = \frac{\alpha + y_j}{\beta + n_j}
   = \frac{\beta}{\beta + n_j} \cdot \underbrace{\frac{\alpha}{\beta}}_{\text{prior rate}}
   \;+\; \frac{n_j}{\beta + n_j} \cdot \underbrace{\frac{y_j}{n_j}}_{\text{raw rate}} .

Small counties (:math:`n_j \ll \beta`) are pulled hard toward the national rate; large counties keep
their own. The prior does not distort — it **stabilises**.

.. code-block:: python

   from scipy import stats
   alpha, beta = 20, 1_000_000        # prior: ~20 cases per million person-years
   for y, n in [(7, 17_000), (4, 60_000), (250, 1_200_000)]:
       raw = 1e5 * y / n
       shrunk = 1e5 * (alpha + y) / (beta + n)
       print(f"raw {raw:6.1f}   shrunk {shrunk:6.1f}   (n={n:,})")
   # small counties move a lot; the large county barely moves

Shrinkage, honestly
---------------------

The estimates are **shrunk** toward a common centre by an amount governed by how much data each county
supplies. This buys enormous stability at the price of a small bias toward the mean — an excellent
trade when the alternative is ranking noise. And note what this analysis is quietly reaching for: the
prior rate :math:`\alpha/\beta` should really be **estimated from the counties themselves**. That is a
**hierarchical model**, and it arrives in Stage 5.

.. hint::

   **Related lessons:** :doc:`Informative Prior Distributions <014-informative-prior-distributions>`  ·  :doc:`Other Standard Single-Parameter Models <016-other-standard-single-parameter-models>`  ·  :doc:`Exchangeability and hierarchical models <034-exchangeability-and-hierarchical-models>`  ·  :doc:`Posterior as a Compromise Between Data and Prior Information <012-posterior-as-a-compromise-between-data-and-prior-information>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/informative-prior-distribution-for-cancer-rates/ <https://insightful-data-lab.com/2025/11/09/informative-prior-distribution-for-cancer-rates/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
