.. _bda-state-level-opinons-from-national-polls:

========================================================================
State-level opinons from national polls
========================================================================

**Part 4 · Stage 13 · 🔗 Generalized Linear Models**  ·  Lesson 110 of 144  ·  *advanced*

:doc:`◀ Previous · Overdispersed Poisson regression for police stops <109-overdispersed-poisson-regression-for-police-stops>`   ·   :doc:`Next · Models for multivariate and multinomial responses ▶ <111-models-for-multivariate-and-multinomial-responses>`   ·   :doc:`↑ Section <index>`


Small areas from big surveys
------------------------------

A national poll of a couple of thousand respondents estimates *national* opinion well but says little
about any single **state** — some states contain a handful of respondents. Yet state-level opinion is
exactly what redistricting, forecasting and representation require. **Multilevel regression and
poststratification** (MRP) extracts reliable small-area estimates from national data, and it is one of
the most consequential applications of the hierarchical models built in this part.

The two steps
---------------

MRP is a **model** followed by a **weighting**.

**Regression.** Fit a multilevel logistic model for the individual response, with demographic predictors
(age, race, sex, education) as varying effects and a **state effect** that is partially pooled — often
with a state-level predictor such as past vote:

.. math::

   \Pr(y_i = 1) = \mathrm{logit}^{-1}\!\bigl(\alpha_{\text{state}[i]}
                  + \beta_{\text{age}[i]} + \beta_{\text{race}[i]} + \cdots\bigr),
   \qquad \alpha_s \sim \mathrm{N}(\gamma_0 + \gamma_1 v_s, \sigma^2).

Partial pooling stabilises the estimate for every demographic-by-state cell, even cells with almost no
respondents — the shrinkage of this entire part, doing the heavy lifting.

**Poststratification.** Reweight the model's cell predictions by the **known population frequency** of
each cell from the census, so the state estimate reflects that state's actual demographic composition:

.. math::

   \theta_s = \frac{\sum_{c \in s} N_c \, \hat{p}_c}{\sum_{c \in s} N_c} .

.. code-block:: python

   import numpy as np
   # 1. multilevel model gives p_hat for every (demographic x state) cell
   p_cell = posterior_cell_predictions(idata)              # shape: (draws, n_cells)
   # 2. reweight by census population counts N_cell within each state
   theta_state = (N_cell * p_cell).sum(axis=1) / N_cell.sum()   # per state, with uncertainty

Why it works
--------------

The pieces cover each other's weaknesses. The **model** borrows strength across states so that sparse
cells get sensible estimates instead of noise; **poststratification** corrects the sample's demographic
imbalances against known population totals, removing the bias that makes raw subgroup means unreliable.
Together they turn a survey never designed for state estimates into a state-level instrument — and the
same machinery **adjusts for non-representative samples** generally, which is why MRP has become central
to modern survey inference and to forecasting from imperfect polls. It is the hierarchical logistic model
of this stage, put to work on the small-area problem.

.. hint::

   **Related lessons:** :doc:`Working with generalized linear models <107-working-with-generalized-linear-models>`  ·  :doc:`Weakly informative priors for logistic regression <108-weakly-informative-priors-for-logistic-regression>`  ·  :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`  ·  :doc:`Sample surveys <052-sample-surveys>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/state-level-opinons-from-national-polls/ <https://insightful-data-lab.com/2025/12/06/state-level-opinons-from-national-polls/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
