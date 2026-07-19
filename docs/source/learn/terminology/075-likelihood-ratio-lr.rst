:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-likelihood-ratio-lr:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🔬&nbsp;&nbsp;<b>Likelihood Ratio (LR)</b></div>`

=======================
Likelihood Ratio (LR)
=======================

*The ratio of data likelihoods under two hypotheses; central to many sequential and diagnostic tests.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

The **likelihood ratio (LR)** measures **how much more likely the observed data is under
one hypothesis than another**, by dividing their likelihoods:

.. math::

   \Lambda = \frac{L(\text{data} \mid H_1)}{L(\text{data} \mid H_0)}.

:math:`\Lambda = 1` means the data is equally likely under both; :math:`\Lambda > 1`
favours :math:`H_1`; :math:`\Lambda < 1` favours :math:`H_0`.

Two roles in testing
--------------------

- **Likelihood-ratio test (LRT)** — the generalised statistic
  :math:`\Lambda = \sup_{\theta \in \Theta_0} L(\theta) / \sup_{\theta \in \Theta} L(\theta)`
  compares the best fit under the null to the best fit overall; a small :math:`\Lambda`
  is strong evidence against :math:`H_0`. (By the Neyman–Pearson lemma, the LR is the
  *most powerful* test for simple hypotheses.)
- **Sequential probability ratio test (SPRT)** — accumulate the ratio as data arrives,
  :math:`\Lambda_n = L(\text{data}_{1:n} \mid H_1)/L(\text{data}_{1:n} \mid H_0)`, and
  stop when it crosses an upper bound :math:`A` (accept :math:`H_1`) or lower bound
  :math:`B` (accept :math:`H_0`), else continue.

Worked example
--------------

Seven heads in 10 tosses, with :math:`H_0: p = 0.5` vs :math:`H_1: p = 0.7`:

.. math::

   L(H_0) = \binom{10}{7}(0.5)^{10} \approx 0.117, \quad
   L(H_1) = \binom{10}{7}(0.7)^7(0.3)^3 \approx 0.266, \quad
   \Lambda = \frac{0.266}{0.117} \approx 2.27,

so the data is about **2.3 times more likely** under the biased hypothesis. Rough
reading: :math:`\Lambda \approx 1` no evidence, :math:`> 3` moderate, :math:`> 10`
strong.

Where it shows up
-----------------

The LR is the backbone of likelihood-based inference: the **LRT** and **SPRT**, nested
**model comparison**, and — closely related — the **Bayes factor**, which *is* a
likelihood ratio when the hypotheses are simple (and an evidence ratio of marginal
likelihoods when they are composite). In medicine, diagnostic **LR+ and LR−** update the
odds of disease from a test result.

----

*Theme:* :ref:`Model Evaluation & Uncertainty <term-theme-evaluation>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>` · :doc:`Binomial Likelihood <060-binomial-likelihood>` · :doc:`Frequentist <059-frequentist>` · :doc:`Posterior Probability <073-posterior-probability>`

----

.. hint::
   **More in Model Evaluation & Uncertainty**

   :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>` · :doc:`Baseline Heuristics <428-baseline-heuristics>` · :doc:`Bootstrap <365-bootstrap>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Coverage <411-coverage>` · :doc:`Cramér's V <180-cramer-s-v>` · :doc:`DeLong's Test <352-delong-s-test>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>` · :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>` · :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>` · :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>` · :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>` · :doc:`Relative accuracy <258-relative-accuracy>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Likelihood Ratio (LR) <https://insightful-data-lab.com/2025/08/25/likelihood-ratio-lr/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
