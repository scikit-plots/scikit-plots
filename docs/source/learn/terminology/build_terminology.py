#!/usr/bin/env python3
# ======================================================================
# build_terminology.py  —  deterministic generator for learn/terminology
# ----------------------------------------------------------------------
# Reads the verified inventory (term_inventory.tsv: title <TAB> url ...)
# joins it with the ENRICH table below (theme + rewritten gloss), and
# emits a complete, idempotent index.rst (in place, inside terminology/):
#   * centred banner + intro + how-to note
#   * "Discovery at a Glance" tab-set (by level) -> theme section cards
#   * one section per theme, each term a sphinx_design dropdown carrying
#     a rewritten gloss, a verified source link, and auto "Related in
#     this area" mind-map cross-links (siblings in the same theme)
#   * an alphabetical A-Z quick index (:ref: links to every term)
#   * sphinx_tags block (controlled vocabulary) + verified Sources
#
# Re-runnable: same inputs -> byte-identical output (sorted, no RNG).
# To grow toward all 431 terms: append rows to term_inventory.tsv,
# add their {title: (theme, gloss)} here, and re-run.  Missing/extra
# titles are reported and abort the build (fail-fast, no silent gaps).
# ======================================================================
from __future__ import annotations
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # make term_content importable regardless of CWD
INVENTORY = HERE / "term_inventory.tsv"
OUT = HERE / "index.rst"               # this script now lives INSIDE terminology/

# ----------------------------------------------------------------------
# Theme registry: key -> (emoji, display title, level, one-line blurb)
# level in {"foundations", "applied", "advanced"} drives the discovery
# tabs and the dropdown colour.
# ----------------------------------------------------------------------
THEMES: dict[str, tuple[str, str, str, str]] = {
    "probstats": ("🎲", "Probability & Statistics Foundations", "foundations",
                  "The vocabulary of estimates, spread and uncertainty that everything else builds on."),
    "inference": ("🧮", "Statistical Inference & Power", "foundations",
                  "Hypothesis testing, error rates and how big a study needs to be."),
    "imbalance": ("🧪", "Imbalanced Learning & Resampling", "applied",
                  "Techniques for training useful models when one class is rare."),
    "metrics": ("📏", "Classification & Averaging Metrics", "applied",
                "Precision, recall, F1 and AUROC — and how they are averaged across classes."),
    "evaluation": ("🔬", "Model Evaluation & Uncertainty", "applied",
                   "Putting error bars and significance around a model's score."),
    "abtest": ("🧫", "A/B Testing & Experimentation", "applied",
               "Designing and reading controlled online experiments."),
    "growth": ("💼", "Business & Growth Analytics", "applied",
               "The unit-economics and customer metrics that experiments optimise."),
    "fairness": ("⚖️", "Fairness & Calibration", "advanced",
                 "Group-level fairness criteria and what each one equalises."),
    "bayes": ("🔁", "Bayesian Inference", "advanced",
              "Reasoning with priors, posteriors and the machinery that computes them."),
    "bandits": ("🎰", "Sequential Methods & Bandits", "advanced",
                "Adaptive decisions and tests that update as evidence arrives."),
    "signal": ("📈", "Signal Processing & Time Series", "advanced",
               "Working with ordered, time-indexed and filtered data."),
    "validation": ("🧷", "Validation & Cross-Validation", "applied",
                   "Splitting strategies that estimate performance honestly, including for time series."),
    "training": ("🏋️", "Model Training & Optimization", "applied",
                 "Fitting, tuning and compressing models effectively."),
    "mlops": ("⚙️", "MLOps, Serving & Monitoring", "advanced",
              "Deploying, serving and keeping models healthy in production."),
    "drift": ("🌊", "Distribution Shift & Drift", "advanced",
              "Detecting and measuring when data or representations change."),
    "repr": ("🧬", "Representations & Embeddings", "advanced",
             "Learned feature spaces and the encoders that produce them."),
    "ranking": ("🔎", "Ranking & Interleaving", "advanced",
                "Ordering items and comparing rankers with online evaluation."),
    "causal": ("🔗", "Causal Inference & Uplift", "advanced",
               "Estimating cause-and-effect and the incremental impact of actions."),
    "concepts": ("💡", "AI & ML Concepts", "foundations",
                 "Umbrella terms for the field and its major model families."),
    "platforms": ("🧰", "ML Platforms & Tools", "applied",
                  "Cloud services and APIs for building, serving and experimenting."),
    "ops": ("📦", "Operations & Supply Chain", "applied",
            "Inventory, demand and fulfillment metrics for operations."),
    "risk": ("📉", "Risk & Probabilistic Forecasting", "advanced",
             "Quantifying uncertainty and tail risk in forecasts."),
    "recsys": ("🎁", "Recommender Systems", "advanced",
               "Relevance, diversity and coverage in recommendation."),
    "calibration": ("🎯", "Probability Calibration", "advanced",
                    "Making predicted probabilities trustworthy and measuring miscalibration."),
    "features": ("🧮", "Data Preparation & Features", "applied",
                 "Transforming, encoding and organising raw data into model-ready form."),
    "xai": ("🔍", "Explainability & Governance", "advanced",
            "Interpreting model decisions and meeting regulatory and ethical duties."),
}
LEVEL_ORDER = ["foundations", "applied", "advanced"]
LEVEL_TAB = {
    "foundations": "🟢 Foundations",
    "applied": "🔵 Applied",
    "advanced": "🔴 Advanced",
}
LEVEL_COLOR = {"foundations": "primary", "applied": "info", "advanced": "secondary"}
# theme display order = grouped by level, then registry order
THEME_ORDER = [k for lvl in LEVEL_ORDER for k in THEMES if THEMES[k][2] == lvl]

# Legacy anchors referenced by the topic hubs (data_prep/bayesian/time_series/
# deep_learning). Emitted as stacked aliases on the matching theme section so
# those cross-links keep resolving after every regeneration.
COMPAT: dict[str, str] = {
    "terminology-statistics": "probstats",
    "terminology-averaging": "metrics",
    "terminology-classification-types": "metrics",
    "terminology-multiclass-auroc": "metrics",
    "terminology-precision-recall": "metrics",
    "terminology-roc-auroc": "metrics",
    "terminology-bootstrap": "evaluation",
    "terminology-calibration": "fairness",
    "terminology-signal-timeseries": "signal",
}

# ----------------------------------------------------------------------
# ENRICH: verified-title -> (theme_key, rewritten gloss).
# Glosses are concise, original re-expressions (not copied from source).
# ----------------------------------------------------------------------
ENRICH: dict[str, tuple[str, str]] = {
    # --- Imbalanced Learning & Resampling ---
    "Subsampling": ("imbalance", "Selecting a representative subset of rows, often to speed up training or to rebalance class frequencies."),
    "Class Weighting": ("imbalance", "Giving minority-class errors a larger weight in the loss so the model stops ignoring rare classes."),
    "SMOTE (Synthetic Minority Over-sampling Technique)": ("imbalance", "Creates new minority examples by interpolating between nearest neighbours instead of duplicating existing rows."),
    "Oversampling": ("imbalance", "Rebalancing classes by replicating or synthesising additional minority-class examples."),
    "NearMiss (Distance-based Undersampling)": ("imbalance", "Undersamples the majority class by keeping the majority points nearest to the minority, chosen by distance heuristics."),
    "Cluster-based undersampling": ("imbalance", "Clusters the majority class and keeps representatives of each cluster, shrinking it while preserving its structure."),
    "Random Undersampling": ("imbalance", "Drops random majority-class rows to balance classes — simple, but can discard useful information."),
    # --- Signal Processing & Time Series ---
    "Low-pass Filtering": ("signal", "A filter that keeps low-frequency content and attenuates high-frequency noise in a signal."),
    "Signal Processing": ("signal", "Analysing and transforming signals (audio, sensor, time series) to extract or clean information."),
    "Time Series": ("signal", "Observations indexed in time order, where sequence and dependence carry information."),
    "Bayesian Time Series": ("signal", "Time-series modelling in a Bayesian framework, yielding full posterior uncertainty over parameters and forecasts."),
    # --- Classification & Averaging Metrics ---
    "Micro AUROC": ("metrics", "AUROC pooled across classes by aggregating individual decisions first; weights every sample equally."),
    "Multi-label Classification": ("metrics", "Tasks where each instance may carry several non-exclusive labels at once."),
    "Micro F1": ("metrics", "F1 computed from globally pooled TP/FP/FN; dominated by the more frequent classes."),
    "Single-label Classification": ("metrics", "Tasks where each instance is assigned exactly one label."),
    "Micro Recall": ("metrics", "Recall computed from globally pooled true positives and false negatives."),
    "Micro Precision": ("metrics", "Precision computed from globally pooled true positives and false positives."),
    "One-vs-Rest (OvR) AUROC": ("metrics", "Multiclass AUROC obtained by scoring each class against all others and averaging."),
    "Macro AUROC (Macro-Averaged AUROC)": ("metrics", "The mean of per-class AUROC values, weighting every class equally regardless of size."),
    "Macro F1": ("metrics", "The unweighted mean of per-class F1 scores; exposes weak performance on rare classes."),
    "Macro Recall": ("metrics", "The unweighted mean of per-class recall values."),
    "Macro Precision": ("metrics", "The unweighted mean of per-class precision values."),
    "Multiclass AUROC": ("metrics", "AUROC extended beyond two classes via One-vs-Rest or One-vs-One schemes."),
    "Gini Coefficient": ("metrics", "A ranking-quality score linearly tied to AUROC: Gini = 2 x AUROC - 1."),
    # --- Model Evaluation & Uncertainty ---
    "Bootstrap Confidence Intervals (CIs)": ("evaluation", "Interval estimates built by resampling the data with replacement and recomputing the statistic many times."),
    "Mann\u2013Whitney U Test (also called the Wilcoxon rank-sum test)": ("evaluation", "A nonparametric, rank-based test of whether one group's values tend to exceed another's."),
    "Likelihood Ratio (LR)": ("evaluation", "The ratio of data likelihoods under two hypotheses; central to many sequential and diagnostic tests."),
    # --- Fairness & Calibration ---
    "Predictive Parity (Calibration)": ("fairness", "A fairness criterion requiring equal positive predictive value (precision) across groups."),
    "Equalized Odds (Fairness)": ("fairness", "Requires equal true-positive and false-positive rates across groups."),
    "Equal Opportunity (Fairness)": ("fairness", "Requires equal true-positive rates across groups — a relaxation of equalized odds."),
    "Demographic Parity (Statistical Parity)": ("fairness", "Requires the positive-prediction rate to be equal across groups, independent of the true label."),
    # --- Probability & Statistics Foundations ---
    "Probability": ("probstats", "A number in [0, 1] quantifying how likely an event is."),
    "Standard Error (SE)": ("probstats", "The standard deviation of a statistic's sampling distribution — how much an estimate varies across samples."),
    "True Mean (Population Mean)": ("probstats", "The actual mean of the whole population that a sample mean estimates."),
    "Margin of Error (MoE)": ("probstats", "The half-width of a confidence interval: the plus/minus range around a point estimate."),
    "Critical Value": ("probstats", "The cutoff from a reference distribution that a test statistic must pass to reject the null."),
    "Sample Standard Deviation": ("probstats", "An estimate of population spread from a sample, using n - 1 in the denominator."),
    "Sample Mean": ("probstats", "The arithmetic average of a sample, used to estimate the population mean."),
    "Regression Coefficient": ("probstats", "The estimated effect of a predictor on the response in a regression model."),
    "Proportion": ("probstats", "The fraction of a sample or population that has a given attribute."),
    "True Population Parameter": ("probstats", "The fixed, usually unknown quantity (mean, proportion, coefficient) that estimation targets."),
    "Z-Score": ("probstats", "How many standard deviations a value lies from the mean."),
    "Statistical Significance": ("probstats", "Evidence that an observed effect is unlikely under the null hypothesis, judged against a chosen threshold."),
    "Frequentist": ("probstats", "The school of statistics treating probability as long-run frequency and parameters as fixed unknowns."),
    # --- Statistical Inference & Power ---
    "Compromise Power Analysis": ("inference", "Sizing a study by trading Type I against Type II error at a fixed ratio rather than fixing one."),
    "Post Hoc Power Analysis": ("inference", "Computing achieved power after a study from the observed effect — widely criticised as uninformative."),
    "A Priori Power Analysis": ("inference", "Computing the sample size required before a study for a target power and effect size."),
    "Type I Error": ("inference", "Rejecting a true null hypothesis — a false positive, controlled at level alpha."),
    "Two-Proportion Z-Test": ("inference", "A hypothesis test for whether two groups' success proportions differ, using a normal approximation."),
    # --- Bayesian Inference ---
    "Bayesian Decision Theory (BDT)": ("bayes", "Choosing the action that minimises expected loss under the posterior distribution."),
    "Posterior probability of uplift": ("bayes", "The posterior probability that a treatment's effect exceeds zero (or a chosen threshold)."),
    "Gaussian Processes (GPs)": ("bayes", "A nonparametric Bayesian prior over functions that yields predictions with calibrated uncertainty."),
    "Bayesian Neural Networks (BNNs)": ("bayes", "Neural networks with distributions over their weights, producing predictive uncertainty."),
    "Variational Inference (VI)": ("bayes", "Approximating an intractable posterior by optimising a simpler distribution to be close to it."),
    "MCMC (Markov Chain Monte Carlo)": ("bayes", "Sampling from a posterior via a Markov chain whose stationary distribution is that posterior."),
    "Binomial Likelihood": ("bayes", "The probability model for the number of successes in a fixed number of independent Bernoulli trials."),
    "Posterior belief": ("bayes", "Updated belief about a parameter after combining prior and data through Bayes' theorem."),
    "Marginal Likelihood (also called The Model Evidence or Integrated Likelihood)": ("bayes", "The probability of the data averaged over the prior — the normaliser and a model-comparison score."),
    "Posterior": ("bayes", "The distribution of parameters given the data; the central object of Bayesian inference."),
    "Prior Belief (or Prior Probability)": ("bayes", "Belief about a parameter before observing the current data."),
    "Parameter(s) of Interest": ("bayes", "The unknown quantities an analysis sets out to estimate."),
    "Bayes' Theorem": ("bayes", "The rule that turns a prior into a posterior using the likelihood: posterior is proportional to likelihood x prior."),
    "Posterior Probability": ("bayes", "The probability of a hypothesis or event evaluated under the posterior distribution."),
    # --- A/B Testing & Experimentation ---
    "Conversion Rate Uplift": ("abtest", "The increase in conversion rate attributable to a treatment versus control."),
    "Bayesian Stopping Rules": ("abtest", "Criteria for ending an experiment based on posterior quantities, such as probability of being best."),
    "Optimizely": ("abtest", "A commercial online experimentation and A/B-testing platform."),
    "Online Experimentation Platforms": ("abtest", "Systems that randomise, run and analyse controlled experiments on live traffic."),
    "Stopping Rules": ("abtest", "Predefined conditions that determine when to stop collecting data in a test."),
    "Treatment Effect": ("abtest", "The causal difference in outcome between treated and untreated units."),
    "Bayesian Sequential Testing": ("abtest", "Continuously monitoring an experiment in a Bayesian framework, valid to stop at any time."),
    "Group Sequential Testing": ("abtest", "Frequentist designs allowing interim analyses at planned points with adjusted error spending."),
    "Traditional A/B Test (Fixed-Horizon A/B Test)": ("abtest", "A test analysed once at a pre-committed sample size in order to control error rates."),
    "Fixed-Horizon Testing": ("abtest", "Testing where the sample size is fixed in advance and analysed only at the end."),
    "True Conversion Rate": ("abtest", "The unknown underlying probability that a user converts, estimated from observed conversions."),
    # --- Sequential Methods & Bandits ---
    "Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB))": ("bandits", "A bandit strategy that samples from posterior beliefs to balance exploration and exploitation."),
    "Sequential Settings": ("bandits", "Decision problems where data arrive over time and choices adapt as evidence accrues."),
    "Sequential Probability Ratio Test (SPRT)": ("bandits", "A sequential test that accumulates the likelihood ratio and stops when it crosses preset bounds."),
    "Pocock Method": ("bandits", "A group-sequential boundary using a constant significance threshold at every interim look."),
    "O'Brien\u2013Fleming (OBF) Method": ("bandits", "A group-sequential boundary that is very strict early and relaxes toward the final analysis."),
    # --- Business & Growth Analytics ---
    "Cross-Selling": ("growth", "Encouraging existing customers to buy complementary products."),
    "Upselling": ("growth", "Encouraging customers to buy a higher-tier or larger version of a product."),
    "Customer Segmentation": ("growth", "Dividing customers into groups with similar attributes or behaviour for targeting."),
    "SaaS (Software as a Service)": ("growth", "Software delivered and billed as an ongoing subscription service."),
    "Valuation Metric": ("growth", "A quantitative measure used to value a company or a customer relationship."),
    "D2C (Direct-to-Consumer)": ("growth", "A model where brands sell straight to consumers, bypassing intermediaries."),
    "LTV:CAC Ratio": ("growth", "The ratio of customer lifetime value to acquisition cost — a core unit-economics health metric."),
    "Net LTV (sometimes called Contribution LTV)": ("growth", "Customer lifetime value after subtracting variable and serving costs."),
    "Gross LTV (Customer Lifetime Value)": ("growth", "Total expected revenue from a customer before any cost deductions."),
    "Predictive LTV (pLTV)": ("growth", "A model-based forecast of a customer's future lifetime value."),
    "Cohort-Based LTV (Simple Version)": ("growth", "Lifetime value estimated by tracking the revenue of customer cohorts over time."),
    "Customer Lifetime": ("growth", "The expected duration of a customer's active relationship with a business."),
    "Gross Margin": ("growth", "Revenue minus cost of goods sold, expressed as an amount or a percentage."),
    "Fully Loaded CAC (Customer Acquisition Cost)": ("growth", "Acquisition cost including all sales, marketing and overhead."),
    "Organic CAC (Customer Acquisition Cost)": ("growth", "Acquisition cost attributed to unpaid, organic channels."),
    "Paid CAC (Customer Acquisition Cost)": ("growth", "Acquisition cost attributed to paid marketing spend."),
    "Channel-Specific CAC (Customer Acquisition Cost)": ("growth", "Acquisition cost computed separately for each marketing channel."),
    "Blended CAC (Customer Acquisition Cost)": ("growth", "Total acquisition spend divided by all customers acquired, across channels."),
    "Lead-Gen Software": ("growth", "Tools that capture and manage prospective-customer leads."),

    # ===================== batch 2 — pages 11-20 (+2 recovered) =====================
    # --- Statistical Inference, Testing & Power ---
    "Minimum Detectable Lift (MDL)": ("inference", "The smallest effect an experiment is powered to detect reliably."),
    "Trivial Effects": ("inference", "Effects too small to matter in practice even if statistically detectable."),
    "Sample size": ("inference", "The number of observations collected; larger samples shrink estimation error."),
    "Power (1 – β)": ("inference", "The probability a test correctly detects a real effect (rejects a false null)."),
    "Significance Level (α)": ("inference", "The tolerated false-positive probability, fixed before testing."),
    "Effect Size (δ)": ("inference", "The magnitude of a difference or relationship, independent of sample size."),
    "Hypothesis Testing": ("inference", "A framework for deciding between a null and an alternative using data."),
    "P-Value (probability value)": ("inference", "The probability of data at least as extreme as observed, assuming the null is true."),
    "Z-Test": ("inference", "A hypothesis test using the normal distribution when variance is known or n is large."),
    "T-Test": ("inference", "A hypothesis test comparing means using the t-distribution for smaller samples."),
    # --- Ranking & Interleaving ---
    "Ranking Algorithms": ("ranking", "Methods that order items by relevance or predicted value."),
    "Probabilistic Interleaving": ("ranking", "An online method that probabilistically mixes two rankers' results to compare them."),
    "Team Draft Interleaving (TDI)": ("ranking", "Interleaving where rankers alternately draft items into one list for fair comparison."),
    "Balanced Interleaving": ("ranking", "An early interleaving scheme that merges two ranked lists to attribute clicks."),
    # --- Causal Inference & Uplift ---
    "Causal Impact": ("causal", "The estimated effect of an intervention, often via a counterfactual time-series model."),
    "Causal Inference": ("causal", "Drawing cause-and-effect conclusions from data, not merely associations."),
    "Causal ML (Causal Machine Learning)": ("causal", "ML methods that estimate causal effects rather than predictive associations."),
    "Treatment Cost": ("causal", "The cost of applying an intervention to a unit in an experiment or campaign."),
    "Incremental Revenue": ("causal", "Additional revenue caused by an intervention beyond the baseline."),
    "Incremental Recovery Rate (IRR)": ("causal", "The added recovery (e.g. in collections) attributable to a treatment."),
    "Incremental Sales": ("causal", "Extra sales generated by an intervention versus doing nothing."),
    "Random Targeting Strategy": ("causal", "A random-selection baseline used to benchmark uplift models."),
    "Cumulative Uplift": ("causal", "The running incremental effect as more of the targeted population is treated."),
    "Incremental Gain": ("causal", "The improvement in outcome from targeting versus a baseline."),
    # --- Bandits & A/B Testing ---
    "Bandit Algorithms": ("bandits", "Strategies that allocate trials to options to maximise reward while learning."),
    "A/B/n Test": ("abtest", "An experiment comparing more than two variants simultaneously."),
    "Multivariate Test (MVT)": ("abtest", "An experiment varying several elements at once to estimate combined effects."),
    "Risk of Peeking": ("abtest", "The inflated false-positive risk from repeatedly checking a fixed-horizon test early."),
    # --- Business & Growth Analytics ---
    "Session Length": ("growth", "The duration of a user's single visit or session."),
    "Revenue per User (RPU / ARPU)": ("growth", "Average revenue generated per user over a period."),
    "Churn": ("growth", "The rate at which customers stop using a product or cancel."),
    "Retention": ("growth", "The share of users who remain active over a given period."),
    "KYC": ("growth", "'Know Your Customer' — regulatory identity-verification processes."),
    "FTEs": ("growth", "Full-Time Equivalents — staffing normalised to full-time headcount."),
    "OpEx": ("growth", "Operating expenses — the ongoing cost of running operations."),
    "Lagging Indicators": ("growth", "Metrics that confirm trends after they occur, such as revenue or churn."),
    "Leading Indicators": ("growth", "Early-signal metrics that predict future outcomes."),
    "Cohort": ("growth", "A group of users sharing a common start characteristic, tracked over time."),
    "ROI (Return on Investment)": ("growth", "The ratio of net gain to the cost of an investment."),
    # --- Probability & Statistics Foundations ---
    "Statistically Significant": ("probstats", "Describing a result unlikely under the null hypothesis at the chosen level."),
    "IID (Independent and Identically Distributed)": ("probstats", "An assumption that samples are mutually independent and share one distribution."),
    "Beta Distribution": ("probstats", "A continuous distribution on [0, 1]; the conjugate prior for a binomial probability."),
    "Population Proportion": ("probstats", "The fraction of an entire population with a given attribute."),
    # --- Signal Processing & Time Series ---
    "Temporal autocorrelation (Serial Correlation)": ("signal", "Correlation of a time series with its own past values."),
    "Windows (in Time-Series)": ("signal", "Fixed spans over which time-series features or aggregates are computed."),
    # --- Validation & Cross-Validation ---
    "Blocked Splits (Single Holdout)": ("validation", "Splitting time-ordered data into contiguous train/test blocks to avoid leakage."),
    "Sliding Window (Rolling Window) Cross-Validation": ("validation", "Time-series CV using a fixed-size window that moves forward through time."),
    "Expanding Window Cross-Validation": ("validation", "Time-series CV that grows the training window as it walks forward."),
    "Data Leakage": ("validation", "When information from outside the training set leaks in, inflating performance."),
    "Stratified Group K-Fold": ("validation", "K-fold CV preserving class balance while keeping groups intact across folds."),
    "Stratified Shuffle Split": ("validation", "Repeated random splits that preserve class proportions in each split."),
    "Multiclass stratified CV": ("validation", "Stratified cross-validation maintaining each class's proportion across folds."),
    "k-fold cross-validation": ("validation", "Train on k-1 folds and test on the held-out fold, rotating through all k."),
    "Cross-Validation (CV)": ("validation", "Estimating generalisation by repeatedly training and testing on different splits."),
    # --- Model Training & Optimization ---
    "Model Distillation (Knowledge Distillation)": ("training", "Training a small student model to mimic a larger teacher for efficiency."),
    "Early Stopping": ("training", "Halting training when validation performance stops improving, to curb overfitting."),
    "Epochs": ("training", "One full pass of the training algorithm over the entire dataset."),
    "Hyperparameter": ("training", "A setting fixed before training (e.g. learning rate) rather than learned."),
    "Ensemble": ("training", "Combining several models' predictions to improve accuracy and robustness."),
    "Model Weights": ("training", "The learned parameters that define a trained model's behaviour."),
    "FLOPs": ("training", "Floating-point operations — a measure of a model's compute cost."),
    "Active Learning": ("training", "Iteratively querying the most informative examples to label, cutting labelling cost."),
    # --- MLOps, Serving & Monitoring ---
    "Re-scoring": ("mlops", "Recomputing model scores on data, e.g. after retraining or recalibration."),
    "Recalibration": ("mlops", "Re-aligning predicted probabilities with observed frequencies after drift."),
    "Reweighting": ("mlops", "Adjusting sample or class weights to correct bias or distribution shift."),
    "Continuous Retraining": ("mlops", "Automatically retraining models on fresh data to counter drift."),
    "Monitoring Pipelines": ("mlops", "Automated systems that track model and data health in production."),
    "Recalibrate Thresholds": ("mlops", "Updating decision cut-offs as data or costs change."),
    "Guardrails (in ML & Data Systems)": ("mlops", "Automated checks that keep model behaviour within safe limits."),
    "Model KPIs (Key Performance Indicators)": ("mlops", "Metrics tracked to judge a deployed model's ongoing performance."),
    "Model Stability": ("mlops", "The consistency of a model's predictions and performance over time."),
    "Feature Values": ("mlops", "The actual input feature values fed to a model at scoring time."),
    "SLI (Service Level Indicator)": ("mlops", "A measured signal such as latency or error rate used to track service health."),
    "AWS SageMaker Endpoints": ("mlops", "Managed HTTPS endpoints that serve SageMaker model predictions."),
    "Cloud Inference with Big Payloads": ("mlops", "Serving predictions on large inputs, needing batching or async handling."),
    "Cloud Inference": ("mlops", "Running model predictions on managed cloud infrastructure."),
    # --- Distribution Shift & Drift ---
    "Drift Detection": ("drift", "Monitoring for changes in data or target distributions that degrade a model."),
    "Representation Shift": ("drift", "A change in learned feature representations between training and serving."),
    "Classifier Two-Sample Tests (C2STs)": ("drift", "Testing for shift by checking whether a classifier can tell two samples apart."),
    "Energy Distance": ("drift", "A distance between distributions used to test whether two samples differ."),
    "Maximum Mean Discrepancy (MMD)": ("drift", "A kernel-based distance between distributions for two-sample and drift tests."),
    "Cardinality in Categorical Data": ("drift", "The number of distinct values a categorical feature can take."),
    "Categorical Drift": ("drift", "Shifts in the distribution of categorical feature values over time."),
    "Macro Shifts": ("drift", "Broad, large-scale changes in data distribution affecting many features."),
    "Categorical Explosions": ("drift", "A surge in distinct categorical values that strains encoders and models."),
    "Off-Distribution": ("drift", "Inputs that fall outside the distribution a model was trained on."),
    # --- Representations & Embeddings ---
    "Autoencoder": ("repr", "A neural network trained to reconstruct its input through a compressed latent code."),
    "Frozen Encoder": ("repr", "A pretrained encoder whose weights stay fixed while downstream layers train."),
    "Embedding": ("repr", "A dense vector representation that places similar items near each other."),
    # --- Classification & Averaging Metrics ---
    "Discriminatory Power": ("metrics", "A model's ability to separate positive from negative cases."),
    # --- Model Evaluation & Uncertainty ---
    "Cramér's V": ("evaluation", "A measure of association between two categorical variables, derived from chi-squared."),
    "KS Statistic (Kolmogorov–Smirnov Statistic)": ("evaluation", "The maximum gap between two cumulative distributions; measures separation or shift."),
    # --- Fairness & Calibration ---
    "Four-Fifths (80%) Rule": ("fairness", "An adverse-impact guideline flagging a group selection rate below 80% of the top group's."),
    # --- Bayesian Inference ---
    "Bayesian Correction": ("bayes", "Adjusting estimates using prior information within a Bayesian framework."),
    # --- AI & ML Concepts ---
    "AI (Artificial Intelligence)": ("concepts", "Systems that perform tasks normally requiring human intelligence."),
    "Machine Learning (ML)": ("concepts", "Algorithms that learn patterns from data rather than being explicitly programmed."),
    "Medical AI": ("concepts", "The application of AI to clinical and healthcare problems."),
    "LLMs (Large Language Models)": ("concepts", "Large neural networks trained on text to understand and generate language."),
    # --- ML Platforms & Tools ---
    "AWS SageMaker": ("platforms", "Amazon's managed platform for building, training and deploying ML models."),
    "Vertex AI": ("platforms", "Google Cloud's managed platform for ML model development and deployment."),
    "OpenAI API (ML API)": ("platforms", "A hosted API for accessing OpenAI's models programmatically."),
    "Google Experiments": ("platforms", "Google's online experimentation / A-B testing tooling."),

    # ===================== batch 3 — pages 21-30 =====================
    # --- Causal Inference & Uplift ---
    "Total Incremental Benefit (TIB)": ("causal", "The total added benefit from treating a targeted population versus not."),
    "Cumulative Incremental Gain (CIG)": ("causal", "The running sum of incremental gain as more of the ranked population is treated."),
    "Qini Curve": ("causal", "An uplift-evaluation curve plotting incremental gain against the fraction targeted."),
    "Uplift Score": ("causal", "A model's estimate of how much an action changes an individual's outcome."),
    "Uplift Models": ("causal", "Models that predict the incremental effect of treating each individual."),
    # --- MLOps, Serving & Monitoring ---
    "Ops Health Dashboard": ("mlops", "A dashboard tracking the operational health of deployed systems."),
    "SLA Breach Rate": ("mlops", "The fraction of time a service fails to meet its agreed service level."),
    "SLA (Service Level Agreement)": ("mlops", "A contract specifying expected service performance and reliability targets."),
    # --- Operations & Supply Chain ---
    "Supplier Constraints": ("ops", "Limits imposed by suppliers (capacity, timing) that affect planning."),
    "Long Lead Times": ("ops", "Extended delays between ordering and receiving goods."),
    "Slow-Moving SKUs": ("ops", "Products that sell slowly and risk tying up inventory."),
    "SKU": ("ops", "Stock-Keeping Unit — a unique identifier for a distinct product."),
    "Real-Time Inventory Tracking": ("ops", "Continuously monitoring stock levels as they change."),
    "Supplier Management": ("ops", "Coordinating and optimising relationships with suppliers."),
    "Demand Forecasting": ("ops", "Predicting future demand to guide inventory and production."),
    "Reorder Point (ROP) Optimization": ("ops", "Setting the stock level at which to reorder to avoid stockouts."),
    "Safety Stock": ("ops", "Extra inventory held to buffer against demand or supply variability."),
    "Backorder Rate": ("ops", "The share of orders that cannot be filled immediately from stock."),
    "Lost Sales Value": ("ops", "Revenue forgone when demand cannot be met from available stock."),
    "Fill Rate": ("ops", "The fraction of demand satisfied directly from inventory."),
    "Stockout Rate": ("ops", "The frequency with which an item is out of stock."),
    "Long-Tail Items": ("ops", "Numerous low-demand products that collectively matter but are hard to forecast."),
    # --- Signal Processing & Time Series ---
    "Prophet — Time Series Forecasting by Facebook (Meta)": ("signal", "An open-source library for decomposable time-series forecasting with trend and seasonality."),
    "LSTM — Long Short-Term Memory Networks": ("signal", "A recurrent network with gating that captures long-range dependencies in sequences."),
    "ARIMA (AutoRegressive Integrated Moving Average)": ("signal", "A classic model combining autoregression, differencing and moving-average terms for forecasting."),
    "M-Competitions (Makridakis Competitions)": ("signal", "Influential forecasting competitions benchmarking time-series methods."),
    "Forecasting Benchmarks": ("signal", "Standard datasets and baselines for comparing forecasting methods."),
    "Seasonal Lag": ("signal", "The offset to the same point in a previous season, used in seasonal models."),
    "Simple Baseline Methods": ("signal", "Easy reference forecasts (last value, mean) that stronger models must beat."),
    "Naïve Baseline Forecast": ("signal", "A baseline predicting the next value equals the most recent observation."),
    "Forecast Error": ("signal", "The difference between a forecast and the realised value."),
    "Forecasting Competitions": ("signal", "Public contests that benchmark forecasting accuracy across methods."),
    "Time Series Forecasting": ("signal", "Predicting future values of a time-ordered series."),
    "Log-Space": ("signal", "Working with log-transformed values to stabilise variance or handle scale."),
    # --- Risk & Probabilistic Forecasting ---
    "Return Distribution": ("risk", "The distribution of asset returns, central to financial risk modelling."),
    "Value-at-Risk (VaR)": ("risk", "A threshold loss unlikely to be exceeded at a given confidence over a horizon."),
    "Risk Forecast": ("risk", "A forward-looking estimate of potential loss or adverse outcomes."),
    "Probabilistic Scoring": ("risk", "Evaluating forecasts by how well their predicted probabilities match outcomes."),
    "Full Distribution": ("risk", "Predicting the entire outcome distribution rather than a single value."),
    "Continuous Probabilistic Forecasts": ("risk", "Forecasts expressed as continuous probability distributions over outcomes."),
    "Quantile Forecasts": ("risk", "Forecasts of specific quantiles of the outcome distribution."),
    "Point Forecasts": ("risk", "A single best-estimate prediction, without an uncertainty range."),
    "Strictly Proper Scoring Rules": ("risk", "Scoring rules minimised only by reporting the true probability distribution."),
    "Probability Forecasts": ("risk", "Forecasts stated as probabilities of events."),
    "Probabilistic Forecasts": ("risk", "Forecasts that quantify uncertainty as a full probability distribution."),
    "Deterministic forecasts": ("risk", "Single-valued forecasts that carry no explicit uncertainty."),
    "Predicting Percentiles": ("risk", "Forecasting specific percentiles to convey the outcome distribution."),
    "Prediction Intervals (PI)": ("risk", "A range expected to contain the outcome with a stated probability."),
    "Quantile Regression": ("risk", "Regression that estimates conditional quantiles rather than the mean."),
    "Quantile Level": ("risk", "The probability level (e.g. 0.9) targeted by a quantile forecast."),
    "Risk-Based Decisions": ("risk", "Choosing actions by weighing outcome probabilities against their costs."),
    # --- Classification & Averaging Metrics ---
    "Classification Probability": ("metrics", "The probability a model assigns to a class for an instance."),
    "ROC Curve (Receiver Operating Characteristic)": ("metrics", "A plot of true- versus false-positive rate across thresholds."),
    "Binary Classification": ("metrics", "Predicting one of two classes for each instance."),
    # --- AI & ML Concepts ---
    "Target Variable": ("concepts", "The quantity a supervised model is trained to predict."),
    "Support Vector Machines (SVMs)": ("concepts", "Classifiers that find the maximum-margin boundary, optionally via kernels."),
    "Neural Networks": ("concepts", "Layered models of interconnected units that learn complex functions from data."),
    "Logistic Regression": ("concepts", "A linear model mapping features to a probability via the logistic function."),
    "Classification Models": ("concepts", "Models that assign inputs to discrete categories."),
    # --- Probability & Statistics Foundations ---
    "Probability Density": ("probstats", "The relative likelihood of a continuous variable at a value, given by its PDF."),
    "Normal Distribution": ("probstats", "The bell-shaped Gaussian distribution defined by its mean and variance."),
    "Probability Mass": ("probstats", "The probability assigned to each value of a discrete variable."),
    "Probability Distribution": ("probstats", "A description of how probability is spread over a variable's possible values."),
    "Cumulative Distribution Function (CDF)": ("probstats", "The probability that a variable is at most a given value."),
    "Confidence Level": ("probstats", "The long-run proportion of intervals expected to contain the true parameter."),
    # --- Model Evaluation & Uncertainty ---
    "Average Absolute Error (AAE)": ("evaluation", "The mean of absolute differences between forecasts and outcomes."),
    "Relative accuracy": ("evaluation", "Forecast accuracy measured against a baseline rather than in absolute terms."),
    "R² (R-squared)": ("evaluation", "The share of variance in the target explained by a regression model."),
    # --- Recommender Systems ---
    "Self-Information of Popularity": ("recsys", "An information-theoretic weighting that rewards recommending less popular items."),
    "Relevance in Recommender Systems": ("recsys", "How well a recommended item matches a user's interests."),
    "Genre Overlap": ("recsys", "The degree to which recommended items share genres, a diversity signal."),
    "Jaccard index": ("recsys", "Intersection over union of two sets, a similarity measure."),
    "Cosine Similarity of Item Features": ("recsys", "Similarity as the cosine of the angle between two items' feature vectors."),
    "Intra-List Diversity (ILD)": ("recsys", "The average dissimilarity among items within a single recommendation list."),
    "Dominating in Recommender Systems": ("recsys", "When a few popular items crowd out the rest of recommendations."),
    "Catalog Coverage": ("recsys", "The share of the item catalog that ever gets recommended."),
    "User Coverage": ("recsys", "The share of users for whom the system can make useful recommendations."),
    "Item Coverage": ("recsys", "The share of items that the system is able to recommend."),
    "Diminishing Utility": ("recsys", "The decreasing marginal value of additional similar or lower-ranked items."),
    # --- Ranking & Interleaving ---
    "DCG (Discounted Cumulative Gain)": ("ranking", "A ranking metric that rewards relevant items more when ranked higher."),
    "TREC (Text REtrieval Conference)": ("ranking", "A long-running benchmark effort for information-retrieval evaluation."),
    # --- ML Platforms & Tools ---
    "Kaggle": ("platforms", "A platform hosting data-science competitions, datasets and notebooks."),
    # --- Probability Calibration ---
    "Adaptive ECE (Expected Calibration Error with Adaptive Binning)": ("calibration", "A calibration error using adaptive bins so each holds a similar count."),
    "Maximum Calibration Error (MCE)": ("calibration", "The largest gap between confidence and accuracy across calibration bins."),
    "Murphy's Decomposition": ("calibration", "A breakdown of a probabilistic score into calibration and refinement parts."),
    "Temperature Scaling": ("calibration", "A simple post-hoc method that rescales logits to calibrate probabilities."),
    "Platt Scaling": ("calibration", "Fitting a logistic function to scores to produce calibrated probabilities."),
    "Isotonic Regression": ("calibration", "A nonparametric, monotonic fit used to calibrate predicted probabilities."),
    "Underconfident": ("calibration", "When predicted probabilities are less extreme than the true accuracy warrants."),
    "Overconfident": ("calibration", "When predicted probabilities are more extreme than the true accuracy warrants."),
    # --- Model Training & Optimization ---
    "Binary Cross-Entropy (BCE)": ("training", "The standard loss for binary classification, penalising confident wrong predictions."),
    "Loss Functions": ("training", "Objectives quantifying prediction error that training seeks to minimise."),
    "Underflow": ("training", "Numerical loss of precision when values become too small to represent."),
    "Logit Space": ("training", "The pre-activation, log-odds scale on which linear models and nets operate."),
    "Log-Odds": ("training", "The logarithm of the odds, the natural scale for logistic models."),
    "Softmax Function": ("training", "Converts a vector of scores into a probability distribution over classes."),
    "Sigmoid Function": ("training", "Maps any real value to (0, 1), used for probabilities and gating."),
    "Squashing Function": ("training", "Any bounded nonlinearity (sigmoid, tanh) that compresses its input range."),
    # --- Business & Growth Analytics ---
    "Conversion Rate (CR)": ("growth", "The share of users who complete a desired action."),
    "Cost-Per-Click (CPC) Models": ("growth", "Advertising pricing where payment is per click received."),

    # ===================== batch 4 — pages 31-44 (completes 431) =====================
    # --- Causal Inference & Uplift ---
    "Causal Trees": ("causal", "Decision trees that partition data by differences in treatment effect."),
    "Uplift Random Forests": ("causal", "An ensemble of trees that estimates individual-level treatment effects."),
    "Uplift Curve": ("causal", "A curve showing cumulative incremental gain as more of the ranked population is treated."),
    "Causal Effect": ("causal", "The change in an outcome caused by an intervention, all else equal."),
    "Revenue net of treatment cost": ("causal", "Incremental revenue from a treatment after subtracting its cost."),
    "Incremental Conversions": ("causal", "Extra conversions caused by a treatment beyond the baseline."),
    "Uplift@k": ("causal", "The incremental gain captured within the top-k targeted population."),
    "AUUC (Area Under the Uplift Curve)": ("causal", "A summary of uplift-model quality as the area under its uplift curve."),
    "Qini Coefficient": ("causal", "A normalised summary of the Qini curve measuring uplift-model performance."),
    "Uplift": ("causal", "The incremental effect of an action on an individual's outcome."),
    # --- Probability & Statistics Foundations ---
    "Likelihood": ("probstats", "How probable observed data are under a model's parameters."),
    "Correlation": ("probstats", "The strength and direction of a linear relationship between variables."),
    "Outlier": ("probstats", "An observation far from the bulk of the data."),
    "Median": ("probstats", "The middle value separating the higher and lower halves of data."),
    "Mean": ("probstats", "The arithmetic average of a set of values."),
    # --- Model Evaluation & Uncertainty ---
    "Mean Squared Error (MSE)": ("evaluation", "The average of squared differences between predictions and actuals."),
    "DeLong's Test": ("evaluation", "A statistical test comparing the AUCs of two models."),
    "Bootstrap": ("evaluation", "Resampling with replacement to estimate the variability of a statistic."),
    "MASE (Mean Absolute Scaled Error)": ("evaluation", "Forecast error scaled by a naive baseline's error, comparable across series."),
    "WMAPE (Weighted Mean Absolute Percentage Error)": ("evaluation", "MAPE weighted by volume so large items count more."),
    "sMAPE (Symmetric Mean Absolute Percentage Error)": ("evaluation", "A symmetric percentage error bounded between 0 and 200%."),
    "RMSLE (Root Mean Squared Logarithmic Error)": ("evaluation", "RMSE on log-scaled values, penalising under-prediction and easing large magnitudes."),
    "Mean Absolute Error (MAE)": ("evaluation", "The average absolute difference between predictions and actuals."),
    "Coverage": ("evaluation", "The share of outcomes that fall within predicted intervals."),
    "WAPE (Weighted Absolute Percentage Error)": ("evaluation", "Total absolute error divided by total actuals."),
    "Mean Absolute Percentage Error (MAPE)": ("evaluation", "The average absolute error expressed as a percentage of actuals."),
    "Root Mean Squared Error (RMSE)": ("evaluation", "The square root of mean squared error, in the target's units."),
    "Baseline Heuristics": ("evaluation", "Simple rules used as reference points to judge whether a model adds value."),
    # --- Classification & Averaging Metrics ---
    "One-vs-Rest (OvR)": ("metrics", "A multiclass strategy fitting one binary classifier per class."),
    "Multiclass Classification": ("metrics", "Assigning each instance to one of three or more classes."),
    "Partial AUC (pAUC)": ("metrics", "AUC restricted to a region of interest of the ROC curve."),
    "Micro AUC": ("metrics", "AUC pooled across all classes by aggregating decisions before averaging."),
    "Macro AUC": ("metrics", "AUC averaged equally across per-class scores."),
    "Accuracy": ("metrics", "The fraction of predictions that are correct."),
    "Per-class Precision (sometimes called class-wise precision)": ("metrics", "Precision computed separately for each class."),
    "Multiclass Precision": ("metrics", "Precision aggregated across the classes of a multiclass problem."),
    "Multilabel Precision": ("metrics", "Precision when each instance may carry several labels at once."),
    "Weighted Averaging": ("metrics", "Averaging per-class metrics weighted by class support."),
    "Harmonic Mean": ("metrics", "The reciprocal-based mean underlying the F1 score."),
    "F1-score": ("metrics", "The harmonic mean of precision and recall."),
    "Model Score": ("metrics", "The raw numeric output a model assigns before thresholding."),
    "Average Precision (AP)": ("metrics", "The area under the precision-recall curve summarising ranked retrieval."),
    "Micro Averaging": ("metrics", "Aggregating counts across classes before computing a metric."),
    "Macro Averaging": ("metrics", "Computing a metric per class then averaging them equally."),
    "AUC (Area Under the Curve)": ("metrics", "The area under a ROC or PR curve summarising performance across thresholds."),
    "Log Loss (also called Logarithmic Loss or Cross-Entropy Loss)": ("metrics", "A loss penalising confident wrong probabilistic predictions."),
    "Recall": ("metrics", "The share of actual positives the model correctly identifies."),
    "ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)": ("metrics", "The probability a random positive outranks a random negative."),
    "Precision (a.k.a. Positive Predictive Value, PPV)": ("metrics", "The share of predicted positives that are truly positive."),
    "Precision–Recall AUC (PR-AUC)": ("metrics", "Area under the precision-recall curve, informative under class imbalance."),
    # --- AI & ML Concepts ---
    "Regression Models": ("concepts", "Models that predict continuous numeric outcomes."),
    "Computer Vision (CV)": ("concepts", "The field of teaching machines to interpret images and video."),
    "Natural Language Processing (NLP)": ("concepts", "The field of processing and understanding human language with machines."),
    "Decision Trees": ("concepts", "Models that split data on feature thresholds to reach predictions."),
    "Linear Models": ("concepts", "Models predicting from a weighted sum of features."),
    # --- Statistical Inference, Testing & Power ---
    "Chi-square (χ²) Test": ("inference", "A test of association between categorical variables using expected counts."),
    "Kolmogorov–Smirnov (KS) Test": ("inference", "A test comparing distributions via their largest cumulative gap."),
    "Statistical Tests": ("inference", "Procedures for deciding whether data support a hypothesis."),
    "Statistical Power": ("inference", "The probability of detecting a true effect when one exists."),
    "Clopper–Pearson Interval": ("inference", "An exact confidence interval for a binomial proportion."),
    "Wilson Score Interval": ("inference", "An accurate confidence interval for a proportion, robust for small samples."),
    "Confidence Intervals (CIs)": ("inference", "A range that would contain the true parameter a stated fraction of the time."),
    "Power Analysis": ("inference", "Planning sample size so a test can detect an effect of interest."),
    # --- Distribution Shift & Drift ---
    "Jensen–Shannon (JS) Divergence": ("drift", "A symmetric, bounded measure of difference between two distributions."),
    "Kullback–Leibler (KL) Divergence": ("drift", "An asymmetric measure of how one distribution differs from another."),
    "Concept Drift": ("drift", "When the relationship between inputs and target changes over time."),
    "Data Drift": ("drift", "When the distribution of input data shifts away from training."),
    "Dataset Shift": ("drift", "Any mismatch between training and deployment data distributions."),
    "Drift Guardrails": ("drift", "Automated thresholds that flag or block on detected drift."),
    "Label Drift (a.k.a. Target Drift)": ("drift", "A change over time in the distribution of the target variable."),
    "Covariate Drift (a.k.a. Covariate Shift)": ("drift", "A change in the input distribution while the input-output relationship holds."),
    "KS shift (Kolmogorov–Smirnov shift)": ("drift", "Using the KS statistic to quantify distribution shift in a feature."),
    "PSI (Population Stability Index)": ("drift", "A widely used score measuring how much a distribution has shifted."),
    # --- Explainability & Governance ---
    "Fair Lending laws": ("xai", "Regulations prohibiting discrimination in credit decisions."),
    "Basel III": ("xai", "International banking rules on capital and risk management."),
    "High-Stakes Domains": ("xai", "Settings like health or credit where model errors carry serious consequences."),
    "Counterfactual Explanations": ("xai", "Explaining a decision by the smallest input change that would flip it."),
    "LIME (Local Interpretable Model-agnostic Explanations)": ("xai", "Explaining a prediction by fitting a simple local surrogate model."),
    "SHAP (SHapley Additive exPlanations)": ("xai", "Attributing a prediction to features using Shapley values from game theory."),
    "Post-hoc Explainability": ("xai", "Interpreting an already-trained model rather than building it interpretable."),
    # --- Model Training & Optimization ---
    "Deep Ensembles": ("training", "Averaging several independently trained networks for accuracy and uncertainty."),
    "Quantization": ("training", "Reducing numeric precision of weights to shrink and speed up models."),
    "Full Annotation": ("training", "Labelling every example fully, the most costly supervision setting."),
    "Weak Supervision": ("training", "Training from noisy, heuristic or partial labels instead of clean ones."),
    "Label Noise": ("training", "Errors in training labels that can mislead a model."),
    "Logits": ("training", "Raw pre-activation scores before a sigmoid or softmax."),
    # --- Data Preparation & Features ---
    "Sensitivity in Feature Engineering": ("features", "How much a model's output responds to changes in a feature."),
    "Encode (in Feature Engineering)": ("features", "Converting categorical or text data into numeric form for models."),
    "Normalize (in Feature Engineering)": ("features", "Rescaling features to a common range or distribution."),
    "Advanced Sorting in Spreadsheets": ("features", "Ordering and arranging tabular data, including with dynamic sort functions."),
    # --- Representations & Embeddings ---
    "Embedding Similarity": ("repr", "Measuring how close two items are in a learned embedding space."),
    # --- Signal Processing & Time Series ---
    "Seasonality": ("signal", "Regular, calendar-linked cycles in a time series."),
    # --- MLOps, Serving & Monitoring ---
    "Caching": ("mlops", "Storing computed results to serve repeated requests faster."),
    "Latency Guardrails": ("mlops", "Thresholds that alert or act when response times exceed limits."),
    "Compute budgets": ("mlops", "Limits on the compute resources a workload may consume."),
    "Manual review minutes": ("mlops", "Human-review effort spent per period, an operational cost metric."),
    "Inference Cost (Inference $)": ("mlops", "The monetary cost of serving model predictions."),
    "SLOs (Service Level Objectives)": ("mlops", "Internal targets for reliability that a service aims to meet."),
    "SLA Breaches": ("mlops", "Events where service performance falls below its agreement."),
    # --- ML Platforms & Tools ---
    "ONNX (Open Neural Network Exchange)": ("platforms", "An open format for exchanging trained models across frameworks."),
    "TPU Clusters": ("platforms", "Groups of tensor-processing units for large-scale training and inference."),
    # --- Validation & Cross-Validation ---
    "Evaluation Set": ("validation", "Held-out data used to measure model performance."),
    "Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation)": ("validation", "Validation that respects time order to avoid using the future to predict the past."),
    "k-fold Stratified Cross-Validation (Stratified CV)": ("validation", "K-fold CV that preserves class proportions in every fold."),
    # --- Fairness & Calibration ---
    "Fairness Guardrails": ("fairness", "Automated checks that enforce fairness constraints on outputs."),
    "Fairness parity": ("fairness", "Equalising a chosen metric across protected groups."),
    "Selection Rate": ("fairness", "The fraction of a group that receives a positive decision."),
    # --- Imbalanced Learning & Resampling ---
    "Upsampling": ("imbalance", "Increasing minority-class examples to balance a dataset."),
    "Downsampling": ("imbalance", "Reducing majority-class examples to balance a dataset."),
    # --- Bayesian Inference ---
    "Bayesian Inference.": ("bayes", "Updating beliefs about parameters using priors and observed data."),
    # --- A/B Testing & Experimentation ---
    "Sequential Testing (also called sequential analysis)": ("abtest", "Analysing results as data arrive while controlling error from repeated looks."),
    "A/B Testing": ("abtest", "A randomised experiment comparing two variants to measure an effect."),
    # --- Ranking & Interleaving ---
    "Interleaving Tests": ("ranking", "Online ranker comparison that blends two result lists and attributes clicks."),
    "NDCG (Normalized Discounted Cumulative Gain)": ("ranking", "DCG normalised by the ideal ordering, bounded between 0 and 1."),
    "Mean Average Precision (MAP)": ("ranking", "The mean of average-precision scores across queries."),
    # --- Business & Growth Analytics ---
    "LTV (Customer Lifetime Value)": ("growth", "The total value a customer is expected to generate over time."),
    "CAC (Customer Acquisition Cost)": ("growth", "The average cost to acquire one new customer."),
    "Cannibalization": ("growth", "When a new offering eats into sales of an existing one."),
    "CTR (Click-Through Rate)": ("growth", "The share of impressions that result in a click."),
    # --- Operations & Supply Chain ---
    "Crew Overtime": ("ops", "Extra paid hours worked beyond scheduled time, an operations cost."),
    "Overstock %": ("ops", "The share of inventory held in excess of demand."),
    "Stockouts": ("ops", "Events where demand cannot be met because an item is unavailable."),
    # --- Risk & Probabilistic Forecasting ---
    "Continuous Ranked Probability Score (CRPS)": ("risk", "A proper score comparing a full predicted distribution to the outcome."),
    "Pinball Loss (a.k.a. Quantile Loss)": ("risk", "The loss minimised by an accurate quantile forecast."),
    # --- Recommender Systems ---
    "Novelty (in Recommender Systems)": ("recsys", "How unfamiliar or unexpected recommended items are to the user."),
    "Diversity (in Recommender Systems)": ("recsys", "How varied the items within a recommendation list are."),
    "Hit Rate (HR)": ("recsys", "The share of users for whom a relevant item appears in the top-N list."),
    # --- Probability Calibration ---
    "Expected Calibration Error (ECE)": ("calibration", "The average gap between confidence and accuracy across probability bins."),
    "Reliability Curves (also called Calibration Curves)": ("calibration", "Plots of predicted probability against observed frequency."),
    "Brier Score": ("calibration", "The mean squared error of probabilistic predictions."),
    "Calibration quality (Model Calibration)": ("calibration", "How well predicted probabilities match real-world frequencies."),
}

RULE = "=" * 70
SUB = "-" * 70


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-{2,}", "-", text)


def load_inventory() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with INVENTORY.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            rows.append((r["title"].strip(), r["url"].strip()))
    return rows




# ----------------------------------------------------------------------
# Emission: one self-contained page per term + a semantic grid hub.
#   terminology/index.rst        -> banner + intro + per-level grid of
#                                   theme cards (each lists its terms),
#                                   theme/legacy anchors, hidden toctree
#   terminology/NNN-<slug>.rst   -> one page per term (stable NNN id =
#                                   inventory order): banner, overlined
#                                   title, lead gloss, FULL body from
#                                   term_content.CONTENT[title] when
#                                   present (else a transitional summary),
#                                   related-terms links, theme/back nav,
#                                   provenance, controlled tags.
# Re-runnable: same inputs -> identical tree. Old NNN-*.rst are cleared
# first so the set never drifts.
# ----------------------------------------------------------------------
PAGES_DIR = HERE                        # NNN-*.rst pages are siblings of this script
RAW_HTML = ".. role:: raw-html(raw)\n   :format: html"
BR_HTML = ".. |br| raw:: html\n\n   <br/>"
LEVEL_WORD = {"foundations": "beginner", "applied": "intermediate", "advanced": "advanced"}


def _h(s: str) -> str:
    """Escape for raw-HTML banner text."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bar(title: str) -> str:
    return "=" * (len(title) + 2)


def _banner(emoji: str, title: str, big: bool = False) -> str:
    # w(":raw-html:`<div style=\"text-align:center\"><strong>` 📖 Terminology")
    size = "1.55rem;font-weight:700" if big else "1.12rem"
    return (f':raw-html:`<div align="center" style="text-align:center;font-size:{size};'
            f'margin:0.45rem 0 0.2rem">{emoji}&nbsp;&nbsp;'
            f'{"" if big else "<b>"}{_h(title)}{"" if big else "</b>"}</div>`')


def main() -> int:
    inv = load_inventory()
    inv_titles = [t for t, _ in inv]
    inv_set = set(inv_titles)
    enrich_titles = set(ENRICH)

    missing = sorted(inv_set - enrich_titles)
    extra = sorted(enrich_titles - inv_set)
    if missing or extra:
        if missing:
            print("ERROR: inventory titles with no ENRICH entry:", file=sys.stderr)
            for m in missing:
                print(f"   - {m!r}", file=sys.stderr)
        if extra:
            print("ERROR: ENRICH entries not present in inventory:", file=sys.stderr)
            for e in extra:
                print(f"   - {e!r}", file=sys.stderr)
        return 1

    # optional full-content store: {title: rst-body-str}
    try:
        from term_content import CONTENT  # type: ignore
    except Exception:
        CONTENT = {}
    bad_keys = sorted(set(CONTENT) - inv_set)
    if bad_keys:
        print("ERROR: term_content.CONTENT keys not in inventory:", file=sys.stderr)
        for b in bad_keys:
            print(f"   - {b!r}", file=sys.stderr)
        return 1

    # optional curated cross-topic mind map: {title: [related title, ...]}
    try:
        from term_content import MINDMAP  # type: ignore
    except Exception:
        MINDMAP = {}
    bad_mm = sorted((set(MINDMAP) | {m for v in MINDMAP.values() for m in v}) - inv_set)
    if bad_mm:
        print("ERROR: term_content.MINDMAP references not in inventory:", file=sys.stderr)
        for b in bad_mm:
            print(f"   - {b!r}", file=sys.stderr)
        return 1

    url = {t: u for t, u in inv}
    num = {t: i + 1 for i, t in enumerate(inv_titles)}   # stable id = inventory order

    # unique per-term anchors (term-<slug>, deduped)
    anchor: dict[str, str] = {}
    seen: set[str] = set()
    for t in inv_titles:
        base = "term-" + slugify(t)
        a, i = base, 2
        while a in seen:
            a = f"{base}-{i}"
            i += 1
        seen.add(a)
        anchor[t] = a

    def fileslug(t: str) -> str:
        s = slugify(t)[:48].strip("-")
        return s or "term"

    docname = {t: f"{num[t]:03d}-{fileslug(t)}" for t in inv_titles}

    # group titles by theme (alpha within theme)
    by_theme: dict[str, list[str]] = {k: [] for k in THEMES}
    for t in inv_titles:
        by_theme[ENRICH[t][0]].append(t)
    for k in by_theme:
        by_theme[k].sort(key=str.lower)

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAGES_DIR.glob("[0-9][0-9][0-9]-*.rst"):
        old.unlink()

    # ---------------- per-term pages ----------------
    n_rich = 0
    for t in inv_titles:
        key = ENRICH[t][0]
        emoji, disp, lvl, _blurb = THEMES[key]
        gloss = ENRICH[t][1]
        body = CONTENT.get(t)

        L: list[str] = []
        a = L.append
        a(":html_theme.sidebar_secondary.remove:")
        a("")
        a("..")
        a("   GENERATED FILE - do not edit by hand.")
        a("   Full body lives in term_content.py (CONTENT[title]).")
        a("   Regenerate: python build_terminology.py")
        a("")
        a(RAW_HTML)
        a("")
        a(BR_HTML)
        a("")

        a(f".. _{anchor[t]}:")
        a("")
        a(_banner(emoji, t))
        a("")
        bar = _bar(t)
        a(bar); a(t); a(bar)
        a("")
        a(f"*{gloss}*")
        a("")

        # body: full content if written, else a working stub
        if body:
            n_rich += 1
            a(body.strip())
            a("")
        else:
            a(".. note::")
            a("")
            a(f"   A full, self-contained explanation of this term is being written. "
              f"The definition above is the working summary; meanwhile, explore the "
              f"related **{_h(disp)}** terms below.")
            a("")

        a("----")
        a("")
        a(f"*Theme:* :ref:`{disp} <term-theme-{key}>` :raw-html:`&nbsp;·&nbsp;` "
          f":doc:`All terminology <index>`")
        a("")

        # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#admonitions-messages-and-warnings
        # Note      → "Be aware of this clarification or detail."          # 📝 Neutral observations, assumptions, clarifications, conventions, or exceptions.
        # See also  → "Explore these related topics and resources."        # 📚 Internal/external references, further reading, related topics, prerequisites.
        # Hint      → "This may help you understand the concept."          # 💡 Intuition, conceptual connections, mind maps, learning aids.
        # Tip       → "This may help you work more effectively."           # 💡 Best practices, shortcuts, recommendations, efficient/advice workflows.
        # Info      → "Here's additional background or context."           # ℹ️ Background, implementation notes, **sources used by this page**, supplementary factual information where the information came from.
        # Important → "Do not overlook this; it's essential."              # ⭐ Critical/Essential concepts, requirements, or limitations.

        # lateral cross-links
        mm = [s for s in MINDMAP.get(t, []) if s != t and s in docname]
        if mm:
            a("----")
            a("")
            a(".. hint::")
            a("   " + "**Mind map — connected ideas**")
            a("")
            a("   " + " · ".join(f":doc:`{s} <{docname[s]}>`" for s in mm))
            a("")

        # lateral cross-links
        sibs = [s for s in by_theme[key] if s != t]
        if sibs:
            a("----")
            a("")
            a(".. hint::")
            a("   " + f"**More in {disp}**")
            a("")
            shown = sibs[:14]
            a("   " + " · ".join(f":doc:`{s} <{docname[s]}>`" for s in shown))
            a("")

        # source (context/traceability)
        a(".. seealso::")
        a("")
        a(f"   **Source article** Adapted (context, re-expressed) in our own words from: `{t} <{url[t]}>`__ "
          f"(insightful-data-lab.com).")
        a("")

        # tags
        a(f".. tags:: purpose: reference, topic: terminology, level: {LEVEL_WORD[lvl]}")
        a("")
        (PAGES_DIR / f"{docname[t]}.rst").write_text("\n".join(L), encoding="utf-8")

    # ---------------- index hub ----------------
    active_themes = [k for k in THEME_ORDER if by_theme[k]]
    I: list[str] = []
    w = I.append
    w(":html_theme.sidebar_secondary.remove:")
    w("")
    w("..")
    w("   GENERATED FILE - do not edit by hand. Regenerate: python build_terminology.py")
    w("")
    w(RAW_HTML)
    w("")
    w(BR_HTML)
    w("")
    w(".. _terminology-index:")
    w("")
    # w(_banner("📖", "Terminology", big=True))  # ·
    w(":raw-html:`<div align=\"center\" style=\"text-align:center\"><strong>` 📖 Terminology")
    w("|br| Glossary of Artificial Intelligence and Machine Learning Terms")
    w("|br| |full_version| - |today|")
    w(":raw-html:`</strong></div>`")
    w("")
    title = "Terminology"
    w(_bar(title)); w(title); w(_bar(title))
    w("")
    w(f"A working glossary of **{len(inv_titles)} terms** spanning statistics, machine "
      f"learning, forecasting and MLOps. Every term has **its own page** with a "
      f"self-contained explanation — no need to leave to read the full context. "
      f"Entries are grouped into **{len(active_themes)} themes** across three depth "
      f"levels. **Type in the filter box** for instant lookup by name or keyword, "
      f"expand a theme to browse, or open the A–Z index at the bottom.")
    w("")

    # ---- live filter: type-to-search across every term (progressive JS) ----
    # Static, dependency-free, deterministic. Without JS the page degrades
    # gracefully to plain collapsible dropdowns.
    # ---- v2 hub: live filter + collapsed stage dropdowns + A-Z --------
    # Same pattern/classes as learn/terminology (details.sd-dropdown, .term-az)
    n_terms = len(inv_titles)
    w(".. raw:: html")
    w("")
    w('   <div style="text-align:center;margin:0.4rem 0 0.4rem">')
    w('   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"')
    w(f'          placeholder="&#128269;&nbsp; Type to filter {n_terms} terms &mdash; by name or keyword&hellip;"')
    w('          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;')
    w('                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;')  # 1px solid rgba(128,128,128,0.45)
    w('                 background:transparent;color:inherit"/>')
    w('   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;')
    w('        min-height:1.2em;margin-top:0.35rem"></div>')
    w("   </div>")
    w("   <script>")
    w("   document.addEventListener('DOMContentLoaded',function(){")
    w("     var inp=document.getElementById('term-filter');if(!inp){return;}")
    w("     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));")
    w("     var az=document.querySelector('details.term-az');")
    w("     var items=[];")
    w("     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(")
    w("       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});")
    w("     var cnt=document.getElementById('term-filter-count');")
    w("     inp.addEventListener('input',function(){")
    w("       var q=inp.value.trim().toLowerCase();var n=0;")
    w("       dds.forEach(function(d){d.tHits=0;});")
    w("       items.forEach(function(it){")
    w("         var hit=!q||it.t.indexOf(q)!==-1;")
    w("         it.li.style.display=hit?'':'none';")
    w("         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});")
    w("       dds.forEach(function(d){")
    w("         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}")
    w("         else{d.style.display='';d.open=false;}});")
    w(f"       if(cnt){{cnt.textContent=(q&&az)?(n+' of {n_terms} match'+(n===1?'':'s')):'';}}")
    w("     });")
    w("   });")
    w("   </script>")
    w("")

    for lvl in LEVEL_ORDER:
        themes_in = [k for k in active_themes if THEMES[k][2] == lvl]
        if not themes_in:
            continue
        # legacy hub anchors that point into this level (resolve to the section)
        for lbl, tk in COMPAT.items():
            if THEMES.get(tk, ("", "", ""))[2] == lvl:
                w(f".. _{lbl}:")
        head = LEVEL_TAB[lvl]
        w("")
        w(head)
        w("-" * (len(head) + 2))
        w("")
        for k in themes_in:
            emoji, disp, _, blurb = THEMES[k]
            terms = by_theme[k]
            # stable anchor kept from the card era (referenced externally,
            # e.g. by the bayesian_data_analysis hub) - do not rename.
            w(f".. _term-theme-{k}:")
            w("")
            w(f".. dropdown:: {emoji} {disp}  \u00b7  {len(terms)} terms")
            w("   :animate: fade-in-slide-down")
            w("")
            w(f"   *{blurb}*")
            w("")
            w("   .. hlist::")
            w("      :columns: 2")
            w("")
            for s in terms:
                w(f"      * :doc:`{s} <{docname[s]}>`")
            w("")

    # ---- dictionary view: one A-Z master list (auto-sorted) ----------
    az_head = "\U0001F524 Every term, A\u2013Z"
    w(az_head)
    w("-" * (len(az_head) + 2))
    w("")
    w(".. dropdown:: Open the full alphabetical index")
    w("   :class-container: term-az")
    w("")
    w("   .. hlist::")
    w("      :columns: 2")
    w("")
    for s in sorted(inv_titles, key=str.casefold):
        w(f"      * :doc:`{s} <{docname[s]}>`")
    w("")

    w("")
    w(".. toctree::")
    w("   :hidden:")
    w("")
    for k in THEME_ORDER:
        for s in by_theme[k]:
            w(f"   {docname[s]}")
    w("")
    w(".. tags:: purpose: reference, topic: terminology")
    w("")
    OUT.write_text("\n".join(I), encoding="utf-8")

    n_pages = len(inv_titles)
    print(f"Wrote index.rst + {n_pages} term pages "
          f"({n_rich} with full content, {n_pages - n_rich} summary) "
          f"across {len(active_themes)} themes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
