:html_theme.sidebar_secondary.remove:

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _data-preparation-and-analysis-index:

:raw-html:`<div style="text-align:center"><strong>` 📊 Data Preparation & Analysis
|br| Building, scoring and trusting predictive models
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Data Preparation & Analysis
======================================================================

This course covers the **applied predictive-modelling workflow** end to end — framing a
prediction problem, exploring and preparing data, fitting models, and, most importantly for
**scikit-plots**, *evaluating* them with the right chart for the right question. It is the
practical companion to the :ref:`Terminology reference <terminology-index>`: terminology
defines the metrics, this course shows the workflow that produces and reads them — as an
ordered, self-contained sequence of 56 lessons across 8 stages.

Read it at any depth:

* **newcomers** — the intuition behind analysis, associations and model evaluation;
* **practitioners** — choosing between regression, trees, ROC, lift and threshold tuning;
* **reviewers** — diagnostics (residuals, outliers, separation) before shipping a model.

.. warning::

   Report performance on **held-out test data**, never the data a model was fit on. The
   sampling-and-partitioning stage exists precisely so the numbers you quote are honest.

.. note::

   Follow the lessons in order with **Next ▶**, or jump in by stage below. Snippets use
   real ``scikitplot`` / ``scikit-learn`` / ``mlxtend`` calls, and the evaluation charts
   (ROC, lift, calibration, residuals) are scikit-plots' specialty. This course pairs with
   the :ref:`Terminology reference <terminology-index>` (which defines every metric used).

======================================================================

📋 Stage 1 — Foundations
------------------------------------------------------------------------

*Why we analyse data, the CRISP-DM process, big data, and how numbers are stored.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 01 · Why Do We Analyze Data?
      :link: dpa-why-do-we-analyze-data
      :link-type: ref

      The goals of data analysis — turning raw records into decisions, predictions, and understanding.

   .. grid-item-card:: 02 · The Process of Data Analysis
      :link: dpa-the-process-of-data-analysis
      :link-type: ref

      The end-to-end workflow from question to insight, and where each technique fits.

   .. grid-item-card:: 03 · CRISP-DM for Data Science
      :link: dpa-crisp-dm-for-data-science
      :link-type: ref

      The Cross-Industry Standard Process: six phases from business understanding to deployment.

   .. grid-item-card:: 04 · Big Data: Definition, Characteristics, Evolution, and Business Impact
      :link: dpa-big-data-definition-characteristics-evolution-and-business-impact
      :link-type: ref

      The volume, velocity and variety of big data, and why it reshaped analytics.

   .. grid-item-card:: 05 · The First Step in Knowing Your Data
      :link: dpa-the-first-step-in-knowing-your-data
      :link-type: ref

      Inspecting types, ranges and missingness before any modelling begins.

   .. grid-item-card:: 06 · IEEE 754 Floating-Point Standard
      :link: dpa-ieee-754-floating-point-standard
      :link-type: ref

      How computers store real numbers — precision, rounding, and the pitfalls that follow.

🔗 Stage 2 — Associations & Correlation
------------------------------------------------------------------------

*Measuring relationships between variables: correlation, statistical tests and effect size.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 07 · Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022)
      :link: dpa-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022
      :link-type: ref

      Finding relationships in data, illustrated with the Chicago taxi-trips dataset.

   .. grid-item-card:: 08 · Taxi Trips – 2022 dataset from the City of Chicago open data portal
      :link: dpa-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal
      :link-type: ref

      The open dataset used throughout — its fields, scale, and how to load it.

   .. grid-item-card:: 09 · Objective Selection of the Bin Width for a Time Histogram
      :link: dpa-objective-selection-of-the-bin-width-for-a-time-histogram
      :link-type: ref

      Choosing histogram bin width by a principled rule rather than guesswork.

   .. grid-item-card:: 10 · Measuring Associations in Data
      :link: dpa-measuring-associations-in-data
      :link-type: ref

      How to quantify whether and how strongly two variables move together.

   .. grid-item-card:: 11 · Measuring Associations Between Two Continuous Variables
      :link: dpa-measuring-associations-between-two-continuous-variables
      :link-type: ref

      Covariance and correlation for two numeric variables, and what they capture.

   .. grid-item-card:: 12 · Correlation Coefficients in Python (Pearson, Spearman, Kendall)
      :link: dpa-correlation-coefficients-in-python-pearson-spearman-kendall
      :link-type: ref

      Three correlation measures — linear, rank, and concordance — computed in Python.

   .. grid-item-card:: 13 · Karl Pearson
      :link: dpa-karl-pearson
      :link-type: ref

      The statistician behind the correlation coefficient and much of modern statistics.

   .. grid-item-card:: 14 · Harald Cramér
      :link: dpa-harald-cramer
      :link-type: ref

      The mathematician whose V measures association between categorical variables.

   .. grid-item-card:: 15 · What Are Statistical Tests?
      :link: dpa-what-are-statistical-tests
      :link-type: ref

      The logic of hypothesis testing: null hypotheses, p-values, and significance.

   .. grid-item-card:: 16 · Eta Squared (η²): Effect Size in ANOVA
      :link: dpa-eta-squared-2-effect-size-in-anova
      :link-type: ref

      An effect-size measure for how much a grouping explains a numeric variable's variance.

🛒 Stage 3 — Market Basket & Association Rules
------------------------------------------------------------------------

*Mining frequent itemsets and association rules for cross-selling with Apriori.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 17 · Understanding Market Baskets and Ideal Customers
      :link: dpa-understanding-market-baskets-and-ideal-customers
      :link-type: ref

      What transaction baskets reveal about customer behaviour and product affinity.

   .. grid-item-card:: 18 · What Can Association Rules Tell Us?
      :link: dpa-what-can-association-rules-tell-us
      :link-type: ref

      Reading if-then rules over co-purchased items to guide merchandising.

   .. grid-item-card:: 19 · How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach
      :link: dpa-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach
      :link-type: ref

      Support, confidence and lift, and how Apriori tames the combinatorial search.

   .. grid-item-card:: 20 · Apriori: Frequent Itemsets via the Apriori Algorithm
      :link: dpa-apriori-frequent-itemsets-via-the-apriori-algorithm
      :link-type: ref

      Finding frequent itemsets efficiently using the downward-closure property.

   .. grid-item-card:: 21 · association_rules: Generating Association Rules from Frequent Itemsets (mlxtend)
      :link: dpa-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend
      :link-type: ref

      Turning frequent itemsets into ranked rules with mlxtend in Python.

   .. grid-item-card:: 22 · Cross-Selling
      :link: dpa-cross-selling
      :link-type: ref

      Using discovered rules to recommend complementary products.

🧩 Stage 4 — Sampling, Partitioning & Segmentation
------------------------------------------------------------------------

*Sampling and train / test partitioning, then grouping observations by clustering and RFM.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 23 · Stratified Random Sampling
      :link: dpa-stratified-random-sampling
      :link-type: ref

      Sampling that preserves subgroup proportions for representative data.

   .. grid-item-card:: 24 · Linear Congruential Random Number Generator (LCG)
      :link: dpa-linear-congruential-random-number-generator-lcg
      :link-type: ref

      A classic pseudo-random generator — how reproducible randomness is produced.

   .. grid-item-card:: 25 · Partitioning Observations to Train Objective Models
      :link: dpa-partitioning-observations-to-train-objective-models
      :link-type: ref

      Splitting data into train and test sets to measure honest performance.

   .. grid-item-card:: 26 · Putting Similar Observations into Clusters
      :link: dpa-putting-similar-observations-into-clusters
      :link-type: ref

      The idea of grouping records by similarity — unsupervised segmentation.

   .. grid-item-card:: 27 · Clustering
      :link: dpa-clustering
      :link-type: ref

      Algorithms like k-means that partition observations into homogeneous groups.

   .. grid-item-card:: 28 · Recency, Frequency, and Monetary Value (RFM)
      :link: dpa-recency-frequency-and-monetary-value-rfm
      :link-type: ref

      Three behavioural dimensions that summarise a customer's value.

   .. grid-item-card:: 29 · RFM Analysis
      :link: dpa-rfm-analysis
      :link-type: ref

      Scoring and ranking customers on recency, frequency, and monetary value.

   .. grid-item-card:: 30 · Creating Segments of Observations for Business Reasons (RFM)
      :link: dpa-creating-segments-of-observations-for-business-reasons-rfm
      :link-type: ref

      Turning RFM scores into actionable customer segments.

📈 Stage 5 — Regression
------------------------------------------------------------------------

*Linear and multiple regression, feature selection, and feature importance.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 31 · Least Squares Regression
      :link: dpa-least-squares-regression
      :link-type: ref

      Fitting a line by minimising the sum of squared residuals.

   .. grid-item-card:: 32 · Multiple Linear Regression
      :link: dpa-multiple-linear-regression
      :link-type: ref

      Predicting an outcome from several features with a linear model.

   .. grid-item-card:: 33 · Feature Importance in Linear Regression
      :link: dpa-feature-importance-in-linear-regression
      :link-type: ref

      Judging which predictors matter most from coefficients and standardisation.

   .. grid-item-card:: 34 · Forward Selection: Definition and Core Idea
      :link: dpa-forward-selection-definition-and-core-idea
      :link-type: ref

      Building a model by adding the most useful feature one at a time.

   .. grid-item-card:: 35 · Forward Selection and Model Interpretation in Linear Regression
      :link: dpa-forward-selection-and-model-interpretation-in-linear-regression
      :link-type: ref

      Applying forward selection to a regression and reading the result.

   .. grid-item-card:: 36 · Understanding Forward and Backward Stepwise Regression
      :link: dpa-understanding-forward-and-backward-stepwise-regression
      :link-type: ref

      Adding and removing features by a criterion to search model space.

   .. grid-item-card:: 37 · How Shapley Values Work
      :link: dpa-how-shapley-values-work
      :link-type: ref

      Fairly attributing a prediction to each feature using cooperative game theory.

🎯 Stage 6 — Classification & Logistic Regression
------------------------------------------------------------------------

*Modelling binary outcomes with logistic regression, maximum likelihood, and model fit.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 38 · Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds
      :link: dpa-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds
      :link-type: ref

      Modelling a yes/no probability through the log-odds linear form.

   .. grid-item-card:: 39 · Maximum Likelihood (MLE): Fitting a Distribution to Observed Data
      :link: dpa-maximum-likelihood-mle-fitting-a-distribution-to-observed-data
      :link-type: ref

      Choosing parameters that make the observed data most probable.

   .. grid-item-card:: 40 · Assessing Model Fit in Logistic Regression
      :link: dpa-assessing-model-fit-in-logistic-regression
      :link-type: ref

      Predictive-power and pseudo-R² measures for a fitted logistic model.

   .. grid-item-card:: 41 · Complete and Quasi-Complete Separation in Logistic Regression
      :link: dpa-complete-and-quasi-complete-separation-in-logistic-regression
      :link-type: ref

      When perfect class separation breaks maximum-likelihood estimation.

   .. grid-item-card:: 42 · Forward Selection with Nested Models and Deviance Tests
      :link: dpa-forward-selection-with-nested-models-and-deviance-tests
      :link-type: ref

      Comparing nested logistic models via the deviance likelihood-ratio test.

   .. grid-item-card:: 43 · Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention
      :link: dpa-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention
      :link-type: ref

      A worked case study fitting and reading a student-retention model.

🌳 Stage 7 — Decision Trees
------------------------------------------------------------------------

*CART decision trees as piecewise models, the interactions they capture, and profiling clusters.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 44 · Motivation of Decision Trees: An Incremental Model of Decision-Making
      :link: dpa-motivation-of-decision-trees-an-incremental-model-of-decision-making
      :link-type: ref

      Why splitting data by questions yields an interpretable predictor.

   .. grid-item-card:: 45 · The CART Algorithm
      :link: dpa-the-cart-algorithm
      :link-type: ref

      Classification and Regression Trees — recursive binary splitting by impurity.

   .. grid-item-card:: 46 · Decision Trees as Piecewise Models and Their Predictive Structure
      :link: dpa-decision-trees-as-piecewise-models-and-their-predictive-structure
      :link-type: ref

      Reading a tree as a piecewise-constant model over feature regions.

   .. grid-item-card:: 47 · How CART Decision Trees Model Interactions
      :link: dpa-how-cart-decision-trees-model-interactions
      :link-type: ref

      Why trees naturally capture feature interactions through nested splits.

   .. grid-item-card:: 48 · Cluster Profiling Using Decision Trees
      :link: dpa-cluster-profiling-using-decision-trees
      :link-type: ref

      Describing clusters by training a tree to predict cluster membership.

   .. grid-item-card:: 49 · Using Decision Trees to Explain Clustering Results
      :link: dpa-using-decision-trees-to-explain-clustering-results
      :link-type: ref

      Turning opaque clusters into readable rules with a surrogate tree.

📊 Stage 8 — Model Evaluation
------------------------------------------------------------------------

*Judging models: prediction quality, classification metrics, ROC / AUC, lift, and residuals.*

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 50 · Assessing the Quality of Prediction Models
      :link: dpa-assessing-the-quality-of-prediction-models
      :link-type: ref

      The general question of how good a model is, and by which yardstick.

   .. grid-item-card:: 51 · Binary Classification Models – Conceptual Framework and Evaluation Metrics
      :link: dpa-binary-classification-models-conceptual-framework-and-evaluation-metrics
      :link-type: ref

      The confusion matrix and the metrics derived from it for two classes.

   .. grid-item-card:: 52 · Nominal Classification Models: Model State and Evaluation Metrics
      :link: dpa-nominal-classification-models-model-state-and-evaluation-metrics
      :link-type: ref

      Extending evaluation to multi-class, unordered outcomes.

   .. grid-item-card:: 53 · Binary Classification Model Evaluation and Threshold Optimization
      :link: dpa-binary-classification-model-evaluation-and-threshold-optimization
      :link-type: ref

      Choosing a decision threshold to balance precision, recall, and cost.

   .. grid-item-card:: 54 · Identifying Outliers Using Residuals and Studentized Residuals
      :link: dpa-identifying-outliers-using-residuals-and-studentized-residuals
      :link-type: ref

      Spotting influential and anomalous points from scaled residuals.

   .. grid-item-card:: 55 · AUC–ROC Curve: Evaluating Classification Model Performance
      :link: dpa-auc-roc-curve-evaluating-classification-model-performance
      :link-type: ref

      The ROC curve and the area under it as threshold-free performance measures.

   .. grid-item-card:: 56 · Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value
      :link: dpa-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value
      :link-type: ref

      Ranking prospects by model score to target a campaign efficiently.

.. toctree::
   :hidden:
   :maxdepth: 1

   01-why-do-we-analyze-data
   02-the-process-of-data-analysis
   03-crisp-dm-for-data-science
   04-big-data-definition-characteristics-evolution-and-business-impact
   05-the-first-step-in-knowing-your-data
   06-ieee-754-floating-point-standard
   07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022
   08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal
   09-objective-selection-of-the-bin-width-for-a-time-histogram
   10-measuring-associations-in-data
   11-measuring-associations-between-two-continuous-variables
   12-correlation-coefficients-in-python-pearson-spearman-kendall
   13-karl-pearson
   14-harald-cramer
   15-what-are-statistical-tests
   16-eta-squared-2-effect-size-in-anova
   17-understanding-market-baskets-and-ideal-customers
   18-what-can-association-rules-tell-us
   19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach
   20-apriori-frequent-itemsets-via-the-apriori-algorithm
   21-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend
   22-cross-selling
   23-stratified-random-sampling
   24-linear-congruential-random-number-generator-lcg
   25-partitioning-observations-to-train-objective-models
   26-putting-similar-observations-into-clusters
   27-clustering
   28-recency-frequency-and-monetary-value-rfm
   29-rfm-analysis
   30-creating-segments-of-observations-for-business-reasons-rfm
   31-least-squares-regression
   32-multiple-linear-regression
   33-feature-importance-in-linear-regression
   34-forward-selection-definition-and-core-idea
   35-forward-selection-and-model-interpretation-in-linear-regression
   36-understanding-forward-and-backward-stepwise-regression
   37-how-shapley-values-work
   38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds
   39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data
   40-assessing-model-fit-in-logistic-regression
   41-complete-and-quasi-complete-separation-in-logistic-regression
   42-forward-selection-with-nested-models-and-deviance-tests
   43-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention
   44-motivation-of-decision-trees-an-incremental-model-of-decision-making
   45-the-cart-algorithm
   46-decision-trees-as-piecewise-models-and-their-predictive-structure
   47-how-cart-decision-trees-model-interactions
   48-cluster-profiling-using-decision-trees
   49-using-decision-trees-to-explain-clustering-results
   50-assessing-the-quality-of-prediction-models
   51-binary-classification-models-conceptual-framework-and-evaluation-metrics
   52-nominal-classification-models-model-state-and-evaluation-metrics
   53-binary-classification-model-evaluation-and-threshold-optimization
   54-identifying-outliers-using-residuals-and-studentized-residuals
   55-auc-roc-curve-evaluating-classification-model-performance
   56-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value

.. tags:: purpose: reference, topic: data preparation
