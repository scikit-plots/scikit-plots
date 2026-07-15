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

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
          placeholder="&#128269;&nbsp; Type to filter 56 lessons &mdash; by title or keyword&hellip;"
          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;
                 background:transparent;color:inherit"/>
   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;
        min-height:1.2em;margin-top:0.35rem"></div>
   </div>
   <script>
   document.addEventListener('DOMContentLoaded',function(){
     var inp=document.getElementById('term-filter');if(!inp){return;}
     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));
     var az=document.querySelector('details.term-az');
     var items=[];
     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(
       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});
     var cnt=document.getElementById('term-filter-count');
     inp.addEventListener('input',function(){
       var q=inp.value.trim().toLowerCase();var n=0;
       dds.forEach(function(d){d.tHits=0;});
       items.forEach(function(it){
         var hit=!q||it.t.indexOf(q)!==-1;
         it.li.style.display=hit?'':'none';
         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});
       dds.forEach(function(d){
         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}
         else{d.style.display='';d.open=false;}});
       if(cnt){cnt.textContent=(q&&az)?(n+' of 56 match'+(n===1?'':'s')):'';}
     });
   });
   </script>

.. _dpa-stage-foundations:

.. dropdown:: 📋 Stage 1 — Foundations  ·  6 lessons
   :animate: fade-in-slide-down

   *Why we analyse data, the CRISP-DM process, big data, and how numbers are stored.*  ·  *beginner*

   * :doc:`01 · Why Do We Analyze Data? <01-why-do-we-analyze-data>` — The goals of data analysis — turning raw records into decisions, predictions, and understanding.
   * :doc:`02 · The Process of Data Analysis <02-the-process-of-data-analysis>` — The end-to-end workflow from question to insight, and where each technique fits.
   * :doc:`03 · CRISP-DM for Data Science <03-crisp-dm-for-data-science>` — The Cross-Industry Standard Process: six phases from business understanding to deployment.
   * :doc:`04 · Big Data: Definition, Characteristics, Evolution, and Business Impact <04-big-data-definition-characteristics-evolution-and-business-impact>` — The volume, velocity and variety of big data, and why it reshaped analytics.
   * :doc:`05 · The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>` — Inspecting types, ranges and missingness before any modelling begins.
   * :doc:`06 · IEEE 754 Floating-Point Standard <06-ieee-754-floating-point-standard>` — How computers store real numbers — precision, rounding, and the pitfalls that follow.

.. _dpa-stage-associations:

.. dropdown:: 🔗 Stage 2 — Associations & Correlation  ·  10 lessons
   :animate: fade-in-slide-down

   *Measuring relationships between variables: correlation, statistical tests and effect size.*  ·  *beginner*

   * :doc:`07 · Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>` — Finding relationships in data, illustrated with the Chicago taxi-trips dataset.
   * :doc:`08 · Taxi Trips – 2022 dataset from the City of Chicago open data portal <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>` — The open dataset used throughout — its fields, scale, and how to load it.
   * :doc:`09 · Objective Selection of the Bin Width for a Time Histogram <09-objective-selection-of-the-bin-width-for-a-time-histogram>` — Choosing histogram bin width by a principled rule rather than guesswork.
   * :doc:`10 · Measuring Associations in Data <10-measuring-associations-in-data>` — How to quantify whether and how strongly two variables move together.
   * :doc:`11 · Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>` — Covariance and correlation for two numeric variables, and what they capture.
   * :doc:`12 · Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>` — Three correlation measures — linear, rank, and concordance — computed in Python.
   * :doc:`13 · Karl Pearson <13-karl-pearson>` — The statistician behind the correlation coefficient and much of modern statistics.
   * :doc:`14 · Harald Cramér <14-harald-cramer>` — The mathematician whose V measures association between categorical variables.
   * :doc:`15 · What Are Statistical Tests? <15-what-are-statistical-tests>` — The logic of hypothesis testing: null hypotheses, p-values, and significance.
   * :doc:`16 · Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>` — An effect-size measure for how much a grouping explains a numeric variable's variance.

.. _dpa-stage-market_basket:

.. dropdown:: 🛒 Stage 3 — Market Basket & Association Rules  ·  6 lessons
   :animate: fade-in-slide-down

   *Mining frequent itemsets and association rules for cross-selling with Apriori.*  ·  *intermediate*

   * :doc:`17 · Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers>` — What transaction baskets reveal about customer behaviour and product affinity.
   * :doc:`18 · What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>` — Reading if-then rules over co-purchased items to guide merchandising.
   * :doc:`19 · How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>` — Support, confidence and lift, and how Apriori tames the combinatorial search.
   * :doc:`20 · Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>` — Finding frequent itemsets efficiently using the downward-closure property.
   * :doc:`21 · association_rules: Generating Association Rules from Frequent Itemsets (mlxtend) <21-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend>` — Turning frequent itemsets into ranked rules with mlxtend in Python.
   * :doc:`22 · Cross-Selling <22-cross-selling>` — Using discovered rules to recommend complementary products.

.. _dpa-stage-segmentation:

.. dropdown:: 🧩 Stage 4 — Sampling, Partitioning & Segmentation  ·  8 lessons
   :animate: fade-in-slide-down

   *Sampling and train / test partitioning, then grouping observations by clustering and RFM.*  ·  *intermediate*

   * :doc:`23 · Stratified Random Sampling <23-stratified-random-sampling>` — Sampling that preserves subgroup proportions for representative data.
   * :doc:`24 · Linear Congruential Random Number Generator (LCG) <24-linear-congruential-random-number-generator-lcg>` — A classic pseudo-random generator — how reproducible randomness is produced.
   * :doc:`25 · Partitioning Observations to Train Objective Models <25-partitioning-observations-to-train-objective-models>` — Splitting data into train and test sets to measure honest performance.
   * :doc:`26 · Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters>` — The idea of grouping records by similarity — unsupervised segmentation.
   * :doc:`27 · Clustering <27-clustering>` — Algorithms like k-means that partition observations into homogeneous groups.
   * :doc:`28 · Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>` — Three behavioural dimensions that summarise a customer's value.
   * :doc:`29 · RFM Analysis <29-rfm-analysis>` — Scoring and ranking customers on recency, frequency, and monetary value.
   * :doc:`30 · Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>` — Turning RFM scores into actionable customer segments.

.. _dpa-stage-regression:

.. dropdown:: 📈 Stage 5 — Regression  ·  7 lessons
   :animate: fade-in-slide-down

   *Linear and multiple regression, feature selection, and feature importance.*  ·  *intermediate*

   * :doc:`31 · Least Squares Regression <31-least-squares-regression>` — Fitting a line by minimising the sum of squared residuals.
   * :doc:`32 · Multiple Linear Regression <32-multiple-linear-regression>` — Predicting an outcome from several features with a linear model.
   * :doc:`33 · Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>` — Judging which predictors matter most from coefficients and standardisation.
   * :doc:`34 · Forward Selection: Definition and Core Idea <34-forward-selection-definition-and-core-idea>` — Building a model by adding the most useful feature one at a time.
   * :doc:`35 · Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression>` — Applying forward selection to a regression and reading the result.
   * :doc:`36 · Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>` — Adding and removing features by a criterion to search model space.
   * :doc:`37 · How Shapley Values Work <37-how-shapley-values-work>` — Fairly attributing a prediction to each feature using cooperative game theory.

.. _dpa-stage-classification:

.. dropdown:: 🎯 Stage 6 — Classification & Logistic Regression  ·  6 lessons
   :animate: fade-in-slide-down

   *Modelling binary outcomes with logistic regression, maximum likelihood, and model fit.*  ·  *advanced*

   * :doc:`38 · Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>` — Modelling a yes/no probability through the log-odds linear form.
   * :doc:`39 · Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>` — Choosing parameters that make the observed data most probable.
   * :doc:`40 · Assessing Model Fit in Logistic Regression <40-assessing-model-fit-in-logistic-regression>` — Predictive-power and pseudo-R² measures for a fitted logistic model.
   * :doc:`41 · Complete and Quasi-Complete Separation in Logistic Regression <41-complete-and-quasi-complete-separation-in-logistic-regression>` — When perfect class separation breaks maximum-likelihood estimation.
   * :doc:`42 · Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>` — Comparing nested logistic models via the deviance likelihood-ratio test.
   * :doc:`43 · Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention <43-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention>` — A worked case study fitting and reading a student-retention model.

.. _dpa-stage-trees:

.. dropdown:: 🌳 Stage 7 — Decision Trees  ·  6 lessons
   :animate: fade-in-slide-down

   *CART decision trees as piecewise models, the interactions they capture, and profiling clusters.*  ·  *advanced*

   * :doc:`44 · Motivation of Decision Trees: An Incremental Model of Decision-Making <44-motivation-of-decision-trees-an-incremental-model-of-decision-making>` — Why splitting data by questions yields an interpretable predictor.
   * :doc:`45 · The CART Algorithm <45-the-cart-algorithm>` — Classification and Regression Trees — recursive binary splitting by impurity.
   * :doc:`46 · Decision Trees as Piecewise Models and Their Predictive Structure <46-decision-trees-as-piecewise-models-and-their-predictive-structure>` — Reading a tree as a piecewise-constant model over feature regions.
   * :doc:`47 · How CART Decision Trees Model Interactions <47-how-cart-decision-trees-model-interactions>` — Why trees naturally capture feature interactions through nested splits.
   * :doc:`48 · Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>` — Describing clusters by training a tree to predict cluster membership.
   * :doc:`49 · Using Decision Trees to Explain Clustering Results <49-using-decision-trees-to-explain-clustering-results>` — Turning opaque clusters into readable rules with a surrogate tree.

.. _dpa-stage-evaluation:

.. dropdown:: 📊 Stage 8 — Model Evaluation  ·  7 lessons
   :animate: fade-in-slide-down

   *Judging models: prediction quality, classification metrics, ROC / AUC, lift, and residuals.*  ·  *advanced*

   * :doc:`50 · Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>` — The general question of how good a model is, and by which yardstick.
   * :doc:`51 · Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>` — The confusion matrix and the metrics derived from it for two classes.
   * :doc:`52 · Nominal Classification Models: Model State and Evaluation Metrics <52-nominal-classification-models-model-state-and-evaluation-metrics>` — Extending evaluation to multi-class, unordered outcomes.
   * :doc:`53 · Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>` — Choosing a decision threshold to balance precision, recall, and cost.
   * :doc:`54 · Identifying Outliers Using Residuals and Studentized Residuals <54-identifying-outliers-using-residuals-and-studentized-residuals>` — Spotting influential and anomalous points from scaled residuals.
   * :doc:`55 · AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>` — The ROC curve and the area under it as threshold-free performance measures.
   * :doc:`56 · Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value <56-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value>` — Ranking prospects by model score to target a campaign efficiently.

🔤 Every lesson, A–Z index
---------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`Apriori: Frequent Itemsets via the Apriori Algorithm <20-apriori-frequent-itemsets-via-the-apriori-algorithm>`
      * :doc:`Assessing Model Fit in Logistic Regression <40-assessing-model-fit-in-logistic-regression>`
      * :doc:`Assessing the Quality of Prediction Models <50-assessing-the-quality-of-prediction-models>`
      * :doc:`association_rules: Generating Association Rules from Frequent Itemsets (mlxtend) <21-association-rules-generating-association-rules-from-frequent-itemsets-mlxtend>`
      * :doc:`AUC–ROC Curve: Evaluating Classification Model Performance <55-auc-roc-curve-evaluating-classification-model-performance>`
      * :doc:`Big Data: Definition, Characteristics, Evolution, and Business Impact <04-big-data-definition-characteristics-evolution-and-business-impact>`
      * :doc:`Binary Classification Model Evaluation and Threshold Optimization <53-binary-classification-model-evaluation-and-threshold-optimization>`
      * :doc:`Binary Classification Models – Conceptual Framework and Evaluation Metrics <51-binary-classification-models-conceptual-framework-and-evaluation-metrics>`
      * :doc:`Cluster Profiling Using Decision Trees <48-cluster-profiling-using-decision-trees>`
      * :doc:`Clustering <27-clustering>`
      * :doc:`Complete and Quasi-Complete Separation in Logistic Regression <41-complete-and-quasi-complete-separation-in-logistic-regression>`
      * :doc:`Correlation Coefficients in Python (Pearson, Spearman, Kendall) <12-correlation-coefficients-in-python-pearson-spearman-kendall>`
      * :doc:`Creating Segments of Observations for Business Reasons (RFM) <30-creating-segments-of-observations-for-business-reasons-rfm>`
      * :doc:`CRISP-DM for Data Science <03-crisp-dm-for-data-science>`
      * :doc:`Cross-Selling <22-cross-selling>`
      * :doc:`Decision Trees as Piecewise Models and Their Predictive Structure <46-decision-trees-as-piecewise-models-and-their-predictive-structure>`
      * :doc:`Discovering Associations Through Data: From Everyday Patterns to Chicago Taxi Trips (September 2022) <07-discovering-associations-through-data-from-everyday-patterns-to-chicago-taxi-trips-september-2022>`
      * :doc:`Eta Squared (η²): Effect Size in ANOVA <16-eta-squared-2-effect-size-in-anova>`
      * :doc:`Feature Importance in Linear Regression <33-feature-importance-in-linear-regression>`
      * :doc:`Forward Selection and Model Interpretation in Linear Regression <35-forward-selection-and-model-interpretation-in-linear-regression>`
      * :doc:`Forward Selection with Nested Models and Deviance Tests <42-forward-selection-with-nested-models-and-deviance-tests>`
      * :doc:`Forward Selection: Definition and Core Idea <34-forward-selection-definition-and-core-idea>`
      * :doc:`Harald Cramér <14-harald-cramer>`
      * :doc:`How Association Rules Are Discovered: Concepts, Scale, Measures, and the Apriori Approach <19-how-association-rules-are-discovered-concepts-scale-measures-and-the-apriori-approach>`
      * :doc:`How CART Decision Trees Model Interactions <47-how-cart-decision-trees-model-interactions>`
      * :doc:`How Shapley Values Work <37-how-shapley-values-work>`
      * :doc:`Identifying Outliers Using Residuals and Studentized Residuals <54-identifying-outliers-using-residuals-and-studentized-residuals>`
      * :doc:`IEEE 754 Floating-Point Standard <06-ieee-754-floating-point-standard>`
      * :doc:`Interpreting and Assessing a Forward-Selection Logistic Regression Model for College Student Retention <43-interpreting-and-assessing-a-forward-selection-logistic-regression-model-for-college-student-retention>`
      * :doc:`Karl Pearson <13-karl-pearson>`
      * :doc:`Least Squares Regression <31-least-squares-regression>`
      * :doc:`Lift Analysis for Direct Mail Campaigns: Concept, Process, and Business Value <56-lift-analysis-for-direct-mail-campaigns-concept-process-and-business-value>`
      * :doc:`Linear Congruential Random Number Generator (LCG) <24-linear-congruential-random-number-generator-lcg>`
      * :doc:`Logistic Regression: Modeling Binary Outcomes via Odds and Log-Odds <38-logistic-regression-modeling-binary-outcomes-via-odds-and-log-odds>`
      * :doc:`Maximum Likelihood (MLE): Fitting a Distribution to Observed Data <39-maximum-likelihood-mle-fitting-a-distribution-to-observed-data>`
      * :doc:`Measuring Associations Between Two Continuous Variables <11-measuring-associations-between-two-continuous-variables>`
      * :doc:`Measuring Associations in Data <10-measuring-associations-in-data>`
      * :doc:`Motivation of Decision Trees: An Incremental Model of Decision-Making <44-motivation-of-decision-trees-an-incremental-model-of-decision-making>`
      * :doc:`Multiple Linear Regression <32-multiple-linear-regression>`
      * :doc:`Nominal Classification Models: Model State and Evaluation Metrics <52-nominal-classification-models-model-state-and-evaluation-metrics>`
      * :doc:`Objective Selection of the Bin Width for a Time Histogram <09-objective-selection-of-the-bin-width-for-a-time-histogram>`
      * :doc:`Partitioning Observations to Train Objective Models <25-partitioning-observations-to-train-objective-models>`
      * :doc:`Putting Similar Observations into Clusters <26-putting-similar-observations-into-clusters>`
      * :doc:`Recency, Frequency, and Monetary Value (RFM) <28-recency-frequency-and-monetary-value-rfm>`
      * :doc:`RFM Analysis <29-rfm-analysis>`
      * :doc:`Stratified Random Sampling <23-stratified-random-sampling>`
      * :doc:`Taxi Trips – 2022 dataset from the City of Chicago open data portal <08-taxi-trips-2022-dataset-from-the-city-of-chicago-open-data-portal>`
      * :doc:`The CART Algorithm <45-the-cart-algorithm>`
      * :doc:`The First Step in Knowing Your Data <05-the-first-step-in-knowing-your-data>`
      * :doc:`The Process of Data Analysis <02-the-process-of-data-analysis>`
      * :doc:`Understanding Forward and Backward Stepwise Regression <36-understanding-forward-and-backward-stepwise-regression>`
      * :doc:`Understanding Market Baskets and Ideal Customers <17-understanding-market-baskets-and-ideal-customers>`
      * :doc:`Using Decision Trees to Explain Clustering Results <49-using-decision-trees-to-explain-clustering-results>`
      * :doc:`What Are Statistical Tests? <15-what-are-statistical-tests>`
      * :doc:`What Can Association Rules Tell Us? <18-what-can-association-rules-tell-us>`
      * :doc:`Why Do We Analyze Data? <01-why-do-we-analyze-data>`

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

.. tags:: purpose: reference, topic: data analysis, topic: data preparation
