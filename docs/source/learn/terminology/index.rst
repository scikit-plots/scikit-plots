:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand. Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _terminology-index:

:raw-html:`<div align="center" style="text-align:center"><strong>` 📖 Terminology
|br| Glossary of Artificial Intelligence and Machine Learning Terms
|br| |full_version| - |today|
:raw-html:`</strong></div>`

=============
Terminology
=============

A working glossary of **431 terms** spanning statistics, machine learning, forecasting and MLOps. Every term has **its own page** with a self-contained explanation — no need to leave to read the full context. Entries are grouped into **26 themes** across three depth levels. **Type in the filter box** for instant lookup by name or keyword, expand a theme to browse, or open the A–Z index at the bottom.

.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
          placeholder="&#128269;&nbsp; Type to filter 431 terms &mdash; by name or keyword&hellip;"
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
       if(cnt){cnt.textContent=(q&&az)?(n+' of 431 match'+(n===1?'':'s')):'';}
     });
   });
   </script>

.. _terminology-statistics:

🟢 Foundations
---------------

.. _term-theme-probstats:

.. dropdown:: 🎲 Probability & Statistics Foundations  ·  28 terms
   :animate: fade-in-slide-down

   *The vocabulary of estimates, spread and uncertainty that everything else builds on.*

   .. hlist::
      :columns: 2

      * :doc:`Beta Distribution <099-beta-distribution>`
      * :doc:`Confidence Level <285-confidence-level>`
      * :doc:`Correlation <305-correlation>`
      * :doc:`Critical Value <087-critical-value>`
      * :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>`
      * :doc:`Frequentist <059-frequentist>`
      * :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>`
      * :doc:`Likelihood <304-likelihood>`
      * :doc:`Margin of Error (MoE) <086-margin-of-error-moe>`
      * :doc:`Mean <316-mean>`
      * :doc:`Median <315-median>`
      * :doc:`Normal Distribution <238-normal-distribution>`
      * :doc:`Outlier <307-outlier>`
      * :doc:`Population Proportion <199-population-proportion>`
      * :doc:`Probability <025-probability>`
      * :doc:`Probability Density <237-probability-density>`
      * :doc:`Probability Distribution <240-probability-distribution>`
      * :doc:`Probability Mass <239-probability-mass>`
      * :doc:`Proportion <091-proportion>`
      * :doc:`Regression Coefficient <090-regression-coefficient>`
      * :doc:`Sample Mean <089-sample-mean>`
      * :doc:`Sample Standard Deviation <088-sample-standard-deviation>`
      * :doc:`Standard Error (SE) <084-standard-error-se>`
      * :doc:`Statistical Significance <096-statistical-significance>`
      * :doc:`Statistically Significant <125-statistically-significant>`
      * :doc:`True Mean (Population Mean) <085-true-mean-population-mean>`
      * :doc:`True Population Parameter <092-true-population-parameter>`
      * :doc:`Z-Score <097-z-score>`

.. _term-theme-inference:

.. dropdown:: 🧮 Statistical Inference & Power  ·  23 terms
   :animate: fade-in-slide-down

   *Hypothesis testing, error rates and how big a study needs to be.*

   .. hlist::
      :columns: 2

      * :doc:`A Priori Power Analysis <095-a-priori-power-analysis>`
      * :doc:`Chi-square (χ²) Test <324-chi-square-2-test>`
      * :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>`
      * :doc:`Compromise Power Analysis <093-compromise-power-analysis>`
      * :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>`
      * :doc:`Effect Size (δ) <106-effect-size>`
      * :doc:`Hypothesis Testing <107-hypothesis-testing>`
      * :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>`
      * :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>`
      * :doc:`P-Value (probability value) <118-p-value-probability-value>`
      * :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>`
      * :doc:`Power (1 – β) <104-power-1>`
      * :doc:`Power Analysis <378-power-analysis>`
      * :doc:`Sample size <103-sample-size>`
      * :doc:`Significance Level (α) <105-significance-level>`
      * :doc:`Statistical Power <348-statistical-power>`
      * :doc:`Statistical Tests <328-statistical-tests>`
      * :doc:`T-Test <120-t-test>`
      * :doc:`Trivial Effects <102-trivial-effects>`
      * :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>`
      * :doc:`Type I Error <080-type-i-error>`
      * :doc:`Wilson Score Interval <357-wilson-score-interval>`
      * :doc:`Z-Test <119-z-test>`

.. _term-theme-concepts:

.. dropdown:: 💡 AI & ML Concepts  ·  14 terms
   :animate: fade-in-slide-down

   *Umbrella terms for the field and its major model families.*

   .. hlist::
      :columns: 2

      * :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>`
      * :doc:`Classification Models <294-classification-models>`
      * :doc:`Computer Vision (CV) <321-computer-vision-cv>`
      * :doc:`Decision Trees <340-decision-trees>`
      * :doc:`Linear Models <341-linear-models>`
      * :doc:`LLMs (Large Language Models) <158-llms-large-language-models>`
      * :doc:`Logistic Regression <292-logistic-regression>`
      * :doc:`Machine Learning (ML) <144-machine-learning-ml>`
      * :doc:`Medical AI <145-medical-ai>`
      * :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>`
      * :doc:`Neural Networks <287-neural-networks>`
      * :doc:`Regression Models <309-regression-models>`
      * :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>`
      * :doc:`Target Variable <236-target-variable>`

.. _terminology-averaging:
.. _terminology-classification-types:
.. _terminology-multiclass-auroc:
.. _terminology-precision-recall:
.. _terminology-roc-auroc:
.. _terminology-bootstrap:

🔵 Applied
-----------

.. _term-theme-imbalance:

.. dropdown:: 🧪 Imbalanced Learning & Resampling  ·  9 terms
   :animate: fade-in-slide-down

   *Techniques for training useful models when one class is rare.*

   .. hlist::
      :columns: 2

      * :doc:`Class Weighting <002-class-weighting>`
      * :doc:`Cluster-based undersampling <007-cluster-based-undersampling>`
      * :doc:`Downsampling <368-downsampling>`
      * :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>`
      * :doc:`Oversampling <004-oversampling>`
      * :doc:`Random Undersampling <008-random-undersampling>`
      * :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>`
      * :doc:`Subsampling <001-subsampling>`
      * :doc:`Upsampling <367-upsampling>`

.. _term-theme-metrics:

.. dropdown:: 📏 Classification & Averaging Metrics  ·  39 terms
   :animate: fade-in-slide-down

   *Precision, recall, F1 and AUROC — and how they are averaged across classes.*

   .. hlist::
      :columns: 2

      * :doc:`Accuracy <323-accuracy>`
      * :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>`
      * :doc:`Average Precision (AP) <366-average-precision-ap>`
      * :doc:`Binary Classification <293-binary-classification>`
      * :doc:`Classification Probability <231-classification-probability>`
      * :doc:`Discriminatory Power <185-discriminatory-power>`
      * :doc:`F1-score <363-f1-score>`
      * :doc:`Gini Coefficient <023-gini-coefficient>`
      * :doc:`Harmonic Mean <362-harmonic-mean>`
      * :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>`
      * :doc:`Macro AUC <314-macro-auc>`
      * :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`
      * :doc:`Macro Averaging <370-macro-averaging>`
      * :doc:`Macro F1 <019-macro-f1>`
      * :doc:`Macro Precision <021-macro-precision>`
      * :doc:`Macro Recall <020-macro-recall>`
      * :doc:`Micro AUC <313-micro-auc>`
      * :doc:`Micro AUROC <011-micro-auroc>`
      * :doc:`Micro Averaging <369-micro-averaging>`
      * :doc:`Micro F1 <013-micro-f1>`
      * :doc:`Micro Precision <016-micro-precision>`
      * :doc:`Micro Recall <015-micro-recall>`
      * :doc:`Model Score <364-model-score>`
      * :doc:`Multi-label Classification <012-multi-label-classification>`
      * :doc:`Multiclass AUROC <022-multiclass-auroc>`
      * :doc:`Multiclass Classification <311-multiclass-classification>`
      * :doc:`Multiclass Precision <359-multiclass-precision>`
      * :doc:`Multilabel Precision <360-multilabel-precision>`
      * :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>`
      * :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>`
      * :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>`
      * :doc:`Per-class Precision (sometimes called class-wise precision) <358-per-class-precision-sometimes-called-class-wise>`
      * :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>`
      * :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>`
      * :doc:`Recall <423-recall>`
      * :doc:`ROC Curve (Receiver Operating Characteristic) <277-roc-curve-receiver-operating-characteristic>`
      * :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>`
      * :doc:`Single-label Classification <014-single-label-classification>`
      * :doc:`Weighted Averaging <361-weighted-averaging>`

.. _term-theme-evaluation:

.. dropdown:: 🔬 Model Evaluation & Uncertainty  ·  21 terms
   :animate: fade-in-slide-down

   *Putting error bars and significance around a model's score.*

   .. hlist::
      :columns: 2

      * :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>`
      * :doc:`Baseline Heuristics <428-baseline-heuristics>`
      * :doc:`Bootstrap <365-bootstrap>`
      * :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`
      * :doc:`Coverage <411-coverage>`
      * :doc:`Cramér's V <180-cramer-s-v>`
      * :doc:`DeLong's Test <352-delong-s-test>`
      * :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>`
      * :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>`
      * :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>`
      * :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>`
      * :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>`
      * :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>`
      * :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`
      * :doc:`Relative accuracy <258-relative-accuracy>`
      * :doc:`RMSLE (Root Mean Squared Logarithmic Error) <407-rmsle-root-mean-squared-logarithmic-error>`
      * :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>`
      * :doc:`R² (R-squared) <259-r2-r-squared>`
      * :doc:`sMAPE (Symmetric Mean Absolute Percentage Error) <406-smape-symmetric-mean-absolute-percentage-error>`
      * :doc:`WAPE (Weighted Absolute Percentage Error) <422-wape-weighted-absolute-percentage-error>`
      * :doc:`WMAPE (Weighted Mean Absolute Percentage Error) <405-wmape-weighted-mean-absolute-percentage-error>`

.. _term-theme-abtest:

.. dropdown:: 🧫 A/B Testing & Experimentation  ·  16 terms
   :animate: fade-in-slide-down

   *Designing and reading controlled online experiments.*

   .. hlist::
      :columns: 2

      * :doc:`A/B Testing <380-a-b-testing>`
      * :doc:`A/B/n Test <114-a-b-n-test>`
      * :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>`
      * :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>`
      * :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`
      * :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>`
      * :doc:`Group Sequential Testing <079-group-sequential-testing>`
      * :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>`
      * :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`
      * :doc:`Optimizely <069-optimizely>`
      * :doc:`Risk of Peeking <116-risk-of-peeking>`
      * :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>`
      * :doc:`Stopping Rules <071-stopping-rules>`
      * :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`
      * :doc:`Treatment Effect <072-treatment-effect>`
      * :doc:`True Conversion Rate <083-true-conversion-rate>`

.. _term-theme-growth:

.. dropdown:: 💼 Business & Growth Analytics  ·  36 terms
   :animate: fade-in-slide-down

   *The unit-economics and customer metrics that experiments optimise.*

   .. hlist::
      :columns: 2

      * :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>`
      * :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>`
      * :doc:`Cannibalization <392-cannibalization>`
      * :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>`
      * :doc:`Churn <123-churn>`
      * :doc:`Cohort <183-cohort>`
      * :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>`
      * :doc:`Conversion Rate (CR) <299-conversion-rate-cr>`
      * :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>`
      * :doc:`Cross-Selling <031-cross-selling>`
      * :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>`
      * :doc:`Customer Lifetime <042-customer-lifetime>`
      * :doc:`Customer Segmentation <033-customer-segmentation>`
      * :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`
      * :doc:`FTEs <147-ftes>`
      * :doc:`Fully Loaded CAC (Customer Acquisition Cost) <044-fully-loaded-cac-customer-acquisition-cost>`
      * :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>`
      * :doc:`Gross Margin <043-gross-margin>`
      * :doc:`KYC <146-kyc>`
      * :doc:`Lagging Indicators <168-lagging-indicators>`
      * :doc:`Lead-Gen Software <049-lead-gen-software>`
      * :doc:`Leading Indicators <169-leading-indicators>`
      * :doc:`LTV (Customer Lifetime Value) <373-ltv-customer-lifetime-value>`
      * :doc:`LTV:CAC Ratio <037-ltv-cac-ratio>`
      * :doc:`Net LTV (sometimes called Contribution LTV) <038-net-ltv-sometimes-called-contribution-ltv>`
      * :doc:`OpEx <157-opex>`
      * :doc:`Organic CAC (Customer Acquisition Cost) <045-organic-cac-customer-acquisition-cost>`
      * :doc:`Paid CAC (Customer Acquisition Cost) <046-paid-cac-customer-acquisition-cost>`
      * :doc:`Predictive LTV (pLTV) <040-predictive-ltv-pltv>`
      * :doc:`Retention <124-retention>`
      * :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>`
      * :doc:`ROI (Return on Investment) <191-roi-return-on-investment>`
      * :doc:`SaaS (Software as a Service) <034-saas-software-as-a-service>`
      * :doc:`Session Length <121-session-length>`
      * :doc:`Upselling <032-upselling>`
      * :doc:`Valuation Metric <035-valuation-metric>`

.. _term-theme-validation:

.. dropdown:: 🧷 Validation & Cross-Validation  ·  12 terms
   :animate: fade-in-slide-down

   *Splitting strategies that estimate performance honestly, including for time series.*

   .. hlist::
      :columns: 2

      * :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>`
      * :doc:`Cross-Validation (CV) <136-cross-validation-cv>`
      * :doc:`Data Leakage <131-data-leakage>`
      * :doc:`Evaluation Set <355-evaluation-set>`
      * :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>`
      * :doc:`k-fold cross-validation <135-k-fold-cross-validation>`
      * :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>`
      * :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>`
      * :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>`
      * :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>`
      * :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>`
      * :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`

.. _term-theme-training:

.. dropdown:: 🏋️ Model Training & Optimization  ·  22 terms
   :animate: fade-in-slide-down

   *Fitting, tuning and compressing models effectively.*

   .. hlist::
      :columns: 2

      * :doc:`Active Learning <163-active-learning>`
      * :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>`
      * :doc:`Deep Ensembles <335-deep-ensembles>`
      * :doc:`Early Stopping <140-early-stopping>`
      * :doc:`Ensemble <154-ensemble>`
      * :doc:`Epochs <141-epochs>`
      * :doc:`FLOPs <156-flops>`
      * :doc:`Full Annotation <345-full-annotation>`
      * :doc:`Hyperparameter <142-hyperparameter>`
      * :doc:`Label Noise <354-label-noise>`
      * :doc:`Log-Odds <295-log-odds>`
      * :doc:`Logit Space <291-logit-space>`
      * :doc:`Logits <420-logits>`
      * :doc:`Loss Functions <289-loss-functions>`
      * :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`
      * :doc:`Model Weights <155-model-weights>`
      * :doc:`Quantization <343-quantization>`
      * :doc:`Sigmoid Function <297-sigmoid-function>`
      * :doc:`Softmax Function <296-softmax-function>`
      * :doc:`Squashing Function <298-squashing-function>`
      * :doc:`Underflow <290-underflow>`
      * :doc:`Weak Supervision <346-weak-supervision>`

.. _term-theme-platforms:

.. dropdown:: 🧰 ML Platforms & Tools  ·  7 terms
   :animate: fade-in-slide-down

   *Cloud services and APIs for building, serving and experimenting.*

   .. hlist::
      :columns: 2

      * :doc:`AWS SageMaker <148-aws-sagemaker>`
      * :doc:`Google Experiments <100-google-experiments>`
      * :doc:`Kaggle <273-kaggle>`
      * :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>`
      * :doc:`OpenAI API (ML API) <150-openai-api-ml-api>`
      * :doc:`TPU Clusters <347-tpu-clusters>`
      * :doc:`Vertex AI <149-vertex-ai>`

.. _term-theme-ops:

.. dropdown:: 📦 Operations & Supply Chain  ·  17 terms
   :animate: fade-in-slide-down

   *Inventory, demand and fulfillment metrics for operations.*

   .. hlist::
      :columns: 2

      * :doc:`Backorder Rate <218-backorder-rate>`
      * :doc:`Crew Overtime <398-crew-overtime>`
      * :doc:`Demand Forecasting <215-demand-forecasting>`
      * :doc:`Fill Rate <220-fill-rate>`
      * :doc:`Long Lead Times <210-long-lead-times>`
      * :doc:`Long-Tail Items <260-long-tail-items>`
      * :doc:`Lost Sales Value <219-lost-sales-value>`
      * :doc:`Overstock % <400-overstock>`
      * :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>`
      * :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>`
      * :doc:`Safety Stock <217-safety-stock>`
      * :doc:`SKU <212-sku>`
      * :doc:`Slow-Moving SKUs <211-slow-moving-skus>`
      * :doc:`Stockout Rate <221-stockout-rate>`
      * :doc:`Stockouts <401-stockouts>`
      * :doc:`Supplier Constraints <209-supplier-constraints>`
      * :doc:`Supplier Management <214-supplier-management>`

.. _term-theme-features:

.. dropdown:: 🧮 Data Preparation & Features  ·  4 terms
   :animate: fade-in-slide-down

   *Transforming, encoding and organising raw data into model-ready form.*

   .. hlist::
      :columns: 2

      * :doc:`Advanced Sorting in Spreadsheets <431-advanced-sorting-in-spreadsheets>`
      * :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>`
      * :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>`
      * :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>`

.. _terminology-calibration:
.. _terminology-signal-timeseries:

🔴 Advanced
------------

.. _term-theme-fairness:

.. dropdown:: ⚖️ Fairness & Calibration  ·  8 terms
   :animate: fade-in-slide-down

   *Group-level fairness criteria and what each one equalises.*

   .. hlist::
      :columns: 2

      * :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>`
      * :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>`
      * :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>`
      * :doc:`Fairness Guardrails <351-fairness-guardrails>`
      * :doc:`Fairness parity <372-fairness-parity>`
      * :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>`
      * :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`
      * :doc:`Selection Rate <390-selection-rate>`

.. _term-theme-bayes:

.. dropdown:: 🔁 Bayesian Inference  ·  16 terms
   :animate: fade-in-slide-down

   *Reasoning with priors, posteriors and the machinery that computes them.*

   .. hlist::
      :columns: 2

      * :doc:`Bayes' Theorem <066-bayes-theorem>`
      * :doc:`Bayesian Correction <164-bayesian-correction>`
      * :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>`
      * :doc:`Bayesian Inference. <375-bayesian-inference>`
      * :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>`
      * :doc:`Binomial Likelihood <060-binomial-likelihood>`
      * :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>`
      * :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>`
      * :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>`
      * :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>`
      * :doc:`Posterior <063-posterior>`
      * :doc:`Posterior belief <061-posterior-belief>`
      * :doc:`Posterior Probability <073-posterior-probability>`
      * :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`
      * :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`
      * :doc:`Variational Inference (VI) <056-variational-inference-vi>`

.. _term-theme-bandits:

.. dropdown:: 🎰 Sequential Methods & Bandits  ·  6 terms
   :animate: fade-in-slide-down

   *Adaptive decisions and tests that update as evidence arrives.*

   .. hlist::
      :columns: 2

      * :doc:`Bandit Algorithms <113-bandit-algorithms>`
      * :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>`
      * :doc:`Pocock Method <077-pocock-method>`
      * :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>`
      * :doc:`Sequential Settings <058-sequential-settings>`
      * :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`

.. _term-theme-signal:

.. dropdown:: 📈 Signal Processing & Time Series  ·  19 terms
   :animate: fade-in-slide-down

   *Working with ordered, time-indexed and filtered data.*

   .. hlist::
      :columns: 2

      * :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>`
      * :doc:`Bayesian Time Series <052-bayesian-time-series>`
      * :doc:`Forecast Error <250-forecast-error>`
      * :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>`
      * :doc:`Forecasting Competitions <251-forecasting-competitions>`
      * :doc:`Log-Space <257-log-space>`
      * :doc:`Low-pass Filtering <005-low-pass-filtering>`
      * :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>`
      * :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>`
      * :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>`
      * :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>`
      * :doc:`Seasonal Lag <247-seasonal-lag>`
      * :doc:`Seasonality <329-seasonality>`
      * :doc:`Signal Processing <009-signal-processing>`
      * :doc:`Simple Baseline Methods <248-simple-baseline-methods>`
      * :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>`
      * :doc:`Time Series <010-time-series>`
      * :doc:`Time Series Forecasting <256-time-series-forecasting>`
      * :doc:`Windows (in Time-Series) <170-windows-in-time-series>`

.. _term-theme-mlops:

.. dropdown:: ⚙️ MLOps, Serving & Monitoring  ·  24 terms
   :animate: fade-in-slide-down

   *Deploying, serving and keeping models healthy in production.*

   .. hlist::
      :columns: 2

      * :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>`
      * :doc:`Caching <342-caching>`
      * :doc:`Cloud Inference <153-cloud-inference>`
      * :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>`
      * :doc:`Compute budgets <383-compute-budgets>`
      * :doc:`Continuous Retraining <161-continuous-retraining>`
      * :doc:`Feature Values <188-feature-values>`
      * :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>`
      * :doc:`Inference Cost (Inference $) <385-inference-cost-inference>`
      * :doc:`Latency Guardrails <350-latency-guardrails>`
      * :doc:`Manual review minutes <384-manual-review-minutes>`
      * :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>`
      * :doc:`Model Stability <187-model-stability>`
      * :doc:`Monitoring Pipelines <162-monitoring-pipelines>`
      * :doc:`Ops Health Dashboard <206-ops-health-dashboard>`
      * :doc:`Re-scoring <137-re-scoring>`
      * :doc:`Recalibrate Thresholds <165-recalibrate-thresholds>`
      * :doc:`Recalibration <159-recalibration>`
      * :doc:`Reweighting <160-reweighting>`
      * :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>`
      * :doc:`SLA Breach Rate <207-sla-breach-rate>`
      * :doc:`SLA Breaches <399-sla-breaches>`
      * :doc:`SLI (Service Level Indicator) <190-sli-service-level-indicator>`
      * :doc:`SLOs (Service Level Objectives) <391-slos-service-level-objectives>`

.. _term-theme-drift:

.. dropdown:: 🌊 Distribution Shift & Drift  ·  20 terms
   :animate: fade-in-slide-down

   *Detecting and measuring when data or representations change.*

   .. hlist::
      :columns: 2

      * :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>`
      * :doc:`Categorical Drift <179-categorical-drift>`
      * :doc:`Categorical Explosions <182-categorical-explosions>`
      * :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>`
      * :doc:`Concept Drift <330-concept-drift>`
      * :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>`
      * :doc:`Data Drift <331-data-drift>`
      * :doc:`Dataset Shift <353-dataset-shift>`
      * :doc:`Drift Detection <138-drift-detection>`
      * :doc:`Drift Guardrails <349-drift-guardrails>`
      * :doc:`Energy Distance <176-energy-distance>`
      * :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>`
      * :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>`
      * :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`
      * :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`
      * :doc:`Macro Shifts <181-macro-shifts>`
      * :doc:`Maximum Mean Discrepancy (MMD) <177-maximum-mean-discrepancy-mmd>`
      * :doc:`Off-Distribution <184-off-distribution>`
      * :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`
      * :doc:`Representation Shift <174-representation-shift>`

.. _term-theme-repr:

.. dropdown:: 🧬 Representations & Embeddings  ·  4 terms
   :animate: fade-in-slide-down

   *Learned feature spaces and the encoders that produce them.*

   .. hlist::
      :columns: 2

      * :doc:`Autoencoder <171-autoencoder>`
      * :doc:`Embedding <173-embedding>`
      * :doc:`Embedding Similarity <320-embedding-similarity>`
      * :doc:`Frozen Encoder <172-frozen-encoder>`

.. _term-theme-ranking:

.. dropdown:: 🔎 Ranking & Interleaving  ·  9 terms
   :animate: fade-in-slide-down

   *Ordering items and comparing rankers with online evaluation.*

   .. hlist::
      :columns: 2

      * :doc:`Balanced Interleaving <111-balanced-interleaving>`
      * :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>`
      * :doc:`Interleaving Tests <379-interleaving-tests>`
      * :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>`
      * :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`
      * :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>`
      * :doc:`Ranking Algorithms <108-ranking-algorithms>`
      * :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>`
      * :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`

.. _term-theme-causal:

.. dropdown:: 🔗 Causal Inference & Uplift  ·  25 terms
   :animate: fade-in-slide-down

   *Estimating cause-and-effect and the incremental impact of actions.*

   .. hlist::
      :columns: 2

      * :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>`
      * :doc:`Causal Effect <306-causal-effect>`
      * :doc:`Causal Impact <112-causal-impact>`
      * :doc:`Causal Inference <117-causal-inference>`
      * :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>`
      * :doc:`Causal Trees <301-causal-trees>`
      * :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>`
      * :doc:`Cumulative Uplift <198-cumulative-uplift>`
      * :doc:`Incremental Conversions <394-incremental-conversions>`
      * :doc:`Incremental Gain <200-incremental-gain>`
      * :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>`
      * :doc:`Incremental Revenue <193-incremental-revenue>`
      * :doc:`Incremental Sales <195-incremental-sales>`
      * :doc:`Qini Coefficient <397-qini-coefficient>`
      * :doc:`Qini Curve <203-qini-curve>`
      * :doc:`Random Targeting Strategy <196-random-targeting-strategy>`
      * :doc:`Revenue net of treatment cost <393-revenue-net-of-treatment-cost>`
      * :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>`
      * :doc:`Treatment Cost <192-treatment-cost>`
      * :doc:`Uplift <424-uplift>`
      * :doc:`Uplift Curve <303-uplift-curve>`
      * :doc:`Uplift Models <205-uplift-models>`
      * :doc:`Uplift Random Forests <302-uplift-random-forests>`
      * :doc:`Uplift Score <204-uplift-score>`
      * :doc:`Uplift@k <395-uplift-k>`

.. _term-theme-risk:

.. dropdown:: 📉 Risk & Probabilistic Forecasting  ·  19 terms
   :animate: fade-in-slide-down

   *Quantifying uncertainty and tail risk in forecasts.*

   .. hlist::
      :columns: 2

      * :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>`
      * :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>`
      * :doc:`Deterministic forecasts <242-deterministic-forecasts>`
      * :doc:`Full Distribution <229-full-distribution>`
      * :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>`
      * :doc:`Point Forecasts <233-point-forecasts>`
      * :doc:`Predicting Percentiles <252-predicting-percentiles>`
      * :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>`
      * :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>`
      * :doc:`Probabilistic Scoring <228-probabilistic-scoring>`
      * :doc:`Probability Forecasts <235-probability-forecasts>`
      * :doc:`Quantile Forecasts <232-quantile-forecasts>`
      * :doc:`Quantile Level <255-quantile-level>`
      * :doc:`Quantile Regression <254-quantile-regression>`
      * :doc:`Return Distribution <225-return-distribution>`
      * :doc:`Risk Forecast <227-risk-forecast>`
      * :doc:`Risk-Based Decisions <286-risk-based-decisions>`
      * :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`
      * :doc:`Value-at-Risk (VaR) <226-value-at-risk-var>`

.. _term-theme-recsys:

.. dropdown:: 🎁 Recommender Systems  ·  14 terms
   :animate: fade-in-slide-down

   *Relevance, diversity and coverage in recommendation.*

   .. hlist::
      :columns: 2

      * :doc:`Catalog Coverage <268-catalog-coverage>`
      * :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>`
      * :doc:`Diminishing Utility <271-diminishing-utility>`
      * :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>`
      * :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>`
      * :doc:`Genre Overlap <263-genre-overlap>`
      * :doc:`Hit Rate (HR) <412-hit-rate-hr>`
      * :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>`
      * :doc:`Item Coverage <270-item-coverage>`
      * :doc:`Jaccard index <264-jaccard-index>`
      * :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>`
      * :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`
      * :doc:`Self-Information of Popularity <261-self-information-of-popularity>`
      * :doc:`User Coverage <269-user-coverage>`

.. _term-theme-calibration:

.. dropdown:: 🎯 Probability Calibration  ·  12 terms
   :animate: fade-in-slide-down

   *Making predicted probabilities trustworthy and measuring miscalibration.*

   .. hlist::
      :columns: 2

      * :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>`
      * :doc:`Brier Score <418-brier-score>`
      * :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>`
      * :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>`
      * :doc:`Isotonic Regression <281-isotonic-regression>`
      * :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>`
      * :doc:`Murphy's Decomposition <278-murphy-s-decomposition>`
      * :doc:`Overconfident <284-overconfident>`
      * :doc:`Platt Scaling <280-platt-scaling>`
      * :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>`
      * :doc:`Temperature Scaling <279-temperature-scaling>`
      * :doc:`Underconfident <283-underconfident>`

.. _term-theme-xai:

.. dropdown:: 🔍 Explainability & Governance  ·  7 terms
   :animate: fade-in-slide-down

   *Interpreting model decisions and meeting regulatory and ethical duties.*

   .. hlist::
      :columns: 2

      * :doc:`Basel III <333-basel-iii>`
      * :doc:`Counterfactual Explanations <336-counterfactual-explanations>`
      * :doc:`Fair Lending laws <332-fair-lending-laws>`
      * :doc:`High-Stakes Domains <334-high-stakes-domains>`
      * :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>`
      * :doc:`Post-hoc Explainability <339-post-hoc-explainability>`
      * :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`

🔤 Every term, A–Z index
-------------------------

.. dropdown:: 🔠 Open the full alphabetical index
   :class-container: term-az

   .. hlist::
      :columns: 2

      * :doc:`A Priori Power Analysis <095-a-priori-power-analysis>`
      * :doc:`A/B Testing <380-a-b-testing>`
      * :doc:`A/B/n Test <114-a-b-n-test>`
      * :doc:`Accuracy <323-accuracy>`
      * :doc:`Active Learning <163-active-learning>`
      * :doc:`Adaptive ECE (Expected Calibration Error with Adaptive Binning) <275-adaptive-ece-expected-calibration-error-with-ada>`
      * :doc:`Advanced Sorting in Spreadsheets <431-advanced-sorting-in-spreadsheets>`
      * :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>`
      * :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>`
      * :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>`
      * :doc:`Autoencoder <171-autoencoder>`
      * :doc:`AUUC (Area Under the Uplift Curve) <396-auuc-area-under-the-uplift-curve>`
      * :doc:`Average Absolute Error (AAE) <246-average-absolute-error-aae>`
      * :doc:`Average Precision (AP) <366-average-precision-ap>`
      * :doc:`AWS SageMaker <148-aws-sagemaker>`
      * :doc:`AWS SageMaker Endpoints <151-aws-sagemaker-endpoints>`
      * :doc:`Backorder Rate <218-backorder-rate>`
      * :doc:`Balanced Interleaving <111-balanced-interleaving>`
      * :doc:`Bandit Algorithms <113-bandit-algorithms>`
      * :doc:`Basel III <333-basel-iii>`
      * :doc:`Baseline Heuristics <428-baseline-heuristics>`
      * :doc:`Bayes' Theorem <066-bayes-theorem>`
      * :doc:`Bayesian Correction <164-bayesian-correction>`
      * :doc:`Bayesian Decision Theory (BDT) <051-bayesian-decision-theory-bdt>`
      * :doc:`Bayesian Inference. <375-bayesian-inference>`
      * :doc:`Bayesian Neural Networks (BNNs) <055-bayesian-neural-networks-bnns>`
      * :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>`
      * :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>`
      * :doc:`Bayesian Time Series <052-bayesian-time-series>`
      * :doc:`Beta Distribution <099-beta-distribution>`
      * :doc:`Binary Classification <293-binary-classification>`
      * :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>`
      * :doc:`Binomial Likelihood <060-binomial-likelihood>`
      * :doc:`Blended CAC (Customer Acquisition Cost) <048-blended-cac-customer-acquisition-cost>`
      * :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>`
      * :doc:`Bootstrap <365-bootstrap>`
      * :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`
      * :doc:`Brier Score <418-brier-score>`
      * :doc:`CAC (Customer Acquisition Cost) <374-cac-customer-acquisition-cost>`
      * :doc:`Caching <342-caching>`
      * :doc:`Calibration quality (Model Calibration) <419-calibration-quality-model-calibration>`
      * :doc:`Cannibalization <392-cannibalization>`
      * :doc:`Cardinality in Categorical Data <178-cardinality-in-categorical-data>`
      * :doc:`Catalog Coverage <268-catalog-coverage>`
      * :doc:`Categorical Drift <179-categorical-drift>`
      * :doc:`Categorical Explosions <182-categorical-explosions>`
      * :doc:`Causal Effect <306-causal-effect>`
      * :doc:`Causal Impact <112-causal-impact>`
      * :doc:`Causal Inference <117-causal-inference>`
      * :doc:`Causal ML (Causal Machine Learning) <197-causal-ml-causal-machine-learning>`
      * :doc:`Causal Trees <301-causal-trees>`
      * :doc:`Channel-Specific CAC (Customer Acquisition Cost) <047-channel-specific-cac-customer-acquisition-cost>`
      * :doc:`Chi-square (χ²) Test <324-chi-square-2-test>`
      * :doc:`Churn <123-churn>`
      * :doc:`Class Weighting <002-class-weighting>`
      * :doc:`Classification Models <294-classification-models>`
      * :doc:`Classification Probability <231-classification-probability>`
      * :doc:`Classifier Two-Sample Tests (C2STs) <175-classifier-two-sample-tests-c2sts>`
      * :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>`
      * :doc:`Cloud Inference <153-cloud-inference>`
      * :doc:`Cloud Inference with Big Payloads <152-cloud-inference-with-big-payloads>`
      * :doc:`Cluster-based undersampling <007-cluster-based-undersampling>`
      * :doc:`Cohort <183-cohort>`
      * :doc:`Cohort-Based LTV (Simple Version) <041-cohort-based-ltv-simple-version>`
      * :doc:`Compromise Power Analysis <093-compromise-power-analysis>`
      * :doc:`Compute budgets <383-compute-budgets>`
      * :doc:`Computer Vision (CV) <321-computer-vision-cv>`
      * :doc:`Concept Drift <330-concept-drift>`
      * :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>`
      * :doc:`Confidence Level <285-confidence-level>`
      * :doc:`Continuous Probabilistic Forecasts <230-continuous-probabilistic-forecasts>`
      * :doc:`Continuous Ranked Probability Score (CRPS) <402-continuous-ranked-probability-score-crps>`
      * :doc:`Continuous Retraining <161-continuous-retraining>`
      * :doc:`Conversion Rate (CR) <299-conversion-rate-cr>`
      * :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>`
      * :doc:`Correlation <305-correlation>`
      * :doc:`Cosine Similarity of Item Features <265-cosine-similarity-of-item-features>`
      * :doc:`Cost-Per-Click (CPC) Models <300-cost-per-click-cpc-models>`
      * :doc:`Counterfactual Explanations <336-counterfactual-explanations>`
      * :doc:`Covariate Drift (a.k.a. Covariate Shift) <387-covariate-drift-a-k-a-covariate-shift>`
      * :doc:`Coverage <411-coverage>`
      * :doc:`Cramér's V <180-cramer-s-v>`
      * :doc:`Crew Overtime <398-crew-overtime>`
      * :doc:`Critical Value <087-critical-value>`
      * :doc:`Cross-Selling <031-cross-selling>`
      * :doc:`Cross-Validation (CV) <136-cross-validation-cv>`
      * :doc:`CTR (Click-Through Rate) <421-ctr-click-through-rate>`
      * :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>`
      * :doc:`Cumulative Incremental Gain (CIG) <202-cumulative-incremental-gain-cig>`
      * :doc:`Cumulative Uplift <198-cumulative-uplift>`
      * :doc:`Customer Lifetime <042-customer-lifetime>`
      * :doc:`Customer Segmentation <033-customer-segmentation>`
      * :doc:`D2C (Direct-to-Consumer) <036-d2c-direct-to-consumer>`
      * :doc:`Data Drift <331-data-drift>`
      * :doc:`Data Leakage <131-data-leakage>`
      * :doc:`Dataset Shift <353-dataset-shift>`
      * :doc:`DCG (Discounted Cumulative Gain) <272-dcg-discounted-cumulative-gain>`
      * :doc:`Decision Trees <340-decision-trees>`
      * :doc:`Deep Ensembles <335-deep-ensembles>`
      * :doc:`DeLong's Test <352-delong-s-test>`
      * :doc:`Demand Forecasting <215-demand-forecasting>`
      * :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>`
      * :doc:`Deterministic forecasts <242-deterministic-forecasts>`
      * :doc:`Diminishing Utility <271-diminishing-utility>`
      * :doc:`Discriminatory Power <185-discriminatory-power>`
      * :doc:`Diversity (in Recommender Systems) <410-diversity-in-recommender-systems>`
      * :doc:`Dominating in Recommender Systems <267-dominating-in-recommender-systems>`
      * :doc:`Downsampling <368-downsampling>`
      * :doc:`Drift Detection <138-drift-detection>`
      * :doc:`Drift Guardrails <349-drift-guardrails>`
      * :doc:`Early Stopping <140-early-stopping>`
      * :doc:`Effect Size (δ) <106-effect-size>`
      * :doc:`Embedding <173-embedding>`
      * :doc:`Embedding Similarity <320-embedding-similarity>`
      * :doc:`Encode (in Feature Engineering) <318-encode-in-feature-engineering>`
      * :doc:`Energy Distance <176-energy-distance>`
      * :doc:`Ensemble <154-ensemble>`
      * :doc:`Epochs <141-epochs>`
      * :doc:`Equal Opportunity (Fairness) <029-equal-opportunity-fairness>`
      * :doc:`Equalized Odds (Fairness) <028-equalized-odds-fairness>`
      * :doc:`Evaluation Set <355-evaluation-set>`
      * :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>`
      * :doc:`Expected Calibration Error (ECE) <415-expected-calibration-error-ece>`
      * :doc:`F1-score <363-f1-score>`
      * :doc:`Fair Lending laws <332-fair-lending-laws>`
      * :doc:`Fairness Guardrails <351-fairness-guardrails>`
      * :doc:`Fairness parity <372-fairness-parity>`
      * :doc:`Feature Values <188-feature-values>`
      * :doc:`Fill Rate <220-fill-rate>`
      * :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>`
      * :doc:`FLOPs <156-flops>`
      * :doc:`Forecast Error <250-forecast-error>`
      * :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>`
      * :doc:`Forecasting Competitions <251-forecasting-competitions>`
      * :doc:`Four-Fifths (80%) Rule <189-four-fifths-80-rule>`
      * :doc:`Frequentist <059-frequentist>`
      * :doc:`Frozen Encoder <172-frozen-encoder>`
      * :doc:`FTEs <147-ftes>`
      * :doc:`Full Annotation <345-full-annotation>`
      * :doc:`Full Distribution <229-full-distribution>`
      * :doc:`Fully Loaded CAC (Customer Acquisition Cost) <044-fully-loaded-cac-customer-acquisition-cost>`
      * :doc:`Gaussian Processes (GPs) <054-gaussian-processes-gps>`
      * :doc:`Genre Overlap <263-genre-overlap>`
      * :doc:`Gini Coefficient <023-gini-coefficient>`
      * :doc:`Google Experiments <100-google-experiments>`
      * :doc:`Gross LTV (Customer Lifetime Value) <039-gross-ltv-customer-lifetime-value>`
      * :doc:`Gross Margin <043-gross-margin>`
      * :doc:`Group Sequential Testing <079-group-sequential-testing>`
      * :doc:`Guardrails (in ML & Data Systems) <166-guardrails-in-ml-data-systems>`
      * :doc:`Harmonic Mean <362-harmonic-mean>`
      * :doc:`High-Stakes Domains <334-high-stakes-domains>`
      * :doc:`Hit Rate (HR) <412-hit-rate-hr>`
      * :doc:`Hyperparameter <142-hyperparameter>`
      * :doc:`Hypothesis Testing <107-hypothesis-testing>`
      * :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>`
      * :doc:`Incremental Conversions <394-incremental-conversions>`
      * :doc:`Incremental Gain <200-incremental-gain>`
      * :doc:`Incremental Recovery Rate (IRR) <194-incremental-recovery-rate-irr>`
      * :doc:`Incremental Revenue <193-incremental-revenue>`
      * :doc:`Incremental Sales <195-incremental-sales>`
      * :doc:`Inference Cost (Inference $) <385-inference-cost-inference>`
      * :doc:`Interleaving Tests <379-interleaving-tests>`
      * :doc:`Intra-List Diversity (ILD) <266-intra-list-diversity-ild>`
      * :doc:`Isotonic Regression <281-isotonic-regression>`
      * :doc:`Item Coverage <270-item-coverage>`
      * :doc:`Jaccard index <264-jaccard-index>`
      * :doc:`Jensen–Shannon (JS) Divergence <326-jensenshannon-js-divergence>`
      * :doc:`k-fold cross-validation <135-k-fold-cross-validation>`
      * :doc:`k-fold Stratified Cross-Validation (Stratified CV) <382-k-fold-stratified-cross-validation-stratified-cv>`
      * :doc:`Kaggle <273-kaggle>`
      * :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>`
      * :doc:`KS shift (Kolmogorov–Smirnov shift) <388-ks-shift-kolmogorovsmirnov-shift>`
      * :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>`
      * :doc:`Kullback–Leibler (KL) Divergence <327-kullbackleibler-kl-divergence>`
      * :doc:`KYC <146-kyc>`
      * :doc:`Label Drift (a.k.a. Target Drift) <386-label-drift-a-k-a-target-drift>`
      * :doc:`Label Noise <354-label-noise>`
      * :doc:`Lagging Indicators <168-lagging-indicators>`
      * :doc:`Latency Guardrails <350-latency-guardrails>`
      * :doc:`Lead-Gen Software <049-lead-gen-software>`
      * :doc:`Leading Indicators <169-leading-indicators>`
      * :doc:`Likelihood <304-likelihood>`
      * :doc:`Likelihood Ratio (LR) <075-likelihood-ratio-lr>`
      * :doc:`LIME (Local Interpretable Model-agnostic Explanations) <337-lime-local-interpretable-model-agnostic-explanat>`
      * :doc:`Linear Models <341-linear-models>`
      * :doc:`LLMs (Large Language Models) <158-llms-large-language-models>`
      * :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>`
      * :doc:`Log-Odds <295-log-odds>`
      * :doc:`Log-Space <257-log-space>`
      * :doc:`Logistic Regression <292-logistic-regression>`
      * :doc:`Logit Space <291-logit-space>`
      * :doc:`Logits <420-logits>`
      * :doc:`Long Lead Times <210-long-lead-times>`
      * :doc:`Long-Tail Items <260-long-tail-items>`
      * :doc:`Loss Functions <289-loss-functions>`
      * :doc:`Lost Sales Value <219-lost-sales-value>`
      * :doc:`Low-pass Filtering <005-low-pass-filtering>`
      * :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>`
      * :doc:`LTV (Customer Lifetime Value) <373-ltv-customer-lifetime-value>`
      * :doc:`LTV:CAC Ratio <037-ltv-cac-ratio>`
      * :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>`
      * :doc:`Machine Learning (ML) <144-machine-learning-ml>`
      * :doc:`Macro AUC <314-macro-auc>`
      * :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`
      * :doc:`Macro Averaging <370-macro-averaging>`
      * :doc:`Macro F1 <019-macro-f1>`
      * :doc:`Macro Precision <021-macro-precision>`
      * :doc:`Macro Recall <020-macro-recall>`
      * :doc:`Macro Shifts <181-macro-shifts>`
      * :doc:`Mann–Whitney U Test (also called the Wilcoxon rank-sum test) <026-mannwhitney-u-test-also-called-the-wilcoxon-rank>`
      * :doc:`Manual review minutes <384-manual-review-minutes>`
      * :doc:`Margin of Error (MoE) <086-margin-of-error-moe>`
      * :doc:`Marginal Likelihood (also called The Model Evidence or Integrated Likelihood) <062-marginal-likelihood-also-called-the-model-eviden>`
      * :doc:`MASE (Mean Absolute Scaled Error) <403-mase-mean-absolute-scaled-error>`
      * :doc:`Maximum Calibration Error (MCE) <276-maximum-calibration-error-mce>`
      * :doc:`Maximum Mean Discrepancy (MMD) <177-maximum-mean-discrepancy-mmd>`
      * :doc:`MCMC (Markov Chain Monte Carlo) <057-mcmc-markov-chain-monte-carlo>`
      * :doc:`Mean <316-mean>`
      * :doc:`Mean Absolute Error (MAE) <408-mean-absolute-error-mae>`
      * :doc:`Mean Absolute Percentage Error (MAPE) <425-mean-absolute-percentage-error-mape>`
      * :doc:`Mean Average Precision (MAP) <414-mean-average-precision-map>`
      * :doc:`Mean Squared Error (MSE) <308-mean-squared-error-mse>`
      * :doc:`Median <315-median>`
      * :doc:`Medical AI <145-medical-ai>`
      * :doc:`Micro AUC <313-micro-auc>`
      * :doc:`Micro AUROC <011-micro-auroc>`
      * :doc:`Micro Averaging <369-micro-averaging>`
      * :doc:`Micro F1 <013-micro-f1>`
      * :doc:`Micro Precision <016-micro-precision>`
      * :doc:`Micro Recall <015-micro-recall>`
      * :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>`
      * :doc:`Model Distillation (Knowledge Distillation) <139-model-distillation-knowledge-distillation>`
      * :doc:`Model KPIs (Key Performance Indicators) <167-model-kpis-key-performance-indicators>`
      * :doc:`Model Score <364-model-score>`
      * :doc:`Model Stability <187-model-stability>`
      * :doc:`Model Weights <155-model-weights>`
      * :doc:`Monitoring Pipelines <162-monitoring-pipelines>`
      * :doc:`Multi-label Classification <012-multi-label-classification>`
      * :doc:`Multiclass AUROC <022-multiclass-auroc>`
      * :doc:`Multiclass Classification <311-multiclass-classification>`
      * :doc:`Multiclass Precision <359-multiclass-precision>`
      * :doc:`Multiclass stratified CV <134-multiclass-stratified-cv>`
      * :doc:`Multilabel Precision <360-multilabel-precision>`
      * :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>`
      * :doc:`Murphy's Decomposition <278-murphy-s-decomposition>`
      * :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>`
      * :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>`
      * :doc:`NDCG (Normalized Discounted Cumulative Gain) <413-ndcg-normalized-discounted-cumulative-gain>`
      * :doc:`NearMiss (Distance-based Undersampling) <006-nearmiss-distance-based-undersampling>`
      * :doc:`Net LTV (sometimes called Contribution LTV) <038-net-ltv-sometimes-called-contribution-ltv>`
      * :doc:`Neural Networks <287-neural-networks>`
      * :doc:`Normal Distribution <238-normal-distribution>`
      * :doc:`Normalize (in Feature Engineering) <319-normalize-in-feature-engineering>`
      * :doc:`Novelty (in Recommender Systems) <409-novelty-in-recommender-systems>`
      * :doc:`O'Brien–Fleming (OBF) Method <078-o-brienfleming-obf-method>`
      * :doc:`Off-Distribution <184-off-distribution>`
      * :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>`
      * :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>`
      * :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`
      * :doc:`ONNX (Open Neural Network Exchange) <344-onnx-open-neural-network-exchange>`
      * :doc:`OpenAI API (ML API) <150-openai-api-ml-api>`
      * :doc:`OpEx <157-opex>`
      * :doc:`Ops Health Dashboard <206-ops-health-dashboard>`
      * :doc:`Optimizely <069-optimizely>`
      * :doc:`Organic CAC (Customer Acquisition Cost) <045-organic-cac-customer-acquisition-cost>`
      * :doc:`Outlier <307-outlier>`
      * :doc:`Overconfident <284-overconfident>`
      * :doc:`Oversampling <004-oversampling>`
      * :doc:`Overstock % <400-overstock>`
      * :doc:`P-Value (probability value) <118-p-value-probability-value>`
      * :doc:`Paid CAC (Customer Acquisition Cost) <046-paid-cac-customer-acquisition-cost>`
      * :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>`
      * :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>`
      * :doc:`Per-class Precision (sometimes called class-wise precision) <358-per-class-precision-sometimes-called-class-wise>`
      * :doc:`Pinball Loss (a.k.a. Quantile Loss) <404-pinball-loss-a-k-a-quantile-loss>`
      * :doc:`Platt Scaling <280-platt-scaling>`
      * :doc:`Pocock Method <077-pocock-method>`
      * :doc:`Point Forecasts <233-point-forecasts>`
      * :doc:`Population Proportion <199-population-proportion>`
      * :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>`
      * :doc:`Post-hoc Explainability <339-post-hoc-explainability>`
      * :doc:`Posterior <063-posterior>`
      * :doc:`Posterior belief <061-posterior-belief>`
      * :doc:`Posterior Probability <073-posterior-probability>`
      * :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>`
      * :doc:`Power (1 – β) <104-power-1>`
      * :doc:`Power Analysis <378-power-analysis>`
      * :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>`
      * :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>`
      * :doc:`Predicting Percentiles <252-predicting-percentiles>`
      * :doc:`Prediction Intervals (PI) <253-prediction-intervals-pi>`
      * :doc:`Predictive LTV (pLTV) <040-predictive-ltv-pltv>`
      * :doc:`Predictive Parity (Calibration) <027-predictive-parity-calibration>`
      * :doc:`Prior Belief (or Prior Probability) <064-prior-belief-or-prior-probability>`
      * :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>`
      * :doc:`Probabilistic Interleaving <109-probabilistic-interleaving>`
      * :doc:`Probabilistic Scoring <228-probabilistic-scoring>`
      * :doc:`Probability <025-probability>`
      * :doc:`Probability Density <237-probability-density>`
      * :doc:`Probability Distribution <240-probability-distribution>`
      * :doc:`Probability Forecasts <235-probability-forecasts>`
      * :doc:`Probability Mass <239-probability-mass>`
      * :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>`
      * :doc:`Proportion <091-proportion>`
      * :doc:`PSI (Population Stability Index) <389-psi-population-stability-index>`
      * :doc:`Qini Coefficient <397-qini-coefficient>`
      * :doc:`Qini Curve <203-qini-curve>`
      * :doc:`Quantile Forecasts <232-quantile-forecasts>`
      * :doc:`Quantile Level <255-quantile-level>`
      * :doc:`Quantile Regression <254-quantile-regression>`
      * :doc:`Quantization <343-quantization>`
      * :doc:`Random Targeting Strategy <196-random-targeting-strategy>`
      * :doc:`Random Undersampling <008-random-undersampling>`
      * :doc:`Ranking Algorithms <108-ranking-algorithms>`
      * :doc:`Re-scoring <137-re-scoring>`
      * :doc:`Real-Time Inventory Tracking <213-real-time-inventory-tracking>`
      * :doc:`Recalibrate Thresholds <165-recalibrate-thresholds>`
      * :doc:`Recalibration <159-recalibration>`
      * :doc:`Recall <423-recall>`
      * :doc:`Regression Coefficient <090-regression-coefficient>`
      * :doc:`Regression Models <309-regression-models>`
      * :doc:`Relative accuracy <258-relative-accuracy>`
      * :doc:`Relevance in Recommender Systems <262-relevance-in-recommender-systems>`
      * :doc:`Reliability Curves (also called Calibration Curves) <416-reliability-curves-also-called-calibration-curve>`
      * :doc:`Reorder Point (ROP) Optimization <216-reorder-point-rop-optimization>`
      * :doc:`Representation Shift <174-representation-shift>`
      * :doc:`Retention <124-retention>`
      * :doc:`Return Distribution <225-return-distribution>`
      * :doc:`Revenue net of treatment cost <393-revenue-net-of-treatment-cost>`
      * :doc:`Revenue per User (RPU / ARPU) <122-revenue-per-user-rpu-arpu>`
      * :doc:`Reweighting <160-reweighting>`
      * :doc:`Risk Forecast <227-risk-forecast>`
      * :doc:`Risk of Peeking <116-risk-of-peeking>`
      * :doc:`Risk-Based Decisions <286-risk-based-decisions>`
      * :doc:`RMSLE (Root Mean Squared Logarithmic Error) <407-rmsle-root-mean-squared-logarithmic-error>`
      * :doc:`ROC Curve (Receiver Operating Characteristic) <277-roc-curve-receiver-operating-characteristic>`
      * :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>`
      * :doc:`ROI (Return on Investment) <191-roi-return-on-investment>`
      * :doc:`Root Mean Squared Error (RMSE) <426-root-mean-squared-error-rmse>`
      * :doc:`R² (R-squared) <259-r2-r-squared>`
      * :doc:`SaaS (Software as a Service) <034-saas-software-as-a-service>`
      * :doc:`Safety Stock <217-safety-stock>`
      * :doc:`Sample Mean <089-sample-mean>`
      * :doc:`Sample size <103-sample-size>`
      * :doc:`Sample Standard Deviation <088-sample-standard-deviation>`
      * :doc:`Seasonal Lag <247-seasonal-lag>`
      * :doc:`Seasonality <329-seasonality>`
      * :doc:`Selection Rate <390-selection-rate>`
      * :doc:`Self-Information of Popularity <261-self-information-of-popularity>`
      * :doc:`Sensitivity in Feature Engineering <317-sensitivity-in-feature-engineering>`
      * :doc:`Sequential Probability Ratio Test (SPRT) <076-sequential-probability-ratio-test-sprt>`
      * :doc:`Sequential Settings <058-sequential-settings>`
      * :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>`
      * :doc:`Session Length <121-session-length>`
      * :doc:`SHAP (SHapley Additive exPlanations) <338-shap-shapley-additive-explanations>`
      * :doc:`Sigmoid Function <297-sigmoid-function>`
      * :doc:`Signal Processing <009-signal-processing>`
      * :doc:`Significance Level (α) <105-significance-level>`
      * :doc:`Simple Baseline Methods <248-simple-baseline-methods>`
      * :doc:`Single-label Classification <014-single-label-classification>`
      * :doc:`SKU <212-sku>`
      * :doc:`SLA (Service Level Agreement) <208-sla-service-level-agreement>`
      * :doc:`SLA Breach Rate <207-sla-breach-rate>`
      * :doc:`SLA Breaches <399-sla-breaches>`
      * :doc:`SLI (Service Level Indicator) <190-sli-service-level-indicator>`
      * :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>`
      * :doc:`SLOs (Service Level Objectives) <391-slos-service-level-objectives>`
      * :doc:`Slow-Moving SKUs <211-slow-moving-skus>`
      * :doc:`sMAPE (Symmetric Mean Absolute Percentage Error) <406-smape-symmetric-mean-absolute-percentage-error>`
      * :doc:`SMOTE (Synthetic Minority Over-sampling Technique) <003-smote-synthetic-minority-over-sampling-technique>`
      * :doc:`Softmax Function <296-softmax-function>`
      * :doc:`Squashing Function <298-squashing-function>`
      * :doc:`Standard Error (SE) <084-standard-error-se>`
      * :doc:`Statistical Power <348-statistical-power>`
      * :doc:`Statistical Significance <096-statistical-significance>`
      * :doc:`Statistical Tests <328-statistical-tests>`
      * :doc:`Statistically Significant <125-statistically-significant>`
      * :doc:`Stockout Rate <221-stockout-rate>`
      * :doc:`Stockouts <401-stockouts>`
      * :doc:`Stopping Rules <071-stopping-rules>`
      * :doc:`Stratified Group K-Fold <132-stratified-group-k-fold>`
      * :doc:`Stratified Shuffle Split <133-stratified-shuffle-split>`
      * :doc:`Strictly Proper Scoring Rules <234-strictly-proper-scoring-rules>`
      * :doc:`Subsampling <001-subsampling>`
      * :doc:`Supplier Constraints <209-supplier-constraints>`
      * :doc:`Supplier Management <214-supplier-management>`
      * :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>`
      * :doc:`T-Test <120-t-test>`
      * :doc:`Target Variable <236-target-variable>`
      * :doc:`Team Draft Interleaving (TDI) <110-team-draft-interleaving-tdi>`
      * :doc:`Temperature Scaling <279-temperature-scaling>`
      * :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>`
      * :doc:`Thompson Sampling (TS) in Bandits (Multi-Armed Bandit Problem (MAB)) <050-thompson-sampling-ts-in-bandits-multi-armed-band>`
      * :doc:`Time Series <010-time-series>`
      * :doc:`Time Series Forecasting <256-time-series-forecasting>`
      * :doc:`Time-based splits (a.k.a. Temporal Cross-Validation, Rolling Window Validation) <381-time-based-splits-a-k-a-temporal-cross-validatio>`
      * :doc:`Total Incremental Benefit (TIB) <201-total-incremental-benefit-tib>`
      * :doc:`TPU Clusters <347-tpu-clusters>`
      * :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`
      * :doc:`Treatment Cost <192-treatment-cost>`
      * :doc:`Treatment Effect <072-treatment-effect>`
      * :doc:`TREC (Text REtrieval Conference) <274-trec-text-retrieval-conference>`
      * :doc:`Trivial Effects <102-trivial-effects>`
      * :doc:`True Conversion Rate <083-true-conversion-rate>`
      * :doc:`True Mean (Population Mean) <085-true-mean-population-mean>`
      * :doc:`True Population Parameter <092-true-population-parameter>`
      * :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>`
      * :doc:`Type I Error <080-type-i-error>`
      * :doc:`Underconfident <283-underconfident>`
      * :doc:`Underflow <290-underflow>`
      * :doc:`Uplift <424-uplift>`
      * :doc:`Uplift Curve <303-uplift-curve>`
      * :doc:`Uplift Models <205-uplift-models>`
      * :doc:`Uplift Random Forests <302-uplift-random-forests>`
      * :doc:`Uplift Score <204-uplift-score>`
      * :doc:`Uplift@k <395-uplift-k>`
      * :doc:`Upsampling <367-upsampling>`
      * :doc:`Upselling <032-upselling>`
      * :doc:`User Coverage <269-user-coverage>`
      * :doc:`Valuation Metric <035-valuation-metric>`
      * :doc:`Value-at-Risk (VaR) <226-value-at-risk-var>`
      * :doc:`Variational Inference (VI) <056-variational-inference-vi>`
      * :doc:`Vertex AI <149-vertex-ai>`
      * :doc:`WAPE (Weighted Absolute Percentage Error) <422-wape-weighted-absolute-percentage-error>`
      * :doc:`Weak Supervision <346-weak-supervision>`
      * :doc:`Weighted Averaging <361-weighted-averaging>`
      * :doc:`Wilson Score Interval <357-wilson-score-interval>`
      * :doc:`Windows (in Time-Series) <170-windows-in-time-series>`
      * :doc:`WMAPE (Weighted Mean Absolute Percentage Error) <405-wmape-weighted-mean-absolute-percentage-error>`
      * :doc:`Z-Score <097-z-score>`
      * :doc:`Z-Test <119-z-test>`


.. toctree::
   :hidden:

   099-beta-distribution
   285-confidence-level
   305-correlation
   087-critical-value
   243-cumulative-distribution-function-cdf
   059-frequentist
   126-iid-independent-and-identically-distributed
   304-likelihood
   086-margin-of-error-moe
   316-mean
   315-median
   238-normal-distribution
   307-outlier
   199-population-proportion
   025-probability
   237-probability-density
   240-probability-distribution
   239-probability-mass
   091-proportion
   090-regression-coefficient
   089-sample-mean
   088-sample-standard-deviation
   084-standard-error-se
   096-statistical-significance
   125-statistically-significant
   085-true-mean-population-mean
   092-true-population-parameter
   097-z-score
   095-a-priori-power-analysis
   324-chi-square-2-test
   356-clopperpearson-interval
   093-compromise-power-analysis
   377-confidence-intervals-cis
   106-effect-size
   107-hypothesis-testing
   325-kolmogorovsmirnov-ks-test
   101-minimum-detectable-lift-mdl
   118-p-value-probability-value
   094-post-hoc-power-analysis
   104-power-1
   378-power-analysis
   103-sample-size
   105-significance-level
   348-statistical-power
   328-statistical-tests
   120-t-test
   102-trivial-effects
   098-two-proportion-z-test
   080-type-i-error
   357-wilson-score-interval
   119-z-test
   143-ai-artificial-intelligence
   294-classification-models
   321-computer-vision-cv
   340-decision-trees
   341-linear-models
   158-llms-large-language-models
   292-logistic-regression
   144-machine-learning-ml
   145-medical-ai
   322-natural-language-processing-nlp
   287-neural-networks
   309-regression-models
   282-support-vector-machines-svms
   236-target-variable
   002-class-weighting
   007-cluster-based-undersampling
   368-downsampling
   006-nearmiss-distance-based-undersampling
   004-oversampling
   008-random-undersampling
   003-smote-synthetic-minority-over-sampling-technique
   001-subsampling
   367-upsampling
   323-accuracy
   371-auc-area-under-the-curve
   366-average-precision-ap
   293-binary-classification
   231-classification-probability
   185-discriminatory-power
   363-f1-score
   023-gini-coefficient
   362-harmonic-mean
   417-log-loss-also-called-logarithmic-loss-or-cross-e
   314-macro-auc
   018-macro-auroc-macro-averaged-auroc
   370-macro-averaging
   019-macro-f1
   021-macro-precision
   020-macro-recall
   313-micro-auc
   011-micro-auroc
   369-micro-averaging
   013-micro-f1
   016-micro-precision
   015-micro-recall
   364-model-score
   012-multi-label-classification
   022-multiclass-auroc
   311-multiclass-classification
   359-multiclass-precision
   360-multilabel-precision
   310-one-vs-rest-ovr
   017-one-vs-rest-ovr-auroc
   312-partial-auc-pauc
   358-per-class-precision-sometimes-called-class-wise
   429-precision-a-k-a-positive-predictive-value-ppv
   430-precisionrecall-auc-pr-auc
   423-recall
   277-roc-curve-receiver-operating-characteristic
   427-roc-auc-receiver-operating-characteristic-area-u
   014-single-label-classification
   361-weighted-averaging
   246-average-absolute-error-aae
   428-baseline-heuristics
   365-bootstrap
   024-bootstrap-confidence-intervals-cis
   411-coverage
   180-cramer-s-v
   352-delong-s-test
   186-ks-statistic-kolmogorovsmirnov-statistic
   075-likelihood-ratio-lr
   026-mannwhitney-u-test-also-called-the-wilcoxon-rank
   403-mase-mean-absolute-scaled-error
   408-mean-absolute-error-mae
   425-mean-absolute-percentage-error-mape
   308-mean-squared-error-mse
   258-relative-accuracy
   407-rmsle-root-mean-squared-logarithmic-error
   426-root-mean-squared-error-rmse
   259-r2-r-squared
   406-smape-symmetric-mean-absolute-percentage-error
   422-wape-weighted-absolute-percentage-error
   405-wmape-weighted-mean-absolute-percentage-error
   380-a-b-testing
   114-a-b-n-test
   074-bayesian-sequential-testing
   068-bayesian-stopping-rules
   067-conversion-rate-uplift
   082-fixed-horizon-testing
   079-group-sequential-testing
   115-multivariate-test-mvt
   070-online-experimentation-platforms
   069-optimizely
   116-risk-of-peeking
   376-sequential-testing-also-called-sequential-analys
   071-stopping-rules
   081-traditional-a-b-test-fixed-horizon-a-b-test
   072-treatment-effect
   083-true-conversion-rate
   048-blended-cac-customer-acquisition-cost
   374-cac-customer-acquisition-cost
   392-cannibalization
   047-channel-specific-cac-customer-acquisition-cost
   123-churn
   183-cohort
   041-cohort-based-ltv-simple-version
   299-conversion-rate-cr
   300-cost-per-click-cpc-models
   031-cross-selling
   421-ctr-click-through-rate
   042-customer-lifetime
   033-customer-segmentation
   036-d2c-direct-to-consumer
   147-ftes
   044-fully-loaded-cac-customer-acquisition-cost
   039-gross-ltv-customer-lifetime-value
   043-gross-margin
   146-kyc
   168-lagging-indicators
   049-lead-gen-software
   169-leading-indicators
   373-ltv-customer-lifetime-value
   037-ltv-cac-ratio
   038-net-ltv-sometimes-called-contribution-ltv
   157-opex
   045-organic-cac-customer-acquisition-cost
   046-paid-cac-customer-acquisition-cost
   040-predictive-ltv-pltv
   124-retention
   122-revenue-per-user-rpu-arpu
   191-roi-return-on-investment
   034-saas-software-as-a-service
   121-session-length
   032-upselling
   035-valuation-metric
   128-blocked-splits-single-holdout
   136-cross-validation-cv
   131-data-leakage
   355-evaluation-set
   130-expanding-window-cross-validation
   135-k-fold-cross-validation
   382-k-fold-stratified-cross-validation-stratified-cv
   134-multiclass-stratified-cv
   129-sliding-window-rolling-window-cross-validation
   132-stratified-group-k-fold
   133-stratified-shuffle-split
   381-time-based-splits-a-k-a-temporal-cross-validatio
   163-active-learning
   288-binary-cross-entropy-bce
   335-deep-ensembles
   140-early-stopping
   154-ensemble
   141-epochs
   156-flops
   345-full-annotation
   142-hyperparameter
   354-label-noise
   295-log-odds
   291-logit-space
   420-logits
   289-loss-functions
   139-model-distillation-knowledge-distillation
   155-model-weights
   343-quantization
   297-sigmoid-function
   296-softmax-function
   298-squashing-function
   290-underflow
   346-weak-supervision
   148-aws-sagemaker
   100-google-experiments
   273-kaggle
   344-onnx-open-neural-network-exchange
   150-openai-api-ml-api
   347-tpu-clusters
   149-vertex-ai
   218-backorder-rate
   398-crew-overtime
   215-demand-forecasting
   220-fill-rate
   210-long-lead-times
   260-long-tail-items
   219-lost-sales-value
   400-overstock
   213-real-time-inventory-tracking
   216-reorder-point-rop-optimization
   217-safety-stock
   212-sku
   211-slow-moving-skus
   221-stockout-rate
   401-stockouts
   209-supplier-constraints
   214-supplier-management
   431-advanced-sorting-in-spreadsheets
   318-encode-in-feature-engineering
   319-normalize-in-feature-engineering
   317-sensitivity-in-feature-engineering
   030-demographic-parity-statistical-parity
   029-equal-opportunity-fairness
   028-equalized-odds-fairness
   351-fairness-guardrails
   372-fairness-parity
   189-four-fifths-80-rule
   027-predictive-parity-calibration
   390-selection-rate
   066-bayes-theorem
   164-bayesian-correction
   051-bayesian-decision-theory-bdt
   375-bayesian-inference
   055-bayesian-neural-networks-bnns
   060-binomial-likelihood
   054-gaussian-processes-gps
   062-marginal-likelihood-also-called-the-model-eviden
   057-mcmc-markov-chain-monte-carlo
   065-parameter-s-of-interest
   063-posterior
   061-posterior-belief
   073-posterior-probability
   053-posterior-probability-of-uplift
   064-prior-belief-or-prior-probability
   056-variational-inference-vi
   113-bandit-algorithms
   078-o-brienfleming-obf-method
   077-pocock-method
   076-sequential-probability-ratio-test-sprt
   058-sequential-settings
   050-thompson-sampling-ts-in-bandits-multi-armed-band
   224-arima-autoregressive-integrated-moving-average
   052-bayesian-time-series
   250-forecast-error
   245-forecasting-benchmarks
   251-forecasting-competitions
   257-log-space
   005-low-pass-filtering
   223-lstm-long-short-term-memory-networks
   244-m-competitions-makridakis-competitions
   249-naive-baseline-forecast
   222-prophet-time-series-forecasting-by-facebook-meta
   247-seasonal-lag
   329-seasonality
   009-signal-processing
   248-simple-baseline-methods
   127-temporal-autocorrelation-serial-correlation
   010-time-series
   256-time-series-forecasting
   170-windows-in-time-series
   151-aws-sagemaker-endpoints
   342-caching
   153-cloud-inference
   152-cloud-inference-with-big-payloads
   383-compute-budgets
   161-continuous-retraining
   188-feature-values
   166-guardrails-in-ml-data-systems
   385-inference-cost-inference
   350-latency-guardrails
   384-manual-review-minutes
   167-model-kpis-key-performance-indicators
   187-model-stability
   162-monitoring-pipelines
   206-ops-health-dashboard
   137-re-scoring
   165-recalibrate-thresholds
   159-recalibration
   160-reweighting
   208-sla-service-level-agreement
   207-sla-breach-rate
   399-sla-breaches
   190-sli-service-level-indicator
   391-slos-service-level-objectives
   178-cardinality-in-categorical-data
   179-categorical-drift
   182-categorical-explosions
   175-classifier-two-sample-tests-c2sts
   330-concept-drift
   387-covariate-drift-a-k-a-covariate-shift
   331-data-drift
   353-dataset-shift
   138-drift-detection
   349-drift-guardrails
   176-energy-distance
   326-jensenshannon-js-divergence
   388-ks-shift-kolmogorovsmirnov-shift
   327-kullbackleibler-kl-divergence
   386-label-drift-a-k-a-target-drift
   181-macro-shifts
   177-maximum-mean-discrepancy-mmd
   184-off-distribution
   389-psi-population-stability-index
   174-representation-shift
   171-autoencoder
   173-embedding
   320-embedding-similarity
   172-frozen-encoder
   111-balanced-interleaving
   272-dcg-discounted-cumulative-gain
   379-interleaving-tests
   414-mean-average-precision-map
   413-ndcg-normalized-discounted-cumulative-gain
   109-probabilistic-interleaving
   108-ranking-algorithms
   110-team-draft-interleaving-tdi
   274-trec-text-retrieval-conference
   396-auuc-area-under-the-uplift-curve
   306-causal-effect
   112-causal-impact
   117-causal-inference
   197-causal-ml-causal-machine-learning
   301-causal-trees
   202-cumulative-incremental-gain-cig
   198-cumulative-uplift
   394-incremental-conversions
   200-incremental-gain
   194-incremental-recovery-rate-irr
   193-incremental-revenue
   195-incremental-sales
   397-qini-coefficient
   203-qini-curve
   196-random-targeting-strategy
   393-revenue-net-of-treatment-cost
   201-total-incremental-benefit-tib
   192-treatment-cost
   424-uplift
   303-uplift-curve
   205-uplift-models
   302-uplift-random-forests
   204-uplift-score
   395-uplift-k
   230-continuous-probabilistic-forecasts
   402-continuous-ranked-probability-score-crps
   242-deterministic-forecasts
   229-full-distribution
   404-pinball-loss-a-k-a-quantile-loss
   233-point-forecasts
   252-predicting-percentiles
   253-prediction-intervals-pi
   241-probabilistic-forecasts
   228-probabilistic-scoring
   235-probability-forecasts
   232-quantile-forecasts
   255-quantile-level
   254-quantile-regression
   225-return-distribution
   227-risk-forecast
   286-risk-based-decisions
   234-strictly-proper-scoring-rules
   226-value-at-risk-var
   268-catalog-coverage
   265-cosine-similarity-of-item-features
   271-diminishing-utility
   410-diversity-in-recommender-systems
   267-dominating-in-recommender-systems
   263-genre-overlap
   412-hit-rate-hr
   266-intra-list-diversity-ild
   270-item-coverage
   264-jaccard-index
   409-novelty-in-recommender-systems
   262-relevance-in-recommender-systems
   261-self-information-of-popularity
   269-user-coverage
   275-adaptive-ece-expected-calibration-error-with-ada
   418-brier-score
   419-calibration-quality-model-calibration
   415-expected-calibration-error-ece
   281-isotonic-regression
   276-maximum-calibration-error-mce
   278-murphy-s-decomposition
   284-overconfident
   280-platt-scaling
   416-reliability-curves-also-called-calibration-curve
   279-temperature-scaling
   283-underconfident
   333-basel-iii
   336-counterfactual-explanations
   332-fair-lending-laws
   334-high-stakes-domains
   337-lime-local-interpretable-model-agnostic-explanat
   339-post-hoc-explainability
   338-shap-shapley-additive-explanations

.. tags:: purpose: reference, topic: terminology
