# 🧪 snsx: Evolution-Aware ML Plotting Library

## 🌐 Modular API Vision (`snsx`)

```python
snsx.evaluation.roc_curve(y_true, y_score)
snsx.representation.pca(data=X, hue=clusters)
snsx.explanation.shap_summary(model, X)
snsx.features.mutual_information(X, y)
snsx.dataset.null_heatmap(df)
snsx.training.loss_curve(history)
snsx.selection.feature_ranking(model)
snsx.robustness.adversarial_examples(model, X)
snsx.monitoring.drift_map(production_data, reference_data)
snsx.causality.counterfactual_plot(model, X)
```

### snsx Plotting Function Metadata

```python
snsx.find(task="classification", explainability="high")
```

| Function                              | Category       | Task Type        | Plot Type          | Supervised | Explainability Level  |
|---------------------------------------|----------------|------------------|--------------------|------------|-----------------------|
| snsx.evaluation.roc_curve             | evaluation     | classification   | diagnostic         | yes        | medium                |
| snsx.evaluation.pr_curve              | evaluation     | classification   | diagnostic         | yes        | medium                |
| snsx.evaluation.confusion_matrix      | evaluation     | classification   | matrix             | yes        | low                   |
| snsx.evaluation.calibration_curve     | evaluation     | classification   | calibration        | yes        | medium                |
| snsx.evaluation.lift_curve            | evaluation     | classification   | ranking/performance| yes        | medium                |
| snsx.evaluation.lift_decile_wise      | evaluation     | classification   | bar                | yes        | medium                |
| snsx.evaluation.ks_statistic          | evaluation     | classification   | diagnostic         | yes        | medium                |
| snsx.evaluation.cumulative_gain       | evaluation     | classification   | cumulative         | yes        | medium                |
| snsx.evaluation.decile_table          | evaluation     | classification   | tabular            | yes        | medium                |
| snsx.representation.pca               | representation | unsupervised     | embedding          | no         | low                   |
| snsx.representation.tsne              | representation | unsupervised     | embedding          | no         | low                   |
| snsx.representation.umap              | representation | unsupervised     | embedding          | no         | low                   |
| snsx.explanation.shap_summary         | explanation    | general          | importance         | yes        | high                  |
| snsx.explanation.pdp_plot             | explanation    | general          | partial_dependence | yes        | high                  |
| snsx.features.mutual_information      | features       | general          | importance         | yes        | medium                |
| snsx.features.correlation_matrix      | features       | general          | matrix             | no         | medium                |
| snsx.features.missingness_plot        | features       | general          | distribution       | no         | low                   |
| snsx.dataset.null_heatmap             | dataset        | general          | matrix             | no         | low                   |
| snsx.dataset.dtype_stats              | dataset        | general          | bar                | no         | low                   |
| snsx.training.loss_curve              | training       | general          | curve              | yes        | medium                |
| snsx.selection.feature_ranking        | selection      | general          | importance         | yes        | high                  |
| snsx.robustness.adversarial_examples  | robustness     | classification   | adversarial        | yes        | high                  |
| snsx.monitoring.drift_map             | monitoring     | general          | drift              | yes        | high                  |
| snsx.monitoring.latency_curve         | monitoring     | general          | curve              | no         | low                   |
| snsx.causality.counterfactual_plot    | causality      | general          | counterfactual     | yes        | high                  |
| snsx.pipeline.workflow_dag            | pipeline       | general          | graph              | no         | medium                |
| snsx.fairness.disparity_chart         | fairness       | classification   | fairness           | yes        | high                  |
| snsx.target.class_balance_bar         | target         | classification   | bar                | yes        | low                   |
| snsx.target.leakage_check_plot        | target         | general          | diagnostic         | yes        | medium                |
| snsx.uncertainty.prediction_entropy   | uncertainty    | general          | distribution       | yes        | high                  |
| snsx.comparison.metric_delta_plot     | comparison     | general          | bar                | yes        | medium                |

## Optional / Advanced categories

| Category         | ML Role                        | Purpose                                                | Example Visuals                                            |
|------------------|--------------------------------|--------------------------------------------------------|------------------------------------------------------------|
| causality        | Causal inference               | Discover or visualize causal relationships             | Causal graphs, counterfactual plots, do-calculus           |
| comparison       | Benchmarking                   | Compare models or results side by side                 | Metric deltas, win/loss charts, scoreboards                |
| dataset          | Data profiling & quality       | Summarize dataset structure, schema, and completeness  | Null heatmap, dtype stats, memory usage                    |
| deployment       | Post-training monitoring       | Detect model or data drift in production               | Population stability index, data drift maps                |
| evaluation       | Predictive diagnostics         | Measure model performance across task types            | ROC, PR, confusion matrix, calibration, Brier curve        |
| explanation      | Interpretability & attribution | Explain model outputs or decision logic                | SHAP, LIME, PDP, ICE, attention maps                       |
| fairness         | Bias & ethics                  | Detect and explain fairness issues                     | Disparate impact, demographic parity, fairness dashboard   |
| features         | Feature behavior & engineering | Diagnose and understand input variables                | Correlation, MI, missingness, drift                        |
| monitoring       | Operational metrics            | Track runtime metrics and alerts in deployment         | Latency curves, prediction volume, service errors          |
| pipeline         | Workflow visualization         | Visualize ML pipeline components and data flow         | DAG of transformations, preprocessor traces                |
| representation   | Latent space & embedding       | Explore reduced/learned feature spaces                 | PCA, t-SNE, UMAP, autoencoder latent plots                 |
| robustness       | Stability & perturbation       | Assess model sensitivity to noise or adversaries       | Adversarial examples, perturbation maps, uncertainty bands |
| selection        | Feature selection              | Rank or filter features based on predictive power      | RFE, Lasso path, SHAP ranking, permutation importance      |
| target           | Label structure diagnostics    | Validate and explore supervised learning targets       | Class imbalance, leakage detection, target skewness        |
| training         | Optimization dynamics          | Analyze convergence and training stability             | Loss vs. epoch, accuracy curves, early stopping            |
| uncertainty      | Probabilistic modeling         | Quantify and visualize model uncertainty               | Confidence intervals, prediction distribution, entropy     |

## Taxonomy for Evolution-Aware ML Plotting

### Core ML Phases and Logical Flow

| Core ML Phase       | Categories (Input → Transform → Output)                                                 |
|---------------------|-----------------------------------------------------------------------------------------|
| Data Understanding  | dataset                  → dataset, features, target                                    |
| Representation      | features, target         → representation, selection                                    |
| Modeling            | features, representation → training, selection → explanation, uncertainty, robustness   |
| Evaluation          | target, model output     → evaluation, fairness, comparison                             |
| Deployment          | model, incoming data     → monitoring, deployment, pipeline                             |
| Reasoning Analysis  | features, target         → causality                                                    |

## Tiered Adoption: Core, Extended, Experimental

#### Core
- dataset, features, target, training, evaluation, explanation

#### Extended
- robustness, uncertainty, representation, selection, deployment, monitoring, comparison, fairness

#### Experimental
- pipeline, causality

## Role-Based Views: Who Uses What?

Segment the taxonomy by user role — helps align tooling with personas:

| Role                | Relevant Categories                                                                     |
|---------------------|-----------------------------------------------------------------------------------------|
| Data Scientist      | dataset, features, target, training, evaluation, explanation                            |
| ML Engineer         | pipeline, deployment, monitoring, robustness, uncertainty                               |
| Researcher          | representation, selection, causality, comparison                                        |
| Compliance / Ethics | fairness, causality, explanation                                                        |
