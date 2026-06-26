:html_theme.sidebar_secondary.remove:

..
   ######################################################################
   learn/terminology/index.rst
   ======================================================================
   Terminology hub — machine learning & data science concepts as used in
   scikit-plots.  Written for Sphinx + PyData Theme.
   Extensions used:
     sphinx_design  (tab-set, grid, dropdown, badge)
     sphinx_tags    (auto-generated tag pages)
     sphinx_togglebutton (toggle)
   ======================================================================
   Section underline convention (mirrors the project):
     = sections (with overline)   ########## or ========
     - subsections
     ^ subsubsections
     " paragraphs
   ######################################################################

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _terminology-index:

:raw-html:`<div style="text-align:center"><strong>`
📖 Terminology Reference :raw-html:`<br/>`
Your complete guide to ML & Data Science concepts in scikit-plots
|br| |full_version| - |today|
:raw-html:`</strong></div>`

======================================================================
Terminology
======================================================================

This reference organises every machine-learning and data-science term
you will encounter when using **scikit-plots** — from the most
elementary ideas (What is a True Positive?) to expert-level subtleties
(Macro-averaged AUROC in imbalanced multiclass problems).

Each entry answers four questions concisely:

* **What is it?** — a plain-English definition.
* **Formula / Key Relationship** — the exact mathematical statement.
* **When to use it** — the right context.
* **scikit-plots / scikit-learn connection** — the API call that
  produces or consumes this concept.

Use the level tabs below to start at the depth that suits you, then
follow cross-references to go deeper.

.. note::

   Terms are grouped by **domain**, not alphabetically, so that related
   concepts appear together.  Use your browser's
   :kbd:`Ctrl` + :kbd:`F` or the Sphinx search to jump to a specific
   term.

----------------------------------------------------------------------

.. _terminology-discovery:

Discovery at a Glance
----------------------------------------------------------------------

.. tab-set::
   :class: sd-width-100

   .. tab-item:: 🟢 Start Here — Foundations
      :sync: level-foundations

      Core building blocks every practitioner must know.
      No formulas required — just intuition.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 📋 Confusion Matrix
            :link: terminology-confusion-matrix
            :link-type: ref
            :class-card: sd-border-1

            The 2×2 (or K×K) table that underpins every classification
            metric. Start here.

         .. grid-item-card:: 🎯 Precision & Recall
            :link: terminology-precision-recall
            :link-type: ref
            :class-card: sd-border-1

            The fundamental trade-off: catching more positives vs.
            trusting your predictions.

         .. grid-item-card:: ⚖️ F1 Score
            :link: terminology-f-scores
            :link-type: ref
            :class-card: sd-border-1

            The harmonic mean of precision and recall — a single number
            that balances both.

         .. grid-item-card:: 🏷️ Classification Types
            :link: terminology-classification-types
            :link-type: ref
            :class-card: sd-border-1

            Binary, multiclass, multi-label — which problem are you
            actually solving?

         .. grid-item-card:: 📉 Data Imbalance
            :link: terminology-data-imbalance
            :link-type: ref
            :class-card: sd-border-1

            When one class dominates — oversampling, undersampling, and
            class weighting.

         .. grid-item-card:: 📈 ROC Curve
            :link: terminology-roc-auroc
            :link-type: ref
            :class-card: sd-border-1

            The performance landscape across every classification
            threshold at once.

   .. tab-item:: 🔵 Go Deeper — Metrics
      :sync: level-metrics

      Intermediate concepts for practitioners building and evaluating
      real models.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 📐 Averaging Strategies
            :link: terminology-averaging
            :link-type: ref
            :class-card: sd-border-1

            Macro vs. Micro vs. Weighted — how single numbers are
            derived from per-class scores.

         .. grid-item-card:: 🔢 Multiclass AUROC
            :link: terminology-multiclass-auroc
            :link-type: ref
            :class-card: sd-border-1

            Extending the ROC framework from two classes to K classes
            with OvR and OvO strategies.

         .. grid-item-card:: 🧪 SMOTE & Sampling
            :link: terminology-sampling
            :link-type: ref
            :class-card: sd-border-1

            Synthetic minority oversampling, NearMiss, cluster-based
            strategies — when and how.

         .. grid-item-card:: 📊 Statistical Tests
            :link: terminology-statistics
            :link-type: ref
            :class-card: sd-border-1

            Bootstrap CIs, Mann-Whitney U, and other tools for
            comparing models rigorously.

         .. grid-item-card:: 🎛️ Calibration
            :link: terminology-calibration
            :link-type: ref
            :class-card: sd-border-1

            Does P̂ = 0.8 really mean 80 % likely? Reliability diagrams
            and calibration curves.

         .. grid-item-card:: ⚡ Signal Processing
            :link: terminology-signal-timeseries
            :link-type: ref
            :class-card: sd-border-1

            Subsampling, downsampling, aliasing, low-pass filtering —
            for time-series and DSP work.

   .. tab-item:: 🔴 Expert — Advanced Concepts
      :sync: level-advanced

      Nuanced topics for senior practitioners, researchers, and
      contributors.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: ⚖️ Fairness Metrics
            :link: terminology-fairness
            :link-type: ref
            :class-card: sd-border-1

            Demographic parity, equal opportunity, equalized odds,
            predictive parity — choosing the right fairness criterion.

         .. grid-item-card:: 🔄 OvR vs. OvO
            :link: terminology-ovr-ovo
            :link-type: ref
            :class-card: sd-border-1

            One-vs-Rest and One-vs-One decomposition strategies and
            their impact on AUROC computation.

         .. grid-item-card:: 📉 Gini Coefficient
            :link: terminology-gini
            :link-type: ref
            :class-card: sd-border-1

            The relationship between Gini index and AUROC — and when
            Gini is the preferred reporting metric.

         .. grid-item-card:: 🧬 Bootstrap CIs
            :link: terminology-bootstrap
            :link-type: ref
            :class-card: sd-border-1

            Constructing confidence intervals for any metric without
            parametric assumptions.

         .. grid-item-card:: 🔬 Imbalance + Fairness
            :link: terminology-imbalance-fairness
            :link-type: ref
            :class-card: sd-border-1

            When class imbalance interacts with group fairness — the
            hidden pitfalls.

         .. grid-item-card:: 📡 Aliasing & Nyquist
            :link: terminology-aliasing
            :link-type: ref
            :class-card: sd-border-1

            Why subsampling without a low-pass filter corrupts signals
            — the Nyquist-Shannon theorem.

----------------------------------------------------------------------

.. _terminology-confusion-matrix:

Domain 1 — Confusion Matrix & Core Metrics
----------------------------------------------------------------------

The confusion matrix is the single most important data structure in
classification evaluation.  All threshold-based metrics derive from
its four cells.

.. _terminology-confusion-matrix-entry:

.. dropdown:: Confusion Matrix
   :color: primary
   :icon: table
   :open:

   **What is it?**

   A square table that tallies the agreement and disagreement between
   a classifier's predicted labels and the true labels on a held-out
   dataset.  For a binary problem it has four cells:

   .. list-table::
      :header-rows: 1
      :widths: 25 37 38

      * - Cell
        - Full Name
        - Meaning
      * - **TP**
        - True Positive
        - Predicted positive, actually positive
      * - **TN**
        - True Negative
        - Predicted negative, actually negative
      * - **FP**
        - False Positive (Type I error)
        - Predicted positive, actually negative
      * - **FN**
        - False Negative (Type II error)
        - Predicted negative, actually positive

   For a K-class problem the matrix is K×K: row ``i``, column ``j``
   counts samples with true label ``i`` predicted as label ``j``.
   The diagonal contains correct predictions.

   **scikit-plots connection**

   .. code-block:: python

      from sklearn.metrics import confusion_matrix
      import scikitplot as skplt

      # Plot normalised confusion matrix
      skplt.metrics.plot_confusion_matrix(
          y_true, y_pred, normalize=True
      )

   **When to use it**

   Always — it is the foundation for every derived metric.  Inspect
   the raw counts before trusting any single-number summary.

.. _terminology-tp-tn-fp-fn:

.. dropdown:: True Positive · True Negative · False Positive · False Negative
   :color: primary
   :icon: check-circle

   **Definitions**

   These four quantities are the atoms of classification evaluation.

   .. list-table::
      :header-rows: 1
      :widths: 15 42 43

      * - Symbol
        - Intuition
        - Domain example (disease screening)
      * - **TP**
        - Correct positive detection
        - Test says "sick", patient is sick ✅
      * - **TN**
        - Correct negative detection
        - Test says "healthy", patient is healthy ✅
      * - **FP**
        - False alarm
        - Test says "sick", patient is healthy ❌
      * - **FN**
        - Missed detection
        - Test says "healthy", patient is sick ❌

   The cost of FP and FN is **domain-specific** — in fraud detection,
   FN (missed fraud) is often far costlier than FP (flagging a
   legitimate transaction).  Always decide which error is worse *before*
   selecting a threshold.

.. dropdown:: Accuracy
   :color: primary
   :icon: check

   **Formula**

   .. math::

      \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

   **Intuition** — the fraction of all predictions that are correct.

   **When NOT to use it**

   Accuracy is misleading for **imbalanced datasets**.  A model that
   predicts the majority class for every sample achieves high accuracy
   while being completely useless.  Prefer F1, balanced accuracy,
   AUROC, or class-specific metrics when class frequencies differ
   substantially.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import accuracy_score, balanced_accuracy_score

      acc    = accuracy_score(y_true, y_pred)
      b_acc  = balanced_accuracy_score(y_true, y_pred)  # better for imbalance

.. dropdown:: Precision (Positive Predictive Value)
   :color: primary
   :icon: goal
   :name: terminology-precision-recall

   **Formula**

   .. math::

      \text{Precision} = \frac{TP}{TP + FP}

   **Intuition** — "Of everything I labelled positive, what fraction
   really was positive?"  High precision means few false alarms.

   **Trade-off** — raising the classification threshold raises precision
   but typically lowers recall.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import precision_score

      # binary
      p = precision_score(y_true, y_pred)
      # multiclass (macro)
      p = precision_score(y_true, y_pred, average='macro')

.. dropdown:: Recall (Sensitivity · True Positive Rate)
   :color: primary
   :icon: search

   **Formula**

   .. math::

      \text{Recall} = \frac{TP}{TP + FN}

   **Aliases** — Sensitivity, True Positive Rate (TPR), Hit Rate.

   **Intuition** — "Of everything that really was positive, what
   fraction did I catch?"  High recall means few missed detections.

   **Trade-off** — lowering the threshold increases recall but
   typically reduces precision (more false alarms).

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import recall_score

      r = recall_score(y_true, y_pred)
      r = recall_score(y_true, y_pred, average='macro')  # multiclass

.. dropdown:: Specificity (True Negative Rate)
   :color: primary
   :icon: shield-check

   **Formula**

   .. math::

      \text{Specificity} = \frac{TN}{TN + FP} = 1 - \text{FPR}

   **Intuition** — "Of everything that really was negative, what
   fraction did I correctly label as negative?"  Closely related to
   the x-axis of the ROC curve.

   .. note::

      ``scikit-learn`` does not have a standalone ``specificity_score``.
      Use ``recall_score(y_true, y_pred, pos_label=0)`` to compute it
      for the negative class in a binary problem.

.. dropdown:: F1 Score
   :color: primary
   :icon: graph
   :name: terminology-f-scores

   **Formula**

   .. math::

      F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}
                         {\text{Precision} + \text{Recall}}

   The F1 is the **harmonic mean** of precision and recall.  The
   harmonic mean penalises extreme imbalances between the two: a
   model with precision = 1.0 and recall = 0.0 gets F1 = 0.0.

   **When to use it** — when both false positives and false negatives
   matter, and you want a single summary number.

   **When NOT to use it** — when true negatives matter (e.g. spam
   filtering, where correctly rejecting spam is also valuable).
   Use the Matthews Correlation Coefficient (MCC) instead.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import f1_score

      f1 = f1_score(y_true, y_pred)
      f1 = f1_score(y_true, y_pred, average='macro')  # multiclass

.. dropdown:: F-beta Score
   :color: primary
   :icon: sliders

   **Formula**

   .. math::

      F_\beta = (1 + \beta^2) \cdot
                \frac{\text{Precision} \cdot \text{Recall}}
                     {(\beta^2 \cdot \text{Precision}) + \text{Recall}}

   The scalar :math:`\beta` controls the relative weight of recall:

   * :math:`\beta < 1` → precision-heavy  (F0.5 emphasises precision)
   * :math:`\beta = 1` → balanced F1
   * :math:`\beta > 1` → recall-heavy  (F2 emphasises recall)

   **Use case** — information retrieval, medical screening where
   missing a positive is far costlier than a false alarm.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import fbeta_score

      f2 = fbeta_score(y_true, y_pred, beta=2)

----------------------------------------------------------------------

.. _terminology-roc-auroc:

Domain 2 — ROC Curve & AUROC
----------------------------------------------------------------------

The Receiver Operating Characteristic curve and the Area Under it are
the standard threshold-independent evaluation framework for
classification models.

.. dropdown:: ROC Curve
   :color: secondary
   :icon: graph
   :open:

   **What is it?**

   A plot of **True Positive Rate (Recall)** on the y-axis against
   **False Positive Rate (1 − Specificity)** on the x-axis, traced
   by sweeping the classification threshold from 1.0 down to 0.0.

   * Bottom-left corner (0, 0) — threshold = 1.0, predict everything
     negative.
   * Top-right corner (1, 1) — threshold = 0.0, predict everything
     positive.
   * **Diagonal** — random classifier (AUROC = 0.5).
   * **Top-left corner** (0, 1) — perfect classifier (AUROC = 1.0).

   **Key insight** — the shape of the curve exposes the trade-off
   across *all* thresholds simultaneously, so you can pick the right
   operating point for your application *after* training.

   **scikit-plots**

   .. code-block:: python

      import scikitplot as skplt

      skplt.metrics.plot_roc(
          y_true,
          y_probas,       # shape (n_samples, n_classes)
          plot_macro=True,
          plot_micro=True,
      )

.. dropdown:: AUROC (Area Under the ROC Curve)
   :color: secondary
   :icon: meter

   **Formula**

   The AUROC equals the **probability** that the model assigns a
   higher score to a randomly chosen positive example than to a
   randomly chosen negative example:

   .. math::

      \text{AUROC} = P\!\bigl(\hat{s}_{\text{pos}} > \hat{s}_{\text{neg}}\bigr)

   This interpretation, due to Bamber (1975), makes AUROC a purely
   **rank-based** measure — it is invariant to monotone score
   transformations.

   **Interpretation scale**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - AUROC
        - Interpretation
      * - 0.50
        - Random guessing — no discriminative power
      * - 0.70 – 0.80
        - Acceptable discrimination
      * - 0.80 – 0.90
        - Good discrimination
      * - 0.90 – 1.00
        - Excellent / near-perfect discrimination
      * - 1.00
        - Perfect ranking (no real problem is this clean)

   **Limitation** — AUROC can be optimistic on highly imbalanced
   datasets because it counts TN (the large negative class) heavily.
   Consider the PR-AUC in that scenario.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import roc_auc_score

      # binary
      auc = roc_auc_score(y_true, y_score)
      # multiclass (macro OvR)
      auc = roc_auc_score(
          y_true, y_score_matrix,
          multi_class='ovr', average='macro'
      )

.. dropdown:: Precision-Recall Curve & PR-AUC
   :color: secondary
   :icon: graph

   **What is it?**

   A plot of **Precision** on the y-axis against **Recall** on the
   x-axis as the threshold varies.  The **Average Precision (AP)**,
   or the area under this curve, summarises performance.

   **When to prefer PR-AUC over AUROC**

   In severely imbalanced datasets (e.g., rare-event detection,
   fraud, medical screening where positives are < 5 % of data),
   the PR curve exposes model weaknesses that the ROC curve can hide —
   because the ROC curve's x-axis (FPR) is diluted by the enormous
   TN pool.

   **scikit-plots**

   .. code-block:: python

      skplt.metrics.plot_precision_recall(
          y_true, y_probas, plot_micro=True
      )

.. dropdown:: Gini Coefficient (in ML context)
   :color: secondary
   :icon: number
   :name: terminology-gini

   **Relationship to AUROC**

   In machine learning (particularly credit scoring and finance),
   the Gini coefficient is defined as:

   .. math::

      \text{Gini} = 2 \cdot \text{AUROC} - 1

   It maps the AUROC from the range [0.5, 1.0] to the range [0.0, 1.0]:

   .. list-table::
      :header-rows: 1
      :widths: 30 30 40

      * - AUROC
        - Gini
        - Interpretation
      * - 0.50
        - 0.00
        - Random
      * - 0.75
        - 0.50
        - Good
      * - 1.00
        - 1.00
        - Perfect

   **When you will see it** — credit risk models, insurance, any
   domain that adopted the Gini metric before AUROC became standard.
   The two metrics are equivalent information.

----------------------------------------------------------------------

.. _terminology-averaging:
.. _terminology-multiclass-auroc:

Domain 3 — Averaging Strategies & Multiclass Metrics
----------------------------------------------------------------------

When K > 2, every binary metric must be extended to the multi-class
case.  The choice of averaging strategy changes the answer.

.. dropdown:: Binary · Multiclass · Multi-label Classification
   :color: info
   :icon: tag
   :open:
   :name: terminology-classification-types

   .. tab-set::

      .. tab-item:: Binary

         **Definition** — exactly two mutually exclusive classes:
         positive (1) and negative (0).

         **Output** — a single probability score per sample, or a
         binary label.

         **Example** — spam vs. not-spam; disease positive vs. negative.

      .. tab-item:: Multiclass (Single-label)

         **Definition** — K > 2 mutually exclusive classes.  Each
         sample belongs to exactly one class.

         **Output** — a probability vector of length K; the argmax
         is the predicted class.

         **Example** — MNIST digit recognition (0–9); flower species
         classification (Iris dataset).

         **Also called** — single-label classification, multi-class
         classification.

      .. tab-item:: Multi-label

         **Definition** — each sample can belong to *multiple* classes
         simultaneously (classes are not mutually exclusive).

         **Output** — a binary indicator vector of length K per sample.

         **Example** — image tagging (a photo can be "dog", "outdoor",
         "daytime" simultaneously); document topic labelling.

         **Key difference** — the per-class AUROC and F1 are computed
         independently for each label.

.. dropdown:: Macro Averaging
   :color: info
   :icon: columns

   **Definition**

   Compute the metric separately for each class, then take the
   **unweighted (simple) mean** across all K classes:

   .. math::

      \text{Metric}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Metric}_i

   **Properties**

   * Each class contributes **equally**, regardless of how many
     samples it has.
   * Sensitive to performance on **minority classes** — poor
     discrimination on a rare class lowers the macro score noticeably.
   * Best choice when you care about **per-class fairness**.

   **Example (Macro F1)**

   .. code-block:: python

      from sklearn.metrics import f1_score

      f1_macro = f1_score(y_true, y_pred, average='macro')

.. dropdown:: Micro Averaging
   :color: info
   :icon: rows

   **Definition**

   Aggregate the TP, TN, FP, FN counts across *all* classes into
   global totals, then compute the metric once from those totals:

   .. math::

      \text{Precision}_{\text{micro}} =
      \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)}

   **Properties**

   * Dominated by **majority classes** because large classes
     contribute more counts.
   * Equivalent to accuracy for multiclass single-label problems.
   * Best choice when overall sample-level performance matters and
     class size differences are acceptable.

   **Example (Micro F1)**

   .. code-block:: python

      from sklearn.metrics import f1_score

      f1_micro = f1_score(y_true, y_pred, average='micro')

.. dropdown:: Weighted Averaging
   :color: info
   :icon: sort-asc

   **Definition**

   Compute the metric per class, then take a **weighted mean** where
   each class's weight equals its proportion of samples (support):

   .. math::

      \text{Metric}_{\text{weighted}} =
      \sum_{i=1}^{K} w_i \cdot \text{Metric}_i,
      \quad w_i = \frac{n_i}{\sum_j n_j}

   **Properties**

   * Accounts for class imbalance without completely ignoring minority
     classes (unlike micro).
   * The default in many scikit-learn reports.
   * Can still mask poor minority-class performance if the majority
     class is large enough.

   **Example**

   .. code-block:: python

      from sklearn.metrics import f1_score

      f1_weighted = f1_score(y_true, y_pred, average='weighted')

.. dropdown:: Macro AUROC (Macro-Averaged AUROC)
   :color: info
   :icon: graph

   **Definition**

   Extend binary AUROC to K classes using the **One-vs-Rest (OvR)**
   strategy, then average:

   .. math::

      \text{AUROC}_{\text{macro}} = \frac{1}{K}
      \sum_{i=1}^{K} \text{AUROC}(\text{class}_i \text{ vs. rest})

   **Example computation**

   For a 3-class problem (A, B, C):

   .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - Binary AUROC
        - Value
      * - AUROC(A vs. B+C)
        - 0.85
      * - AUROC(B vs. A+C)
        - 0.72
      * - AUROC(C vs. A+B)
        - 0.65

   .. math::

      \text{AUROC}_{\text{macro}} = \frac{0.85 + 0.72 + 0.65}{3} = 0.74

   If class C is rare but performs poorly, Macro AUROC reflects this
   because every class has equal weight.

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import roc_auc_score

      auc = roc_auc_score(
          y_true, y_probas, multi_class='ovr', average='macro'
      )

.. dropdown:: Micro AUROC
   :color: info
   :icon: graph

   **Definition**

   Flatten all one-vs-rest binary predictions into a single long
   vector of true labels and scores, then compute a single AUROC:

   .. math::

      \text{AUROC}_{\text{micro}} =
      \text{AUROC}\!\left(\,
        \bigoplus_{i=1}^{K} y_i^{(\text{bin})},\;
        \bigoplus_{i=1}^{K} \hat{s}_i
      \right)

   **Properties**

   * Heavily influenced by **majority classes** (more samples → more
     weight).
   * Provides an overall view of ranking quality across all decisions.
   * Can look good even when rare classes are poorly ranked.

   **Macro vs. Micro — when to use which**

   .. list-table::
      :header-rows: 1
      :widths: 35 65

      * - Situation
        - Recommended
      * - Classes should be treated equally
        - **Macro AUROC**
      * - Overall sample-level ranking matters
        - **Micro AUROC**
      * - Imbalanced and every class matters
        - **Macro AUROC** (reveals minority weaknesses)
      * - Class sizes are acceptable to weight
        - Weighted AUROC

.. dropdown:: One-vs-Rest (OvR) and One-vs-One (OvO)
   :color: info
   :icon: arrow-both
   :name: terminology-ovr-ovo

   **One-vs-Rest (OvR)** — also called One-vs-All (OvA)

   For K classes, train K binary classifiers.  Classifier *i* treats
   class *i* as the positive class and all others as the negative class.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Property
        - OvR
      * - Number of classifiers
        - K
      * - Training set size per classifier
        - Full dataset (imbalanced: 1 positive class vs. K-1 negatives)
      * - Prediction
        - Argmax of K confidence scores
      * - AUROC computation
        - Average of K binary AUROCs

   **One-vs-One (OvO)**

   Train one binary classifier for every pair of classes.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Property
        - OvO
      * - Number of classifiers
        - K(K−1)/2
      * - Training set size per classifier
        - Only the two relevant classes (balanced)
      * - Prediction
        - Majority vote over all pairwise classifiers
      * - AUROC computation
        - Average of all pairwise AUROCs (Hand & Till, 2001)

   **scikit-learn**

   .. code-block:: python

      from sklearn.metrics import roc_auc_score

      # OvR macro
      auc_ovr = roc_auc_score(
          y_true, y_probas, multi_class='ovr', average='macro'
      )
      # OvO macro (Hand & Till)
      auc_ovo = roc_auc_score(
          y_true, y_probas, multi_class='ovo', average='macro'
      )

----------------------------------------------------------------------

.. _terminology-data-imbalance:

Domain 4 — Class Imbalance & Sampling Strategies
----------------------------------------------------------------------

Most real-world classification datasets are imbalanced.  The severity
ranges from mildly unequal class frequencies (1:2) to extreme
imbalance (fraud: 1:10 000).  The response strategies fall into
three groups: **re-weighting**, **oversampling**, and **undersampling**.

.. dropdown:: Class Imbalance — Overview
   :color: warning
   :icon: alert
   :open:

   **Definition**

   A dataset is *imbalanced* when the class frequencies differ
   substantially — typically taken as a ratio exceeding 1:5 (minority :
   majority).

   **Why it matters for metrics**

   A classifier that predicts the majority class for every sample
   achieves misleadingly high accuracy.  Standard metrics (accuracy,
   Macro F1) can therefore be poor guides.

   **Summary of strategies**

   .. list-table::
      :header-rows: 1
      :widths: 30 35 35

      * - Strategy
        - Mechanism
        - Best when
      * - Class weighting
        - Penalise majority class errors more
        - Small to moderate imbalance
      * - Random oversampling
        - Duplicate minority samples
        - Quick baseline
      * - SMOTE
        - Synthesise minority samples
        - Feature-space interpolation is valid
      * - Random undersampling
        - Remove majority samples
        - Very large majority class
      * - NearMiss
        - Keep majority samples closest to minority
        - Hard-boundary learning
      * - Cluster-based undersampling
        - Keep one majority representative per cluster
        - Structured majority class

.. dropdown:: Class Weighting
   :color: warning
   :icon: law

   **Mechanism**

   Assign each sample a loss weight inversely proportional to its
   class frequency, so that minority-class errors are penalised more
   heavily during training:

   .. math::

      w_i = \frac{n_{\text{total}}}{K \cdot n_i}

   where :math:`n_i` is the count of class *i*.

   **Advantages**

   * No data is discarded or synthesised — uses the original
     distribution.
   * Straightforward — most scikit-learn estimators accept
     ``class_weight='balanced'``.

   **Limitations** — does not change the decision boundary in the
   input space; only adjusts the training loss.

   **scikit-learn**

   .. code-block:: python

      from sklearn.linear_model import LogisticRegression
      from sklearn.utils.class_weight import compute_class_weight
      import numpy as np

      weights = compute_class_weight(
          'balanced', classes=np.unique(y_train), y=y_train
      )
      model = LogisticRegression(class_weight='balanced')

.. dropdown:: Oversampling
   :color: warning
   :icon: arrow-up
   :name: terminology-sampling

   **Mechanism**

   Increase the size of the minority class by generating additional
   samples — either by **random duplication** or **synthetic
   generation** (SMOTE).

   **Random oversampling** — duplicate existing minority samples with
   replacement until the desired ratio is reached.  Risk: overfitting
   to duplicated points.

   **When to use** — when the minority class is too small to learn
   meaningful boundaries.  Always apply oversampling only to the
   training set, never to validation or test sets.

.. dropdown:: SMOTE (Synthetic Minority Over-sampling Technique)
   :color: warning
   :icon: sparkle-fill

   **Definition**

   SMOTE generates **synthetic** minority-class samples by
   interpolating in feature space between an existing minority sample
   and one of its k nearest minority neighbours:

   .. math::

      x_{\text{new}} = x_i + \lambda \cdot (x_{\tilde{\text{nn}}} - x_i),
      \quad \lambda \sim \mathcal{U}(0, 1)

   **Properties**

   * Creates genuinely new points (not copies) → less overfitting
     than random oversampling.
   * Can create noisy samples in overlapping class regions.
   * Assumes that interpolation in feature space is meaningful
     (invalid for categorical features without encoding).

   **Library**

   .. code-block:: python

      from imblearn.over_sampling import SMOTE

      sm = SMOTE(k_neighbors=5, random_state=42)
      X_res, y_res = sm.fit_resample(X_train, y_train)

   .. note::

      ``imbalanced-learn`` (``imblearn``) is the standard library for
      SMOTE and related techniques.  It is not part of scikit-learn but
      shares the same API.

.. dropdown:: Undersampling Strategies
   :color: warning
   :icon: arrow-down

   Undersampling reduces the majority class to match the minority.

   .. tab-set::

      .. tab-item:: Random Undersampling

         **Mechanism** — randomly remove majority-class samples.

         **Risk** — discards potentially useful information from the
         majority class.  Use stratified splits to preserve class
         proportions in validation.

         .. code-block:: python

            from imblearn.under_sampling import RandomUnderSampler

            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)

      .. tab-item:: NearMiss

         **Mechanism** — keep majority-class samples that are *closest*
         to minority-class samples (distance-based selection).

         * **NearMiss-1** — select majority samples whose average
           distance to the *nearest* minority neighbours is smallest.
         * **NearMiss-2** — select majority samples whose average
           distance to the *farthest* minority neighbours is smallest.
         * **NearMiss-3** — for each minority sample, keep its M nearest
           majority neighbours.

         **When to use** — when you want the majority class to
         concentrate near the decision boundary (hard-margin learning).

         .. code-block:: python

            from imblearn.under_sampling import NearMiss

            nm = NearMiss(version=1)
            X_res, y_res = nm.fit_resample(X_train, y_train)

      .. tab-item:: Cluster-based Undersampling

         **Mechanism** — cluster the majority class with k-means (or
         another algorithm), then retain one representative per cluster
         (typically the centroid or the sample closest to it).

         **Advantages**

         * Preserves the structure of the majority class.
         * Less information loss than random removal.
         * Works well when the majority class has natural sub-groups.

.. dropdown:: Subsampling
   :color: warning
   :icon: diff-removed

   **Dual meaning** — "subsampling" appears in two completely different
   contexts:

   .. tab-set::

      .. tab-item:: In Machine Learning

         Selecting a random subset of the dataset for training or
         evaluation — analogous to random undersampling, but often
         applied for **computational efficiency** rather than class
         balancing.

         .. code-block:: python

            from sklearn.utils import resample

            X_sub, y_sub = resample(
                X, y, n_samples=10_000, random_state=42
            )

      .. tab-item:: In Signal Processing / DSP

         **Reducing the sampling rate** of a discrete signal.  Also
         called **decimation** or **downsampling**.

         **Critical rule** — always apply a **low-pass filter before
         subsampling** to prevent aliasing (see
         :ref:`terminology-aliasing`).

         .. code-block:: python

            import scipy.signal as sps

            # Decimate signal by factor 4 (includes anti-aliasing filter)
            signal_down = sps.decimate(signal, q=4)

----------------------------------------------------------------------

.. _terminology-statistics:

Domain 5 — Statistical Foundations
----------------------------------------------------------------------

.. dropdown:: Probability & Probability Distributions
   :color: success
   :icon: number
   :open:

   **Probability** — a number in [0, 1] expressing the likelihood of
   an event.  A model's output score is a *probability estimate*
   (not necessarily a calibrated probability — see
   :ref:`terminology-calibration`).

   **Key distributions in ML**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Distribution
        - Role in ML
      * - Bernoulli
        - Binary label; output of a binary classifier
      * - Categorical
        - Multiclass label; output of a softmax classifier
      * - Gaussian (Normal)
        - Assumption in linear discriminant analysis, GPs
      * - Beta
        - Prior / posterior for probabilities (Bayesian)
      * - Dirichlet
        - Prior over class probability vectors (Bayesian multiclass)

.. dropdown:: Bootstrap Confidence Intervals
   :color: success
   :icon: graph
   :name: terminology-bootstrap

   **What is it?**

   A non-parametric resampling method to estimate the sampling
   distribution — and hence confidence intervals — of *any* statistic.

   **Algorithm**

   1. Draw *B* bootstrap samples of size *n* with replacement from
      the original dataset.
   2. Compute the statistic (e.g., AUROC, F1) on each bootstrap sample.
   3. The 95 % CI is the (2.5th, 97.5th) percentile of the *B*
      computed statistics.

   **Why it matters for scikit-plots** — every metric visualised by
   scikit-plots can be accompanied by a bootstrap CI to quantify
   uncertainty.

   .. code-block:: python

      import numpy as np
      from sklearn.metrics import roc_auc_score

      rng = np.random.default_rng(42)
      B   = 1000
      aucs = [
          roc_auc_score(
              y_true[idx := rng.integers(len(y_true), size=len(y_true))],
              y_score[idx]
          )
          for _ in range(B)
      ]
      ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
      print(f"AUROC 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

.. dropdown:: Mann-Whitney U Test (Wilcoxon Rank-Sum Test)
   :color: success
   :icon: pulse

   **What is it?**

   A non-parametric test for whether two independent samples were
   drawn from the same distribution.  It makes no assumptions about
   the underlying distribution (no normality required).

   **Relationship to AUROC** — remarkably, the Mann-Whitney U
   statistic is algebraically equivalent to the AUROC:

   .. math::

      \text{AUROC} = \frac{U}{n_{\text{pos}} \cdot n_{\text{neg}}}

   where :math:`U` is the Mann-Whitney statistic.  This confirms the
   AUROC's rank-statistic interpretation.

   **Use in model comparison** — test whether model A's AUROC is
   significantly greater than model B's using a permutation test or
   DeLong's method.

   .. code-block:: python

      from scipy.stats import mannwhitneyu

      stat, p_value = mannwhitneyu(
          scores_positive, scores_negative, alternative='greater'
      )

----------------------------------------------------------------------

.. _terminology-fairness:
.. _terminology-imbalance-fairness:

Domain 6 — Fairness & Bias Metrics
----------------------------------------------------------------------

Fairness metrics quantify whether a model treats different demographic
groups equitably.  **No single definition of fairness is universally
correct** — the appropriate criterion depends on the application and
its societal context.

.. dropdown:: Demographic Parity (Statistical Parity)
   :color: danger
   :icon: people
   :open:

   **Definition**

   The positive prediction rate is equal across all demographic
   groups *A* and *B*:

   .. math::

      P(\hat{Y}=1 \mid A) = P(\hat{Y}=1 \mid B)

   **Interpretation** — the selection rate (e.g., loan approval,
   job interview, parole) is independent of group membership.

   **Limitation** — if the true positive rate differs between groups
   (due to legitimate factors), forcing equal selection rates may
   require ignoring relevant features.

   **When to apply** — allocation decisions where equal access is
   the primary concern (e.g., advertising, content recommendation).

.. dropdown:: Equal Opportunity
   :color: danger
   :icon: verified

   **Definition**

   The **True Positive Rate (Recall)** is equal across groups:

   .. math::

      P(\hat{Y}=1 \mid Y=1, A) = P(\hat{Y}=1 \mid Y=1, B)

   **Interpretation** — among truly qualified/positive individuals,
   the model identifies them at the same rate regardless of group.

   **Use case** — hiring, academic admission, loan approval —
   where it is critical that deserving applicants are equally
   detected across groups.

.. dropdown:: Equalized Odds
   :color: danger
   :icon: x-circle

   **Definition**

   Both the **True Positive Rate** and the **False Positive Rate**
   are equal across groups:

   .. math::

      P(\hat{Y}=1 \mid Y=y, A) = P(\hat{Y}=1 \mid Y=y, B),
      \quad \text{for } y \in \{0, 1\}

   **Interpretation** — a stronger requirement than equal opportunity:
   the model must treat both positive and negative individuals equally
   across groups.

   **Trade-off** — equalized odds, equal opportunity, and demographic
   parity are mathematically incompatible in general (except in
   degenerate cases).  You must choose which criterion fits the
   application.

.. dropdown:: Predictive Parity (Calibration Fairness)
   :color: danger
   :icon: goal

   **Definition**

   The **Positive Predictive Value (Precision)** is equal across
   groups:

   .. math::

      P(Y=1 \mid \hat{Y}=1, A) = P(Y=1 \mid \hat{Y}=1, B)

   **Interpretation** — when the model predicts "positive" for a
   member of either group, the prediction is equally trustworthy.

   **Use case** — risk scoring tools (recidivism, credit risk) where
   the model score is used as a probability estimate and must be
   equally reliable for all groups.

   .. note::

      Predictive parity and equalized odds cannot both be satisfied
      simultaneously unless base rates are equal across groups
      (the Chouldechova impossibility result, 2017).

----------------------------------------------------------------------

.. _terminology-calibration:

Domain 7 — Calibration
----------------------------------------------------------------------

A model is *calibrated* if its output probabilities match empirical
event frequencies.  Calibration is independent of discrimination
(AUROC): a model can rank perfectly but be poorly calibrated, or be
well-calibrated but with low AUROC.

.. dropdown:: Calibration & Reliability Diagrams
   :color: secondary
   :icon: goal
   :open:

   **What is it?**

   A **reliability diagram** (calibration curve) plots the mean
   predicted probability (x-axis) against the observed event rate
   (y-axis) for bins of predictions.  A perfectly calibrated model
   lies on the diagonal y = x.

   * **Over-confident** — predicted probabilities are too high
     (curve below the diagonal).
   * **Under-confident** — predicted probabilities are too low
     (curve above the diagonal).

   **Brier Score**

   A calibration-sensitive proper scoring rule:

   .. math::

      \text{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2

   Lower is better; 0 = perfect; 0.25 = random for a balanced binary
   problem.

   **scikit-plots**

   .. code-block:: python

      skplt.metrics.plot_calibration_curve(
          y_true,
          [y_prob_model_1, y_prob_model_2],
          clf_names=['Model 1', 'Model 2'],
          n_bins=10,
      )

   **Post-hoc calibration**

   .. code-block:: python

      from sklearn.calibration import CalibratedClassifierCV

      cal = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
      cal.fit(X_train, y_train)
      cal_probs = cal.predict_proba(X_test)

----------------------------------------------------------------------

.. _terminology-signal-timeseries:

Domain 8 — Signal Processing & Time Series
----------------------------------------------------------------------

Relevant when scikit-plots is used to evaluate models applied to
sequential or temporal data.

.. dropdown:: Time Series
   :color: secondary
   :icon: clock
   :open:

   **Definition**

   A sequence of observations ordered in time,
   :math:`\{x_t\}_{t=1}^{T}`.  Key properties:

   * **Stationarity** — statistical properties (mean, variance) do
     not change over time.
   * **Autocorrelation** — observations at time *t* are correlated
     with observations at :math:`t - k` (lag *k*).
   * **Seasonality** — periodic patterns (daily, weekly, yearly).

   **Evaluation difference from i.i.d. data**

   Evaluating ML models on time series requires **temporal
   cross-validation** (walk-forward validation), not random K-fold,
   to avoid data leakage from the future into the training set.

   .. code-block:: python

      from sklearn.model_selection import TimeSeriesSplit

      tscv = TimeSeriesSplit(n_splits=5)
      for train_idx, test_idx in tscv.split(X):
          X_tr, X_te = X[train_idx], X[test_idx]

.. dropdown:: Signal Processing
   :color: secondary
   :icon: broadcast

   **Definition**

   The analysis, synthesis, and transformation of signals —
   including audio, sensor data, EEG, accelerometers, financial
   tick data.

   **Key concepts for ML practitioners**

   * **Sampling rate** :math:`f_s` — how many samples per second.
   * **Nyquist frequency** :math:`f_N = f_s / 2` — the highest
     frequency representable without aliasing.
   * **Frequency domain** — the Fourier transform converts a time-
     domain signal to its frequency components.

.. dropdown:: Low-pass Filtering
   :color: secondary
   :icon: filter

   **Definition**

   A filter that attenuates frequencies above a cut-off frequency
   :math:`f_c` and passes frequencies below it.

   **Why it is required before subsampling**

   If you reduce the sampling rate by factor *d* without filtering,
   frequency components above :math:`f_s / (2d)` fold back into the
   representable range — this is **aliasing** and causes irreversible
   distortion.

   **scikit-learn / scipy**

   .. code-block:: python

      import scipy.signal as sps

      # Design a Butterworth low-pass filter
      b, a = sps.butter(N=4, Wn=0.25, btype='low')
      filtered = sps.filtfilt(b, a, signal)

      # Decimate (apply anti-alias filter + downsample)
      downsampled = sps.decimate(signal, q=4)

.. dropdown:: Aliasing & the Nyquist-Shannon Theorem
   :color: secondary
   :icon: alert-fill
   :name: terminology-aliasing

   **Nyquist-Shannon Sampling Theorem**

   A continuous signal that has no frequency component above
   :math:`f_{\max}` can be perfectly reconstructed from its samples
   if and only if the sampling rate satisfies:

   .. math::

      f_s \geq 2 f_{\max}

   **Aliasing** — when :math:`f_s < 2 f_{\max}`, high-frequency
   components "fold" into lower-frequency aliases, creating distortion
   that cannot be undone post-hoc.

   **Practical rule**

   Before any downsampling by factor *d*, apply a low-pass filter
   with cut-off :math:`f_c \leq f_s / (2d)`.  The ``scipy.signal.decimate``
   function does this automatically.

----------------------------------------------------------------------

.. _terminology-quick-reference:

Quick Reference — Metric Selector
----------------------------------------------------------------------

Use this table to choose the right metric for your problem:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Situation
     - Avoid
     - Use Instead
     - If Multiclass
     - If Imbalanced
   * - Balanced binary classification
     - —
     - F1, AUROC
     - Macro F1, OvR AUROC
     - Macro F1
   * - Severely imbalanced binary
     - Accuracy
     - PR-AUC, F1
     - Macro F1
     - PR-AUC
   * - All classes equally important
     - Micro avg.
     - **Macro avg.**
     - Macro AUROC
     - Macro AUROC
   * - Overall sample-level performance
     - Macro avg.
     - **Micro avg.**
     - Micro F1
     - Check macro too
   * - Probability quality (not just ranking)
     - AUROC alone
     - AUROC + Brier Score
     - Calibration curve
     - Brier Score
   * - Fairness audit required
     - Global accuracy
     - Group-level TPR/FPR
     - Equal Opportunity
     - Equalized Odds

----------------------------------------------------------------------

.. _terminology-sources:

Sources
----------------------------------------------------------------------

The following sources were consulted in preparing this page.
All links were verified as of the documentation build date.

**Core API & Framework Documentation**

* scikit-learn — ``sklearn.metrics`` module reference:
  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

* imbalanced-learn — SMOTE and sampling API reference:
  https://imbalanced-learn.org/stable/references/index.html

* SciPy — ``scipy.signal`` for digital signal processing:
  https://docs.scipy.org/doc/scipy/reference/signal.html

**Authoritative Papers & Textbooks**

* Fawcett, T. (2006). An introduction to ROC analysis.
  *Pattern Recognition Letters*, 27(8), 861–874.
  https://doi.org/10.1016/j.patrec.2005.10.010

* Bamber, D. (1975). The area above the ordinal dominance graph and the
  area below the receiver operating characteristic graph.
  *Journal of Mathematical Psychology*, 12(4), 387–415.
  https://doi.org/10.1016/0022-2496(75)90001-2

* Hand, D. J., & Till, R. J. (2001). A simple generalisation of the
  area under the ROC curve for multiple class classification problems.
  *Machine Learning*, 45(2), 171–186.
  https://doi.org/10.1023/A:1010920819831

* Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling
  Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
  https://doi.org/10.1613/jair.953

* Chouldechova, A. (2017). Fair prediction with disparate impact.
  *Big Data*, 5(2), 153–163.
  https://doi.org/10.1089/big.2016.0047

* Shannon, C. E. (1949). Communication in the presence of noise.
  *Proceedings of the IRE*, 37(1), 10–21.
  https://doi.org/10.1109/jrproc.1949.232969

**Learning Resources**

* Google Machine Learning Crash Course — Classification:
  https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

* insightful-data-lab.com — Terminology category (source of context
  and domain framing for this page):
  https://insightful-data-lab.com/category/00terminology/

* scikit-plots documentation — Metrics API:
  https://scikit-plots.github.io/dev/apis/index.html

..
   ##################################################################
   Page tags — placed at the bottom per the project tagging guide
   (https://scikit-plots.github.io/dev/devel/guide_document_tag.html).
   Uses only the existing controlled vocabulary shown on the tag
   index (https://scikit-plots.github.io/dev/_tags/tagsindex.html).
   ##################################################################

.. tags::
   purpose: reference,
   domain: statistics,
   model-type: classification,
   level: beginner,
   level: intermediate,
   level: advanced
