:html_theme.sidebar_secondary.remove:

..
   ##################################################################
   learn/data_preparation_and_analysis/index.rst
   ==================================================================
   Predictive-modelling & evaluation hub, framed for scikit-plots
   users.  Source context (framing only, re-expressed here):
   https://insightful-data-lab.com/category/data-preparation-and-analysis/ (56 posts)
   ------------------------------------------------------------------
   Extensions relied upon (all enabled in conf.py):
     sphinx_design  (tab-set, grid, grid-item-card, dropdown, badge)
     sphinx_tags    (tag directive — placed at page bottom)
     sphinx_copybutton, sphinx_togglebutton
   Section underline convention (mirrors the project):
     = section (with overline)   - subsection   ^ subsubsection
   ##################################################################

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

This hub covers the **applied predictive-modelling workflow**: framing a
prediction problem, fitting a model, and — most importantly for
**scikit-plots** — *evaluating* it with the right chart for the right
question. It is the practical companion to the
:ref:`Terminology reference <terminology-index>`: terminology defines the
metrics, this page shows the workflow that produces and reads them.

It is written for three readers at once:

* **newcomers** who want the intuition behind model evaluation;
* **practitioners** choosing between ROC, lift, gains and threshold tuning;
* **reviewers** who need diagnostics (residuals, outliers) before shipping.

.. note::

   Detail is collapsed by default. Open the dropdown for a term, follow
   the **See also** cross-links to wander related ideas, and use
   :kbd:`Ctrl` + :kbd:`F` or the Sphinx search to jump straight to a topic.
   Every code snippet uses a real ``scikitplot`` / ``scikit-learn`` call.

----------------------------------------------------------------------

.. _data-prep-discovery:

Discovery at a Glance
----------------------------------------------------------------------

.. tab-set::
   :class: sd-width-100

   .. tab-item:: 🟢 Start Here — Foundations
      :sync: level-foundations

      What a predictive model is, and what "good" means.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🧭 What is a Prediction Model?
            :link: data-prep-prediction-model
            :link-type: ref
            :class-card: sd-border-1

            Inputs → learned mapping → scored output, and the train /
            validate / test discipline that keeps it honest.

         .. grid-item-card:: ✔️ Assessing Model Quality
            :link: data-prep-model-quality
            :link-type: ref
            :class-card: sd-border-1

            Discrimination vs. calibration vs. business value — three
            different questions, three different checks.

         .. grid-item-card:: 🔀 Binary vs. Nominal Targets
            :link: data-prep-target-types
            :link-type: ref
            :class-card: sd-border-1

            How the target's shape (two classes vs. unordered many)
            changes which metrics apply.

   .. tab-item:: 🔵 Core — Evaluation
      :sync: level-core

      The everyday toolkit for scoring classifiers.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 📈 ROC & AUC
            :link: data-prep-roc-auc
            :link-type: ref
            :class-card: sd-border-1

            Ranking quality across every threshold at once — and how to
            plot it in scikit-plots.

         .. grid-item-card:: 🎚️ Threshold Optimization
            :link: data-prep-threshold
            :link-type: ref
            :class-card: sd-border-1

            Turning scores into decisions: choosing the cut-off that
            matches your cost trade-off.

         .. grid-item-card:: 📊 Gains, Lift & Deciles
            :link: data-prep-lift-gains
            :link-type: ref
            :class-card: sd-border-1

            "If I contact the top 20 %, how much better than random?" —
            the campaign manager's metric.

   .. tab-item:: 🔴 Advanced — Models & Diagnostics
      :sync: level-advanced

      Interpretable models and what to check before trusting them.

      .. grid:: 2 2 3 3
         :gutter: 2

         .. grid-item-card:: 🌳 Decision Trees & CART
            :link: data-prep-cart
            :link-type: ref
            :class-card: sd-border-1

            Piecewise, rule-based models that capture interactions and
            explain themselves.

         .. grid-item-card:: 🔬 Residuals & Outliers
            :link: data-prep-residuals
            :link-type: ref
            :class-card: sd-border-1

            Studentized residuals to find the points your model cannot
            explain.

         .. grid-item-card:: 🧩 Explaining Clusters
            :link: data-prep-explain-clusters
            :link-type: ref
            :class-card: sd-border-1

            Using a tree to turn an opaque clustering into human-readable
            rules.

----------------------------------------------------------------------

.. _data-prep-foundations:

Part 1 — Prediction Models & What "Good" Means
----------------------------------------------------------------------

Before any chart, fix the question: *what are we predicting, and how will
we know the model is any good?*

.. dropdown:: What is a Prediction Model?
   :color: primary
   :icon: workflow
   :name: data-prep-prediction-model
   :open:

   **What is it?**

   A prediction (or *supervised*) model learns a mapping from input
   features :math:`X` to a target :math:`y` from labelled examples, so
   that it can score **new, unseen** inputs. Classification predicts a
   category; regression predicts a number.

   **The honesty discipline**

   Performance must be measured on data the model never saw during
   fitting. The standard split is train → validation (for tuning) →
   test (for the final, untouched estimate):

   .. code-block:: python

      from sklearn.model_selection import train_test_split

      X_tr, X_tmp, y_tr, y_tmp = train_test_split(
          X, y, test_size=0.4, stratify=y, random_state=0
      )
      X_val, X_te, y_val, y_te = train_test_split(
          X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0
      )

   **When to use it** — any task where past labelled outcomes can guide
   future decisions (churn, fraud, response, risk).

   .. seealso::

      :ref:`data-prep-model-quality` · :ref:`terminology-classification-types`

.. dropdown:: Assessing the Quality of a Prediction Model
   :color: primary
   :icon: verified
   :name: data-prep-model-quality

   **Three independent questions**

   A model can be strong on one axis and weak on another, so check all
   three:

   * **Discrimination** — does it rank positives above negatives?
     (ROC-AUC, gains, lift).
   * **Calibration** — do predicted probabilities match observed
     frequencies? (reliability curve, Brier score).
   * **Business value** — does acting on it beat the baseline at *your*
     operating point? (lift at the contacted fraction, profit curve).

   **scikit-plots connection**

   .. code-block:: python

      import scikitplot as skplt

      # One call renders confusion matrix + ROC + PR for a quick read
      skplt.metrics.plot_classifier_eval(y_true, y_pred, y_probas)

   .. seealso::

      :ref:`data-prep-roc-auc` · :ref:`data-prep-lift-gains` ·
      :ref:`terminology-calibration`

.. dropdown:: Binary vs. Nominal (Multiclass) Targets
   :color: primary
   :icon: git-merge
   :name: data-prep-target-types

   **Binary** — exactly two outcomes (positive / negative). The full
   confusion-matrix vocabulary (TP/FP/FN/TN) and threshold tuning apply
   directly.

   **Nominal / multiclass** — three or more *unordered* categories. Each
   metric must be **averaged** across classes (macro / micro / weighted),
   and ROC-AUC is computed One-vs-Rest or One-vs-One.

   .. code-block:: python

      from sklearn.metrics import classification_report
      print(classification_report(y_true, y_pred))   # per-class + averages

   .. seealso::

      :ref:`terminology-averaging` · :ref:`terminology-multiclass-auroc`

----------------------------------------------------------------------

.. _data-prep-evaluation:

Part 2 — Classification Evaluation
----------------------------------------------------------------------

The core loop: score → rank → choose a threshold → read the trade-off.

.. dropdown:: ROC Curve & AUC
   :color: info
   :icon: graph
   :name: data-prep-roc-auc

   **What is it?**

   The ROC curve plots True Positive Rate against False Positive Rate as
   the decision threshold sweeps from 1 to 0. The **AUC** (area under it)
   summarises ranking quality in a single number:

   .. math::

      \text{AUC} = P\big(\hat{s}(x^{+}) > \hat{s}(x^{-})\big)

   i.e. the probability a random positive is scored above a random
   negative. 0.5 = random, 1.0 = perfect.

   **scikit-plots connection**

   .. code-block:: python

      import scikitplot as skplt
      skplt.metrics.plot_roc(y_true, y_probas)

   **When to use it** — ranking/threshold-free comparison. For
   **imbalanced** problems, read it alongside Precision–Recall / lift,
   which are more sensitive to the minority class.

   .. seealso::

      :ref:`terminology-roc-auroc` · :ref:`data-prep-lift-gains`

.. dropdown:: Threshold Optimization
   :color: info
   :icon: sliders
   :name: data-prep-threshold

   **The problem**

   A classifier outputs a *score*; a decision needs a *cut-off*. The
   default 0.5 is rarely optimal — the right threshold depends on the
   relative cost of false positives vs. false negatives.

   **A cost-aware choice**

   Pick the threshold :math:`t` that minimises expected cost:

   .. math::

      t^{\*} = \arg\min_{t}\; C_{FP}\,\text{FP}(t) + C_{FN}\,\text{FN}(t)

   .. code-block:: python

      import numpy as np
      from sklearn.metrics import precision_recall_curve

      prec, rec, thr = precision_recall_curve(y_true, y_score)
      f1 = 2 * prec * rec / (prec + rec + 1e-12)
      best_t = thr[np.nanargmax(f1[:-1])]   # threshold maximising F1

   **When to use it** — whenever a model's output drives an action with
   asymmetric consequences (medical screening, fraud holds, mailing cost).

   .. seealso::

      :ref:`terminology-precision-recall` · :ref:`data-prep-model-quality`

----------------------------------------------------------------------

.. _data-prep-business:

Part 3 — Gains, Lift & Decile Analysis
----------------------------------------------------------------------

The "how much better than random, at the fraction I can afford to act on"
family — a particular strength of scikit-plots' decile plots.

.. dropdown:: Cumulative Gains, Lift & Deciles
   :color: warning
   :icon: rocket
   :name: data-prep-lift-gains

   **What is it?**

   Rank all cases by predicted score, descending, and bin into **deciles**
   (top 10 %, next 10 %, …).

   * **Cumulative gains** — the share of all true positives captured by
     the top *k* % of ranked cases.
   * **Lift** — gains divided by the baseline (random) rate:

   .. math::

      \text{Lift}(k) = \frac{\text{response rate in top } k\%}
                            {\text{overall response rate}}

   A lift of 3 at the top decile means that group responds 3× more than
   average — exactly the question behind targeted campaigns.

   **scikit-plots connection**

   .. code-block:: python

      import scikitplot as skplt

      skplt.metrics.plot_cumulative_gain(y_true, y_probas)
      skplt.metrics.plot_lift_curve(y_true, y_probas)
      skplt.metrics.plot_ks_statistic(y_true, y_probas)   # max separation

   **When to use it** — ranked-action problems with a budget: direct
   mail, retention offers, lead scoring, collections.

   .. seealso::

      :ref:`data-prep-roc-auc` · :ref:`data-prep-model-quality`

----------------------------------------------------------------------

.. _data-prep-models:

Part 4 — Decision Trees & Diagnostics
----------------------------------------------------------------------

Interpretable models, and the residual checks that reveal where any model
breaks down.

.. dropdown:: Decision Trees & CART (Interactions, Piecewise Structure)
   :color: danger
   :icon: git-branch
   :name: data-prep-cart

   **What is it?**

   A **CART** (Classification And Regression Tree) recursively splits the
   feature space into axis-aligned regions, predicting a constant in each
   leaf. It is therefore a **piecewise-constant** model that captures
   **interactions** automatically: a split on one feature changes which
   splits matter below it.

   Splits are chosen to reduce impurity — Gini for classification:

   .. math::

      G = \sum_{c} p_c\,(1 - p_c)

   **scikit-learn**

   .. code-block:: python

      from sklearn.tree import DecisionTreeClassifier, plot_tree

      tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)
      tree.fit(X_tr, y_tr)
      plot_tree(tree, filled=True, feature_names=cols)

   **When to use it** — when an interpretable, rule-based model and
   explicit interactions matter more than squeezing out the last point of
   accuracy. Control depth / leaf size to avoid overfitting.

   .. seealso::

      :ref:`data-prep-explain-clusters` · :ref:`terminology-classification-types`

.. dropdown:: Explaining Clustering Results with a Tree
   :color: danger
   :icon: list-unordered
   :name: data-prep-explain-clusters

   **The idea**

   Clustering (e.g. k-means) produces group labels but no explanation.
   Treat those cluster labels as a **target** and fit a shallow decision
   tree on the original features — the tree's splits become a
   human-readable description of *what makes each cluster different*.

   .. code-block:: python

      from sklearn.tree import DecisionTreeClassifier

      explainer = DecisionTreeClassifier(max_depth=3)
      explainer.fit(X, cluster_labels)        # labels from KMeans, etc.

   **When to use it** — segmentation deliverables where stakeholders need
   "Cluster 2 = high spend, low tenure" rather than centroid coordinates.

   .. seealso::

      :ref:`data-prep-cart`

.. dropdown:: Residual Diagnostics & Outliers
   :color: danger
   :icon: search
   :name: data-prep-residuals

   **What is it?**

   A **residual** is the gap between observed and predicted value,
   :math:`e_i = y_i - \hat{y}_i`. **Studentized residuals** rescale each
   residual by its estimated standard deviation so they are comparable;
   points with :math:`|e_i^{\text{stud}}| > 3` are candidate **outliers**
   the model cannot explain.

   **scikit-plots connection**

   .. code-block:: python

      import scikitplot as skplt
      skplt.api.metrics.plot_residuals_distribution(y_true, y_pred)

   **When to use it** — after fitting any regression (or probability)
   model, to check for structure, heteroscedasticity, and influential
   outliers before trusting predictions.

   .. seealso::

      :ref:`data-prep-model-quality`

----------------------------------------------------------------------

.. _data-prep-skplt-map:

Map to scikit-plots Examples
----------------------------------------------------------------------

Worked, runnable galleries for the workflow above (verified links):

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Classifier evaluation
      :link: https://scikit-plots.github.io/dev/auto_examples/classification/plot_classifier_eval_script.html

      Confusion matrix, ROC and PR in one figure.

   .. grid-item-card:: ROC curve
      :link: https://scikit-plots.github.io/dev/auto_examples/classification/plot_roc_script.html

      Per-class and averaged ROC with AUC.

   .. grid-item-card:: Precision–Recall
      :link: https://scikit-plots.github.io/dev/auto_examples/classification/plot_precision_recall_script.html

      The imbalance-aware companion to ROC.

   .. grid-item-card:: Cumulative gains
      :link: https://scikit-plots.github.io/dev/auto_examples/decile/plot_cumulative_gain_script.html

      Share of positives captured by the top deciles.

   .. grid-item-card:: Lift curve
      :link: https://scikit-plots.github.io/dev/auto_examples/decile/plot_lift_script.html

      Improvement over random at each decile.

   .. grid-item-card:: KS statistic
      :link: https://scikit-plots.github.io/dev/auto_examples/decile/plot_ks_statistic_script.html

      Maximum class separation along the ranked score.

   .. grid-item-card:: modelplotpy
      :link: https://scikit-plots.github.io/dev/auto_examples/decile/plot_modelplotpy_script.html

      Business-facing gains / lift / response reports.

   .. grid-item-card:: Residuals distribution
      :link: https://scikit-plots.github.io/dev/auto_examples/regression/plot_residuals_distribution_script.html

      Residual and Q–Q diagnostics for fitted models.

----------------------------------------------------------------------

.. _data-prep-sources:

Sources
----------------------------------------------------------------------

Verified during preparation of this page; links were resolvable at the
documentation build date.

**Source context (framing only, re-expressed in our own words)**

* Data Preparation and Analysis category (56 posts):
  https://insightful-data-lab.com/category/data-preparation-and-analysis/

**Official documentation (API calls used above)**

* scikit-learn — model evaluation metrics:
  https://scikit-learn.org/stable/modules/model_evaluation.html
* scikit-learn — decision trees:
  https://scikit-learn.org/stable/modules/tree.html
* scikit-learn — train/test splitting:
  https://scikit-learn.org/stable/modules/cross_validation.html

**scikit-plots (this project)**

* Example gallery: https://scikit-plots.github.io/dev/auto_examples/index.html
* API reference: https://scikit-plots.github.io/dev/apis/index.html
* Terminology reference: :ref:`terminology-index`

**Standard references**

* James, Witten, Hastie & Tibshirani, *An Introduction to Statistical
  Learning* (free): https://www.statlearning.com/
* Breiman, Friedman, Olshen & Stone, *Classification and Regression
  Trees* (CART), 1984.

..
   ##################################################################
   Tags — bottom of page, project controlled vocabulary only.
   ##################################################################

.. tags::
   purpose: reference,
   domain: statistics,
   model-type: classification,
   model-workflow: model evaluation,
   level: beginner,
   level: intermediate,
   level: advanced
