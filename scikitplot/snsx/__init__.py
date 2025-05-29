"""
Seaborn extension module.

Evolution-aware ML plotting visualization library hypothetically speaking:

# Category: evaluation
snsx.evaluation.roc_curve(y_true, y_score)
snsx.evaluation.confusion_matrix(y_true, y_pred)

# Category: projection
snsx.projection.pca(data=X, hue=labels)
snsx.projection.tsne(data=X, hue=labels)

# Category: model
snsx.model.feature_importance(model, feature_names=X.columns)
snsx.model.shap_summary(model, X)
snsx.model.coefficients(model)

Optional unified entry point:
snsx.plot(kind="roc_curve", y_true=y, y_score=probs)
snsx.plot(kind="pca", data=X, hue=clusters)
snsx.plot(kind="feature_importance", model=clf, data=X)
"""

# scikitplot/_utils/__init__.py
