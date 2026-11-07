.. _metadata_routing:

======================================================================
Metadata routing
======================================================================

*Metadata routing* is scikit-learn's mechanism for passing extra arguments
such as ``sample_weight`` from a meta-estimator (a pipeline, ``cross_validate``,
a search) to the steps, estimators and scorers that ask for them. It is off by
default; turn it on for a block of code or for the session::

    import sklearn

    with sklearn.config_context(enable_metadata_routing=True):
        ...

    sklearn.set_config(enable_metadata_routing=True)

The rules (requests, ``set_<method>_request``, the errors) are scikit-learn's;
see the `scikit-learn metadata routing guide
<https://scikit-learn.org/stable/metadata_routing.html>`_. This page lists
where scikit-plots takes part. Every result below was checked with
scikit-learn 1.9.1.

Estimators
----------

A scikit-plots estimator that subclasses scikit-learn's ``BaseEstimator``
inherits its request methods (``set_fit_request`` and the others), so it is
routed like any scikit-learn estimator. scikit-plots adds no routing of its
own to these estimators.

``scikitplot.experimental.pipeline.Pipeline``
---------------------------------------------

The experimental ``Pipeline`` subclasses :class:`sklearn.pipeline.Pipeline`
and inherits its ``get_metadata_routing``, so the rules are scikit-learn's:

.. list-table::
   :header-rows: 1

   * - Routing
     - Call
     - Result
   * - off
     - ``pipe.fit(X, y, clf__sample_weight=w)``
     - passed to the ``clf`` step's ``fit``
   * - off
     - ``pipe.fit(X, y, sample_weight=w)``
     - ``ValueError``: name the step, ``<step>__sample_weight``
   * - on
     - ``pipe.fit(X, y, sample_weight=w)``, each step having set its request
     - passed to the steps that requested it
   * - on
     - ``pipe.fit(X, y, sample_weight=w)``, a step has not set its request
     - ``UnsetMetadataPassedError`` naming that step
   * - on
     - ``pipe.fit(X, y, clf__sample_weight=w)``
     - ``TypeError``: the ``<step>__`` form is for routing off only

.. code-block:: python

    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from scikitplot.experimental.pipeline import Pipeline

    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            ("scale", StandardScaler().set_fit_request(sample_weight=False)),
            ("clf", LogisticRegression().set_fit_request(sample_weight=True)),
        ])
        pipe.fit(X, y, sample_weight=w)   # only "clf" receives sample_weight

Corpus scorers
--------------

:func:`scikitplot.corpus.make_corpus_scorer` and
:func:`scikitplot.corpus.get_corpus_scorer` build scorers on scikit-learn's
scorer base when scikit-learn is installed
(:func:`scikitplot.corpus.scorer_backend_info` says which base is in use):

* A direct call takes ``sample_weight`` with or without routing:
  ``scorer(estimator, X, y, sample_weight=w)``.
* With routing on, ``scorer.set_score_request(sample_weight=True)`` lets
  ``cross_validate(..., params={"sample_weight": w})`` (scikit-learn 1.4 and
  later) weight each fold's score by its share of ``w``. With routing off,
  scikit-learn's ``set_score_request`` raises ``RuntimeError``.

Without scikit-learn (a ``scikit-plots-corpus`` install), the scorer uses a
vendored base: a direct call with ``sample_weight`` behaves the same, but
``set_score_request`` raises ``RuntimeError`` and extra keyword arguments at
call time raise ``ValueError``; pass them to ``make_corpus_scorer`` instead.
See :ref:`the corpus evaluation guide <corpus-evaluation>` for the scorers.
