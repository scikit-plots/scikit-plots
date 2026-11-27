- :class:`~scikitplot.corpus.LeanReader` without ``scikitplot.lean``
  installed raises :class:`ModuleNotFoundError` (an :class:`ImportError`)
  naming ``scikitplot.lean``, instead of a plain :class:`ImportError`, so
  the partial ``scikit-plots-corpus`` test run reports a test that needs
  the absent part as skipped rather than failed.
