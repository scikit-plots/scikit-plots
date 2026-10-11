.. _preprocessing-dummy-code-encoder:

DummyCodeEncoder
================

:class:`DummyCodeEncoder` is the scikit-learn-style fitted encoder. It supports
``fit``, ``transform``, ``inverse_transform`` and ``get_feature_names_out`` and
can produce sparse output.

Unknown-category handling should be chosen deliberately. ``ignore`` can keep
inference running but may map an unseen category to an all-zero representation;
``error`` is safer when schema drift must be surfaced immediately.

.. warning::

   The current maintenance review records two correctness findings in this
   encoder: cross-feature category collapse and an infrequent-category grouping
   path that is not fully applied. Do not assume those advanced behaviors are
   release-ready until the findings are closed and regression-tested.
