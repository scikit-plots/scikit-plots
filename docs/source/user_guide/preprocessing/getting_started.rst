.. _preprocessing-getting-started:

Getting started
===============

Use :class:`DummyCodeEncoder` when you need fit/transform semantics:

.. code-block:: python

   from scikitplot.preprocessing import DummyCodeEncoder

   encoder = DummyCodeEncoder(columns=["city"], handle_unknown="ignore")
   X_train_enc = encoder.fit_transform(X_train)
   X_test_enc = encoder.transform(X_test)

Use :class:`GetDummies` when a pandas-style dummy expansion is the desired
surface and its column/unknown-category policy matches your workflow.
