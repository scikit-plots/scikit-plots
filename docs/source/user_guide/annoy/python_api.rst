.. _annoy-python-api:

Python API
==========

Primary classes and helpers
---------------------------

.. autosummary::
   :nosignatures:

   Annoy
   AnnoyIndex
   Index
   threads_info
   IndexIOMixin
   MetaMixin
   NDArrayMixin
   PickleMixin
   PlottingMixin
   VectorOpsMixin

:class:`Index` is the normal starting point. The mixins are public for advanced
users constructing compatible wrappers around the same backend, but composing a
custom wrapper also makes you responsible for preserving the backend and
lifecycle assumptions those mixins expect.

Pickle controls
---------------

``PickleMode`` and ``CompressMode`` make persistence policy explicit. Prefer
native Annoy persistence when cross-version Python-object compatibility is not
required.
