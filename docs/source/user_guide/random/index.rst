.. currentmodule:: scikitplot.random

.. _random-index:

KISS random number generation
=============================

``scikitplot.random`` provides a NumPy-like KISS random-number-generation surface with seed sequences, bit generators, stateful generators, serialization, and convenience distribution functions.

.. important::

   A deterministic pseudo-random stream is reproducible state, not cryptographic randomness. Do not use this module for passwords, tokens, keys, nonces, or any other security-sensitive entropy.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   generators_and_seeding
   state_and_serialization
   numpy_interop_and_reproducibility
   python_api
   limits_and_troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: random-getting-started
      :link-type: ref

      Create a KISS generator with ``default_rng`` or ``kiss_context``.

   .. grid-item-card:: Generators and seeding
      :link: random-generators-and-seeding
      :link-type: ref

      Choose 32/64-bit generators, seed sequences, spawning, and deterministic streams.

   .. grid-item-card:: State and serialization
      :link: random-state-and-serialization
      :link-type: ref

      Snapshot, pickle, JSON-style serialization, and restore state deliberately.

   .. grid-item-card:: NumPy interop and reproducibility
      :link: random-numpy-interop-and-reproducibility
      :link-type: ref

      Understand NumPy-like semantics, distributions, and what reproducibility does and does not promise.

   .. grid-item-card:: Python API
      :link: random-python-api
      :link-type: ref

      Generator, bit generator, seed sequence, RandomState compatibility, and convenience functions.

   .. grid-item-card:: Limits and troubleshooting
      :link: random-limits-and-troubleshooting
      :link-type: ref

      Compiled-extension requirements, thread/process use, and state-version compatibility.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`random_examples` for executable examples that accompany this guide.

Further reading
---------------

.. seealso::
   * https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
   * :download:`"Download GoodPracticeRNG.pdf" <./GoodPracticeRNG.pdf>`.
