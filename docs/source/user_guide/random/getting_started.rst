.. _random-getting-started:

Getting started
===============

The preferred entry point mirrors NumPy's generator style:

.. code-block:: python

   from scikitplot.random import default_rng

   rng = default_rng(42)
   values = rng.random(4)
   integers = rng.integers(0, 10, size=4)

A generator is stateful: calls advance its stream. Re-create the generator with
the same seed when you need the same initial stream rather than resetting it
implicitly in the middle of an algorithm.

Context-managed use
-------------------

``kiss_context(seed)`` and generator context-manager support make lifetime
explicit for code that wants a scoped stream. A context manager does not make a
pseudo-random generator cryptographically secure.
