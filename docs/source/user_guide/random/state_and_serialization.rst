.. _random-state-and-serialization:

State and serialization
=======================

The KISS objects expose explicit state and serialization helpers in addition to
pickle support. Prefer the explicit representation when you need an auditable or
language-neutral-ish configuration boundary.

.. code-block:: python

   import json
   from scikitplot.random import default_rng, KissGenerator

   rng = default_rng(123)
   payload = rng.serialize()
   text = json.dumps(payload)
   restored = KissGenerator.deserialize(json.loads(text))

State payloads include compatibility metadata. Treat serialized state as part
of a reproducibility record: keep the package/runtime version alongside long-
lived experiment artifacts.

Pickle is convenient for trusted Python environments but must not be used with
untrusted input.
