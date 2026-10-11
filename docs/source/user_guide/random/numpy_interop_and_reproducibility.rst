.. _random-numpy-interop-and-reproducibility:

NumPy interop and reproducibility
=================================

The API intentionally resembles ``numpy.random.Generator`` and provides common
distributions such as ``random``, ``integers``, ``uniform``, ``normal``,
``choice``, ``shuffle``, ``permutation``.

Reproducibility has a scope
---------------------------

A fixed seed controls this generator's state transition. It does not by itself
make an entire machine-learning experiment deterministic: thread scheduling,
BLAS libraries, GPU kernels, data ordering, third-party estimators, and process
fork/spawn behavior can introduce additional randomness.

Record the seed, generator/bit width, package version, and other algorithmic
random-state settings when results must be reproduced later.

Further reading
---------------

The repository ships :download:`Good Practice in (Pseudo) Random Number
Generation <GoodPracticeRNG.pdf>` alongside this guide. It is useful background
for thinking about independent streams, seeding discipline, and reproducibility.
