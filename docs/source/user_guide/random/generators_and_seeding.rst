.. _random-generators-and-seeding:

Generators and seeding
======================

The public hierarchy includes :class:`KissSeedSequence`, 32-bit and 64-bit
legacy generators, :class:`KissBitGenerator`, :class:`KissGenerator`, and
:class:`KissRandomState` compatibility behavior.

Seed sequences
--------------

Use :class:`KissSeedSequence` when one root seed must produce multiple explicit
child streams. ``spawn`` makes stream derivation visible and is preferable to
inventing ad-hoc arithmetic on integer seeds.

Bit width
---------

``KissRandom`` / ``default_rng`` accept a bit-width policy. If reproducibility
must survive different machines or future defaults, choose the width explicitly
rather than depending on ``"auto"``.

Parallel work
-------------

Do not share one mutable generator across independent workers and assume their
scheduling will be reproducible. Give each worker an explicitly derived child
seed/stream.
