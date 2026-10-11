.. _memmap-lifecycle-and-io:

Lifecycle and I/O
=================

A :class:`MemoryMap` owns a native mapping until it is closed. Context-manager
use is preferred because cleanup remains deterministic when an exception is
raised.

``read(size, offset)`` and ``write(data, offset)`` operate on byte ranges.
Validate bounds before treating a mapping as a structured record store; this
module does not know your file format, element type, endianness, or record
layout.

File-backed mappings
--------------------

A file-backed mapping exposes bytes through the operating-system page cache.
Changes and visibility depend on mapping flags and synchronization choices.
Do not assume that Python object lifetime alone publishes data durably to disk.

Anonymous mappings
------------------

Anonymous mappings are not associated with a file. They are useful for native
scratch space and shared-memory-like patterns where the process controls both
the layout and lifetime.
