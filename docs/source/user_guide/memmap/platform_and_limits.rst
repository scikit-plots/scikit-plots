.. _memmap-platform-and-limits:

Platform and limits
===================

The public API is cross-platform, but native memory mapping is necessarily
platform-sensitive.

* file offsets commonly need page alignment;
* executable mappings may be restricted by platform policy;
* locking pages can fail because of process or system limits;
* mapping a file beyond its valid extent is not a portable data-growth
  strategy; and
* this package is a compiled extension and therefore needs a compatible build
  for the target Python/platform.

``scikitplot.memmap`` consumes the shared ``mman.h`` owned by the independent
Annoy C-external subsystem. That shared native header is not part of the memmap
user API and should not be copied or patched as an application workaround.
