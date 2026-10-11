.. currentmodule:: scikitplot.memmap

.. _memmap-index:

Memory mapping
==============

``scikitplot.memmap`` exposes a small native memory-mapping API for anonymous or file-backed regions. It is intended for explicit lifetime management and random byte access, not for parsing file formats or replacing ordinary stream I/O.

.. important::

   A memory mapping is a resource with operating-system lifetime and protection rules. Prefer a context manager, validate offsets and sizes, and never treat a raw address as a portable data interface.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   lifecycle_and_io
   protection_sync_locking
   python_api
   platform_and_limits

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: memmap-getting-started
      :link-type: ref

      Create an anonymous or file-backed mapping and close it deterministically.

   .. grid-item-card:: Lifecycle and I/O
      :link: memmap-lifecycle-and-io
      :link-type: ref

      Read/write mapped ranges, context management, offsets, and ownership.

   .. grid-item-card:: Protection, sync, and locking
      :link: memmap-protection-sync-locking
      :link-type: ref

      Use mprotect, msync, and mlock with platform-aware expectations.

   .. grid-item-card:: Python API
      :link: memmap-python-api
      :link-type: ref

      MemoryMap, mmap_region, flags, validation helpers, and exceptions.

   .. grid-item-card:: Platform and limits
      :link: memmap-platform-and-limits
      :link-type: ref

      Page alignment, permissions, native-build requirements, and portability.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`memmap_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`annoy-index`

Further reading
---------------

.. seealso::
   * https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
   * numpy.memmap
   * scipy.io
   * mmap (Python standard library)
