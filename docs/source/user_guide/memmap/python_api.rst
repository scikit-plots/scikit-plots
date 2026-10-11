.. _memmap-python-api:

Python API
==========

.. autosummary::
   :nosignatures:

   MemoryMap
   mmap_region
   py_get_page_size
   py_is_page_aligned
   py_validate_prot_flags
   py_validate_map_flags
   MMapError
   MMapAllocationError
   MMapInvalidParameterError

The module also exports ``PY_PROT_*``, ``PY_MAP_*``, ``PY_MS_*`` and Windows
compatibility constants. Prefer intent-oriented factory methods and validation
helpers over hard-coding platform-specific numeric values.
